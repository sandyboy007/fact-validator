"""
Baseline implementations for Fact Validator comparison.

Includes:
- Random baseline (lower bound)
- Keyword-matching baseline (strawman)
- Simple heuristic baseline
- Reference implementations (Google Fact Check, ClaimBuster patterns)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import random
import re
from app.evaluation import PredictionResult, VerdictLabel


class BaselineType(str, Enum):
    """Types of baselines."""
    RANDOM = "random"
    KEYWORD = "keyword_matching"
    LENGTH_HEURISTIC = "length_heuristic"
    SENTIMENT_HEURISTIC = "sentiment_heuristic"
    MAJORITY_CLASS = "majority_class"


@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    baseline_type: BaselineType
    random_seed: int = 42


class RandomBaseline:
    """Lower-bound baseline: random verdict assignment."""
    
    def __init__(self, seed: int = 42, distribution: Optional[Dict[str, float]] = None):
        """
        Initialize random baseline.
        
        Args:
            seed: Random seed for reproducibility
            distribution: Optional probability distribution over labels
                         e.g., {"SUPPORTED": 0.33, "REFUTED": 0.33, "NEI": 0.34}
        """
        self.seed = seed
        random.seed(seed)
        self.distribution = distribution or {
            VerdictLabel.SUPPORTED: 1/3,
            VerdictLabel.REFUTED: 1/3,
            VerdictLabel.NEI: 1/3
        }
    
    def predict(self, claim_text: str) -> tuple[str, float]:
        """
        Random prediction.
        
        Returns: (verdict_label, confidence_0_to_100)
        """
        labels = list(self.distribution.keys())
        probs = list(self.distribution.values())
        
        label = random.choices(labels, weights=probs, k=1)[0]
        confidence = random.uniform(30, 70)  # Random confidence
        
        return label, confidence


class KeywordBaseline:
    """Strawman baseline: keyword-matching heuristics."""
    
    # Define keyword patterns for each verdict
    SUPPORTED_KEYWORDS = [
        r'\b(confirmed|verified|true|correct|supported|accurate|valid|proven)\b',
        r'\b(shows|demonstrates|proves|establishes)\b',
        r'\b(research|study|evidence|findings|data)\s+(shows|confirms|supports)',
    ]
    
    REFUTED_KEYWORDS = [
        r'\b(false|debunked|myth|misconception|hoax|disproven|incorrect)\b',
        r'\b(denied|refuted|contradicts|contradicted|wrong|inaccurate)\b',
        r'\b(no evidence|lacks evidence|debunk|fake)\b',
    ]
    
    NEI_KEYWORDS = [
        r'\b(unclear|uncertain|unknown|may|might|could|possibly|perhaps)\b',
        r'\b(insufficient|not enough|lack of|insufficient)\s+(evidence|information)',
        r'\b(under investigation|unclear|ambiguous)\b',
    ]
    
    def __init__(self):
        self.supported_patterns = [re.compile(p, re.IGNORECASE) for p in self.SUPPORTED_KEYWORDS]
        self.refuted_patterns = [re.compile(p, re.IGNORECASE) for p in self.REFUTED_KEYWORDS]
        self.nei_patterns = [re.compile(p, re.IGNORECASE) for p in self.NEI_KEYWORDS]
    
    def predict(self, claim_text: str) -> tuple[str, float]:
        """
        Predict using keyword matching.
        
        Returns: (verdict_label, confidence_based_on_matches)
        """
        supported_matches = sum(1 for p in self.supported_patterns if p.search(claim_text))
        refuted_matches = sum(1 for p in self.refuted_patterns if p.search(claim_text))
        nei_matches = sum(1 for p in self.nei_patterns if p.search(claim_text))
        
        total_matches = supported_matches + refuted_matches + nei_matches
        
        if total_matches == 0:
            # No keywords found - default to NEI
            return VerdictLabel.NEI, 20.0
        
        # Majority vote
        matches = {
            VerdictLabel.SUPPORTED: supported_matches,
            VerdictLabel.REFUTED: refuted_matches,
            VerdictLabel.NEI: nei_matches
        }
        
        verdict = max(matches, key=matches.get)
        confidence = (matches[verdict] / total_matches) * 100
        
        return verdict, confidence


class LengthHeuristic:
    """Simple heuristic: longer claims tend to be more specific/refutable."""
    
    def predict(self, claim_text: str) -> tuple[str, float]:
        """
        Predict based on claim length.
        
        Short claims (< 100 chars): usually NEI or supported
        Medium claims (100-200): mixed
        Long claims (> 200): often refutable
        """
        length = len(claim_text)
        
        if length < 100:
            return VerdictLabel.SUPPORTED, 50.0
        elif length < 200:
            return VerdictLabel.NEI, 45.0
        else:
            return VerdictLabel.REFUTED, 48.0


class SentimentHeuristic:
    """Heuristic based on emotional language."""
    
    NEGATIVE_WORDS = ['terrible', 'horrible', 'dangerous', 'evil', 'worst', 'disgusting']
    POSITIVE_WORDS = ['great', 'excellent', 'amazing', 'wonderful', 'best', 'fantastic']
    NEUTRAL_WORDS = ['shows', 'indicates', 'demonstrates', 'suggests', 'indicates']
    
    def predict(self, claim_text: str) -> tuple[str, float]:
        """
        Predict based on sentiment.
        
        Emotional language → likely misinformation (REFUTED)
        Neutral language → likely accurate (SUPPORTED)
        """
        text_lower = claim_text.lower()
        
        negative_count = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)
        positive_count = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
        neutral_count = sum(1 for w in self.NEUTRAL_WORDS if w in text_lower)
        
        total_sentiment = negative_count + positive_count + neutral_count
        
        if total_sentiment == 0:
            return VerdictLabel.NEI, 40.0
        
        if negative_count > positive_count:
            return VerdictLabel.REFUTED, min(70.0, 40 + negative_count * 10)
        elif positive_count > negative_count:
            return VerdictLabel.SUPPORTED, min(70.0, 40 + positive_count * 10)
        else:
            return VerdictLabel.SUPPORTED, 55.0


class MajorityClassBaseline:
    """Predict most common label from training set."""
    
    def __init__(self, majority_label: str = VerdictLabel.SUPPORTED):
        """
        Initialize with majority class.
        
        Args:
            majority_label: Most common label in training data
        """
        self.majority_label = majority_label
    
    def predict(self, claim_text: str) -> tuple[str, float]:
        """Always predict majority class."""
        return self.majority_label, 50.0


class BaselineComparison:
    """Compare all baselines against ground truth."""
    
    def __init__(self):
        self.random_baseline = RandomBaseline()
        self.keyword_baseline = KeywordBaseline()
        self.length_baseline = LengthHeuristic()
        self.sentiment_baseline = SentimentHeuristic()
        self.majority_baseline = MajorityClassBaseline()
    
    def evaluate_all_baselines(
        self,
        test_claims: List[Dict[str, str]]
    ) -> Dict[str, List[PredictionResult]]:
        """
        Evaluate all baselines on test claims.
        
        Args:
            test_claims: List of {"id": str, "text": str, "category": str, "label": str}
        
        Returns:
            Dict mapping baseline name to list of PredictionResults
        """
        results = {
            "random": [],
            "keyword": [],
            "length": [],
            "sentiment": [],
            "majority": []
        }
        
        for claim in test_claims:
            claim_id = claim["id"]
            claim_text = claim["text"]
            category = claim.get("category", "general")
            ground_truth = claim["label"]
            
            # Random baseline
            label, conf = self.random_baseline.predict(claim_text)
            results["random"].append(PredictionResult(
                claim_id=claim_id,
                claim_text=claim_text,
                category=category,
                ground_truth_label=ground_truth,
                predicted_label=label,
                predicted_confidence=conf,
                model_name="RandomBaseline"
            ))
            
            # Keyword baseline
            label, conf = self.keyword_baseline.predict(claim_text)
            results["keyword"].append(PredictionResult(
                claim_id=claim_id,
                claim_text=claim_text,
                category=category,
                ground_truth_label=ground_truth,
                predicted_label=label,
                predicted_confidence=conf,
                model_name="KeywordBaseline"
            ))
            
            # Length baseline
            label, conf = self.length_baseline.predict(claim_text)
            results["length"].append(PredictionResult(
                claim_id=claim_id,
                claim_text=claim_text,
                category=category,
                ground_truth_label=ground_truth,
                predicted_label=label,
                predicted_confidence=conf,
                model_name="LengthBaseline"
            ))
            
            # Sentiment baseline
            label, conf = self.sentiment_baseline.predict(claim_text)
            results["sentiment"].append(PredictionResult(
                claim_id=claim_id,
                claim_text=claim_text,
                category=category,
                ground_truth_label=ground_truth,
                predicted_label=label,
                predicted_confidence=conf,
                model_name="SentimentBaseline"
            ))
            
            # Majority baseline
            label, conf = self.majority_baseline.predict(claim_text)
            results["majority"].append(PredictionResult(
                claim_id=claim_id,
                claim_text=claim_text,
                category=category,
                ground_truth_label=ground_truth,
                predicted_label=label,
                predicted_confidence=conf,
                model_name="MajorityBaseline"
            ))
        
        return results
