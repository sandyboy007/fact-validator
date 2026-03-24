"""
Comparative Analysis Framework

Enables comparison with:
- Human judges/annotators
- Existing systems (Google Fact Check API, ClaimBuster, etc.)
- Multiple baselines
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
from app.statistics import StatisticalAnalyzer, ComparisonAnalyzer


class SystemType(str, Enum):
    """Types of systems to compare against."""
    FACT_VALIDATOR = "FactValidator"
    GOOGLE_FACT_CHECK = "GoogleFactCheck"
    CLAIMBUSTER = "ClaimBuster"
    HUMAN_ANNOTATOR = "HumanAnnotator"
    BASELINE_RANDOM = "RandomBaseline"
    BASELINE_KEYWORD = "KeywordBaseline"
    BASELINE_HEURISTIC = "HeuristicBaseline"


@dataclass
class ComparativeJudgment:
    """Single comparative judgment from human or system."""
    claim_id: str
    judge: str  # human name or system name
    verdict: str  # SUPPORTED, REFUTED, NEI
    confidence: float  # 0-100
    reasoning: Optional[str] = None
    annotation_time_sec: Optional[float] = None  # For humans
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class InterraterAgreement:
    """Interrater agreement statistics."""
    cohen_kappa: float  # -1 to 1, higher is better
    fleiss_kappa: Optional[float] = None  # For 3+ judges
    percent_agreement: float = 0.0
    n_claims: int = 0
    n_judges: int = 0


class HumanEvaluationFramework:
    """Framework for human evaluation."""
    
    @staticmethod
    def create_evaluation_instructions() -> str:
        """Generate instructions for human annotators."""
        return """
=== HUMAN FACT-CHECKING INSTRUCTIONS ===

For each claim, provide a verdict:

1. SUPPORTED
   - Multiple credible sources confirm the claim
   - Evidence is recent and from reputable sources
   - Claim is consistent with expert consensus
   Example: "Vaccines prevent disease" → SUPPORTED

2. REFUTED
   - Credible sources contradict the claim
   - Evidence directly contradicts the claim
   - Expert consensus disputes the claim
   Example: "Vaccines cause autism" → REFUTED

3. NEI (Not Enough Information)
   - No sufficient evidence found
   - Evidence is conflicting or unclear
   - Claim is too new/niche for evidence
   - Claim is a prediction/opinion, not checkable fact
   Example: "AI will cause unemployment by 2030" → NEI

GUIDELINES:
- Focus on factual accuracy, not opinion
- Use your best judgment within 2 minutes per claim
- Indicate your confidence (0-100%)
- Note reasoning if claim is difficult

TIME TARGET: 30-60 seconds per claim
"""
    
    @staticmethod
    def calculate_interrater_agreement(
        judgments: List[ComparativeJudgment],
        claims: Optional[List[str]] = None
    ) -> InterraterAgreement:
        """
        Calculate interrater agreement (Cohen's kappa or Fleiss' kappa).
        
        Args:
            judgments: List of judgments from multiple judges
            claims: Optional list of claim IDs to limit comparison
        
        Returns:
            InterraterAgreement metrics
        """
        # Group judgments by judge
        judgments_by_judge: Dict[str, Dict[str, str]] = {}
        judges_set = set()
        claims_set = set()
        
        for judgment in judgments:
            if claims and judgment.claim_id not in claims:
                continue
            
            if judgment.judge not in judgments_by_judge:
                judgments_by_judge[judgment.judge] = {}
            
            judgments_by_judge[judgment.judge][judgment.claim_id] = judgment.verdict
            judges_set.add(judgment.judge)
            claims_set.add(judgment.claim_id)
        
        judges_list = sorted(list(judges_set))
        claims_list = sorted(list(claims_set))
        
        if len(judges_list) < 2:
            return InterraterAgreement(
                cohen_kappa=float('nan'),
                percent_agreement=0.0,
                n_claims=len(claims_list),
                n_judges=len(judges_list)
            )
        
        # Calculate percent agreement (simple)
        agreements = 0
        total_pairs = 0
        
        for claim_id in claims_list:
            verdicts = []
            for judge in judges_list:
                if claim_id in judgments_by_judge[judge]:
                    verdicts.append(judgments_by_judge[judge][claim_id])
            
            if len(verdicts) >= 2:
                # Check if all verdicts match
                first_verdict = verdicts[0]
                if all(v == first_verdict for v in verdicts):
                    agreements += 1
                total_pairs += 1
        
        percent_agreement = agreements / total_pairs if total_pairs > 0 else 0
        
        # Cohen's kappa for 2 judges
        if len(judges_list) == 2:
            cohen_kappa = HumanEvaluationFramework._calculate_cohens_kappa(
                judgments, judges_list, claims_list
            )
        else:
            cohen_kappa = float('nan')
        
        return InterraterAgreement(
            cohen_kappa=cohen_kappa,
            percent_agreement=percent_agreement,
            n_claims=len(claims_list),
            n_judges=len(judges_list)
        )
    
    @staticmethod
    def _calculate_cohens_kappa(
        judgments: List[ComparativeJudgment],
        judges: List[str],
        claims: List[str]
    ) -> float:
        """Calculate Cohen's kappa for 2 judges."""
        labels = ["SUPPORTED", "REFUTED", "NEI"]
        
        # Build confusion matrix
        confusion = {l1: {l2: 0 for l2 in labels} for l1 in labels}
        
        judgments_by_judge_claim = {}
        for j in judgments:
            key = (j.judge, j.claim_id)
            judgments_by_judge_claim[key] = j.verdict
        
        n_total = 0
        for claim_id in claims:
            if (judges[0], claim_id) in judgments_by_judge_claim and \
               (judges[1], claim_id) in judgments_by_judge_claim:
                v1 = judgments_by_judge_claim[(judges[0], claim_id)]
                v2 = judgments_by_judge_claim[(judges[1], claim_id)]
                confusion[v1][v2] += 1
                n_total += 1
        
        if n_total == 0:
            return 0.0
        
        # Observed agreement
        p_o = sum(confusion[l][l] for l in labels) / n_total
        
        # Expected agreement by chance
        p_e = 0
        for label in labels:
            n_1 = sum(confusion[label][l] for l in labels)
            n_2 = sum(confusion[l][label] for l in labels)
            p_e += (n_1 / n_total) * (n_2 / n_total)
        
        # Cohen's kappa
        if p_e == 1.0:
            return 0.0
        
        kappa = (p_o - p_e) / (1 - p_e)
        return max(-1, min(1, kappa))  # Clamp to [-1, 1]
    
    @staticmethod
    def agreement_interpretation(kappa: float) -> str:
        """Interpret Cohen's kappa value."""
        kappa = max(-1, min(1, kappa))  # Clamp
        
        if kappa < 0:
            return "poor"
        elif kappa < 0.20:
            return "slight"
        elif kappa < 0.40:
            return "fair"
        elif kappa < 0.60:
            return "moderate"
        elif kappa < 0.80:
            return "substantial"
        else:
            return "excellent"


class ComparativeAnalysis:
    """Compare system performance across multiple judges/systems."""
    
    @staticmethod
    def build_comparison_matrix(
        system_predictions: Dict[str, List[float]],  # system_name -> [scores]
        system_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, any]]:
        """
        Build comparison matrix for multiple systems.
        
        Args:
            system_predictions: Dict mapping system name to accuracy scores
            system_names: Optional list to control order
        
        Returns:
            Comparison matrix with pairwise statistical comparisons
        """
        if system_names is None:
            system_names = sorted(system_predictions.keys())
        
        comparison_matrix = {}
        analyzer = StatisticalAnalyzer()
        
        for system_name in system_names:
            comparison_matrix[system_name] = {
                "accuracy_mean": float(np.mean(system_predictions[system_name])),
                "accuracy_std": float(np.std(system_predictions[system_name])),
                "n_samples": len(system_predictions[system_name]),
                "comparisons": {}
            }
        
        # Pairwise comparisons
        for i, system1 in enumerate(system_names):
            for system2 in system_names[i+1:]:
                comparison = ComparisonAnalyzer.compare_system_vs_baseline(
                    system_predictions[system1],
                    system_predictions[system2],
                    system_name=system1,
                    baseline_name=system2
                )
                
                comparison_matrix[system1]["comparisons"][system2] = {
                    "improvement_pct": comparison.improvement_pct,
                    "p_value": comparison.significance_test.p_value,
                    "is_significant": comparison.significance_test.is_significant,
                    "cohens_d": comparison.effect_size,
                    "effect_interpretation": comparison.effect_interpretation
                }
        
        return comparison_matrix
    
    @staticmethod
    def generate_comparative_report(
        comparison_matrix: Dict,
        system_names: List[str]
    ) -> str:
        """Generate formatted comparative report."""
        report = "=== COMPARATIVE PERFORMANCE ANALYSIS ===\n\n"
        
        # Summary table
        report += "| System | Accuracy | Std Dev | N |\n"
        report += "|--------|----------|---------|---|\n"
        for system in system_names:
            acc = comparison_matrix[system]["accuracy_mean"]
            std = comparison_matrix[system]["accuracy_std"]
            n = comparison_matrix[system]["n_samples"]
            report += f"| {system} | {acc:.3f} | {std:.3f} | {n} |\n"
        
        report += "\n\n### Pairwise Comparisons\n\n"
        
        for i, system1 in enumerate(system_names):
            for system2 in system_names[i+1:]:
                if system2 in comparison_matrix[system1]["comparisons"]:
                    comp = comparison_matrix[system1]["comparisons"][system2]
                    sig = "✓" if comp["is_significant"] else "✗"
                    
                    report += f"**{system1} vs {system2}:**\n"
                    report += f"  {sig} Improvement: {comp['improvement_pct']:+.2f}%\n"
                    report += f"  Effect size: {comp['cohens_d']:.3f} ({comp['effect_interpretation']})\n"
                    report += f"  p-value: {comp['p_value']:.4f}\n\n"
        
        return report


@dataclass
class BenchmarkResult:
    """Result from benchmark evaluation."""
    system_name: str
    benchmark_name: str
    test_set_size: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    human_agreement: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BenchmarkFramework:
    """Framework for standardized benchmarking against existing systems."""
    
    # Reference implementations for comparison
    REFERENCE_BENCHMARKS = {
        "GoogleFactCheck": {
            "description": "Google Fact Check API (reference system)",
            "url": "https://toolbox.google.com/factcheck/",
            "api_available": False  # Requires API key
        },
        "ClaimBuster": {
            "description": "ClaimBuster (CMU-based fact verification)",
            "url": "https://claimbuster.org",
            "api_available": True
        },
        "FEVER": {
            "description": "Fact Extraction and VERification dataset baseline",
            "url": "https://fever.ai",
            "api_available": False
        }
    }
    
    @staticmethod
    def export_benchmark_results(
        results: List[BenchmarkResult],
        output_file: str
    ) -> str:
        """Export benchmark results to JSON."""
        data = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "results_count": len(results)
            },
            "results": [
                {
                    "system": r.system_name,
                    "benchmark": r.benchmark_name,
                    "test_set_size": r.test_set_size,
                    "metrics": {
                        "accuracy": r.accuracy,
                        "precision": r.precision,
                        "recall": r.recall,
                        "f1": r.f1
                    },
                    "human_agreement": r.human_agreement,
                    "timestamp": r.timestamp
                }
                for r in results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_file


# Add numpy import at module level
import numpy as np
