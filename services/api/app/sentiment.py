"""Sentiment analysis module for detecting emotional tone in claims."""

from typing import Literal, Dict, Any, List
from dataclasses import dataclass
import re


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    score: float  # -1 (negative) to +1 (positive)
    label: Literal["positive", "negative", "neutral"]
    emotional_intensity: float  # 0-1, how emotionally charged
    flags: List[str]  # red flags for emotional manipulation


# Sentiment lexicons for rule-based analysis (VADER-inspired)
_POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
    "best", "better", "improved", "success", "successful", "progress",
    "benefits", "beneficial", "positive", "uplifting", "inspiring",
    "genuine", "authentic", "proven", "confirmed", "supported",
    "breakthrough", "innovative", "revolutionary", "advanced",
    "helpful", "supportive", "thriving", "flourishing", "growth",
    "safe", "effective", "reliable", "trustworthy", "confident",
}

_NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "awful", "worst", "worse",
    "failed", "failure", "disaster", "crisis", "emergency",
    "danger", "dangerous", "threat", "scary", "terrifying",
    "evil", "wicked", "corrupt", "corrupt", "conspiracy",
    "hoax", "fake", "false", "lie", "deceive", "betrayal",
    "harmful", "toxic", "poisoned", "lethal", "deadly",
    "collapse", "ruin", "destroyed", "devastated", "catastrophic",
    "controversial", "blamed", "accused", "guilty", "criminal",
    "problematic", "misleading", "wrong", "flawed", "defective",
}

# Emotional amplifiers and intensifiers
_INTENSIFIERS = {
    "extremely", "very", "incredibly", "absolutely", "definitely",
    "completely", "totally", "utterly", "pure", "sheer",
    "massive", "huge", "enormous", "tremendous", "shocking",
}

# Red flag words for emotional manipulation
_MANIPULATION_FLAGS = {
    "shocking": "sensationalism",
    "bombshell": "sensationalism",
    "exposed": "alarmism",
    "coverup": "conspiracy-thinking",
    "must see": "urgency-manipulation",
    "everyone knows": "false-consensus",
    "proof": "false-certainty",
    "guaranteed": "false-certainty",
    "awakening": "conspiratorial",
    "sheeple": "ad-hominem",
    "wake up": "alarmism",
    "miracle": "exaggeration",
    "secret": "mystique",
    "mainstream media": "propaganda-framing",
    "fake news": "propaganda-framing",
    "censorship": "persecution-narrative",
    "evil": "demonization",
    "traitor": "demonization",
    "criminal": "demonization",
    "scam": "dismissiveness",
    "propaganda": "propaganda-framing",
}


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Analyze sentiment of text using rule-based approach.
    
    Returns:
        SentimentResult with score (-1 to +1), label, intensity, and flags
    """
    if not text or len(text.strip()) < 5:
        return SentimentResult(
            score=0.0,
            label="neutral",
            emotional_intensity=0.0,
            flags=[]
        )
    
    low_text = text.lower()
    words = re.findall(r'\b\w+\b', low_text)
    
    # Count sentiment words
    positive_count = sum(1 for w in words if w in _POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in _NEGATIVE_WORDS)
    
    # Check for intensifiers
    intensifier_count = sum(1 for w in words if w in _INTENSIFIERS)
    
    # Calculate base sentiment score
    total_words = len(words)
    if total_words == 0:
        return SentimentResult(
            score=0.0,
            label="neutral",
            emotional_intensity=0.0,
            flags=[]
        )
    
    # Normalize scores
    pos_ratio = positive_count / total_words
    neg_ratio = negative_count / total_words
    
    # Calculate sentiment score (-1 to +1)
    sentiment_score = pos_ratio - neg_ratio
    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    
    # Determine label
    if sentiment_score > 0.1:
        label = "positive"
    elif sentiment_score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    
    # Calculate emotional intensity (how emotionally charged)
    emotional_intensity = (positive_count + negative_count + intensifier_count) / (total_words * 0.5)
    emotional_intensity = min(1.0, emotional_intensity)
    
    # Detect manipulation flags
    flags = []
    for flag_word, flag_category in _MANIPULATION_FLAGS.items():
        if flag_word in low_text:
            flags.append(flag_category)
    
    # Remove duplicates
    flags = list(set(flags))
    
    return SentimentResult(
        score=round(sentiment_score, 3),
        label=label,
        emotional_intensity=round(emotional_intensity, 3),
        flags=flags
    )


def batch_analyze_sentiment(texts: List[str]) -> List[SentimentResult]:
    """Analyze sentiment of multiple texts."""
    return [analyze_sentiment(text) for text in texts]


def estimate_bias_risk(sentiment_label: str, emotional_intensity: float, flags: List[str]) -> str:
    """
    Estimate bias/manipulation risk based on sentiment characteristics.
    
    Returns: "low", "medium", or "high"
    """
    risk_score = 0
    
    # Strong positive or negative sentiment + high emotion = higher risk
    if sentiment_label in ("positive", "negative") and emotional_intensity > 0.6:
        risk_score += 2
    elif sentiment_label in ("positive", "negative"):
        risk_score += 1
    
    # Manipulation flags increase risk
    risk_score += len(flags) * 0.5
    
    if risk_score >= 2:
        return "high"
    elif risk_score >= 1:
        return "medium"
    else:
        return "low"
