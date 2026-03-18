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
    # General positive
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
    "best", "better", "improved", "success", "successful", "progress",
    "benefits", "beneficial", "positive", "uplifting", "inspiring",
    "genuine", "authentic", "proven", "confirmed", "supported",
    "breakthrough", "innovative", "revolutionary", "advanced",
    "helpful", "supportive", "thriving", "flourishing", "growth",
    "safe", "effective", "reliable", "trustworthy", "confident",
    # Science/research
    "discover", "discovered", "finding", "found", "study", "research",
    "academic", "peer-reviewed", "evidence", "data", "scientific",
    # Success/achievement
    "award", "winning", "triumph", "victory", "achieve", "accomplished",
    "outperform", "leading", "top", "first", "champion",
    # Health/wellbeing
    "healing", "cure", "recovery", "wellness", "healthy", "vital",
    "natural", "organic", "pure", "clean", "fresh",
    # Trust/credibility
    "transparent", "honest", "integrity", "ethical", "legitimate",
    "verified", "certified", "official", "authorized", "valid",
    # Quality
    "quality", "premium", "superior", "exceptional", "outstanding",
    "remarkable", "impressive", "stunning", "beautiful", "elegant",
}

_NEGATIVE_WORDS = {
    # General negative
    "bad", "terrible", "horrible", "awful", "worst", "worse",
    "failed", "failure", "disaster", "crisis", "emergency",
    "danger", "dangerous", "threat", "scary", "terrifying",
    "evil", "wicked", "corrupt", "corrupt", "conspiracy",
    "hoax", "fake", "false", "lie", "deceive", "betrayal",
    "harmful", "toxic", "poisoned", "lethal", "deadly",
    "collapse", "ruin", "destroyed", "devastated", "catastrophic",
    "controversial", "blamed", "accused", "guilty", "criminal",
    "problematic", "misleading", "wrong", "flawed", "defective",
    # Health/disease
    "disease", "illness", "sick", "infected", "virus", "pandemic",
    "epidemic", "plague", "outbreak", "contagion", "contamination",
    "side-effect", "toxin", "poison", "overdose", "death", "dying",
    # Distrust
    "suspicious", "doubt", "questionable", "unreliable", "untrustworthy",
    "fraudulent", "scam", "scheme", "corruption", "bribery",
    "censorship", "suppression", "coverup", "hidden", "secret",
    # Decline/failure
    "decline", "declining", "falling", "crash", "plummet", "bankruptcy",
    "unemployment", "poverty", "inequality", "suffering", "crisis",
    # Violation/abuse
    "abuse", "assault", "violation", "attack", "aggression", "violence",
    "oppression", "discrimination", "injustice", "exploitation",
    # Intensity negatives
    "disgusting", "repugnant", "abhorrent", "revolting", "sickening",
    "grotesque", "vile", "despicable", "heinous", "monstrous",
}

# Emotional amplifiers and intensifiers
_INTENSIFIERS = {
    "extremely", "very", "incredibly", "absolutely", "definitely",
    "completely", "totally", "utterly", "pure", "sheer",
    "massive", "huge", "enormous", "tremendous", "shocking",
    # Additional intensifiers
    "profoundly", "deeply", "severely", "dramatically", "significantly",
    "remarkably", "exceptionally", "extraordinarily", "exceptionally",
    "undeniably", "unquestionably", "indisputably", "incontrovertibly",
    "devastatingly", "overwhelmingly", "stunningly", "shockingly",
}

# Red flag words for emotional manipulation
_MANIPULATION_FLAGS = {
    # Sensationalism
    "shocking": "sensationalism",
    "bombshell": "sensationalism",
    "exposed": "alarmism",
    "breaking": "sensationalism",
    "urgent": "sensationalism",
    "must-see": "sensationalism",
    "incredible": "sensationalism",
    
    # Conspiracy/Cover-up
    "coverup": "conspiracy-thinking",
    "hidden": "conspiracy-thinking",
    "suppressed": "conspiracy-thinking",
    "shadow": "conspiracy-thinking",
    "elite": "conspiracy-thinking",
    
    # Alarmism
    "wake-up": "alarmism",
    "awakening": "conspiratorial",
    "warning": "alarmism",
    "beware": "alarmism",
    "danger": "alarmism",
    
    # False consensus
    "everyone-knows": "false-consensus",
    "obviously": "false-certainty",
    "clearly": "false-certainty",
    "evidently": "false-certainty",
    
    # Certainty claims
    "proof": "false-certainty",
    "guaranteed": "false-certainty",
    "proven": "false-certainty",
    "undeniable": "false-certainty",
    "incontrovertible": "false-certainty",
    
    # Manipulation tactics
    "miracle": "exaggeration",
    "cure-all": "exaggeration",
    "secret-weapon": "mystique",
    "ancient-secret": "mystique",
    "they-don't-want": "persecution-narrative",
    
    # Propaganda framing
    "mainstream-media": "propaganda-framing",
    "mainstream": "propaganda-framing",
    "fake-news": "propaganda-framing",
    "lamestream": "propaganda-framing",
    "legacy-media": "propaganda-framing",
    "state-media": "propaganda-framing",
    
    # Demonization
    "evil": "demonization",
    "traitor": "demonization",
    "enemy": "demonization",
    "villain": "demonization",
    "scum": "demonization",
    "corrupt": "demonization",
    
    # Ad hominem / Dismissiveness
    "sheeple": "ad-hominem",
    "sheep": "ad-hominem",
    "idiot": "ad-hominem",
    "moron": "ad-hominem",
    "brainwashed": "ad-hominem",
    "asleep": "ad-hominem",
    
    # Dismissal of evidence
    "scam": "dismissiveness",
    "hoax": "dismissiveness",
    "propaganda": "propaganda-framing",
    "lies": "dismissiveness",
    
    # Persecution narratives
    "censored": "persecution-narrative",
    "silenced": "persecution-narrative",
    "attacked": "persecution-narrative",
    "hunted": "persecution-narrative",
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


def calculate_sentiment_misinformation_adjustment(
    sentiment_result: SentimentResult,
) -> float:
    """
    Calculate adjustment factor (0-0.4) to apply to misinformation likelihood based on sentiment.
    
    Higher emotional intensity + manipulative flags = higher adjustment.
    This can be added to the base misinformation likelihood.
    
    Returns:
        float: Adjustment factor between 0 and 0.4
    """
    adjustment = 0.0
    
    # Base adjustment from emotional intensity
    if sentiment_result.emotional_intensity > 0.7:
        adjustment += 0.2  # High emotion = 20% boost
    elif sentiment_result.emotional_intensity > 0.5:
        adjustment += 0.1  # Medium emotion = 10% boost
    
    # Negative sentiment slightly increases risk
    if sentiment_result.label == "negative":
        adjustment += 0.05
    
    # Each manipulation flag adds 5% adjustment
    flag_adjustment = min(0.15, len(sentiment_result.flags) * 0.05)
    adjustment += flag_adjustment
    
    # Cap at 0.4
    return min(0.4, adjustment)


def get_sentiment_summary(sentiment_result: SentimentResult) -> str:
    """Generate a human-readable summary of sentiment analysis."""
    intensity_level = "highly" if sentiment_result.emotional_intensity > 0.6 else "moderately" if sentiment_result.emotional_intensity > 0.3 else "slightly"
    
    summary = f"This claim is {intensity_level} emotionally charged with {sentiment_result.label} sentiment"
    
    if sentiment_result.flags:
        summary += f" with red flags detected: {', '.join(sentiment_result.flags[:3])}"
        if len(sentiment_result.flags) > 3:
            summary += f" and {len(sentiment_result.flags) - 3} more"
        summary += " (possible manipulation)."
    else:
        summary += "."
    
    return summary
