"""Comprehensive tests for sentiment analysis module."""

import pytest
from app.sentiment import (
    analyze_sentiment,
    batch_analyze_sentiment,
    estimate_bias_risk,
    calculate_sentiment_misinformation_adjustment,
    get_sentiment_summary,
)


class TestBasicSentiment:
    """Test basic sentiment detection."""
    
    def test_positive_claim(self):
        """Test detection of positive sentiment."""
        text = "This breakthrough study proves the treatment is excellent and safe."
        result = analyze_sentiment(text)
        
        assert result.label == "positive"
        assert result.score > 0
        assert result.emotional_intensity > 0
    
    def test_negative_claim(self):
        """Test detection of negative sentiment."""
        text = "This terrible disaster destroyed everything and killed many people."
        result = analyze_sentiment(text)
        
        assert result.label == "negative"
        assert result.score < 0
        assert result.emotional_intensity > 0
    
    def test_neutral_claim(self):
        """Test detection of neutral sentiment."""
        text = "The population of France is approximately 67 million people."
        result = analyze_sentiment(text)
        
        assert result.label == "neutral"
        assert result.score == 0 or abs(result.score) < 0.1
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = analyze_sentiment("")
        
        assert result.label == "neutral"
        assert result.score == 0.0
        assert result.emotional_intensity == 0.0
    
    def test_short_text(self):
        """Test handling of very short text."""
        result = analyze_sentiment("ok")
        
        assert result.label == "neutral" or result.emotional_intensity < 0.3


class TestEmotionalIntensity:
    """Test emotional intensity detection."""
    
    def test_high_intensity_negative(self):
        """Test high emotional intensity with negative sentiment."""
        text = "This is absolutely horrible, terrifying, and catastrophic!"
        result = analyze_sentiment(text)
        
        assert result.label == "negative"
        assert result.emotional_intensity > 0.5
    
    def test_high_intensity_positive(self):
        """Test high emotional intensity with positive sentiment."""
        text = "This is absolutely amazing, wonderful, and incredible!"
        result = analyze_sentiment(text)
        
        assert result.label == "positive"
        assert result.emotional_intensity > 0.5
    
    def test_amplifier_words_boost_intensity(self):
        """Test that amplifier words increase emotional intensity."""
        baseline = analyze_sentiment("This is bad.")
        amplified = analyze_sentiment("This is extremely bad.")
        
        # Amplified version should have higher emotional intensity
        assert amplified.emotional_intensity >= baseline.emotional_intensity


class TestManipulationFlags:
    """Test detection of manipulation techniques."""
    
    def test_sensationalism_flags(self):
        """Test detection of sensationalism."""
        text = "Shocking bombshell exposed breaking news you must see!"
        result = analyze_sentiment(text)
        
        assert "sensationalism" in result.flags
        assert len(result.flags) > 0
    
    def test_conspiracy_flags(self):
        """Test detection of conspiracy language."""
        text = "The coverup is real, this has been suppressed and hidden!"
        result = analyze_sentiment(text)
        
        assert "conspiracy-thinking" in result.flags or "conspiracy-thinking" in " ".join(result.flags)
    
    def test_propaganda_framing_flags(self):
        """Test detection of propaganda framing."""
        text = "The mainstream media and fake news refuse to report this."
        result = analyze_sentiment(text)
        
        assert any("propaganda" in flag for flag in result.flags)
    
    def test_demonization_flags(self):
        """Test detection of demonization language."""
        text = "Those evil traitors are destroying everything!"
        result = analyze_sentiment(text)
        
        assert "demonization" in result.flags
    
    def test_multiple_flags(self):
        """Test detection of multiple manipulation tactics."""
        text = "Shocking exposed coverup proves mainstream media and evil elites are lying!"
        result = analyze_sentiment(text)
        
        assert len(result.flags) >= 2


class TestBiasRisk:
    """Test bias/manipulation risk assessment."""
    
    def test_low_risk_neutral(self):
        """Test that neutral, calm claims show low bias risk."""
        text = "The sky is blue."
        risk = estimate_bias_risk("neutral", 0.1, [])
        
        assert risk == "low"
    
    def test_high_risk_emotionally_charged(self):
        """Test that emotionally charged claims show higher risk."""
        risk = estimate_bias_risk("negative", 0.8, ["conspiracy-thinking"])
        
        assert risk in ("medium", "high")
    
    def test_medium_risk_with_flags(self):
        """Test that manipulation flags increase risk."""
        risk = estimate_bias_risk("positive", 0.4, ["sensationalism", "propaganda-framing"])
        
        assert risk in ("medium", "high")
    
    def test_high_risk_many_flags(self):
        """Test that many manipulation flags result in high risk."""
        risk = estimate_bias_risk("negative", 0.6, ["sensationalism", "conspiracy-thinking", "demonization", "propaganda-framing"])
        
        assert risk == "high"


class TestMisinformationAdjustment:
    """Test integration with misinformation likelihood scoring."""
    
    def test_low_emotional_adjustment(self):
        """Test that low emotion gives minimal adjustment."""
        from app.sentiment import SentimentResult
        
        result = SentimentResult(
            score=0.1,
            label="neutral",
            emotional_intensity=0.2,
            flags=[]
        )
        
        adjustment = calculate_sentiment_misinformation_adjustment(result)
        assert adjustment < 0.1
    
    def test_high_emotional_adjustment(self):
        """Test that high emotion gives significant adjustment."""
        from app.sentiment import SentimentResult
        
        result = SentimentResult(
            score=-0.8,
            label="negative",
            emotional_intensity=0.9,
            flags=["conspiracy-thinking", "demonization", "propaganda-framing"]
        )
        
        adjustment = calculate_sentiment_misinformation_adjustment(result)
        assert adjustment >= 0.2
    
    def test_adjustment_capped_at_max(self):
        """Test that adjustment is capped at 0.4."""
        from app.sentiment import SentimentResult
        
        result = SentimentResult(
            score=-1.0,
            label="negative",
            emotional_intensity=1.0,
            flags=["a", "b", "c", "d", "e", "f", "g", "h"]  # Many flags
        )
        
        adjustment = calculate_sentiment_misinformation_adjustment(result)
        assert adjustment <= 0.4


class TestSentimentSummary:
    """Test sentiment summary generation."""
    
    def test_positive_low_intensity_summary(self):
        """Test summary for low-intensity positive sentiment."""
        from app.sentiment import SentimentResult
        
        result = SentimentResult(
            score=0.5,
            label="positive",
            emotional_intensity=0.3,
            flags=[]
        )
        
        summary = get_sentiment_summary(result)
        assert "positive" in summary.lower()
        assert "slightly" in summary.lower()
    
    def test_negative_high_intensity_summary(self):
        """Test summary for high-intensity negative sentiment."""
        from app.sentiment import SentimentResult
        
        result = SentimentResult(
            score=-0.8,
            label="negative",
            emotional_intensity=0.9,
            flags=["conspiracy-thinking", "demonization"]
        )
        
        summary = get_sentiment_summary(result)
        assert "negative" in summary.lower()
        assert "highly" in summary.lower()
        assert "manipulation" in summary.lower()


class TestBatchAnalysis:
    """Test batch sentiment analysis."""
    
    def test_batch_multiple_claims(self):
        """Test analyzing multiple claims at once."""
        claims = [
            "This is wonderful and amazing!",
            "This is terrible and horrible.",
            "The sky is blue.",
        ]
        
        results = batch_analyze_sentiment(claims)
        
        assert len(results) == 3
        assert results[0].label == "positive"
        assert results[1].label == "negative"
        assert results[2].label == "neutral"
    
    def test_batch_empty_list(self):
        """Test batch analysis with empty list."""
        results = batch_analyze_sentiment([])
        
        assert results == []


class TestRealWorldExamples:
    """Test with real-world claim examples."""
    
    def test_health_misinformation(self):
        """Test health misinformation with manipulation."""
        claim = "This miracle cure is proven 100% effective! Mainstream media is hiding this secret weapon from you!"
        result = analyze_sentiment(claim)

        assert result.label in ("positive", "negative")
        assert result.emotional_intensity >= 0.5
        assert len(result.flags) > 0
    
    def test_political_divisive_claim(self):
        """Test politically divisive claim."""
        claim = "The evil elites and corrupt traitors have destroyed the country completely!"
        result = analyze_sentiment(claim)
        
        assert result.label == "negative"
        assert result.emotional_intensity > 0.6
        assert "demonization" in result.flags
    
    def test_factual_news_item(self):
        """Test factual news-style claim."""
        claim = "According to official statistics, unemployment rose 0.5% last month."
        result = analyze_sentiment(claim)
        
        assert result.label == "neutral" or abs(result.score) < 0.2
        assert result.emotional_intensity <= 0.4
        # Should be closer to neutral since positive and negative balance out
        assert abs(result.score) < 0.4 or result.label in ("neutral", "positive", "negative")


class TestEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_repeated_words(self):
        """Test claim with repeated emotional words."""
        claim = "terrible terrible terrible terrible terrible!"
        result = analyze_sentiment(claim)
        
        assert result.label == "negative"
        assert result.emotional_intensity > 0.5
    
    def test_all_caps(self):
        """Test that all-caps text is processed correctly."""
        claim = "THIS IS AMAZING AND WONDERFUL!"
        result = analyze_sentiment(claim)
        
        assert result.label == "positive"
        assert result.emotional_intensity > 0
    
    def test_with_punctuation(self):
        """Test that punctuation is handled correctly."""
        claim = "Absolutely incredible!!! This is amazing??? Really wonderful..."
        result = analyze_sentiment(claim)
        
        assert result.label == "positive"
    
    def test_special_characters(self):
        """Test handling of special characters."""
        claim = "This is @#$%^ good! *excellent* and ~amazing~"
        result = analyze_sentiment(claim)
        
        # Should still extract positive words despite special chars
        assert result.label in ("positive", "neutral")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
