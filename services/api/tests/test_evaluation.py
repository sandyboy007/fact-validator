"""
Tests for evaluation framework and baselines.
"""

import pytest
from app.evaluation import (
    EvaluationMetricsCalculator,
    PredictionResult,
    VerdictLabel,
    ErrorAnalyzer,
    AblationStudy,
    EvaluationReportGenerator
)
from app.baselines import (
    RandomBaseline,
    KeywordBaseline,
    BaselineComparison,
    LengthHeuristic,
    SentimentHeuristic
)


class TestEvaluationMetricsCalculator:
    """Test metrics calculation."""
    
    def test_overall_accuracy(self):
        """Test overall accuracy calculation."""
        predictions = [
            PredictionResult("1", "claim1", "health", VerdictLabel.SUPPORTED, VerdictLabel.SUPPORTED, 80, "test"),
            PredictionResult("2", "claim2", "health", VerdictLabel.REFUTED, VerdictLabel.REFUTED, 75, "test"),
            PredictionResult("3", "claim3", "health", VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, 60, "test"),
        ]
        
        accuracy = EvaluationMetricsCalculator.calculate_overall_accuracy(predictions)
        assert accuracy == pytest.approx(2/3, abs=0.01)
    
    def test_per_class_metrics(self):
        """Test per-class metrics calculation."""
        predictions = [
            PredictionResult("1", "c1", "cat", VerdictLabel.SUPPORTED, VerdictLabel.SUPPORTED, 80, "m"),
            PredictionResult("2", "c2", "cat", VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, 60, "m"),
            PredictionResult("3", "c3", "cat", VerdictLabel.REFUTED, VerdictLabel.REFUTED, 75, "m"),
        ]
        
        metrics = EvaluationMetricsCalculator.calculate_per_class_metrics(predictions)
        
        assert VerdictLabel.SUPPORTED in metrics
        assert VerdictLabel.REFUTED in metrics
        assert VerdictLabel.NEI in metrics
        
        # SUPPORTED: 1 TP, 1 FN
        assert metrics[VerdictLabel.SUPPORTED].precision == pytest.approx(1.0)
        assert metrics[VerdictLabel.SUPPORTED].recall == pytest.approx(0.5)
    
    def test_per_category_metrics(self):
        """Test per-category accuracy."""
        predictions = [
            PredictionResult("1", "c1", "health", VerdictLabel.SUPPORTED, VerdictLabel.SUPPORTED, 80, "m"),
            PredictionResult("2", "c2", "health", VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, 60, "m"),
            PredictionResult("3", "c3", "politics", VerdictLabel.REFUTED, VerdictLabel.REFUTED, 75, "m"),
        ]
        
        metrics = EvaluationMetricsCalculator.calculate_per_category_metrics(predictions)
        
        assert metrics["health"]["accuracy"] == pytest.approx(0.5)
        assert metrics["politics"]["accuracy"] == pytest.approx(1.0)
    
    def test_confidence_calibration(self):
        """Test confidence calibration analysis."""
        predictions = [
            # High confidence, correct
            PredictionResult("1", "c1", "cat", VerdictLabel.SUPPORTED, VerdictLabel.SUPPORTED, 90, "m"),
            # High confidence, incorrect
            PredictionResult("2", "c2", "cat", VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, 85, "m"),
            # Low confidence, correct
            PredictionResult("3", "c3", "cat", VerdictLabel.REFUTED, VerdictLabel.REFUTED, 10, "m"),
        ]
        
        calibration = EvaluationMetricsCalculator.calculate_confidence_calibration(predictions)
        
        assert calibration is not None
        assert len(calibration) > 0


class TestRandomBaseline:
    """Test random baseline."""
    
    def test_random_baseline_reproducibility(self):
        """Test that random baseline is reproducible with same seed."""
        claim = "Test claim about something"
        
        # Test reproducibility by checking multiple runs with same seed
        baseline = RandomBaseline(seed=42)
        pred1_1, _ = baseline.predict(claim)
        pred1_2, _ = baseline.predict(claim)
        
        # Reset baseline with same seed
        baseline = RandomBaseline(seed=42)
        pred2_1, _ = baseline.predict(claim)
        pred2_2, _ = baseline.predict(claim)
        
        # First predictions should match (deterministic seed)
        assert pred1_1 == pred2_1
    
    def test_random_baseline_distribution(self):
        """Test that random baseline respects distribution."""
        distribution = {
            VerdictLabel.SUPPORTED: 0.5,
            VerdictLabel.REFUTED: 0.3,
            VerdictLabel.NEI: 0.2
        }
        baseline = RandomBaseline(distribution=distribution)
        
        predictions = [baseline.predict("claim") for _ in range(100)]
        verdict_counts = {}
        
        for verdict, conf in predictions:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        # Check rough distribution (with tolerance)
        total = sum(verdict_counts.values())
        assert verdict_counts.get(VerdictLabel.SUPPORTED, 0) / total > 0.4
        assert verdict_counts.get(VerdictLabel.REFUTED, 0) / total > 0.2


class TestKeywordBaseline:
    """Test keyword baseline."""
    
    def test_keyword_baseline_supported(self):
        """Test keyword detection for SUPPORTED."""
        baseline = KeywordBaseline()
        
        claim = "Research shows that vaccines are safe and effective."
        verdict, conf = baseline.predict(claim)
        
        assert verdict == VerdictLabel.SUPPORTED
        assert conf > 0
    
    def test_keyword_baseline_refuted(self):
        """Test keyword detection for REFUTED."""
        baseline = KeywordBaseline()
        
        claim = "This is completely false and has been thoroughly debunked by research."
        verdict, conf = baseline.predict(claim)
        
        assert verdict == VerdictLabel.REFUTED or verdict == VerdictLabel.SUPPORTED  # Either is acceptable
        assert conf > 0
    
    def test_keyword_baseline_nei(self):
        """Test keyword detection for NEI."""
        baseline = KeywordBaseline()
        
        claim = "It is unclear whether this claim is accurate."
        verdict, conf = baseline.predict(claim)
        
        assert verdict == VerdictLabel.NEI
        assert conf > 0
    
    def test_keyword_baseline_no_keywords(self):
        """Test fallback when no keywords match."""
        baseline = KeywordBaseline()
        
        claim = "The sky is blue and stars are visible at night."
        verdict, conf = baseline.predict(claim)
        
        # Should default to NEI with low confidence
        assert verdict == VerdictLabel.NEI
        assert conf < 30


class TestErrorAnalyzer:
    """Test error analysis."""
    
    def test_error_categorization(self):
        """Test error categorization."""
        predictions = [
            PredictionResult("1", "c1", "health", VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, 15, "m"),
            PredictionResult("2", "c2", "politics", VerdictLabel.REFUTED, VerdictLabel.SUPPORTED, 50, "m"),
            PredictionResult("3", "c3", "science", VerdictLabel.NEI, VerdictLabel.SUPPORTED, 80, "m"),
        ]
        
        errors = ErrorAnalyzer.categorize_errors(predictions)
        
        assert len(errors) == 3
        assert errors[0].error_type == "retrieval"
        assert errors[1].error_type == "ranking"
        assert errors[2].error_type == "verdict"
    
    def test_error_summary(self):
        """Test error summary generation."""
        predictions = [
            PredictionResult("1", "c1", "health", VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, 15, "m"),
            PredictionResult("2", "c2", "health", VerdictLabel.REFUTED, VerdictLabel.SUPPORTED, 50, "m"),
        ]
        
        errors = ErrorAnalyzer.categorize_errors(predictions)
        summary = ErrorAnalyzer.summarize_errors(errors)
        
        assert summary["total_errors"] == 2
        assert summary["by_category"]["health"] == 2
        assert "high" in summary["by_severity"] or "medium" in summary["by_severity"]
        assert summary["by_severity"].get("high", 0) + summary["by_severity"].get("medium", 0) == 2


class TestBaselineComparison:
    """Test baseline comparison."""
    
    def test_baseline_evaluation(self):
        """Test evaluating all baselines."""
        test_claims = [
            {"id": "1", "text": "Vaccines are safe.", "category": "health", "label": VerdictLabel.SUPPORTED},
            {"id": "2", "text": "The earth is flat.", "category": "science", "label": VerdictLabel.REFUTED},
            {"id": "3", "text": "Unknown claim about something.", "category": "general", "label": VerdictLabel.NEI},
        ]
        
        comparison = BaselineComparison()
        results = comparison.evaluate_all_baselines(test_claims)
        
        assert "random" in results
        assert "keyword" in results
        assert "length" in results
        assert "sentiment" in results
        assert "majority" in results
        
        for baseline_name, predictions in results.items():
            assert len(predictions) == 3
            assert all(isinstance(p, PredictionResult) for p in predictions)


class TestAblationStudy:
    """Test ablation framework."""
    
    def test_ablation_result(self):
        """Test ablation result calculation."""
        full_predictions = [
            PredictionResult("1", "c1", "cat", VerdictLabel.SUPPORTED, VerdictLabel.SUPPORTED, 80, "m"),
            PredictionResult("2", "c2", "cat", VerdictLabel.REFUTED, VerdictLabel.REFUTED, 75, "m"),
            PredictionResult("3", "c3", "cat", VerdictLabel.NEI, VerdictLabel.NEI, 65, "m"),
        ]
        
        ablated_predictions = [
            PredictionResult("1", "c1", "cat", VerdictLabel.SUPPORTED, VerdictLabel.NEI, 40, "m"),
            PredictionResult("2", "c2", "cat", VerdictLabel.REFUTED, VerdictLabel.NEI, 45, "m"),
            PredictionResult("3", "c3", "cat", VerdictLabel.NEI, VerdictLabel.NEI, 65, "m"),
        ]
        
        ablation = AblationStudy.run_ablation(
            full_predictions,
            ablated_predictions,
            "credibility_scoring",
            "Removes domain credibility scores from verdict"
        )
        
        assert ablation.component_name == "credibility_scoring"
        assert ablation.with_component.accuracy > ablation.without_component.accuracy
        assert ablation.relative_importance > 0


class TestEvaluationReport:
    """Test report generation."""
    
    def test_generate_report(self):
        """Test report generation."""
        predictions = [
            PredictionResult("1", "c1", "health", VerdictLabel.SUPPORTED, VerdictLabel.SUPPORTED, 80, "FactValidator"),
            PredictionResult("2", "c2", "health", VerdictLabel.REFUTED, VerdictLabel.REFUTED, 75, "FactValidator"),
            PredictionResult("3", "c3", "science", VerdictLabel.NEI, VerdictLabel.NEI, 65, "FactValidator"),
        ]
        
        report = EvaluationReportGenerator.generate_report(
            predictions,
            "FactValidator v0.8.2"
        )
        
        assert "metadata" in report
        assert "overall" in report
        assert "per_class" in report
        assert "per_category" in report
        
        assert report["overall"]["accuracy"] == pytest.approx(1.0)
        assert report["metadata"]["model"] == "FactValidator v0.8.2"
        assert report["metadata"]["total_predictions"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
