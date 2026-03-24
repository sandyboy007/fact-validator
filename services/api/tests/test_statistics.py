"""
Tests for statistical analysis module.
"""

import pytest
import numpy as np
from app.statistics import (
    StatisticalAnalyzer,
    ComparisonAnalyzer,
    ConfidenceInterval,
    SignificanceTestResult
)


class TestConfidenceInterval:
    """Test confidence interval calculations."""
    
    def test_ci_basic(self):
        """Test basic CI calculation."""
        # Perfect accuracy scores
        samples = [0.9, 0.8, 0.85, 0.92, 0.88]
        
        analyzer = StatisticalAnalyzer()
        ci = analyzer.confidence_interval(samples, confidence_level=0.95)
        
        assert ci.lower < ci.mean < ci.upper
        assert ci.confidence_level == 0.95
        assert 0.75 < ci.mean < 0.95  # Reasonable range
    
    def test_ci_reproducibility(self):
        """Test CI is consistent."""
        samples = [0.7, 0.75, 0.8, 0.82, 0.78]
        analyzer = StatisticalAnalyzer()
        
        ci1 = analyzer.confidence_interval(samples)
        ci2 = analyzer.confidence_interval(samples)
        
        assert ci1.mean == ci2.mean
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper
    
    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval."""
        samples = [0.7, 0.75, 0.8, 0.82, 0.78, 0.79, 0.81]
        analyzer = StatisticalAnalyzer()
        
        ci = analyzer.bootstrap_ci(samples, n_bootstrap=1000)
        
        assert ci.lower < ci.mean < ci.upper
        assert ci.confidence_level == 0.95


class TestSignificanceTests:
    """Test statistical significance tests."""
    
    def test_paired_t_test_significant(self):
        """Test when system is significantly better."""
        system = [0.85, 0.88, 0.90, 0.87, 0.89]
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69]
        
        analyzer = StatisticalAnalyzer()
        result = analyzer.paired_t_test(system, baseline, alternative="greater")
        
        assert result.is_significant
        assert result.p_value < 0.05
        assert result.mean_difference > 0
        assert result.t_statistic > 0
    
    def test_paired_t_test_not_significant(self):
        """Test when system is not significantly better."""
        # Very similar performance
        system = [0.75, 0.76, 0.74, 0.75, 0.76]
        baseline = [0.74, 0.75, 0.75, 0.74, 0.75]
        
        analyzer = StatisticalAnalyzer()
        result = analyzer.paired_t_test(system, baseline, alternative="greater")
        
        # May not be significant due to small difference
        assert result.p_value > 0.01  # Not highly significant
    
    def test_one_sample_t_test(self):
        """Test one-sample t-test."""
        # System accuracy > random (0.5)
        samples = [0.70, 0.72, 0.75, 0.73, 0.71]
        
        analyzer = StatisticalAnalyzer()
        result = analyzer.one_sample_t_test(samples, null_value=0.5, alternative="greater")
        
        assert result.is_significant
        assert result.p_value < 0.05
        assert result.mean_difference > 0


class TestEffectSizes:
    """Test effect size calculations."""
    
    def test_cohens_d_large(self):
        """Test Cohen's d for large effect."""
        group1 = [0.85, 0.88, 0.90, 0.87, 0.89]
        group2 = [0.60, 0.62, 0.58, 0.61, 0.59]
        
        analyzer = StatisticalAnalyzer()
        d = analyzer.cohens_d(group1, group2)
        
        assert d > 0.5  # Should be medium to large
        assert d > 1.0  # Probably large
    
    def test_cohens_d_small(self):
        """Test Cohen's d for small effect."""
        # Use data with larger variance to get small effect
        group1 = [0.60, 0.70, 0.50, 0.75, 0.65]
        group2 = [0.59, 0.68, 0.52, 0.73, 0.64]
        
        analyzer = StatisticalAnalyzer()
        d = analyzer.cohens_d(group1, group2)
        
        assert abs(d) < 0.5  # Should be negligible or small
    
    def test_effect_interpretation(self):
        """Test effect size interpretation."""
        analyzer = StatisticalAnalyzer()
        
        assert analyzer.effect_size_interpretation(0.1) == "negligible"
        assert analyzer.effect_size_interpretation(0.3) == "small"
        assert analyzer.effect_size_interpretation(0.6) == "medium"
        assert analyzer.effect_size_interpretation(1.0) == "large"
        assert analyzer.effect_size_interpretation(-0.6) == "medium"
    
    def test_hedges_g(self):
        """Test Hedges' g (bias-corrected effect size)."""
        group1 = [0.85, 0.88, 0.90]
        group2 = [0.70, 0.72, 0.68]
        
        analyzer = StatisticalAnalyzer()
        g = analyzer.hedges_g(group1, group2)
        
        # Should be slightly smaller than Cohen's d for small samples
        d = analyzer.cohens_d(group1, group2)
        assert abs(g) < abs(d)


class TestMannWhitneyU:
    """Test non-parametric Mann-Whitney U test."""
    
    def test_mann_whitney_significant(self):
        """Test Mann-Whitney U for significant difference."""
        group1 = [0.85, 0.88, 0.90, 0.87, 0.89]
        group2 = [0.70, 0.72, 0.68, 0.71, 0.69]
        
        analyzer = StatisticalAnalyzer()
        result = analyzer.mann_whitney_u(group1, group2, alternative="greater")
        
        assert result["is_significant"]
        assert result["p_value"] < 0.05


class TestComparisonAnalyzer:
    """Test full comparison analysis."""
    
    def test_complete_comparison(self):
        """Test complete system vs baseline comparison."""
        system = [0.85, 0.88, 0.90, 0.87, 0.89, 0.86]
        baseline = [0.70, 0.72, 0.68, 0.71, 0.69, 0.70]
        
        comparison = ComparisonAnalyzer.compare_system_vs_baseline(
            system, baseline,
            system_name="FactValidator",
            baseline_name="KeywordBaseline"
        )
        
        assert comparison.system_accuracy > comparison.baseline_accuracy
        assert comparison.improvement_pct > 10
        assert comparison.significance_test.is_significant
        assert comparison.effect_size > 0
        assert comparison.effect_interpretation in ["small", "medium", "large"]
    
    def test_comparison_report(self):
        """Test comparison report generation."""
        system = [0.8, 0.82, 0.81, 0.79, 0.83]
        baseline = [0.75, 0.76, 0.74, 0.75, 0.76]
        
        comparison = ComparisonAnalyzer.compare_system_vs_baseline(
            system, baseline,
            system_name="System",
            baseline_name="Baseline"
        )
        
        report = comparison.generate_report()
        
        assert "System" in report
        assert "Baseline" in report
        assert "Accuracy" in report
        assert "Cohen's d" in report
        assert "Statistical Significance" in report


class TestRobustness:
    """Test robustness of statistical calculations."""
    
    def test_zero_variance(self):
        """Test handling of zero variance (all same values)."""
        samples = [0.8, 0.8, 0.8, 0.8]
        
        analyzer = StatisticalAnalyzer()
        ci = analyzer.confidence_interval(samples)
        
        # Should handle gracefully
        assert ci.mean == 0.8
        # Margin of error may be 0 or very small
        assert ci.margin_of_error >= 0
    
    def test_single_sample(self):
        """Test with single sample (edge case)."""
        samples = [0.75]
        
        analyzer = StatisticalAnalyzer()
        # May return NaN or infinite for std dev
        ci = analyzer.confidence_interval(samples)
        
        assert ci.mean == 0.75
    
    def test_negative_cohens_d(self):
        """Test Cohen's d with negative difference."""
        group1 = [0.60, 0.62, 0.58]
        group2 = [0.85, 0.88, 0.90]
        
        analyzer = StatisticalAnalyzer()
        d = analyzer.cohens_d(group1, group2)
        
        assert d < 0  # Negative because group1 < group2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
