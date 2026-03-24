"""
Statistical analysis module for rigorous evaluation.

Provides:
- Confidence intervals (95%)
- Significance testing (paired t-test, one-sample)
- Effect size calculations (Cohen's d, Hedges' g)
- Bootstrap resampling for robust estimates
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    mean: float
    lower: float
    upper: float
    confidence_level: float
    margin_of_error: float
    
    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class SignificanceTestResult:
    """Result from significance test."""
    test_name: str
    t_statistic: float
    p_value: float
    degrees_freedom: int
    mean_difference: float
    is_significant: bool  # At alpha=0.05
    effect_size: float  # Cohen's d or similar
    
    def __str__(self) -> str:
        sig_marker = "✓" if self.is_significant else "✗"
        return f"{sig_marker} {self.test_name}: t({self.degrees_freedom})={self.t_statistic:.3f}, p={self.p_value:.4f}, d={self.effect_size:.3f}"


class StatisticalAnalyzer:
    """Rigorous statistical analysis for evaluation."""
    
    @staticmethod
    def confidence_interval(
        samples: List[float],
        confidence_level: float = 0.95,
        method: str = "t"
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval using t-distribution.
        
        Args:
            samples: List of accuracy scores or metrics
            confidence_level: 0.95 for 95% CI
            method: "t" for t-distribution (preferred for small n), "z" for normal
        
        Returns:
            ConfidenceInterval object
        """
        n = len(samples)
        mean = np.mean(samples)
        std_err = np.std(samples, ddof=1) / np.sqrt(n)
        
        if method == "t":
            alpha = 1 - confidence_level
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        else:  # method == "z"
            alpha = 1 - confidence_level
            t_crit = stats.norm.ppf(1 - alpha/2)
        
        margin_error = t_crit * std_err
        
        return ConfidenceInterval(
            mean=mean,
            lower=mean - margin_error,
            upper=mean + margin_error,
            confidence_level=confidence_level,
            margin_of_error=margin_error
        )
    
    @staticmethod
    def bootstrap_ci(
        samples: List[float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        metric_fn=None
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval using bootstrap resampling.
        
        More robust for non-normal distributions.
        
        Args:
            samples: Original data
            confidence_level: 0.95 for 95% CI
            n_bootstrap: Number of bootstrap samples
            metric_fn: Function to compute on each bootstrap sample (default: mean)
        
        Returns:
            ConfidenceInterval object
        """
        if metric_fn is None:
            metric_fn = np.mean
        
        n = len(samples)
        np.random.seed(42)  # Reproducibility
        
        bootstrap_metrics = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(samples, size=n, replace=True)
            metric = metric_fn(bootstrap_sample)
            bootstrap_metrics.append(metric)
        
        mean = np.mean(samples)
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha/2) * 100
        
        lower = np.percentile(bootstrap_metrics, lower_percentile)
        upper = np.percentile(bootstrap_metrics, upper_percentile)
        margin_error = (upper - lower) / 2
        
        return ConfidenceInterval(
            mean=mean,
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            margin_of_error=margin_error
        )
    
    @staticmethod
    def paired_t_test(
        system_scores: List[float],
        baseline_scores: List[float],
        alternative: str = "greater",  # greater, less, two-sided
        alpha: float = 0.05
    ) -> SignificanceTestResult:
        """
        Paired t-test: is system significantly better than baseline?
        
        Args:
            system_scores: Accuracy scores from system
            baseline_scores: Accuracy scores from baseline
            alternative: "greater" (system > baseline), "less", "two-sided"
            alpha: Significance level (0.05 for 95% confidence)
        
        Returns:
            SignificanceTestResult
        """
        assert len(system_scores) == len(baseline_scores)
        
        # Compute paired differences
        diffs = np.array(system_scores) - np.array(baseline_scores)
        n = len(diffs)
        
        # T-statistic
        mean_diff = np.mean(diffs)
        std_err = np.std(diffs, ddof=1) / np.sqrt(n)
        
        if std_err == 0:
            t_stat = float('inf') if mean_diff > 0 else float('-inf')
        else:
            t_stat = mean_diff / std_err
        
        df = n - 1
        
        # P-value
        if alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, df)
        elif alternative == "less":
            p_value = stats.t.cdf(t_stat, df)
        else:  # two-sided
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Effect size (Cohen's d)
        cohens_d = StatisticalAnalyzer.cohens_d(system_scores, baseline_scores)
        
        is_significant = p_value < alpha
        
        return SignificanceTestResult(
            test_name="Paired t-test",
            t_statistic=t_stat,
            p_value=p_value,
            degrees_freedom=df,
            mean_difference=mean_diff,
            is_significant=is_significant,
            effect_size=cohens_d
        )
    
    @staticmethod
    def cohens_d(
        group1: List[float],
        group2: List[float],
        pooled: bool = True
    ) -> float:
        """
        Calculate Cohen's d effect size.
        
        Interpretation:
            |d| < 0.2: negligible
            0.2 ≤ |d| < 0.5: small
            0.5 ≤ |d| < 0.8: medium
            |d| ≥ 0.8: large
        
        Args:
            group1: First group scores
            group2: Second group scores
            pooled: Use pooled standard deviation (True for equal variance assumption)
        
        Returns:
            Cohen's d value
        """
        mean_diff = np.mean(group1) - np.mean(group2)
        
        if pooled:
            # Pooled standard deviation
            n1, n2 = len(group1), len(group2)
            var1 = np.var(group1, ddof=1)
            var2 = np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        else:
            pooled_std = np.std(np.concatenate([group1, group2]), ddof=1)
        
        if pooled_std == 0:
            return 0
        
        return mean_diff / pooled_std
    
    @staticmethod
    def hedges_g(
        group1: List[float],
        group2: List[float]
    ) -> float:
        """
        Calculate Hedges' g (bias-corrected Cohen's d).
        
        Better for small samples.
        
        Args:
            group1: First group scores
            group2: Second group scores
        
        Returns:
            Hedges' g value
        """
        cohens_d = StatisticalAnalyzer.cohens_d(group1, group2, pooled=True)
        n1, n2 = len(group1), len(group2)
        n = n1 + n2
        
        # Bias correction
        correction = 1 - (3 / (4 * n - 9))
        
        return cohens_d * correction
    
    @staticmethod
    def one_sample_t_test(
        samples: List[float],
        null_value: float = 0.5,  # Null hypothesis: accuracy = 0.5
        alternative: str = "greater",
        alpha: float = 0.05
    ) -> SignificanceTestResult:
        """
        One-sample t-test: is mean significantly different from null value?
        
        Useful for: "Is system accuracy > 0.5 (random baseline)?"
        
        Args:
            samples: Observed accuracy scores
            null_value: Hypothesized population mean
            alternative: "greater", "less", "two-sided"
            alpha: Significance level
        
        Returns:
            SignificanceTestResult
        """
        n = len(samples)
        mean = np.mean(samples)
        std_err = np.std(samples, ddof=1) / np.sqrt(n)
        
        if std_err == 0:
            t_stat = float('inf') if mean > null_value else float('-inf')
        else:
            t_stat = (mean - null_value) / std_err
        
        df = n - 1
        
        # P-value
        if alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, df)
        elif alternative == "less":
            p_value = stats.t.cdf(t_stat, df)
        else:  # two-sided
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        Cohen_d = (mean - null_value) / np.std(samples, ddof=1)
        
        is_significant = p_value < alpha
        
        return SignificanceTestResult(
            test_name="One-sample t-test",
            t_statistic=t_stat,
            p_value=p_value,
            degrees_freedom=df,
            mean_difference=mean - null_value,
            is_significant=is_significant,
            effect_size=Cohen_d
        )
    
    @staticmethod
    def mann_whitney_u(
        group1: List[float],
        group2: List[float],
        alternative: str = "greater"
    ) -> Dict:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).
        
        Use when data is not normally distributed.
        
        Args:
            group1: First group scores
            group2: Second group scores
            alternative: "greater", "less", "two-sided"
        
        Returns:
            Test result dictionary
        """
        u_stat, p_value = stats.mannwhitneyu(
            group1, group2,
            alternative=alternative
        )
        
        return {
            "test": "Mann-Whitney U",
            "u_statistic": u_stat,
            "p_value": p_value,
            "n1": len(group1),
            "n2": len(group2),
            "is_significant": p_value < 0.05
        }
    
    @staticmethod
    def effect_size_interpretation(cohens_d: float) -> str:
        """Interpret Cohen's d magnitude."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


@dataclass
class ComparisonResult:
    """Complete comparison result (system vs baseline)."""
    system_name: str
    baseline_name: str
    system_accuracy: float
    baseline_accuracy: float
    improvement_pct: float
    ci_system: ConfidenceInterval
    ci_baseline: ConfidenceInterval
    significance_test: SignificanceTestResult
    effect_size: float
    effect_interpretation: str
    
    def generate_report(self) -> str:
        """Generate formatted comparison report."""
        return f"""
=== COMPARISON: {self.system_name} vs {self.baseline_name} ===

Accuracy:
  {self.system_name}: {self.system_accuracy:.1%} CI95 [{self.ci_system.lower:.1%}, {self.ci_system.upper:.1%}]
  {self.baseline_name}: {self.baseline_accuracy:.1%} CI95 [{self.ci_baseline.lower:.1%}, {self.ci_baseline.upper:.1%}]

Improvement:
  Δ Accuracy: +{self.improvement_pct:.2f} percentage points
  
Statistical Significance:
  {self.significance_test}
  
Effect Size:
  Cohen's d: {self.effect_size:.3f} ({self.effect_interpretation} effect)
  
Interpretation:
  The {self.system_name} system is {self.effect_interpretation} better than {self.baseline_name},
  with statistical significance p={self.significance_test.p_value:.4f}.
"""


class ComparisonAnalyzer:
    """Analyze system vs baseline comparison."""
    
    @staticmethod
    def compare_system_vs_baseline(
        system_predictions: List[float],  # Accuracy scores per prediction
        baseline_predictions: List[float],
        system_name: str = "System",
        baseline_name: str = "Baseline",
        alpha: float = 0.05
    ) -> ComparisonResult:
        """
        Complete statistical comparison of system vs baseline.
        
        Args:
            system_predictions: List of correct/incorrect (1/0) or accuracy scores
            baseline_predictions: Corresponding baseline results
            system_name: Name for system
            baseline_name: Name for baseline
            alpha: Significance level
        
        Returns:
            ComparisonResult with full analysis
        """
        analyzer = StatisticalAnalyzer()
        
        # Confidence intervals
        ci_system = analyzer.confidence_interval(system_predictions)
        ci_baseline = analyzer.confidence_interval(baseline_predictions)
        
        # Significance test
        sig_test = analyzer.paired_t_test(system_predictions, baseline_predictions, alpha=alpha)
        
        # Effect size
        effect_size = analyzer.cohens_d(system_predictions, baseline_predictions)
        effect_interp = analyzer.effect_size_interpretation(effect_size)
        
        improvement_pct = (ci_system.mean - ci_baseline.mean) * 100
        
        return ComparisonResult(
            system_name=system_name,
            baseline_name=baseline_name,
            system_accuracy=ci_system.mean,
            baseline_accuracy=ci_baseline.mean,
            improvement_pct=improvement_pct,
            ci_system=ci_system,
            ci_baseline=ci_baseline,
            significance_test=sig_test,
            effect_size=effect_size,
            effect_interpretation=effect_interp
        )
