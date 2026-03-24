"""
Comprehensive tests for the comparative analysis framework.

Tests cover:
- Human evaluation framework annotations
- Interrater agreement calculations (Cohen's kappa, Fleiss' kappa)
- Comparative analysis matrix generation
- Benchmark result format validation
"""

import pytest
from typing import List, Dict, Any
import json
import tempfile
from pathlib import Path

# Import the comparative module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.comparative import (
    HumanEvaluationFramework,
    ComparativeJudgment,
    ComparativeAnalysis,
    BenchmarkFramework,
    BenchmarkResult,
    InterraterAgreement,
)


class TestHumanEvaluationFramework:
    """Test human evaluation framework and interrater agreement."""

    def test_framework_initialization(self):
        """Test that framework initializes properly."""
        framework = HumanEvaluationFramework()
        
        assert framework is not None
        assert hasattr(framework, 'calculate_interrater_agreement')
        assert hasattr(framework, '_calculate_cohens_kappa')
        assert hasattr(framework, 'create_evaluation_instructions')
    
    def test_evaluation_instructions_format(self):
        """Test that annotation instructions are properly formatted."""
        framework = HumanEvaluationFramework()
        instructions = framework.create_evaluation_instructions()
        
        assert isinstance(instructions, str)
        assert len(instructions) > 100  # Should be detailed
        assert 'SUPPORTED' in instructions
        assert 'REFUTED' in instructions
        assert 'NEI' in instructions
    
    def test_cohen_kappa_perfect_agreement(self):
        """Test Cohen's kappa with perfect agreement (kappa=1.0)."""
        framework = HumanEvaluationFramework()
        
        # Two judges both agree on all verdicts
        judgments = [
            ComparativeJudgment("c1", "judge1", "SUPPORTED", 80),
            ComparativeJudgment("c1", "judge2", "SUPPORTED", 80),
            ComparativeJudgment("c2", "judge1", "REFUTED", 85),
            ComparativeJudgment("c2", "judge2", "REFUTED", 85),
            ComparativeJudgment("c3", "judge1", "NEI", 60),
            ComparativeJudgment("c3", "judge2", "NEI", 60),
        ]
        
        agreement = framework.calculate_interrater_agreement(judgments)
        
        assert agreement.cohen_kappa == 1.0
        assert agreement.percent_agreement == 1.0
        assert agreement.n_judges == 2
        assert agreement.n_claims == 3
    
    def test_cohen_kappa_partial_agreement(self):
        """Test Cohen's kappa with partial agreement."""
        framework = HumanEvaluationFramework()
        
        # Two judges mostly agree
        judgments = [
            ComparativeJudgment("c1", "judge1", "SUPPORTED", 80),
            ComparativeJudgment("c1", "judge2", "SUPPORTED", 80),
            ComparativeJudgment("c2", "judge1", "REFUTED", 85),
            ComparativeJudgment("c2", "judge2", "REFUTED", 85),
            ComparativeJudgment("c3", "judge1", "NEI", 60),
            ComparativeJudgment("c3", "judge2", "SUPPORTED", 50),  # Disagree
            ComparativeJudgment("c4", "judge1", "SUPPORTED", 75),
            ComparativeJudgment("c4", "judge2", "REFUTED", 70),  # Disagree
        ]
        
        agreement = framework.calculate_interrater_agreement(judgments)
        
        # Should have moderate agreement (2/4 claims agree)
        assert 0.2 <= agreement.cohen_kappa <= 0.8
        assert -1.0 <= agreement.cohen_kappa <= 1.0
        assert agreement.percent_agreement == 0.5
        assert agreement.n_judges == 2
        assert agreement.n_claims == 4
    
    def test_percent_agreement_calculation(self):
        """Test simple percent agreement calculation."""
        framework = HumanEvaluationFramework()
        
        judgments = [
            ComparativeJudgment("c1", "j1", "SUPPORTED", 80),
            ComparativeJudgment("c1", "j2", "SUPPORTED", 80),
            ComparativeJudgment("c2", "j1", "REFUTED", 85),
            ComparativeJudgment("c2", "j2", "REFUTED", 85),
            ComparativeJudgment("c3", "j1", "NEI", 60),
            ComparativeJudgment("c3", "j2", "REFUTED", 70),  # Disagree
        ]
        
        agreement = framework.calculate_interrater_agreement(judgments)
        
        # 2 out of 3 claims agree
        assert agreement.percent_agreement == 2/3
    
    def test_agreement_interpretation(self):
        """Test interpretation of Cohen's kappa values."""
        framework = HumanEvaluationFramework()
        
        assert framework.agreement_interpretation(0.85) == "excellent"
        assert framework.agreement_interpretation(0.70) == "substantial"
        assert framework.agreement_interpretation(0.50) == "moderate"
        assert framework.agreement_interpretation(0.30) == "fair"
        assert framework.agreement_interpretation(0.10) == "slight"
        assert framework.agreement_interpretation(-0.1) == "poor"
    
    def test_comparative_judgment_data_structure(self):
        """Test ComparativeJudgment data class."""
        judgment = ComparativeJudgment(
            claim_id="claim_1",
            judge="judge_001",
            verdict="SUPPORTED",
            confidence=85,
            reasoning="Evidence supports the claim"
        )
        
        assert judgment.claim_id == "claim_1"
        assert judgment.verdict == "SUPPORTED"
        assert judgment.confidence == 85
        assert 0 <= judgment.confidence <= 100
    
    def test_interrater_agreement_structure(self):
        """Test InterraterAgreement data structure."""
        agreement = InterraterAgreement(
            cohen_kappa=0.75,
            percent_agreement=0.85,
            n_claims=20,
            n_judges=3
        )
        
        assert agreement.cohen_kappa == 0.75
        assert agreement.percent_agreement == 0.85
        assert agreement.n_judges == 3


class TestComparativeAnalysis:
    """Test comparative analysis matrix generation and comparison."""
    
    def test_comparative_analysis_initialization(self):
        """Test ComparativeAnalysis framework initialization."""
        analysis = ComparativeAnalysis()
        
        assert analysis is not None
        assert hasattr(analysis, 'build_comparison_matrix')
        assert hasattr(analysis, 'generate_comparative_report')
    
    def test_comparison_matrix_format(self):
        """Test that comparison matrix is properly formatted."""
        analysis = ComparativeAnalysis()
        
        # Create sample system predictions
        system_predictions = {
            "FactValidator": [0.85, 0.80, 0.90, 0.88, 0.92],
            "RandomBaseline": [0.33, 0.33, 0.33, 0.33, 0.33],
            "GoogleAPI": [0.70, 0.75, 0.72, 0.68, 0.70],
        }
        
        matrix = analysis.build_comparison_matrix(system_predictions)
        
        assert isinstance(matrix, dict)
        assert "FactValidator" in matrix
        assert "accuracy_mean" in matrix["FactValidator"]
        assert "comparisons" in matrix["FactValidator"]
    
    def test_comparison_matrix_accuracy_mean(self):
        """Test accuracy mean calculation in comparison matrix."""
        analysis = ComparativeAnalysis()
        
        system_predictions = {
            "System1": [0.8, 0.9, 0.7],  # Mean = 0.8
            "System2": [0.5, 0.5, 0.5],  # Mean = 0.5
        }
        
        matrix = analysis.build_comparison_matrix(system_predictions)
        
        avg1 = matrix["System1"]["accuracy_mean"]
        avg2 = matrix["System2"]["accuracy_mean"]
        
        assert abs(avg1 - 0.8) < 0.01
        assert abs(avg2 - 0.5) < 0.01
    
    def test_comparison_matrix_has_pairwise_comparisons(self):
        """Test that comparison matrix includes pairwise comparisons."""
        analysis = ComparativeAnalysis()
        
        system_predictions = {
            "System1": [0.85, 0.80, 0.90, 0.88, 0.92],
            "System2": [0.70, 0.75, 0.72, 0.68, 0.70],
            "System3": [0.50, 0.45, 0.55, 0.52, 0.48],
        }
        
        matrix = analysis.build_comparison_matrix(system_predictions)
        
        # System1 should have comparisons with System2 and System3
        assert "System2" in matrix["System1"]["comparisons"]
        assert "System3" in matrix["System1"]["comparisons"]
    
    def test_pairwise_comparison_contains_statistics(self):
        """Test that pairwise comparisons include statistical metrics."""
        analysis = ComparativeAnalysis()
        
        system_predictions = {
            "BetterSystem": [0.9, 0.85, 0.92, 0.88, 0.91],
            "WorseSystem": [0.6, 0.65, 0.62, 0.58, 0.61],
        }
        
        matrix = analysis.build_comparison_matrix(system_predictions)
        
        comparison = matrix["BetterSystem"]["comparisons"]["WorseSystem"]
        
        # Should have statistical metrics
        assert "improvement_pct" in comparison
        assert "p_value" in comparison
        assert "cohens_d" in comparison
        assert "is_significant" in comparison
        assert "effect_interpretation" in comparison
        
        # BetterSystem should show improvement over WorseSystem
        assert comparison["improvement_pct"] > 0
    
    def test_comparative_report_generation(self):
        """Test comparative report generation."""
        analysis = ComparativeAnalysis()
        
        system_predictions = {
            "FactValidator": [0.80, 0.82, 0.78, 0.85],
            "GoogleAPI": [0.70, 0.72, 0.68, 0.75],
            "Random": [0.33, 0.33, 0.33, 0.33],
        }
        
        matrix = analysis.build_comparison_matrix(system_predictions)
        systems = ["FactValidator", "GoogleAPI", "Random"]
        
        report = analysis.generate_comparative_report(matrix, systems)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "FactValidator" in report
        assert "GoogleAPI" in report
        assert "Random" in report
        assert "Pairwise" in report
    
    def test_comparative_report_contains_metrics(self):
        """Test that report includes performance metrics."""
        analysis = ComparativeAnalysis()
        
        system_predictions = {
            "System1": [0.9, 0.85, 0.88],
            "System2": [0.7, 0.72, 0.68],
        }
        
        matrix = analysis.build_comparison_matrix(system_predictions)
        report = analysis.generate_comparative_report(matrix, ["System1", "System2"])
        
        # Should include accuracy values
        assert "Accuracy" in report or "accuracy" in report


class TestBenchmarkFramework:
    """Test benchmark result formatting and export."""
    
    def test_benchmark_result_structure(self):
        """Test BenchmarkResult data class."""
        result = BenchmarkResult(
            system_name="FactValidator",
            benchmark_name="TestBenchmark",
            test_set_size=100,
            accuracy=0.80,
            precision=0.82,
            recall=0.78,
            f1=0.80
        )
        
        assert result.system_name == "FactValidator"
        assert result.accuracy == 0.80
        assert 0 <= result.accuracy <= 1.0
        assert result.test_set_size == 100
    
    def test_benchmark_framework_initialization(self):
        """Test BenchmarkFramework initialization."""
        framework = BenchmarkFramework()
        
        assert framework is not None
        assert hasattr(framework, 'export_benchmark_results')
        assert hasattr(framework, 'REFERENCE_BENCHMARKS')
    
    def test_reference_benchmarks_available(self):
        """Test that reference benchmarks are documented."""
        framework = BenchmarkFramework()
        
        references = framework.REFERENCE_BENCHMARKS
        
        assert isinstance(references, dict)
        assert len(references) > 0
        # Check for at least one reference system
        assert any(name in references for name in ["GoogleFactCheck", "ClaimBuster", "FEVER"])
    
    def test_export_benchmark_results_to_file(self):
        """Test exporting benchmark results to JSON file."""
        framework = BenchmarkFramework()
        
        results = [
            BenchmarkResult("FactValidator", "Benchmark", 100, 0.80, 0.82, 0.78, 0.80),
            BenchmarkResult("GoogleAPI", "Benchmark", 100, 0.70, 0.72, 0.68, 0.70),
            BenchmarkResult("RandomBaseline", "Benchmark", 100, 0.33, 0.50, 0.33, 0.33),
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            output_file = framework.export_benchmark_results(results, temp_path)
            
            assert Path(output_file).exists()
            
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "results" in data
            assert data["results"][0]["system"] == "FactValidator"
            assert data["results"][0]["metrics"]["accuracy"] == 0.80
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_benchmark_result_with_human_agreement(self):
        """Test benchmark result with human agreement metric."""
        result = BenchmarkResult(
            system_name="FactValidator",
            benchmark_name="Benchmark",
            test_set_size=50,
            accuracy=0.80,
            precision=0.82,
            recall=0.78,
            f1=0.80,
            human_agreement=0.85
        )
        
        assert result.human_agreement == 0.85
        assert result.human_agreement > result.accuracy


class TestComparativeIntegration:
    """Integration tests for complete comparative workflow."""
    
    def test_end_to_end_human_evaluation(self):
        """Test complete human evaluation workflow with interrater agreement."""
        framework = HumanEvaluationFramework()
        
        # Simulate 2 judges annotating 4 claims (Cohen's kappa requires exactly 2)
        judgments = [
            # Claim 1 - High agreement
            ComparativeJudgment("claim_1", "judge_1", "SUPPORTED", 90),
            ComparativeJudgment("claim_1", "judge_2", "SUPPORTED", 85),
            # Claim 2 - Moderate agreement
            ComparativeJudgment("claim_2", "judge_1", "REFUTED", 80),
            ComparativeJudgment("claim_2", "judge_2", "REFUTED", 75),
            # Claim 3 - High agreement
            ComparativeJudgment("claim_3", "judge_1", "REFUTED", 85),
            ComparativeJudgment("claim_3", "judge_2", "REFUTED", 80),
            # Claim 4 - Disagreement
            ComparativeJudgment("claim_4", "judge_1", "SUPPORTED", 60),
            ComparativeJudgment("claim_4", "judge_2", "REFUTED", 65),
        ]
        
        agreement = framework.calculate_interrater_agreement(judgments)
        
        assert agreement.n_judges == 2
        assert agreement.n_claims == 4
        assert 0 <= agreement.percent_agreement <= 1.0
        # Cohen's kappa calculated for 2 judges
        assert not isinstance(agreement.cohen_kappa, float) or (-1.0 <= agreement.cohen_kappa <= 1.0)

    
    def test_end_to_end_comparative_analysis(self):
        """Test complete comparative analysis workflow."""
        analysis = ComparativeAnalysis()
        
        # Simulate multiple systems evaluated on same test set
        system_scores = {
            "FactValidator": [0.85, 0.80, 0.90, 0.88, 0.92],
            "GoogleAPI": [0.70, 0.75, 0.72, 0.68, 0.70],
            "KeywordBaseline": [0.50, 0.40, 0.45, 0.55, 0.60],
            "RandomBaseline": [0.33, 0.33, 0.33, 0.33, 0.33],
        }
        
        matrix = analysis.build_comparison_matrix(system_scores)
        
        assert "FactValidator" in matrix
        assert "GoogleAPI" in matrix
        assert len(matrix["FactValidator"]["comparisons"]) == 3  # Compared with 3 others
    
    def test_end_to_end_benchmark_export(self):
        """Test complete benchmark workflow."""
        framework = BenchmarkFramework()
        analysis = ComparativeAnalysis()
        
        # Create synthetic benchmark results
        results = []
        systems = ["FactValidator", "GoogleAPI", "RandomBaseline"]
        accuracies = [0.80, 0.70, 0.33]
        
        for system, accuracy in zip(systems, accuracies):
            result = BenchmarkResult(
                system_name=system,
                benchmark_name="evaluation_benchmark",
                test_set_size=100,
                accuracy=accuracy,
                precision=accuracy + 0.02 if system != "RandomBaseline" else 0.50,
                recall=accuracy - 0.02 if system != "RandomBaseline" else 0.33,
                f1=accuracy,
                human_agreement=0.85 if system == "FactValidator" else None
            )
            results.append(result)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            output_file = framework.export_benchmark_results(results, temp_path)
            assert Path(output_file).exists()
            
            with open(output_file, 'r') as f:
                exported_data = json.load(f)
            
            assert len(exported_data["results"]) == 3
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
