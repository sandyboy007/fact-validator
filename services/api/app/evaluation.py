"""
Comprehensive evaluation framework for Fact Validator.

Provides metrics calculation, baseline comparisons, ablation studies,
and error analysis for thesis-grade evaluation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
import math


class VerdictLabel(str, Enum):
    """Standard verdict labels."""
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEI = "NEI"  # Not Enough Information


@dataclass
class EvaluationMetrics:
    """Core evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConfusionMatrix:
    """Per-class confusion matrix."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    
    def calculate_metrics(self) -> EvaluationMetrics:
        """Calculate precision, recall, F1 from confusion matrix."""
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        
        if total == 0:
            return EvaluationMetrics(accuracy=0, precision=0, recall=0, f1=0, support=0)
        
        accuracy = (self.true_positives + self.true_negatives) / total if total > 0 else 0
        
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0 else 0
        )
        
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0 else 0
        )
        
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0
        )
        
        support = self.true_positives + self.false_negatives
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support
        )


@dataclass
class PredictionResult:
    """Single prediction result for evaluation."""
    claim_id: str
    claim_text: str
    category: str
    ground_truth_label: str  # SUPPORTED, REFUTED, NEI
    predicted_label: str
    predicted_confidence: float  # 0-100
    model_name: str
    
    def is_correct(self) -> bool:
        return self.predicted_label == self.ground_truth_label


class EvaluationMetricsCalculator:
    """Calculate comprehensive evaluation metrics."""
    
    @staticmethod
    def calculate_overall_accuracy(predictions: List[PredictionResult]) -> float:
        """Calculate overall accuracy across all predictions."""
        if not predictions:
            return 0.0
        correct = sum(1 for p in predictions if p.is_correct())
        return correct / len(predictions)
    
    @staticmethod
    def calculate_per_class_metrics(
        predictions: List[PredictionResult],
        labels: Optional[List[str]] = None
    ) -> Dict[str, EvaluationMetrics]:
        """Calculate metrics per verdict class (SUPPORTED, REFUTED, NEI)."""
        if labels is None:
            labels = [VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, VerdictLabel.NEI]
        
        metrics_per_class = {}
        
        for label in labels:
            cm = ConfusionMatrix()
            
            for pred in predictions:
                if pred.ground_truth_label == label and pred.predicted_label == label:
                    cm.true_positives += 1
                elif pred.ground_truth_label == label and pred.predicted_label != label:
                    cm.false_negatives += 1
                elif pred.ground_truth_label != label and pred.predicted_label == label:
                    cm.false_positives += 1
                else:
                    cm.true_negatives += 1
            
            metrics_per_class[label] = cm.calculate_metrics()
        
        return metrics_per_class
    
    @staticmethod
    def calculate_per_category_metrics(
        predictions: List[PredictionResult]
    ) -> Dict[str, float]:
        """Calculate accuracy per domain category (health, politics, etc.)."""
        category_groups: Dict[str, List[PredictionResult]] = {}
        
        for pred in predictions:
            if pred.category not in category_groups:
                category_groups[pred.category] = []
            category_groups[pred.category].append(pred)
        
        per_category = {}
        for category, preds in category_groups.items():
            correct = sum(1 for p in preds if p.is_correct())
            per_category[category] = {
                "accuracy": correct / len(preds) if preds else 0,
                "count": len(preds)
            }
        
        return per_category
    
    @staticmethod
    def calculate_confidence_calibration(
        predictions: List[PredictionResult],
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Measure confidence calibration.
        
        Are 80% confident predictions actually correct 80% of the time?
        Returns bin-wise accuracy vs. confidence.
        """
        bins = {i: {"confidences": [], "corrects": []} for i in range(n_bins)}
        
        for pred in predictions:
            bin_idx = min(int(pred.predicted_confidence / 100 * n_bins), n_bins - 1)
            bins[bin_idx]["confidences"].append(pred.predicted_confidence)
            bins[bin_idx]["corrects"].append(1 if pred.is_correct() else 0)
        
        calibration = {}
        for bin_idx, data in bins.items():
            if data["confidences"]:
                avg_confidence = sum(data["confidences"]) / len(data["confidences"])
                avg_accuracy = sum(data["corrects"]) / len(data["corrects"])
                calibration[f"bin_{bin_idx}"] = {
                    "avg_confidence": avg_confidence,
                    "avg_accuracy": avg_accuracy,
                    "calibration_gap": abs(avg_confidence - avg_accuracy)
                }
        
        return calibration
    
    @staticmethod
    def calculate_roc_auc(
        predictions: List[PredictionResult],
        positive_label: str = VerdictLabel.SUPPORTED
    ) -> float:
        """Calculate AUC-ROC for binary classification (Supported vs Others)."""
        # Sort by confidence descending
        sorted_preds = sorted(predictions, key=lambda p: p.predicted_confidence, reverse=True)
        
        # Calculate TPR and FPR at each threshold
        tp = 0
        fp = 0
        total_positives = sum(1 for p in predictions if p.ground_truth_label == positive_label)
        total_negatives = len(predictions) - total_positives
        
        if total_positives == 0 or total_negatives == 0:
            return 0.0
        
        tpr_values = [0]
        fpr_values = [0]
        
        for pred in sorted_preds:
            if pred.ground_truth_label == positive_label:
                tp += 1
            else:
                fp += 1
            
            tpr = tp / total_positives
            fpr = fp / total_negatives
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
        
        return auc


@dataclass
class ErrorCategory:
    """Categorized error for analysis."""
    error_type: str  # extraction, retrieval, verdict, etc.
    subcategory: str
    prediction: PredictionResult
    explanation: str
    severity: str  # low, medium, high


class ErrorAnalyzer:
    """Analyze and categorize prediction errors."""
    
    @staticmethod
    def categorize_errors(predictions: List[PredictionResult]) -> List[ErrorCategory]:
        """
        Categorize errors into semantic types.
        
        Error Types:
        - extraction: claim extraction failed/wrong
        - retrieval: no relevant evidence found
        - ranking: evidence exists but ranked poorly
        - verdict: wrong classification despite good evidence
        - confidence: correct verdict but very wrong confidence
        """
        errors = []
        
        for pred in predictions:
            if not pred.is_correct():
                # Determine error type based on confidence and label
                if pred.predicted_confidence < 30:
                    error_type = "retrieval"
                    subcategory = "weak_evidence"
                    severity = "high"
                elif pred.predicted_confidence < 60:
                    error_type = "ranking"
                    subcategory = "suboptimal_ranking"
                    severity = "medium"
                else:
                    error_type = "verdict"
                    subcategory = "classification_error"
                    severity = "high"
                
                explanation = f"Predicted {pred.predicted_label} (conf: {pred.predicted_confidence}%) but ground truth is {pred.ground_truth_label}"
                
                errors.append(ErrorCategory(
                    error_type=error_type,
                    subcategory=subcategory,
                    prediction=pred,
                    explanation=explanation,
                    severity=severity
                ))
        
        return errors
    
    @staticmethod
    def summarize_errors(errors: List[ErrorCategory]) -> Dict:
        """Generate error summary statistics."""
        summary = {
            "total_errors": len(errors),
            "by_type": {},
            "by_category": {},
            "by_severity": {}
        }
        
        for error in errors:
            # By type
            summary["by_type"][error.error_type] = summary["by_type"].get(error.error_type, 0) + 1
            
            # By category
            cat = error.prediction.category
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            
            # By severity
            summary["by_severity"][error.severity] = summary["by_severity"].get(error.severity, 0) + 1
        
        return summary


@dataclass
class AblationResult:
    """Result of ablating a system component."""
    component_name: str
    description: str
    without_component: EvaluationMetrics
    with_component: EvaluationMetrics
    impact_delta: Dict[str, float]  # changes in metrics
    relative_importance: float  # % contribution to overall performance


class AblationStudy:
    """Framework for ablation studies."""
    
    @staticmethod
    def run_ablation(
        full_model_predictions: List[PredictionResult],
        ablated_predictions: List[PredictionResult],
        component_name: str,
        description: str
    ) -> AblationResult:
        """
        Compare full model vs. ablated version.
        
        Returns impact metrics showing what component contributes.
        """
        full_accuracy = EvaluationMetricsCalculator.calculate_overall_accuracy(full_model_predictions)
        ablated_accuracy = EvaluationMetricsCalculator.calculate_overall_accuracy(ablated_predictions)
        
        full_metrics = EvaluationMetricsCalculator.calculate_per_class_metrics(full_model_predictions)
        ablated_metrics = EvaluationMetricsCalculator.calculate_per_class_metrics(ablated_predictions)
        
        # Calculate deltas
        accuracy_drop = full_accuracy - ablated_accuracy
        
        impact_delta = {
            "accuracy_drop": accuracy_drop,
            "accuracy_drop_pct": (accuracy_drop / full_accuracy * 100) if full_accuracy > 0 else 0
        }
        
        relative_importance = (accuracy_drop / full_accuracy * 100) if full_accuracy > 0 else 0
        
        return AblationResult(
            component_name=component_name,
            description=description,
            without_component=EvaluationMetrics(
                accuracy=ablated_accuracy,
                precision=0,
                recall=0,
                f1=0,
                support=len(ablated_predictions)
            ),
            with_component=EvaluationMetrics(
                accuracy=full_accuracy,
                precision=0,
                recall=0,
                f1=0,
                support=len(full_model_predictions)
            ),
            impact_delta=impact_delta,
            relative_importance=relative_importance
        )


class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    @staticmethod
    def generate_report(
        predictions: List[PredictionResult],
        model_name: str,
        errors: Optional[List[ErrorCategory]] = None,
        ablations: Optional[List[AblationResult]] = None
    ) -> Dict:
        """Generate complete evaluation report."""
        
        report = {
            "metadata": {
                "model": model_name,
                "timestamp": datetime.utcnow().isoformat(),
                "total_predictions": len(predictions)
            },
            "overall": {
                "accuracy": EvaluationMetricsCalculator.calculate_overall_accuracy(predictions)
            },
            "per_class": EvaluationMetricsCalculator.calculate_per_class_metrics(predictions),
            "per_category": EvaluationMetricsCalculator.calculate_per_category_metrics(predictions),
            "calibration": EvaluationMetricsCalculator.calculate_confidence_calibration(predictions),
            "auc_roc": EvaluationMetricsCalculator.calculate_roc_auc(predictions)
        }
        
        if errors:
            report["errors"] = {
                "summary": ErrorAnalyzer.summarize_errors(errors),
                "total": len(errors)
            }
        
        if ablations:
            report["ablations"] = [
                {
                    "component": a.component_name,
                    "description": a.description,
                    "accuracy_with": a.with_component.accuracy,
                    "accuracy_without": a.without_component.accuracy,
                    "impact_drop_pct": a.impact_delta.get("accuracy_drop_pct", 0),
                    "relative_importance_pct": a.relative_importance
                }
                for a in ablations
            ]
        
        return report
