"""
Run Step 2 baseline comparison on prepared benchmark splits.

Outputs:
- JSON metrics report
- CSV predictions table
- Markdown summary table

Usage:
  python Scripts/run_baseline_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.baselines import BaselineComparison, MajorityClassBaseline, RandomBaseline
from app.evaluation import EvaluationMetricsCalculator, PredictionResult, VerdictLabel


def _label_to_str(label: object) -> str:
    if hasattr(label, "value"):
        return str(getattr(label, "value"))
    return str(label)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline comparison on benchmark split")
    parser.add_argument(
        "--train",
        default=str(REPO_ROOT / "data" / "benchmarks" / "splits" / "train.json"),
        help="Path to train split JSON (used to compute majority class)",
    )
    parser.add_argument(
        "--test",
        default=str(REPO_ROOT / "data" / "benchmarks" / "splits" / "test.json"),
        help="Path to test split JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory where reports will be written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic baselines",
    )
    return parser.parse_args()


def _load_claims(file_path: Path) -> List[Dict[str, str]]:
    with open(file_path, "r") as f:
        payload = json.load(f)

    claims = payload.get("claims", [])
    normalized = []
    for claim in claims:
        claim_text = claim.get("claim") or claim.get("text") or ""
        normalized.append(
            {
                "id": str(claim.get("id", "")),
                "text": str(claim_text),
                "category": str(claim.get("category", "general")),
                "label": str(claim.get("label", "NEI")),
            }
        )
    return normalized


def _majority_label(train_claims: List[Dict[str, str]]) -> str:
    labels = [str(c.get("label", "NEI")).upper() for c in train_claims]
    if not labels:
        return VerdictLabel.SUPPORTED.value
    return Counter(labels).most_common(1)[0][0]


def _macro_metrics(per_class: Dict[object, object]) -> Dict[str, float]:
    metrics = list(per_class.values())
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = sum(m.precision for m in metrics) / len(metrics)
    recall = sum(m.recall for m in metrics) / len(metrics)
    f1 = sum(m.f1 for m in metrics) / len(metrics)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _serialize_per_class(per_class: Dict[object, object]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label, metric in per_class.items():
        out[_label_to_str(label)] = {
            "accuracy": metric.accuracy,
            "precision": metric.precision,
            "recall": metric.recall,
            "f1": metric.f1,
            "support": metric.support,
        }
    return out


def _build_markdown_summary(report: Dict) -> str:
    lines = [
        "# Baseline Comparison Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Test claims: {report['metadata']['test_claim_count']}",
        f"- Train claims: {report['metadata']['train_claim_count']}",
        "",
        "| Baseline | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, metrics in report["results"].items():
        lines.append(
            "| "
            f"{name} | {metrics['overall_accuracy']:.3f} | "
            f"{metrics['macro']['precision']:.3f} | "
            f"{metrics['macro']['recall']:.3f} | "
            f"{metrics['macro']['f1']:.3f} |"
        )

    lines.append("")
    lines.append("## Majority Baseline")
    lines.append("")
    lines.append(f"- Computed majority class from train split: **{report['metadata']['majority_label']}**")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_claims = _load_claims(train_path)
    test_claims = _load_claims(test_path)

    comparison = BaselineComparison()
    comparison.random_baseline = RandomBaseline(seed=args.seed)
    majority = _majority_label(train_claims)
    comparison.majority_baseline = MajorityClassBaseline(majority_label=majority)

    results_by_baseline = comparison.evaluate_all_baselines(test_claims)

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "train_split": str(train_path),
            "test_split": str(test_path),
            "train_claim_count": len(train_claims),
            "test_claim_count": len(test_claims),
            "majority_label": majority,
            "seed": args.seed,
        },
        "results": {},
    }

    csv_rows = []

    for baseline_name, predictions in results_by_baseline.items():
        overall_accuracy = EvaluationMetricsCalculator.calculate_overall_accuracy(predictions)
        per_class = EvaluationMetricsCalculator.calculate_per_class_metrics(predictions)
        per_category = EvaluationMetricsCalculator.calculate_per_category_metrics(predictions)
        calibration = EvaluationMetricsCalculator.calculate_confidence_calibration(predictions)

        report["results"][baseline_name] = {
            "overall_accuracy": overall_accuracy,
            "macro": _macro_metrics(per_class),
            "per_class": _serialize_per_class(per_class),
            "per_category": per_category,
            "calibration": calibration,
            "n_predictions": len(predictions),
        }

        for pred in predictions:
            csv_rows.append(
                {
                    "baseline": baseline_name,
                    "claim_id": pred.claim_id,
                    "category": pred.category,
                    "ground_truth_label": _label_to_str(pred.ground_truth_label),
                    "predicted_label": _label_to_str(pred.predicted_label),
                    "predicted_confidence": round(float(pred.predicted_confidence), 4),
                    "is_correct": pred.is_correct(),
                }
            )

    json_path = output_dir / "baseline_comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "baseline_comparison_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "baseline",
                "claim_id",
                "category",
                "ground_truth_label",
                "predicted_label",
                "predicted_confidence",
                "is_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path = output_dir / "baseline_comparison_summary.md"
    with open(md_path, "w") as f:
        f.write(_build_markdown_summary(report))

    print("Baseline comparison completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV predictions: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
