"""Evaluate the evidence-relation baseline on a frozen claim-passage dataset.

Input JSON must be a list, or an object with an ``items`` list. Each item needs
``claim``, ``passage`` (or ``evidence``), and ``label`` using support/refute/
neutral or SUPPORTED/REFUTED/NEI aliases.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
sys.path.insert(0, str(API_ROOT))

from app.relation_classifier import classify_relation  # noqa: E402


LABELS = ("support", "refute", "neutral")
LABEL_ALIASES = {
    "support": "support",
    "supported": "support",
    "entailment": "support",
    "entails": "support",
    "refute": "refute",
    "refuted": "refute",
    "contradiction": "refute",
    "contradicts": "refute",
    "neutral": "neutral",
    "nei": "neutral",
    "not enough information": "neutral",
}


def normalize_label(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in LABEL_ALIASES:
        raise ValueError(f"unsupported relation label: {value}")
    return LABEL_ALIASES[normalized]


def macro_f1(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    per_label: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    for label in LABELS:
        true_positive = sum(1 for row in rows if row["gold"] == label and row["prediction"] == label)
        false_positive = sum(1 for row in rows if row["gold"] != label and row["prediction"] == label)
        false_negative = sum(1 for row in rows if row["gold"] == label and row["prediction"] != label)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for row in rows if row["gold"] == label),
        }
        f1_values.append(f1)
    return {"macro_f1": round(sum(f1_values) / len(f1_values), 4), "per_label": per_label}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen claim-passage relation dataset")
    parser.add_argument("--input", required=True, help="Path to JSON relation dataset")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "relation_baseline_report.json"),
        help="Path to the JSON evaluation report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("input must be a JSON list or an object with an items list")

    predictions: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        passage = str(item.get("passage") or item.get("evidence") or "").strip()
        if not claim or not passage:
            continue
        gold = normalize_label(item.get("label"))
        prediction, classifier = classify_relation(claim, passage)
        predictions.append(
            {
                "id": item.get("id", index),
                "claim": claim,
                "passage": passage,
                "gold": gold,
                "prediction": prediction,
                "correct": prediction == gold,
                "classifier": classifier,
            }
        )

    if not predictions:
        raise ValueError("no valid claim-passage relation items found")

    report = {
        "version": "relation-baseline-report-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(input_path.resolve()),
        "items_evaluated": len(predictions),
        "label_distribution": dict(Counter(item["gold"] for item in predictions)),
        "accuracy": round(sum(1 for item in predictions if item["correct"]) / len(predictions), 4),
        "metrics": macro_f1(predictions),
        "predictions": predictions,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {len(predictions)} claim-passage pairs.")
    print(f"Macro-F1: {report['metrics']['macro_f1']:.4f}")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())