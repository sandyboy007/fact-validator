"""
Compare external systems from prediction dumps.

This harness is provider-agnostic. It does not call GPT-4o, Gemini, or Claude
directly; instead it consumes real prediction files produced by those systems,
plus FacTool, FEVER baselines, and RAG baselines. That makes the comparison
auditable and reproducible for paper submission.

Supported input formats per system:
  - CSV with columns: claim_id, ground_truth_label, predicted_label, predicted_confidence, is_correct, category
  - JSONL or JSON array with equivalent keys

Usage examples:
  python Scripts/run_external_system_comparison.py \
    --input gpt-4o=results/gpt4o_predictions.csv \
    --input gemini=results/gemini_predictions.csv \
    --input claude=results/claude_predictions.csv \
    --input factool=results/factool_predictions.csv \
    --input fever_baseline=results/fever_baseline_predictions.csv \
    --input rag_baseline=results/rag_baseline_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import sys

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.statistics import StatisticalAnalyzer


@dataclass
class SystemPrediction:
    system: str
    claim_id: str
    category: str
    ground_truth_label: str
    predicted_label: str
    predicted_confidence: float
    is_correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare external fact-checking systems")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Prediction source in the form system_name=path (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory for report outputs",
    )
    parser.add_argument(
        "--reference-system",
        default="gpt-4o",
        help="System name used as the main comparison anchor",
    )
    return parser.parse_args()


def _label(value: Any) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value or "NEI").upper()


def _load_json_payload(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else []
    if text.startswith("{"):
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("predictions", "results", "rows"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    return _load_json_payload(path)


def _parse_prediction_rows(system_name: str, path: Path) -> list[SystemPrediction]:
    rows = _load_rows(path)
    parsed: list[SystemPrediction] = []
    for index, row in enumerate(rows, start=1):
        claim_id = str(row.get("claim_id") or row.get("id") or f"{system_name}-{index}")
        category = str(row.get("category") or "general")
        ground_truth = _label(row.get("ground_truth_label") or row.get("label"))
        predicted = _label(row.get("predicted_label") or row.get("prediction") or row.get("verdict"))
        confidence_raw = row.get("predicted_confidence") or row.get("confidence") or 0.0
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        is_correct_raw = row.get("is_correct")
        if is_correct_raw is None:
            is_correct = predicted == ground_truth
        else:
            is_correct = str(is_correct_raw).strip().lower() in {"1", "true", "yes", "y"}

        parsed.append(
            SystemPrediction(
                system=system_name,
                claim_id=claim_id,
                category=category,
                ground_truth_label=ground_truth,
                predicted_label=predicted,
                predicted_confidence=confidence,
                is_correct=is_correct,
            )
        )
    return parsed


def _group_by_system(rows: Iterable[SystemPrediction]) -> dict[str, dict[str, SystemPrediction]]:
    grouped: dict[str, dict[str, SystemPrediction]] = defaultdict(dict)
    for row in rows:
        grouped[row.system][row.claim_id] = row
    return grouped


def _order_claim_ids(grouped: dict[str, dict[str, SystemPrediction]], anchor: str) -> list[str]:
    if not grouped:
        return []

    common_ids: set[str] | None = None
    for system_rows in grouped.values():
        system_ids = set(system_rows.keys())
        common_ids = system_ids if common_ids is None else common_ids.intersection(system_ids)

    if common_ids:
        return sorted(common_ids)

    if anchor in grouped and grouped[anchor]:
        return sorted(grouped[anchor].keys())

    claim_ids = set()
    for system_rows in grouped.values():
        claim_ids.update(system_rows.keys())
    return sorted(claim_ids)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_scores(rows: dict[str, SystemPrediction], claim_ids: list[str]) -> tuple[list[float], list[SystemPrediction]]:
    ordered = [rows[cid] for cid in claim_ids if cid in rows]
    scores = [1.0 if r.is_correct else 0.0 for r in ordered]
    return scores, ordered


def _calibration_error(rows: list[SystemPrediction], n_bins: int = 10) -> float:
    if not rows:
        return 0.0
    bins = {i: {"n": 0, "conf": 0.0, "correct": 0.0} for i in range(n_bins)}
    for row in rows:
        confidence = max(0.0, min(100.0, row.predicted_confidence)) / 100.0
        idx = min(int(confidence * n_bins), n_bins - 1)
        bins[idx]["n"] += 1
        bins[idx]["conf"] += confidence
        bins[idx]["correct"] += 1.0 if row.is_correct else 0.0

    total = len(rows)
    ece = 0.0
    for bin_data in bins.values():
        if bin_data["n"] == 0:
            continue
        avg_conf = bin_data["conf"] / bin_data["n"]
        avg_acc = bin_data["correct"] / bin_data["n"]
        ece += (bin_data["n"] / total) * abs(avg_acc - avg_conf)
    return ece


def _system_summary(rows: list[SystemPrediction]) -> dict[str, Any]:
    if not rows:
        return {"n_claims": 0, "accuracy": 0.0, "avg_confidence": 0.0, "calibration_error": 0.0, "ece": 0.0}
    accuracy = _mean([1.0 if r.is_correct else 0.0 for r in rows])
    avg_confidence = _mean([r.predicted_confidence for r in rows])
    return {
        "n_claims": len(rows),
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "calibration_error": abs(accuracy - (avg_confidence / 100.0)),
        "ece": _calibration_error(rows),
    }


def _pairwise_report(
    grouped: dict[str, dict[str, SystemPrediction]],
    claim_ids: list[str],
    system_names: list[str],
    reference_system: str,
) -> dict[str, Any]:
    analyzer = StatisticalAnalyzer()
    report: dict[str, Any] = {}
    for system in system_names:
        scores, aligned_rows = _build_scores(grouped.get(system, {}), claim_ids)
        report[system] = {
            "summary": _system_summary(aligned_rows),
            "n_aligned": len(aligned_rows),
            "comparisons": {},
        }

    for system in system_names:
        if system == reference_system:
            continue
        system_scores, _ = _build_scores(grouped.get(reference_system, {}), claim_ids)
        baseline_scores, _ = _build_scores(grouped.get(system, {}), claim_ids)
        if len(system_scores) == len(baseline_scores) and system_scores:
            test = analyzer.paired_t_test(system_scores, baseline_scores, alternative="greater")
            effect = analyzer.cohens_d(system_scores, baseline_scores)
            if abs(effect) < 0.2:
                effect_label = "negligible"
            elif abs(effect) < 0.5:
                effect_label = "small"
            elif abs(effect) < 0.8:
                effect_label = "medium"
            else:
                effect_label = "large"
            report[reference_system]["comparisons"][system] = {
                "improvement_pct_points": (_mean(system_scores) - _mean(baseline_scores)) * 100.0,
                "p_value": test.p_value,
                "t_statistic": test.t_statistic,
                "cohens_d": effect,
                "effect_interpretation": effect_label,
                "is_significant": test.is_significant,
            }

    return report


def _markdown(report: dict[str, Any], system_names: list[str], reference_system: str) -> str:
    lines = [
        "# External System Comparison",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Reference system: {reference_system}",
        "",
        "| System | Accuracy | Avg. Confidence | Calibration Error | ECE | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in system_names:
        summary = report["systems"][system]["summary"]
        lines.append(
            f"| {system} | {summary['accuracy']:.3f} | {summary['avg_confidence']:.1f} | {summary['calibration_error']:.3f} | {summary['ece']:.3f} | {summary['n_claims']} |"
        )

    lines.append("")
    lines.append("## Pairwise Comparisons Against Reference")
    lines.append("")
    lines.append("| Comparator | Delta Accuracy (pp) | p-value | Cohen's d | Significant |")
    lines.append("|---|---:|---:|---:|:---:|")
    for comparator, comp in report["systems"][reference_system]["comparisons"].items():
        sig = "yes" if comp["is_significant"] else "no"
        lines.append(
            f"| {comparator} | {comp['improvement_pct_points']:+.2f} | {comp['p_value']:.4f} | {comp['cohens_d']:.3f} | {sig} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[SystemPrediction] = []
    source_map: dict[str, str] = {}
    for spec in args.input:
        if "=" not in spec:
            raise ValueError(f"invalid input spec '{spec}', expected system=path")
        system_name, path_str = spec.split("=", 1)
        system_name = system_name.strip()
        path = Path(path_str.strip())
        if not path.exists():
            raise FileNotFoundError(f"prediction file not found: {path}")
        rows.extend(_parse_prediction_rows(system_name, path))
        source_map[system_name] = str(path)

    grouped = _group_by_system(rows)
    system_names = list(source_map.keys())
    claim_ids = _order_claim_ids(grouped, args.reference_system)
    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "reference_system": args.reference_system,
            "input_sources": source_map,
            "claim_count": len(claim_ids),
            "system_count": len(system_names),
        },
        "systems": _pairwise_report(grouped, claim_ids, system_names, args.reference_system),
    }

    json_path = output_dir / "external_system_comparison_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output_dir / "external_system_comparison_ranking.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["system", "n_claims", "accuracy", "avg_confidence", "calibration_error", "ece"],
        )
        writer.writeheader()
        for system in system_names:
            summary = report["systems"][system]["summary"]
            writer.writerow(
                {
                    "system": system,
                    "n_claims": summary["n_claims"],
                    "accuracy": round(summary["accuracy"], 6),
                    "avg_confidence": round(summary["avg_confidence"], 4),
                    "calibration_error": round(summary["calibration_error"], 6),
                    "ece": round(summary["ece"], 6),
                }
            )

    md_path = output_dir / "external_system_comparison_summary.md"
    md_path.write_text(_markdown(report, system_names, args.reference_system), encoding="utf-8")

    print("External system comparison completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV ranking: {csv_path}")
    print(f"- Markdown summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())