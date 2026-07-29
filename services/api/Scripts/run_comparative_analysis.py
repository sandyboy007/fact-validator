"""
Run Step 4 comparative analysis using Step 2 and Step 3 outputs.

This script compares the full proxy system against all baseline systems,
adds statistical significance tests, and reports confidence calibration.

Outputs:
- JSON comparative report
- CSV ranking table
- Markdown summary

Usage:
  python Scripts/run_comparative_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


@dataclass
class SystemPrediction:
    system: str
    claim_id: str
    category: str
    ground_truth_label: str
    predicted_label: str
    predicted_confidence: float
    is_correct: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 4 comparative analysis")
    parser.add_argument(
        "--baseline-csv",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "baseline_comparison_predictions.csv"),
        help="Path to Step 2 baseline predictions CSV",
    )
    parser.add_argument(
        "--ablation-csv",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "ablation_study_predictions.csv"),
        help="Path to Step 3 ablation predictions CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory where comparative artifacts will be written",
    )
    parser.add_argument(
        "--full-variant",
        default="full_proxy",
        help="Ablation variant name that represents the full system",
    )
    parser.add_argument(
        "--no-debate-variant",
        default="ablate_debate",
        help="Ablation variant to use for debate lift comparison",
    )
    return parser.parse_args()


def _load_rows(path: Path, system_column: str) -> List[SystemPrediction]:
    rows: List[SystemPrediction] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            confidence_raw = row.get("predicted_confidence", "0")
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.0

            correct_raw = str(row.get("is_correct", "false")).strip().lower()
            is_correct = correct_raw in {"true", "1", "yes"}

            rows.append(
                SystemPrediction(
                    system=str(row.get(system_column, "unknown")),
                    claim_id=str(row.get("claim_id", "")),
                    category=str(row.get("category", "general")),
                    ground_truth_label=str(row.get("ground_truth_label", "NEI")),
                    predicted_label=str(row.get("predicted_label", "NEI")),
                    predicted_confidence=confidence,
                    is_correct=is_correct,
                )
            )
    return rows


def _order_claim_ids(rows: List[SystemPrediction], anchor_system: str) -> List[str]:
    anchor_rows = [r for r in rows if r.system == anchor_system]
    if anchor_rows:
        return [r.claim_id for r in anchor_rows]

    unique = sorted({r.claim_id for r in rows})
    return unique


def _by_system(rows: List[SystemPrediction]) -> Dict[str, Dict[str, SystemPrediction]]:
    by_name: Dict[str, Dict[str, SystemPrediction]] = defaultdict(dict)
    for r in rows:
        by_name[r.system][r.claim_id] = r
    return by_name


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_confidence_interval(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "margin_of_error": 0.0}
    if len(values) == 1:
        v = values[0]
        return {"mean": v, "lower": v, "upper": v, "margin_of_error": 0.0}

    # Normal approximation CI for Bernoulli mean (accuracy on claim-level 0/1 scores).
    mean = _mean(values)
    z = 1.96
    se = math.sqrt((mean * (1.0 - mean)) / len(values))
    margin = z * se
    return {
        "mean": mean,
        "lower": max(0.0, mean - margin),
        "upper": min(1.0, mean + margin),
        "margin_of_error": margin,
    }


def _ece(rows: List[SystemPrediction], n_bins: int = 10) -> float:
    if not rows:
        return 0.0

    bin_totals = [0 for _ in range(n_bins)]
    bin_conf_sums = [0.0 for _ in range(n_bins)]
    bin_corr_sums = [0.0 for _ in range(n_bins)]

    for r in rows:
        conf = max(0.0, min(100.0, float(r.predicted_confidence))) / 100.0
        idx = min(int(conf * n_bins), n_bins - 1)
        bin_totals[idx] += 1
        bin_conf_sums[idx] += conf
        bin_corr_sums[idx] += 1.0 if r.is_correct else 0.0

    total = len(rows)
    ece = 0.0
    for i in range(n_bins):
        if bin_totals[i] == 0:
            continue
        avg_conf = bin_conf_sums[i] / bin_totals[i]
        avg_acc = bin_corr_sums[i] / bin_totals[i]
        ece += (bin_totals[i] / total) * abs(avg_acc - avg_conf)
    return ece


def _build_system_metrics(
    system_rows: Dict[str, SystemPrediction],
    ordered_claim_ids: List[str],
) -> Tuple[List[float], List[SystemPrediction], Dict]:
    aligned_rows = [system_rows[cid] for cid in ordered_claim_ids if cid in system_rows]
    scores = [1.0 if r.is_correct else 0.0 for r in aligned_rows]

    ci = _safe_confidence_interval(scores)
    accuracy = _mean(scores)
    avg_confidence = _mean([r.predicted_confidence for r in aligned_rows])
    calibration_error = abs(accuracy - (avg_confidence / 100.0))

    metrics = {
        "n_claims": len(aligned_rows),
        "accuracy": accuracy,
        "accuracy_ci95": ci,
        "avg_confidence": avg_confidence,
        "calibration_error": calibration_error,
        "ece": _ece(aligned_rows),
    }
    return scores, aligned_rows, metrics


def _safe_comparison(
    system_scores: List[float],
    baseline_scores: List[float],
    system_name: str,
    baseline_name: str,
) -> Dict:
    if len(system_scores) != len(baseline_scores) or not system_scores:
        return {
            "error": "Incompatible score vectors for paired comparison",
            "system_name": system_name,
            "baseline_name": baseline_name,
        }

    ci_system = _safe_confidence_interval(system_scores)
    ci_baseline = _safe_confidence_interval(baseline_scores)

    wins = 0
    losses = 0
    for s, b in zip(system_scores, baseline_scores):
        if s > b:
            wins += 1
        elif s < b:
            losses += 1

    # Exact two-sided McNemar test on discordant paired correctness outcomes.
    n_non_tie = wins + losses
    if n_non_tie == 0:
        p_value = 1.0
        z_stat = 0.0
    else:
        tail_end = min(wins, losses)
        log_terms = []
        log_half = math.log(0.5)
        for k in range(0, tail_end + 1):
            log_terms.append(
                math.lgamma(n_non_tie + 1)
                - math.lgamma(k + 1)
                - math.lgamma(n_non_tie - k + 1)
                + n_non_tie * log_half
            )

        max_log = max(log_terms)
        one_tail = math.exp(max_log) * sum(math.exp(t - max_log) for t in log_terms)
        p_value = 2.0 * one_tail
        p_value = min(1.0, max(0.0, p_value))

        # Continuity-corrected z approximation is display-only; the exact
        # two-sided p-value above is the reported inferential result.
        expected = n_non_tie * 0.5
        std = math.sqrt(n_non_tie * 0.25)
        correction = 0.5 if wins >= losses else -0.5
        z_stat = ((wins - expected) - correction) / std if std > 0 else 0.0

    system_accuracy = _mean(system_scores)
    baseline_accuracy = _mean(baseline_scores)
    improvement_pct = (system_accuracy - baseline_accuracy) * 100.0

    return {
        "system_name": system_name,
        "baseline_name": baseline_name,
        "improvement_pct_points": improvement_pct,
        "system_accuracy": system_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "ci_system": {
            "mean": ci_system["mean"],
            "lower": ci_system["lower"],
            "upper": ci_system["upper"],
        },
        "ci_baseline": {
            "mean": ci_baseline["mean"],
            "lower": ci_baseline["lower"],
            "upper": ci_baseline["upper"],
        },
        "significance_test": {
            "name": "Exact paired McNemar test (two-sided)",
            "z_statistic": z_stat,
            "p_value": p_value,
            "p_value_holm": None,
            "degrees_freedom": n_non_tie,
            "is_significant_alpha_0_05": bool(p_value < 0.05),
            "is_significant_holm_alpha_0_05": False,
        },
        "effect_size": {
            "matched_pair_odds_ratio": (wins + 0.5) / (losses + 0.5),
            "paired_risk_difference": system_accuracy - baseline_accuracy,
            "full_wins": wins,
            "full_losses": losses,
        },
    }


def _apply_holm_correction(comparisons: List[Dict]) -> None:
    valid = [
        (index, item)
        for index, item in enumerate(comparisons)
        if "error" not in item
    ]
    ordered = sorted(
        valid,
        key=lambda pair: pair[1]["significance_test"]["p_value"],
    )
    running_max = 0.0
    total = len(ordered)
    for rank, (index, item) in enumerate(ordered):
        adjusted = min(
            1.0,
            (total - rank) * item["significance_test"]["p_value"],
        )
        running_max = max(running_max, adjusted)
        comparisons[index]["significance_test"]["p_value_holm"] = running_max
        comparisons[index]["significance_test"][
            "is_significant_holm_alpha_0_05"
        ] = bool(running_max < 0.05)


def _build_markdown(report: Dict) -> str:
    lines = [
        "# Comparative Analysis Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Full system variant: {report['metadata']['full_variant']}",
        f"- Claims compared: {report['metadata']['claims_compared']}",
        "",
        "## System Ranking",
        "",
        "| System | Accuracy | 95% CI | Avg Confidence | Calibration Error | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for item in report["ranking"]:
        ci = item["accuracy_ci95"]
        lines.append(
            "| "
            f"{item['system']} | {item['accuracy']:.3f} | "
            f"[{ci['lower']:.3f}, {ci['upper']:.3f}] | "
            f"{item['avg_confidence']:.1f} | "
            f"{item['calibration_error']:.3f} | {item['ece']:.3f} |"
        )

    lines.extend([
        "",
        "## Full System vs Comparators",
        "",
        "| Comparator | Delta Accuracy (pp) | exact p | Holm p | Matched OR | Holm significant |",
        "|---|---:|---:|---:|---:|:---:|",
    ])

    for cmp_row in report["comparisons"]:
        if "error" in cmp_row:
            continue
        p_value = cmp_row["significance_test"]["p_value"]
        p_text = "NA" if p_value is None else f"{p_value:.4f}"
        adjusted = cmp_row["significance_test"]["p_value_holm"]
        adjusted_text = "NA" if adjusted is None else f"{adjusted:.4f}"
        lines.append(
            "| "
            f"{cmp_row['baseline_name']} | "
            f"{cmp_row['improvement_pct_points']:+.2f} | "
            f"{p_text} | "
            f"{adjusted_text} | "
            f"{cmp_row['effect_size']['matched_pair_odds_ratio']:.3f} | "
            f"{'yes' if cmp_row['significance_test']['is_significant_holm_alpha_0_05'] else 'no'} |"
        )

    debate_lift = report.get("debate_lift")
    if debate_lift:
        lines.extend([
            "",
            "## Debate Lift",
            "",
            f"- Variant compared: {debate_lift['variant']}",
            f"- Accuracy delta (full - no-debate): {debate_lift['accuracy_delta_pct_points']:+.2f} pp",
            f"- Prediction change rate: {debate_lift['prediction_change_rate']:.3f}",
        ])

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    baseline_csv = Path(args.baseline_csv)
    ablation_csv = Path(args.ablation_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = _load_rows(baseline_csv, system_column="baseline")
    ablation_rows = _load_rows(ablation_csv, system_column="variant")

    all_rows = baseline_rows + ablation_rows
    by_system = _by_system(all_rows)

    full_variant = args.full_variant
    if full_variant not in by_system:
        raise FileNotFoundError(
            f"Full system variant '{full_variant}' was not found in {ablation_csv}"
        )

    ordered_claim_ids = _order_claim_ids(all_rows, anchor_system=full_variant)

    system_metrics: Dict[str, Dict] = {}
    system_scores: Dict[str, List[float]] = {}
    aligned_rows_by_system: Dict[str, List[SystemPrediction]] = {}

    for system_name, system_rows in by_system.items():
        scores, aligned_rows, metrics = _build_system_metrics(system_rows, ordered_claim_ids)
        if not scores:
            continue
        system_scores[system_name] = scores
        aligned_rows_by_system[system_name] = aligned_rows
        system_metrics[system_name] = metrics

    comparisons: List[Dict] = []
    for comparator, scores in system_scores.items():
        if comparator == full_variant:
            continue
        if len(scores) != len(system_scores[full_variant]):
            continue
        comparisons.append(
            _safe_comparison(
                system_scores[full_variant],
                scores,
                system_name=full_variant,
                baseline_name=comparator,
            )
        )
    _apply_holm_correction(comparisons)

    ranking = []
    for name, metrics in system_metrics.items():
        ranking.append(
            {
                "system": name,
                **metrics,
            }
        )
    ranking.sort(key=lambda x: x["accuracy"], reverse=True)

    debate_lift = None
    no_debate_variant = args.no_debate_variant
    if no_debate_variant in system_scores and full_variant in system_scores:
        full_rows = aligned_rows_by_system[full_variant]
        nd_rows = aligned_rows_by_system[no_debate_variant]

        changed = 0
        total = min(len(full_rows), len(nd_rows))
        nd_by_claim = {r.claim_id: r.predicted_label for r in nd_rows}
        for row in full_rows:
            if row.claim_id in nd_by_claim and row.predicted_label != nd_by_claim[row.claim_id]:
                changed += 1

        debate_lift = {
            "variant": no_debate_variant,
            "accuracy_delta_pct_points": (system_metrics[full_variant]["accuracy"] - system_metrics[no_debate_variant]["accuracy"]) * 100.0,
            "prediction_change_rate": (changed / total) if total else 0.0,
        }

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "baseline_csv": str(baseline_csv),
            "ablation_csv": str(ablation_csv),
            "full_variant": full_variant,
            "claims_compared": len(ordered_claim_ids),
        },
        "system_metrics": system_metrics,
        "ranking": ranking,
        "comparisons": comparisons,
        "debate_lift": debate_lift,
    }

    json_path = output_dir / "comparative_analysis_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "comparative_analysis_ranking.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "n_claims",
                "accuracy",
                "ci95_lower",
                "ci95_upper",
                "avg_confidence",
                "calibration_error",
                "ece",
            ],
        )
        writer.writeheader()
        for row in ranking:
            writer.writerow(
                {
                    "system": row["system"],
                    "n_claims": row["n_claims"],
                    "accuracy": round(row["accuracy"], 6),
                    "ci95_lower": round(row["accuracy_ci95"]["lower"], 6),
                    "ci95_upper": round(row["accuracy_ci95"]["upper"], 6),
                    "avg_confidence": round(row["avg_confidence"], 3),
                    "calibration_error": round(row["calibration_error"], 6),
                    "ece": round(row["ece"], 6),
                }
            )

    md_path = output_dir / "comparative_analysis_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print("Comparative analysis completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV ranking: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
