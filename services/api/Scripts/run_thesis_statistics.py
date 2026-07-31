"""Generate publication-safe statistics for the 5,000-claim proxy experiment.

The analysis treats correctness as paired binary data. It reports exact
two-sided McNemar tests, Holm-adjusted p-values, paired bootstrap confidence
intervals for accuracy differences, matched-pair odds ratios, confusion
matrices, and class/domain metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import binomtest

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.dataset import normalize_claim_text

DEFAULT_RESULT_DIR = REPO_ROOT / "data" / "benchmarks" / "results_5000"
SPLIT_DIR = REPO_ROOT / "data" / "benchmarks" / "splits_5000"
LABELS = ("SUPPORTED", "REFUTED", "NEI")


@dataclass(frozen=True)
class Prediction:
    system: str
    claim_id: str
    dataset: str
    gold: str
    predicted: str
    confidence: float

    @property
    def correct(self) -> bool:
        return self.gold == self.predicted


def _dataset_from_claim_id(claim_id: str) -> str:
    """Recover the frozen source dataset from its stable claim-ID prefix."""
    prefix = claim_id.split("-", 1)[0].strip().lower()
    names = {
        "fever": "FEVER",
        "liar": "LIAR",
        "scifact": "SciFact",
        "health": "PUBHEALTH_health_fact",
    }
    try:
        return names[prefix]
    except KeyError as exc:
        raise ValueError(f"Unrecognized dataset prefix in claim_id={claim_id!r}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis statistics")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    return parser.parse_args()


def _load(path: Path, system_column: str) -> dict[str, dict[str, Prediction]]:
    systems: dict[str, dict[str, Prediction]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            system = row[system_column]
            claim_id = row["claim_id"]
            if claim_id in systems[system]:
                raise ValueError(f"duplicate claim_id for {system}: {claim_id}")
            systems[system][claim_id] = Prediction(
                system=system,
                claim_id=claim_id,
                dataset=_dataset_from_claim_id(claim_id),
                gold=row["ground_truth_label"],
                predicted=row["predicted_label"],
                confidence=float(row.get("predicted_confidence", 0.0)),
            )
    return dict(systems)


def _accuracy(rows: Iterable[Prediction]) -> float:
    values = list(rows)
    return sum(row.correct for row in values) / len(values) if values else 0.0


def _wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return ((centre - margin) / denominator, (centre + margin) / denominator)


def _class_metrics(rows: Iterable[Prediction], label: str) -> dict[str, float | int]:
    values = list(rows)
    tp = sum(r.gold == label and r.predicted == label for r in values)
    fp = sum(r.gold != label and r.predicted == label for r in values)
    fn = sum(r.gold == label and r.predicted != label for r in values)
    support = sum(r.gold == label for r in values)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "label": label,
        "support": support,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _paired_bootstrap(
    full: list[Prediction],
    other: list[Prediction],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.asarray([
        (1 if left.correct else 0) - (1 if right.correct else 0)
        for left, right in zip(full, other)
    ], dtype=np.int8)
    count = len(deltas)
    estimates = np.empty(iterations, dtype=np.float64)
    chunk_size = 500
    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        indices = rng.integers(0, count, size=(stop - start, count))
        estimates[start:stop] = deltas[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def _holm_adjust(rows: list[dict]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value_unadjusted"])
    running_max = 0.0
    total = len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * row["p_value_unadjusted"])
        running_max = max(running_max, adjusted)
        rows[original_index]["p_value_holm"] = running_max
        rows[original_index]["significant_holm_0_05"] = running_max < 0.05


def _known_contaminated_test_ids() -> set[str]:
    """Return test IDs whose normalized text occurs in train or validation."""
    development_text: set[str] = set()
    for filename in ("train.json", "val.json"):
        payload = json.loads((SPLIT_DIR / filename).read_text(encoding="utf-8"))
        development_text.update(
            normalize_claim_text(row.get("claim", ""))
            for row in payload["claims"]
        )
    test_payload = json.loads((SPLIT_DIR / "test.json").read_text(encoding="utf-8"))
    return {
        str(row.get("id") or row.get("source_id"))
        for row in test_payload["claims"]
        if normalize_claim_text(row.get("claim", "")) in development_text
    }


def _subset_metrics(
    systems: dict[str, dict[str, Prediction]],
    claim_ids: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for name, rows_by_id in systems.items():
        predictions = [rows_by_id[claim_id] for claim_id in claim_ids]
        correct = sum(row.correct for row in predictions)
        lower, upper = _wilson(correct, len(predictions))
        class_rows = [_class_metrics(predictions, label) for label in LABELS]
        rows.append(
            {
                "system": name,
                "n": len(predictions),
                "accuracy": correct / len(predictions),
                "macro_f1": sum(float(item["f1"]) for item in class_rows) / len(LABELS),
                "accuracy_wilson_lower": lower,
                "accuracy_wilson_upper": upper,
            }
        )
    return rows


def _write_decontamination_sensitivity(
    output_dir: Path,
    systems: dict[str, dict[str, Prediction]],
    anchor_ids: list[str],
) -> None:
    contaminated_ids = _known_contaminated_test_ids()
    filtered_ids = [claim_id for claim_id in anchor_ids if claim_id not in contaminated_ids]
    if len(contaminated_ids) != 39 or len(filtered_ids) != 4961:
        raise ValueError(
            "decontamination audit changed: "
            f"{len(contaminated_ids)} contaminated, {len(filtered_ids)} retained"
        )

    full_metrics = _subset_metrics(systems, anchor_ids)
    filtered_metrics = _subset_metrics(systems, filtered_ids)
    by_full = {row["system"]: row for row in full_metrics}
    by_filtered = {row["system"]: row for row in filtered_metrics}
    comparison_rows = []
    for name in sorted(systems):
        full = by_full[name]
        filtered = by_filtered[name]
        comparison_rows.append(
            {
                "system": name,
                "full_n": full["n"],
                "full_accuracy": full["accuracy"],
                "full_macro_f1": full["macro_f1"],
                "decontaminated_n": filtered["n"],
                "decontaminated_accuracy": filtered["accuracy"],
                "decontaminated_macro_f1": filtered["macro_f1"],
                "accuracy_change_pp": 100.0
                * (filtered["accuracy"] - full["accuracy"]),
                "macro_f1_change": filtered["macro_f1"] - full["macro_f1"],
            }
        )

    report = {
        "analysis": "exact-normalized-overlap sensitivity analysis",
        "status": "exploratory robustness check",
        "normalization": (
            "NFKC + casefold + punctuation-to-space + whitespace collapse"
        ),
        "full_test_claims": len(anchor_ids),
        "known_contaminated_test_claims": len(contaminated_ids),
        "decontaminated_test_claims": len(filtered_ids),
        "excluded_claim_ids": sorted(contaminated_ids),
        "metrics": comparison_rows,
        "interpretation": (
            "Filtering known exact normalized overlaps is a sensitivity analysis, "
            "not a new untouched confirmatory test. Near-duplicate and test-guided "
            "development risks remain."
        ),
    }
    (output_dir / "sensitivity_analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "sensitivity_analysis_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    lines = [
        "# Exact-Overlap Sensitivity Analysis",
        "",
        "This exploratory robustness check removes the 39 test claims with exact",
        "normalized matches in train or validation, retaining 4,961 claims.",
        "It does not remove all likely near-duplicates and does not restore",
        "confirmatory independence after test-guided proxy development.",
        "",
        "| System | Full n | Full accuracy | Full macro-F1 | Filtered n | Filtered accuracy | Filtered macro-F1 | Accuracy change (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['system']} | {row['full_n']} | {row['full_accuracy']:.4f} | "
            f"{row['full_macro_f1']:.4f} | {row['decontaminated_n']} | "
            f"{row['decontaminated_accuracy']:.4f} | "
            f"{row['decontaminated_macro_f1']:.4f} | "
            f"{row['accuracy_change_pp']:+.3f} |"
        )
    (output_dir / "sensitivity_analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_path = output_dir / "ablation_study_predictions.csv"
    baseline_path = output_dir / "baseline_comparison_predictions.csv"
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        generated_utc = json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("generated_utc")
    else:
        generated_utc = None
    generated_utc = generated_utc or datetime.now(timezone.utc).isoformat()

    systems = _load(ablation_path, "variant")
    for name, rows in _load(baseline_path, "baseline").items():
        if name in systems:
            raise ValueError(f"duplicate system name across inputs: {name}")
        systems[name] = rows

    if "full_proxy" not in systems:
        raise ValueError("full_proxy is missing")
    anchor_ids = list(systems["full_proxy"])
    anchor_id_set = set(anchor_ids)
    validation: dict[str, dict[str, int | bool]] = {}
    for name, rows in systems.items():
        validation[name] = {
            "prediction_count": len(rows),
            "claim_ids_match_full_proxy": set(rows) == anchor_id_set,
        }
        if len(rows) != 5000 or set(rows) != anchor_id_set:
            raise ValueError(f"{name} does not align to the 5,000-claim anchor")

    _write_decontamination_sensitivity(output_dir, systems, anchor_ids)

    full_rows = [systems["full_proxy"][claim_id] for claim_id in anchor_ids]
    confusion = [
        {
            "gold_label": gold,
            **{
                predicted: sum(r.gold == gold and r.predicted == predicted for r in full_rows)
                for predicted in LABELS
            },
        }
        for gold in LABELS
    ]
    with (output_dir / "confusion_matrix_full_proxy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["gold_label", *LABELS])
        writer.writeheader()
        writer.writerows(confusion)

    per_class = [_class_metrics(full_rows, label) for label in LABELS]
    with (output_dir / "per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)

    system_metrics: list[dict] = []
    per_dataset: list[dict] = []
    for name, rows_by_id in systems.items():
        rows = [rows_by_id[claim_id] for claim_id in anchor_ids]
        correct = sum(row.correct for row in rows)
        lower, upper = _wilson(correct, len(rows))
        class_rows = [_class_metrics(rows, label) for label in LABELS]
        system_metrics.append(
            {
                "system": name,
                "n": len(rows),
                "accuracy": correct / len(rows),
                "accuracy_wilson_lower": lower,
                "accuracy_wilson_upper": upper,
                "macro_f1": sum(float(item["f1"]) for item in class_rows) / len(LABELS),
                "average_raw_confidence": sum(r.confidence for r in rows) / len(rows),
            }
        )
        datasets: dict[str, list[Prediction]] = defaultdict(list)
        for row in rows:
            datasets[row.dataset].append(row)
        for dataset, dataset_rows in sorted(datasets.items()):
            dataset_correct = sum(row.correct for row in dataset_rows)
            ds_lower, ds_upper = _wilson(dataset_correct, len(dataset_rows))
            per_dataset.append(
                {
                    "system": name,
                    "dataset": dataset,
                    "n": len(dataset_rows),
                    "accuracy": dataset_correct / len(dataset_rows),
                    "accuracy_wilson_lower": ds_lower,
                    "accuracy_wilson_upper": ds_upper,
                }
            )

    with (output_dir / "per_dataset_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_dataset[0]))
        writer.writeheader()
        writer.writerows(per_dataset)

    comparisons: list[dict] = []
    for index, name in enumerate(sorted(systems)):
        if name == "full_proxy":
            continue
        other_rows = [systems[name][claim_id] for claim_id in anchor_ids]
        full_wins = sum(left.correct and not right.correct for left, right in zip(full_rows, other_rows))
        full_losses = sum(not left.correct and right.correct for left, right in zip(full_rows, other_rows))
        discordant = full_wins + full_losses
        p_value = (
            float(binomtest(full_wins, discordant, 0.5, alternative="two-sided").pvalue)
            if discordant
            else 1.0
        )
        ci_lower, ci_upper = _paired_bootstrap(
            full_rows,
            other_rows,
            iterations=args.bootstrap_iterations,
            seed=args.seed + index,
        )
        comparisons.append(
            {
                "system": "full_proxy",
                "comparator": name,
                "n": len(anchor_ids),
                "full_accuracy": _accuracy(full_rows),
                "comparator_accuracy": _accuracy(other_rows),
                "paired_risk_difference": _accuracy(full_rows) - _accuracy(other_rows),
                "paired_bootstrap_lower": ci_lower,
                "paired_bootstrap_upper": ci_upper,
                "full_wins": full_wins,
                "full_losses": full_losses,
                "discordant_pairs": discordant,
                "matched_pair_odds_ratio": (full_wins + 0.5) / (full_losses + 0.5),
                "p_value_unadjusted": p_value,
            }
        )
    _holm_adjust(comparisons)

    with (output_dir / "paired_tests.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    report = {
        "metadata": {
            "generated_utc": generated_utc,
            "evaluation_type": "deterministic_proxy",
            "full_system_label": "full_proxy",
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "confidence_note": (
                "The proxy stores one heuristic raw confidence score. "
                "It does not store a multiclass probability vector, so a proper "
                "multiclass Brier score is not available."
            ),
        },
        "artifact_validation": validation,
        "system_metrics": system_metrics,
        "confusion_matrix_full_proxy": confusion,
        "per_class_metrics_full_proxy": per_class,
        "per_dataset_metrics": per_dataset,
        "paired_tests": comparisons,
    }
    (output_dir / "statistics_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    selected = {
        row["comparator"]: row
        for row in comparisons
        if row["comparator"] in {"majority", "length", "ablate_debate"}
    }
    lines = [
        "# Thesis Statistics Summary",
        "",
        "The 5,000-claim experiment evaluates the deterministic FactValidator-Proxy,",
        "not the live SerpAPI/SentenceTransformer/Ollama application pipeline.",
        "",
        "## System metrics",
        "",
        "| System | Accuracy | Macro-F1 | Wilson 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(system_metrics, key=lambda item: item["accuracy"], reverse=True):
        lines.append(
            f"| {row['system']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"[{row['accuracy_wilson_lower']:.4f}, {row['accuracy_wilson_upper']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Selected exact paired tests",
            "",
            "| Comparison | Full wins | Full losses | Exact two-sided p | Holm p | Matched OR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparator in ("majority", "length", "ablate_debate"):
        row = selected[comparator]
        lines.append(
            f"| full_proxy vs {comparator} | {row['full_wins']} | {row['full_losses']} | "
            f"{row['p_value_unadjusted']:.5f} | {row['p_value_holm']:.5f} | "
            f"{row['matched_pair_odds_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "No selected comparison is statistically significant after Holm correction.",
            "The no-debate proxy has the best observed point estimates, but its comparison",
            "with the full proxy has Holm-adjusted p approximately 0.0573. The result is",
            "descriptive evidence against always-on proxy debate, not confirmatory proof.",
            "",
            "All tests are descriptive and exploratory because proxy development was",
            "informed by observations on this benchmark and normalized split overlaps",
            "were identified retrospectively. No confirmatory superiority claim is made.",
            "",
            "The confidence output is reported only as a raw-score calibration diagnostic.",
            "A proper multiclass Brier score requires prob_supported, prob_refuted, and",
            "prob_nei for every prediction.",
        ]
    )
    (output_dir / "statistics_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote thesis statistics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
