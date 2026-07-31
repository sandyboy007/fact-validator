"""Fail when frozen thesis artifacts are incomplete or internally inconsistent."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.dataset import normalize_claim_text
SPLIT_DIR = REPO_ROOT / "data" / "benchmarks" / "splits_5000"
RESULT_DIR = REPO_ROOT / "data" / "benchmarks" / "results_5000"
EXPECTED_SPLIT_COUNTS = {"train.json": 11986, "val.json": 2997, "test.json": 5000}
EXPECTED_DATASET_COUNTS = {
    "FEVER": 1829,
    "LIAR": 1490,
    "PUBHEALTH_health_fact": 1490,
    "SciFact": 191,
}
REQUIRED_MANIFEST_FIELDS = {
    "experiment_name",
    "evaluation_type",
    "git_commit",
    "generated_utc",
    "random_seed",
    "python_version",
    "platform",
    "train_claims",
    "validation_claims",
    "test_claims",
    "proxy_components",
    "live_components_not_executed",
    "input_files",
    "sha256",
    "sha256_mode",
    "evidence_status",
    "confirmatory_claims_permitted",
    "split_isolation",
}


def _sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-disjoint",
        action="store_true",
        help="Fail on any normalized cross-split overlap (for new confirmatory sets).",
    )
    return parser.parse_args()


def _load_split(path: Path) -> tuple[set[str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ValueError(f"{path}: claims list missing")
    ids = {str(row.get("id") or row.get("source_id") or "") for row in claims}
    normalized = {normalize_claim_text(str(row.get("claim", ""))) for row in claims}
    if "" in ids or "" in normalized:
        raise ValueError(f"{path}: blank claim identifier or text")
    if len(ids) != len(claims):
        raise ValueError(f"{path}: duplicate claim identifiers")
    return ids, normalized


def _prediction_systems(path: Path, column: str) -> dict[str, set[str]]:
    systems: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row[column]
            systems.setdefault(name, set()).add(row["claim_id"])
            counts[name] = counts.get(name, 0) + 1
    for name, ids in systems.items():
        if counts[name] != 5000 or len(ids) != 5000:
            raise ValueError(
                f"{path.name}:{name} has {counts[name]} rows and {len(ids)} unique IDs"
            )
    return systems


def _validate_per_dataset_metrics(path: Path, systems: set[str]) -> None:
    observed: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            system = row["system"]
            dataset = row["dataset"]
            if dataset in observed.setdefault(system, {}):
                raise ValueError(f"{path.name}: duplicate {system}/{dataset} row")
            observed[system][dataset] = int(row["n"])
    if set(observed) != systems:
        raise ValueError(f"{path.name}: system set differs from prediction artifacts")
    for system, counts in observed.items():
        if counts != EXPECTED_DATASET_COUNTS:
            raise ValueError(
                f"{path.name}:{system} dataset counts differ: {counts}"
            )


def _validate_sensitivity_report(path: Path, systems: set[str]) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("full_test_claims") != 5000:
        raise ValueError("sensitivity report must retain the 5,000-claim reference")
    if report.get("known_contaminated_test_claims") != 39:
        raise ValueError("sensitivity report must exclude exactly 39 test claims")
    if report.get("decontaminated_test_claims") != 4961:
        raise ValueError("sensitivity report must contain exactly 4,961 claims")
    observed_systems = {row["system"] for row in report.get("metrics", [])}
    if observed_systems != systems:
        raise ValueError("sensitivity report system set differs from predictions")
    if any(row["decontaminated_n"] != 4961 for row in report["metrics"]):
        raise ValueError("sensitivity metrics are not aligned to 4,961 claims")


def main() -> int:
    args = parse_args()
    split_ids: dict[str, set[str]] = {}
    split_text: dict[str, set[str]] = {}
    for filename, expected in EXPECTED_SPLIT_COUNTS.items():
        ids, normalized = _load_split(SPLIT_DIR / filename)
        if len(ids) != expected:
            raise ValueError(f"{filename}: expected {expected}, found {len(ids)}")
        split_ids[filename] = ids
        split_text[filename] = normalized

    observed_overlaps: dict[str, int] = {}
    filenames = list(EXPECTED_SPLIT_COUNTS)
    for index, left in enumerate(filenames):
        for right in filenames[index + 1 :]:
            id_overlap = split_ids[left] & split_ids[right]
            text_overlap = split_text[left] & split_text[right]
            if id_overlap:
                raise ValueError(
                    f"split overlap {left}/{right}: "
                    f"{len(id_overlap)} IDs, {len(text_overlap)} normalized claims"
                )
            observed_overlaps[f"{left}/{right}"] = len(text_overlap)
            if args.require_disjoint and text_overlap:
                raise ValueError(
                    f"split overlap {left}/{right}: "
                    f"{len(text_overlap)} normalized claims"
                )

    systems = {}
    systems.update(
        _prediction_systems(
            RESULT_DIR / "ablation_study_predictions.csv", "variant"
        )
    )
    systems.update(
        _prediction_systems(
            RESULT_DIR / "baseline_comparison_predictions.csv", "baseline"
        )
    )
    anchor = systems.get("full_proxy")
    if not anchor:
        raise ValueError("full_proxy predictions missing")
    for name, ids in systems.items():
        if ids != anchor:
            raise ValueError(f"claim IDs differ for {name}")
    if anchor != split_ids["test.json"]:
        raise ValueError("prediction claim IDs differ from test split")

    manifest_path = RESULT_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"run_manifest.json missing fields: {sorted(missing)}")
    declared_overlaps = manifest["split_isolation"][
        "known_pairwise_overlap_counts"
    ]
    if observed_overlaps != declared_overlaps:
        raise ValueError(
            "observed normalized overlaps differ from the exploratory manifest: "
            f"{observed_overlaps}"
        )
    if manifest["evidence_status"] != "exploratory":
        raise ValueError("overlapping historical benchmark must be marked exploratory")
    if manifest["confirmatory_claims_permitted"] is not False:
        raise ValueError("exploratory benchmark cannot permit confirmatory claims")
    if manifest["sha256_mode"] != "canonical-lf-bytes":
        raise ValueError("manifest must use line-ending-independent hashes")
    for relative, expected_hash in manifest["sha256"].items():
        path = REPO_ROOT / relative
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"hash mismatch: {relative}")

    required_outputs = [
        "statistics_report.json",
        "confusion_matrix_full_proxy.csv",
        "per_class_metrics.csv",
        "per_dataset_metrics.csv",
        "paired_tests.csv",
        "statistics_summary.md",
        "sensitivity_analysis_report.json",
        "sensitivity_analysis_metrics.csv",
        "sensitivity_analysis_summary.md",
    ]
    missing_outputs = [name for name in required_outputs if not (RESULT_DIR / name).is_file()]
    if missing_outputs:
        raise ValueError(f"missing statistical outputs: {missing_outputs}")
    _validate_per_dataset_metrics(
        RESULT_DIR / "per_dataset_metrics.csv", set(systems)
    )
    _validate_sensitivity_report(
        RESULT_DIR / "sensitivity_analysis_report.json", set(systems)
    )

    print(
        f"Validated {len(systems)} systems, 5,000 aligned predictions each, "
        f"declared exploratory overlaps {observed_overlaps}, canonical-LF "
        "manifest hashes, and statistical outputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
