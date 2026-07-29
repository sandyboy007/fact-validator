"""Fail when frozen thesis artifacts are incomplete or internally inconsistent."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
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
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _load_split(path: Path) -> tuple[set[str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ValueError(f"{path}: claims list missing")
    ids = {str(row.get("id") or row.get("source_id") or "") for row in claims}
    normalized = {_normalize(str(row.get("claim", ""))) for row in claims}
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


def main() -> int:
    split_ids: dict[str, set[str]] = {}
    split_text: dict[str, set[str]] = {}
    for filename, expected in EXPECTED_SPLIT_COUNTS.items():
        ids, normalized = _load_split(SPLIT_DIR / filename)
        if len(ids) != expected:
            raise ValueError(f"{filename}: expected {expected}, found {len(ids)}")
        split_ids[filename] = ids
        split_text[filename] = normalized

    filenames = list(EXPECTED_SPLIT_COUNTS)
    for index, left in enumerate(filenames):
        for right in filenames[index + 1 :]:
            id_overlap = split_ids[left] & split_ids[right]
            text_overlap = split_text[left] & split_text[right]
            if id_overlap or text_overlap:
                raise ValueError(
                    f"split overlap {left}/{right}: "
                    f"{len(id_overlap)} IDs, {len(text_overlap)} normalized claims"
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
    ]
    missing_outputs = [name for name in required_outputs if not (RESULT_DIR / name).is_file()]
    if missing_outputs:
        raise ValueError(f"missing statistical outputs: {missing_outputs}")
    _validate_per_dataset_metrics(
        RESULT_DIR / "per_dataset_metrics.csv", set(systems)
    )

    print(
        f"Validated {len(systems)} systems, 5,000 aligned predictions each, "
        "disjoint splits, manifest hashes, and statistical outputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
