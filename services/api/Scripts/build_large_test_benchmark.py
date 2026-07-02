"""
Build an exact-size large test benchmark from genuine public claim datasets.

This script is intended to replace the earlier 51-claim evaluation split with a
much larger, review-grade benchmark. It preserves provenance, deduplicates on
normalized claim text, and can export a fixed-size test set of 5000 claims when
enough source claims are provided.

Recommended sources:
  - FEVER
  - LIAR (after remapping / label normalization)
  - SciFact
  - HealthVer

Example usage:
  python Scripts/build_large_test_benchmark.py \
    --input fever=data/fever.json \
    --input liar=data/liar.csv \
    --input scifact=data/scifact.json \
    --input healthver=data/healthver.csv \
    --target-test-size 5000 \
    --output data/benchmarks/results/large_benchmark_manifest.json \
    --splits-dir data/benchmarks/splits_5000

Notes:
  - If fewer than 5000 unique claims are available, the script exits with an
    informative error rather than silently padding with synthetic claims.
  - A secondary robustness set can be derived later from the exported test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import random
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.dataset import DatasetManager


LABEL_ALIASES = {
    "supported": "SUPPORTED",
    "support": "SUPPORTED",
    "true": "SUPPORTED",
    "refuted": "REFUTED",
    "false": "REFUTED",
    "nei": "NEI",
    "not enough information": "NEI",
    "insufficient evidence": "NEI",
    "mixed": "NEI",
    "disputed": "NEI",
    "mixed / disputed": "NEI",
}


@dataclass
class BenchmarkClaim:
    source_dataset: str
    source_path: str
    source_id: str
    claim: str
    label: str
    category: str
    difficulty: str
    source_url: str = ""
    provenance_note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a large exact-size benchmark")
    parser.add_argument("--input", action="append", required=True, help="Dataset input in the form name=path")
    parser.add_argument("--target-test-size", type=int, default=5000, help="Exact test-set size")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "large_benchmark_manifest.json"))
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "data" / "benchmarks" / "splits_5000"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for train split from remaining claims")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio for val split from remaining claims")
    parser.add_argument("--test-ratio", type=float, default=1.0, help="Kept for backward compatibility")
    return parser.parse_args()


def normalize_label(label: str) -> str:
    raw = (label or "").strip()
    if not raw:
        raise ValueError("missing label")
    upper = raw.upper()
    if upper in {"SUPPORTED", "REFUTED", "NEI"}:
        return upper
    mapped = LABEL_ALIASES.get(raw.lower())
    if mapped:
        return mapped
    raise ValueError(f"unsupported label: {label}")


def normalize_text(text: str) -> str:
    low = (text or "").lower().strip()
    low = re.sub(r"[^a-z0-9\s]", " ", low)
    low = re.sub(r"\s+", " ", low)
    return low[:500]


def _load_json_claims(path: Path, dataset_name: str) -> list[BenchmarkClaim]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    claims = payload.get("claims", []) if isinstance(payload, dict) else []
    out: list[BenchmarkClaim] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("claim") or claim.get("text") or "").strip()
        if not text:
            continue
        out.append(
            BenchmarkClaim(
                source_dataset=dataset_name,
                source_path=str(path),
                source_id=str(claim.get("id") or claim.get("claim_id") or f"{dataset_name}-{index + 1}"),
                claim=text,
                label=normalize_label(str(claim.get("label") or claim.get("verdict") or "")),
                category=str(claim.get("category") or claim.get("topic") or "general"),
                difficulty=str(claim.get("difficulty") or "medium"),
                source_url=str(claim.get("source_url") or claim.get("url") or ""),
                provenance_note=str(claim.get("provenance_note") or claim.get("notes") or ""),
            )
        )
    return out


def _load_csv_claims(path: Path, dataset_name: str) -> list[BenchmarkClaim]:
    out: list[BenchmarkClaim] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            text = str(row.get("claim") or row.get("text") or row.get("claim_original") or "").strip()
            if not text:
                continue
            out.append(
                BenchmarkClaim(
                    source_dataset=dataset_name,
                    source_path=str(path),
                    source_id=str(row.get("id") or row.get("claim_id") or f"{dataset_name}-{index}"),
                    claim=text,
                    label=normalize_label(str(row.get("label") or row.get("verdict") or row.get("ground_truth_label") or "")),
                    category=str(row.get("category") or row.get("topic") or "general"),
                    difficulty=str(row.get("difficulty") or "medium"),
                    source_url=str(row.get("source_url") or row.get("url") or ""),
                    provenance_note=str(row.get("notes") or row.get("provenance_note") or ""),
                )
            )
    return out


def load_dataset(spec: str) -> list[BenchmarkClaim]:
    if "=" not in spec:
        raise ValueError(f"invalid input spec '{spec}', expected name=path")
    dataset_name, path_str = spec.split("=", 1)
    path = Path(path_str.strip())
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    if path.suffix.lower() == ".json":
        return _load_json_claims(path, dataset_name.strip())
    if path.suffix.lower() == ".csv":
        return _load_csv_claims(path, dataset_name.strip())
    raise ValueError(f"unsupported input format: {path}")


def deduplicate_claims(claims: Iterable[BenchmarkClaim]) -> list[dict[str, Any]]:
    seen: dict[str, BenchmarkClaim] = {}
    aliases: dict[str, list[dict[str, str]]] = {}
    for claim in claims:
        key = normalize_text(claim.claim)
        if not key:
            continue
        if key not in seen:
            seen[key] = claim
        aliases.setdefault(key, []).append(
            {
                "source_dataset": claim.source_dataset,
                "source_id": claim.source_id,
                "source_path": claim.source_path,
            }
        )
    deduped: list[dict[str, Any]] = []
    for key, claim in seen.items():
        item = asdict(claim)
        # Keep a stable generic id field so downstream evaluators can align rows.
        item["id"] = claim.source_id
        item["aliases"] = aliases.get(key, [])
        deduped.append(item)
    return deduped


def _split_remaining_claims(
    remaining_claims: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not remaining_claims:
        return [], []

    total_ratio = train_ratio + val_ratio
    if total_ratio <= 0:
        return [], []

    train_ratio_norm = train_ratio / total_ratio
    by_label: dict[str, list[dict[str, Any]]] = {}
    for claim in remaining_claims:
        label = str(claim.get("label", "NEI"))
        by_label.setdefault(label, []).append(claim)

    rng = random.Random(seed)
    train_claims: list[dict[str, Any]] = []
    val_claims: list[dict[str, Any]] = []

    for label_claims in by_label.values():
        rng.shuffle(label_claims)
        cutoff = int(round(len(label_claims) * train_ratio_norm))
        train_claims.extend(label_claims[:cutoff])
        val_claims.extend(label_claims[cutoff:])

    rng.shuffle(train_claims)
    rng.shuffle(val_claims)
    return train_claims, val_claims


def main() -> int:
    args = parse_args()
    all_claims: list[BenchmarkClaim] = []
    source_datasets: list[str] = []
    for spec in args.input:
        source_datasets.append(spec.split("=", 1)[0].strip())
        all_claims.extend(load_dataset(spec))

    deduped = deduplicate_claims(all_claims)
    if len(deduped) < args.target_test_size:
        print(
            f"Insufficient unique claims for a {args.target_test_size}-claim test set: "
            f"found {len(deduped)} after deduplication."
        )
        return 1

    # Use the dataset utilities to create a deterministic split where the test set
    # is exactly the requested size.
    temp_path = Path(args.output).with_suffix(".source.json")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(json.dumps({"claims": deduped}, indent=2), encoding="utf-8")

    manager = DatasetManager(str(temp_path))
    manager.normalize_claim_labels()
    canonical_path = manager.export_canonical_dataset(str(Path(args.output).with_suffix(".canonical.json")))

    split = manager.stratified_split(
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
        stratify_by="label",
        seed=args.seed,
    )
    _, _, test_claims = manager.get_split_data(split)

    if len(test_claims) < args.target_test_size:
        print(
            f"The stratified split produced only {len(test_claims)} test claims; "
            f"need {args.target_test_size}. Adjust the source pool or split ratios."
        )
        return 1

    # Keep only the requested number of test claims for evaluation.
    test_claims = test_claims[: args.target_test_size]

    test_keys = {normalize_text(str(item.get("claim", ""))) for item in test_claims}
    remaining_claims = [item for item in deduped if normalize_text(str(item.get("claim", ""))) not in test_keys]
    train_claims, val_claims = _split_remaining_claims(
        remaining_claims,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    test_payload = {
        "claims": test_claims,
        "metadata": {
            "seed": args.seed,
            "test_count": len(test_claims),
            "target_test_size": args.target_test_size,
            "source_datasets": source_datasets,
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }

    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / "train.json"
    val_path = splits_dir / "val.json"
    test_path = splits_dir / "test.json"
    train_path.write_text(
        json.dumps(
            {
                "claims": train_claims,
                "metadata": {
                    "seed": args.seed,
                    "split": "train",
                    "count": len(train_claims),
                    "source_datasets": source_datasets,
                    "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    val_path.write_text(
        json.dumps(
            {
                "claims": val_claims,
                "metadata": {
                    "seed": args.seed,
                    "split": "val",
                    "count": len(val_claims),
                    "source_datasets": source_datasets,
                    "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    test_path.write_text(json.dumps(test_payload, indent=2), encoding="utf-8")

    label_counts = Counter(item["label"] for item in test_claims)
    dataset_counts = Counter(item["source_dataset"] for item in test_claims)

    manifest = {
        "version": "large-test-benchmark-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_datasets": source_datasets,
        "target_test_size": args.target_test_size,
        "retained_claims": len(deduped),
        "test_claim_count": len(test_claims),
        "train_claim_count": len(train_claims),
        "val_claim_count": len(val_claims),
        "label_distribution": dict(label_counts),
        "dataset_distribution": dict(dataset_counts),
        "canonical_dataset": canonical_path,
        "train_split_path": str(train_path),
        "val_split_path": str(val_path),
        "test_split_path": str(test_path),
        "claims": test_claims,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Large benchmark test set built successfully.")
    print(f"- Output: {output_path}")
    print(f"- Train split: {train_path} ({len(train_claims)} claims)")
    print(f"- Val split: {val_path} ({len(val_claims)} claims)")
    print(f"- Test split: {test_path}")
    print(f"- Test claims: {len(test_claims)}")
    print(f"- Label distribution: {dict(label_counts)}")
    print(f"- Dataset distribution: {dict(dataset_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())