"""
Prepare a research-grade benchmark dataset from the seed benchmark JSON.

Outputs:
- Canonical dataset with normalized labels
- Deterministic stratified train/val/test split
- Split metadata + balance verification

Usage:
  python Scripts/prepare_research_benchmark.py
  python Scripts/prepare_research_benchmark.py --input docs/evaluation_benchmark.json --output data/benchmarks/research_benchmark_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.dataset import DatasetManager


def parse_args() -> argparse.Namespace:
    default_input = str(REPO_ROOT / "docs" / "evaluation_benchmark.json")
    default_output = str(REPO_ROOT / "data" / "benchmarks" / "research_benchmark_v1.json")
    default_splits = str(REPO_ROOT / "data" / "benchmarks" / "splits")

    parser = argparse.ArgumentParser(description="Prepare research benchmark artifacts")
    parser.add_argument(
        "--input",
        default=default_input,
        help="Input benchmark JSON path",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Output canonical benchmark JSON path",
    )
    parser.add_argument(
        "--splits-dir",
        default=default_splits,
        help="Output directory for train/val/test split files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manager = DatasetManager(args.input)

    validation = manager.validate_dataset_quality()
    if not validation["ok"]:
        print("Dataset validation failed with errors:")
        for err in validation["errors"]:
            print(f"  - {err}")
        return 1

    changed = manager.normalize_claim_labels()
    canonical_path = manager.export_canonical_dataset(args.output)

    split = manager.stratified_split(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify_by="label",
        seed=args.seed,
    )
    split_paths = manager.export_split(split, output_dir=args.splits_dir)

    balance = manager.verify_split_balance(split, stratify_by="label", tolerance=0.35)
    split_info_path = Path(split_paths["info"])
    with open(split_info_path, "r") as f:
        split_info = json.load(f)
    split_info["balance_verification"] = balance
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print("Research benchmark prepared successfully.")
    print(f"- Input claims: {len(manager.claims)}")
    print(f"- Labels normalized: {changed}")
    print(f"- Canonical dataset: {canonical_path}")
    print(f"- Split files: {args.splits_dir}")
    print(f"- Split balanced: {balance['balanced']}")

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
