"""
Build benchmark JSON from annotated CSV template.

Input CSV columns expected:
- id, claim, label, category, difficulty
Optional:
- source_url, annotator_1, annotator_2, annotator_3, notes

Usage:
  python Scripts/build_benchmark_from_csv.py \
    --input data/benchmarks/claim_annotation_template_240.csv \
    --output docs/evaluation_benchmark_v2.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


VALID_LABELS = {
    "SUPPORTED": "SUPPORTED",
    "REFUTED": "REFUTED",
    "NEI": "NEI",
    "INSUFFICIENT EVIDENCE": "NEI",
    "INSUFFICIENT_EVIDENCE": "NEI",
    "MIXED / DISPUTED": "NEI",
    "MIXED": "NEI",
}

VALID_DIFFICULTY = {"easy", "medium", "hard"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark JSON from annotated CSV")
    parser.add_argument(
        "--input",
        default="data/benchmarks/claim_annotation_template_240.csv",
        help="Annotated CSV input file",
    )
    parser.add_argument(
        "--output",
        default="docs/evaluation_benchmark_v2.json",
        help="Output benchmark JSON path",
    )
    parser.add_argument(
        "--min-claims",
        type=int,
        default=30,
        help="Minimum number of valid annotated claims required",
    )
    return parser.parse_args()


def normalize_label(label: str) -> str | None:
    key = (label or "").strip().upper()
    return VALID_LABELS.get(key)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    claims = []
    errors = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            cid = (row.get("id") or "").strip()
            claim = (row.get("claim") or "").strip()
            raw_label = (row.get("label") or "").strip()
            label = normalize_label(raw_label)
            category = (row.get("category") or "general").strip().lower() or "general"
            difficulty = (row.get("difficulty") or "medium").strip().lower() or "medium"

            # Skip empty template rows.
            if not claim and not raw_label:
                continue

            if not cid:
                errors.append(f"Line {idx}: missing id")
                continue
            if not claim:
                errors.append(f"Line {idx}: missing claim text for id={cid}")
                continue
            if label is None:
                errors.append(f"Line {idx}: invalid label '{raw_label}' for id={cid}")
                continue
            if difficulty not in VALID_DIFFICULTY:
                errors.append(f"Line {idx}: invalid difficulty '{difficulty}' for id={cid}")
                continue

            claims.append(
                {
                    "id": cid,
                    "claim": claim,
                    "label": label,
                    "category": category,
                    "difficulty": difficulty,
                }
            )

    if errors:
        print("Validation errors found:")
        for err in errors[:50]:
            print(f"- {err}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more")
        return 1

    if len(claims) < args.min_claims:
        print(
            f"Not enough valid claims: {len(claims)} found, need at least {args.min_claims}. "
            "Fill more rows in the annotation CSV and retry."
        )
        return 1

    label_counts = Counter(c["label"] for c in claims)
    diff_counts = Counter(c["difficulty"] for c in claims)
    cat_counts = Counter(c["category"] for c in claims)

    payload = {
        "version": "research-v2",
        "updated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": "Expanded benchmark built from annotated CSV template for statistical-significance evaluation.",
        "metadata": {
            "claim_count": len(claims),
            "label_distribution": dict(label_counts),
            "difficulty_distribution": dict(diff_counts),
            "category_distribution": dict(cat_counts),
            "source_csv": str(input_path),
        },
        "claims": claims,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Benchmark JSON built successfully.")
    print(f"- Output: {output_path}")
    print(f"- Claims: {len(claims)}")
    print(f"- Labels: {dict(label_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
