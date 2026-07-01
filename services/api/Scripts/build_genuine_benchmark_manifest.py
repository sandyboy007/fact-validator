"""
Build a provenance-preserving benchmark manifest from one or more public datasets.

This script is intended for genuine benchmark claims only. It preserves original
dataset identifiers, source provenance, and label normalization so the final
IEEE submission can report a defensible benchmark construction process.

Usage examples:
  python Scripts/build_genuine_benchmark_manifest.py \
    --input fever=C:\data\fever.json \
    --input liar=C:\data\liar.csv \
    --output data\benchmarks\results\genuine_benchmark_manifest.json

Supported input formats:
  - JSON with a top-level "claims" list
  - CSV with columns such as id, claim, label, category, difficulty
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
class ManifestClaim:
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
    parser = argparse.ArgumentParser(description="Build a genuine benchmark manifest")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Dataset input in the form name=path (repeatable)",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/results/genuine_benchmark_manifest.json",
        help="Output manifest JSON path",
    )
    parser.add_argument(
        "--min-claims",
        type=int,
        default=500,
        help="Minimum number of retained claims required for the manifest",
    )
    parser.add_argument(
        "--write-markdown",
        action="store_true",
        help="Also write a human-readable markdown summary next to the JSON output",
    )
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


def load_json_claims(path: Path, dataset_name: str) -> list[ManifestClaim]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    claims = payload.get("claims", []) if isinstance(payload, dict) else []
    rows: list[ManifestClaim] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("claim") or claim.get("text") or "").strip()
        if not text:
            continue
        label = normalize_label(str(claim.get("label") or claim.get("verdict") or ""))
        rows.append(
            ManifestClaim(
                source_dataset=dataset_name,
                source_path=str(path),
                source_id=str(claim.get("id") or claim.get("claim_id") or f"{dataset_name}-{index + 1}"),
                claim=text,
                label=label,
                category=str(claim.get("category") or claim.get("topic") or "general"),
                difficulty=str(claim.get("difficulty") or "medium"),
                source_url=str(claim.get("source_url") or claim.get("url") or ""),
                provenance_note=str(claim.get("provenance_note") or claim.get("notes") or ""),
            )
        )
    return rows


def load_csv_claims(path: Path, dataset_name: str) -> list[ManifestClaim]:
    rows: list[ManifestClaim] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            text = str(row.get("claim") or row.get("text") or row.get("claim_original") or "").strip()
            if not text:
                continue
            label = normalize_label(str(row.get("label") or row.get("verdict") or row.get("ground_truth_label") or ""))
            rows.append(
                ManifestClaim(
                    source_dataset=dataset_name,
                    source_path=str(path),
                    source_id=str(row.get("id") or row.get("claim_id") or f"{dataset_name}-{index}"),
                    claim=text,
                    label=label,
                    category=str(row.get("category") or row.get("topic") or "general"),
                    difficulty=str(row.get("difficulty") or "medium"),
                    source_url=str(row.get("source_url") or row.get("url") or ""),
                    provenance_note=str(row.get("notes") or row.get("provenance_note") or ""),
                )
            )
    return rows


def load_dataset(spec: str) -> list[ManifestClaim]:
    if "=" not in spec:
        raise ValueError(f"invalid input spec '{spec}', expected name=path")
    dataset_name, path_str = spec.split("=", 1)
    dataset_name = dataset_name.strip()
    path = Path(path_str.strip())
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_claims(path, dataset_name)
    if suffix == ".csv":
        return load_csv_claims(path, dataset_name)
    raise ValueError(f"unsupported input format for {path}")


def deduplicate_claims(claims: Iterable[ManifestClaim]) -> list[ManifestClaim]:
    by_norm: dict[str, ManifestClaim] = {}
    aliases: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for claim in claims:
        key = normalize_text(claim.claim)
        if not key:
            continue
        if key not in by_norm:
            by_norm[key] = claim
        aliases[key].append({
            "source_dataset": claim.source_dataset,
            "source_id": claim.source_id,
            "source_path": claim.source_path,
        })

    deduped: list[ManifestClaim] = []
    for key, claim in by_norm.items():
        payload = asdict(claim)
        payload["aliases"] = aliases[key]
        deduped.append(payload)
    return deduped


def build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Genuine Benchmark Manifest",
        "",
        f"- Generated UTC: {report['generated_utc']}",
        f"- Retained claims: {report['retained_claims']}",
        f"- Source datasets: {', '.join(report['source_datasets'])}",
        "",
        "## Label Distribution",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(report["label_distribution"].items()):
        lines.append(f"| {label} | {count} |")

    lines.extend([
        "",
        "## Dataset Distribution",
        "",
        "| Dataset | Count |",
        "|---|---:|",
    ])
    for dataset, count in sorted(report["dataset_distribution"].items()):
        lines.append(f"| {dataset} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    all_claims: list[ManifestClaim] = []
    source_datasets: list[str] = []
    for spec in args.input:
        dataset_name = spec.split("=", 1)[0].strip()
        source_datasets.append(dataset_name)
        all_claims.extend(load_dataset(spec))

    deduped = deduplicate_claims(all_claims)
    if len(deduped) < args.min_claims:
        print(f"Retained only {len(deduped)} claims; need at least {args.min_claims}.")
        return 1

    label_counts = Counter(item["label"] for item in deduped)
    dataset_counts = Counter(item["source_dataset"] for item in deduped)

    report = {
        "version": "genuine-benchmark-manifest-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_datasets": source_datasets,
        "retained_claims": len(deduped),
        "label_distribution": dict(label_counts),
        "dataset_distribution": dict(dataset_counts),
        "claims": deduped,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Genuine benchmark manifest built successfully.")
    print(f"- Output: {output_path}")
    print(f"- Retained claims: {len(deduped)}")
    print(f"- Labels: {dict(label_counts)}")
    print(f"- Datasets: {dict(dataset_counts)}")

    if args.write_markdown:
        md_path = output_path.with_suffix(".md")
        md_path.write_text(build_markdown(report), encoding="utf-8")
        print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())