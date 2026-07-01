"""
Empirically calibrate domain credibility weights from annotated evidence data.

The script expects a CSV file with at least the following columns:
  - domain: evidence domain or host
  - label: binary or ordinal trust outcome

Accepted positive labels:
  SUPPORTED, TRUE, 1, YES, TRUSTED

Accepted negative labels:
  REFUTED, FALSE, 0, NO, UNTRUSTED

The output is a JSON mapping from domain to an empirically justified delta
relative to the neutral 50-point credibility baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


POSITIVE_LABELS = {"supported", "true", "1", "yes", "trusted"}
NEGATIVE_LABELS = {"refuted", "false", "0", "no", "untrusted"}


@dataclass
class DomainCalibration:
    domain: str
    n: int
    positive_rate: float
    delta: int
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirically calibrate credibility weights")
    parser.add_argument("--input", required=True, help="CSV with domain and label columns")
    parser.add_argument("--domain-column", default="domain", help="Domain column name")
    parser.add_argument("--label-column", default="label", help="Label column name")
    parser.add_argument("--output", default="data/benchmarks/results/credibility_calibration_report.json")
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def normalize_label(label: str) -> int:
    raw = (label or "").strip().lower()
    if raw in POSITIVE_LABELS:
        return 1
    if raw in NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unsupported calibration label: {label}")


def read_rows(path: Path, domain_column: str, label_column: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            domain = (row.get(domain_column) or "").strip().lower()
            label = (row.get(label_column) or "").strip()
            if not domain:
                raise ValueError(f"Missing domain at row {index}")
            rows.append((domain, normalize_label(label)))
    return rows


def learn_domain_weights(rows: Iterable[tuple[str, int]]) -> list[DomainCalibration]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for domain, target in rows:
        grouped[domain].append(int(target))

    global_rate = sum(sum(values) for values in grouped.values()) / max(1, sum(len(values) for values in grouped.values()))
    calibrations: list[DomainCalibration] = []

    for domain, values in sorted(grouped.items(), key=lambda item: item[0]):
        n = len(values)
        positive_rate = sum(values) / n if n else 0.0
        # Smoothed delta centered around the neutral 50-point baseline.
        raw_delta = round((positive_rate - global_rate) * 40)
        delta = int(max(-30, min(35, raw_delta)))
        if delta > 0:
            rationale = "Observed trust rate above global average; boost domain credibility."
        elif delta < 0:
            rationale = "Observed trust rate below global average; penalize domain credibility."
        else:
            rationale = "Observed trust rate near global average; keep neutral."
        calibrations.append(
            DomainCalibration(
                domain=domain,
                n=n,
                positive_rate=positive_rate,
                delta=delta,
                rationale=rationale,
            )
        )

    return calibrations


def build_markdown(report: Dict) -> str:
    lines = [
        "# Credibility Calibration Report",
        "",
        f"- Generated UTC: {report['generated_utc']}",
        f"- Global positive rate: {report['global_positive_rate']:.4f}",
        "",
        "| Domain | n | Positive rate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for item in report["calibrations"]:
        lines.append(f"| {item['domain']} | {item['n']} | {item['positive_rate']:.3f} | {item['delta']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    rows = read_rows(input_path, args.domain_column, args.label_column)
    calibrations = learn_domain_weights(rows)

    global_rate = sum(item["positive_rate"] * item["n"] for item in [c.__dict__ for c in calibrations]) / max(1, sum(item["n"] for item in [c.__dict__ for c in calibrations]))
    report = {
        "version": "credibility-calibration-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": str(input_path),
        "global_positive_rate": global_rate,
        "calibrations": [c.__dict__ for c in calibrations],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Credibility calibration completed.")
    print(f"- Output: {output_path}")
    print(f"- Domains calibrated: {len(calibrations)}")

    if args.write_markdown:
        md_path = output_path.with_suffix(".md")
        md_path.write_text(build_markdown(report), encoding="utf-8")
        print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())