"""
Fetch and normalize public fact-checking datasets into CSV files compatible
with the 5000 benchmark pipeline.

Datasets used:
- FEVER (v1.0)
- LIAR
- SciFact (claims config)
- Health-Fact (used as health-domain substitute for HealthVer)

Outputs default to:
- data/benchmarks/external_templates/fever_filled.csv
- data/benchmarks/external_templates/liar_filled.csv
- data/benchmarks/external_templates/scifact_filled.csv
- data/benchmarks/external_templates/healthver_filled.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize external benchmark datasets")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "external_templates"),
        help="Directory where normalized CSV files are written",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fever-max", type=int, default=8000)
    parser.add_argument("--liar-max", type=int, default=6000)
    parser.add_argument("--scifact-max", type=int, default=1200)
    parser.add_argument("--health-max", type=int, default=6000)
    return parser.parse_args()


def _canonical_label(raw: Any, source: str) -> str | None:
    if source == "fever":
        text = str(raw).strip().upper()
        mapping = {
            "SUPPORTS": "SUPPORTED",
            "REFUTES": "REFUTED",
            "NOT ENOUGH INFO": "NEI",
        }
        return mapping.get(text)

    if source == "liar":
        try:
            label_id = int(raw)
        except Exception:
            return None
        # LIAR labels: 0 false, 1 half-true, 2 mostly-true, 3 true, 4 barely-true, 5 pants-fire
        if label_id in {2, 3}:
            return "SUPPORTED"
        if label_id in {0, 5}:
            return "REFUTED"
        if label_id in {1, 4}:
            return "NEI"
        return None

    if source == "scifact":
        text = str(raw).strip().upper()
        if text == "SUPPORT":
            return "SUPPORTED"
        if text == "CONTRADICT":
            return "REFUTED"
        if text == "":
            return "NEI"
        return None

    if source == "health":
        # health_fact labels: 0 false, 1 mixture, 2 true, 3 unproven, -1 missing
        try:
            label_id = int(raw)
        except Exception:
            return None
        mapping = {
            0: "REFUTED",
            1: "NEI",
            2: "SUPPORTED",
            3: "NEI",
        }
        return mapping.get(label_id)

    return None


def _difficulty_from_text(claim: str) -> str:
    length = len((claim or "").split())
    if length <= 10:
        return "easy"
    if length <= 20:
        return "medium"
    return "hard"


def _sample_rows(rows: list[dict[str, str]], max_count: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= max_count:
        return rows
    rng = random.Random(seed)
    out = rows[:]
    rng.shuffle(out)
    return out[:max_count]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "claim", "label", "category", "difficulty", "source_url", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_fever(limit: int, seed: int) -> list[dict[str, str]]:
    ds = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)
    rows: list[dict[str, str]] = []
    for item in ds:
        claim = str(item.get("claim") or "").strip()
        label = _canonical_label(item.get("label"), "fever")
        if not claim or not label:
            continue
        rows.append(
            {
                "id": f"fever-{item.get('id')}",
                "claim": claim,
                "label": label,
                "category": "general",
                "difficulty": _difficulty_from_text(claim),
                "source_url": "https://huggingface.co/datasets/fever",
                "notes": "source_dataset=fever;split=train;version=v1.0",
            }
        )
    return _sample_rows(rows, limit, seed)


def _load_liar(limit: int, seed: int) -> list[dict[str, str]]:
    ds = load_dataset("liar", "default", split="train", trust_remote_code=True)
    rows: list[dict[str, str]] = []
    for item in ds:
        claim = str(item.get("statement") or "").strip()
        label = _canonical_label(item.get("label"), "liar")
        if not claim or not label:
            continue
        category = str(item.get("subject") or "general")
        rows.append(
            {
                "id": f"liar-{item.get('id')}",
                "claim": claim,
                "label": label,
                "category": category,
                "difficulty": _difficulty_from_text(claim),
                "source_url": "https://huggingface.co/datasets/liar",
                "notes": "source_dataset=liar;split=train",
            }
        )
    return _sample_rows(rows, limit, seed)


def _load_scifact(limit: int, seed: int) -> list[dict[str, str]]:
    ds = load_dataset("scifact", "claims", split="train", trust_remote_code=True)
    rows: list[dict[str, str]] = []
    for item in ds:
        claim = str(item.get("claim") or "").strip()
        label = _canonical_label(item.get("evidence_label"), "scifact")
        if not claim or not label:
            continue
        rows.append(
            {
                "id": f"scifact-{item.get('id')}",
                "claim": claim,
                "label": label,
                "category": "science",
                "difficulty": _difficulty_from_text(claim),
                "source_url": "https://huggingface.co/datasets/scifact",
                "notes": "source_dataset=scifact;config=claims;split=train",
            }
        )
    return _sample_rows(rows, limit, seed)


def _load_health(limit: int, seed: int) -> list[dict[str, str]]:
    ds = load_dataset("health_fact", "default", split="train", trust_remote_code=True)
    rows: list[dict[str, str]] = []
    for item in ds:
        claim = str(item.get("claim") or "").strip()
        label = _canonical_label(item.get("label"), "health")
        if not claim or not label:
            continue
        category = "health"
        rows.append(
            {
                "id": f"health-{item.get('claim_id')}",
                "claim": claim,
                "label": label,
                "category": category,
                "difficulty": _difficulty_from_text(claim),
                "source_url": "https://huggingface.co/datasets/health_fact",
                "notes": "source_dataset=health_fact;split=train;used_as_healthver_substitute=true",
            }
        )
    return _sample_rows(rows, limit, seed)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)

    fever_rows = _load_fever(args.fever_max, args.seed)
    liar_rows = _load_liar(args.liar_max, args.seed)
    scifact_rows = _load_scifact(args.scifact_max, args.seed)
    health_rows = _load_health(args.health_max, args.seed)

    fever_path = out_dir / "fever_filled.csv"
    liar_path = out_dir / "liar_filled.csv"
    scifact_path = out_dir / "scifact_filled.csv"
    health_path = out_dir / "healthver_filled.csv"

    _write_csv(fever_path, fever_rows)
    _write_csv(liar_path, liar_rows)
    _write_csv(scifact_path, scifact_rows)
    _write_csv(health_path, health_rows)

    total = len(fever_rows) + len(liar_rows) + len(scifact_rows) + len(health_rows)

    print("External benchmark CSVs written.")
    print(f"- FEVER rows: {len(fever_rows)} -> {fever_path}")
    print(f"- LIAR rows: {len(liar_rows)} -> {liar_path}")
    print(f"- SciFact rows: {len(scifact_rows)} -> {scifact_path}")
    print(f"- Health rows: {len(health_rows)} -> {health_path}")
    print(f"- Total rows: {total}")
    if total < 5000:
        print("WARNING: total rows under 5000; increase per-dataset limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
