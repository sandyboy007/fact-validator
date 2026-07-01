from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "services" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.baselines import KeywordBaseline, LengthHeuristic


LABEL_TO_ORDINAL = {"REFUTED": 0, "NEI": 1, "SUPPORTED": 2}

SHORTCUT_KEYWORDS = {
    "confirmed",
    "verified",
    "supported",
    "proven",
    "false",
    "debunked",
    "refuted",
    "unclear",
    "uncertain",
    "insufficient",
    "evidence",
    "study",
    "research",
    "data",
}


@dataclass
class ClaimRow:
    claim_id: str
    label: str
    original: str
    perturbed: str
    category: str = "general"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shortcut sensitivity analysis for fact checking claims")
    parser.add_argument("--input-csv", required=True, help="CSV with original and perturbed claims")
    parser.add_argument("--output-md", default=None, help="Optional markdown output path")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000", help="Fact Validator API base URL")
    parser.add_argument("--original-column", default="claim_original", help="Original claim text column")
    parser.add_argument("--perturbed-column", default="claim_perturbed", help="Perturbed claim text column")
    parser.add_argument("--label-column", default="label", help="Ground-truth label column")
    parser.add_argument("--id-column", default="id", help="Claim id column")
    parser.add_argument("--category-column", default="category", help="Category column")
    parser.add_argument("--sentiment-mode", choices=["compound", "absolute"], default="compound")
    return parser.parse_args()


def load_rows(path: Path, args: argparse.Namespace) -> list[ClaimRow]:
    rows: list[ClaimRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            original = (row.get(args.original_column) or row.get("claim") or "").strip()
            perturbed = (row.get(args.perturbed_column) or "").strip()
            label = (row.get(args.label_column) or "").strip().upper()
            claim_id = (row.get(args.id_column) or f"row-{index}").strip()
            category = (row.get(args.category_column) or "general").strip() or "general"

            if not original:
                raise ValueError(f"Missing original claim text on row {index}")
            if label not in LABEL_TO_ORDINAL:
                raise ValueError(f"Invalid label '{label}' on row {index}")

            rows.append(ClaimRow(claim_id=claim_id, label=label, original=original, perturbed=perturbed, category=category))
    return rows


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def ensure_vader_lexicon() -> None:
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception:
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            pass


def keyword_frequency(text: str) -> int:
    tokens = tokenize(text)
    return sum(1 for token in tokens if token in SHORTCUT_KEYWORDS)


def sentiment_score(analyzer: SentimentIntensityAnalyzer, text: str, mode: str) -> float:
    compound = analyzer.polarity_scores(text).get("compound", 0.0)
    return abs(compound) if mode == "absolute" else compound


async def score_claim(client: httpx.AsyncClient, api_base_url: str, text: str) -> str:
    response = await client.post(
        f"{api_base_url.rstrip('/')}/analyze",
        json={"text": text, "verifier": "baseline", "max_claims": 1, "max_evidence_per_claim": 5, "mode": "snapshot"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    claims = payload.get("claims") or []
    if not claims:
        return "NEI"
    verdict = str(claims[0].get("verdict") or "NEI").upper()
    return verdict if verdict in LABEL_TO_ORDINAL else "NEI"


async def score_rows(rows: list[ClaimRow], api_base_url: str) -> list[dict[str, Any]]:
    keyword_baseline = KeywordBaseline()
    length_baseline = LengthHeuristic()
    analyzer = SentimentIntensityAnalyzer()

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for row in rows:
            orig_pred = await score_claim(client, api_base_url, row.original)
            pert_pred = await score_claim(client, api_base_url, row.perturbed or row.original)

            length_orig_pred, _ = length_baseline.predict(row.original)
            length_pert_pred, _ = length_baseline.predict(row.perturbed or row.original)
            keyword_orig_pred, _ = keyword_baseline.predict(row.original)
            keyword_pert_pred, _ = keyword_baseline.predict(row.perturbed or row.original)

            row_data = {
                "id": row.claim_id,
                "label": row.label,
                "original": row.original,
                "perturbed": row.perturbed or row.original,
                "token_count": len(tokenize(row.original)),
                "char_count": len(row.original),
                "keyword_frequency": keyword_frequency(row.original),
                "vader_compound": sentiment_score(analyzer, row.original, "compound"),
                "orig_full_pred": orig_pred,
                "pert_full_pred": pert_pred,
                "orig_length_pred": length_orig_pred,
                "pert_length_pred": length_pert_pred,
                "orig_keyword_pred": keyword_orig_pred,
                "pert_keyword_pred": keyword_pert_pred,
            }
            results.append(row_data)
    return results


def accuracy(records: Iterable[dict[str, Any]], pred_key: str) -> float:
    items = list(records)
    if not items:
        return float("nan")
    correct = sum(1 for item in items if str(item[pred_key]).upper() == str(item["label"]).upper())
    return correct / len(items)


def corr_report(values: list[float], labels: list[int], method: str) -> dict[str, float]:
    if len(values) < 2:
        return {"r": float("nan"), "p": float("nan")}
    if method == "pearson":
        r, p = pearsonr(values, labels)
    else:
        r, p = spearmanr(values, labels)
    return {"r": float(r), "p": float(p)}


def build_output(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [LABEL_TO_ORDINAL[str(item["label"]).upper()] for item in results]

    perturbation = {
        "full_system": {
            "original_accuracy": accuracy(results, "orig_full_pred"),
            "perturbed_accuracy": accuracy(results, "pert_full_pred"),
        },
        "length_heuristic": {
            "original_accuracy": accuracy(results, "orig_length_pred"),
            "perturbed_accuracy": accuracy(results, "pert_length_pred"),
        },
        "keyword_heuristic": {
            "original_accuracy": accuracy(results, "orig_keyword_pred"),
            "perturbed_accuracy": accuracy(results, "pert_keyword_pred"),
        },
    }

    correlation = {
        "char_count": corr_report([float(item["char_count"]) for item in results], labels, "spearman"),
        "token_count": corr_report([float(item["token_count"]) for item in results], labels, "spearman"),
        "keyword_frequency": corr_report([float(item["keyword_frequency"]) for item in results], labels, "spearman"),
        "vader_compound": corr_report([float(item["vader_compound"]) for item in results], labels, "spearman"),
    }

    sorted_by_length = sorted(results, key=lambda item: item["token_count"])
    chunks = np.array_split(sorted_by_length, 3)
    tertile_rows = []
    tertile_names = ["Short", "Medium", "Long"]
    for name, chunk in zip(tertile_names, chunks):
        chunk_list = list(chunk)
        tertile_rows.append(
            {
                "tertile": name,
                "n": len(chunk_list),
                "full_system_accuracy": accuracy(chunk_list, "orig_full_pred"),
                "length_heuristic_accuracy": accuracy(chunk_list, "orig_length_pred"),
                "avg_token_count": float(np.mean([item["token_count"] for item in chunk_list])) if chunk_list else float("nan"),
            }
        )

    return {
        "perturbation_matrix": perturbation,
        "feature_label_correlation": correlation,
        "length_tertile_analysis": tertile_rows,
    }


def format_pct(value: float) -> str:
    return "nan" if value != value else f"{value:.3f}"


def render_markdown(output: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Shortcut Analysis Results")
    lines.append("")
    lines.append("## Table III. Perturbation Matrix")
    lines.append("")
    lines.append("| System | Original Accuracy | Perturbed Accuracy |")
    lines.append("|---|---:|---:|")
    for system_name, item in output["perturbation_matrix"].items():
        lines.append(
            f"| {system_name.replace('_', ' ').title()} | {format_pct(item['original_accuracy'])} | {format_pct(item['perturbed_accuracy'])} |"
        )
    lines.append("")
    lines.append("## Table IV. Feature-Label Correlation")
    lines.append("")
    lines.append("| Feature | Spearman r | p-value |")
    lines.append("|---|---:|---:|")
    for feature_name, item in output["feature_label_correlation"].items():
        lines.append(f"| {feature_name.replace('_', ' ').title()} | {format_pct(item['r'])} | {format_pct(item['p'])} |")
    lines.append("")
    lines.append("## Table V. Length Stratified Error Analysis")
    lines.append("")
    lines.append("| Tertile | n | Full System Accuracy | Length Heuristic Accuracy | Avg. Token Count |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in output["length_tertile_analysis"]:
        lines.append(
            f"| {row['tertile']} | {row['n']} | {format_pct(row['full_system_accuracy'])} | {format_pct(row['length_heuristic_accuracy'])} | {row['avg_token_count']:.1f} |"
        )
    lines.append("")
    lines.append("Note: label correlations use an ordinal encoding of REFUTED=0, NEI=1, SUPPORTED=2 for exploratory analysis.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    ensure_vader_lexicon()
    rows = load_rows(input_path, args)
    output = asyncio.run(score_rows(rows, args.api_base_url))
    report = build_output(output)
    markdown = render_markdown(report)

    print(markdown)

    if args.output_md:
        Path(args.output_md).write_text(markdown, encoding="utf-8")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())