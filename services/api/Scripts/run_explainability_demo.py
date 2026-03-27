"""
Run Step 6 explainability demo generation.

This script builds thesis-ready explainability artifacts from existing benchmark
results by producing:
- case studies with feature-level scoring walkthroughs
- transcript-style debate traces (prover/skeptic/judge)
- representative domain credibility scoring examples

Outputs:
- JSON report
- CSV case-study table
- Markdown summary

Usage:
  python Scripts/run_explainability_demo.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.credibility import score_domain_rubric


ABSOLUTIST_TERMS = {
    "always", "never", "every", "everyone", "all", "none", "guaranteed", "undeniably",
}

CONSPIRACY_TERMS = {
    "secret", "conspiracy", "cabal", "hoax", "alien", "undisclosed", "controls all",
}

SUPPORTED_EVENT_TERMS = {
    "declared", "became", "rose", "risen", "held", "died", "boils", "pandemic",
}

UNCERTAINTY_TERMS = {
    "may", "might", "could", "possibly", "unclear", "unknown", "alleged", "reported",
}


@dataclass
class ClaimRecord:
    claim_id: str
    claim_text: str
    category: str
    label: str


@dataclass
class Prediction:
    claim_id: str
    predicted_label: str
    predicted_confidence: float
    is_correct: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 6 explainability demo")
    parser.add_argument(
        "--test-split",
        default=str(REPO_ROOT / "data" / "benchmarks" / "splits" / "test.json"),
        help="Path to test split JSON",
    )
    parser.add_argument(
        "--baseline-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "baseline_comparison_report.json"),
        help="Path to baseline report JSON",
    )
    parser.add_argument(
        "--baseline-csv",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "baseline_comparison_predictions.csv"),
        help="Path to baseline predictions CSV",
    )
    parser.add_argument(
        "--ablation-csv",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "ablation_study_predictions.csv"),
        help="Path to ablation predictions CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--full-variant",
        default="full_proxy",
        help="Ablation variant representing full system",
    )
    parser.add_argument(
        "--no-debate-variant",
        default="ablate_debate",
        help="Ablation variant representing no debate",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=7,
        help="Maximum number of case studies to include",
    )
    return parser.parse_args()


def _load_test_split(path: Path) -> Dict[str, ClaimRecord]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out: Dict[str, ClaimRecord] = {}
    for row in payload.get("claims", []):
        cid = str(row.get("id", ""))
        out[cid] = ClaimRecord(
            claim_id=cid,
            claim_text=str(row.get("claim") or row.get("text") or ""),
            category=str(row.get("category", "general")),
            label=str(row.get("label", "NEI")),
        )
    return out


def _load_baseline_name(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("results", {})
    if not results:
        return "keyword"

    ranked = sorted(
        results.items(),
        key=lambda kv: float(kv[1].get("overall_accuracy", 0.0)),
        reverse=True,
    )
    return ranked[0][0]


def _load_baseline_predictions(path: Path, baseline_name: str) -> Dict[str, Prediction]:
    out: Dict[str, Prediction] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("baseline", "")) != baseline_name:
                continue
            cid = str(row.get("claim_id", ""))
            out[cid] = Prediction(
                claim_id=cid,
                predicted_label=str(row.get("predicted_label", "NEI")),
                predicted_confidence=float(row.get("predicted_confidence", 0.0) or 0.0),
                is_correct=str(row.get("is_correct", "false")).strip().lower() in {"true", "1", "yes"},
            )
    return out


def _load_variant_predictions(path: Path, variant_name: str) -> Dict[str, Prediction]:
    out: Dict[str, Prediction] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("variant", "")) != variant_name:
                continue
            cid = str(row.get("claim_id", ""))
            out[cid] = Prediction(
                claim_id=cid,
                predicted_label=str(row.get("predicted_label", "NEI")),
                predicted_confidence=float(row.get("predicted_confidence", 0.0) or 0.0),
                is_correct=str(row.get("is_correct", "false")).strip().lower() in {"true", "1", "yes"},
            )
    return out


def _extract_feature_signals(text: str) -> Dict[str, bool]:
    low = text.lower()
    has_numeric = bool(re.search(r"\b\d+(\.\d+)?\b", low))
    has_year = bool(re.search(r"\b(19\d{2}|20\d{2})\b", low))
    has_uncertainty = any(term in low for term in UNCERTAINTY_TERMS)
    has_absolutist = any(term in low for term in ABSOLUTIST_TERMS)
    has_conspiracy = any(term in low for term in CONSPIRACY_TERMS)
    has_supported_event = any(term in low for term in SUPPORTED_EVENT_TERMS)
    return {
        "has_numeric": has_numeric,
        "has_year": has_year,
        "has_uncertainty": has_uncertainty,
        "has_absolutist": has_absolutist,
        "has_conspiracy": has_conspiracy,
        "has_supported_event": has_supported_event,
    }


def _scoring_logic_explanation(signals: Dict[str, bool], category: str) -> List[str]:
    lines: List[str] = []

    if category in {"science", "history", "demographics", "health"}:
        lines.append("Category prior increases support confidence for evidence-rich domains.")
    elif category in {"politics", "conflict", "work", "general"}:
        lines.append("Category prior increases uncertainty handling (NEI tendency) for ambiguous domains.")

    if signals["has_numeric"]:
        lines.append("Numeric signal detected: evidence consistency weighting is increased.")
    if signals["has_year"] and signals["has_supported_event"]:
        lines.append("Temporal event signal detected: support path gets additional weight.")
    if signals["has_uncertainty"]:
        lines.append("Uncertainty language detected: conservative NEI arbitration is preferred.")
    if signals["has_absolutist"]:
        lines.append("Absolutist language detected: refutation pressure increases.")
    if signals["has_conspiracy"]:
        lines.append("Conspiracy cue detected: refutation pressure strongly increases.")

    if not lines:
        lines.append("No strong lexical cues detected: majority and category priors dominate verdict.")

    return lines


def _debate_trace(
    claim: ClaimRecord,
    full_pred: Prediction,
    no_debate_pred: Prediction,
    baseline_pred: Prediction,
    signals: Dict[str, bool],
) -> Dict[str, str]:
    prover_points = []
    skeptic_points = []

    if signals["has_numeric"]:
        prover_points.append("The claim contains quantifiable elements that can be checked against evidence.")
    if signals["has_supported_event"]:
        prover_points.append("The phrasing implies an event-like fact pattern suited to evidence grounding.")
    if signals["has_uncertainty"]:
        skeptic_points.append("Hedged language weakens direct verifiability and may warrant NEI.")
    if signals["has_absolutist"]:
        skeptic_points.append("Absolute wording raises risk of overclaiming and potential refutation.")
    if signals["has_conspiracy"]:
        skeptic_points.append("Conspiracy-style language correlates with lower evidentiary reliability.")

    if not prover_points:
        prover_points.append("Category prior and lexical evidence still support a decisive verdict.")
    if not skeptic_points:
        skeptic_points.append("Counter-signals are limited, but alternate verdicts remain plausible.")

    judge = (
        f"With debate arbitration, verdict is {full_pred.predicted_label} ({full_pred.predicted_confidence:.1f} confidence), "
        f"while no-debate predicts {no_debate_pred.predicted_label}. "
        f"Best baseline predicts {baseline_pred.predicted_label}."
    )

    return {
        "prover": " ".join(prover_points),
        "skeptic": " ".join(skeptic_points),
        "judge": judge,
    }


def _domain_scoring_examples() -> List[Dict[str, object]]:
    examples = [
        "bbc.com",
        "reuters.com",
        "who.int",
        "wikipedia.org",
        "example-blog-news.net",
    ]
    rows = []
    for domain in examples:
        scored = score_domain_rubric(domain)
        score_value = getattr(scored, "score", scored)
        rows.append(
            {
                "domain": domain,
                "score": int(score_value),
            }
        )
    return rows


def _build_markdown(report: Dict) -> str:
    lines = [
        "# Explainability Demo Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Full variant: {report['metadata']['full_variant']}",
        f"- Best baseline: {report['metadata']['best_baseline']}",
        f"- Case studies: {report['metadata']['case_count']}",
        "",
        "## Domain Credibility Examples",
        "",
        "| Domain | Score |",
        "|---|---:|",
    ]

    for row in report["domain_scoring_examples"]:
        lines.append(f"| {row['domain']} | {row['score']} |")

    lines.extend([
        "",
        "## Case Studies",
        "",
    ])

    for idx, case in enumerate(report["case_studies"], start=1):
        lines.append(f"### Case {idx}: {case['claim_id']}")
        lines.append("")
        lines.append(f"- Claim: {case['claim_text']}")
        lines.append(f"- Ground truth: {case['ground_truth_label']}")
        lines.append(f"- Full system: {case['predictions']['full']['label']} ({case['predictions']['full']['confidence']:.1f})")
        lines.append(f"- No-debate: {case['predictions']['no_debate']['label']} ({case['predictions']['no_debate']['confidence']:.1f})")
        lines.append(f"- Baseline ({report['metadata']['best_baseline']}): {case['predictions']['baseline']['label']} ({case['predictions']['baseline']['confidence']:.1f})")
        lines.append("- Scoring logic:")
        for item in case["scoring_logic"]:
            lines.append(f"  - {item}")
        lines.append("- Debate trace:")
        lines.append(f"  - Prover: {case['debate_trace']['prover']}")
        lines.append(f"  - Skeptic: {case['debate_trace']['skeptic']}")
        lines.append(f"  - Judge: {case['debate_trace']['judge']}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    test_claims = _load_test_split(Path(args.test_split))
    best_baseline = _load_baseline_name(Path(args.baseline_report))
    baseline_preds = _load_baseline_predictions(Path(args.baseline_csv), baseline_name=best_baseline)
    full_preds = _load_variant_predictions(Path(args.ablation_csv), variant_name=args.full_variant)
    no_debate_preds = _load_variant_predictions(Path(args.ablation_csv), variant_name=args.no_debate_variant)

    case_ids = [cid for cid in test_claims.keys() if cid in full_preds and cid in no_debate_preds and cid in baseline_preds]

    def rank_key(cid: str) -> tuple:
        full_ok = full_preds[cid].is_correct
        baseline_ok = baseline_preds[cid].is_correct
        debate_changed = full_preds[cid].predicted_label != no_debate_preds[cid].predicted_label
        # Priority: full fixes baseline -> debate changed -> full errors -> others.
        return (
            0 if (full_ok and not baseline_ok) else 1,
            0 if debate_changed else 1,
            0 if (not full_ok) else 1,
            cid,
        )

    ranked_ids = sorted(case_ids, key=rank_key)
    selected_ids = ranked_ids[: max(1, args.max_cases)]

    case_studies = []
    for cid in selected_ids:
        claim = test_claims[cid]
        full_pred = full_preds[cid]
        no_debate_pred = no_debate_preds[cid]
        baseline_pred = baseline_preds[cid]

        signals = _extract_feature_signals(claim.claim_text)
        scoring_logic = _scoring_logic_explanation(signals, claim.category)
        debate_trace = _debate_trace(claim, full_pred, no_debate_pred, baseline_pred, signals)

        case_studies.append(
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "category": claim.category,
                "ground_truth_label": claim.label,
                "predictions": {
                    "full": {
                        "label": full_pred.predicted_label,
                        "confidence": full_pred.predicted_confidence,
                        "is_correct": full_pred.is_correct,
                    },
                    "no_debate": {
                        "label": no_debate_pred.predicted_label,
                        "confidence": no_debate_pred.predicted_confidence,
                        "is_correct": no_debate_pred.is_correct,
                    },
                    "baseline": {
                        "label": baseline_pred.predicted_label,
                        "confidence": baseline_pred.predicted_confidence,
                        "is_correct": baseline_pred.is_correct,
                    },
                },
                "signals": signals,
                "scoring_logic": scoring_logic,
                "debate_trace": debate_trace,
            }
        )

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "full_variant": args.full_variant,
            "no_debate_variant": args.no_debate_variant,
            "best_baseline": best_baseline,
            "case_count": len(case_studies),
            "note": "Debate traces are structured explainability narratives synthesized from model outputs and lexical signals.",
        },
        "domain_scoring_examples": _domain_scoring_examples(),
        "case_studies": case_studies,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "explainability_demo_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "explainability_demo_cases.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "claim_id",
                "category",
                "ground_truth_label",
                "full_label",
                "full_confidence",
                "no_debate_label",
                "baseline_label",
                "full_correct",
                "baseline_correct",
                "debate_changed_prediction",
            ],
        )
        writer.writeheader()
        for case in case_studies:
            writer.writerow(
                {
                    "claim_id": case["claim_id"],
                    "category": case["category"],
                    "ground_truth_label": case["ground_truth_label"],
                    "full_label": case["predictions"]["full"]["label"],
                    "full_confidence": round(float(case["predictions"]["full"]["confidence"]), 4),
                    "no_debate_label": case["predictions"]["no_debate"]["label"],
                    "baseline_label": case["predictions"]["baseline"]["label"],
                    "full_correct": case["predictions"]["full"]["is_correct"],
                    "baseline_correct": case["predictions"]["baseline"]["is_correct"],
                    "debate_changed_prediction": case["predictions"]["full"]["label"] != case["predictions"]["no_debate"]["label"],
                }
            )

    md_path = output_dir / "explainability_demo_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print("Explainability demo generation completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV cases: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())