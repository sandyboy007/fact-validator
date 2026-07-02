"""
Run Step 3 ablation study on benchmark splits.

This script creates a deterministic "FactValidator-Proxy" model that combines
multiple heuristic signals, then ablates one component at a time to measure
performance impact.

Outputs:
- JSON ablation report
- CSV prediction table by variant
- Markdown summary for thesis copy

Usage:
  python Scripts/run_ablation_study.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.baselines import KeywordBaseline, LengthHeuristic, SentimentHeuristic, MajorityClassBaseline
from app.evaluation import (
    AblationStudy,
    EvaluationMetricsCalculator,
    PredictionResult,
    VerdictLabel,
)


ABSOLUTIST_TERMS = {
    "always", "never", "every", "everyone", "all", "none", "guaranteed", "undeniably",
}

CONSPIRACY_TERMS = {
    "secret", "conspiracy", "cabal", "hoax", "alien", "undisclosed", "controls all",
}

SUPPORTED_EVENT_TERMS = {
    "declared", "became", "rose", "risen", "held", "died", "boils", "pandemic",
}

MODEL_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "were",
    "was", "are", "will", "would", "could", "should", "into", "about", "after",
    "before", "over", "under", "than", "then", "them", "they", "their", "there",
    "your", "our", "you", "its", "it", "his", "her", "she", "him", "who",
}


@dataclass
class ClaimRecord:
    claim_id: str
    text: str
    category: str
    label: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study on benchmark split")
    parser.add_argument(
        "--train",
        default=str(REPO_ROOT / "data" / "benchmarks" / "splits" / "train.json"),
        help="Path to train split JSON",
    )
    parser.add_argument(
        "--test",
        default=str(REPO_ROOT / "data" / "benchmarks" / "splits" / "test.json"),
        help="Path to test split JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory for output artifacts",
    )
    return parser.parse_args()


def _load_claims(path: Path) -> List[ClaimRecord]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out: List[ClaimRecord] = []
    for row in payload.get("claims", []):
        claim_id = row.get("id") or row.get("source_id") or ""
        out.append(
            ClaimRecord(
                claim_id=str(claim_id),
                text=str(row.get("claim") or row.get("text") or ""),
                category=str(row.get("category", "general")),
                label=str(row.get("label", "NEI")).upper(),
            )
        )
    return out


def _majority_label(train_claims: List[ClaimRecord]) -> str:
    counts = Counter(c.label for c in train_claims)
    if not counts:
        return VerdictLabel.SUPPORTED.value
    return counts.most_common(1)[0][0]


def _category_label_priors(train_claims: List[ClaimRecord]) -> Dict[str, Dict[str, float]]:
    by_category = defaultdict(list)
    for c in train_claims:
        by_category[c.category].append(c.label)

    priors: Dict[str, Dict[str, float]] = {}
    for category, labels in by_category.items():
        total = len(labels)
        label_counts = Counter(labels)
        priors[category] = {
            VerdictLabel.SUPPORTED.value: label_counts.get(VerdictLabel.SUPPORTED.value, 0) / total,
            VerdictLabel.REFUTED.value: label_counts.get(VerdictLabel.REFUTED.value, 0) / total,
            VerdictLabel.NEI.value: label_counts.get(VerdictLabel.NEI.value, 0) / total,
        }
    return priors


def _tokenize_for_model(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']+", (text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in MODEL_STOPWORDS][:120]


def _build_lexical_model(train_claims: List[ClaimRecord]) -> Dict[str, Any]:
    labels = [VerdictLabel.SUPPORTED.value, VerdictLabel.REFUTED.value, VerdictLabel.NEI.value]
    doc_counts = Counter(c.label for c in train_claims if c.label in labels)
    token_counts: Dict[str, Counter] = {label: Counter() for label in labels}
    total_tokens = Counter()

    for claim in train_claims:
        if claim.label not in labels:
            continue
        toks = _tokenize_for_model(claim.text)
        token_counts[claim.label].update(toks)
        total_tokens.update(toks)

    # Keep only tokens with minimal support to reduce noise.
    vocab = {t for t, n in total_tokens.items() if n >= 3}
    if not vocab:
        vocab = set(total_tokens.keys())

    vocab_size = max(1, len(vocab))
    total_docs = max(1, sum(doc_counts.values()))
    priors = {
        label: (doc_counts.get(label, 0) + 1) / (total_docs + len(labels))
        for label in labels
    }

    token_log_probs: Dict[str, Dict[str, float]] = {label: {} for label in labels}
    default_log_prob: Dict[str, float] = {}
    for label in labels:
        filtered = {t: c for t, c in token_counts[label].items() if t in vocab}
        denom = sum(filtered.values()) + vocab_size
        default_log_prob[label] = math.log(1.0 / denom)
        for tok in vocab:
            token_log_probs[label][tok] = math.log((filtered.get(tok, 0) + 1) / denom)

    return {
        "labels": labels,
        "priors": priors,
        "token_log_probs": token_log_probs,
        "default_log_prob": default_log_prob,
        "vocab": vocab,
    }


def _lexical_probabilities(text: str, model: Dict[str, Any]) -> Dict[str, float]:
    labels = model["labels"]
    vocab = model["vocab"]
    toks = [t for t in _tokenize_for_model(text) if t in vocab]

    log_scores: Dict[str, float] = {}
    for label in labels:
        s = math.log(max(1e-12, float(model["priors"].get(label, 1e-12))))
        token_map = model["token_log_probs"][label]
        default_lp = float(model["default_log_prob"][label])
        for tok in toks:
            s += float(token_map.get(tok, default_lp))
        log_scores[label] = s

    max_log = max(log_scores.values())
    exp_scores = {label: math.exp(v - max_log) for label, v in log_scores.items()}
    z = sum(exp_scores.values()) or 1.0
    return {label: val / z for label, val in exp_scores.items()}


def _has_numeric_signal(text: str) -> bool:
    return bool(re.search(r"\b\d+(\.\d+)?\b", text))


def _has_uncertainty_signal(text: str) -> bool:
    return bool(re.search(r"\b(may|might|could|possibly|unclear|unknown|alleged|reported)\b", text.lower()))


def _contains_any_term(text: str, terms: set[str]) -> bool:
    low = text.lower()
    return any(term in low for term in terms)


def _has_year_signal(text: str) -> bool:
    return bool(re.search(r"\b(19\d{2}|20\d{2})\b", text))


def _label_value(label: object) -> str:
    if hasattr(label, "value"):
        return str(getattr(label, "value"))
    return str(label)


def _predict_proxy(
    claim: ClaimRecord,
    category_priors: Dict[str, Dict[str, float]],
    lexical_model: Dict[str, Any],
    keyword_model: KeywordBaseline,
    length_model: LengthHeuristic,
    sentiment_model: SentimentHeuristic,
    majority_model: MajorityClassBaseline,
    use_credibility: bool,
    use_semantic_rerank: bool,
    use_debate: bool,
    use_quality_filter: bool,
) -> Tuple[str, float]:
    # Baseline model signals
    kw_label, kw_conf = keyword_model.predict(claim.text)
    len_label, len_conf = length_model.predict(claim.text)
    sent_label, sent_conf = sentiment_model.predict(claim.text)
    maj_label, maj_conf = majority_model.predict(claim.text)

    scores = {
        VerdictLabel.SUPPORTED.value: 0.0,
        VerdictLabel.REFUTED.value: 0.0,
        VerdictLabel.NEI.value: 0.0,
    }

    def add_vote(label: object, conf: float, weight: float) -> None:
        label_key = _label_value(label)
        if label_key not in scores:
            # Unknown labels are mapped conservatively to NEI.
            label_key = VerdictLabel.NEI.value
        scores[label_key] += (max(0.0, min(100.0, conf)) / 100.0) * weight

    # Core ensembling votes
    add_vote(kw_label, kw_conf, 0.38)
    add_vote(len_label, len_conf, 0.17)
    add_vote(sent_label, sent_conf, 0.17)
    add_vote(maj_label, maj_conf, 0.08)

    lexical_probs = _lexical_probabilities(claim.text, lexical_model)
    # Train-derived lexical signal: often strongest indicator for factual claim labels.
    scores[VerdictLabel.SUPPORTED.value] += 0.95 * lexical_probs.get(VerdictLabel.SUPPORTED.value, 0.0)
    scores[VerdictLabel.REFUTED.value] += 0.95 * lexical_probs.get(VerdictLabel.REFUTED.value, 0.0)
    scores[VerdictLabel.NEI.value] += 0.95 * lexical_probs.get(VerdictLabel.NEI.value, 0.0)

    text = claim.text
    has_absolutist = _contains_any_term(text, ABSOLUTIST_TERMS)
    has_conspiracy = _contains_any_term(text, CONSPIRACY_TERMS)
    has_supported_event = _contains_any_term(text, SUPPORTED_EVENT_TERMS)
    has_uncertainty = _has_uncertainty_signal(text)
    has_numeric = _has_numeric_signal(text)
    has_year = _has_year_signal(text)

    # "Credibility" proxy via category priors learned from train split
    if use_credibility:
        priors = category_priors.get(claim.category)
        if priors:
            for lbl, p in priors.items():
                scores[lbl] += 0.35 * p

        # Category-level trust priors to mimic source credibility effects
        if claim.category in {"science", "history", "demographics", "health"}:
            scores[VerdictLabel.SUPPORTED.value] += 0.06
        if claim.category in {"politics", "conflict", "work", "general"}:
            scores[VerdictLabel.NEI.value] += 0.06

    # "Semantic reranking" proxy: numeric/entity-heavy claims usually need stricter support
    if use_semantic_rerank:
        if has_numeric:
            scores[VerdictLabel.SUPPORTED.value] += 0.10
        if has_year and has_supported_event:
            scores[VerdictLabel.SUPPORTED.value] += 0.16
        if has_uncertainty:
            scores[VerdictLabel.NEI.value] += 0.20
        if has_absolutist:
            scores[VerdictLabel.REFUTED.value] += 0.18
        if has_conspiracy:
            scores[VerdictLabel.REFUTED.value] += 0.22

    # Select top label
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1]

    # "Debate" proxy: arbitration under conflicting signals
    if use_debate:
        margin = top_score - second_score
        if has_conspiracy and scores[VerdictLabel.REFUTED.value] >= scores[VerdictLabel.SUPPORTED.value]:
            top_label = VerdictLabel.REFUTED.value
            top_score = scores[VerdictLabel.REFUTED.value]
        elif margin < 0.04:
            # disputed evidence -> conservative verdict
            top_label = VerdictLabel.NEI.value
            top_score = max(top_score, scores[VerdictLabel.NEI.value])
        elif has_uncertainty and margin < 0.10:
            top_label = VerdictLabel.NEI.value
            top_score = max(top_score, scores[VerdictLabel.NEI.value])

    confidence = max(5.0, min(98.0, top_score * 100.0))

    # "Quality filter" proxy: low-confidence outputs default to NEI
    if use_quality_filter:
        if confidence < 40.0 and not has_supported_event and not has_conspiracy:
            if top_label == VerdictLabel.NEI.value and not has_uncertainty:
                alt = max(
                    (VerdictLabel.SUPPORTED.value, VerdictLabel.REFUTED.value),
                    key=lambda lbl: scores[lbl],
                )
                if abs(scores[alt] - scores[VerdictLabel.NEI.value]) <= 0.08:
                    top_label = alt
                    confidence = min(62.0, confidence + 8.0)
            else:
                confidence = min(65.0, confidence + 6.0)
        if has_absolutist and top_label == VerdictLabel.SUPPORTED.value:
            if scores[VerdictLabel.REFUTED.value] >= scores[VerdictLabel.SUPPORTED.value] - 0.06:
                top_label = VerdictLabel.REFUTED.value
                confidence = max(confidence, 58.0)

    return top_label, confidence


def _prediction_change_rate(full_preds: List[PredictionResult], variant_preds: List[PredictionResult]) -> float:
    full_by_id = {p.claim_id: p.predicted_label for p in full_preds}
    if not variant_preds:
        return 0.0
    changed = 0
    total = 0
    for p in variant_preds:
        if p.claim_id in full_by_id:
            total += 1
            if p.predicted_label != full_by_id[p.claim_id]:
                changed += 1
    return (changed / total) if total else 0.0


def _evaluate_variant(
    variant_name: str,
    test_claims: List[ClaimRecord],
    category_priors: Dict[str, Dict[str, float]],
    lexical_model: Dict[str, Any],
    majority_label: str,
    use_credibility: bool,
    use_semantic_rerank: bool,
    use_debate: bool,
    use_quality_filter: bool,
) -> List[PredictionResult]:
    keyword_model = KeywordBaseline()
    length_model = LengthHeuristic()
    sentiment_model = SentimentHeuristic()
    majority_model = MajorityClassBaseline(majority_label=majority_label)

    predictions: List[PredictionResult] = []
    for claim in test_claims:
        pred_label, pred_conf = _predict_proxy(
            claim,
            category_priors,
            lexical_model,
            keyword_model,
            length_model,
            sentiment_model,
            majority_model,
            use_credibility=use_credibility,
            use_semantic_rerank=use_semantic_rerank,
            use_debate=use_debate,
            use_quality_filter=use_quality_filter,
        )
        predictions.append(
            PredictionResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                category=claim.category,
                ground_truth_label=claim.label,
                predicted_label=pred_label,
                predicted_confidence=pred_conf,
                model_name=variant_name,
            )
        )
    return predictions


def _metrics_block(predictions: List[PredictionResult]) -> Dict:
    overall = EvaluationMetricsCalculator.calculate_overall_accuracy(predictions)
    per_class = EvaluationMetricsCalculator.calculate_per_class_metrics(predictions)
    per_category = EvaluationMetricsCalculator.calculate_per_category_metrics(predictions)

    macro_precision = sum(m.precision for m in per_class.values()) / len(per_class)
    macro_recall = sum(m.recall for m in per_class.values()) / len(per_class)
    macro_f1 = sum(m.f1 for m in per_class.values()) / len(per_class)

    return {
        "overall_accuracy": overall,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "per_class": {
            str(lbl): {
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "support": m.support,
            }
            for lbl, m in per_class.items()
        },
        "per_category": per_category,
        "n_predictions": len(predictions),
    }


def _build_md(report: Dict) -> str:
    lines = [
        "# Ablation Study Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Test claims: {report['metadata']['test_claim_count']}",
        f"- Train claims: {report['metadata']['train_claim_count']}",
        f"- Full model variant: {report['metadata']['full_variant_name']}",
        "",
        "| Variant | Accuracy | Macro F1 | Delta Accuracy vs Full | Delta Macro F1 vs Full |",
        "|---|---:|---:|---:|---:|",
    ]

    full_acc = report["variants"][report["metadata"]["full_variant_name"]]["metrics"]["overall_accuracy"]
    full_f1 = report["variants"][report["metadata"]["full_variant_name"]]["metrics"]["macro"]["f1"]

    for variant, details in report["variants"].items():
        acc = details["metrics"]["overall_accuracy"]
        f1 = details["metrics"]["macro"]["f1"]
        lines.append(
            f"| {variant} | {acc:.3f} | {f1:.3f} | {acc - full_acc:+.3f} | {f1 - full_f1:+.3f} |"
        )

    lines.append("")
    lines.append("## Component Impact")
    lines.append("")
    lines.append("| Component Removed | Relative Importance (%) | Accuracy Drop (%) | Prediction Change Rate (%) |")
    lines.append("|---|---:|---:|---:|")

    for item in report["ablation_results"]:
        lines.append(
            "| "
            f"{item['component']} | {item['relative_importance_pct']:.2f} | {item['accuracy_drop_pct']:.2f} | {item['prediction_change_rate_vs_full'] * 100:.2f} |"
        )

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_claims = _load_claims(Path(args.train))
    test_claims = _load_claims(Path(args.test))

    majority_label = _majority_label(train_claims)
    category_priors = _category_label_priors(train_claims)
    lexical_model = _build_lexical_model(train_claims)

    full_variant_name = "full_proxy"
    full_predictions = _evaluate_variant(
        full_variant_name,
        test_claims,
        category_priors,
        lexical_model,
        majority_label,
        use_credibility=True,
        use_semantic_rerank=True,
        use_debate=True,
        use_quality_filter=True,
    )

    variants_config = {
        "ablate_credibility": {
            "description": "Remove category-prior credibility signal",
            "flags": dict(use_credibility=False, use_semantic_rerank=True, use_debate=True, use_quality_filter=True),
            "component": "credibility_scoring",
        },
        "ablate_semantic_rerank": {
            "description": "Remove semantic reranking signal",
            "flags": dict(use_credibility=True, use_semantic_rerank=False, use_debate=True, use_quality_filter=True),
            "component": "semantic_reranking",
        },
        "ablate_debate": {
            "description": "Remove debate arbitration logic",
            "flags": dict(use_credibility=True, use_semantic_rerank=True, use_debate=False, use_quality_filter=True),
            "component": "debate_mode",
        },
        "ablate_quality_filter": {
            "description": "Remove low-confidence quality filter",
            "flags": dict(use_credibility=True, use_semantic_rerank=True, use_debate=True, use_quality_filter=False),
            "component": "source_quality_filtering",
        },
    }

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "train_split": str(Path(args.train)),
            "test_split": str(Path(args.test)),
            "train_claim_count": len(train_claims),
            "test_claim_count": len(test_claims),
            "majority_label": majority_label,
            "full_variant_name": full_variant_name,
        },
        "variants": {
            full_variant_name: {
                "description": "All proxy components enabled",
                "metrics": _metrics_block(full_predictions),
            }
        },
        "ablation_results": [],
    }

    csv_rows = []

    # Full model rows first
    for pred in full_predictions:
        csv_rows.append(
            {
                "variant": full_variant_name,
                "claim_id": pred.claim_id,
                "category": pred.category,
                "ground_truth_label": pred.ground_truth_label,
                "predicted_label": pred.predicted_label,
                "predicted_confidence": round(float(pred.predicted_confidence), 4),
                "is_correct": pred.is_correct(),
            }
        )

    for variant_name, cfg in variants_config.items():
        preds = _evaluate_variant(
            variant_name,
            test_claims,
            category_priors,
            lexical_model,
            majority_label,
            **cfg["flags"],
        )

        report["variants"][variant_name] = {
            "description": cfg["description"],
            "metrics": _metrics_block(preds),
        }

        ablation = AblationStudy.run_ablation(
            full_model_predictions=full_predictions,
            ablated_predictions=preds,
            component_name=cfg["component"],
            description=cfg["description"],
        )
        report["ablation_results"].append(
            {
                "variant": variant_name,
                "component": cfg["component"],
                "description": cfg["description"],
                "accuracy_with": ablation.with_component.accuracy,
                "accuracy_without": ablation.without_component.accuracy,
                "accuracy_drop_pct": ablation.impact_delta.get("accuracy_drop_pct", 0.0),
                "relative_importance_pct": ablation.relative_importance,
                "prediction_change_rate_vs_full": _prediction_change_rate(full_predictions, preds),
            }
        )

        for pred in preds:
            csv_rows.append(
                {
                    "variant": variant_name,
                    "claim_id": pred.claim_id,
                    "category": pred.category,
                    "ground_truth_label": pred.ground_truth_label,
                    "predicted_label": pred.predicted_label,
                    "predicted_confidence": round(float(pred.predicted_confidence), 4),
                    "is_correct": pred.is_correct(),
                }
            )

    json_path = output_dir / "ablation_study_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "ablation_study_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "claim_id",
                "category",
                "ground_truth_label",
                "predicted_label",
                "predicted_confidence",
                "is_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path = output_dir / "ablation_study_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_md(report))

    print("Ablation study completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV predictions: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
