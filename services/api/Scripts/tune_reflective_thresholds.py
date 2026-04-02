"""
Run reflective-threshold tuning over the annotated benchmark CSV.

This script does a deterministic grid search on reflective abstention thresholds
using synthetic, label-aware evidence profiles derived from benchmark rows. It is
intended for fast local tuning before running expensive live retrieval benchmarks.

It also supports a live-sampled mode that calls the in-process /analyze pipeline
on a stratified sample of benchmark claims and then tunes thresholds against the
retrieved evidence.

Outputs:
- JSON report with scored candidates and a recommended threshold set.
- Markdown summary for quick review.

Usage:
    python Scripts/tune_reflective_thresholds.py
    python Scripts/tune_reflective_thresholds.py --mode live-sampled --live-sample-size 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.analysis_features import decompose_claim, determine_verdict, enrich_evidence
from app.reflective import DEFAULT_REFLECTIVE_THRESHOLDS, reflective_analysis


VALID_LABELS = {"SUPPORTED", "REFUTED", "NEI"}


@dataclass
class ClaimRow:
    claim_id: str
    claim: str
    label: str
    category: str
    difficulty: str
    source_url: str


@dataclass
class PreparedClaim:
    row: ClaimRow
    claim_profile: Dict[str, Any]
    enriched_evidence: List[Dict[str, Any]]
    baseline_verdict: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune reflective abstention thresholds on benchmark CSV")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "live-sampled"],
        default="synthetic",
        help="Tuning mode: synthetic (fast heuristic evidence) or live-sampled (real retrieval on sampled claims)",
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "data" / "benchmarks" / "claim_annotation_template_240.csv"),
        help="Input benchmark CSV path",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "reflective_threshold_tuning_report.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to include in report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling",
    )
    parser.add_argument(
        "--live-sample-size",
        type=int,
        default=30,
        help="Number of claims to sample in live-sampled mode",
    )
    parser.add_argument(
        "--live-max-evidence",
        type=int,
        default=5,
        help="max_evidence_per_claim used when calling /analyze in live-sampled mode",
    )
    return parser.parse_args()


def _normalize_label(value: str) -> str:
    label = (value or "").strip().upper()
    if label in {"INSUFFICIENT_EVIDENCE", "INSUFFICIENT EVIDENCE", "MIXED", "MIXED / DISPUTED"}:
        return "NEI"
    return label


def _difficulty_bucket(value: str) -> str:
    low = (value or "medium").strip().lower()
    if low in {"easy", "medium", "hard"}:
        return low
    return "medium"


def _domain_from_url(url: str) -> str:
    if not url:
        return "example.org"
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host or "example.org"


def _load_claims(path: Path) -> List[ClaimRow]:
    rows: List[ClaimRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim_id = (row.get("id") or "").strip()
            claim = (row.get("claim") or "").strip()
            label = _normalize_label(row.get("label") or "")
            if not claim_id or not claim or label not in VALID_LABELS:
                continue

            rows.append(
                ClaimRow(
                    claim_id=claim_id,
                    claim=claim,
                    label=label,
                    category=(row.get("category") or "general").strip().lower() or "general",
                    difficulty=_difficulty_bucket(row.get("difficulty") or "medium"),
                    source_url=(row.get("source_url") or "").strip(),
                )
            )
    return rows


def _quality_profile(label: str, difficulty: str) -> Dict[str, Tuple[float, float, float]]:
    # (domain_score, overlap, semantic_score) for primary and secondary evidence.
    if label == "NEI":
        if difficulty == "easy":
            return {
                "primary": (62.0, 2.0, 0.28),
                "secondary": (58.0, 1.0, 0.18),
            }
        if difficulty == "hard":
            return {
                "primary": (56.0, 1.0, 0.22),
                "secondary": (52.0, 1.0, 0.15),
            }
        return {
            "primary": (59.0, 2.0, 0.24),
            "secondary": (55.0, 1.0, 0.16),
        }

    if difficulty == "easy":
        return {
            "primary": (88.0, 8.0, 0.82),
            "secondary": (84.0, 7.0, 0.74),
            "conflict": (70.0, 4.0, 0.45),
        }
    if difficulty == "hard":
        return {
            "primary": (72.0, 5.0, 0.55),
            "secondary": (67.0, 4.0, 0.47),
            "conflict": (68.0, 4.0, 0.48),
        }
    return {
        "primary": (78.0, 6.0, 0.64),
        "secondary": (73.0, 5.0, 0.56),
        "conflict": (66.0, 4.0, 0.42),
    }


def _raw_evidence_templates(row: ClaimRow) -> List[Dict[str, Any]]:
    claim = row.claim
    category = row.category
    label = row.label
    profile = _quality_profile(label, row.difficulty)

    trusted_domain = _domain_from_url(row.source_url)
    vague_domain = "analysis.example.org"

    def make_item(
        *,
        title: str,
        snippet: str,
        domain: str,
        domain_score: float,
        overlap: float,
        semantic_score: float,
        idx: int,
    ) -> Dict[str, Any]:
        return {
            "url": f"https://{domain}/doc/{row.claim_id}/{idx}",
            "title": title,
            "snippet": snippet,
            "domain": domain,
            "base_domain": domain,
            "domain_score": int(round(domain_score)),
            "overlap": int(round(overlap)),
            "semantic_score": float(semantic_score),
        }

    if label == "SUPPORTED":
        items = [
            make_item(
                title="Institutional monitoring report",
                snippet=f"Official reports provide direct evidence that supports this statement: {claim}",
                domain=trusted_domain,
                domain_score=profile["primary"][0],
                overlap=profile["primary"][1],
                semantic_score=profile["primary"][2],
                idx=1,
            ),
            make_item(
                title="Independent verification summary",
                snippet="Independent analysts confirm the same measurable trend described in the claim.",
                domain=trusted_domain,
                domain_score=profile["secondary"][0],
                overlap=profile["secondary"][1],
                semantic_score=profile["secondary"][2],
                idx=2,
            ),
        ]
        if row.difficulty == "hard":
            items.append(
                make_item(
                    title="Disputed interpretation",
                    snippet="A commentary source argues this claim is false or overstated in some contexts.",
                    domain=trusted_domain,
                    domain_score=profile["conflict"][0],
                    overlap=profile["conflict"][1],
                    semantic_score=profile["conflict"][2],
                    idx=3,
                )
            )
        return items

    if label == "REFUTED":
        items = [
            make_item(
                title="Fact-check ruling",
                snippet=f"This claim is false. Credible evidence directly contradicts this statement: {claim}",
                domain=trusted_domain,
                domain_score=profile["primary"][0],
                overlap=profile["primary"][1],
                semantic_score=profile["primary"][2],
                idx=1,
            ),
            make_item(
                title="Methodology review",
                snippet="Multiple measurements reject the statement and do not support the claimed conclusion.",
                domain=trusted_domain,
                domain_score=profile["secondary"][0],
                overlap=profile["secondary"][1],
                semantic_score=profile["secondary"][2],
                idx=2,
            ),
        ]
        if row.difficulty == "hard":
            items.append(
                make_item(
                    title="Ambiguous support article",
                    snippet="One opinion article supports portions of this claim without rigorous evidence.",
                    domain=trusted_domain,
                    domain_score=profile["conflict"][0],
                    overlap=profile["conflict"][1],
                    semantic_score=profile["conflict"][2],
                    idx=3,
                )
            )
        return items

    # NEI
    return [
        make_item(
            title="General commentary",
            snippet=f"General discussion about {category} exists, but this source does not verify this specific statement.",
            domain=vague_domain,
            domain_score=profile["primary"][0],
            overlap=profile["primary"][1],
            semantic_score=profile["primary"][2],
            idx=1,
        ),
        make_item(
            title="Opinion roundup",
            snippet="The article contains broad claims and lacks direct data, primary records, or clear corroborating measurements.",
            domain=vague_domain,
            domain_score=profile["secondary"][0],
            overlap=profile["secondary"][1],
            semantic_score=profile["secondary"][2],
            idx=2,
        ),
    ]


def _stratified_sample(rows: List[ClaimRow], sample_size: int, seed: int) -> List[ClaimRow]:
    if not rows:
        return []

    sample_size = max(1, min(int(sample_size), len(rows)))
    if sample_size >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    labels = sorted({r.label for r in rows})
    by_label: Dict[str, List[ClaimRow]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)
    for label in labels:
        rng.shuffle(by_label[label])

    total = len(rows)
    base_targets: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for label in labels:
        exact = sample_size * (len(by_label[label]) / total)
        base = int(exact)
        base_targets[label] = base
        remainders.append((exact - base, label))

    assigned = sum(base_targets.values())
    needed = sample_size - assigned
    for _, label in sorted(remainders, reverse=True):
        if needed <= 0:
            break
        base_targets[label] += 1
        needed -= 1

    if sample_size >= len(labels):
        for label in labels:
            if base_targets[label] == 0 and by_label[label]:
                donor = max(labels, key=lambda x: base_targets[x])
                if base_targets[donor] > 1:
                    base_targets[donor] -= 1
                    base_targets[label] = 1

    selected: List[ClaimRow] = []
    selected_ids = set()
    for label in labels:
        take = min(base_targets[label], len(by_label[label]))
        for row in by_label[label][:take]:
            selected.append(row)
            selected_ids.add(row.claim_id)

    if len(selected) < sample_size:
        pool = [r for r in rows if r.claim_id not in selected_ids]
        rng.shuffle(pool)
        selected.extend(pool[: sample_size - len(selected)])

    rng.shuffle(selected)
    return selected[:sample_size]


def _prepare_claims_synthetic(rows: List[ClaimRow]) -> List[PreparedClaim]:
    prepared: List[PreparedClaim] = []
    for row in rows:
        profile = decompose_claim(row.claim)
        enriched = [
            enrich_evidence(row.claim, profile, ev)
            for ev in _raw_evidence_templates(row)
        ]
        enriched.sort(
            key=lambda e: (float(e.get("quality_score") or 0.0), int(e.get("domain_score") or 0), int(e.get("overlap") or 0)),
            reverse=True,
        )
        baseline = determine_verdict(row.claim, profile, enriched).get("legacy_verdict", "NEI")
        if baseline not in VALID_LABELS:
            baseline = "NEI"
        prepared.append(
            PreparedClaim(
                row=row,
                claim_profile=profile,
                enriched_evidence=enriched,
                baseline_verdict=baseline,
            )
        )
    return prepared


def _prepare_claims_live(
    rows: List[ClaimRow],
    sample_size: int,
    seed: int,
    max_evidence_per_claim: int,
) -> Tuple[List[PreparedClaim], Dict[str, Any]]:
    from fastapi.testclient import TestClient
    from app.main import app

    sampled_rows = _stratified_sample(rows, sample_size=sample_size, seed=seed)
    client = TestClient(app)

    prepared: List[PreparedClaim] = []
    cached_claims: Dict[str, Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = {}
    failed: List[str] = []
    serpapi_enabled: bool | None = None

    for idx, row in enumerate(sampled_rows, start=1):
        key = row.claim.strip().lower()
        cached = cached_claims.get(key)

        if cached is None:
            response = client.post(
                "/analyze",
                json={
                    "text": row.claim,
                    "mode": "live",
                    "verifier": "baseline",
                    "max_claims": 1,
                    "max_evidence_per_claim": int(max_evidence_per_claim),
                    "enable_reflective_abstention": False,
                    "enable_faithful_correction": False,
                },
            )
            if response.status_code != 200:
                failed.append(f"{row.claim_id}: status={response.status_code}")
                continue

            payload = response.json()
            meta = payload.get("metadata") or {}
            if serpapi_enabled is None:
                serpapi_enabled = bool(meta.get("serpapi_enabled", False))

            claims = payload.get("claims") or []
            if not claims:
                failed.append(f"{row.claim_id}: empty claims")
                continue

            claim_out = claims[0]
            baseline = str(claim_out.get("verdict") or "NEI").upper()
            if baseline not in VALID_LABELS:
                baseline = "NEI"

            profile = claim_out.get("claim_profile")
            if not isinstance(profile, dict):
                profile = decompose_claim(row.claim)

            evidence = list(claim_out.get("evidence") or [])
            cached_claims[key] = (baseline, profile, evidence)

        baseline, profile, evidence = cached_claims[key]
        prepared.append(
            PreparedClaim(
                row=row,
                claim_profile=profile,
                enriched_evidence=evidence,
                baseline_verdict=baseline,
            )
        )

        if idx % 10 == 0:
            print(f"Live sampling progress: {idx}/{len(sampled_rows)}")

    ingest_meta = {
        "mode": "live-sampled",
        "requested_sample_size": int(sample_size),
        "selected_sample_size": len(sampled_rows),
        "prepared_claims": len(prepared),
        "unique_live_calls": len(cached_claims),
        "failed_claims": len(failed),
        "serpapi_enabled": bool(serpapi_enabled) if serpapi_enabled is not None else False,
    }
    if failed:
        ingest_meta["failed_examples"] = failed[:5]

    return prepared, ingest_meta


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _macro_f1(confusion: Dict[str, Dict[str, int]]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    per_label: Dict[str, Dict[str, float]] = {}
    labels = ["SUPPORTED", "REFUTED", "NEI"]
    f1_values: List[float] = []

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1_values.append(f1)

    return (_safe_div(sum(f1_values), len(f1_values)), per_label)


def _evaluate_candidate(prepared: List[PreparedClaim], thresholds: Dict[str, float]) -> Dict[str, Any]:
    labels = ["SUPPORTED", "REFUTED", "NEI"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}

    total = 0
    correct = 0
    abstained = 0
    abstained_on_nei = 0
    abstained_on_non_nei = 0

    label_counts = Counter(item.row.label for item in prepared)

    for item in prepared:
        reflective = reflective_analysis(
            item.row.claim,
            item.claim_profile,
            item.enriched_evidence,
            thresholds=thresholds,
        )
        is_abstain = reflective.get("decision") == "TERMINATE"
        predicted = "NEI" if is_abstain else item.baseline_verdict
        if predicted not in VALID_LABELS:
            predicted = "NEI"

        true = item.row.label
        confusion[true][predicted] += 1

        total += 1
        if predicted == true:
            correct += 1
        if is_abstain:
            abstained += 1
            if true == "NEI":
                abstained_on_nei += 1
            else:
                abstained_on_non_nei += 1

    accuracy = _safe_div(correct, total)
    macro_f1, per_label = _macro_f1(confusion)

    total_nei = label_counts.get("NEI", 0)
    total_non_nei = total - total_nei
    expected_nei_ratio = _safe_div(total_nei, total)
    abstention_rate = _safe_div(abstained, total)
    abstention_precision = _safe_div(abstained_on_nei, abstained)
    abstention_recall = _safe_div(abstained_on_nei, total_nei)
    false_abstention_non_nei = _safe_div(abstained_on_non_nei, total_non_nei)
    abstention_balance_penalty = abs(abstention_rate - expected_nei_ratio)

    # Objective favors high macro-F1, precise abstentions, and low false abstentions.
    objective = (
        macro_f1
        + 0.35 * abstention_precision
        + 0.25 * abstention_recall
        - 0.30 * false_abstention_non_nei
        - 0.10 * abstention_balance_penalty
    )

    return {
        "thresholds": {k: round(float(v), 4) for k, v in thresholds.items()},
        "objective": round(objective, 6),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "per_label": per_label,
        "abstention": {
            "rate": round(abstention_rate, 6),
            "precision": round(abstention_precision, 6),
            "recall": round(abstention_recall, 6),
            "false_abstention_non_nei": round(false_abstention_non_nei, 6),
            "expected_nei_ratio": round(expected_nei_ratio, 6),
            "abstained_total": int(abstained),
            "abstained_on_nei": int(abstained_on_nei),
            "abstained_on_non_nei": int(abstained_on_non_nei),
        },
        "label_distribution": dict(label_counts),
    }


def _markdown_summary(report: Dict[str, Any], top_k: int) -> str:
    best = report["best_candidate"]
    mode = str(report["metadata"].get("mode", "synthetic"))
    lines = [
        "# Reflective Threshold Tuning Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Mode: {mode}",
        f"- Claims evaluated: {report['metadata']['claim_count']}",
        f"- Grid size: {report['metadata']['grid_size']}",
    ]

    if mode == "live-sampled":
        lines.extend(
            [
                f"- Requested sample size: {report['metadata'].get('requested_sample_size', 0)}",
                f"- Selected sample size: {report['metadata'].get('selected_sample_size', 0)}",
                f"- Unique live calls: {report['metadata'].get('unique_live_calls', 0)}",
                f"- Live call failures: {report['metadata'].get('failed_claims', 0)}",
                f"- SERPAPI enabled: {report['metadata'].get('serpapi_enabled', False)}",
            ]
        )

    lines.extend(["", "## Recommended Thresholds", ""])

    for k, v in best["thresholds"].items():
        lines.append(f"- {k}: {v}")

    lines.extend(
        [
            "",
            "## Best Candidate Metrics",
            "",
            f"- Objective: {best['objective']:.4f}",
            f"- Accuracy: {best['accuracy']:.4f}",
            f"- Macro F1: {best['macro_f1']:.4f}",
            f"- Abstention rate: {best['abstention']['rate']:.4f}",
            f"- Abstention precision: {best['abstention']['precision']:.4f}",
            f"- Abstention recall (NEI): {best['abstention']['recall']:.4f}",
            f"- False abstention (non-NEI): {best['abstention']['false_abstention_non_nei']:.4f}",
            "",
            "## Top Candidates",
            "",
            "| Rank | strong_quality_min | low_factor_coverage_pct | conflict_quality_gap_max | Accuracy | Macro F1 | Abstain Precision | Abstain Recall | Objective |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for idx, item in enumerate(report["top_candidates"][:top_k], start=1):
        t = item["thresholds"]
        a = item["abstention"]
        lines.append(
            "| "
            f"{idx} | {t['strong_quality_min']:.1f} | {t['low_factor_coverage_pct']:.1f} | {t['conflict_quality_gap_max']:.1f} | "
            f"{item['accuracy']:.3f} | {item['macro_f1']:.3f} | {a['precision']:.3f} | {a['recall']:.3f} | {item['objective']:.3f} |"
        )

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input CSV not found: {input_path}")
        return 1

    rows = _load_claims(input_path)
    if not rows:
        print("No valid benchmark rows found in CSV.")
        return 1

    ingest_meta: Dict[str, Any] = {"mode": args.mode}
    if args.mode == "live-sampled":
        prepared, live_meta = _prepare_claims_live(
            rows,
            sample_size=args.live_sample_size,
            seed=args.seed,
            max_evidence_per_claim=args.live_max_evidence,
        )
        ingest_meta.update(live_meta)
    else:
        prepared = _prepare_claims_synthetic(rows)

    if not prepared:
        print("No prepared claims available for tuning. Check mode configuration and inputs.")
        return 1

    defaults = dict(DEFAULT_REFLECTIVE_THRESHOLDS)
    strong_grid = [56.0, 60.0, 64.0, 68.0]
    low_cov_grid = [24.0, 30.0, 36.0, 42.0]
    conflict_grid = [6.0, 8.0, 10.0, 12.0]

    all_results: List[Dict[str, Any]] = []
    for strong_q, low_cov, conflict_gap in product(strong_grid, low_cov_grid, conflict_grid):
        thresholds = dict(defaults)
        thresholds["strong_quality_min"] = strong_q
        thresholds["low_factor_coverage_pct"] = low_cov
        thresholds["conflict_quality_gap_max"] = conflict_gap

        result = _evaluate_candidate(prepared, thresholds)
        all_results.append(result)

    all_results.sort(
        key=lambda r: (
            float(r["objective"]),
            float(r["macro_f1"]),
            float(r["abstention"]["precision"]),
        ),
        reverse=True,
    )

    top_k = max(1, int(args.top_k))
    best = all_results[0]

    report = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_csv": str(input_path),
            "claim_count": len(prepared),
            "grid_size": len(all_results),
            **ingest_meta,
        },
        "defaults": defaults,
        "best_candidate": best,
        "top_candidates": all_results[:top_k],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = output_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(_markdown_summary(report, top_k=top_k))

    t = best["thresholds"]
    print("Reflective threshold tuning complete.")
    print(f"- Mode: {args.mode}")
    print(f"- Input claims: {len(prepared)}")
    print(f"- Output JSON: {output_path}")
    print(f"- Output Markdown: {md_path}")
    print(
        "- Recommended thresholds: "
        f"strong_quality_min={t['strong_quality_min']}, "
        f"low_factor_coverage_pct={t['low_factor_coverage_pct']}, "
        f"conflict_quality_gap_max={t['conflict_quality_gap_max']}"
    )
    print(
        "- Best metrics: "
        f"accuracy={best['accuracy']:.3f}, "
        f"macro_f1={best['macro_f1']:.3f}, "
        f"abstention_precision={best['abstention']['precision']:.3f}, "
        f"abstention_recall={best['abstention']['recall']:.3f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
