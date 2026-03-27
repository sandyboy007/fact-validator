"""
Run Step 7 limitations assessment synthesis.

This script compiles an evidence-backed limitations register from existing
evaluation artifacts and exports thesis-ready outputs covering:
- observed failure modes
- bias/generalization risks
- mitigation priorities

Outputs:
- JSON report
- CSV limitations table
- Markdown summary

Usage:
  python Scripts/run_limitations_assessment.py
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 7 limitations assessment")
    parser.add_argument(
        "--ablation-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "ablation_study_report.json"),
        help="Path to ablation report JSON",
    )
    parser.add_argument(
        "--production-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "production_metrics_report.json"),
        help="Path to production metrics report JSON",
    )
    parser.add_argument(
        "--explainability-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "explainability_demo_report.json"),
        help="Path to explainability report JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory where outputs are written",
    )
    parser.add_argument(
        "--low-category-threshold",
        type=float,
        default=0.6,
        help="Accuracy threshold below which category is flagged as weak",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _severity_from_ratio(x: float) -> str:
    if x >= 0.4:
        return "high"
    if x >= 0.2:
        return "medium"
    return "low"


def _register_item(
    item_id: str,
    title: str,
    severity: str,
    impact: str,
    evidence: str,
    mitigation: str,
) -> Dict[str, str]:
    return {
        "id": item_id,
        "title": title,
        "severity": severity,
        "impact": impact,
        "evidence": evidence,
        "mitigation": mitigation,
    }


def _build_markdown(report: Dict) -> str:
    lines = [
        "# Limitations Assessment Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Registered limitations: {report['metadata']['limitation_count']}",
        f"- High severity items: {report['metadata']['high_severity_count']}",
        "",
        "## Limitations Register",
        "",
        "| ID | Limitation | Severity | Impact |",
        "|---|---|---|---|",
    ]

    for item in report["limitations"]:
        lines.append(f"| {item['id']} | {item['title']} | {item['severity']} | {item['impact']} |")

    lines.extend(["", "## Evidence & Mitigation", ""])
    for item in report["limitations"]:
        lines.append(f"### {item['id']} - {item['title']}")
        lines.append("")
        lines.append(f"- Severity: {item['severity']}")
        lines.append(f"- Evidence: {item['evidence']}")
        lines.append(f"- Mitigation: {item['mitigation']}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    ablation = _load_json(Path(args.ablation_report))
    production = _load_json(Path(args.production_report))
    explainability = _load_json(Path(args.explainability_report))

    limitations: List[Dict[str, str]] = []

    # L1: observed error rate
    quality = production.get("quality", {})
    error_rate = float(quality.get("error_rate", 0.0))
    limitations.append(
        _register_item(
            "L1",
            "Residual classification errors",
            _severity_from_ratio(error_rate),
            "Incorrect verdicts remain possible in real-world usage.",
            f"Observed error rate is {error_rate:.3f} ({error_rate * 100:.1f} errors per 100 claims).",
            "Route low-confidence outputs to human review; expand evaluation set and tune calibration.",
        )
    )

    # L2: domain/category generalization gaps from per-category accuracy
    full_metrics = ablation.get("variants", {}).get("full_proxy", {}).get("metrics", {})
    per_category = full_metrics.get("per_category", {})
    weak_categories = []
    for category, metric in per_category.items():
        acc = float(metric.get("accuracy", 0.0))
        if acc < args.low_category_threshold:
            weak_categories.append((category, acc, int(metric.get("count", 0))))

    if weak_categories:
        weak_categories.sort(key=lambda x: x[1])
        weak_txt = "; ".join([f"{c}={a:.2f} (n={n})" for c, a, n in weak_categories])
        limitations.append(
            _register_item(
                "L2",
                "Category-specific generalization gaps",
                "medium" if len(weak_categories) <= 2 else "high",
                "Performance is uneven across claim domains.",
                f"Categories below threshold {args.low_category_threshold:.2f}: {weak_txt}",
                "Increase domain-balanced benchmark size and add domain-specific retrieval prompts/models.",
            )
        )

    # L3: debate arbitration instability
    cases = explainability.get("case_studies", [])
    debate_changes = 0
    debate_regressions = 0
    for case in cases:
        full_label = case.get("predictions", {}).get("full", {}).get("label")
        no_debate_label = case.get("predictions", {}).get("no_debate", {}).get("label")
        gt = case.get("ground_truth_label")
        if full_label != no_debate_label:
            debate_changes += 1
            if full_label != gt and no_debate_label == gt:
                debate_regressions += 1

    total_cases = max(1, len(cases))
    change_rate = debate_changes / total_cases
    regression_rate = debate_regressions / total_cases
    limitations.append(
        _register_item(
            "L3",
            "Debate arbitration can introduce regressions",
            _severity_from_ratio(regression_rate),
            "Debate mode may change verdicts without guaranteed net improvement.",
            (
                f"Debate changed {debate_changes}/{total_cases} cases ({change_rate:.2f}); "
                f"regressions observed in {debate_regressions}/{total_cases} ({regression_rate:.2f})."
            ),
            "Trigger debate selectively (only uncertain baseline cases) and validate with guardrail thresholds.",
        )
    )

    # L4: confidence calibration quality
    calibration_error = float(quality.get("calibration_error", 0.0))
    ece = float(quality.get("ece", 0.0))
    limitations.append(
        _register_item(
            "L4",
            "Confidence calibration mismatch",
            _severity_from_ratio(max(calibration_error, ece)),
            "Displayed confidence may overstate or understate true correctness.",
            f"Calibration error={calibration_error:.3f}, ECE={ece:.3f} in current split.",
            "Apply post-hoc calibration (temperature scaling / isotonic regression) on larger validation data.",
        )
    )

    # L5: source/credibility bias (documented structural limitation)
    limitations.append(
        _register_item(
            "L5",
            "Source-selection and credibility-rubric bias",
            "medium",
            "System trust signals may favor mainstream indexed domains and under-represent local/novel sources.",
            "Credibility scoring relies on domain rubric and search-retrieved evidence, not exhaustive ground truth.",
            "Introduce external expert calibration panel and diversify retrieval sources beyond a single search index.",
        )
    )

    # L6: small test split statistical fragility
    claims_in_split = int(production.get("metadata", {}).get("claims_in_split", 0))
    limitations.append(
        _register_item(
            "L6",
            "Limited statistical power from small evaluation split",
            "high" if claims_in_split < 30 else "medium",
            "Point estimates and p-values are sensitive to a few cases.",
            f"Current evaluated split contains {claims_in_split} claims.",
            "Expand benchmark to 100+ claims per major domain and recompute all comparison statistics.",
        )
    )

    high_severity_count = sum(1 for x in limitations if x["severity"] == "high")

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "limitation_count": len(limitations),
            "high_severity_count": high_severity_count,
            "sources": {
                "ablation_report": str(Path(args.ablation_report)),
                "production_report": str(Path(args.production_report)),
                "explainability_report": str(Path(args.explainability_report)),
            },
        },
        "limitations": limitations,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "limitations_assessment_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "limitations_assessment_table.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "title", "severity", "impact", "evidence", "mitigation"],
        )
        writer.writeheader()
        writer.writerows(limitations)

    md_path = output_dir / "limitations_assessment_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print("Limitations assessment completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV table: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())