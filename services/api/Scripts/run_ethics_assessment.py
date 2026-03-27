"""
Run Step 9 ethical discussion synthesis.

This script generates thesis-ready ethics artifacts focused on:
- bias and fairness risks
- societal impact analysis
- misuse and governance controls

Outputs:
- JSON report
- CSV ethics risk register
- Markdown summary

Usage:
  python Scripts/run_ethics_assessment.py
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
    parser = argparse.ArgumentParser(description="Run Step 9 ethics assessment")
    parser.add_argument(
        "--limitations-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "limitations_assessment_report.json"),
        help="Path to limitations assessment report",
    )
    parser.add_argument(
        "--production-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "production_metrics_report.json"),
        help="Path to production metrics report",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Output directory",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _risk(
    risk_id: str,
    domain: str,
    title: str,
    likelihood: str,
    impact: str,
    severity: str,
    evidence: str,
    mitigation: str,
    owner: str,
) -> Dict[str, str]:
    return {
        "id": risk_id,
        "domain": domain,
        "title": title,
        "likelihood": likelihood,
        "impact": impact,
        "severity": severity,
        "evidence": evidence,
        "mitigation": mitigation,
        "owner": owner,
    }


def _build_markdown(report: Dict) -> str:
    lines = [
        "# Ethics Assessment Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Total ethical risks: {report['metadata']['risk_count']}",
        f"- High-severity risks: {report['metadata']['high_severity_count']}",
        "",
        "## Ethical Risk Register",
        "",
        "| ID | Domain | Risk | Severity | Owner |",
        "|---|---|---|---|---|",
    ]

    for r in report["ethical_risks"]:
        lines.append(f"| {r['id']} | {r['domain']} | {r['title']} | {r['severity']} | {r['owner']} |")

    lines.extend(["", "## Guardrails", ""])
    for g in report["deployment_guardrails"]:
        lines.append(f"- {g}")

    lines.extend(["", "## Mitigation Roadmap", ""])
    for phase, items in report["mitigation_roadmap"].items():
        lines.append(f"### {phase}")
        lines.append("")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    limitations = _load_json(Path(args.limitations_report))
    production = _load_json(Path(args.production_report))

    quality = production.get("quality", {})
    error_rate = float(quality.get("error_rate", 0.0))
    calibration_error = float(quality.get("calibration_error", 0.0))
    ece = float(quality.get("ece", 0.0))

    # Use limitations evidence to support ethics framing.
    limitation_titles = {x.get("id", ""): x.get("title", "") for x in limitations.get("limitations", [])}

    ethical_risks: List[Dict[str, str]] = []

    ethical_risks.append(
        _risk(
            "E1",
            "fairness",
            "Source-selection bias can under-represent minority or local viewpoints",
            likelihood="medium",
            impact="high",
            severity="high",
            evidence=f"Linked limitations: {limitation_titles.get('L5', 'source bias')}.",
            mitigation="Add multi-source retrieval diversification and periodic expert review of credibility rubric.",
            owner="ML + Policy",
        )
    )

    ethical_risks.append(
        _risk(
            "E2",
            "safety",
            "Overconfident outputs may mislead users in high-stakes domains",
            likelihood="medium",
            impact="high",
            severity="high" if max(calibration_error, ece) >= 0.25 else "medium",
            evidence=f"Calibration error={calibration_error:.3f}, ECE={ece:.3f}.",
            mitigation="Calibrate confidence and display uncertainty reasons with explicit human-review triggers.",
            owner="ML + Product",
        )
    )

    ethical_risks.append(
        _risk(
            "E3",
            "harm",
            "Residual model error can amplify misinformation if used as sole authority",
            likelihood="medium",
            impact="high",
            severity="high" if error_rate >= 0.25 else "medium",
            evidence=f"Observed error rate={error_rate:.3f} in evaluation split.",
            mitigation="Position system as triage aid, not final truth authority; require human-in-loop for critical claims.",
            owner="Product + Trust & Safety",
        )
    )

    ethical_risks.append(
        _risk(
            "E4",
            "governance",
            "Debate mode may alter verdicts without guaranteed net safety improvement",
            likelihood="medium",
            impact="medium",
            severity="medium",
            evidence=f"Linked limitations: {limitation_titles.get('L3', 'debate regressions')}.",
            mitigation="Enable debate only when baseline uncertainty exceeds policy threshold.",
            owner="ML",
        )
    )

    ethical_risks.append(
        _risk(
            "E5",
            "transparency",
            "Users may misunderstand confidence and think verdicts are definitive",
            likelihood="high",
            impact="medium",
            severity="medium",
            evidence="Current UX exposes confidence numbers but users may interpret them as certainty.",
            mitigation="Add concise disclaimers and confidence interpretation bands in UI.",
            owner="UX + Product",
        )
    )

    deployment_guardrails = [
        "Do not use as sole decision-maker for legal, medical, or electoral enforcement decisions.",
        "Automatically require human review for low-confidence or high-risk domain claims.",
        "Log model decisions and uncertainty reasons for post-hoc auditing.",
        "Track domain-level disparity metrics and review monthly for drift or bias.",
        "Publish model limitations and update rubric changelog transparently.",
    ]

    mitigation_roadmap = {
        "Immediate (0-2 weeks)": [
            "Display explicit 'assistive tool' warning in results UI.",
            "Enable policy rule: mandatory human review under confidence threshold.",
            "Add operational alert on spikes in disagreement/error rate.",
        ],
        "Near-term (2-6 weeks)": [
            "Run expert panel audit for credibility rubric and source weighting.",
            "Implement confidence calibration (temperature scaling / isotonic).",
            "Expand benchmark with underrepresented domains and multilingual samples.",
        ],
        "Mid-term (6-12 weeks)": [
            "Introduce fairness dashboard with group/domain parity monitoring.",
            "Add independent retrieval providers to reduce single-source bias.",
            "Formalize governance review cadence and incident response playbook.",
        ],
    }

    high_severity_count = sum(1 for r in ethical_risks if r["severity"] == "high")

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "risk_count": len(ethical_risks),
            "high_severity_count": high_severity_count,
            "sources": {
                "limitations_report": str(Path(args.limitations_report)),
                "production_report": str(Path(args.production_report)),
            },
        },
        "ethical_risks": ethical_risks,
        "deployment_guardrails": deployment_guardrails,
        "mitigation_roadmap": mitigation_roadmap,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ethics_assessment_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "ethics_assessment_risks.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "domain", "title", "likelihood", "impact", "severity", "evidence", "mitigation", "owner"],
        )
        writer.writeheader()
        writer.writerows(ethical_risks)

    md_path = output_dir / "ethics_assessment_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print("Ethics assessment completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV risks: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())