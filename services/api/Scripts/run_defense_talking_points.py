"""
Run Step 10 defense talking points synthesis.

This script compiles thesis defense-ready talking points from previously
generated reports and produces:
- structured Q&A with evidence-backed answers
- likely objection handling responses
- quick metrics cheat-sheet

Outputs:
- JSON report
- CSV Q&A table
- Markdown summary

Usage:
  python Scripts/run_defense_talking_points.py
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
    parser = argparse.ArgumentParser(description="Run Step 10 defense talking points synthesis")
    parser.add_argument(
        "--comparative-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "comparative_analysis_report.json"),
        help="Path to comparative analysis report",
    )
    parser.add_argument(
        "--production-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "production_metrics_report.json"),
        help="Path to production metrics report",
    )
    parser.add_argument(
        "--limitations-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "limitations_assessment_report.json"),
        help="Path to limitations assessment report",
    )
    parser.add_argument(
        "--ethics-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "ethics_assessment_report.json"),
        help="Path to ethics assessment report",
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


def _build_qa(comp: Dict, prod: Dict, lim: Dict, ethics: Dict) -> List[Dict[str, str]]:
    ranking = comp.get("ranking", [])
    top = ranking[0] if ranking else {}
    quality = prod.get("quality", {})
    cost = prod.get("cost", {})
    limitations = lim.get("limitations", [])
    ethics_risks = ethics.get("ethical_risks", [])

    top_acc = float(top.get("accuracy", 0.0))
    top_sys = str(top.get("system", "full_proxy"))
    err_rate = float(quality.get("error_rate", 0.0))
    monthly_saved = float(cost.get("monthly_savings_usd", 0.0))
    savings_pct = float(cost.get("monthly_savings_pct", 0.0)) * 100.0
    high_lim = [x for x in limitations if x.get("severity") == "high"]
    high_ethics = [x for x in ethics_risks if x.get("severity") == "high"]

    qa: List[Dict[str, str]] = []

    qa.append(
        {
            "question": "What is your main empirical contribution?",
            "answer": (
                f"Our full system variant ({top_sys}) ranks highest on the current benchmark with accuracy {top_acc:.3f}, "
                "while integrating comparative evaluation, production metrics, explainability, limitations, reproducibility, and ethics workflows in one deployable stack."
            ),
            "evidence": "comparative_analysis_report.json + production_metrics_report.json",
            "category": "contribution",
        }
    )

    qa.append(
        {
            "question": "How do you justify practical value beyond accuracy?",
            "answer": (
                f"Caching-aware operation reduces estimated monthly API spend by about ${monthly_saved:.2f} "
                f"({savings_pct:.1f}% savings at the configured workload), with explicit latency and throughput reporting."
            ),
            "evidence": "production_metrics_report.json",
            "category": "practicality",
        }
    )

    qa.append(
        {
            "question": "What are your system's biggest weaknesses?",
            "answer": (
                f"Current error rate is {err_rate:.3f}, and the most critical limitation is small-sample statistical fragility. "
                "We explicitly track these in a limitations register and attach mitigation actions."
            ),
            "evidence": "limitations_assessment_report.json",
            "category": "limitations",
        }
    )

    qa.append(
        {
            "question": "How do you address bias and societal risk?",
            "answer": (
                "We maintain an ethics risk register with explicit guardrails, ownership, and phased mitigation. "
                f"Current report flags {len(high_ethics)} high-severity ethics risks, primarily around source-selection bias and overconfidence harms."
            ),
            "evidence": "ethics_assessment_report.json",
            "category": "ethics",
        }
    )

    qa.append(
        {
            "question": "How reproducible are your results?",
            "answer": (
                "The project includes dedicated scripts and machine-readable artifacts for each evaluation step, plus a reproducibility audit endpoint "
                "that validates report presence and runtime availability."
            ),
            "evidence": "reproducibility_audit_report.json",
            "category": "reproducibility",
        }
    )

    qa.append(
        {
            "question": "What is your defense strategy when asked about small benchmark size?",
            "answer": (
                f"We acknowledge the limitation directly (high-severity items: {len(high_lim)}) and frame current results as controlled pilot evidence. "
                "Our next milestone is expanding to domain-balanced 100+ claim slices with repeated significance analysis."
            ),
            "evidence": "limitations_assessment_report.json + comparative_analysis_report.json",
            "category": "defense",
        }
    )

    qa.append(
        {
            "question": "What should evaluators remember in one line?",
            "answer": (
                "This work is not just a classifier; it is an evidence-aware fact-checking platform with measurable tradeoffs, explicit uncertainty, and governance-ready reporting."
            ),
            "evidence": "steps 1-10 integrated outputs",
            "category": "summary",
        }
    )

    return qa


def _build_metrics_cheatsheet(comp: Dict, prod: Dict, lim: Dict, ethics: Dict) -> List[Dict[str, str]]:
    ranking = comp.get("ranking", [])
    top = ranking[0] if ranking else {}
    quality = prod.get("quality", {})
    latency = prod.get("latency", {})
    throughput = prod.get("throughput", {})
    cost = prod.get("cost", {})

    return [
        {"metric": "Top System", "value": str(top.get("system", "-")), "source": "comparative"},
        {"metric": "Top Accuracy", "value": f"{float(top.get('accuracy', 0.0)):.3f}", "source": "comparative"},
        {"metric": "Error Rate", "value": f"{float(quality.get('error_rate', 0.0)):.3f}", "source": "production"},
        {"metric": "Macro F1", "value": f"{float(quality.get('macro_f1', 0.0)):.3f}", "source": "production"},
        {"metric": "Baseline Latency (sec)", "value": f"{float(latency.get('baseline_avg_sec', 0.0)):.2f}", "source": "production"},
        {"metric": "Debate Latency (sec)", "value": f"{float(latency.get('debate_avg_sec', 0.0)):.2f}", "source": "production"},
        {"metric": "Baseline Throughput (claims/hour)", "value": f"{float(throughput.get('baseline_claims_per_hour', 0.0)):.2f}", "source": "production"},
        {"metric": "Debate Throughput (claims/hour)", "value": f"{float(throughput.get('debate_claims_per_hour', 0.0)):.2f}", "source": "production"},
        {"metric": "Monthly Savings (USD)", "value": f"{float(cost.get('monthly_savings_usd', 0.0)):.2f}", "source": "production"},
        {"metric": "High-Severity Limitations", "value": str(lim.get('metadata', {}).get('high_severity_count', 0)), "source": "limitations"},
        {"metric": "High-Severity Ethics Risks", "value": str(ethics.get('metadata', {}).get('high_severity_count', 0)), "source": "ethics"},
    ]


def _build_markdown(report: Dict) -> str:
    lines = [
        "# Defense Talking Points Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Q&A items: {len(report['qa'])}",
        "",
        "## Rapid Q&A",
        "",
    ]

    for idx, row in enumerate(report["qa"], start=1):
        lines.append(f"### Q{idx}. {row['question']}")
        lines.append("")
        lines.append(f"- Answer: {row['answer']}")
        lines.append(f"- Evidence: {row['evidence']}")
        lines.append("")

    lines.extend([
        "## Metrics Cheat-Sheet",
        "",
        "| Metric | Value | Source |",
        "|---|---|---|",
    ])
    for item in report["metrics_cheatsheet"]:
        lines.append(f"| {item['metric']} | {item['value']} | {item['source']} |")

    lines.extend([
        "",
        "## Closing Statement",
        "",
        report["closing_statement"],
    ])

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    comparative = _load_json(Path(args.comparative_report))
    production = _load_json(Path(args.production_report))
    limitations = _load_json(Path(args.limitations_report))
    ethics = _load_json(Path(args.ethics_report))

    qa = _build_qa(comparative, production, limitations, ethics)
    metrics_cheatsheet = _build_metrics_cheatsheet(comparative, production, limitations, ethics)

    closing_statement = (
        "The system demonstrates a full research-to-production pipeline with measurable performance, explicit uncertainty, "
        "artifact-level reproducibility, and a concrete ethics governance layer."
    )

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "sources": {
                "comparative_report": str(Path(args.comparative_report)),
                "production_report": str(Path(args.production_report)),
                "limitations_report": str(Path(args.limitations_report)),
                "ethics_report": str(Path(args.ethics_report)),
            },
        },
        "qa": qa,
        "metrics_cheatsheet": metrics_cheatsheet,
        "closing_statement": closing_statement,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "defense_talking_points_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "defense_talking_points_qa.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "question", "answer", "evidence"],
        )
        writer.writeheader()
        writer.writerows(qa)

    md_path = output_dir / "defense_talking_points_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print("Defense talking points synthesis completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV Q&A: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())