"""
Run Step 8 reproducibility audit.

This script evaluates whether the project is reproducible from an artifact and
engineering-process perspective. It checks:
- required documentation and scripts
- generated evaluation artifacts (Steps 1-7)
- expected API evaluation endpoints
- environment/tooling metadata snapshot

Outputs:
- JSON report
- CSV checklist
- Markdown summary

Usage:
  python Scripts/run_reproducibility_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]


@dataclass
class CheckResult:
    section: str
    check_id: str
    description: str
    passed: bool
    details: str
    weight: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 8 reproducibility audit")
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="API base URL for endpoint checks",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory where outputs are written",
    )
    return parser.parse_args()


def _run_cmd(cmd: List[str], cwd: Path) -> str:
    try:
        output = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True)
        return output.strip()
    except Exception:
        return ""


def _check_file_exists(path: Path, min_size_bytes: int = 1) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Missing: {path}"
    size = path.stat().st_size
    if size < min_size_bytes:
        return False, f"Too small ({size} bytes): {path}"
    return True, f"OK ({size} bytes): {path}"


def _check_endpoint(url: str) -> tuple[bool, str]:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                return True, f"HTTP 200: {url}"
            return False, f"HTTP {resp.status}: {url}"
    except Exception as exc:
        return False, f"Unavailable: {url} ({exc})"


def _weighted_score(checks: List[CheckResult]) -> Dict[str, float]:
    total = sum(c.weight for c in checks)
    achieved = sum(c.weight for c in checks if c.passed)
    pct = (achieved / total * 100.0) if total else 0.0
    return {
        "achieved_weight": achieved,
        "total_weight": total,
        "score_percent": pct,
    }


def _build_markdown(report: Dict) -> str:
    lines = [
        "# Reproducibility Audit Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Git commit: {report['metadata']['git_commit'] or 'unknown'}",
        f"- Python: {report['environment']['python_version']}",
        f"- Platform: {report['environment']['platform']}",
        "",
        "## Score",
        "",
        f"- Reproducibility score: {report['score']['score_percent']:.1f}%",
        f"- Passed checks: {report['summary']['passed_checks']} / {report['summary']['total_checks']}",
        "",
        "## Checklist",
        "",
        "| Section | Check | Status | Details |",
        "|---|---|---|---|",
    ]

    for row in report["checks"]:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"| {row['section']} | {row['check_id']} - {row['description']} | {status} | {row['details']} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This audit checks artifact completeness and runtime availability, not semantic correctness of every model decision.",
        "- For stronger reproducibility claims, run the full test suite and record dependency lockfiles in CI.",
    ])

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    checks: List[CheckResult] = []

    # Documentation checks
    doc_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "DEPLOYMENT.md",
        REPO_ROOT / "docs" / "METHODS.md",
        REPO_ROOT / "docs" / "LIMITATIONS.md",
        REPO_ROOT / "docs" / "COMPARATIVE_ANALYSIS.md",
        REPO_ROOT / "docs" / "THESIS_COMPARATIVE_EVALUATION.md",
    ]
    for i, path in enumerate(doc_files, start=1):
        ok, details = _check_file_exists(path, min_size_bytes=200)
        checks.append(
            CheckResult(
                section="docs",
                check_id=f"D{i}",
                description=f"Required documentation present: {path.name}",
                passed=ok,
                details=details,
                weight=1.0,
            )
        )

    # Script checks (Steps 1-8)
    script_files = [
        "prepare_research_benchmark.py",
        "run_baseline_comparison.py",
        "run_ablation_study.py",
        "run_comparative_analysis.py",
        "run_production_metrics.py",
        "run_explainability_demo.py",
        "run_limitations_assessment.py",
        "run_reproducibility_audit.py",
    ]
    for i, name in enumerate(script_files, start=1):
        path = API_ROOT / "Scripts" / name
        ok, details = _check_file_exists(path, min_size_bytes=100)
        checks.append(
            CheckResult(
                section="scripts",
                check_id=f"S{i}",
                description=f"Pipeline script present: {name}",
                passed=ok,
                details=details,
                weight=1.0,
            )
        )

    # Artifact checks (Steps 1-8)
    result_files = [
        REPO_ROOT / "data" / "benchmarks" / "research_benchmark_v1.json",
        REPO_ROOT / "data" / "benchmarks" / "results" / "baseline_comparison_report.json",
        REPO_ROOT / "data" / "benchmarks" / "results" / "ablation_study_report.json",
        REPO_ROOT / "data" / "benchmarks" / "results" / "comparative_analysis_report.json",
        REPO_ROOT / "data" / "benchmarks" / "results" / "production_metrics_report.json",
        REPO_ROOT / "data" / "benchmarks" / "results" / "explainability_demo_report.json",
        REPO_ROOT / "data" / "benchmarks" / "results" / "limitations_assessment_report.json",
    ]
    for i, path in enumerate(result_files, start=1):
        ok, details = _check_file_exists(path, min_size_bytes=50)
        checks.append(
            CheckResult(
                section="artifacts",
                check_id=f"A{i}",
                description=f"Generated artifact present: {path.name}",
                passed=ok,
                details=details,
                weight=1.5,
            )
        )

    # Endpoint checks
    endpoints = [
        "/health",
        "/evaluation/benchmark",
        "/evaluation/baselines",
        "/evaluation/ablations",
        "/evaluation/comparative",
        "/evaluation/production-metrics",
        "/evaluation/explainability",
        "/evaluation/limitations",
    ]
    for i, suffix in enumerate(endpoints, start=1):
        ok, details = _check_endpoint(args.api_base.rstrip("/") + suffix)
        checks.append(
            CheckResult(
                section="runtime",
                check_id=f"R{i}",
                description=f"Endpoint available: {suffix}",
                passed=ok,
                details=details,
                weight=1.5,
            )
        )

    score = _weighted_score(checks)

    passed_checks = sum(1 for c in checks if c.passed)
    total_checks = len(checks)

    git_commit = _run_cmd(["git", "rev-parse", "--short", "HEAD"], REPO_ROOT)
    git_branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], REPO_ROOT)

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "git_commit": git_commit,
            "git_branch": git_branch,
            "api_base": args.api_base,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": {
            "passed_checks": passed_checks,
            "total_checks": total_checks,
        },
        "score": score,
        "checks": [
            {
                "section": c.section,
                "check_id": c.check_id,
                "description": c.description,
                "passed": c.passed,
                "details": c.details,
                "weight": c.weight,
            }
            for c in checks
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "reproducibility_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "reproducibility_audit_checks.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "check_id", "description", "passed", "details", "weight"],
        )
        writer.writeheader()
        for row in report["checks"]:
            writer.writerow(row)

    md_path = output_dir / "reproducibility_audit_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print("Reproducibility audit completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV checks: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())