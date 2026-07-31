"""Generate the final thesis reproducibility audit from the correction branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
RESULT_DIR = REPO_ROOT / "data" / "benchmarks" / "results_5000"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis reproducibility audit")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Record tests as not executed (intended only for quick local inspection)",
    )
    return parser.parse_args()


def _run(command: list[str], timeout: int = 300) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "output": completed.stdout[-20000:],
    }


def _git(*args: str) -> str:
    result = _run(["git", *args], timeout=30)
    return result["output"].strip() if result["exit_code"] == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except (ImportError, OSError):
        return None


def _gpu() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        value = result.stdout.strip()
        return value or "not detected"
    except (OSError, subprocess.TimeoutExpired):
        return "not detected"


def main() -> int:
    args = _parse_args()
    commands: list[dict[str, Any]] = []
    if args.skip_tests:
        test_result = {
            "command": "pytest services/api/tests -q",
            "exit_code": None,
            "output": "not executed (--skip-tests)",
        }
    else:
        test_result = _run(["pytest", "services/api/tests", "-q"], timeout=600)
    commands.append(test_result)
    validation = _run(
        ["python", "services/api/Scripts/validate_thesis_artifacts.py"],
        timeout=300,
    )
    commands.append(validation)

    tracked_artifacts = [
        REPO_ROOT / "services" / "api" / "requirements.lock",
        REPO_ROOT / "data" / "benchmarks" / "splits_5000" / "train.json",
        REPO_ROOT / "data" / "benchmarks" / "splits_5000" / "val.json",
        REPO_ROOT / "data" / "benchmarks" / "splits_5000" / "test.json",
        RESULT_DIR / "ablation_study_predictions.csv",
        RESULT_DIR / "baseline_comparison_predictions.csv",
        RESULT_DIR / "statistics_report.json",
    ]
    hashes = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _sha256(path)
        for path in tracked_artifacts
        if path.is_file()
    }
    report = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_tag_exact": _git("describe", "--tags", "--exact-match"),
            # Untracked local build outputs do not alter the committed source
            # snapshot being audited. Track only modifications to versioned files.
            "working_tree_dirty": bool(
                _git("status", "--porcelain", "--untracked-files=no")
            ),
        },
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "gpu": _gpu(),
            "ram_bytes": _ram_bytes(),
        },
        "dependency_lock": {
            "path": "services/api/requirements.lock",
            "sha256": hashes.get("services/api/requirements.lock"),
        },
        "artifact_hashes": hashes,
        "commands": commands,
        "test_result": {
            "exit_code": test_result["exit_code"],
            "passed": test_result["exit_code"] == 0,
            "summary_output": test_result["output"][-4000:],
        },
        "artifact_validation": {
            "exit_code": validation["exit_code"],
            "passed": validation["exit_code"] == 0,
            "summary_output": validation["output"][-4000:],
        },
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "reproducibility_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    summary = [
        "# Reproducibility Audit Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Branch: `{report['metadata']['git_branch']}`",
        f"- Commit: `{report['metadata']['git_commit']}`",
        f"- Python: `{report['environment']['python_version']}`",
        f"- OS: `{report['environment']['os']}`",
        f"- CPU: `{report['environment']['cpu']}`",
        f"- GPU: `{report['environment']['gpu']}`",
        f"- RAM bytes: `{report['environment']['ram_bytes']}`",
        f"- Tests passed: **{report['test_result']['passed']}**",
        f"- Artifact validation passed: **{report['artifact_validation']['passed']}**",
        "",
        "## Executed commands",
        "",
    ]
    for item in commands:
        summary.append(f"- `{item['command']}` -> exit {item['exit_code']}")
    summary.extend(
        [
            "",
            "## Test output",
            "",
            "```text",
            test_result["output"][-4000:].strip(),
            "```",
            "",
            "The audit records artifact integrity and executed software checks. "
            "It does not convert proxy benchmark results into live-application results.",
        ]
    )
    (RESULT_DIR / "reproducibility_audit_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(f"Wrote reproducibility audit to {RESULT_DIR}")
    return 0 if report["test_result"]["passed"] and report["artifact_validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
