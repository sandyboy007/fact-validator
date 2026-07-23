"""Replay frozen evidence and evaluate graph-only versus audited graph decisions.

Input is a JSON list or object with ``items``. Each item needs ``id``, ``claim``,
and ``expected_verdict``. It may include ``evidence`` and a ``corruptions`` object
whose values are frozen evidence lists. The script never invents corrupted evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
sys.path.insert(0, str(API_ROOT))

from app.robustness_evaluation import build_scenarios, evaluate_case, selective_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval robustness from frozen evidence")
    parser.add_argument("--input", required=True, help="JSON challenge set with frozen evidence scenarios")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "retrieval_robustness_report.json"),
        help="JSON report path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    payload: Any = json.loads(input_path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("input must be a JSON list or an object with an items list")

    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("claim") or not item.get("expected_verdict"):
            continue
        for scenario_name, evidence in build_scenarios(item).items():
            rows.append(evaluate_case(item, scenario_name, evidence))
    if not rows:
        raise ValueError("no valid frozen-evidence scenarios found")

    report = {
        "version": "retrieval-robustness-report-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(input_path.resolve()),
        "warning": "Interpret results only after using independently sourced, annotated evidence and predeclared corruptions.",
        "scenarios_evaluated": len(rows),
        "systems": {
            "graph_only": selective_metrics(rows, "graph_only"),
            "full_audited_graph": selective_metrics(rows, "full_audited_graph"),
        },
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {len(rows)} frozen-evidence scenarios.")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())