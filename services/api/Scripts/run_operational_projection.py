"""Generate explicitly labelled operational scenario projections."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
OUTPUT = (
    REPO_ROOT
    / "data"
    / "benchmarks"
    / "results_5000"
    / "operational_projection_report.json"
)


def main() -> int:
    baseline_latency = 8.2
    debate_latency = 72.0
    no_cache = 77.0
    with_cache = 22.0
    report = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_type": "operational_projection",
            "measurement_status": (
                "Scenario projection from stated assumptions; not a controlled "
                "concurrent load-test measurement."
            ),
        },
        "assumptions": {
            "baseline_seconds_per_claim": baseline_latency,
            "debate_seconds_per_claim": debate_latency,
            "claims_per_month": 1000,
            "monthly_usd_without_cache": no_cache,
            "monthly_usd_with_cache": with_cache,
        },
        "projections": {
            "baseline_claims_per_hour": 3600.0 / baseline_latency,
            "debate_claims_per_hour": 3600.0 / debate_latency,
            "debate_latency_ratio": debate_latency / baseline_latency,
            "monthly_savings_usd": no_cache - with_cache,
            "monthly_savings_fraction": (no_cache - with_cache) / no_cache,
        },
        "required_for_measured_claim": [
            "repeated warm-cache and cold-cache runs",
            "median, p90, and p95 latency",
            "declared concurrency",
            "failure and timeout counts",
            "hardware and provider request counts",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
