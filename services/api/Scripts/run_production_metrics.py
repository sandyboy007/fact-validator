"""
Run Step 5 production metrics synthesis.

This script aggregates quality metrics from existing evaluation artifacts and
computes estimated production characteristics for thesis reporting:
- latency
- throughput
- cost
- error rates

Outputs:
- JSON report
- CSV key-value table
- Markdown summary

Usage:
  python Scripts/run_production_metrics.py
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
    parser = argparse.ArgumentParser(description="Run Step 5 production metrics")
    parser.add_argument(
        "--ablation-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "ablation_study_report.json"),
        help="Path to Step 3 ablation report",
    )
    parser.add_argument(
        "--comparative-report",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "comparative_analysis_report.json"),
        help="Path to Step 4 comparative report",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory where Step 5 outputs are written",
    )

    # Assumptions (overridable)
    parser.add_argument("--baseline-latency-sec", type=float, default=8.2)
    parser.add_argument("--debate-latency-sec", type=float, default=72.0)
    parser.add_argument("--claims-per-month", type=int, default=1000)
    parser.add_argument("--serpapi-cost-per-query", type=float, default=0.022)
    parser.add_argument("--baseline-calls-per-claim", type=float, default=3.5)
    parser.add_argument("--cached-calls-per-claim", type=float, default=1.0)
    parser.add_argument("--full-variant", default="full_proxy")
    parser.add_argument(
        "--telemetry-json",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "production_telemetry.json"),
        help="Optional runtime telemetry JSON with cache, CPU, memory, and resilience metrics",
    )

    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _flatten_report_rows(report: Dict) -> List[Dict[str, str]]:
    rows = []

    def add(metric: str, value: float | int | str, unit: str = "") -> None:
        rows.append({"metric": metric, "value": str(value), "unit": unit})

    latency = report["latency"]
    throughput = report["throughput"]
    cost = report["cost"]
    quality = report["quality"]

    add("latency.baseline_avg_sec", round(latency["baseline_avg_sec"], 4), "sec")
    add("latency.debate_avg_sec", round(latency["debate_avg_sec"], 4), "sec")
    add("latency.debate_over_baseline_ratio", round(latency["debate_over_baseline_ratio"], 4), "x")

    add("throughput.baseline_claims_per_hour", round(throughput["baseline_claims_per_hour"], 2), "claims/hour")
    add("throughput.debate_claims_per_hour", round(throughput["debate_claims_per_hour"], 2), "claims/hour")

    add("cost.claims_per_month", cost["claims_per_month"], "claims")
    add("cost.monthly_usd_no_cache", round(cost["monthly_usd_no_cache"], 2), "USD")
    add("cost.monthly_usd_with_cache", round(cost["monthly_usd_with_cache"], 2), "USD")
    add("cost.monthly_savings_usd", round(cost["monthly_savings_usd"], 2), "USD")
    add("cost.monthly_savings_pct", round(cost["monthly_savings_pct"], 4), "ratio")

    add("quality.accuracy", round(quality["accuracy"], 6), "ratio")
    add("quality.error_rate", round(quality["error_rate"], 6), "ratio")
    add("quality.expected_errors_per_100_claims", round(quality["expected_errors_per_100_claims"], 3), "errors")
    add("quality.macro_f1", round(quality["macro_f1"], 6), "ratio")
    add("quality.calibration_error", round(quality["calibration_error"], 6), "ratio")
    add("quality.ece", round(quality["ece"], 6), "ratio")

    runtime = report.get("runtime") or {}
    if runtime:
        add("runtime.cache_hit_rate", round(float(runtime.get("cache_hit_rate", 0.0)), 6), "ratio")
        add("runtime.cache_miss_rate", round(float(runtime.get("cache_miss_rate", 0.0)), 6), "ratio")
        add("runtime.cpu_utilization_mean_pct", round(float(runtime.get("cpu_utilization_mean_pct", 0.0)), 4), "pct")
        add("runtime.cpu_utilization_peak_pct", round(float(runtime.get("cpu_utilization_peak_pct", 0.0)), 4), "pct")
        add("runtime.memory_usage_mean_mb", round(float(runtime.get("memory_usage_mean_mb", 0.0)), 4), "MB")
        add("runtime.memory_usage_peak_mb", round(float(runtime.get("memory_usage_peak_mb", 0.0)), 4), "MB")
        add("runtime.concurrent_requests_tested", int(runtime.get("concurrent_requests_tested", 0)), "count")
        add("runtime.p95_latency_sec", round(float(runtime.get("p95_latency_sec", 0.0)), 4), "sec")
        add("runtime.failure_recovery_rate", round(float(runtime.get("failure_recovery_rate", 0.0)), 6), "ratio")
        add("runtime.median_recovery_time_sec", round(float(runtime.get("median_recovery_time_sec", 0.0)), 4), "sec")
        add("runtime.scaling_slope", round(float(runtime.get("scaling_slope", 0.0)), 6), "ratio")

    return rows


def _build_markdown(report: Dict) -> str:
    latency = report["latency"]
    throughput = report["throughput"]
    cost = report["cost"]
    quality = report["quality"]

    lines = [
        "# Production Metrics Summary",
        "",
        f"- Generated UTC: {report['metadata']['generated_utc']}",
        f"- Full system variant: {report['metadata']['full_variant']}",
        f"- Claims in evaluation split: {report['metadata']['claims_in_split']}",
        "",
        "## Latency & Throughput",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline avg latency (sec) | {latency['baseline_avg_sec']:.2f} |",
        f"| Debate avg latency (sec) | {latency['debate_avg_sec']:.2f} |",
        f"| Debate / Baseline latency ratio | {latency['debate_over_baseline_ratio']:.2f}x |",
        f"| Baseline throughput (claims/hour) | {throughput['baseline_claims_per_hour']:.2f} |",
        f"| Debate throughput (claims/hour) | {throughput['debate_claims_per_hour']:.2f} |",
        "",
        "## Cost",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Monthly claims (assumed) | {cost['claims_per_month']} |",
        f"| Monthly cost without cache (USD) | {cost['monthly_usd_no_cache']:.2f} |",
        f"| Monthly cost with cache (USD) | {cost['monthly_usd_with_cache']:.2f} |",
        f"| Monthly savings (USD) | {cost['monthly_savings_usd']:.2f} |",
        f"| Monthly savings (%) | {cost['monthly_savings_pct'] * 100:.2f}% |",
        "",
        "## Quality & Error",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {quality['accuracy']:.3f} |",
        f"| Error rate | {quality['error_rate']:.3f} |",
        f"| Expected errors / 100 claims | {quality['expected_errors_per_100_claims']:.2f} |",
        f"| Macro F1 | {quality['macro_f1']:.3f} |",
        f"| Calibration error | {quality['calibration_error']:.3f} |",
        f"| ECE | {quality['ece']:.3f} |",
        "",
        "## Assumptions",
        "",
        f"- Baseline latency assumption: {latency['assumptions']['baseline_latency_sec']} sec/claim",
        f"- Debate latency assumption: {latency['assumptions']['debate_latency_sec']} sec/claim",
        f"- Cost per search query: ${cost['assumptions']['serpapi_cost_per_query']:.4f}",
        f"- Calls/claim without cache: {cost['assumptions']['baseline_calls_per_claim']}",
        f"- Calls/claim with cache: {cost['assumptions']['cached_calls_per_claim']}",
    ]

    if report.get("runtime"):
        runtime = report["runtime"]
        lines.extend([
            "",
            "## Runtime Telemetry",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Cache hit rate | {runtime.get('cache_hit_rate', 0.0) * 100:.2f}% |",
            f"| Cache miss rate | {runtime.get('cache_miss_rate', 0.0) * 100:.2f}% |",
            f"| Mean CPU utilization | {runtime.get('cpu_utilization_mean_pct', 0.0):.2f}% |",
            f"| Peak CPU utilization | {runtime.get('cpu_utilization_peak_pct', 0.0):.2f}% |",
            f"| Mean memory usage | {runtime.get('memory_usage_mean_mb', 0.0):.2f} MB |",
            f"| Peak memory usage | {runtime.get('memory_usage_peak_mb', 0.0):.2f} MB |",
            f"| Concurrent requests tested | {runtime.get('concurrent_requests_tested', 0)} |",
            f"| P95 latency under load | {runtime.get('p95_latency_sec', 0.0):.2f} sec |",
            f"| Failure recovery rate | {runtime.get('failure_recovery_rate', 0.0) * 100:.2f}% |",
            f"| Median recovery time | {runtime.get('median_recovery_time_sec', 0.0):.2f} sec |",
            f"| Scaling slope | {runtime.get('scaling_slope', 0.0):.3f} |",
        ])

    return "\n".join(lines)


def _load_telemetry(path_str: str | None) -> Dict:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    args = _parse_args()

    ablation_report = _load_json(Path(args.ablation_report))
    comparative_report = _load_json(Path(args.comparative_report))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_variant = args.full_variant
    full_metrics = ablation_report.get("variants", {}).get(full_variant, {}).get("metrics", {})
    comparative_metrics = comparative_report.get("system_metrics", {}).get(full_variant, {})
    claims_in_split = int(ablation_report.get("metadata", {}).get("test_claim_count", 0))
    telemetry = _load_telemetry(args.telemetry_json)

    accuracy = float(full_metrics.get("overall_accuracy", 0.0))
    macro_f1 = float(full_metrics.get("macro", {}).get("f1", 0.0))
    calibration_error = float(comparative_metrics.get("calibration_error", 0.0))
    ece = float(comparative_metrics.get("ece", 0.0))
    error_rate = 1.0 - accuracy

    baseline_avg_sec = float(args.baseline_latency_sec)
    debate_avg_sec = float(args.debate_latency_sec)
    debate_over_baseline_ratio = _safe_div(debate_avg_sec, baseline_avg_sec)

    baseline_claims_per_hour = _safe_div(3600.0, baseline_avg_sec)
    debate_claims_per_hour = _safe_div(3600.0, debate_avg_sec)

    monthly_no_cache = (
        args.claims_per_month * args.baseline_calls_per_claim * args.serpapi_cost_per_query
    )
    monthly_with_cache = (
        args.claims_per_month * args.cached_calls_per_claim * args.serpapi_cost_per_query
    )
    monthly_savings_usd = monthly_no_cache - monthly_with_cache
    monthly_savings_pct = _safe_div(monthly_savings_usd, monthly_no_cache)

    report = {
        "metadata": {
            "generated_utc": datetime.utcnow().isoformat(),
            "full_variant": full_variant,
            "claims_in_split": claims_in_split,
            "source_reports": {
                "ablation_report": str(Path(args.ablation_report)),
                "comparative_report": str(Path(args.comparative_report)),
            },
        },
        "latency": {
            "baseline_avg_sec": baseline_avg_sec,
            "debate_avg_sec": debate_avg_sec,
            "debate_over_baseline_ratio": debate_over_baseline_ratio,
            "assumptions": {
                "baseline_latency_sec": baseline_avg_sec,
                "debate_latency_sec": debate_avg_sec,
            },
        },
        "throughput": {
            "baseline_claims_per_hour": baseline_claims_per_hour,
            "debate_claims_per_hour": debate_claims_per_hour,
        },
        "cost": {
            "claims_per_month": args.claims_per_month,
            "monthly_usd_no_cache": monthly_no_cache,
            "monthly_usd_with_cache": monthly_with_cache,
            "monthly_savings_usd": monthly_savings_usd,
            "monthly_savings_pct": monthly_savings_pct,
            "assumptions": {
                "serpapi_cost_per_query": args.serpapi_cost_per_query,
                "baseline_calls_per_claim": args.baseline_calls_per_claim,
                "cached_calls_per_claim": args.cached_calls_per_claim,
            },
        },
        "quality": {
            "accuracy": accuracy,
            "error_rate": error_rate,
            "expected_errors_per_100_claims": error_rate * 100.0,
            "macro_f1": macro_f1,
            "calibration_error": calibration_error,
            "ece": ece,
        },
        "runtime": telemetry.get("runtime", {}),
        "resilience": telemetry.get("resilience", {}),
        "scalability": telemetry.get("scalability", {}),
    }

    json_path = output_dir / "production_metrics_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = output_dir / "production_metrics_values.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "unit"])
        writer.writeheader()
        writer.writerows(_flatten_report_rows(report))

    md_path = output_dir / "production_metrics_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))

    print("Production metrics synthesis completed.")
    print(f"- JSON report: {json_path}")
    print(f"- CSV values: {csv_path}")
    print(f"- Markdown summary: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())