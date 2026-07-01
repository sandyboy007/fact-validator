"""
Run the publication-grade 5000-claim benchmark pipeline.

This helper executes the benchmark builder first and, if comparison inputs are
provided, runs the external-system comparison harness on the resulting test set.

Typical use:
  python Scripts/run_5000_benchmark_pipeline.py \
    --input fever=data/fever.json \
    --input liar=data/liar.csv \
    --input scifact=data/scifact.json \
    --input healthver=data/healthver.csv \
    --target-test-size 5000 \
    --comparison-input gpt-4o=results/gpt4o_predictions.csv \
    --comparison-input gemini=results/gemini_predictions.csv \
    --comparison-input claude=results/claude_predictions.csv \
    --comparison-input factool=results/factool_predictions.csv \
    --comparison-input fever_baseline=results/fever_baseline_predictions.csv \
    --comparison-input rag_baseline=results/rag_baseline_predictions.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 5000-claim benchmark pipeline")
    parser.add_argument("--input", action="append", required=True, help="Dataset input in the form name=path")
    parser.add_argument("--target-test-size", type=int, default=5000)
    parser.add_argument("--benchmark-output", default=str(REPO_ROOT / "data" / "benchmarks" / "results" / "large_benchmark_manifest.json"))
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "data" / "benchmarks" / "splits_5000"))
    parser.add_argument("--comparison-input", action="append", default=[], help="External comparison input in the form system=path")
    parser.add_argument("--comparison-output-dir", default=str(REPO_ROOT / "data" / "benchmarks" / "results"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    args = parse_args()

    build_cmd = [
        PYTHON,
        str(Path(__file__).resolve().with_name("build_large_test_benchmark.py")),
        "--target-test-size",
        str(args.target_test_size),
        "--output",
        args.benchmark_output,
        "--splits-dir",
        args.splits_dir,
        "--seed",
        str(args.seed),
    ]
    for spec in args.input:
        build_cmd.extend(["--input", spec])

    _run(build_cmd)

    if args.comparison_input:
        compare_cmd = [
            PYTHON,
            str(Path(__file__).resolve().with_name("run_external_system_comparison.py")),
            "--output-dir",
            args.comparison_output_dir,
        ]
        for spec in args.comparison_input:
            compare_cmd.extend(["--input", spec])
        _run(compare_cmd)

    print("5000-claim pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())