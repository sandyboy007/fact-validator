"""
Run full architecture comparison suite on a benchmark split.

This script chains:
1) run_baseline_comparison.py
2) run_ablation_study.py
3) run_comparative_analysis.py

Use this for consistent, reproducible architecture-vs-architecture reporting.

Examples:
  python Scripts/run_benchmark_architecture_suite.py \
    --train data/benchmarks/splits_224/train.json \
    --test data/benchmarks/splits_224/test.json \
    --output-dir data/benchmarks/results_224

  python Scripts/run_benchmark_architecture_suite.py \
    --train data/benchmarks/splits_5000/train.json \
    --test data/benchmarks/splits_5000/test.json \
    --output-dir data/benchmarks/results_5000
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
    parser = argparse.ArgumentParser(description="Run full benchmark architecture suite")
    parser.add_argument(
        "--train",
        default=str(REPO_ROOT / "data" / "benchmarks" / "splits" / "train.json"),
        help="Train split JSON path",
    )
    parser.add_argument(
        "--test",
        default=str(REPO_ROOT / "data" / "benchmarks" / "splits" / "test.json"),
        help="Test split JSON path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "benchmarks" / "results"),
        help="Directory for all generated reports",
    )
    parser.add_argument(
        "--full-variant",
        default="full_proxy",
        help="Ablation variant name used as the full-system anchor",
    )
    parser.add_argument(
        "--no-debate-variant",
        default="ablate_debate",
        help="Ablation variant used for debate-lift comparison",
    )
    return parser.parse_args()


def run_checked(command: list[str]) -> None:
    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    args = parse_args()

    scripts_dir = Path(__file__).resolve().parent
    train_path = str(Path(args.train).resolve())
    test_path = str(Path(args.test).resolve())
    output_dir = str(Path(args.output_dir).resolve())

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    baseline_cmd = [
        PYTHON,
        str(scripts_dir / "run_baseline_comparison.py"),
        "--train",
        train_path,
        "--test",
        test_path,
        "--output-dir",
        output_dir,
    ]

    ablation_cmd = [
        PYTHON,
        str(scripts_dir / "run_ablation_study.py"),
        "--train",
        train_path,
        "--test",
        test_path,
        "--output-dir",
        output_dir,
    ]

    comparative_cmd = [
        PYTHON,
        str(scripts_dir / "run_comparative_analysis.py"),
        "--baseline-csv",
        str(Path(output_dir) / "baseline_comparison_predictions.csv"),
        "--ablation-csv",
        str(Path(output_dir) / "ablation_study_predictions.csv"),
        "--output-dir",
        output_dir,
        "--full-variant",
        args.full_variant,
        "--no-debate-variant",
        args.no_debate_variant,
    ]

    run_checked(baseline_cmd)
    run_checked(ablation_cmd)
    run_checked(comparative_cmd)

    print("Architecture comparison suite completed.")
    print(f"- Summary: {Path(output_dir) / 'comparative_analysis_summary.md'}")
    print(f"- Ranking CSV: {Path(output_dir) / 'comparative_analysis_ranking.csv'}")
    print(f"- JSON report: {Path(output_dir) / 'comparative_analysis_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
