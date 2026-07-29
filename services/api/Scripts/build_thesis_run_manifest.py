"""Build the immutable-input manifest for the 5,000-claim proxy experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "benchmarks" / "splits_5000"
RESULT_DIR = REPO_ROOT / "data" / "benchmarks" / "results_5000"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    inputs = {
        "train": SPLIT_DIR / "train.json",
        "validation": SPLIT_DIR / "val.json",
        "test": SPLIT_DIR / "test.json",
    }
    predictions = {
        "ablation_predictions": RESULT_DIR / "ablation_study_predictions.csv",
        "baseline_predictions": RESULT_DIR / "baseline_comparison_predictions.csv",
    }
    for path in [*inputs.values(), *predictions.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "experiment_name": "factvalidator_proxy_5000",
        "evaluation_type": "deterministic_proxy",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "train_claims": 11986,
        "validation_claims": 2997,
        "test_claims": 5000,
        "proxy_components": {
            "lexical_model": True,
            "category_priors": True,
            "heuristic_semantic_signals": True,
            "deterministic_debate_rule": True,
            "quality_filter": True,
        },
        "live_components_not_executed": [
            "SerpAPI retrieval",
            "live domain credibility scoring",
            "SentenceTransformer reranking",
            "Ollama debate",
        ],
        "input_files": {
            key: str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            for key, path in inputs.items()
        },
        "prediction_files": {
            key: str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            for key, path in predictions.items()
        },
        "sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _sha256(path)
            for path in [*inputs.values(), *predictions.values()]
        },
    }
    output = RESULT_DIR / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
