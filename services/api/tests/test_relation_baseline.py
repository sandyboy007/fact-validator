import importlib.util
import json
import sys
from pathlib import Path

from app.relation_classifier import classify_relation


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "Scripts" / "run_relation_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_relation_baseline", SCRIPT_PATH)
assert SPEC and SPEC.loader
relation_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relation_baseline)


def test_relation_classifier_identifies_fallback_when_nli_is_disabled(monkeypatch):
    monkeypatch.setenv("FACTVALIDATOR_NLI_ENABLED", "false")
    relation, metadata = classify_relation(
        "The Earth is flat.",
        "Satellite imagery confirms that the Earth is a sphere and the flat-Earth claim is false.",
    )

    assert relation == "refute"
    assert metadata["method"] == "heuristic-fallback"
    assert metadata["enabled"] is False


def test_relation_baseline_command_writes_versioned_report(tmp_path, monkeypatch):
    input_path = tmp_path / "relations.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "id": "support-1",
                    "claim": "The Earth is round.",
                    "passage": "Evidence confirms the Earth is round.",
                    "label": "support",
                },
                {
                    "id": "refute-1",
                    "claim": "The Earth is flat.",
                    "passage": "The flat-Earth claim is false.",
                    "label": "refute",
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run_relation_baseline.py", "--input", str(input_path), "--output", str(output_path)])

    assert relation_baseline.main() == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["version"] == "relation-baseline-report-v1"
    assert report["items_evaluated"] == 2
    assert "macro_f1" in report["metrics"]
    assert all(item["classifier"]["method"] == "heuristic-fallback" for item in report["predictions"])