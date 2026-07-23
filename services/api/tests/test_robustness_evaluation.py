from app.robustness_evaluation import build_scenarios, evaluate_case, selective_metrics


def _evidence(passage: str, url: str) -> dict:
    return {
        "url": url,
        "domain": "example.org",
        "base_domain": "example.org",
        "domain_score": 90,
        "passage": passage,
    }


def test_explicit_missing_evidence_corruption_forces_audited_abstention():
    item = {
        "id": "missing-1",
        "claim": "The study reported a 20 percent reduction in hospital admissions.",
        "expected_verdict": "SUPPORTED",
        "evidence": [_evidence("The study reported a 20 percent reduction in hospital admissions.", "https://example.org/report")],
        "corruptions": {"missing_evidence": []},
    }
    scenarios = build_scenarios(item)
    missing = evaluate_case(item, "missing_evidence", scenarios["missing_evidence"])

    assert set(scenarios) == {"clean", "missing_evidence"}
    assert missing["full_audited_graph"] == "NEI"
    assert any("No fetched evidence passage" in reason for reason in missing["audit"]["violations"])


def test_duplicate_corruption_is_reported_and_metrics_are_available():
    item = {
        "id": "duplicate-1",
        "claim": "The study reported a 20 percent reduction in hospital admissions.",
        "expected_verdict": "SUPPORTED",
        "evidence": [_evidence("The study reported a 20 percent reduction in hospital admissions.", "https://example.org/a")],
        "corruptions": {
            "duplicate_reporting": [
                _evidence("The study reported a 20 percent reduction in hospital admissions.", "https://example.org/a"),
                _evidence("The study reported a 20 percent reduction in hospital admissions.", "https://example.org/b"),
            ]
        },
    }
    row = evaluate_case(item, "duplicate_reporting", build_scenarios(item)["duplicate_reporting"])
    metrics = selective_metrics([row], "full_audited_graph")

    assert row["full_audited_graph"] == "HUMAN_REVIEW"
    assert any("correlated" in reason for reason in row["audit"]["violations"])
    assert set(metrics) == {"coverage", "selective_risk", "false_accept_rate"}