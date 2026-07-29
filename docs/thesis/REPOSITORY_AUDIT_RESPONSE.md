# Repository audit response

Audit date: 2026-07-29

## Finding

The external report inspected the older `origin/main` snapshot
(`8923cf71d94ca6cba17cd4202ff32cf3d9ea7a13`). The thesis reproducibility
artifacts are tracked and pushed on `thesis/reproducibility-corrections`.
Commit `95c514c578f940f84dab7381263d2117555a644f` contains the complete set.

The thesis previously failed to make that branch boundary sufficiently clear
and incorrectly named a proposed `thesis-v1.0.0` tag that was never created.
The final source now names the exact branch and commits and explicitly states
that no release tag exists.

## Claim-by-claim verification

| Reported concern | Verified result |
|---|---|
| `data/benchmarks/results_5000/reproducibility_audit_report.json` does not exist | False on the thesis branch; true on the older `main` snapshot. The file is tracked in commit `95c514c`. |
| `run_thesis_statistics.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/run_thesis_statistics.py`. |
| `validate_thesis_artifacts.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/validate_thesis_artifacts.py`. |
| `build_thesis_run_manifest.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/build_thesis_run_manifest.py`. |
| `validate_cache.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/validate_cache.py`. |
| `run_operational_projection.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/run_operational_projection.py`. |
| `requirements.in` and `requirements.lock` do not exist | False on the thesis branch. Both are tracked under `services/api/`. |
| `thesis-v1.0.0` exists | False. No such tag exists. The thesis claim was removed. |
| The suite has only 162 tests | True for the older `main` snapshot; false for the thesis branch. Commit `72ad99c` adds one integration test for the persisted recent-results display, producing 163 tests on the thesis branch. |
| The suite produces 126 warnings | Not reproduced in the pinned Python 3.10 thesis environment. A fresh thesis-branch run produced 163 passed and three warning records. Warning totals can differ under another dependency environment. |

## Commands re-executed

```powershell
python -m pytest services/api/tests -q
python services/api/Scripts/run_thesis_statistics.py
python services/api/Scripts/validate_thesis_artifacts.py
```

Observed results:

- 163 tests passed and three warning records.
- Statistical regeneration completed without changing committed statistical
  outputs.
- Artifact validation passed for 11 systems with 5,000 aligned predictions
  each, disjoint splits, manifest hashes, and generated reports.

## Repository snapshots

- Older main snapshot inspected by the report:
  `8923cf71d94ca6cba17cd4202ff32cf3d9ea7a13`
- Statistics and reproducibility implementation:
  `4612740da4682542fc1772fdad873ff10fd7ade1`
- Commit containing the generated audit report:
  `95c514c578f940f84dab7381263d2117555a644f`

The corrected thesis avoids describing branch-specific evidence as though it
were already present on `main`.
