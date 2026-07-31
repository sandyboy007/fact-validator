# Repository audit response

Audit date: 2026-07-29

## Finding

The external report inspected the older `origin/main` snapshot
(`8923cf71d94ca6cba17cd4202ff32cf3d9ea7a13`). The thesis reproducibility
artifacts have since been merged into `main`. The final audit was executed
from clean tracked source commit
`a6bbafc9d61e6fc002707ef16190491e4d2b973e`.

The thesis previously failed to make that branch boundary sufficiently clear
and incorrectly named a proposed `thesis-v1.0.0` tag that was never created.
The final source now names the exact branch and commits and explicitly states
that no release tag exists.

## Claim-by-claim verification

| Reported concern | Verified result |
|---|---|
| `data/benchmarks/results_5000/reproducibility_audit_report.json` does not exist | False on current `main`; true on the older `main` snapshot inspected by the report. |
| `run_thesis_statistics.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/run_thesis_statistics.py`. |
| `validate_thesis_artifacts.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/validate_thesis_artifacts.py`. |
| `build_thesis_run_manifest.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/build_thesis_run_manifest.py`. |
| `validate_cache.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/validate_cache.py`. |
| `run_operational_projection.py` does not exist | False on the thesis branch. It is tracked at `services/api/Scripts/run_operational_projection.py`. |
| `requirements.in` and `requirements.lock` do not exist | False on the thesis branch. Both are tracked under `services/api/`. |
| `thesis-v1.0.0` exists | False. No such tag exists. The thesis claim was removed. |
| The suite has only 162 tests | True for the older `main` snapshot; false for current `main`. Commit `72ad99c` adds one integration test for the persisted recent-results display, producing 163 tests in the final repository. |
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
- Final clean source snapshot executed by the reproducibility audit:
  `a6bbafc9d61e6fc002707ef16190491e4d2b973e`

The corrected thesis and its supporting evidence are now published together
on `main`.
