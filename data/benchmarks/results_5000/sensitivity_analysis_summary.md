# Exact-Overlap Sensitivity Analysis

This exploratory robustness check removes the 39 test claims with exact
normalized matches in train or validation, retaining 4,961 claims.
It does not remove all likely near-duplicates and does not restore
confirmatory independence after test-guided proxy development.

| System | Full n | Full accuracy | Full macro-F1 | Filtered n | Filtered accuracy | Filtered macro-F1 | Accuracy change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ablate_credibility | 5000 | 0.4974 | 0.4393 | 4961 | 0.4967 | 0.4368 | -0.073 |
| ablate_debate | 5000 | 0.5134 | 0.4427 | 4961 | 0.5130 | 0.4402 | -0.040 |
| ablate_quality_filter | 5000 | 0.5086 | 0.4387 | 4961 | 0.5082 | 0.4362 | -0.044 |
| ablate_semantic_rerank | 5000 | 0.5068 | 0.4371 | 4961 | 0.5063 | 0.4345 | -0.045 |
| full_proxy | 5000 | 0.5082 | 0.4384 | 4961 | 0.5078 | 0.4359 | -0.044 |
| keyword | 5000 | 0.2354 | 0.1364 | 4961 | 0.2348 | 0.1362 | -0.057 |
| length | 5000 | 0.4942 | 0.3483 | 4961 | 0.4961 | 0.3486 | +0.187 |
| majority | 5000 | 0.4926 | 0.2200 | 4961 | 0.4945 | 0.2206 | +0.186 |
| random | 5000 | 0.3320 | 0.3240 | 4961 | 0.3312 | 0.3230 | -0.082 |
| sentiment | 5000 | 0.2378 | 0.1400 | 4961 | 0.2375 | 0.1399 | -0.035 |
| tune_fever | 5000 | 0.5098 | 0.4421 | 4961 | 0.5094 | 0.4397 | -0.043 |
