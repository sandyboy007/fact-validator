# Thesis Statistics Summary

The 5,000-claim experiment evaluates the deterministic FactValidator-Proxy,
not the live SerpAPI/SentenceTransformer/Ollama application pipeline.

## System metrics

| System | Accuracy | Macro-F1 | Wilson 95% CI |
|---|---:|---:|---:|
| ablate_debate | 0.5134 | 0.4427 | [0.4995, 0.5272] |
| tune_fever | 0.5098 | 0.4421 | [0.4959, 0.5236] |
| ablate_quality_filter | 0.5086 | 0.4387 | [0.4947, 0.5224] |
| full_proxy | 0.5082 | 0.4384 | [0.4943, 0.5220] |
| ablate_semantic_rerank | 0.5068 | 0.4371 | [0.4929, 0.5206] |
| ablate_credibility | 0.4974 | 0.4393 | [0.4835, 0.5113] |
| length | 0.4942 | 0.3483 | [0.4804, 0.5081] |
| majority | 0.4926 | 0.2200 | [0.4788, 0.5065] |
| random | 0.3320 | 0.3240 | [0.3191, 0.3452] |
| sentiment | 0.2378 | 0.1400 | [0.2262, 0.2498] |
| keyword | 0.2354 | 0.1364 | [0.2238, 0.2474] |

## Selected exact paired tests

| Comparison | Full wins | Full losses | Exact two-sided p | Holm p | Matched OR |
|---|---:|---:|---:|---:|---:|
| full_proxy vs majority | 765 | 687 | 0.04327 | 0.21636 | 1.113 |
| full_proxy vs length | 634 | 564 | 0.04616 | 0.21636 | 1.124 |
| full_proxy vs ablate_debate | 34 | 60 | 0.00955 | 0.05729 | 0.570 |

No selected comparison is statistically significant after Holm correction.
The no-debate proxy has the best observed point estimates, but its comparison
with the full proxy has Holm-adjusted p approximately 0.0573. The result is
descriptive evidence against always-on proxy debate, not confirmatory proof.

All tests are descriptive and exploratory because proxy development was
informed by observations on this benchmark and normalized split overlaps
were identified retrospectively. No confirmatory superiority claim is made.

The confidence output is reported only as a raw-score calibration diagnostic.
A proper multiclass Brier score requires prob_supported, prob_refuted, and
prob_nei for every prediction.
