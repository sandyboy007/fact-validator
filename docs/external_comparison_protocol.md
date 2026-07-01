# External Model Comparison Protocol

This protocol defines how to compare Fact Validator against external systems and baselines in a review-safe way.

## Target Systems

Use real prediction dumps for the following systems when available:

1. GPT-4o
2. Gemini
3. Claude
4. FacTool
5. FEVER baseline
6. RAG baseline

## Dataset Scale

For the publication-grade benchmark, use a minimum of 5000 claims in the test split if source data permits. The previous 51-claim split is only a pilot result and should not be used in the final IEEE submission.

## Rules

1. Do not mix predictions from different benchmark splits.
2. Keep a claim-level join key so all systems are evaluated on the exact same claims.
3. Record the prompt, retrieval settings, and temperature used for each external system.
4. Store model outputs in a frozen artifact before computing summary metrics.
5. Report accuracy, macro-F1, calibration error, and significance tests.
6. If a system cannot be run in the current environment, use an exported prediction dump and cite the generation settings in the paper.

## Output Files

The comparison harness writes three artifacts:

1. JSON report
2. CSV ranking
3. Markdown summary
