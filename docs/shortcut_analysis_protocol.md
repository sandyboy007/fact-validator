# Shortcut Analysis Protocol for n=500+ Claims

This protocol defines the data collection, annotation, perturbation, and analysis workflow for the expanded shortcut-sensitivity experiment.

## 1. Goal

Measure whether the full Fact Validator system relies on shallow shortcuts such as length, keyword presence, or sentiment cues, and whether those shortcuts collapse under paraphrastic perturbation.

This protocol is intended for IEEE-style publication. The primary benchmark must be genuine, provenance-tracked, and externally verifiable. Synthetic claims may only be used as a separate internal stress test, not as the main evidence base.

The final dataset should support three analyses:

1. Perturbation matrix: original versus paraphrased claims.
2. Feature-label correlation matrix: length, keyword frequency, sentiment versus label.
3. Length-stratified error analysis: short, medium, long tertiles.

## 2. Source Claims

Use a balanced pool of at least 500 genuine claims.

Preferred source mix:

1. FEVER for directly supported/refuted claims.
2. LIAR only if claims are revalidated, provenance-checked, and remapped to the project label space.
3. Optional additional public datasets only if the original citation, license, and label schema are preserved.

Recommended target mix:

1. 250 supported claims.
2. 250 refuted claims.

If a third class is retained, keep it separate and report it explicitly. Do not silently collapse ambiguous claims into supported or refuted.

For IEEE submission, include a data provenance table with dataset name, original label schema, source citation or URL, inclusion criteria, and any relabeling notes.

## 3. Collection Rules

1. Deduplicate claims before annotation.
2. Remove near-duplicates using token overlap and normalized-string similarity.
3. Keep only claims with a clear factual truth condition.
4. Exclude subjective opinions, predictions, and value judgments unless the manuscript explicitly treats them as NEI.
5. Store only the claim text, source metadata, and derived annotations needed for the experiment.
6. Preserve the original benchmark identifier for every claim so reviewers can audit provenance.
7. Keep the original genuine claim as the primary scientific unit; the paraphrase is only a paired robustness sample.

## 4. Annotation Protocol

Each claim should be annotated by three annotators.

Annotators assign one label from:

1. SUPPORTED
2. REFUTED
3. NEI

Operational guidance:

1. Label SUPPORTED only when the claim is directly backed by strong evidence.
2. Label REFUTED only when evidence directly contradicts the claim.
3. Label NEI when evidence is absent, mixed, or too weak to decide.
4. Require a short rationale for every difficult claim.
5. Record disagreement flags for adjudication.

Acceptance criteria:

1. Target Cohen/Fleiss agreement: at least moderate, preferably $\kappa \ge 0.60$.
2. Resolve disagreements through adjudication before final split creation.

## 5. Perturbation Pipeline

Generate a paired paraphrase for every original claim.

Perturbation rules:

1. Preserve truth value.
2. Change surface form: word order, syntax, synonym choice, and sentence length.
3. Avoid changing named entities, dates, quantities, or polarity.
4. Create one paraphrase per claim.
5. Manually sample-check a subset before full-scale use.

Recommended perturbation methods:

1. Local LLM rewrite with a strict preservation prompt.
2. Back-translation.
3. Rule-based surface edits as a fallback.

Perturbations must be reported separately from the main benchmark results. The original genuine claims are the primary dataset.

## 6. Split Policy

Use a stratified split after final annotation.

Recommended split:

1. 60% training.
2. 20% validation.
3. 20% test.

Stratify by:

1. label.
2. domain or category.
3. length bucket.
4. source family if available.

## 7. File Schema

Recommended CSV fields:

1. `id`
2. `claim_original`
3. `claim_perturbed`
4. `label`
5. `category`
6. `difficulty`
7. `source`
8. `annotator_1`
9. `annotator_2`
10. `annotator_3`
11. `adjudicated_label`
12. `notes`

## 8. Analysis Outputs

The analysis script should print or save three tables:

1. Perturbation matrix: full system, length heuristic, and keyword heuristic on original versus paraphrased claims.
2. Feature-label correlations: character count, token count, keyword frequency, and VADER sentiment versus label.
3. Length tertile analysis: full system versus length heuristic on short, medium, and long claims.

## 9. Reporting Guidance

1. Report the exact dataset source and annotation date.
2. Report the number of claims retained after cleaning.
3. Report agreement and adjudication rate.
4. State clearly whether the final evaluation uses the original, perturbed, or both claim sets.
5. If the 500+ dataset is not yet available, mark the tables as pending rather than estimating them from the smaller benchmark.
6. For IEEE, add a reproducibility statement describing whether the dataset is public, redistributable, or citation-only.
