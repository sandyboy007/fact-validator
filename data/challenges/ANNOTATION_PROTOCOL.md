# Conflict And Undercut Annotation Protocol

## Purpose

This protocol defines labels for a future double-annotated challenge set. The included `conflict_undercut_v1_seed.json` is synthetic test data only; it must not be reported as a benchmark result.

## Unit Of Annotation

Annotate one atomic claim and one exact evidence passage. Preserve the source URL, publication date when available, passage offsets, content hash, and retrieval-manifest identifier.

## Relation Labels

| Label | Definition |
|---|---|
| `SUPPORTS` | The passage directly entails the claim at its stated scope, time, quantity, and modality. |
| `REFUTES` | The passage directly contradicts the claim. |
| `UNDERCUTS` | The passage challenges an inference, scope, applicability, reliability, or completeness of support without necessarily asserting the claim's opposite. |
| `NEUTRAL` | The passage does not establish a support, refutation, or undercut relation. |
| `MISSING_EVIDENCE` | No usable passage was retrieved or preserved. |

## Verdict Labels

Assign `SUPPORTED` or `REFUTED` only when a grounded decisive relation exists. Assign `CONFLICTING` when comparable support and attack relations remain unresolved. Assign `NEI` for missing, neutral, or insufficiently grounded evidence.

## Quality Controls

- Use two independent annotators for at least 20 percent of the dataset.
- Record disagreements and adjudication rationale.
- Report Cohen's $\kappa$ or Krippendorff's $\alpha$ by relation type.
- Hold out a test set before tuning thresholds or prompts.
- Include retrieval-corruption variants: removal, distractor injection, duplication, contradiction, context truncation, and temporal substitution.