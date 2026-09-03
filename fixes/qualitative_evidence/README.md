# Qualitative-evidence audit

The 904 run records a scorecard update that should be promoted only after the user reruns the isolated notebook and confirms the generated artifacts. This audit verifies the stored CSV evidence without inspecting images.

## Current audited findings from the recorded 904 outputs

| Theme | Audited calculation | Interpretation |
|---|---:|---|
| Representation invariance | existing graph/attribution tables | established by the recorded invariance suite |
| Explainability fidelity | existing intervention table | established separately from cross-model rank agreement |
| Cross-model rank agreement | mean Spearman **0.472223** | retain as the one explicit limitation |
| Conformal calibration | max absolute coverage deviation **0.023529** | satisfies the predeclared ±3% check in the archived table |
| Error–uncertainty association | ρ≥0.30 on **5/7** targets | satisfies the recorded scorecard threshold |
| Augmentation | invariance spread 6.932313 → 0.650983 at k=8; Tg OOF R² 0.900136 → 0.896014 | a real stability/accuracy trade-off, not a free gain |

## Before promotion

1. Rerun the isolated notebook.
2. Run `python validate_archive_evidence.py --outputs <new-output-directory>`.
3. Compare newly created CSVs with the recorded results; do not overwrite existing public evidence when a result differs.
4. Copy only verified CSV/PNG pairs and regenerate the public scorecard from the verified output directory.
