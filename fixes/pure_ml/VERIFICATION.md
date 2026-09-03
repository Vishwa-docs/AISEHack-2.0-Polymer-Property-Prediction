# Pure-ML baseline verification

Verified on 2026-09-02 from `outputs_full/` without retraining.

| Check | Result |
|---|---|
| Evaluation method | grouped canonical-structure CV, as documented by this workstream |
| Mean target-wise R² | 0.816344 |
| Submission rows | 4,940 |
| Submission schema | `id`, `target` |
| Duplicate ids | 0 |
| Finite prediction values | yes |
| Serialized model bundle | `outputs_full/pure_ml_models.joblib` present |

This is a reproducible ExtraTrees baseline, not a replacement for the submitted
competition ensemble. Do not compare this score to a different panel or promote
this folder into the public codebase without a separate decision.
