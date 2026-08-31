# analysis/

EDA and dataset analysis for Round 3. To be produced by the assigned agent
(do not duplicate work already here).

Planned contents:
- `eda_train_test.md` — the train/test EDA is mostly DONE in Round 2 (data is
  byte-identical; see `EXPERIMENT_LOOP.md` data facts). Only re-verify, don't
  redo.
- `eda_smile_r3.md` — NEW: the 5,973,369-SMILES auxiliary dataset (validity,
  chemical space, size distribution, polymer-vs-molecule composition, similarity
  to train/test/PI1M, subset selection strategies for pretraining).
- `invariance_audit.md` — canonical vs randomized SMILES stability of the final
  pipeline (Round 3 theme).
- `explainability_audit.md` — SHAP/permutation attribution per target (Round 3
  theme).
