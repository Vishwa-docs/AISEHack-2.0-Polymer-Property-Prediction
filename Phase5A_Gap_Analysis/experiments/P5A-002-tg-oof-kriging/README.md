# P5A-002 — P5A-002-tg-oof-kriging

**Focus:** Tg OOF residual kriging: Tanimoto kNN correction of tg predictions, alpha fitted on train OOF only

**Mechanism:** built on the shared core (exp_core.py): stage-1 per-target ensemble
(LGBM/XGB/ExtraTrees, grouped 5-fold OOF, best-by-OOF), stage-2 cross-property partner
features (exact train partner labels where the same SMILES has another property, else
stage-1 predictions), physics identity overlays (eps=nc^2+ionic, egb~egc, ei~egc+eea)
with alphas fitted on OOF only, final clip to train bounds.

**Experiment-specific config:** {}

**Gate:** tg oracle R2 >= baseline_tg + 0.005

**Run:** bash ../run.sh P5A-002   (or all: bash ../run.sh)
**Score:** frozen predictions are scored post-hoc against Oracle/final_oracle.csv by run.sh
(mean of 7 per-target R2; incumbent V57 = 0.9024; target 0.9350; est private = mean - 0.011).
