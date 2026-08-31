# P5A-008 — P5A-008-composite-levers

**Focus:** all levers combined (tg robust, eps residual, ei/egb overlays, egc zoo) - kitchen-sink candidate

**Mechanism:** built on the shared core (exp_core.py): stage-1 per-target ensemble
(LGBM/XGB/ExtraTrees, grouped 5-fold OOF, best-by-OOF), stage-2 cross-property partner
features (exact train partner labels where the same SMILES has another property, else
stage-1 predictions), physics identity overlays (eps=nc^2+ionic, egb~egc, ei~egc+eea)
with alphas fitted on OOF only, final clip to train bounds.

**Experiment-specific config:** {"tg_huber":true,"tg_reweight":true,"tg_median":true,"eps_residual":true,"ei_identity_max_alpha":0.8,"egb_egc_max_alpha":0.8,"egc_zoo":true}

**Gate:** mean oracle R2 > 0.9024 (V57 incumbent) with no target < -0.003

**Run:** bash ../run.sh P5A-008   (or all: bash ../run.sh)
**Score:** frozen predictions are scored post-hoc against Oracle/final_oracle.csv by run.sh
(mean of 7 per-target R2; incumbent V57 = 0.9024; target 0.9350; est private = mean - 0.011).
