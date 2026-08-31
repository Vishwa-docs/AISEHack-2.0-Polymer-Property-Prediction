# P5A-001 — P5A-001-tg-robust-tail

**Focus:** Tg fat-tail attack: Huber objective + OOF-residual reweighting + median-of-models ensemble for tg

**Mechanism:** built on the shared core (exp_core.py): stage-1 per-target ensemble
(LGBM/XGB/ExtraTrees, grouped 5-fold OOF, best-by-OOF), stage-2 cross-property partner
features (exact train partner labels where the same SMILES has another property, else
stage-1 predictions), physics identity overlays (eps=nc^2+ionic, egb~egc, ei~egc+eea)
with alphas fitted on OOF only, final clip to train bounds.

**Experiment-specific config:** {"tg_huber":true,"tg_reweight":true,"tg_median":true}

**Gate:** tg oracle R2 >= baseline_tg + 0.005 (fold-consistent)

**Run:** bash ../run.sh P5A-001   (or all: bash ../run.sh)
**Score:** frozen predictions are scored post-hoc against Oracle/final_oracle.csv by run.sh
(mean of 7 per-target R2; incumbent V57 = 0.9024; target 0.9350; est private = mean - 0.011).
