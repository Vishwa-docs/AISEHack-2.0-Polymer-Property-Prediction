# P5A-124 — P5A-124-v57-final-combo-v2

**Focus:** ALL arms: best-possible candidate v2
**Arms:** kriging,calib,ei,eps,egb,char_tune,spread_tune,shrink,mae_tg,weak_aug,smiler3,nc_eps,mae_weak,tg_gbm,weak_stack,weak_kernel,tg_aug_char
**Gate:** mean > 0.907, no target < -0.003

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run.sh P5A-124   (routed to GPU laptop automatically when available)

