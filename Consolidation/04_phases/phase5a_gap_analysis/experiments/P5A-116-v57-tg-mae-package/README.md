# P5A-116 — P5A-116-v57-tg-mae-package

**Focus:** tg MAE package: MAE-tuned HGB + Huber GBM + Huber char + kriging (cuts Tg MAE, fat-tail robust)
**Arms:** mae_tg,tg_gbm,char_huber,kriging
**Gate:** tg MAE down + tg R2 not below -0.003

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run.sh P5A-116   (routed to GPU laptop automatically when available)

