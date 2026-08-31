# P5A-112 — P5A-112-v57-tg-mae

**Focus:** tg MAE-optimal residual model (absolute_error HGB), MAE-tuned alpha
**Arms:** mae_tg
**Gate:** tg MAE down, R2 not below -0.003

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run_final.sh P5A-112   (or all: bash run_final.sh)
Expected wall time on Mac: ~1.5-4 h (PI1M SVD + ~40 models + arm).

