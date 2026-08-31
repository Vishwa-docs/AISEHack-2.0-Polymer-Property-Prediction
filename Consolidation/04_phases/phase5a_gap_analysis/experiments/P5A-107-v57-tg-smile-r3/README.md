# P5A-107 — P5A-107-v57-tg-smile-r3

**Focus:** tg residual model on smile_r3-fitted char-SVD features (400k sample)
**Arms:** smiler3
**Gate:** tg +0.010

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run_final.sh P5A-107   (or all: bash run_final.sh)
Expected wall time on Mac: ~1.5-4 h (PI1M SVD + ~40 models + arm).

