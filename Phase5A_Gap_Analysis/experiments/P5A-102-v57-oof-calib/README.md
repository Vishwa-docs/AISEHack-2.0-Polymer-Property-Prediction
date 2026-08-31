# P5A-102 — P5A-102-v57-oof-calib

**Focus:** per-target linear calibration of base on OOF
**Arms:** calib
**Gate:** mean +0.002

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run_final.sh P5A-102   (or all: bash run_final.sh)
Expected wall time on Mac: ~1.5-4 h (PI1M SVD + ~40 models + arm).

