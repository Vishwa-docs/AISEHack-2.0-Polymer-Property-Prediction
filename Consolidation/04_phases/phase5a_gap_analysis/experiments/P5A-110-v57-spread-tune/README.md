# P5A-110 — P5A-110-v57-spread-tune

**Focus:** ei/eea spread scale fitted on OOF (replaces fixed 1.05)
**Arms:** spread_tune
**Gate:** ei/eea +0.005

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run_final.sh P5A-110   (or all: bash run_final.sh)
Expected wall time on Mac: ~1.5-4 h (PI1M SVD + ~40 models + arm).

