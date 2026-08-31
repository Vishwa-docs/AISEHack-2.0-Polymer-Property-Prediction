# P5A-106 — P5A-106-v57-arms-core

**Focus:** core arms combined (kriging+calib+identities+char alpha tune)
**Arms:** kriging,calib,ei,eps,egb,char_tune
**Gate:** mean > 0.907

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run_final.sh P5A-106   (or all: bash run_final.sh)
Expected wall time on Mac: ~1.5-4 h (PI1M SVD + ~40 models + arm).

