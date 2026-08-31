# P5A-101 — P5A-101-v57-tg-kriging

**Focus:** tg Tanimoto kNN residual kriging arm (alpha on C282 OOF)
**Arms:** kriging
**Gate:** tg +0.005 or kill

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run_final.sh P5A-101   (or all: bash run_final.sh)
Expected wall time on Mac: ~1.5-4 h (PI1M SVD + ~40 models + arm).

