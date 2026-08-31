# P5A-111 — P5A-111-v57-tg-char-huber

**Focus:** tg char arm with Huber loss + OOF alpha (fat-tail robust)
**Arms:** char_huber
**Gate:** tg +0.005

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run_final.sh P5A-111   (or all: bash run_final.sh)
Expected wall time on Mac: ~1.5-4 h (PI1M SVD + ~40 models + arm).

