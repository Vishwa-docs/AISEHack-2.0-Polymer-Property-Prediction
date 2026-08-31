# P5A-113 — P5A-113-v57-weak-augment

**Focus:** ei/eea/nc/eps residual models with x8 random-SMILES augmentation (train-only)
**Arms:** weak_aug
**Gate:** weak targets +0.005 each

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run_final.sh P5A-113   (or all: bash run_final.sh)
Expected wall time on Mac: ~1.5-4 h (PI1M SVD + ~40 models + arm).

