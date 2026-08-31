# P5A-123 — P5A-123-v57-tg-aug-char

**Focus:** tg char residual model with x8 random-SMILES augmentation (train-only)
**Arms:** tg_aug_char
**Gate:** tg +0.005

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run.sh P5A-123   (routed to GPU laptop automatically when available)

