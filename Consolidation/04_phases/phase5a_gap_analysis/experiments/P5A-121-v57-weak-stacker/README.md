# P5A-121 — P5A-121-v57-weak-stacker

**Focus:** weak-target stacker: Ridge on [partner preds, base] per target (imputation cascade)
**Arms:** weak_stack
**Gate:** weak targets +0.005 each

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run.sh P5A-121   (routed to GPU laptop automatically when available)

