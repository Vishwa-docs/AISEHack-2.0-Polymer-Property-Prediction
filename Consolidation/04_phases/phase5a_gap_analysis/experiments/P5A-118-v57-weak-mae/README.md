# P5A-118 — P5A-118-v57-weak-mae

**Focus:** ei/eea/nc/eps MAE-tuned residual models (alpha by OOF MAE)
**Arms:** mae_weak
**Gate:** MAE down, R2 not below -0.003 each

Built by scripts/patch_v57_arms.py from the pristine standalone copy (v57_pristine.py).
All arm alphas are fitted on the C282 OOF residuals (train-only; no answer panels).
Run: bash run.sh P5A-118   (routed to GPU laptop automatically when available)

