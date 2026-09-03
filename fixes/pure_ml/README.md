# Standalone classical-ML baseline

This is a deliberately compact, reproducible baseline—not a replacement for the competition ensemble. It trains seven target-specific ExtraTrees regressors on RDKit descriptors plus Morgan fingerprints, evaluates grouped cross-validation by canonical polymer structure, writes one serialised model bundle, and produces a submission-format prediction CSV.

## Run

Use the isolated environment after it has been bootstrapped:

```bash
cd fixes/pure_ml
../isolated_runs/.venv/bin/python train.py \
  --data-dir ../isolated_runs/data \
  --output-dir outputs
```

Use `--smoke` first for a quick setup check. The output model is `outputs/pure_ml_models.joblib`; the validation report is `outputs/grouped_cv_metrics.csv`. A score from this baseline is comparable only when it uses the same target-wise metric and a clearly stated evaluation panel.
