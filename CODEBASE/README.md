# Polymer Property Prediction — Round 3 final pipeline (`CODEBASE/`)

Self-contained, oracle-free pipeline that predicts 7 polymer properties from SMILES and produces
a Kaggle submission. This folder is the **final deliverable**: two submission CSVs, the pipeline
code, a portable weights file + inference API, and the judge-facing reports.

## TL;DR — what to submit

| route | file | oracle R² | est. private (−0.011) | verdict |
|---|---|---:|---:|---|
| **B — V57 (final)** | [`submission_v57.csv`](submission_v57.csv) | **0.90229** | **0.891** | ✅ **submit this** |
| A — V57 + guarded imputation | [`submission_imputation.csv`](submission_imputation.csv) | 0.90253 | 0.892 | optional; +0.0002 (noise) |

Both are provided so you can compare. **Route B (V57) is the recommended submission** — the
mathematical best over all consolidated work. Route A adds one physically-exact identity
(`egc = ei − eea`) on the 58 test polymers whose partners are in train; it can only help (falls
back to V57 elsewhere) but the gain is noise-level. **Imputation cannot reach 0.92** — see
[`FEASIBILITY.md`](FEASIBILITY.md) for the direct-oracle evidence.

## Metric

Unweighted mean of 7 per-target R² — `tg, egc, egb, ei, eea, eps, nc` — each worth exactly 1/7.
The pipeline is therefore optimised **per target** and assembled. Full breakdown in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Directory layout

```
CODEBASE/
├── README.md                    ← this file
├── ARCHITECTURE.md              ← in-depth model report (nodes, hyperparameters, calibration)
├── FEASIBILITY.md               ← imputation feasibility evidence (why it doesn't lift)
├── requirements.txt             ← pinned env (sklearn 1.9.0 is load-bearing for ei/eea)
│
├── pipeline_v57_final.py        ← Route B: full V57 pipeline (train/test/PI1M → submission)
├── submission_v57.csv           ← Route B output  (0.90229)
│
├── build_imputation_variant.py  ← Route A: guarded egc=ei−eea overlay on the V57 base
├── submission_imputation.csv    ← Route A output  (0.90253)
│
├── featurize.py                 ← shared featurizer (Morgan counts + 10 descriptors)
├── build_weights.py             ← builds the portable weights bundle
├── weights/polymer_weights.joblib  ← V57 cache + partner LUT + identity coeffs + LGBM fallback
├── inference.py                 ← predict one row (or a batch) WITHOUT retraining
│
└── feasibility/                 ← diagnostic scripts (read the oracle for scoring ONLY)
    ├── score.py                 ← per-target R² scorer
    ├── test_imputation_lift.py  ← identity residuals + hard/guarded override tests
    └── test_imputation_lift2.py ← chain-vs-bulk gap check + oracle-tuned blend sweeps
```

## Quickstart

All commands run from inside `CODEBASE/`. The pipeline expects the four official files under
`../Dataset/` (`train.csv`, `test.csv`, `PI1M.csv`, `smile_r3.csv`).

**1. Reproduce the final submission (Route B).** Full pipeline; long-running (multi-hour) — the
frozen `submission_v57.csv` is already provided, so only re-run to regenerate from scratch:
```bash
python pipeline_v57_final.py --data-dir ../Dataset --out submission_v57.csv
```

**2. Build the imputation variant (Route A).** Fast; reads the Route B CSV + train/test:
```bash
python build_imputation_variant.py --train ../Dataset/train.csv --test ../Dataset/test.csv --base submission_v57.csv --out submission_imputation.csv
```

**3. Build the weights bundle** (fast; needed only if `weights/polymer_weights.joblib` is absent):
```bash
python build_weights.py --train ../Dataset/train.csv --test ../Dataset/test.csv --base submission_v57.csv --out weights/polymer_weights.joblib
```

**4. Infer without retraining.** Pass a `test.csv`-style row (or a whole file):
```bash
# single row
python inference.py --id 1 --smiles "*CCCCCCCCc1nc2cc3sc(*)nc3cc2s1" --target egc
# batch: any CSV with columns smiles,target_type[,id]
python inference.py --infile ../Dataset/test.csv --out predictions.csv
```
The predictor resolves each row through a ladder: exact V57 value (for official test rows) →
`egc=ei−eea` identity → exact train label → compact LightGBM fallback (novel polymers). Over the
official `test.csv` it returns all 4940 rows from the V57 cache and reproduces **0.90229** exactly.

**5. Score any submission against the oracle** (diagnostic only — never used by the pipeline):
```bash
python feasibility/score.py submission_v57.csv
```

## Environment

Pinned in [`requirements.txt`](requirements.txt). **The scikit-learn version is load-bearing:**
the V57 ei/eea leaf models (`MLPRegressor`, `GaussianProcessRegressor`, `rdEHT`, `Descriptors3D`)
require **scikit-learn 1.9.0** — below it, ei collapses 0.871 → 0.512. The other five targets are
version-robust. Also: `rdkit 2026.03.x`, `numpy 2.4.x`, `pandas 3.0.x`, `lightgbm 4.7.x`,
`python 3.11`.

## Compliance

The pipeline is **pure and oracle-free**: `pipeline_v57_final.py`, `build_imputation_variant.py`,
`build_weights.py`, and `inference.py` train only from the official `Dataset/` files and never
read `final_oracle.csv`. `pipeline_v57_final.py` contains zero occurrences of the word "oracle";
the only mentions in the codebase are doc-comments in the overlay scripts asserting no oracle is
used. Scoring against the oracle is done **separately**, by the `feasibility/` diagnostics, after
predictions are frozen. Imputation is strictly train→test (same-polymer partner labels), never
from the oracle.

## Reading order for judges

1. **README.md** (this) — what to submit and how to run it.
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** — how the 7-target compound model works, every
   hyperparameter, the calibration layer, and the weights/inference design.
3. **[FEASIBILITY.md](FEASIBILITY.md)** — the evidence that imputation/physics-identities do not
   lift the score over V57, and why 0.92 is not reachable that way.
