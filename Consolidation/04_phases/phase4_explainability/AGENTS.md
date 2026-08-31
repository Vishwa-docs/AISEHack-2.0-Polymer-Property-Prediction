# AGENTS.md — Phase 4: Explainability, Robustness & Generalization
## AISEHack 2.0 Polymer Property Prediction · Round 3

> **Read this file first. Then read REQUIREMENTS.md, then PLAN.md, then PROMPT.md.**
> PROMPT.md is the executable instruction set. This file is the operating contract.

---

## 1. Your Mission

You are a coding agent executing Phase 4 of the AISEHack 2.0 Round 3 challenge.
**Do NOT improve the prediction score.** That is handled in a separate track.

Your mission is to produce rigorous, visual, and quantitative evidence that the
existing best model is:

1. **Explainable** — its predictions are driven by chemically meaningful features
2. **Polymer-invariant** — equivalent SMILES representations give consistent predictions AND consistent explanations
3. **Methodologically sound** — validated beyond random CV, with honest degradation curves
4. **Reliable** — predictions come with calibrated uncertainty intervals
5. **Generalizable** — performance is proven on held-out structural families and external data

All evidence takes the form of plots, tables, and JSON summaries written to
`Phase4_Round3_Explainability/outputs/`. The agent that grades you reads
REQUIREMENTS.md and checks whether every artifact listed there exists and
passes its stated criterion.

---

## 2. What You Are Analyzing (the existing model)

The best current model lives at:
```
../final_submissions/v57_reproduction_standalone.py
../final_submissions/submission.csv          (4,940-row test predictions)
```

Verified oracle mean R² = **0.9024** (private LB calibrated to ~0.891).

The pipeline is a 339-node DAG ensemble with these components:
- **RDKit descriptors** (200+ physicochemical features)
- **Morgan count fingerprints** (radius 2/3, 512/768/1024 bits)
- **Character n-gram TF-IDF** on SMILES strings
- **PI1M SVD** (unsupervised on 995,799 official polymer SMILES)
- **Physics overlays**: `eps = nc² + ionic`, `ei ≈ egc + eea` as covariate routes
- **Polymer Genome atomic-triple fingerprint**
- **Tanimoto KRR**, Ridge, ExtraTrees, HistGradientBoosting, LightGBM per target
- **OOF NNLS blend** across arms

To run explainability analysis you do NOT re-run the full ~2.5-hour training pipeline.
Instead you:
1. Re-train **lightweight proxy models** (Ridge + LightGBM + ExtraTrees, same features,
   same seeds, 5-fold CV) solely to produce OOF predictions and SHAP values.
2. Use the final `submission.csv` predictions directly for test-side analyses
   (invariance tests, AD, conformal calibration on a split of train data).

**Why proxies?** The full V57 pipeline takes 2.5 hours and ~60 GB RAM. The proxy
models capture the same feature space and reproduce ~98% of the predictive signal,
which is sufficient for explainability and invariance analysis.

---

## 3. Inputs You Have Access To

| Path (relative to this folder) | Description |
|---|---|
| `../Dataset/train.csv` | 7,409 rows, `smiles, target, target_type` |
| `../Dataset/test.csv` | 4,940 rows, `id, smiles, target_type` |
| `../Dataset/PI1M.csv` | 995,799 unlabeled polymer SMILES (for SVD features only) |
| `../Dataset/smile_r3.csv` | 5,973,369 unlabeled molecular SMILES (optional, large) |
| `../final_submissions/submission.csv` | Final test predictions (id, target) |
| `../Oracle/final_oracle.csv` | **POST-FREEZE ONLY** — for R4.2 Khazana verification |

**Never read from `../Oracle/` during any training, featurization, or prediction step.**
Oracle data is only permitted for computing post-freeze evaluation scores in R4.2.

---

## 4. Seven Targets

| Code | Property | Train rows | Test rows | Notes |
|---|---|---|---|---|
| `tg` | Glass transition temperature (K) | 4,143 | 2,763 | Experimental, noisy |
| `egc` | Chain bandgap (eV) | 2,028 | 1,352 | DFT/Khazana |
| `egb` | Bulk bandgap (eV) | 337 | 224 | DFT/Khazana |
| `ei` | Ionisation energy (eV) | 222 | 148 | DFT/Khazana |
| `eea` | Electron affinity (eV) | 221 | 147 | DFT/Khazana |
| `nc` | Refractive index | 229 | 153 | DFT/Khazana |
| `eps` | Dielectric constant | 229 | 153 | DFT/Khazana |

The small-data targets (egb, ei, eea, nc, eps — ≤337 train rows each) need
extra care in explainability: use LOO-style CV or 3-fold instead of 5-fold.

---

## 5. Hard Rules (Same As Parent AGENTS.md)

### 5.1 Data rules
- **Only official competition data at runtime**: `train.csv`, `test.csv`, `PI1M.csv`.
  `smile_r3.csv` is allowed but not required (it is large; skip if it slows you down).
- **No external datasets, no pretrained weights, no external vocabularies.**
  Every embedding, SVD, or vocabulary must be fit from scratch inside this run.
- `Oracle/` is read-only post-freeze for R4.2 only.

### 5.2 Output rules
- Every artifact goes in `Phase4_Round3_Explainability/outputs/`.
- File naming must match REQUIREMENTS.md exactly (the grader checks filenames).
- Every plot must have axis labels, a title, and a legend if there are multiple series.
- Every CSV must have a header row.
- All plots: `dpi=150`, `bbox_inches='tight'`, saved as PNG.
- Color scheme: use matplotlib's `tab10` or `viridis`; be accessible (no pure red/green pairs).

### 5.3 Reproducibility
- Fix all random seeds: `numpy.random.seed(42)`, `random.seed(42)`, all sklearn/lgbm/xgb seeds = 42.
- Every script must run end-to-end without manual steps.
- Python version: 3.10+. Libraries available: rdkit, shap, lightgbm, scikit-learn, numpy, pandas, matplotlib, seaborn, scipy.

### 5.4 Oracle compliance scan
Before any output is finalized, grep the analysis scripts for:
`oracle|ORACLE|Oracle|sources/|final_oracle|oracle_proxy`
These strings must appear **only** in the clearly demarcated R4.2 Khazana block.
Everywhere else they must be absent.

### 5.5 Scope
- **Do NOT modify `../final_submissions/` in any way.**
- **Do NOT retrain the full V57 pipeline.** Use proxy models as described in §2.
- **Do NOT add features not already in V57** (no new external data sources).
- Any new code files go inside `Phase4_Round3_Explainability/` only.

---

## 6. Proxy Model Specification

For all analysis that requires OOF predictions or SHAP values, use these exact specs:

```python
FEATURE_SETS = {
    "morgan":      Morgan count FP, radius=2, nBits=1024
    "rdkit_desc":  RDKit 200-descriptor block (same as V57 Stage A)
    "char_ngram":  CountVectorizer(ngram_range=(2,6), max_features=8192, analyzer='char')
                   on canonical SMILES
    "physics":     [nc_sq (=nc²), ionic (=eps−nc²), egc_eea_sum (=egc+eea)] — only
                   available for rows where both source targets are in train
}

PROXY_MODELS_PER_TARGET = {
    "ridge":        Ridge(alpha=100, random_state=42)  on combined morgan + rdkit_desc
    "et":           ExtraTreesRegressor(n_estimators=200, random_state=42)
    "lgbm":         LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=42,
                    num_leaves=31, min_child_samples=5)
    "ensemble":     NNLS blend of ridge + et + lgbm OOF predictions
}

CV_STRATEGY = {
    # for large targets (tg=4143, egc=2028): StratifiedKFold(n_splits=5) on target quantiles
    # for small targets (egb≤337, ei, eea, nc, eps): KFold(n_splits=3)
    # group constraint: GroupKFold on canonical SMILES to prevent leakage from duplicates
}
```

These proxies are **fast** (total runtime < 15 minutes on a laptop). They are
used solely for analysis — never submitted to Kaggle.

---

## 7. Required Outputs Summary

All outputs go in `outputs/`. Full specifications are in REQUIREMENTS.md.
Quick reference:

### Explainability (R1)
```
outputs/shap_beeswarm_<target>.png          × 7
outputs/shap_summary_global.png
outputs/shap_top20_per_target.csv
outputs/local_shap_<target>_<polymer_id>.png  (2-3 per target = 14-21 files)
outputs/shap_force_<target>_<polymer_id>.png
outputs/fidelity_curve_<target>.png          × 7
outputs/fidelity_table.csv
outputs/explanation_agreement_heatmap.png
outputs/explanation_agreement.csv
outputs/physics_decomp_eps_shap.png
```

### Invariance (R2)
```
outputs/smiles_invariance_per_target.csv
outputs/smiles_invariance_boxplot.png
outputs/smiles_invariance_violation_rate.csv
outputs/canonicalization_check.txt
outputs/attribution_invariance_per_target.csv
outputs/attribution_invariance_scatter.png
outputs/oligomer_invariance.csv
outputs/oligomer_invariance_plot.png
```

### Reliability (R3)
```
outputs/cv_validation_table.csv
outputs/cv_validation_barplot.png
outputs/conformal_coverage_table.csv
outputs/conformal_calibration_plot.png
outputs/test_predictions_with_intervals.csv
outputs/error_vs_uncertainty_scatter_<target>.png  × 7
outputs/error_uncertainty_correlation.csv
outputs/ad_analysis_table.csv
outputs/ad_analysis_plot.png
outputs/ad_test_similarity.csv
outputs/seed_stability.csv
```

### Generalization (R4)
```
outputs/generalization_ladder.csv
outputs/generalization_ladder_plot.png
outputs/khazana_holdout_scores.csv
outputs/khazana_scatter_<target>.png  × 6 (DFT targets only)
outputs/tail_performance.csv
outputs/tail_performance_plot.png
```

### Summary
```
outputs/scorecard.md        (auto-generated: lists every artifact, its pass/fail status
                             against REQUIREMENTS.md criteria, and a one-line note)
outputs/trustworthiness_radar.png  (radar chart across 8 axes)
```

---

## 8. Session-End Checklist

After all scripts complete:

1. Verify every filename in §7 exists in `outputs/`.
2. Run `python check_outputs.py` (you will create this script; it reads REQUIREMENTS.md
   criteria and checks each artifact, prints PASS/FAIL per item).
3. Update `outputs/scorecard.md` with the actual pass/fail results.
4. Confirm no oracle strings leaked into any analysis script (grep check §5.4).
5. Confirm `../final_submissions/` is unmodified.
6. Write a brief session summary to `outputs/SESSION_SUMMARY.md`:
   - which artifacts passed, which failed
   - any surprising findings (e.g., attribution invariance was unexpectedly low for tg)
   - recommended follow-up

---

## 9. Grading Contract

A separate grading agent will:
1. Read `REQUIREMENTS.md` to get the list of artifacts and pass criteria.
2. Check `outputs/` for each file.
3. For plots: visually inspect axis labels, titles, data presence.
4. For CSVs: check that the required columns exist and values are numeric/reasonable.
5. For numerical criteria (e.g., conformal coverage within ±3%): read the CSV and verify.
6. Output a verdict: PASS / PARTIAL / FAIL per requirement, with reasoning.

You pass Phase 4 when the minimum viable set (R1.1, R1.2, R2.1, R2.3, R3.1,
R3.2, R4.1, R4.2) all receive PASS.
