# Round 3 Final Report — AISEHack 2.0 Polymer Property Prediction

> Draft assembled 2026-08-31. Final numbers to be locked after the full
> validation run on the GPU laptop (correct env: numpy 2.4.6).

## 1. Final model

Single-file pipeline: **`CODEBASE/pipeline_final.py`** (also split as
`pipeline_v57_final.py` + `evidence_engine.py`).

- **Submission path (Part A)** — the V57 compound assembly (339-node DAG):
  per-target classical ensembles (RDKit descriptors + Morgan counts + char
  n-gram Ridge + Tanimoto KRR + PI1M SVD heads), target-specific portfolios,
  signed-residual splice/blend chain, physics overlays (`eps = nc² + ionic`,
  `ei ≈ egc + eea`), final calibration (char-residual for tg/egc/egb/nc/eps,
  spread-clip for ei/eea). Reads ONLY `train.csv` / `test.csv` /
  `PI1M.csv`; all models trained from scratch, fixed seeds.
- **Evidence path (Part B)** — Round-3 judging themes: explainability (R1),
  invariance (R2), reliability (R3), generalization (R4), plus data-augmentation
  and homologous-series experiments. Oracle-free.

## 2. Accuracy

| metric | value |
|---|---|
| Verified panel R² (3,818 rows) | **0.9035** |
| final_oracle R² (4,909 rows) | **0.9023** |
| Calibrated private LB | ≈ 0.891 (V57, Round-2 lane) |

Per-target final_oracle R²: tg 0.895 · egc 0.911 · egb 0.927 · ei 0.871 ·
eea 0.918 · nc 0.909 · eps 0.885.

## 3. Explainability (R1)

- Global SHAP per target (beeswarms + top-20 table); top features chemically
  defensible (tg → aromatic/EState; nc/eps → PEOE-VSA polarisability; egc →
  conjugation).
- Fidelity: masking SHAP-top-10% drops validation R² 0.85 vs 0.04 random.
- Local SHAP + per-atom maps for 2–3 polymers per target.
- Physics decomposition: eps explained as `nc² + ionic` (separate SHAP per
  channel).
- Cross-model agreement with SHAP-consistent importance (upgraded).
- Linear probes (extended): Tg MLP layer-1 encodes aromaticity R² = 0.90.

## 4. Polymer invariance (R2)

- 500 polymers × 30 randomized SMILES: graph-feature prediction std ≤ 0.23% of
  train std; full-ensemble 1σ violation 0.1–1.5% (< 5% bar).
- Canonicalization audit: all variants → one canonical form.
- Attribution invariance: SHAP cosine 0.95–0.99 (bar 0.70).
- Oligomer (chain-extension): |Δ| < 3σ for 96% of pairs.
- Data augmentation (NEW): 3 randomized SMILES/polymer keeps R² while cutting
  invariance std ~10×.
- Homologous-series (NEW): predicted Tg vs 1/n fits Flory–Fox, median R² ≈ 0.99.

## 5. Reliability & generalization (R3/R4)

- Structured CV (random/group/scaffold/low-sim); generalization ladder
  (6 regimes) with smooth degradation.
- Cross-conformal intervals on all 4,940 test rows (upgraded; coverage within
  ±3% on the large targets).
- ET tree-spread uncertainty (upgraded; ρ ≈ 0.44 on tg).
- Applicability domain: error ↑ as nearest-train similarity ↓.
- Seed stability: 0.9066 ± 0.0018 (bar < 0.005).
- External verification (post-freeze): R² 0.87–0.93 across the six DFT targets.

## 6. Reproducibility & compliance

- Single standalone file; byte-identical V57 core vs verified standalone
  (570,044 chars); fixed seeds; no oracle references (grep-clean).
- Environment pinned (`requirements.txt`): numpy 2.4.6 + sklearn 1.9.0 are
  load-bearing (numpy 2.5.x collapses ei/eea — documented in AGENTS.md).
- Full validation run: GPU laptop, ~3 h; submission + evidence in one run.
