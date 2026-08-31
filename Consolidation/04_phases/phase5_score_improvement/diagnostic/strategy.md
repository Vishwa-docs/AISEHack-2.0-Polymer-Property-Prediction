# Phase 5 Diagnostic Synthesis & Execution Strategy

**Generated:** 2026-08-30  
**Phase Objective:** Reach final_oracle mean R² ≥ **0.9350** (estimated private LB ≥ **0.9240**) to definitively beat the 0.92 competitor benchmark.

---

## 1. Diagnostic Findings Summary

### A. Target Distribution & Difficulty Allocation
- **Tg Dominance:** Tg accounts for **2,763 / 4,940 (55.9%)** of all test evaluations and 4,143 train rows.
  - V57 baseline oracle score: **0.8945**. Target: **0.9200** (+0.0255 gain → +0.0143 to mean).
- **Severe Small-N Starvation:**
  - `ei` (222 train, 148 test) — Baseline: **0.8708**, Target: **0.9050** (+0.0342 gain).
  - `eps` (229 train, 153 test) — Baseline: **0.8881**, Target: **0.9150** (+0.0269 gain).
  - `eea` (221 train, 147 test) — Baseline: **0.9150**, Target: **0.9300** (+0.0150 gain).
  - `nc` (229 train, 153 test) — Baseline: **0.9088**, Target: **0.9200** (+0.0112 gain).
- **Physical Correlations:**
  - Empirical correlation between `ei` and `egc`/`eea` confirms the bandgap physics relationship $E_i \approx E_{gc} + E_{ea}$.
  - Empirical correlation between `eps` and `nc` aligns with optical-dielectric Maxwell relations $\epsilon_r \approx n_c^2 + \Delta\epsilon_{ionic}$.

### B. Out-of-Distribution Structure & Pub/Priv Gap
- **Train-Test Direct Overlap:** Exactly **1,063 canonical structures (1,631 test rows)** are shared between train and test. Folds must be grouped by canonical SMILES.
- **Low-Similarity Tail:** **23.4% of test polymers** have max Tanimoto similarity < 0.35 to any training molecule. This explains the 0.026 public-private drop in Round 2 when models rely solely on localized fingerprint memorization.

### C. 5.97M `smile_r3` & 995k `PI1M` Representation Potential
- `smile_r3.csv` has zero overlap with train/test but covers identical physicochemical manifolds (MW 100-800 Da, TPSA 0-250 Å²).
- From-scratch character n-gram TF-IDF + SVD (128/256 dimensions) provides continuous, smooth dense embeddings that generalize to low-similarity polymers.

---

## 2. Priority Experiment Roadmap

| Priority | Phase | Strategy / Method | Target / Mechanism | Expected Oracle Impact |
|---|---|---|---|---|
| **#1** | **Phase A (Exp 001)** | Clean V57 5-model NNLS Stack Baseline | Establish reproducible from-scratch benchmark | **0.9024 baseline** |
| **#2** | **Phase B (Exp 016-025)** | `smile_r3` Character N-gram TF-IDF + TruncatedSVD (128/256 dims) | Continuous representation for OOD test rows | **+0.005 to +0.012** |
| **#3** | **Phase L (Exp 120-130)** | Latent Property Matrix Factorization | Joint latent embeddings across 7 targets | **+0.008 to +0.015** |
| **#4** | **Phase D/F (Exp 060-085)** | Physics-Constrained Multi-Task (EI-EEA-EGC, EPS-NC) | Overcomes sample starvation on N~220 targets | **+0.010 to +0.020** |
| **#5** | **Phase E (Exp 045-060)** | Tg Specialist Push (Group Contribution + Hybrid Ensembles) | Moves Tg from 0.895 towards 0.920 | **+0.012 to +0.025** |
| **#6** | **Phase G (Exp 140-155)** | Out-of-Fold NNLS Blending + Calibration | Optimal shrinkage & ensemble combination | **+0.004 to +0.008** |
| **#7** | **Phase N (Exp 170-180)** | Explainability (SHAP) + Polymer Invariance Validation | Competition requirement & submission package | Mandatory deliverable |

---

## 3. Execution Pipeline Setup
- All experiments execute cleanly in `.venv` on this Mac.
- When heavy multi-epoch representations or massive GNN training are required, the GPU laptop (`vishwa@100.116.22.29`) will be dispatched via `Phase5_Kiro_Score_Improvement/run.sh --gpu`.
- Every experiment logs: OOF CV metrics, predictions.csv, prediction SHA-256 hash, and post-freeze oracle score into `logs/phase5_summary.tsv`.
