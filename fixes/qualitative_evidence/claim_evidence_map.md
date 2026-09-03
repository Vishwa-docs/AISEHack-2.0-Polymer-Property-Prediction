# Claim-to-Evidence Map — Round 3 Polymer Property Prediction

Generated from ARCHITECTURE.md and evidence tables in `outputs/evidence_tables/`. Every claim below is linked to a specific, reproducible artifact.

---

## 1. Problem Structure Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| No exact (polymer, property) pair overlap for 6/7 targets | ARCHITECTURE.md Table (row 26-27): egc, egb, ei, eea, nc, eps all 0% exact overlap | VERIFIED |
| Tg has only 12.3% polymer overlap with train | ARCHITECTURE.md row 27: "polymer in train under any property" for tg = 28.0%, but §0 says 12.3% for Tg evaluation polymers | NEEDS VERIFICATION (exact number in §0 vs table) |
| DFT targets are cross-property imputation; Tg is structure→property extrapolation | ARCHITECTURE.md §0 rationale | DESIGN RATIONALE |
| Mean R² ceiling for perfect Tg model = 0.9172 | ARCHITECTURE.md §0: "even a perfect Tg model... reaches a mean of only 0.9172" | CALCULATED (not directly in evidence tables) |
| Equal target weighting: each target = 1/7 of score | ARCHITECTURE.md §1: `score = (1/7) · Σ_t R²_t` | SPECIFICATION |

---

## 2. Representation & Feature Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| Stage A shared representation: RDKit 2D, 3D, Morgan counts, Tanimoto KRR, character n-grams, rdEHT, PI1M SVD, Polymer-Genome triples | ARCHITECTURE.md §3 | ARCHITECTURE DOC |
| Character n-gram (2,7) + Ridge is the only non-invariant component | ARCHITECTURE.md §3 bullet 5: "This is the only non-invariant component of the whole system" | ARCHITECTURE DOC |
| PI1M character SVD is label-free, fitted from random init inside run | ARCHITECTURE.md §3 bullet 7 | ARCHITECTURE DOC |
| Polymer-Genome atomic triples worth egb +0.0092, nc +0.0081 | ARCHITECTURE.md §3 bullet 8: "worth egb 0.9167 → 0.9259 and nc 0.8438 → 0.8519" | ARCHITECTURE DOC |

---

## 3. Per-Target Lane Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| Tg: GB + ET + Huber/RF + character-residual correction | ARCHITECTURE.md §4 Table | ARCHITECTURE DOC |
| Egc: Boosting + PI1M-SVD residual arm | ARCHITECTURE.md §4 Table | ARCHITECTURE DOC |
| Egb: Affine identity (egb = a·egc + b) + ET residual | ARCHITECTURE.md §4 Table: "0.9205 → 0.9478" | ARCHITECTURE DOC |
| Ei: MLP + GP + Tanimoto KRR + bare band-edge identity | ARCHITECTURE.md §4 Table | ARCHITECTURE DOC |
| Eea: Spread-calibrated residual + identity | ARCHITECTURE.md §4 Table | ARCHITECTURE DOC |
| Nc: Ionic → n projection + classical ensemble | ARCHITECTURE.md §4 Table: "physics reparametrisation was worth +0.0434" | ARCHITECTURE DOC |
| Eps: n² + ionic with ET on 26 polar-group features | ARCHITECTURE.md §4 Table: "best single mechanism... +0.0666" | ARCHITECTURE DOC |
| Train duplicates deduplicated by median per (canon SMILES, target) | ARCHITECTURE.md §4 | ARCHITECTURE DOC |

---

## 4. Physics Overlay Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| egc = ei − eea: n=59, corr 0.9882, R² 0.9716, MAE 0.0716 eV, bias +0.0443 eV; used bare, no ML residual (LOO R² −0.82 if added) | ARCHITECTURE.md §5 Table | ARCHITECTURE DOC |
| eps = n² + ionic, ionic ≥ 0: n=134, 0 violations, min 0.0240, median 0.6896, std 0.4088 vs eps std 1.0697; 2.62× better-conditioned | ARCHITECTURE.md §5 Table | ARCHITECTURE DOC |
| egb = a·egc + b: n=175, a=1.1586, b=−1.0437, R² 0.9282; affine map + ET residual | ARCHITECTURE.md §5 Table | ARCHITECTURE DOC |
| All overlays guarded: only fire when inputs exist, spliced so row never worse than model prediction | ARCHITECTURE.md §5 | ARCHITECTURE DOC |

---

## 5. Assembly & Calibration Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| Per-target OOF NNLS blend, then blend/splice/reflected-source chain | ARCHITECTURE.md §6 | ARCHITECTURE DOC |
| NNLS chosen over unconstrained stacking to avoid negative weights fitting noise | ARCHITECTURE.md §6 | ARCHITECTURE DOC |
| Blend beats every single family by 0.02–0.05 | ARCHITECTURE.md §6 | ARCHITECTURE DOC |
| Tg, egc, egb, nc, eps: character-residual term `final = base + 0.20 × (char-ngram Ridge)` | ARCHITECTURE.md §6 | ARCHITECTURE DOC |
| Ei, eea: spread calibration `spread = median + 1.05×(base − median)`, clipped to ±0.25σ beyond q0.001/q0.999 | ARCHITECTURE.md §6 | ARCHITECTURE DOC |
| Contract check: exactly 4,940 unique ids, all-finite predictions | ARCHITECTURE.md §6 | ARCHITECTURE DOC |

---

## 6. Inference Stack Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| Spine is transductive → cannot be serialized to per-row weights | ARCHITECTURE.md §7 | ARCHITECTURE DOC |
| `inference.py` layers exact cache over approximate portable model | ARCHITECTURE.md §7 | ARCHITECTURE DOC |
| Runs model by default (`--mode model`) not cache | ARCHITECTURE.md §7 | ARCHITECTURE DOC |
| `weights/polymer_weights.joblib` contains: exact pipeline predictions for eval rows, partner lookup (5,920 polymers), identity coefficients, compact LightGBM per target | ARCHITECTURE.md §7 | ARCHITECTURE DOC |
| Served model honest 5-fold OOF R²: tg 0.897, egc 0.860, egb 0.875, eea 0.884, nc 0.795, ei 0.773, eps 0.694 | ARCHITECTURE.md §7 & inference.py `BASE_OOF_R2` | VERIFIED (matches code) |
| LightGBM params: seed 2026, deterministic, n_jobs=1; small targets n_est=300/leaves=15/min_child=5/colsample=0.6/reg_lambda=1; large targets n_est=500/leaves=31/min_child=20/colsample=0.7; lr=0.05, subsample=0.8 | ARCHITECTURE.md §7 | ARCHITECTURE DOC |
| Every prediction returned with source, nearest-training Tanimoto, AD tier, conformal interval | ARCHITECTURE.md §7 | VERIFIED (website shows all) |

---

## 7. Evidence Suite Claims (Proxy Model)

| Claim | Evidence | Status |
|-------|----------|--------|
| Proxy: Ridge + ET + LightGBM on Stage-A features, GroupKFold on canon SMILES, NNLS blend | ARCHITECTURE.md §8 | ARCHITECTURE DOC |
| Proxy OOF: tg 0.909, egc 0.895, egb 0.871, ei 0.800, eea 0.853, nc 0.803, eps 0.744 | ARCHITECTURE.md §8 | ARCHITECTURE DOC |
| Proxy explanations are about feature space, not production assembly chain | ARCHITECTURE.md §8 | LIMITATION ACKNOWLEDGED |

---

## 8. Quantitative Evidence Results (from evidence tables)

### 8.1 Invariance (R2)

| Metric | Claim | Evidence File | Value | Status |
|--------|-------|---------------|-------|--------|
| Graph prediction violation rate (0.5σ, 1σ, 2σ) | Zero across all targets | `smiles_invariance_graph_violation_summary.csv` | All 0.0 | VERIFIED |
| Graph-feature prediction std (Tg) | ≤0.23% of training std | `smiles_invariance_per_target.csv` | 0.2301% | VERIFIED |
| SHAP attribution cosine similarity (mean) | 0.9797 | `attribution_invariance_per_target.csv` | Mean 0.9797, min 0.947 (tg), max 0.996 (nc) | VERIFIED |

### 8.2 Explainability (R1)

| Metric | Claim | Evidence File | Value | Status |
|--------|-------|---------------|-------|--------|
| Fidelity test: 5% SHAP masking drop | Mean R² drop 0.810 vs 0.022 random | `fidelity_table.csv` (frac_masked=0.05) | Mean drop_top_shap ≈ 0.807, mean drop_random ≈ 0.032 | APPROX (proxy model) |
| Cross-model explanation agreement (Spearman) | ρ = 0.471 vs 0.60 bar | `explanation_agreement.csv` | Mean across all pairs ≈ 0.472 | VERIFIED (FAIL vs pre-registered bar) |
| Physics-decomposed attribution for eps | PASS | `physics_decomp_values.csv` | Not checked here | ARCHITECTURE DOC |

### 8.3 Robustness (R3)

| Metric | Claim | Evidence File | Value | Status |
|--------|-------|---------------|-------|--------|
| Conformal coverage max |Δ| | ≤0.03 for all targets | `conformal_coverage_table.csv` | Max |Δ| = 0.078 (nc at 90%) | FAIL (qualitative scorecard says 0.0235 - different table?) |
| Error–uncertainty correlation ρ≥0.30 | ≥5/7 targets | `error_uncertainty_correlation.csv` | Only egc (0.304) ≥0.30 | FAIL (qualitative scorecard says 5/7 - different calc?) |
| AD tiers monotone error rise | Tg MAE: 14.77 → 19.50 → 29.50 → 43.56 °C | `ad_analysis_table.csv` | Verified | VERIFIED |
| Seed stability (Tg) | std ≤ 0.002 | `seed_stability.csv` | 0.00182 | VERIFIED |

### 8.4 Generalizability (R4)

| Metric | Claim | Evidence File | Value | Status |
|--------|-------|---------------|-------|--------|
| Canonical-group mean R² (G1) | 0.824 | `generalization_ladder.csv` G1 rows | 0.8238 | VERIFIED |
| Scaffold-holdout mean R² (G2) | 0.658 | `generalization_ladder.csv` G2 rows | 0.6581 | VERIFIED |
| All 7 targets positive R² on G1 and G2 | Yes | `generalization_ladder.csv` | All >0 | VERIFIED |

### 8.5 Known Failures (from ARCHITECTURE.md §8)

| Check | Result | Cause | Fix Status |
|-------|--------|-------|------------|
| R1.4 Cross-model agreement | FAIL (ρ=0.471) | Collinear features ranked differently by Ridge vs ET vs LGBM | Reported as limitation; feature-group ranking lifts to 0.52 |
| R3.2 Conformal coverage | FAIL (max|Δ|=0.089) | ~45 cal rows on small targets → ±4.5% binomial noise; tg/egc inside ±3% | Cross-conformal implemented, moved 0.100→0.033 in smoke; needs full rerun |
| R3.3 Error–uncertainty ρ≥0.30 | FAIL (1/7 targets) | Shallow trees confidently wrong off-domain | ExtraTrees spread lifted tg 0.224→0.444; MC-dropout/deep ensembles needed |
| AUG Augmentation | FAIL | Artifact not regenerated in last full run | Smoke showed tg OOF held (0.780→0.784), spread fell ~10× |

---

## 9. Additional Evidence (Not in Scorecard)

| Claim | Evidence | Status |
|-------|----------|--------|
| Linear probes: Tg-MLP L1 encodes aromaticity R² 0.895; egc L1 0.901; eps L1 0.934 | ARCHITECTURE.md §8 | ARCHITECTURE DOC |
| Activation patching: randomised-SMILES variant activations → canonical changes prediction by exactly 0.0 | ARCHITECTURE.md §8 | ARCHITECTURE DOC |
| Causal tracing: restoring any single hidden layer fully recovers prediction (recovery = 1.0) | ARCHITECTURE.md §8 | ARCHITECTURE DOC |
| Structural counterfactuals: textbook edits agree 27/40 = 67.5% (best on rigidity 12/13) | ARCHITECTURE.md §8 | ARCHITECTURE DOC |
| Homologous series: predicted Tg vs 1/n linear, median R² ≈ 0.99 (Flory–Fox recovered) | ARCHITECTURE.md §8 | ARCHITECTURE DOC |

---

## 10. Reproducibility & Environment Claims

| Claim | Evidence | Status |
|-------|----------|--------|
| One run, one file, fixed seeds (2026); reads only train.csv, test.csv, PI1M.csv | ARCHITECTURE.md §9 | ARCHITECTURE DOC |
| Runtime: ~2.5h for `--mode submission` on laptop CPU | ARCHITECTURE.md §9 | ARCHITECTURE DOC |
| ei/eea leaves depend on RDKit `rdEHTTools.RunMol` which segfaults on linux-x86_64, works on macOS | ARCHITECTURE.md §9 | PLATFORM SENSITIVE |
| Python 3.12 collapses ei 0.871→0.512 regardless of packages | ARCHITECTURE.md §9 | ENVIRONMENT MISMATCH |
| numpy ≥2.5 collapses ei → 0.516 | ARCHITECTURE.md §9 | ENVIRONMENT MISMATCH |
| scikit-learn <1.9.0 collapses ei → 0.512 | ARCHITECTURE.md §9 | ENVIRONMENT MISMATCH |
| Other 5 targets reproduce at correlation 1.0 across platforms | ARCHITECTURE.md §9 | VERIFIED CLAIM |
| Phase 7 GNN decorrelation blend: structure-grouped GINE, 3 seeds/target, CV on train only, blend weight `w = clip((cv−0.80)/0.25, 0.10, 0.60)` | ARCHITECTURE.md §9 Part C | ARCHITECTURE DOC |
| GNN not better alone (tg 0.8987 vs ensemble 0.8954); errs differently → decorrelation lifts mean to 0.907551 | ARCHITECTURE.md §9 Part C | ARCHITECTURE DOC |

---

## 11. Known Weaknesses (from ARCHITECTURE.md §10)

| Weakness | Detail | Evidence |
|----------|--------|----------|
| Chain depth | 7-arm sibling scored 0.838 standalone; self-generated chain drifts up to 19.5°C on Tg | ARCHITECTURE DOC |
| Transductive design | Leaderboard model cannot be served per row; served fallback weaker | ARCHITECTURE DOC |
| String features not invariant | Character n-gram reads SMILES string; cost measured separately | ARCHITECTURE DOC |
| Tg label noise | Only 4 replicate groups (median spread 5.9°C); practical Tg ceiling ~0.92 | ARCHITECTURE DOC |

---

## 12. Website Demo Claims (Verified)

| Website Element | Claim | Evidence | Status |
|-----------------|-------|----------|--------|
| Predictions show value, 90% conformal interval, AD tier, nearest training analogue, source | All present | `app.py` lines 193-212 | VERIFIED |
| T4 out-of-domain banner shows Tg error 43.56°C (R² 0.635) vs 14.77°C (R² 0.962) | Exact match | `ad_analysis_table.csv` | VERIFIED |
| Invariance demo: 500 polymers (Tg/EgC), 12 spellings, graph std ≤0.23%, violation rate 0.0000, SHAP cosine 0.947-0.996 | Updated caption | `smiles_invariance_per_target.csv`, `smiles_invariance_graph_violation_summary.csv`, `attribution_invariance_per_target.csv` | VERIFIED |
| Literature anchors: PS (104°C) and PMMA (108°C) from Koike & Kumaki 2022; disclosed as illustrative, not benchmark | `app.py` lines 123-130, 213-220 | CITATION PROVIDED | VERIFIED |
| Visual explanation: descriptor importance (gain×presence) for compact model; labeled "not a SHAP claim" | `app.py` lines 224-236 | HONEST LABELING | VERIFIED |
| Candidate screening: forward screen with interval and AD tier retained | `app.py` lines 239-254 | FUNCTIONAL | VERIFIED |
| Footer: served model OOF R² per target; leaderboard 0.907551 local / 0.920 public | `app.py` lines 294-303, `inference.py` | VERIFIED | VERIFIED |

---

## 13. Boundary Conditions to Retain (from STATUS.md)

| Boundary | How Presented |
|----------|---------------|
| Cross-model feature-rank agreement = 0.472 | Secondary sensitivity diagnostic, not primary fidelity evidence |
| Extreme family/low-similarity and tail extrapolation not claimed operating regime | Demo shows applicability tier + uncertainty, not reliable extrapolation |
| No external labels or secret artifacts in public material | "Local held-out verification panel" used instead of oracle/khazana/polyinfo/TgSS/test_answers |

---

## Summary: Claims Requiring Re-verification After Isolated Run

1. **Conformal coverage max |Δ|** — qualitative scorecard says 0.0235 (passes ±3%), but `conformal_coverage_table.csv` shows 0.078 (fails). Need to confirm which table is authoritative.
2. **Error–uncertainty correlation** — qualitative scorecard says 5/7 targets ρ≥0.30, but `error_uncertainty_correlation.csv` shows only 1/7. Need to confirm calculation method.
3. **Fidelity test at 5%** — qualitative scorecard: 0.810 vs 0.022; `fidelity_table.csv`: ~0.807 vs ~0.032. Close but not identical.
4. **All Stage 8 evidence checks** — must be regenerated from isolated run outputs and compared.

**Action**: After isolated notebook completes, run `validate_archive_evidence.py` and `summarize_qualitative.py` on the new output directory, compare with recorded results, update this map accordingly.