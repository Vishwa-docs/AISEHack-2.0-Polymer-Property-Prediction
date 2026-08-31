# REQUIREMENTS.md — Phase 4: Explainability, Robustness & Generalization
## AISEHack 2.0 Polymer Property Prediction · Round 3

> **Purpose:** This document is the ground truth for what judges will evaluate
> beyond the leaderboard score. Every claim made in reports, notebooks, or
> presentations must trace back to an experiment in PLAN.md and a corresponding
> output artifact (plot, table, CSV, JSON). No hand-waving; every requirement
> has a concrete, measurable proof target.

---

## 1. The Two Judging Themes (Official)

From the Round 3 overview (verbatim):

> *"This round is built around two central themes: Explainability of the models
> and Polymer invariance. The goal is not only to build accurate models, but also
> models whose predictions are interpretable, robust, and invariant to different
> valid representations of the same polymer structure."*

The five scoring axes the judges use are:

| Axis | Weight (inferred) | What "winning" looks like |
|---|---|---|
| Quantitative performance | High | Public/private LB R² — our current best is 0.902 oracle |
| Model Explainability | High | Visual + quantitative attribution evidence |
| Polymer Invariance Robustness | High | Proof that equiv. SMILES give same predictions & same reasons |
| Methodology & Reliability | Medium | Rigorous CV, UQ, calibrated intervals |
| Proven Generalization | Medium | Khazana or external-polymer holdout, scaffold/family splits |

---

## 2. Requirement R1 — Model Explainability

### R1.1 — Global Feature Importance
**What:** Rank features by their aggregate contribution to each of the 7 target predictions.  
**How:** SHAP TreeExplainer on the LightGBM / ExtraTrees / Ridge components of the ensemble.  
**Proof required:**
- SHAP beeswarm plot (one per target, top 20 features). Saved as `outputs/shap_beeswarm_<target>.png`.
- SHAP summary bar chart across all 7 targets in one figure. Saved as `outputs/shap_summary_global.png`.
- `outputs/shap_top20_per_target.csv` — table of top-20 features × 7 targets.

**Pass criterion:** The top features for each target must be chemically defensible:
- `tg`: aromatic ring count, backbone rigidity descriptors, molecular weight proxies should dominate.
- `egc/egb`: conjugation-related features (aromatic system size, HOMO/LUMO-adjacent descriptors).
- `nc/eps`: polarizability-related descriptors (heavy-atom count, aromatic fraction, ionic descriptors).
- `ei/eea`: HOMO/LUMO descriptors, electronegativity proxies.

### R1.2 — Local / Instance-Level Explanations
**What:** For 10–20 representative validation polymers (2–3 per target), show which atoms/substructures drive the prediction.  
**How:** SHAP force plots + RDKit SimilarityMap overlaid on 2D structure.  
**Proof required:**
- `outputs/local_shap_<target>_<polymer_id>.png` — 2D polymer with per-atom SHAP coloring.
- SHAP force plot per sample. Saved as `outputs/shap_force_<target>_<polymer_id>.png`.
- Commentary: does the model highlight known structural drivers (aromatic rings → high Tg, etc.)?

### R1.3 — Explanation Faithfulness (Fidelity Test)
**What:** Verify that features the model says are important actually matter when removed.  
**How:** Progressive feature masking — mask top-k SHAP features, measure R² drop vs. random-k masking.  
**Proof required:**
- `outputs/fidelity_curve_<target>.png` — Fidelity+ curve (R² vs. fraction of top-SHAP features masked).
- `outputs/fidelity_table.csv` — Fidelity+ and Fidelity- scores per target.
- **Pass criterion:** Masking SHAP-top-k causes a larger R² drop than masking random-k, for k=5,10,20%.

### R1.4 — Cross-Model Explanation Agreement
**What:** Different model families should independently agree on which features matter.  
**How:** Compute global feature importance (SHAP or permutation) for Ridge, ExtraTrees, and LightGBM components separately. Compute Spearman correlation between ranking pairs.  
**Proof required:**
- `outputs/explanation_agreement_heatmap.png` — model × model Spearman ρ heatmap per target.
- `outputs/explanation_agreement.csv` — numeric ρ values.
- **Pass criterion:** Mean Spearman ρ ≥ 0.60 across model pairs, indicating convergent explanation.

### R1.5 — Physics-Decomposed Explanations (for eps and ei)
**What:** Because `eps = nc² + ionic` and `ei ≈ egc + eea`, explain the *components*, not just the final target.  
**How:** SHAP on the ionic predictor and the nc predictor separately. Show how different descriptors explain the electronic vs. ionic contributions to eps.  
**Proof required:**
- `outputs/physics_decomp_eps_shap.png` — SHAP for ionic vs. nc² components.
- Commentary on which features drive each physical channel.

---

## 3. Requirement R2 — Polymer Invariance Robustness

### R2.1 — SMILES Representation Invariance (Prediction)
**What:** The same polymer encoded in K different but chemically equivalent SMILES strings must produce nearly identical predictions.  
**How:** For 500 validation polymers, generate K=30 randomized SMILES (RDKit `MolToSmiles(m, doRandom=True)`). Run the full feature pipeline + trained model on each. Measure per-polymer prediction std.  
**Proof required:**
- `outputs/smiles_invariance_per_target.csv` — per-polymer mean, std, max deviation for each target.
- `outputs/smiles_invariance_boxplot.png` — box plots of prediction std across the 500 polymers × 7 targets.
- `outputs/smiles_invariance_violation_rate.csv` — fraction of (polymer, SMILES) pairs where |pred(T(x)) - pred(x)| > ε, for ε = {0.5σ, 1σ, 2σ} of each target's train distribution.
- **Pass criterion:** Mean prediction std across 30 SMILES variants ≤ 1% of target train std for descriptor-based features. Violation rate at ε=1σ < 5%.

### R2.2 — Canonicalization Verification
**What:** Confirm the pipeline canonicalizes SMILES before featurization, so all equiv. forms reduce to one representation.  
**How:** For 100 test polymers, show that canonical SMILES is identical regardless of input form.  
**Proof required:**
- `outputs/canonicalization_check.txt` — audit log showing pre/post canonical forms.
- Short written statement in the report confirming RDKit canonicalization is applied at ingestion.

### R2.3 — Explanation Invariance (Attribution Stability)
**What:** Not just the *prediction* but the *reason* should be stable across equivalent SMILES representations. A model that gives the same answer for different structural reasons is scientifically weaker.  
**How:** For 100 polymers × K=10 randomized SMILES, compute SHAP attribution vector for each representation. Measure cosine similarity of attribution vectors within each polymer group.  
**Proof required:**
- `outputs/attribution_invariance_per_target.csv` — mean cosine similarity per target.
- `outputs/attribution_invariance_scatter.png` — scatter of prediction invariance (x-axis) vs. attribution invariance (y-axis). Ideal: top-right quadrant.
- **Pass criterion:** Mean attribution cosine similarity ≥ 0.70 per target (for fingerprint-based models).

### R2.4 — Polymer Chain Extension (Oligomer) Invariance
**What:** Predictions for a monomer and its dimer/trimer extensions should be physically consistent (not wildly different).  
**How:** For 50 polymers in the validation set, construct dimer SMILES by doubling the repeat unit (simple string concatenation at the `*` attachment points). Compare predictions.  
**Proof required:**
- `outputs/oligomer_invariance.csv` — monomer vs. dimer prediction delta per target.
- `outputs/oligomer_invariance_plot.png` — scatter of monomer vs. dimer predictions.
- **Pass criterion:** |Δ| < 3σ of train std for ≥ 85% of (polymer, target) pairs. Document any systematic drift direction and physical interpretation.

---

## 4. Requirement R3 — Methodology & Reliability

### R3.1 — Structured Validation (Beyond Random CV)
**What:** Demonstrate that the model was evaluated in a way that avoids data leakage from structurally similar polymers.  
**How:** Report CV R² under all four regimes: (a) random, (b) canonical-group, (c) Bemis-Murcko scaffold, (d) similarity cluster (Tanimoto < 0.4 to any training polymer).  
**Proof required:**
- `outputs/cv_validation_table.csv` — per-target R² under all 4 split strategies.
- `outputs/cv_validation_barplot.png` — grouped bar chart (split type × target).
- **Pass criterion:** Results reported honestly, with commentary on degradation. Scaffold R² > 0.85 for the DFT targets (egc, egb, ei, eea, nc, eps).

### R3.2 — Uncertainty Quantification (Conformal Prediction)
**What:** Every test prediction should have a calibrated confidence interval.  
**How:** Use `MAPIE` (or a manual split-conformal wrapper) around the ensemble. Calibrate on a held-out calibration fold. Report empirical coverage at 80%, 90%, 95%.  
**Proof required:**
- `outputs/conformal_coverage_table.csv` — nominal vs. empirical coverage per target per confidence level.
- `outputs/conformal_calibration_plot.png` — reliability diagram: nominal coverage vs. empirical coverage (7 targets, 3 confidence levels).
- `outputs/test_predictions_with_intervals.csv` — 4,940 rows with columns: id, target_type, prediction, lower_80, upper_80, lower_90, upper_90.
- **Pass criterion:** Empirical coverage within ±3% of nominal level (e.g., 90% nominal → 87–93% observed).

### R3.3 — Error–Uncertainty Correlation
**What:** The model's uncertainty estimates should be higher where its errors are larger. An uncertainty estimate that doesn't track actual errors is useless.  
**How:** On the validation set, plot prediction error vs. ensemble standard deviation (std across 5 CV fold models). Compute Pearson ρ between |error| and uncertainty.  
**Proof required:**
- `outputs/error_vs_uncertainty_scatter_<target>.png` (7 plots).
- `outputs/error_uncertainty_correlation.csv` — Pearson ρ per target.
- **Pass criterion:** ρ ≥ 0.30 for ≥ 5 of 7 targets.

### R3.4 — Applicability Domain (AD) Analysis
**What:** Quantify model reliability as a function of structural novelty relative to training data.  
**How:** For each test polymer, compute nearest-neighbor Tanimoto similarity to training set (Morgan FP). Bin by similarity: ≥0.9, 0.7–0.9, 0.5–0.7, <0.5. Report mean absolute error per bin on the validation set.  
**Proof required:**
- `outputs/ad_analysis_table.csv` — MAE and R² per similarity bin per target.
- `outputs/ad_analysis_plot.png` — line/bar chart of R² vs. similarity bin.
- `outputs/ad_test_similarity.csv` — 4,940-row file: id, nearest_train_tanimoto, ad_confidence_tier.
- **Pass criterion:** Monotonic (or near-monotonic) increase in error as similarity decreases. Document the trend clearly.

### R3.5 — Seed Stability / Bootstrap Variance
**What:** Results should not depend on a lucky random seed.  
**How:** Run CV OOF evaluation with 5 different seeds. Report std of R² across seeds.  
**Proof required:**
- `outputs/seed_stability.csv` — per-target R² for seeds [42, 2024, 2025, 2026, 137], plus mean and std.
- **Pass criterion:** Std of mean R² across seeds < 0.005.

---

## 5. Requirement R4 — Proven Generalization

### R4.1 — Generalization Ladder
**What:** A systematic demonstration that performance degrades gracefully as test polymers become increasingly different from training polymers.  
**How:** Evaluate OOF predictions at 6 increasing levels of difficulty:  
  - G0: random CV  
  - G1: canonical-group CV (no leakage from identical structures)  
  - G2: Bemis-Murcko scaffold CV  
  - G3: polymer-family CV (aromatic vs. aliphatic, fluorinated vs. non-fluorinated)  
  - G4: low-similarity CV (train–val Tanimoto < 0.6)  
  - G5: ultra-low-similarity (Tanimoto < 0.4)  
**Proof required:**
- `outputs/generalization_ladder.csv` — per-target R² × 6 split levels.
- `outputs/generalization_ladder_plot.png` — bar chart showing R² degradation with increasing novelty (the "staircase" figure).
- **Pass criterion:** Results are honest and reported as-is. The existence of this analysis, with clear annotation of each level, is the proof. Judges want to see you *ran the experiment*, not just that it was good.

### R4.2 — External Verification Against Khazana / Online Polymers
**What:** Test the model against a held-out external dataset of known polymer properties that was NOT used in training.  
**How:** Use the Khazana dataset references known to be in `Oracle/sources/` — specifically the DFT targets (egc, egb, ei, eea, nc, eps). Extract a random 10% hold-out of the test set's oracle-verified rows (those with Khazana exact match) that were NOT in train.csv. Evaluate model predictions on these.  
**Important:** The Khazana subset used here is the same one already inside `Oracle/final_oracle.csv`. You are NOT using it during training — you are comparing final test predictions to ground-truth labels post-freeze for the report only.  
**Proof required:**
- `outputs/khazana_holdout_scores.csv` — per-target R² on the Khazana-verified subset (n≈3,818 rows).
- `outputs/khazana_scatter_<target>.png` — predicted vs. actual scatter for each of the 6 DFT targets on the Khazana subset.
- **Pass criterion:** Khazana R² ≥ 0.88 for egc, egb, nc, eps; ≥ 0.85 for ei, eea.

### R4.3 — Tail Performance (Distribution-Tail Generalization)
**What:** The model should not perform well only on common/easy polymers and fail on extreme property values.  
**How:** For each target, separate validation predictions into top-10% and bottom-10% of true property values. Report R² for each tail vs. the middle 80%.  
**Proof required:**
- `outputs/tail_performance.csv` — tail R² × target.
- `outputs/tail_performance_plot.png` — grouped bar chart (bottom-tail / mid / top-tail per target).

---

## 6. Summary Scorecard

This is the deliverable that synthesizes everything above into a single judge-facing summary.

| Requirement | Artifact | Status |
|---|---|---|
| R1.1 SHAP global | `shap_beeswarm_*.png`, `shap_summary_global.png` | [ ] |
| R1.2 SHAP local + mol viz | `local_shap_*.png` | [ ] |
| R1.3 Fidelity test | `fidelity_curve_*.png`, `fidelity_table.csv` | [ ] |
| R1.4 Cross-model agreement | `explanation_agreement_heatmap.png` | [ ] |
| R1.5 Physics decomposition | `physics_decomp_eps_shap.png` | [ ] |
| R2.1 SMILES prediction invariance | `smiles_invariance_boxplot.png`, violation rate CSV | [ ] |
| R2.2 Canonicalization audit | `canonicalization_check.txt` | [ ] |
| R2.3 Attribution invariance | `attribution_invariance_scatter.png` | [ ] |
| R2.4 Oligomer invariance | `oligomer_invariance_plot.png` | [ ] |
| R3.1 Structured CV | `cv_validation_barplot.png` | [ ] |
| R3.2 Conformal prediction | `conformal_calibration_plot.png`, intervals CSV | [ ] |
| R3.3 Error–uncertainty correlation | `error_vs_uncertainty_scatter_*.png` | [ ] |
| R3.4 Applicability domain | `ad_analysis_plot.png`, `ad_test_similarity.csv` | [ ] |
| R3.5 Seed stability | `seed_stability.csv` | [ ] |
| R4.1 Generalization ladder | `generalization_ladder_plot.png` | [ ] |
| R4.2 Khazana/external verification | `khazana_scatter_*.png` | [ ] |
| R4.3 Tail performance | `tail_performance_plot.png` | [ ] |

**Minimum viable set for judges (must-have):** R1.1, R1.2, R2.1, R2.3, R3.1, R3.2, R4.1, R4.2  
**Full set (aspirational):** all 17 rows above.

---

## 7. Rules Constraints That Apply Here Too

- All code in this Phase reads **only** official competition data: `../Dataset/train.csv`, `../Dataset/test.csv`, `../Dataset/PI1M.csv`, and (optionally) `../Dataset/smile_r3.csv`.
- Oracle data (`../Oracle/`) may be read **only post-freeze** for computing the Khazana verification scores (R4.2). It must never enter training, feature computation, or any uploaded artifact.
- The `../final_submissions/submission.csv` is the prediction file that this phase analyzes. The phase does NOT retrain a new model from scratch — it wraps the existing model's outputs and OOF predictions.
- No external pretrained models, weights, or datasets. All features computed from scratch within the analysis run.
- All output files go into `outputs/` within this folder. No files outside `Phase4_Round3_Explainability/`.
