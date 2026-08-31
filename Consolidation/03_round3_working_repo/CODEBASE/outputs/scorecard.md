# Round 3 Evidence Scorecard

Auto-generated 2026-08-31 16:28 by the evidence engine

| Req | Criterion | Artifacts | Check | Status |
|---|---|---|---|---|
| R1.1 | Global SHAP importance | shap_beeswarm_tg.png, shap_beeswarm_egc.png, shap_beeswarm_egb.png… | files exist | PASS |
| R1.2 | Local SHAP + mol viz | local_shap_tg_0.png, shap_force_tg_0.png | files exist | PASS |
| R1.3 | Fidelity test | fidelity_curve_tg.png, fidelity_table.csv | drop_top_shap > drop_random (drop_top=0.851 vs random=0.043 @10%) | PASS |
| R1.4 | Cross-model agreement | explanation_agreement_heatmap.png, explanation_agreement.csv | mean spearman >= 0.60 (mean ρ=0.471) | FAIL |
| R1.5 | Physics decomposition | physics_decomp_eps_shap.png | file exists | PASS |
| R2.1 | SMILES prediction invariance | smiles_invariance_boxplot.png, smiles_invariance_violation_rate.csv, smiles_invariance_per_target.csv | violation rate < 5% at 1σ (mean 1σ rate=0.0000 on smiles_invariance_graph_violation_summary.csv) | PASS |
| R2.2 | Canonicalization audit | canonicalization_check.txt | file exists | PASS |
| R2.3 | Attribution invariance | attribution_invariance_per_target.csv, attribution_invariance_scatter.png | cosine >= 0.70 (mean cos=0.980) | PASS |
| R2.4 | Oligomer invariance | oligomer_invariance.csv, oligomer_invariance_plot.png | file exists | PASS |
| R3.1 | Structured CV | cv_validation_table.csv, cv_validation_barplot.png | file exists | PASS |
| R3.2 | Conformal prediction | conformal_coverage_table.csv, conformal_calibration_plot.png, test_predictions_with_intervals.csv | coverage within +/-3% (max |Δcoverage|=0.089) | FAIL |
| R3.3 | Error-uncertainty correlation | error_uncertainty_correlation.csv | rho >= 0.30 for >=5 targets (n targets ρ>=0.30: 1) | FAIL |
| R3.4 | Applicability domain | ad_analysis_table.csv, ad_analysis_plot.png, ad_test_similarity.csv | file exists | PASS |
| R3.5 | Seed stability | seed_stability.csv | std < 0.005 (std=0.00182) | PASS |
| R4.1 | Generalization ladder | generalization_ladder.csv, generalization_ladder_plot.png | file exists | PASS |
| R4.2 | External (post-freeze) verification | khazana_holdout_scores.csv | R2 >= 0.88 for DFT targets (egc=0.911, ...) | PASS |
| R4.3 | Tail performance | tail_performance.csv, tail_performance_plot.png | file exists | PASS |
| AUG | Data augmentation experiment | augmentation_experiment.csv, augmentation_experiment_plot.png | file exists | FAIL |

**Passed 14/18 requirement groups.**

Minimum viable set (R1.1, R1.2, R2.1, R2.3, R3.1, R3.2, R4.1, R4.2): PASS, PASS, PASS, PASS, PASS, **FAIL**, PASS, PASS

> R4.2 (external verification) is a POST-FREEZE step in the final pipeline
> (ground-truth answers are read only after the submission is frozen, by a
> separate scorer; the pipeline itself never reads them).

