# Report claim-to-evidence map

This file is the audit companion to `Report.md`. A claim below is public-report
eligible only at the stated scope; it is not permission to reuse the number in a
different evaluation protocol.

| Report claim | Executed artifact | Scope / status |
|---|---|---|
| Submitted local mean R² 0.907551; public score 0.920 | `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/ARCHITECTURE.md` §0; `outputs/CAPTIONS.md` | submitted-system outcome |
| Equal target weighting; Tg data/variance imbalance | `outputs/eda/per_target_counts_vs_metric_weight.png`; `outputs/eda/variance_share_trap.png`; `Personal/docs/00_My Docs.md` | EDA and contest-metric rationale |
| Feature-family proxy ablations | `outputs/evidence_tables/feature_ablation_results.csv` | proxy, grouped-CV feature-space study—not final ensemble attribution |
| Atomic-triple target gains | `ARCHITECTURE.md` §3 | targeted architecture ablation |
| Guarded band-edge, bulk-bandgap and optical/ionic routes | `ARCHITECTURE.md` §5; `outputs/evidence_tables/physics_route_comparison.csv` | co-measured-subset / guarded-route evidence |
| Randomised-SMILES graph invariance | `outputs/evidence_tables/smiles_invariance_per_target.csv`; `smiles_invariance_graph_violation_summary.csv` | valid alternative spellings of the same graph |
| Attribution stability | `outputs/evidence_tables/attribution_invariance_per_target.csv` | proxy feature-space explanations |
| Explanation fidelity | `outputs/evidence_tables/fidelity_table.csv`; `outputs/explainability/fidelity_curve_tg.png` | proxy masking intervention, not chemical causality |
| Structural generalisation | `outputs/evidence_tables/generalization_ladder.csv`; `outputs/generalization/generalization_ladder_plot.png` | recorded canonical-group and scaffold split study |
| Tg applicability-domain error gradient | `outputs/evidence_tables/ad_analysis_table.csv`; `outputs/generalization/ad_analysis_plot.png` | recorded Tg similarity-tier analysis |
| Pure-ML baseline 0.816344 | `fixes/pure_ml/outputs_full/grouped_cv_metrics.csv`; `fixes/pure_ml/VERIFICATION.md` | grouped-CV baseline, not directly comparable to held-out-panel score |
| Uncertainty calibration / error–uncertainty success | *No release claim* | archived tables conflict; see `fixes/qualitative_evidence/claim_evidence_map.md` and isolated-run audit |

## Literature provenance

The numbered citations in `Report.md` are drawn from `Personal/Research/INDEX.md`.
The paper support is methodological (why a feature, model class or test is reasonable);
the project artifacts above support the numerical outcome.
