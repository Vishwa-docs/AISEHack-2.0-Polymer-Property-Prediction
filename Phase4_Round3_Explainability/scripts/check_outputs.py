"""
check_outputs.py — standalone verifier. Prints PASS/FAIL per REQUIREMENTS.md
artifact and exits 0 only when the minimum viable set exists.
Run from the Phase_4 folder root:  python scripts/check_outputs.py
"""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"

REQUIRED = {
    "R1.1_shap_global": ["shap_beeswarm_tg.png", "shap_beeswarm_egc.png",
                          "shap_beeswarm_egb.png", "shap_beeswarm_ei.png",
                          "shap_beeswarm_eea.png", "shap_beeswarm_nc.png",
                          "shap_beeswarm_eps.png", "shap_summary_global.png",
                          "shap_top20_per_target.csv"],
    "R1.2_shap_local": ["local_shap_tg_0.png", "shap_force_tg_0.png"],
    "R1.3_fidelity": ["fidelity_curve_tg.png", "fidelity_table.csv"],
    "R1.4_agreement": ["explanation_agreement_heatmap.png", "explanation_agreement.csv"],
    "R1.5_physics_decomp": ["physics_decomp_eps_shap.png"],
    "R2.1_smiles_invariance": ["smiles_invariance_per_target.csv",
                                "smiles_invariance_boxplot.png",
                                "smiles_invariance_violation_rate.csv"],
    "R2.2_canonicalization": ["canonicalization_check.txt"],
    "R2.3_attribution_invariance": ["attribution_invariance_per_target.csv",
                                     "attribution_invariance_scatter.png"],
    "R2.4_oligomer": ["oligomer_invariance.csv", "oligomer_invariance_plot.png"],
    "R3.1_cv": ["cv_validation_table.csv", "cv_validation_barplot.png"],
    "R3.2_conformal": ["conformal_coverage_table.csv",
                        "conformal_calibration_plot.png",
                        "test_predictions_with_intervals.csv"],
    "R3.3_uq_corr": ["error_vs_uncertainty_scatter_tg.png",
                      "error_uncertainty_correlation.csv"],
    "R3.4_ad": ["ad_analysis_table.csv", "ad_analysis_plot.png",
                 "ad_test_similarity.csv"],
    "R3.5_seed": ["seed_stability.csv"],
    "R4.1_ladder": ["generalization_ladder.csv", "generalization_ladder_plot.png"],
    "R4.2_khazana": ["khazana_holdout_scores.csv", "khazana_scatter_egc.png"],
    "R4.3_tail": ["tail_performance.csv", "tail_performance_plot.png"],
    "R5_scorecard": ["scorecard.md", "trustworthiness_radar.png"],
}

MIN_VIABLE = ["R1.1_shap_global", "R1.2_shap_local", "R2.1_smiles_invariance",
              "R2.3_attribution_invariance", "R3.1_cv", "R3.2_conformal",
              "R4.1_ladder", "R4.2_khazana"]


def main():
    print("=== Phase 4 outputs check ===")
    all_ok = True
    for req, files in REQUIRED.items():
        missing = [f for f in files if not (OUT / f).exists()]
        ok = len(missing) == 0
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {req}" + ("" if ok else f"  missing: {missing}"))
    mv = [r for r in MIN_VIABLE if not all((OUT / f).exists() for f in REQUIRED[r])]
    print("---")
    if mv:
        print(f"Minimum viable set MISSING: {mv}")
        sys.exit(1)
    if not all_ok:
        print("Some optional artifacts missing (see above) — minimum viable set present.")
        sys.exit(0)
    print("ALL REQUIRED ARTIFACTS PRESENT")
    sys.exit(0)


if __name__ == "__main__":
    main()
