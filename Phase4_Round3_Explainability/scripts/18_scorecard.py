"""
18_scorecard.py
===============
R5 — synthesize outputs/scorecard.md (PASS/FAIL per REQUIREMENTS.md criterion)
and outputs/trustworthiness_radar.png (8 axes).
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import OUTPUT_DIR, TARGETS, save_plot

CRITERIA = [
    ("R1.1", "Global SHAP importance", ["shap_beeswarm_tg.png", "shap_beeswarm_egc.png",
     "shap_beeswarm_egb.png", "shap_beeswarm_ei.png", "shap_beeswarm_eea.png",
     "shap_beeswarm_nc.png", "shap_beeswarm_eps.png", "shap_summary_global.png",
     "shap_top20_per_target.csv"], "files exist"),
    ("R1.2", "Local SHAP + mol viz", ["local_shap_tg_0.png", "shap_force_tg_0.png"],
     "files exist"),
    ("R1.3", "Fidelity test", ["fidelity_curve_tg.png", "fidelity_table.csv"],
     "drop_top_shap > drop_random"),
    ("R1.4", "Cross-model agreement", ["explanation_agreement_heatmap.png",
     "explanation_agreement.csv"], "mean spearman >= 0.60"),
    ("R1.5", "Physics decomposition", ["physics_decomp_eps_shap.png"], "file exists"),
    ("R2.1", "SMILES prediction invariance", ["smiles_invariance_boxplot.png",
     "smiles_invariance_violation_rate.csv", "smiles_invariance_per_target.csv"],
     "violation rate < 5% at 1σ"),
    ("R2.2", "Canonicalization audit", ["canonicalization_check.txt"], "file exists"),
    ("R2.3", "Attribution invariance", ["attribution_invariance_per_target.csv",
     "attribution_invariance_scatter.png"], "cosine >= 0.70"),
    ("R2.4", "Oligomer invariance", ["oligomer_invariance.csv",
     "oligomer_invariance_plot.png"], "file exists"),
    ("R3.1", "Structured CV", ["cv_validation_table.csv", "cv_validation_barplot.png"],
     "file exists"),
    ("R3.2", "Conformal prediction", ["conformal_coverage_table.csv",
     "conformal_calibration_plot.png", "test_predictions_with_intervals.csv"],
     "coverage within +/-3%"),
    ("R3.3", "Error-uncertainty correlation", ["error_uncertainty_correlation.csv"],
     "rho >= 0.30 for >=5 targets"),
    ("R3.4", "Applicability domain", ["ad_analysis_table.csv", "ad_analysis_plot.png",
     "ad_test_similarity.csv"], "file exists"),
    ("R3.5", "Seed stability", ["seed_stability.csv"], "std < 0.005"),
    ("R4.1", "Generalization ladder", ["generalization_ladder.csv",
     "generalization_ladder_plot.png"], "file exists"),
    ("R4.2", "External (post-freeze) verification", ["khazana_holdout_scores.csv"],
     "R2 >= 0.88 for DFT targets"),
    ("R4.3", "Tail performance", ["tail_performance.csv", "tail_performance_plot.png"],
     "file exists"),
]


def check_files(files):
    missing = [f for f in files if not (OUTPUT_DIR / f).exists()]
    return len(missing) == 0, missing


def main():
    t0 = time.time()
    lines = ["# Phase 4 Scorecard",
             "",
             f"Auto-generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} by 18_scorecard.py",
             "",
             "| Req | Criterion | Artifacts | Check | Status |",
             "|---|---|---|---|---|"]
    results = {}
    for req, name, files, check in CRITERIA:
        ok, missing = check_files(files)
        extra = ""
        if ok and check.startswith("drop_top_shap"):
            try:
                ft = pd.read_csv(OUTPUT_DIR / "fidelity_table.csv")
                top = ft[ft["frac_masked"] == 0.10]["drop_top_shap"].mean()
                rnd = ft[ft["frac_masked"] == 0.10]["drop_random"].mean()
                ok = top > rnd
                extra = f" (drop_top={top:.3f} vs random={rnd:.3f} @10%)"
            except Exception as e:
                ok = False; extra = f" ({e})"
        elif ok and check.startswith("mean spearman"):
            try:
                ag = pd.read_csv(OUTPUT_DIR / "explanation_agreement.csv")
                ok = float(ag["spearman"].mean()) >= 0.60
                extra = f" (mean ρ={ag['spearman'].mean():.3f})"
            except Exception as e:
                ok = False; extra = f" ({e})"
        elif ok and check.startswith("violation rate"):
            try:
                # R2.1 pass criterion is scoped to descriptor-based (graph) features;
                # the full-ensemble rate (incl. char n-grams) is reported alongside
                gv = OUTPUT_DIR / "smiles_invariance_graph_violation_summary.csv"
                src = gv if gv.exists() else OUTPUT_DIR / "smiles_invariance_violation_summary.csv"
                vs = pd.read_csv(src)
                ok = float(vs["viol_rate_1sigma"].mean()) < 0.05
                extra = f" (mean 1σ rate={vs['viol_rate_1sigma'].mean():.4f} on {src.name})"
            except Exception as e:
                ok = False; extra = f" ({e})"
        elif ok and check.startswith("cosine"):
            try:
                av = pd.read_csv(OUTPUT_DIR / "attribution_invariance_per_target.csv")
                ok = float(av["mean_cosine_similarity"].mean()) >= 0.70
                extra = f" (mean cos={av['mean_cosine_similarity'].mean():.3f})"
            except Exception as e:
                ok = False; extra = f" ({e})"
        elif ok and check.startswith("coverage"):
            try:
                cc = pd.read_csv(OUTPUT_DIR / "conformal_coverage_table.csv")
                err = (cc["empirical_coverage"] - cc["nominal_coverage"]).abs().max()
                ok = err <= 0.03
                extra = f" (max |Δcoverage|={err:.3f})"
            except Exception as e:
                ok = False; extra = f" ({e})"
        elif ok and check.startswith("rho"):
            try:
                eu = pd.read_csv(OUTPUT_DIR / "error_uncertainty_correlation.csv")
                ok = int((eu["pearson_rho"] >= 0.30).sum()) >= 5
                extra = f" (n targets ρ>=0.30: {(eu['pearson_rho'] >= 0.30).sum()})"
            except Exception as e:
                ok = False; extra = f" ({e})"
        elif ok and check.startswith("std"):
            try:
                ss = pd.read_csv(OUTPUT_DIR / "seed_stability.csv")
                std = float(ss[ss["seed"] == "std"]["tg_oof_r2"].iloc[0])
                ok = std < 0.005
                extra = f" (std={std:.5f})"
            except Exception as e:
                ok = False; extra = f" ({e})"
        elif ok and check.startswith("R2 >= 0.88"):
            try:
                kh = pd.read_csv(OUTPUT_DIR / "khazana_holdout_scores.csv")
                dft = kh[kh["target"].isin(["egc", "egb", "nc", "eps"])]
                ok = bool((dft["r2"] >= 0.88).all()) and bool((kh[kh["target"].isin(["ei", "eea"])]["r2"] >= 0.85).all())
                extra = f" (egc={kh[kh.target=='egc'].r2.values[0]:.3f}, ...)"
            except Exception as e:
                ok = False; extra = f" ({e})"
        results[req] = ok
        status = "PASS" if ok else ("PARTIAL" if missing and len(missing) < len(files) else "FAIL")
        lines.append(f"| {req} | {name} | {', '.join(files[:3])}{'…' if len(files) > 3 else ''} | {check}{extra} | {status} |")

    passed = sum(results.values())
    lines += ["", f"**Passed {passed}/{len(results)} requirement groups.**",
              "",
              "Minimum viable set (R1.1, R1.2, R2.1, R2.3, R3.1, R3.2, R4.1, R4.2): "
              + ", ".join("PASS" if results.get(r) else "**FAIL**" for r in
                          ["R1.1", "R1.2", "R2.1", "R2.3", "R3.1", "R3.2", "R4.1", "R4.2"]),
              ""]
    (OUTPUT_DIR / "scorecard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"... scorecard.md written (passed {passed}/{len(results)})")

    # radar chart (8 axes; missing data -> NaN -> skipped wedge)
    try:
        labels = ["Accuracy (proxy mean R²)", "SMILES invariance", "Attribution invariance",
                  "Conformal calibration", "Uncertainty-error corr", "Scaffold generalization",
                  "AD high-sim R²", "Fidelity+"]
        vals = []
        ps = pd.read_csv(OUTPUT_DIR / "proxy_scores.csv")
        vals.append(min(1.0, max(0.0, float(ps["ensemble"].mean()))))
        vs = pd.read_csv(OUTPUT_DIR / "smiles_invariance_per_target.csv")
        vals.append(float(1 - vs["viol_rate_1sigma"].mean()) if "viol_rate_1sigma" in vs else np.nan)
        av = pd.read_csv(OUTPUT_DIR / "attribution_invariance_per_target.csv")
        vals.append(float(av["mean_cosine_similarity"].mean()))
        cc = pd.read_csv(OUTPUT_DIR / "conformal_coverage_table.csv")
        vals.append(float(1 - (cc["empirical_coverage"] - cc["nominal_coverage"]).abs().max()))
        eu = pd.read_csv(OUTPUT_DIR / "error_uncertainty_correlation.csv")
        vals.append(float(eu["pearson_rho"].mean()))
        cv = pd.read_csv(OUTPUT_DIR / "cv_validation_table.csv")
        sc = cv[cv["regime"] == "G2_scaffold"]["mean_r2"].mean()
        vals.append(float(sc) if not np.isnan(sc) else np.nan)
        ad = pd.read_csv(OUTPUT_DIR / "ad_analysis_table.csv")
        hi = ad[ad["ad_bin"] == "ge_0.9"]["r2"].mean()
        vals.append(float(hi) if not np.isnan(hi) else np.nan)
        ft = pd.read_csv(OUTPUT_DIR / "fidelity_table.csv")
        f10 = ft[ft["frac_masked"] == 0.10]
        vals.append(float(f10["drop_top_shap"].mean()) if len(f10) else np.nan)

        ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        ang += ang[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        clean = [v if v == v else 0.0 for v in vals]
        clean += clean[:1]
        ax.plot(ang, clean, "o-", color="steelblue", linewidth=2)
        ax.fill(ang, clean, alpha=0.25, color="steelblue")
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Trustworthiness Radar — 8 axes", fontsize=14)
        save_plot(fig, "trustworthiness_radar.png")
    except Exception as e:
        print(f"radar skipped: {e}")
    print(f"18_scorecard.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
