"""
G1_html_report.py
=================
EXP-G1 - bundle every output artifact into a single self-contained
TRUSTWORTHINESS_REPORT.html (PNGs inline as base64, CSVs as HTML tables,
sections per judging axis, pass/fail from scorecard.md).
"""
import base64
import time

import pandas as pd

from helpers import OUTPUT_DIR

SECTIONS = [
    ("1. Quantitative Performance", ["proxy_scores.csv", "khazana_holdout_scores.csv", "cv_validation_table.csv"]),
    ("2. Model Explainability (R1)", ["shap_summary_global.png", "shap_beeswarm_tg.png", "shap_beeswarm_egc.png",
     "shap_beeswarm_nc.png", "shap_top20_per_target.csv", "fidelity_table.csv",
     "explanation_agreement_heatmap.png", "physics_decomp_eps_shap.png", "linear_probe_results.csv",
     "linear_probe_heatmap_tg.png", "structural_counterfactuals.csv"]),
    ("3. Polymer Invariance (R2)", ["smiles_invariance_boxplot.png", "smiles_invariance_per_target.csv",
     "attribution_invariance_scatter.png", "attribution_invariance_per_target.csv",
     "oligomer_invariance_plot.png", "activation_patch_invariance_plot.png", "canonicalization_check.txt"]),
    ("4. Methodology & Reliability (R3)", ["cv_validation_barplot.png", "conformal_calibration_plot.png",
     "conformal_coverage_table.csv", "error_vs_uncertainty_scatter_tg.png",
     "error_uncertainty_correlation.csv", "ad_analysis_plot.png", "ad_test_similarity.csv",
     "seed_stability.csv", "reliability_tiers_validation.csv"]),
    ("5. Proven Generalization (R4)", ["generalization_ladder_plot.png", "generalization_ladder.csv",
     "khazana_scatter_egc.png", "khazana_scatter_nc.png", "tail_performance_plot.png", "tail_performance.csv"]),
    ("6. Extended Mechanistic Studies", ["activation_patch_invariance.csv", "causal_tracing_summary.csv",
     "attribution_patch_modality_heatmap.png", "counterfactual_directions_tg.csv",
     "uq_comparison_table.csv", "shift_aware_conformal.csv", "physics_route_comparison.csv",
     "feature_ablation_results.csv", "oracle_sweep_scores.csv"]),
]


def csv_to_html(name):
    p = OUTPUT_DIR / name
    if not p.exists():
        return "<p><em>" + name + "</em> (missing)</p>"
    try:
        df = pd.read_csv(p)
        return df.head(200).to_html(index=False, classes="tbl")
    except Exception as e:
        return "<p>" + name + ": " + str(e) + "</p>"


def png_to_img(name):
    p = OUTPUT_DIR / name
    if not p.exists():
        return "<p><em>" + name + "</em> (missing)</p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return ('<img src="data:image/png;base64,' + b64 + '" alt="' + name +
            '" class="plot"/>')


def render(name):
    p = OUTPUT_DIR / name
    if not p.exists():
        return "<p><em>" + name + "</em> (missing)</p>"
    if name.endswith(".png"):
        return png_to_img(name)
    if name.endswith(".csv"):
        return csv_to_html(name)
    if name.endswith(".txt"):
        return "<pre>" + p.read_text()[:4000] + "</pre>"
    return "<p>" + name + "</p>"


def main():
    t0 = time.time()
    html = []
    html.append("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Phase 4 - Trustworthiness Report</title>")
    html.append("<style>body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1100px;color:#222}"
                "h1{color:#1a3a6b}h2{border-bottom:2px solid #eee;padding-bottom:4px;margin-top:3rem}"
                ".plot{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:8px 0}"
                ".tbl{border-collapse:collapse;font-size:12px}.tbl td,.tbl th{border:1px solid #ddd;padding:3px 6px}"
                ".tbl tr:nth-child(even){background:#f6f8fa}</style></head><body>")
    html.append("<h1>Phase 4 - Explainability, Robustness &amp; Generalization</h1>")
    html.append("<p>Generated " + str(pd.Timestamp.now()) + ". Every artifact below is produced by the "
                "proxy-analysis suite from official Round 3 data only (proxy models = Ridge/ExtraTrees/LightGBM "
                "on the V57 Stage-A feature stack).</p>")
    sc = OUTPUT_DIR / "scorecard.md"
    if sc.exists():
        html.append("<h2>Scorecard</h2><pre>" + sc.read_text()[:6000] + "</pre>")
    for title, files in SECTIONS:
        html.append("<h2>" + title + "</h2>")
        for f in files:
            html.append(render(f))
    html.append("</body></html>")
    (OUTPUT_DIR / "TRUSTWORTHINESS_REPORT.html").write_text("\n".join(html))
    print("G1_html_report.py DONE in {:.0f}s - outputs/TRUSTWORTHINESS_REPORT.html".format(time.time() - t0))


if __name__ == "__main__":
    main()
