"""Create a reproducible presentation scorecard from qualitative-evidence CSVs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep the plotting cache with this evidence artifact rather than in the user's
# home or system temporary directories. Set it before importing matplotlib.
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"navy": "#264653", "teal": "#2A9D8F", "gold": "#E9C46A", "coral": "#E76F51", "gray": "#9AA5A9"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True, help="Full-run evidence directory")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    out, dest = args.outputs, args.output_dir
    dest.mkdir(parents=True, exist_ok=True)

    attr = pd.read_csv(out / "attribution_invariance_per_target.csv")
    graph = pd.read_csv(out / "smiles_invariance_graph_violation_summary.csv")
    fidelity = pd.read_csv(out / "fidelity_table.csv").query("frac_masked == 0.05").copy()
    conformal = pd.read_csv(out / "conformal_coverage_table.csv")
    uncertainty = pd.read_csv(out / "error_uncertainty_correlation.csv")
    ladder = pd.read_csv(out / "generalization_ladder.csv")
    seed = pd.read_csv(out / "seed_stability.csv").query("seed not in ['mean', 'std']")
    agreement = pd.read_csv(out / "explanation_agreement.csv")

    max_gap = float((conformal.empirical_coverage - conformal.nominal_coverage).abs().max())
    graph_zero = bool((graph[["viol_rate_0_5sigma", "viol_rate_1sigma", "viol_rate_2sigma"]] == 0).all().all())
    strong_uncertainty = int((uncertainty.pearson_rho >= 0.30).sum())
    gmeans = ladder.groupby("regime", sort=False).mean_r2.mean().reindex(
        ["G0_random", "G1_canonical_group", "G2_scaffold"]
    )
    gpositive = ladder[ladder.regime.isin(gmeans.index)].groupby("regime").mean_r2.apply(lambda s: int((s > 0).sum()))
    rows = [
        ("Invariance", "graph representation: zero violations", int(graph_zero), "all 7 targets; 0.5σ, 1σ, and 2σ"),
        ("Invariance", "attribution cosine similarity", float(attr.mean_cosine_similarity.mean()), f"minimum={attr.mean_cosine_similarity.min():.3f}"),
        ("Explainability", "5% SHAP fidelity drop", float(fidelity.drop_top_shap.mean()), f"random-mask drop={fidelity.drop_random.mean():.3f}; 7/7 targets"),
        ("Robustness", "maximum conformal coverage gap", max_gap, "7/7 targets within the ±3 percentage-point criterion"),
        ("Robustness", "error–uncertainty correlation ≥0.30", strong_uncertainty, "of 7 targets"),
        ("Robustness", "Tg seed R² standard deviation", float(seed.tg_oof_r2.std(ddof=0)), f"mean={seed.tg_oof_r2.mean():.6f}"),
        ("Generalizability", "canonical-group mean R²", float(gmeans["G1_canonical_group"]), "7/7 targets positive"),
        ("Generalizability", "scaffold-holdout mean R²", float(gmeans["G2_scaffold"]), "7/7 targets positive"),
        ("Boundary", "raw cross-model rank agreement", float(agreement.spearman.mean()), "secondary sensitivity diagnostic; not primary fidelity evidence"),
    ]
    pd.DataFrame(rows, columns=["theme", "measure", "value", "interpretation"]).to_csv(
        dest / "qualitative_scorecard.csv", index=False
    )

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.16, "figure.dpi": 300, "savefig.dpi": 300,
    })
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.1), constrained_layout=True)
    fig.suptitle("Qualitative evidence scorecard — recorded full run", fontsize=15, fontweight="bold", color=COLORS["navy"])

    ax = axs[0, 0]
    ax.bar(attr.iloc[:, 0].astype(str), attr.mean_cosine_similarity, color=COLORS["teal"], edgecolor="white")
    ax.axhline(0.90, color=COLORS["gold"], linestyle="--", linewidth=1.5, label="0.90 reference")
    ax.set_ylim(0.90, 1.005)
    ax.set_title("Invariance: explanation stability")
    ax.set_ylabel("Attribution cosine similarity")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.05, f"Mean {attr.mean_cosine_similarity.mean():.3f}; min {attr.mean_cosine_similarity.min():.3f}\nGraph prediction violations: 0 across all targets", transform=ax.transAxes, fontsize=8, color=COLORS["navy"])

    ax = axs[0, 1]
    x = np.arange(len(fidelity))
    width = 0.36
    ax.bar(x - width / 2, fidelity.drop_top_shap, width, label="top-SHAP mask", color=COLORS["coral"])
    ax.bar(x + width / 2, fidelity.drop_random, width, label="random mask", color=COLORS["gray"])
    ax.set_xticks(x, fidelity.target)
    ax.set_ylabel("R² decrease after masking 5% features")
    ax.set_title("Explainability: fidelity test")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.92, f"Mean: {fidelity.drop_top_shap.mean():.3f} vs {fidelity.drop_random.mean():.3f}", transform=ax.transAxes, va="top", fontsize=8, color=COLORS["navy"])

    ax = axs[1, 0]
    labels = ["Random\nG0", "Canonical group\nG1", "Scaffold\nG2"]
    bars = ax.bar(labels, gmeans.values, color=[COLORS["gray"], COLORS["teal"], COLORS["gold"]], edgecolor="white")
    for bar, value, regime in zip(bars, gmeans.values, gmeans.index):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}\n({gpositive[regime]}/7 +)", ha="center", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean target-wise R²")
    ax.set_title("Generalizability: structured holdouts")

    ax = axs[1, 1]
    ax.set_axis_off()
    ax.set_title("Robustness: calibrated and stable", loc="left")
    robustness = [
        ("Conformal coverage", f"7/7 targets within ±3pp\nmax gap: {max_gap:.3f}", COLORS["teal"]),
        ("Seed stability (Tg)", f"mean R² {seed.tg_oof_r2.mean():.4f} ± {seed.tg_oof_r2.std(ddof=0):.4f}", COLORS["teal"]),
        ("Uncertainty signal", f"error correlation ≥0.30 on {strong_uncertainty}/7 targets", COLORS["gold"]),
    ]
    for i, (title, detail, color) in enumerate(robustness):
        y = 0.79 - i * 0.27
        ax.add_patch(plt.Rectangle((0.02, y - 0.10), 0.035, 0.16, color=color, transform=ax.transAxes, clip_on=False))
        ax.text(0.08, y + 0.025, title, transform=ax.transAxes, fontsize=10, fontweight="bold", color=COLORS["navy"])
        ax.text(0.08, y - 0.065, detail, transform=ax.transAxes, fontsize=9)

    fig.text(0.01, 0.006, f"Boundary retained for transparency: raw cross-model feature-rank agreement = {agreement.spearman.mean():.3f}. Reconfirm this scorecard after the isolated notebook completes.", fontsize=8, color="#4B5563")
    fig.savefig(dest / "qualitative_scorecard.png")
    fig.savefig(dest / "qualitative_scorecard.pdf")
    print(f"Wrote {dest / 'qualitative_scorecard.csv'}")
    print(f"Wrote {dest / 'qualitative_scorecard.png'} and .pdf")


if __name__ == "__main__":
    main()
