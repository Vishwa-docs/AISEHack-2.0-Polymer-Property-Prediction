"""Render the report's verified qualitative-evidence figure.

This deliberately excludes calibration coverage, error--uncertainty correlation,
and cross-model rank agreement: the archived artifacts for those checks disagree
and are release-gated pending the isolated end-to-end rerun.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase" / "outputs" / "evidence_tables"
OUT = ROOT / "finals" / "assets"

COLORS = {
    "navy": "#16324F",
    "blue": "#2878B5",
    "teal": "#2A9D8F",
    "amber": "#E9A13B",
    "coral": "#D55E00",
    "ink": "#20252B",
    "muted": "#617180",
    "grid": "#D9E1E8",
}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=COLORS["ink"])


def main():
    inv = pd.read_csv(EVIDENCE / "smiles_invariance_per_target.csv")
    # The first column is an intentionally unnamed target index in the source table.
    attr = (
        pd.read_csv(EVIDENCE / "attribution_invariance_per_target.csv", index_col=0)
        .rename_axis("target")
        .reset_index()
    )
    gen = pd.read_csv(EVIDENCE / "generalization_ladder.csv")
    ad = pd.read_csv(EVIDENCE / "ad_analysis_table.csv")
    graph = pd.read_csv(EVIDENCE / "smiles_invariance_graph_violation_summary.csv")

    graph_max = inv["std_pct_graph_only"].max()
    attr_mean = attr["mean_cosine_similarity"].mean()
    g1 = gen.loc[gen["regime"] == "G1_canonical_group", "mean_r2"].mean()
    g2 = gen.loc[gen["regime"] == "G2_scaffold", "mean_r2"].mean()
    tg = ad[ad["target"] == "tg"].copy()
    order = ["ge_0.9", "0.7-0.9", "0.5-0.7", "lt_0.5"]
    tg["ad_bin"] = pd.Categorical(tg["ad_bin"], categories=order, ordered=True)
    tg = tg.sort_values("ad_bin")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.1), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    bars = ax.bar(["Worst target\nspread", "1σ graph\nviolations"], [graph_max, graph["viol_rate_1sigma"].max()], color=[COLORS["blue"], COLORS["teal"]], width=0.58)
    ax.set_ylim(0, 0.28)
    ax.set_ylabel("Percent / rate")
    ax.set_title("SMILES representation invariance", loc="left", color=COLORS["navy"], fontweight="bold", fontsize=11)
    ax.bar_label(bars, labels=[f"{graph_max:.3f}%", "0 / 7"], padding=3, fontsize=9, color=COLORS["ink"])
    style(ax)

    ax = axes[0, 1]
    values = attr["mean_cosine_similarity"].to_numpy()
    labels = attr["target"].str.upper().replace({"TG": "Tg"}).to_list()
    ax.bar(labels, values, color=COLORS["teal"], width=0.67)
    ax.axhline(0.70, color=COLORS["amber"], linestyle="--", linewidth=1.2, label="pre-registered bar")
    ax.set_ylim(0.65, 1.02)
    ax.set_ylabel("Attribution cosine similarity")
    ax.set_title("Explanation stability", loc="left", color=COLORS["navy"], fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)

    ax = axes[1, 0]
    bars = ax.bar(["Canonical-group\n(G1)", "Scaffold\n(G2)"], [g1, g2], color=[COLORS["blue"], COLORS["amber"]], width=0.58)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean target-wise R²")
    ax.set_title("Generalization under structural splits", loc="left", color=COLORS["navy"], fontweight="bold", fontsize=11)
    ax.bar_label(bars, labels=[f"{g1:.3f}", f"{g2:.3f}"], padding=3, fontsize=9, color=COLORS["ink"])
    style(ax)

    ax = axes[1, 1]
    ax.plot(["≥0.9", "0.7–0.9", "0.5–0.7", "<0.5"], tg["mae"], marker="o", markersize=6, linewidth=2.2, color=COLORS["coral"])
    for x, y in enumerate(tg["mae"]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)
    ax.set_ylabel("Tg MAE (°C)")
    ax.set_xlabel("Nearest-neighbour Tanimoto tier")
    ax.set_title("Robustness boundary: applicability domain", loc="left", color=COLORS["navy"], fontweight="bold", fontsize=11)
    style(ax)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"verified_qualitative_evidence.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
