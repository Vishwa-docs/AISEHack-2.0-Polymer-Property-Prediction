"""Render the compact, report-safe EDA figure for Report_V3.

All values are recorded in ARCHITECTURE.md §0–1 and outputs/CAPTIONS.md.
The chart intentionally separates sample imbalance from cross-property availability
so labels, legend and annotations never overlap.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ["Tg", "Egc", "Egb", "Ei", "Eea", "n", "ε"]
TRAIN_ROWS = [4143, 2028, 337, 222, 221, 229, 229]
# Percentage of test polymers represented in training under any measured property.
CROSS_PROPERTY_SUPPORT = [28.0, 49.8, 89.3, 98.0, 98.0, 98.7, 98.7]
COLORS = ["#D97706", "#188977", "#188977", "#188977", "#188977", "#6260A8", "#6260A8"]


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), gridspec_kw={"wspace": 0.38})
    x = np.arange(len(TARGETS))

    bars = axes[0].bar(x, TRAIN_ROWS, color=COLORS, width=0.7)
    axes[0].set_title("Training labels are highly uneven")
    axes[0].set_ylabel("Number of labelled polymers")
    axes[0].set_xticks(x, TARGETS)
    axes[0].set_ylim(0, 4750)
    axes[0].grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, TRAIN_ROWS):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 95, f"{value:,}",
                     ha="center", va="bottom", fontsize=7.5, color="#303030")
    axes[0].text(0.5, -0.29, "Every target still receives the same 1/7 score weight.",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=8, color="#454545")

    bars = axes[1].bar(x, CROSS_PROPERTY_SUPPORT, color=COLORS, width=0.7)
    axes[1].set_title("Cross-property support differs by target")
    axes[1].set_ylabel("Test polymers also seen in train (%)")
    axes[1].set_xticks(x, TARGETS)
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, CROSS_PROPERTY_SUPPORT):
        # Place labels inside bars to keep the adjacent 98% values readable.
        axes[1].text(bar.get_x() + bar.get_width() / 2, value - 3.5, f"{value:.1f}%",
                     ha="center", va="top", fontsize=7.5, color="white", fontweight="bold",
                     rotation=90)
    axes[1].text(0.5, -0.29, "Almost no exact polymer–property pair is repeated.",
                 transform=axes[1].transAxes, ha="center", va="top", fontsize=8, color="#454545")

    fig.savefig(OUT / "report_v3_eda.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "report_v3_eda.pdf", bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "report_v3_eda.png")


if __name__ == "__main__":
    main()
