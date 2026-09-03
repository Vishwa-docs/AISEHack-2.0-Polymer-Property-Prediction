"""Validate the four qualitative evidence claims needed for release review."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    out = args.outputs

    conformal = pd.read_csv(out / "conformal_coverage_table.csv")
    uncertainty = pd.read_csv(out / "error_uncertainty_correlation.csv")
    agreement = pd.read_csv(out / "explanation_agreement.csv")
    augmentation = pd.read_csv(out / "augmentation_experiment.csv")

    coverage_error = float((conformal.empirical_coverage - conformal.nominal_coverage).abs().max())
    strong_uncertainty_targets = int((uncertainty.pearson_rho >= 0.30).sum())
    agreement_mean = float(agreement.spearman.mean())
    baseline = augmentation.loc[augmentation.setting == "baseline_no_augment"].iloc[0]
    best_stability = augmentation.loc[augmentation.invariance_std.idxmin()]

    print(f"conformal max_abs_coverage_delta={coverage_error:.6f}; pass={coverage_error <= 0.03}")
    print(f"uncertainty targets rho>=0.30={strong_uncertainty_targets}; pass={strong_uncertainty_targets >= 5}")
    print(f"cross-model mean_spearman={agreement_mean:.6f}; pass={agreement_mean >= 0.60}")
    print("augmentation "
          f"{baseline.setting}: R2={baseline.tg_oof_r2:.6f}, std={baseline.invariance_std:.6f}; "
          f"{best_stability.setting}: R2={best_stability.tg_oof_r2:.6f}, std={best_stability.invariance_std:.6f}")
    required = ["augmentation_experiment.csv", "augmentation_experiment_plot.png",
                "conformal_coverage_table.csv", "conformal_calibration_plot.png",
                "error_uncertainty_correlation.csv", "explanation_agreement.csv"]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required evidence artifacts: {missing}")


if __name__ == "__main__":
    main()
