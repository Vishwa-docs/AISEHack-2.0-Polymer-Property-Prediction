"""Leakage-safe grouped-OOF affine calibration experiment.

Reads completed-run proxy OOF/test predictions only. Writes all artifacts next to this file.
No competition submission is modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "fixes" / "isolated_runs" / "outputs"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
GAIN_GATE = 0.005
WIN_FOLD_GATE = 4


def fit_affine(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """OLS affine map y ≈ slope * pred + intercept, with identity fallback."""
    x = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    var = float(np.var(x))
    if not np.isfinite(var) or var < 1e-12:
        return 1.0, 0.0
    slope = float(np.cov(x, y, ddof=0)[0, 1] / var)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def evaluate_target(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = frame["true_value"].to_numpy(float)
    base = frame["oof_ensemble"].to_numpy(float)
    groups = frame["canonical"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    calibrated = np.full(len(frame), np.nan)
    rows = []
    for fold, (train, valid) in enumerate(splitter.split(base, y, groups)):
        slope, intercept = fit_affine(y[train], base[train])
        calibrated[valid] = slope * base[valid] + intercept
        rows.append({
            "fold": fold,
            "baseline_r2": float(r2_score(y[valid], base[valid])),
            "affine_r2": float(r2_score(y[valid], calibrated[valid])),
            "delta_r2": float(r2_score(y[valid], calibrated[valid]) - r2_score(y[valid], base[valid])),
            "slope": slope,
            "intercept": intercept,
        })
    folds = pd.DataFrame(rows)
    baseline = float(r2_score(y, base))
    affine = float(r2_score(y, calibrated))
    delta = affine - baseline
    wins = int((folds.delta_r2 > 0).sum())
    retained = bool(delta >= GAIN_GATE and wins >= WIN_FOLD_GATE)
    slope, intercept = fit_affine(y, base)
    frame = frame.copy()
    frame["oof_affine"] = calibrated
    return {
        "baseline_r2": baseline,
        "affine_r2": affine,
        "delta_r2": delta,
        "positive_folds": wins,
        "retained": retained,
        "full_slope": slope,
        "full_intercept": intercept,
        "folds": rows,
    }, frame


def main() -> None:
    reports: dict[str, dict] = {}
    calibrated_oof = []
    for target in TARGETS:
        source = SOURCE / f"proxy_oof_{target}.csv"
        report, oof = evaluate_target(pd.read_csv(source))
        reports[target] = report
        oof["target_type"] = target
        calibrated_oof.append(oof)
    oof_all = pd.concat(calibrated_oof, ignore_index=True)
    oof_all.to_csv(OUT / "oof_affine_predictions.csv", index=False)
    summary = pd.DataFrame([
        {"target_type": target, **{k: v for k, v in report.items() if k != "folds"}}
        for target, report in reports.items()
    ])
    summary.to_csv(OUT / "calibration_summary.csv", index=False)

    test = pd.read_csv(SOURCE / "test_predictions_with_intervals.csv")
    candidate = test[["id", "target"]].copy()
    candidate["calibration_retained"] = False
    for target, report in reports.items():
        mask = test.target_type.eq(target)
        if report["retained"]:
            candidate.loc[mask, "target"] = (
                report["full_slope"] * candidate.loc[mask, "target"] + report["full_intercept"]
            )
            candidate.loc[mask, "calibration_retained"] = True
    candidate[["id", "target"]].to_csv(OUT / "submission_candidate_oof_affine.csv", index=False)
    with (OUT / "report.json").open("w") as handle:
        json.dump({"gates": {"min_delta_r2": GAIN_GATE, "min_positive_folds": WIN_FOLD_GATE}, "targets": reports}, handle, indent=2)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("retained:", ", ".join(summary.loc[summary.retained, "target_type"]) or "none")
    print("candidate:", OUT / "submission_candidate_oof_affine.csv")


if __name__ == "__main__":
    main()
