"""Per-target scoring, fold statistics, panels, and report helpers.

Everything here is oracle-free: it scores predictions against the official
training labels (OOF) using grouped folds.  Post-freeze oracle scoring happens
OUTSIDE this package (in Oracle/) and is never imported by clean code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


def per_target_r2(target_type: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for target in TARGETS:
        mask = target_type == target
        if mask.sum() < 2:
            continue
        out[target] = float(r2_score(y_true[mask], y_pred[mask]))
    return out


def report_metrics(target_type, y_true, y_pred) -> dict[str, object]:
    scores = per_target_r2(target_type, y_true, y_pred)
    mean_r2 = float(np.mean(list(scores.values()))) if scores else float("nan")
    per_target_detail = {}
    for target in TARGETS:
        mask = target_type == target
        if mask.sum() < 2:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        per_target_detail[target] = {
            "r2": float(r2_score(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(np.mean(np.square(yt - yp)))),
            "rows": int(mask.sum()),
        }
    return {
        "mean_r2": mean_r2,
        "per_target": per_target_detail,
        "covered_rows": int(np.isfinite(y_true).sum()),
    }


def fold_metrics(fold_scores: list[float]) -> dict[str, float]:
    arr = np.asarray(fold_scores, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def similarity_bins(target_type, y_true, y_pred, sim: np.ndarray, edges=(0.30, 0.45, 0.55, 0.65, 0.75)) -> dict[str, object]:
    """Panel: per similarity bin R2 (sim = nearest-train Tanimoto per row)."""
    out = {}
    for lo, hi in zip([0.0] + list(edges), list(edges) + [1.01]):
        mask = (sim >= lo) & (sim < hi) & np.isfinite(y_true)
        if mask.sum() < 5:
            continue
        out[f"{lo:.2f}-{hi:.2f}"] = {
            "rows": int(mask.sum()),
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
        }
    return out


def availability_panel(target_type, y_true, y_pred, available_mask: np.ndarray | None = None) -> dict[str, object]:
    """Panel: score split by partner-availability (None => all rows)."""
    if available_mask is None:
        available_mask = np.isfinite(y_true)
    mask = available_mask & np.isfinite(y_true)
    return {
        "available_rows": int(mask.sum()),
        "r2_available": float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() >= 5 else float("nan"),
        "r2_all": float(r2_score(y_true, y_pred)),
    }


def write_metrics(exp_dir: Path, metrics: dict[str, object]) -> Path:
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_predictions(exp_dir: Path, ids: np.ndarray, target: np.ndarray) -> Path:
    import pandas as pd
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": ids, "target": target})
    path = exp_dir / "predictions.csv"
    frame.to_csv(path, index=False)
    return path
