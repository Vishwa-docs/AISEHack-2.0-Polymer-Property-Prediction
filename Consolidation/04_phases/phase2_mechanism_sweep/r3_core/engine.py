"""Per-target grouped-CV training engine with panels.

Real pipeline: loads official data, builds features from scratch, trains one
per-target model per fold (structure-grouped), produces OOF + test predictions,
evaluates per-target R2/MAE + similarity/availability panels, writes
metrics.json / predictions.csv / decision.md.  Oracle-free.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as d
from . import features as f
from . import metrics as m
from . import models as mo
from . import panels as pn


def run_protocol(
    *,
    name: str,
    exp_id: str,
    output_dir: Path,
    feature_fn,
    model_fn,
    n_splits: int = 5,
    seed: int = d.SEED,
    targets: tuple = d.TARGETS,
    data_dir: Path | None = None,
    smoke: bool = False,
) -> dict[str, object]:
    """Train per-target with grouped folds; write OOF metrics + test CSV.

    feature_fn(smiles_list) -> np.ndarray (rows x features)
    model_fn(X, y) -> object with .predict(X)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train, test = d.load_official_data(data_dir)
    train, test = d.add_structure_keys(train, test)

    n_rows = 400 if smoke else len(train)
    if smoke:
        train = train.iloc[:n_rows].copy()
        # keep every target represented in smoke mode
        for t in targets:
            if not (train["target_type"] == t).any():
                extra = d.load_official_data(data_dir)[0]
                extra = extra[extra["target_type"] == t].head(20)
                train = pd.concat([train, extra], ignore_index=True)
        train, _ = d.add_structure_keys(train, test)

    all_smiles = list(train["smiles"]) + list(test["smiles"])
    X_all = feature_fn(all_smiles)
    X_train = X_all[: len(train)]
    X_test = X_all[len(train):]

    target_type = train["target_type"].to_numpy(object)
    oof = np.full(len(train), np.nan)
    test_preds = np.zeros(len(test))
    fold_r2 = {t: [] for t in targets}
    per_target_rows = {t: int((train["target_type"] == t).sum()) for t in targets}

    for target in targets:
        mask = (train["target_type"] == target).to_numpy()
        pos = np.where(mask)[0]
        if len(pos) < 2 * n_splits:
            continue
        y_full = train["target"].to_numpy(float)
        folds = d.grouped_folds(train, target, n_splits=n_splits, seed=seed)
        fold_test = []
        for tr_idx, va_idx in folds:
            X_tr, y_tr = X_train[tr_idx], y_full[tr_idx]
            X_va, y_va = X_train[va_idx], y_full[va_idx]
            model = model_fn(X_tr, y_tr)
            oof[va_idx] = model.predict(X_va)
            fold_test.append(model.predict(X_test))
        test_preds += np.mean(fold_test, axis=0) / len(fold_test)
        valid = np.isfinite(y_full) & np.isfinite(oof)
        if valid.sum() >= 2:
            from sklearn.metrics import r2_score
            fold_r2[target].append(float(r2_score(y_full[valid], oof[valid])))

    metrics = m.report_metrics(target_type, train["target"].to_numpy(float), oof)
    metrics["fold_stats"] = {t: m.fold_metrics(v) for t, v in fold_r2.items() if v}
    metrics["per_target_rows"] = per_target_rows
    metrics["elapsed_seconds"] = float(time.time() - start)

    # Panels
    try:
        sim = pn.tanimoto_similarity(list(train["smiles"]), list(test["smiles"]))
        metrics["similarity_bins"] = m.similarity_bins(
            test["target_type"].to_numpy(object), np.full(len(test), np.nan), test_preds, sim
        )
    except Exception as exc:
        metrics["similarity_bins"] = {"error": str(exc)}

    # Write artifacts (VALUES ONLY in output dir)
    m.write_metrics(output_dir, metrics)
    m.write_predictions(output_dir, test["id"].to_numpy(int), test_preds)
    # OOF values (train rows): id, target_type, y_true, y_pred
    oof_frame = pd.DataFrame({
        "id": train["id"].to_numpy(int) if "id" in train.columns else np.arange(len(train)),
        "target_type": target_type,
        "y_true": train["target"].to_numpy(float),
        "y_pred": oof,
    })
    oof_frame.to_csv(output_dir / "oof_values.csv", index=False)
    (output_dir / "decision.md").write_text(
        f"# {exp_id} - {name}\n\nReal grouped-CV run. Mean OOF R2: {metrics['mean_r2']:.6f}\n", encoding="utf-8"
    )
    return metrics
