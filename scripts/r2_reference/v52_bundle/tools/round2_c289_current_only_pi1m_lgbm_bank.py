#!/usr/bin/env python3
"""C289 current-only PI1M-SVD + gradient-boosted target bank.

No archive labels, local_eval files, pretrained weights, prior predictions, or
Kaggle state are read.  PI1M is used only as unlabeled text for a from-scratch
character n-gram SVD representation.  Target model family selection is based
only on grouped OOF R2 inside official current train labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282
import round2_c284_current_only_pi1m_svd_reference as c284


TARGETS = tuple(reference.TARGETS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard(path: Path) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden input path: {path}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def maccs_matrix(molecules: list[Any]) -> np.ndarray:
    rows = np.zeros((len(molecules), 167), dtype=np.float32)
    for i, mol in enumerate(molecules):
        fp = MACCSkeys.GenMACCSKeys(mol)
        rows[i] = np.asarray([int(fp.GetBit(j)) for j in range(167)], dtype=np.float32)
    return rows


def sanitize_dense(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(out) | (np.abs(out) > 1.0e12)
    if bad.any():
        out = out.copy()
        out[bad] = np.nan
    return out


def grouped_oof(model_factory, x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for OOF")
    oof = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = model_factory(seed + fold)
        model.fit(x[tr], y[tr])
        oof[va] = np.asarray(model.predict(x[va]), dtype=np.float64)
    if not np.isfinite(oof).all():
        raise RuntimeError("Non-finite OOF prediction")
    return oof


def clipped_r2(y: np.ndarray, pred: np.ndarray) -> tuple[float, np.ndarray]:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1.0e-8)
    clipped = np.clip(np.asarray(pred, dtype=np.float64), q01 - 2.0 * margin, q99 + 2.0 * margin)
    return float(r2_score(y, clipped)), clipped


def clip_to_train_range(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1.0e-8)
    return np.clip(np.asarray(pred, dtype=np.float64), q01 - 2.0 * margin, q99 + 2.0 * margin)


def model_factories(target: str, rows: int) -> dict[str, Any]:
    large = target in {"tg", "egc"}
    factories: dict[str, Any] = {
        "ridge_10": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=10.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "ridge_100": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=100.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "extra_trees": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=500 if large else 700,
                min_samples_leaf=3 if large else 2,
                max_features=0.55 if large else 0.75,
                random_state=seed,
                n_jobs=4,
            ),
        ),
        "random_forest": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=350 if large else 500,
                min_samples_leaf=3 if large else 2,
                max_features=0.45 if large else 0.70,
                random_state=seed,
                n_jobs=4,
            ),
        ),
        "hist_gbdt": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                learning_rate=0.035,
                max_iter=450 if large else 650,
                l2_regularization=0.05,
                max_leaf_nodes=31 if large else 15,
                min_samples_leaf=20 if large else 8,
                random_state=seed,
            ),
        ),
    }
    if lgb is not None:
        factories["lightgbm"] = lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            lgb.LGBMRegressor(
                objective="regression",
                n_estimators=900 if large else 700,
                learning_rate=0.025 if large else 0.03,
                num_leaves=63 if large else 31,
                min_child_samples=25 if large else 8,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.75,
                reg_alpha=0.01,
                reg_lambda=0.10,
                random_state=seed,
                n_jobs=4,
                verbosity=-1,
            ),
        )
    if xgb is not None:
        factories["xgboost"] = make_xgb_factory(large)
    return factories


def parse_models(value: str, available: dict[str, Any]) -> dict[str, Any]:
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    if not requested:
        raise RuntimeError("No models requested")
    bad = [name for name in requested if name not in available]
    if bad:
        raise RuntimeError(f"Requested unavailable models {bad}; available={sorted(available)}")
    return {name: available[name] for name in requested}


def make_xgb_factory(large: bool):
    def factory(seed: int):
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                n_estimators=650 if large else 500,
                learning_rate=0.025 if large else 0.035,
                max_depth=4 if large else 3,
                min_child_weight=8 if large else 3,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.02,
                reg_lambda=0.20,
                random_state=seed,
                n_jobs=4,
            ),
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pi1m-limit", type=int, default=300_000)
    parser.add_argument("--pi1m-svd-components", type=int, default=192)
    parser.add_argument("--pi1m-hash-features", type=int, default=131_072)
    parser.add_argument("--morgan-bits", type=int, default=1024)
    parser.add_argument("--models", default="ridge_10,ridge_100,extra_trees,random_forest,hist_gbdt,lightgbm,xgboost")
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path in (data_dir, output, run_dir):
        guard(path)
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    progress_path.write_text(json.dumps({"stage": "started", "created_at": datetime.now().astimezone().isoformat()}) + "\n")

    train, test, inputs = c282.load_current_only_inputs(data_dir)
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    progress_path.open("a").write(json.dumps({"stage": "rdkit_ready", "keys": len(keys), "descriptors": len(descriptor_names)}) + "\n")

    pi1m_config = {
        **c284.DEFAULT_CONFIG,
        "seed": int(args.seed),
        "pi1m_limit": int(args.pi1m_limit),
        "pi1m_hash_features": int(args.pi1m_hash_features),
        "pi1m_svd_components": int(args.pi1m_svd_components),
    }
    pi1m_features, pi1m_report = c284.pi1m_svd_features(
        keys=keys,
        pi1m_path=data_dir / "PI1M.csv",
        config=pi1m_config,
    )
    progress_path.open("a").write(json.dumps({"stage": "pi1m_svd_ready", **pi1m_report}) + "\n")

    morgan2 = reference.morgan_count_matrix(molecules, radius=2, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    morgan3 = reference.morgan_count_matrix(molecules, radius=3, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    maccs = maccs_matrix(molecules)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    base_dense = np.hstack([descriptor, physical, pi1m_features, morgan2, morgan3, maccs]).astype(np.float32)
    base_dense = sanitize_dense(base_dense)
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(args.morgan_bits))
    key_to_index = {key: i for i, key in enumerate(keys)}

    predictions = np.full(len(test), np.nan, dtype=np.float64)
    target_reports: dict[str, Any] = {}
    detail_rows = []
    for target in TARGETS:
        target_train = pooled[pooled["target_type"].eq(target)].reset_index(drop=True)
        target_test = test[test["target_type"].eq(target)].reset_index(drop=False)
        train_idx = np.asarray([key_to_index[x] for x in target_train["canonical"]], dtype=int)
        test_idx = np.asarray([key_to_index[x] for x in target_test["canonical"]], dtype=int)
        y = target_train["target"].to_numpy(float)
        groups = target_train["canonical"].astype(str).to_numpy(object)
        dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        x_train = dense[train_idx]
        x_test = dense[test_idx]

        candidates = {}
        reports = {}
        selected_factories = parse_models(args.models, model_factories(target, len(y)))
        for name, factory in selected_factories.items():
            t_model = time.time()
            try:
                oof = grouped_oof(factory, x_train, y, groups, int(args.seed))
                score, clipped = clipped_r2(y, oof)
                candidates[name] = clipped
                reports[name] = {"oof_r2": score, "elapsed_seconds": float(time.time() - t_model)}
                progress_path.open("a").write(json.dumps({"stage": "model_oof", "target": target, "model": name, **reports[name]}) + "\n")
            except Exception as exc:
                reports[name] = {"error": repr(exc), "elapsed_seconds": float(time.time() - t_model)}
                progress_path.open("a").write(json.dumps({"stage": "model_failed", "target": target, "model": name, **reports[name]}) + "\n")
        if not candidates:
            raise RuntimeError(f"No successful models for {target}")
        best_name = max(candidates, key=lambda name: reports[name]["oof_r2"])
        final_model = selected_factories[best_name](int(args.seed) + 999)
        final_model.fit(x_train, y)
        final_pred_raw = np.asarray(final_model.predict(x_test), dtype=np.float64)
        final_pred = clip_to_train_range(y, final_pred_raw)

        # Allowed exact same-target current-train override. This affects only
        # rows whose official current train already contains the same canonical
        # structure and target.
        lookup = target_train.groupby("canonical")["target"].mean().to_dict()
        overrides = 0
        for local_pos, row in enumerate(target_test.itertuples(index=False)):
            can = row.canonical
            if can in lookup:
                final_pred[local_pos] = float(lookup[can])
                overrides += 1
        predictions[target_test["index"].to_numpy(int)] = final_pred
        target_reports[target] = {
            "rows": int(len(y)),
            "test_rows": int(len(target_test)),
            "models": reports,
            "selected_model": best_name,
            "selected_oof_r2": float(reports[best_name]["oof_r2"]),
            "exact_current_train_overrides": int(overrides),
        }
        for rid, pred in zip(target_test["id"].to_numpy(int), final_pred, strict=True):
            detail_rows.append({"id": int(rid), "target_type": target, "target": float(pred), "selected_model": best_name})

    if not np.isfinite(predictions).all():
        raise RuntimeError("Missing/non-finite final predictions")
    submission = pd.DataFrame({"id": test["id"].to_numpy(int), "target": predictions})
    if len(submission) != 4940 or not np.array_equal(submission["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Output row contract failed")
    submission.to_csv(output, index=False)
    pd.DataFrame(detail_rows).sort_values("id").to_csv(run_dir / "prediction_detail.csv", index=False)
    report = {
        "schema_version": "ppp.round2.c289.current-only-pi1m-lgbm-bank.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": vars(args),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": Chem.rdBase.rdkitVersion,
            "lightgbm": getattr(lgb, "__version__", None) if lgb is not None else None,
            "xgboost": getattr(xgb, "__version__", None) if xgb is not None else None,
        },
        "inputs": {
            **inputs,
            "PI1M.csv": {
                "path": str(data_dir / "PI1M.csv"),
                "sha256": sha256_file(data_dir / "PI1M.csv"),
                "bytes": (data_dir / "PI1M.csv").stat().st_size,
            },
        },
        "features": {
            "keys": int(len(keys)),
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "pi1m": pi1m_report,
            "morgan_count_radii": [2, 3],
            "morgan_bits_each": int(args.morgan_bits),
            "maccs_bits": 167,
            "dense_shape": list(base_dense.shape),
        },
        "target_reports": target_reports,
        "mean_selected_oof_r2": float(np.mean([target_reports[t]["selected_oof_r2"] for t in TARGETS])),
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(submission)), "bytes": output.stat().st_size},
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", vars(args))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    manifest_lines = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest_lines.append(f"{sha256_file(path)}  {path.name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    progress_path.open("a").write(json.dumps({"stage": "finished", "output_sha256": report["output"]["sha256"], "elapsed_seconds": report["elapsed_seconds"]}) + "\n")
    print(json.dumps({"output": report["output"], "mean_selected_oof_r2": report["mean_selected_oof_r2"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
