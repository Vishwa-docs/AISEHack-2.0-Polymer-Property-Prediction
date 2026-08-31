#!/usr/bin/env python3
"""C385 archive weak-target direct model-zoo bank over C379.

This is an local_eval-free source generator for the with-archive branch.  It reads
only official Round 2 files, trains target-local models on current+archive
labels, replaces only requested targets over a branch-local base CSV, and writes
a complete frozen candidate.  Target-wise acceptance is decided later by the
separate post-freeze scorer and splice builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
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

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
DEFAULT_BASE = (
    "experiments/final_submission_runs/with_archive/"
    "R2-C379-ARCHIVE-TARGET-SPLICE-C378-EEA-EGB-EI-EPS-NC-OVER-C369-20260808.csv"
)
SEED = 20260808


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, role: str, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if "without_archive" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for archive run: {path}")
    if allow_output and "/with_archive/" not in low:
        raise RuntimeError(f"{role} path must stay in with_archive namespace: {path}")


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    bad = [target for target in targets if target not in TARGETS]
    if bad:
        raise RuntimeError(f"Invalid targets: {bad}")
    return targets


def parse_models(value: str, available: dict[str, Any]) -> dict[str, Any]:
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    if not requested:
        raise RuntimeError("No models requested")
    bad = [name for name in requested if name not in available]
    if bad:
        raise RuntimeError(f"Requested unavailable models {bad}; available={sorted(available)}")
    return {name: available[name] for name in requested}


def no_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


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


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path, role="base candidate")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Base contains non-finite predictions: {path}")
    return frame


def grouped_oof(factory: Any, x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for OOF")
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(x, y, groups=groups)):
        model = factory(seed + fold)
        model.fit(x[tr], y[tr])
        pred = np.asarray(model.predict(x[va]), dtype=np.float64)
        oof[va] = pred
        fold_rows.append({"fold": int(fold), "rows": int(len(va)), "r2": float(r2_score(y[va], pred))})
    if not np.isfinite(oof).all():
        raise RuntimeError("Non-finite OOF prediction")
    return oof, fold_rows


def clipped_by_train(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1.0e-8)
    return np.clip(np.asarray(pred, dtype=np.float64), q01 - 2.0 * margin, q99 + 2.0 * margin)


def model_factories(target: str) -> dict[str, Any]:
    weak = target in {"ei", "eea", "egb", "eps", "nc"}
    factories: dict[str, Any] = {
        "ridge_30": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=30.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "ridge_200": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=200.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "extra_trees": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=450 if weak else 300,
                min_samples_leaf=2 if weak else 4,
                max_features=0.70 if weak else 0.50,
                random_state=seed,
                n_jobs=4,
            ),
        ),
        "hist_gbdt": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                learning_rate=0.035,
                max_iter=420 if weak else 300,
                l2_regularization=0.10,
                max_leaf_nodes=15 if weak else 31,
                min_samples_leaf=8 if weak else 20,
                random_state=seed,
            ),
        ),
    }
    if lgb is not None:
        factories["lightgbm"] = lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            lgb.LGBMRegressor(
                objective="regression",
                n_estimators=550 if weak else 400,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=8 if weak else 20,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.75,
                reg_lambda=0.20,
                random_state=seed,
                n_jobs=4,
                verbosity=-1,
            ),
        )
    return factories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--targets", default="ei,eps,nc,egb,eea")
    parser.add_argument("--models", default="ridge_30,ridge_200,extra_trees,hist_gbdt,lightgbm")
    parser.add_argument("--morgan-bits", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir).resolve()
    base_path = Path(args.base_csv).resolve()
    output = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path, role in ((data_dir, "data dir"), (base_path, "base candidate")):
        guard_path(path, role=role)
    guard_path(output, role="output", allow_output=True)
    guard_path(run_dir, role="run dir")
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    progress_path.write_text(json.dumps({"stage": "started", "created_at": datetime.now().astimezone().isoformat()}) + "\n")

    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    for frame in (pooled, test):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
    ids = test["id"].to_numpy(int)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    base = load_base(base_path, ids)
    predictions = base["target"].to_numpy(float).copy()

    active_targets = parse_targets(args.targets)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: i for i, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    morgan2 = reference.morgan_count_matrix(molecules, radius=2, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    morgan3 = reference.morgan_count_matrix(molecules, radius=3, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    maccs = maccs_matrix(molecules)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    base_dense = sanitize_dense(np.hstack([descriptor, physical, morgan2, morgan3, maccs]).astype(np.float32))
    progress_path.open("a").write(json.dumps({"stage": "features_ready", "keys": len(keys), "dense_shape": list(base_dense.shape)}) + "\n")

    target_reports: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    for target in active_targets:
        target_train = pooled[pooled["target_type"].eq(target)].reset_index(drop=True)
        target_test = test[test["target_type"].eq(target)].reset_index(drop=False)
        train_idx = np.asarray([key_to_index[x] for x in target_train["canonical"]], dtype=int)
        test_idx = np.asarray([key_to_index[x] for x in target_test["canonical"]], dtype=int)
        y = target_train["target"].to_numpy(float)
        groups = np.asarray([no_stereo(value) for value in target_train["canonical"].astype(str)], dtype=object)
        dense = sanitize_dense(reference.target_dense_features(base_dense, cross_values, cross_available, target))
        x_train = dense[train_idx]
        x_test = dense[test_idx]
        candidates: dict[str, np.ndarray] = {}
        reports: dict[str, Any] = {}
        selected_factories = parse_models(args.models, model_factories(target))
        for model_name, factory in selected_factories.items():
            model_started = time.time()
            try:
                oof_raw, folds = grouped_oof(factory, x_train, y, groups, int(args.seed) + 100 * TARGETS.index(target))
                oof = clipped_by_train(y, oof_raw)
                reports[model_name] = {
                    "oof_r2": float(r2_score(y, oof)),
                    "folds": folds,
                    "elapsed_seconds": float(time.time() - model_started),
                }
                candidates[model_name] = oof
                progress_path.open("a").write(json.dumps({"stage": "model_oof", "target": target, "model": model_name, "oof_r2": reports[model_name]["oof_r2"]}) + "\n")
            except Exception as exc:
                reports[model_name] = {"error": repr(exc), "elapsed_seconds": float(time.time() - model_started)}
                progress_path.open("a").write(json.dumps({"stage": "model_failed", "target": target, "model": model_name, "error": repr(exc)}) + "\n")
        if not candidates:
            raise RuntimeError(f"No successful models for {target}")
        best_name = max(candidates, key=lambda name: reports[name]["oof_r2"])
        final_model = selected_factories[best_name](int(args.seed) + 999 + TARGETS.index(target))
        final_model.fit(x_train, y)
        final_pred = clipped_by_train(y, np.asarray(final_model.predict(x_test), dtype=np.float64))
        # Preserve official same-target current/archive exact lookups.
        lookup = target_train.groupby("canonical")["target"].mean().to_dict()
        overrides = 0
        for local_pos, row in enumerate(target_test.itertuples(index=False)):
            if row.canonical in lookup:
                final_pred[local_pos] = float(lookup[row.canonical])
                overrides += 1
        positions = target_test["index"].to_numpy(int)
        predictions[positions] = final_pred
        target_reports[target] = {
            "rows": int(len(y)),
            "test_rows": int(len(target_test)),
            "models": reports,
            "selected_model": best_name,
            "selected_oof_r2": float(reports[best_name]["oof_r2"]),
            "exact_current_or_archive_overrides": int(overrides),
        }
        for rid, pred in zip(target_test["id"].to_numpy(int), final_pred, strict=True):
            detail_rows.append({"id": int(rid), "target_type": target, "target": float(pred), "selected_model": best_name})

    if not np.isfinite(predictions).all():
        raise RuntimeError("Missing/non-finite final predictions")
    submission = pd.DataFrame({"id": ids, "target": predictions})
    if len(submission) != 4940 or not np.array_equal(submission["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Output row contract failed")
    submission.to_csv(output, index=False)
    pd.DataFrame(detail_rows).sort_values("id").to_csv(run_dir / "prediction_detail.csv", index=False)
    report = {
        "schema_version": "ppp.round2.c385.archive-weak-target-model-zoo.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "with_archive",
        "official_current_train_used": True,
        "archive_labels_used": True,
        "archive_file_read": True,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
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
        },
        "inputs": inputs,
        "features": {
            "keys": int(len(keys)),
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "morgan_count_radii": [2, 3],
            "morgan_bits_each": int(args.morgan_bits),
            "maccs_bits": 167,
            "dense_shape": list(base_dense.shape),
        },
        "target_reports": target_reports,
        "mean_selected_oof_r2": float(np.mean([target_reports[target]["selected_oof_r2"] for target in active_targets])),
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(submission)), "bytes": output.stat().st_size},
        "elapsed_seconds": float(time.time() - started),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    manifest_lines = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest_lines.append(f"{sha256_file(path)}  {path.name}")
    manifest_lines.append(f"{sha256_file(output)}  {output}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    progress_path.open("a").write(json.dumps({"stage": "finished", "output_sha256": report["output"]["sha256"], "elapsed_seconds": report["elapsed_seconds"]}) + "\n")
    print(json.dumps({"output": report["output"], "mean_selected_oof_r2": report["mean_selected_oof_r2"], "targets": active_targets, "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
