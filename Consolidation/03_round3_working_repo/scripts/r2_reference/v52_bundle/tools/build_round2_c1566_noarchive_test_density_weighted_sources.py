#!/usr/bin/env python3
"""C1566 noarchive test-density weighted target source generator.

Official-input-only experiment:

* reads current train.csv and test.csv only;
* uses test SMILES only as unlabeled covariates to estimate train-row
  similarity/density weights for the requested target distribution;
* trains target-local models from scratch;
* writes complete one-target replacement CSVs over a branch-local base carrier;
* never reads local_eval/external_label/nonofficial files and never touches Kaggle.

Post-freeze scoring and any later compound assembly are separate steps.
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
from rdkit import Chem, DataStructs, RDLogger
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import build_round2_c385_archive_weak_target_model_zoo as c385
import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as graph
import round2_c282_current_only_reference as c282


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
SCHEMA = "ppp.round2.c1566.noarchive-test-density-weighted-sources.v1"
SEED = 20260808


try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def guard_path(path: Path, *, role: str, require_without_archive: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if "with_archive" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for noarchive run: {path}")
    if require_without_archive and "/without_archive/" not in low:
        raise RuntimeError(f"{role} path must stay in /without_archive/: {path}")
    if role in {"output dir", "run dir"} and "polymer prediction challenge round 2" not in low:
        raise RuntimeError(f"{role} outside Round 2 boundary: {path}")


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets supplied")
    invalid = [target for target in targets if target not in TARGETS]
    if invalid:
        raise RuntimeError(f"Invalid targets: {invalid}")
    return targets


def parse_sources(value: str) -> list[str]:
    available = ("ridge_20", "ridge_80", "ridge_250", "extra_trees", "hgb", "lightgbm")
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested:
        raise RuntimeError("No sources supplied")
    invalid = [item for item in requested if item not in available]
    if invalid:
        raise RuntimeError(f"Invalid sources: {invalid}; available={available}")
    if lgb is None and "lightgbm" in requested:
        raise RuntimeError("lightgbm requested but package is not installed")
    return requested


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path, role="base candidate", require_without_archive=True)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema/count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(np.int64), ids):
        raise RuntimeError(f"Base IDs/order do not match test: {path}")
    if not np.isfinite(frame["target"].to_numpy(np.float64)).all():
        raise RuntimeError("Base candidate contains non-finite values")
    return frame


def tanimoto_top_stats(fingerprints: list[Any], train_idx: np.ndarray, test_idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    test_fps = [fingerprints[index] for index in test_idx]
    if not test_fps:
        raise RuntimeError("No target test fingerprints")
    max_sim = np.empty(len(train_idx), dtype=np.float64)
    top_mean = np.empty(len(train_idx), dtype=np.float64)
    take = min(k, len(test_fps))
    for row, index in enumerate(train_idx):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[index], test_fps), dtype=np.float64)
        max_sim[row] = float(np.max(sims))
        top = np.partition(sims, -take)[-take:]
        top_mean[row] = float(np.mean(top))
    return max_sim, top_mean


def make_weights(max_sim: np.ndarray, top_mean: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(max_sim), dtype=np.float64)
    base = max_sim if mode.startswith("max") else top_mean
    power = 1.0
    if mode.endswith("p2"):
        power = 2.0
    elif mode.endswith("p4"):
        power = 4.0
    scaled = np.clip(base, 0.0, 1.0) ** power
    weights = 0.20 + 2.80 * scaled
    weights = weights / np.mean(weights)
    return weights.astype(np.float64)


def impute_scale(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_imp = imputer.fit_transform(x_train)
    test_imp = imputer.transform(x_test)
    return train_imp, test_imp, scaler.fit_transform(train_imp), scaler.transform(test_imp)


def fit_predict_source(
    source: str,
    x_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
) -> np.ndarray:
    train_imp, test_imp, train_scaled, test_scaled = impute_scale(x_train_raw, x_test_raw)
    if source.startswith("ridge"):
        alpha = float(source.split("_", 1)[1])
        model = Ridge(alpha=alpha, solver="lsqr", max_iter=5000, tol=1.0e-4)
        model.fit(train_scaled, y, sample_weight=sample_weight)
        return np.asarray(model.predict(test_scaled), dtype=np.float64)
    if source == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=320,
            min_samples_leaf=1,
            max_features=0.65,
            random_state=seed,
            n_jobs=4,
        )
        model.fit(train_imp, y, sample_weight=sample_weight)
        return np.asarray(model.predict(test_imp), dtype=np.float64)
    if source == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=6,
            l2_regularization=0.15,
            random_state=seed,
        )
        model.fit(train_imp, y, sample_weight=sample_weight)
        return np.asarray(model.predict(test_imp), dtype=np.float64)
    if source == "lightgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is unavailable")
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=420,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=6,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=0.20,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(train_imp, y, sample_weight=sample_weight)
        return np.asarray(model.predict(test_imp), dtype=np.float64)
    raise RuntimeError(f"Unknown source: {source}")


def grouped_oof_source(
    source: str,
    x_train_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    unique = np.unique(groups)
    n_splits = min(5, len(unique))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for OOF")
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for fold, (tr, va) in enumerate(splitter.split(x_train_raw, y, groups=groups)):
        pred = fit_predict_source(source, x_train_raw[tr], x_train_raw[va], y[tr], weights[tr], seed + fold)
        pred = reference.clip_prediction(y[tr], pred)
        oof[va] = pred
        rows.append({"fold": int(fold), "rows": int(len(va)), "r2": float(r2_score(y[va], pred))})
    if not np.isfinite(oof).all():
        raise RuntimeError("Non-finite OOF")
    return oof, rows


def tanimoto_knn_predict(
    fingerprints: list[Any],
    y_global: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    *,
    k: int,
    power: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    train_fps = [fingerprints[index] for index in train_idx]
    pred_fps = [fingerprints[index] for index in pred_idx]
    y = y_global[train_idx]
    if sample_weight is None:
        sw = np.ones(len(train_idx), dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
    take = min(k, len(train_idx))
    out = np.empty(len(pred_idx), dtype=np.float64)
    for row, fp in enumerate(pred_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=np.float64)
        chosen = np.argpartition(sims, -take)[-take:]
        weights = (np.maximum(sims[chosen], 1.0e-6) ** power) * sw[chosen]
        out[row] = float(np.dot(weights, y[chosen]) / np.sum(weights))
    return out


def grouped_tanimoto_oof(
    fingerprints: list[Any],
    train_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    *,
    k: int,
    power: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Not enough groups")
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for fold, (tr, va) in enumerate(splitter.split(train_idx, y, groups=groups)):
        y_global = np.full(len(fingerprints), np.nan, dtype=np.float64)
        y_global[train_idx[tr]] = y[tr]
        pred = tanimoto_knn_predict(fingerprints, y_global, train_idx[tr], train_idx[va], k=k, power=power, sample_weight=weights[tr])
        pred = reference.clip_prediction(y[tr], pred)
        oof[va] = pred
        rows.append({"fold": int(fold), "rows": int(len(va)), "r2": float(r2_score(y[va], pred))})
    return oof, rows


def safe_slug(text: str) -> str:
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in ("-", "_") else "-")
    return "".join(keep)[:96]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--targets", default="tg,egc,egb,ei,eea,nc,eps")
    parser.add_argument("--morgan-bits", type=int, default=256)
    parser.add_argument("--density-k", type=int, default=25)
    parser.add_argument("--sources", default="ridge_20,ridge_80,ridge_250,extra_trees,hgb")
    parser.add_argument("--weight-modes", default="uniform,max_p1,max_p2,top_p1,top_p2")
    parser.add_argument("--skip-tanimoto", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    started = time.time()
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    base_path = Path(args.base_csv)
    if not base_path.is_absolute():
        base_path = (root / base_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    for path, role in ((data_dir, "data dir"), (base_path, "base candidate")):
        guard_path(path, role=role, require_without_archive=(role == "base candidate"))
    guard_path(output_dir, role="output dir", require_without_archive=True)
    guard_path(run_dir, role="run dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output dir: {output_dir}")
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse run dir: {run_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    append_jsonl(progress_path, {"stage": "started", "created_at": datetime.now().astimezone().isoformat()})

    train, test, inputs = c282.load_current_only_inputs(data_dir)
    archive = train.iloc[0:0].copy()
    _, pooled = reference.build_label_pool(train, archive)
    for frame in (pooled, test):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
    ids = test["id"].to_numpy(np.int64)
    if len(test) != 4940 or not np.array_equal(ids, np.arange(1, 4941)):
        raise RuntimeError("Unexpected official test IDs")
    base = load_base(base_path, ids)
    base_values = base["target"].to_numpy(np.float64)
    active_targets = parse_targets(args.targets)

    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    morgan2 = reference.morgan_count_matrix(molecules, radius=2, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    morgan3 = reference.morgan_count_matrix(molecules, radius=3, bits=int(args.morgan_bits)).toarray().astype(np.float32)
    maccs = c385.maccs_matrix(molecules)
    grammar = graph.grammar_features(molecules).astype(np.float32)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    base_dense = c385.sanitize_dense(np.hstack([descriptor, physical, morgan2, morgan3, maccs, grammar]).astype(np.float32))
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
    append_jsonl(progress_path, {"stage": "features_ready", "keys": len(keys), "dense_shape": [int(x) for x in base_dense.shape]})

    source_names = parse_sources(args.sources)
    weight_modes = [item.strip().lower() for item in args.weight_modes.split(",") if item.strip()]
    invalid_modes = [item for item in weight_modes if item not in {"uniform", "max_p1", "max_p2", "top_p1", "top_p2"}]
    if invalid_modes:
        raise RuntimeError(f"Invalid weight modes: {invalid_modes}")
    if not weight_modes:
        raise RuntimeError("No weight modes supplied")
    records: list[dict[str, Any]] = []
    target_reports: dict[str, Any] = {}
    sequence = 0
    manifest_path = output_dir / "manifest.jsonl"
    for target in active_targets:
        target_train = pooled.loc[pooled["target_type"].eq(target)].reset_index(drop=True)
        target_test = test.loc[test["target_type"].eq(target)].reset_index(drop=False)
        train_idx = np.asarray([key_to_index[value] for value in target_train["canonical"]], dtype=np.int64)
        test_idx = np.asarray([key_to_index[value] for value in target_test["canonical"]], dtype=np.int64)
        y = target_train["target"].to_numpy(np.float64)
        groups = np.asarray([c385.no_stereo(value) for value in target_train["canonical"].astype(str)], dtype=object)
        dense = c385.sanitize_dense(reference.target_dense_features(base_dense, cross_values, cross_available, target))
        x_train = dense[train_idx]
        x_test = dense[test_idx]
        max_sim, top_mean = tanimoto_top_stats(fingerprints, train_idx, test_idx, int(args.density_k))
        target_report: dict[str, Any] = {
            "train_rows": int(len(target_train)),
            "test_rows": int(len(target_test)),
            "density": {
                "max_mean": float(np.mean(max_sim)),
                "max_p10": float(np.quantile(max_sim, 0.10)),
                "max_p90": float(np.quantile(max_sim, 0.90)),
                "top_mean_mean": float(np.mean(top_mean)),
            },
            "sources": {},
        }
        append_jsonl(progress_path, {"stage": "target_started", "target": target, "train_rows": len(target_train), "test_rows": len(target_test)})

        target_positions = target_test["index"].to_numpy(np.int64)
        raw_lookup = target_train.groupby("canonical")["target"].median().to_dict()
        exact_overrides = 0
        for source in source_names:
            for mode in weight_modes:
                if mode == "uniform" and source not in {"ridge_80", "extra_trees", "hgb", "lightgbm"}:
                    continue
                weights = make_weights(max_sim, top_mean, mode)
                source_key = f"{source}-{mode}"
                source_seed = int(args.seed) + 10000 * TARGETS.index(target) + 97 * len(target_report["sources"])
                append_jsonl(progress_path, {"stage": "source_started", "target": target, "source": source_key})
                try:
                    oof, folds = grouped_oof_source(source, x_train, y, groups, weights, source_seed)
                    pred = fit_predict_source(source, x_train, x_test, y, weights, source_seed + 1009)
                    pred = reference.clip_prediction(y, pred)
                    for local_pos, row in enumerate(target_test.itertuples(index=False)):
                        if row.canonical in raw_lookup:
                            pred[local_pos] = float(raw_lookup[row.canonical])
                            exact_overrides += 1
                    source_report = {
                        "oof_r2": float(r2_score(y, oof)),
                        "folds": folds,
                        "weight_mode": mode,
                        "weight_min": float(np.min(weights)),
                        "weight_max": float(np.max(weights)),
                    }
                except Exception as exc:
                    target_report["sources"][source_key] = {"error": repr(exc)}
                    append_jsonl(progress_path, {"stage": "source_failed", "target": target, "source": source_key, "error": repr(exc)})
                    continue
                result = base_values.copy()
                result[target_positions] = pred
                if not np.isfinite(result).all():
                    raise RuntimeError(f"Non-finite output for {target}/{source_key}")
                sequence += 1
                output = output_dir / f"R2-C1566-without_archive-{target}-{safe_slug(source_key)}-over-C1535-20260808.csv"
                if output.exists():
                    raise RuntimeError(f"Refusing overwrite: {output}")
                pd.DataFrame({"id": ids, "target": result}).to_csv(output, index=False)
                record = {
                    "schema_version": SCHEMA,
                    "sequence": sequence,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "branch": "without_archive",
                    "classification": "OFFICIAL_SOURCE_OVER_LOCAL_BASE_DIAGNOSTIC",
                    "target": target,
                    "source_name": source,
                    "weight_mode": mode,
                    "changed_rows": int(len(target_positions)),
                    "local_eval_read_by_builder": False,
                    "external_label_file_read_by_builder": False,
                    "nonofficial_file_read_by_builder": False,
                    "archive_labels_used": False,
                    "test_used_as_unlabeled_covariates": True,
                    "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
                    "source_report": source_report,
                    "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(result)), "bytes": output.stat().st_size},
                }
                records.append(record)
                target_report["sources"][source_key] = source_report
                append_jsonl(progress_path, {"stage": "source_finished", "target": target, "source": source_key, "oof_r2": source_report["oof_r2"]})

        if args.skip_tanimoto:
            target_report["tanimoto_skipped"] = True
            target_reports[target] = target_report
            append_jsonl(progress_path, {"stage": "target_finished", "target": target, "records": sequence})
            continue

        # Tanimoto weighted local sources.
        for k, power, mode in ((12, 3.0, "uniform"), (24, 4.0, "top_p1"), (36, 4.0, "top_p2")):
            weights = make_weights(max_sim, top_mean, mode)
            source_key = f"tanimoto_k{k}_p{int(power)}-{mode}"
            append_jsonl(progress_path, {"stage": "source_started", "target": target, "source": source_key})
            y_global = np.full(len(keys), np.nan, dtype=np.float64)
            y_global[train_idx] = y
            try:
                oof, folds = grouped_tanimoto_oof(fingerprints, train_idx, y, groups, weights, k=k, power=power)
                pred = tanimoto_knn_predict(fingerprints, y_global, train_idx, test_idx, k=k, power=power, sample_weight=weights)
                pred = reference.clip_prediction(y, pred)
                for local_pos, row in enumerate(target_test.itertuples(index=False)):
                    if row.canonical in raw_lookup:
                        pred[local_pos] = float(raw_lookup[row.canonical])
                source_report = {
                    "oof_r2": float(r2_score(y, oof)),
                    "folds": folds,
                    "weight_mode": mode,
                    "k": int(k),
                    "power": float(power),
                }
            except Exception as exc:
                target_report["sources"][source_key] = {"error": repr(exc)}
                append_jsonl(progress_path, {"stage": "source_failed", "target": target, "source": source_key, "error": repr(exc)})
                continue
            result = base_values.copy()
            result[target_positions] = pred
            sequence += 1
            output = output_dir / f"R2-C1566-without_archive-{target}-{safe_slug(source_key)}-over-C1535-20260808.csv"
            pd.DataFrame({"id": ids, "target": result}).to_csv(output, index=False)
            record = {
                "schema_version": SCHEMA,
                "sequence": sequence,
                "created_at": datetime.now().astimezone().isoformat(),
                "branch": "without_archive",
                "classification": "OFFICIAL_SOURCE_OVER_LOCAL_BASE_DIAGNOSTIC",
                "target": target,
                "source_name": "tanimoto_knn",
                "weight_mode": mode,
                "changed_rows": int(len(target_positions)),
                "local_eval_read_by_builder": False,
                "external_label_file_read_by_builder": False,
                "nonofficial_file_read_by_builder": False,
                "archive_labels_used": False,
                "test_used_as_unlabeled_covariates": True,
                "base": {"path": str(base_path), "sha256": sha256_file(base_path), "bytes": base_path.stat().st_size},
                "source_report": source_report,
                "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(result)), "bytes": output.stat().st_size},
            }
            records.append(record)
            target_report["sources"][source_key] = source_report
            append_jsonl(progress_path, {"stage": "source_finished", "target": target, "source": source_key, "oof_r2": source_report["oof_r2"]})

        target_report["exact_current_overrides_total_events"] = int(exact_overrides)
        target_reports[target] = target_report
        append_jsonl(progress_path, {"stage": "target_finished", "target": target, "records": sequence})

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "schema_version": SCHEMA + ".summary",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "classification": "OFFICIAL_SOURCE_OVER_LOCAL_BASE_DIAGNOSTIC_SWEEP",
        "official_current_train_used": True,
        "archive_labels_used": False,
        "local_eval_read_by_builder": False,
        "external_label_file_read_by_builder": False,
        "nonofficial_file_read_by_builder": False,
        "test_used_as_unlabeled_covariates": True,
        "config": vars(args),
        "inputs": inputs,
        "features": {
            "keys": int(len(keys)),
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "morgan_bits_each": int(args.morgan_bits),
            "maccs_bits": int(maccs.shape[1]),
            "grammar_features": int(grammar.shape[1]),
            "dense_shape": [int(x) for x in base_dense.shape],
        },
        "target_reports": target_reports,
        "candidate_count": int(len(records)),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "bytes": manifest_path.stat().st_size},
        "elapsed_seconds": float(time.time() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "rdkit": Chem.rdBase.rdkitVersion,
            "lightgbm": getattr(lgb, "__version__", None) if lgb is not None else None,
            "platform": platform.platform(),
        },
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "reference": sha256_file(root / "tools/initial_reference_pipeline.py"),
            "current_only_loader": sha256_file(root / "tools/round2_c282_current_only_reference.py"),
            "c385_features": sha256_file(root / "tools/build_round2_c385_archive_weak_target_model_zoo.py"),
            "graph": sha256_file(root / "tools/round2_c097_graph_grammar_hgb_full.py"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "config.json", vars(args))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    manifest_lines = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest_lines.append(f"{sha256_file(path)}  {path.name}")
    manifest_lines.append(f"{sha256_file(manifest_path)}  OUTPUT_MANIFEST {manifest_path}")
    for record in records:
        manifest_lines.append(f"{record['output']['sha256']}  OUTPUT {record['output']['path']}")
    for name, digest in summary["source_hashes"].items():
        manifest_lines.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    append_jsonl(progress_path, {"stage": "finished", "candidate_count": len(records), "elapsed_seconds": summary["elapsed_seconds"]})
    print(json.dumps({"candidate_count": len(records), "manifest": str(manifest_path), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
