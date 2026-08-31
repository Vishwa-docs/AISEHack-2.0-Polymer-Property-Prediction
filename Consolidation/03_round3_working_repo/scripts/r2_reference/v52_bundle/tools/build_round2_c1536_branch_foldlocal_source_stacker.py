#!/usr/bin/env python3
"""C1536 branch-local fold/group OOF source stacker.

This is a bounded official-input-only weak-target experiment.  It builds a
small source library for requested targets, creates source OOF predictions with
grouped folds on official train labels only, selects a ridge meta-stacker by
OOF evidence only, then freezes a complete branch-local candidate CSV for
separate post-freeze local_eval scoring.

The builder reads no local_eval/external_label/nonofficial files.  The base CSV is used only as
the carrier for unchanged targets; active target predictions are regenerated
from official train/test inputs.
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
from typing import Any, Callable

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_round2_c385_archive_weak_target_model_zoo as c385
import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as graph
import round2_c282_current_only_reference as c282


TARGETS = tuple(reference.TARGETS)
SCHEMA = "ppp.round2.c1536.branch-foldlocal-source-stacker.v1"
SEED = 20260808


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def guard_path(path: Path, *, role: str, branch: str | None = None, require_branch: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if branch is not None:
        opposite = "without_archive" if branch == "with_archive" else "with_archive"
        if f"/{opposite}/" in low:
            raise RuntimeError(f"Refusing cross-branch {role} path for {branch}: {path}")
        if require_branch and f"/{branch}/" not in low:
            raise RuntimeError(f"{role} path must stay in /{branch}/ namespace: {path}")
    if role in {"output", "run dir"} and "polymer prediction challenge round 2" not in low:
        raise RuntimeError(f"{role} outside Round 2 boundary: {path}")


def parse_csv_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    invalid = [target for target in targets if target not in TARGETS]
    if invalid:
        raise RuntimeError(f"Invalid targets: {invalid}")
    return targets


def load_branch_inputs(data_dir: Path, branch: str) -> dict[str, Any]:
    if branch == "with_archive":
        train, test, archive, inputs = reference.load_inputs(data_dir)
    elif branch == "without_archive":
        train, test, inputs = c282.load_current_only_inputs(data_dir)
        archive = train.iloc[0:0].copy()
    else:
        raise RuntimeError(f"Unknown branch: {branch}")
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    return {
        "train": train,
        "test": test,
        "archive": archive,
        "inputs": inputs,
        "raw_labels": raw_labels,
        "pooled": pooled,
        "keys": keys,
        "key_to_index": {key: index for index, key in enumerate(keys)},
        "molecules": molecules,
    }


def load_base(path: Path, ids: np.ndarray, branch: str) -> pd.DataFrame:
    guard_path(path, role="base candidate", branch=branch, require_branch=True)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base candidate schema/count: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(np.int64), ids):
        raise RuntimeError(f"Base candidate IDs do not match official test: {path}")
    if not np.isfinite(frame["target"].to_numpy(np.float64)).all():
        raise RuntimeError("Base candidate contains non-finite values")
    return frame


def build_dense(parent: dict[str, Any], morgan_bits: int) -> tuple[np.ndarray, dict[str, Any]]:
    molecules = parent["molecules"]
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, parent["keys"])
    morgan2 = reference.morgan_count_matrix(molecules, radius=2, bits=morgan_bits).toarray().astype(np.float32)
    morgan3 = reference.morgan_count_matrix(molecules, radius=3, bits=morgan_bits).toarray().astype(np.float32)
    maccs = c385.maccs_matrix(molecules)
    grammar = graph.grammar_features(molecules).astype(np.float32)
    dense = c385.sanitize_dense(np.hstack([descriptor, physical, morgan2, morgan3, maccs, grammar]).astype(np.float32))
    return dense, {
        "shape": [int(value) for value in dense.shape],
        "rdkit_descriptors": int(len(descriptor_names)),
        "physical_features": int(len(physical_names)),
        "morgan_bits_each": int(morgan_bits),
        "maccs_bits": int(maccs.shape[1]),
        "graph_grammar_features": int(grammar.shape[1]),
    }


def source_factories(target: str) -> dict[str, Callable[[int], Any]]:
    weak = target in {"ei", "eea", "egb", "eps", "nc"}
    return {
        "ridge_8": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=8.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "ridge_32": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=32.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "ridge_128": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=128.0, solver="lsqr", max_iter=5000, tol=1.0e-4),
        ),
        "huber": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            HuberRegressor(alpha=0.02, epsilon=1.35, max_iter=500),
        ),
        "hgb": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                max_iter=220 if weak else 160,
                learning_rate=0.035,
                max_leaf_nodes=15 if weak else 31,
                min_samples_leaf=6 if weak else 18,
                l2_regularization=0.15,
                random_state=seed,
            ),
        ),
        "extra_trees": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=320 if weak else 180,
                min_samples_leaf=1 if weak else 3,
                max_features=0.60 if weak else 0.45,
                random_state=seed,
                n_jobs=4,
            ),
        ),
        "random_forest": lambda seed: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=260 if weak else 160,
                min_samples_leaf=1 if weak else 3,
                max_features=0.60 if weak else 0.45,
                random_state=seed,
                n_jobs=4,
            ),
        ),
    }


def clip(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return reference.clip_prediction(np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64))


def split_groups(groups: np.ndarray) -> GroupKFold:
    unique = len(np.unique(groups))
    if unique < 2:
        raise RuntimeError("Not enough unique groups")
    return GroupKFold(n_splits=min(5, unique))


def grouped_model_oof(
    factory: Callable[[int], Any],
    x_train: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    splitter = split_groups(groups)
    for fold, (tr, va) in enumerate(splitter.split(x_train, y, groups=groups)):
        model = factory(seed + fold)
        model.fit(x_train[tr], y[tr])
        pred = clip(y[tr], np.asarray(model.predict(x_train[va]), dtype=np.float64))
        oof[va] = pred
        fold_reports.append({"fold": int(fold), "rows": int(len(va)), "r2": float(r2_score(y[va], pred))})
    if not np.isfinite(oof).all():
        raise RuntimeError("Non-finite source OOF")
    final = factory(seed + 1009)
    final.fit(x_train, y)
    test_pred = clip(y, np.asarray(final.predict(x_test), dtype=np.float64))
    return oof, test_pred, fold_reports


def tanimoto_predict_batch(
    fingerprints: list[Any],
    y_global: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    *,
    k: int,
    power: float,
) -> np.ndarray:
    train_fps = [fingerprints[index] for index in train_idx]
    pred_fps = [fingerprints[index] for index in pred_idx]
    train_y = y_global[train_idx]
    if not np.isfinite(train_y).all():
        raise RuntimeError("Tanimoto train target contains non-finite values")
    take = min(k, len(train_idx))
    output = np.empty(len(pred_idx), dtype=np.float64)
    for start in range(0, len(pred_idx), 256):
        stop = min(start + 256, len(pred_idx))
        for row, fp in enumerate(pred_fps[start:stop]):
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=np.float64)
            chosen = np.argpartition(sims, -take)[-take:]
            weights = np.maximum(sims[chosen], 1.0e-6) ** power
            output[start + row] = float(np.dot(weights, train_y[chosen]) / np.sum(weights))
    return output


def grouped_tanimoto_oof(
    fingerprints: list[Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    k: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    splitter = split_groups(groups)
    for fold, (tr, va) in enumerate(splitter.split(train_idx, y, groups=groups)):
        y_global = np.full(len(fingerprints), np.nan, dtype=np.float64)
        y_global[train_idx[tr]] = y[tr]
        pred = clip(y[tr], tanimoto_predict_batch(fingerprints, y_global, train_idx[tr], train_idx[va], k=k, power=power))
        oof[va] = pred
        fold_reports.append({"fold": int(fold), "rows": int(len(va)), "r2": float(r2_score(y[va], pred))})
    y_global = np.full(len(fingerprints), np.nan, dtype=np.float64)
    y_global[train_idx] = y
    test_pred = clip(y, tanimoto_predict_batch(fingerprints, y_global, train_idx, test_idx, k=k, power=power))
    return oof, test_pred, fold_reports


def add_physics_sources(
    target: str,
    keys: list[str],
    train_keys: pd.Series,
    test_keys: pd.Series,
    cross_values: np.ndarray,
    key_to_index: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return deterministic deployable physics/cross-property sources.

    These are not fitted to local_eval or test external_labels; they use official train
    cross-property values if present for the same canonical structure.
    Missing rows receive NaN and are handled by the meta preprocessor.
    """
    names: list[str] = []
    train_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    train_index = np.asarray([key_to_index[value] for value in train_keys], dtype=np.int64)
    test_index = np.asarray([key_to_index[value] for value in test_keys], dtype=np.int64)

    def values_for(expr: str, array_index: np.ndarray) -> np.ndarray:
        cv = cross_values[array_index]
        tg, egc, egb, ei, eea, nc, eps = [cv[:, TARGETS.index(name)] for name in TARGETS]
        with np.errstate(invalid="ignore"):
            if expr == "ei_from_egc_eea":
                return egc + eea
            if expr == "ei_from_egb_eea":
                return egb + eea
            if expr == "eea_from_ei_egc":
                return ei - egc
            if expr == "eea_from_ei_egb":
                return ei - egb
            if expr == "egb_from_ei_eea":
                return ei - eea
            if expr == "eps_from_nc2":
                return np.square(nc)
            if expr == "nc_from_sqrt_eps":
                return np.sqrt(np.maximum(eps, 0.0))
        raise RuntimeError(f"Unknown physics expression: {expr}")

    expressions_by_target = {
        "ei": ("ei_from_egc_eea", "ei_from_egb_eea"),
        "eea": ("eea_from_ei_egc", "eea_from_ei_egb"),
        "egb": ("egb_from_ei_eea",),
        "eps": ("eps_from_nc2",),
        "nc": ("nc_from_sqrt_eps",),
    }
    for expr in expressions_by_target.get(target, ()):
        names.append(expr)
        train_columns.append(values_for(expr, train_index))
        test_columns.append(values_for(expr, test_index))
    if not names:
        return [], np.empty((len(train_keys), 0), dtype=np.float64), np.empty((len(test_keys), 0), dtype=np.float64)
    return names, np.column_stack(train_columns), np.column_stack(test_columns)


def meta_features(source_oof: np.ndarray, extra: np.ndarray) -> np.ndarray:
    mean = np.nanmean(source_oof, axis=1, keepdims=True)
    median = np.nanmedian(source_oof, axis=1, keepdims=True)
    std = np.nanstd(source_oof, axis=1, keepdims=True)
    minv = np.nanmin(source_oof, axis=1, keepdims=True)
    maxv = np.nanmax(source_oof, axis=1, keepdims=True)
    return np.hstack([source_oof, mean, median, std, minv, maxv, extra]).astype(np.float64, copy=False)


def select_meta(
    x_meta: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    x_test_meta: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, Any]]:
    candidates: dict[str, Any] = {
        "ridge_1": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=1.0)),
        "ridge_5": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=5.0)),
        "ridge_20": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=20.0)),
        "ridge_80": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=80.0)),
    }
    best_name = ""
    best_oof = np.empty(0, dtype=np.float64)
    best_score = -float("inf")
    reports: dict[str, Any] = {}
    splitter = split_groups(groups)
    for name, template in candidates.items():
        oof = np.full(len(y), np.nan, dtype=np.float64)
        folds: list[dict[str, Any]] = []
        for fold, (tr, va) in enumerate(splitter.split(x_meta, y, groups=groups)):
            model = candidates[name]
            model.fit(x_meta[tr], y[tr])
            pred = clip(y[tr], np.asarray(model.predict(x_meta[va]), dtype=np.float64))
            oof[va] = pred
            folds.append({"fold": int(fold), "rows": int(len(va)), "r2": float(r2_score(y[va], pred))})
        score = float(r2_score(y, oof))
        reports[name] = {"oof_r2": score, "folds": folds}
        if score > best_score:
            best_name = name
            best_oof = oof.copy()
            best_score = score
    final_model = candidates[best_name]
    final_model.fit(x_meta, y)
    test_pred = clip(y, np.asarray(final_model.predict(x_test_meta), dtype=np.float64))
    reports["selected"] = {"name": best_name, "oof_r2": float(best_score)}
    return best_name, best_oof, test_pred, reports


def run_target(
    parent: dict[str, Any],
    dense_base: np.ndarray,
    fingerprints: list[Any],
    target: str,
    enabled_sources: tuple[str, ...],
    seed: int,
    progress_path: Path,
) -> dict[str, Any]:
    pooled = parent["pooled"]
    test = parent["test"]
    key_to_index = parent["key_to_index"]
    target_train = pooled.loc[pooled["target_type"].astype(str).eq(target)].reset_index(drop=True)
    target_test = test.loc[test["target_type"].astype(str).eq(target)].reset_index(drop=True)
    train_idx = np.asarray([key_to_index[value] for value in target_train["canonical"]], dtype=np.int64)
    test_idx = np.asarray([key_to_index[value] for value in target_test["canonical"]], dtype=np.int64)
    y = target_train["target"].to_numpy(np.float64)
    groups = np.asarray([graph.no_stereo(value) for value in target_train["canonical"].astype(str)], dtype=object)
    cross_values, cross_available = reference.cross_property_arrays(pooled, parent["keys"])
    dense = c385.sanitize_dense(reference.target_dense_features(dense_base, cross_values, cross_available, target))
    x_train = dense[train_idx]
    x_test = dense[test_idx]

    source_names: list[str] = []
    source_oof: list[np.ndarray] = []
    source_test: list[np.ndarray] = []
    source_reports: dict[str, Any] = {}
    factories = {name: factory for name, factory in source_factories(target).items() if name in enabled_sources}
    if not factories:
        raise RuntimeError(f"No enabled fitted sources for {target}: {enabled_sources}")
    for pos, (name, factory) in enumerate(factories.items()):
        append_jsonl(progress_path, {"stage": "source_started", "target": target, "source": name})
        oof, test_pred, folds = grouped_model_oof(factory, x_train, y, groups, x_test, seed + 101 * pos)
        source_names.append(name)
        source_oof.append(oof)
        source_test.append(test_pred)
        source_reports[name] = {"oof_r2": float(r2_score(y, oof)), "folds": folds}
        append_jsonl(progress_path, {"stage": "source_finished", "target": target, "source": name, "oof_r2": source_reports[name]["oof_r2"]})

    for k, power in ((8, 3.0), (16, 4.0)):
        name = f"tanimoto_k{k}_p{int(power)}"
        if name not in enabled_sources:
            continue
        append_jsonl(progress_path, {"stage": "source_started", "target": target, "source": name})
        oof, test_pred, folds = grouped_tanimoto_oof(fingerprints, train_idx, test_idx, y, groups, k=k, power=power)
        source_names.append(name)
        source_oof.append(oof)
        source_test.append(test_pred)
        source_reports[name] = {"oof_r2": float(r2_score(y, oof)), "folds": folds}
        append_jsonl(progress_path, {"stage": "source_finished", "target": target, "source": name, "oof_r2": source_reports[name]["oof_r2"]})

    phys_names, phys_train, phys_test = add_physics_sources(
        target,
        parent["keys"],
        target_train["canonical"],
        target_test["canonical"],
        cross_values,
        key_to_index,
    )
    for col, name in enumerate(phys_names):
        values = phys_train[:, col]
        if np.isfinite(values).sum() >= 3:
            fill = np.nanmedian(values[np.isfinite(values)])
            oof = np.where(np.isfinite(values), values, fill).astype(np.float64)
            tpred = np.where(np.isfinite(phys_test[:, col]), phys_test[:, col], fill).astype(np.float64)
            oof = clip(y, oof)
            tpred = clip(y, tpred)
            source_names.append(name)
            source_oof.append(oof)
            source_test.append(tpred)
            source_reports[name] = {
                "oof_r2": float(r2_score(y, oof)),
                "finite_train_rows": int(np.isfinite(values).sum()),
                "finite_test_rows": int(np.isfinite(phys_test[:, col]).sum()),
            }

    oof_matrix = np.column_stack(source_oof)
    test_matrix = np.column_stack(source_test)
    x_meta = meta_features(oof_matrix, phys_train)
    x_test_meta = meta_features(test_matrix, phys_test)
    selected_meta, meta_oof, meta_test, meta_reports = select_meta(x_meta, y, groups, x_test_meta)

    nnls_weights, nnls_intercept, nnls_name, nnls_score = reference.blend_from_oof(y, oof_matrix)
    nnls_oof = clip(y, oof_matrix @ nnls_weights + nnls_intercept)
    nnls_test = clip(y, test_matrix @ nnls_weights + nnls_intercept)
    source_scores = [float(r2_score(y, oof_matrix[:, idx])) for idx in range(oof_matrix.shape[1])]
    best_source_idx = int(np.argmax(source_scores))
    candidates = {
        "meta": (float(r2_score(y, meta_oof)), meta_oof, meta_test),
        "nnls": (float(r2_score(y, nnls_oof)), nnls_oof, nnls_test),
        f"best_source:{source_names[best_source_idx]}": (source_scores[best_source_idx], oof_matrix[:, best_source_idx], test_matrix[:, best_source_idx]),
    }
    selected_name = max(candidates, key=lambda key: candidates[key][0])
    selected_score, selected_oof, selected_test = candidates[selected_name]

    raw_lookup = parent["raw_labels"].groupby(["canonical", "target_type"])["target"].median().to_dict()
    exact_overrides = 0
    selected_test = selected_test.copy()
    for pos, row in enumerate(target_test.itertuples(index=False)):
        key = (row.canonical, target)
        if key in raw_lookup:
            selected_test[pos] = float(raw_lookup[key])
            exact_overrides += 1

    return {
        "target": target,
        "train_rows": int(len(y)),
        "test_rows": int(len(target_test)),
        "groups": int(len(np.unique(groups))),
        "source_names": source_names,
        "source_reports": source_reports,
        "physics_sources": phys_names,
        "meta_reports": meta_reports,
        "selected_name": selected_name,
        "selected_oof_r2": float(selected_score),
        "nnls": {
            "name": nnls_name,
            "oof_r2": float(nnls_score),
            "weights": {name: float(value) for name, value in zip(source_names, nnls_weights, strict=True)},
            "intercept": float(nnls_intercept),
        },
        "exact_branch_label_overrides_on_test": int(exact_overrides),
        "test_ids": target_test["id"].to_numpy(np.int64),
        "test_pred": selected_test.astype(np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--targets", default="ei,eps,nc")
    parser.add_argument("--morgan-bits", type=int, default=192)
    parser.add_argument(
        "--sources",
        default="ridge_8,ridge_32,ridge_128,extra_trees,tanimoto_k8_p3,tanimoto_k16_p4",
        help="comma-separated source names; excludes slow hgb/huber by default",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    started = time.time()
    round2_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (round2_root / data_dir).resolve()
    base_path = Path(args.base_csv)
    if not base_path.is_absolute():
        base_path = (round2_root / base_path).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (round2_root / output).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (round2_root / run_dir).resolve()

    guard_path(base_path, role="base candidate", branch=args.branch, require_branch=True)
    guard_path(output, role="output", branch=args.branch, require_branch=True)
    guard_path(run_dir, role="run dir")
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing overwrite/reuse")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    append_jsonl(progress_path, {"stage": "started", "created_at": datetime.now().astimezone().isoformat(), "branch": args.branch})

    parent = load_branch_inputs(data_dir, args.branch)
    ids = parent["test"]["id"].to_numpy(np.int64)
    base = load_base(base_path, ids, args.branch)
    predictions = base["target"].to_numpy(np.float64).copy()
    dense_base, feature_report = build_dense(parent, int(args.morgan_bits))
    fingerprints = reference.morgan_bits(parent["molecules"], radius=2, bits=4096)
    append_jsonl(progress_path, {"stage": "features_ready", "shape": feature_report["shape"], "archive_rows_used": int(len(parent["archive"]))})

    active_targets = parse_csv_targets(args.targets)
    enabled_sources = tuple(item.strip() for item in args.sources.split(",") if item.strip())
    if not enabled_sources:
        raise RuntimeError("No sources enabled")
    target_reports: dict[str, Any] = {}
    for target in active_targets:
        append_jsonl(progress_path, {"stage": "target_started", "target": target})
        result = run_target(parent, dense_base, fingerprints, target, enabled_sources, int(args.seed) + 10000 * TARGETS.index(target), progress_path)
        target_reports[target] = {key: value for key, value in result.items() if key not in {"test_ids", "test_pred"}}
        id_to_value = dict(zip(result["test_ids"].astype(int), result["test_pred"].astype(float), strict=True))
        mask = parent["test"]["target_type"].astype(str).eq(target).to_numpy()
        predictions[mask] = parent["test"].loc[mask, "id"].astype(int).map(id_to_value).to_numpy(np.float64)
        append_jsonl(progress_path, {"stage": "target_finished", "target": target, "selected": result["selected_name"], "selected_oof_r2": result["selected_oof_r2"]})

    out = pd.DataFrame({"id": ids, "target": predictions})
    if len(out) != 4940 or out["id"].duplicated().any() or not np.array_equal(out["id"].to_numpy(np.int64), np.arange(1, 4941)):
        raise RuntimeError("Output row/order contract failed")
    if not np.isfinite(out["target"].to_numpy(np.float64)).all():
        raise RuntimeError("Output contains non-finite predictions")
    out.to_csv(output, index=False)

    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "classification": "CLEAN_OFFICIAL_ONLY_ACTIVE_TARGET_REGENERATOR_WITH_BASE_CARRIER",
        "official_current_train_used": True,
        "archive_labels_used": bool(args.branch == "with_archive"),
        "archive_rows_used": int(len(parent["archive"])),
        "local_eval_read_by_builder": False,
        "external_label_file_read_by_builder": False,
        "nonofficial_file_read_by_builder": False,
        "prior_prediction_as_training_feature": False,
        "base_candidate_used_for_unchanged_targets_only": str(base_path),
        "pi1m_used": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": vars(args),
        "inputs": parent["inputs"],
        "base_candidate": {"path": str(base_path), "sha256": sha256_file(base_path), "rows": int(len(base))},
        "feature_report": feature_report,
        "active_targets": list(active_targets),
        "target_reports": target_reports,
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": int(len(out)), "bytes": output.stat().st_size},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "rdkit": reference.Chem.rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "current_only_loader": sha256_file(round2_root / "tools/round2_c282_current_only_reference.py"),
            "graph": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "c385_features": sha256_file(round2_root / "tools/build_round2_c385_archive_weak_target_model_zoo.py"),
        },
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", vars(args))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "target": target,
                "selected": payload["selected_name"],
                "selected_oof_r2": payload["selected_oof_r2"],
                "train_rows": payload["train_rows"],
                "test_rows": payload["test_rows"],
            }
            for target, payload in target_reports.items()
        ]
    ).to_csv(run_dir / "component_summary.csv", index=False)
    manifest_lines = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest_lines.append(f"{sha256_file(path)}  {path.name}")
    manifest_lines.append(f"{sha256_file(output)}  OUTPUT {output}")
    for name, digest in report["source_hashes"].items():
        manifest_lines.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    append_jsonl(progress_path, {"stage": "finished", "output_sha256": report["output"]["sha256"], "elapsed_seconds": report["elapsed_seconds"]})
    print(json.dumps({"output": report["output"], "active_targets": list(active_targets), "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
