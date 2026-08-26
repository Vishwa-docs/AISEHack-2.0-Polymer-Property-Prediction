#!/usr/bin/env python3
"""C323 no-archive Tg/Egc support-gated read-across overlay over C312.

This builder is intentionally narrow:

* branch: without_archive only;
* targets: Tg and Egc only;
* training data: official current train.csv only;
* test data: official current test.csv only;
* base CSV: branch-local frozen C312 used only as fallback/splice carrier;
* no archive labels, local_eval, external_label file, Kaggle action, prior prediction
  training features, or PI1M.

It trains compact structure/read-across arms with grouped OOF metrics, selects
an arm from clean OOF only, applies a fixed support-gated blend to the test rows,
freezes one complete CSV, and leaves local_eval scoring to the separate post-freeze
scoring script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")

TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("tg", "egc")
DEFAULT_BASE = (
    "experiments/final_submission_runs/without_archive/"
    "R2-C312-EI-EEA-TARGET-SPLICE-V3-without_archive-20260808.csv"
)
DEFAULT_BASELINE_REPORT = "experiments/CLEAN_OFFICIAL_ONLY/R2-C282-20260807-current-only-reference-v1/report.json"


@dataclass(frozen=True)
class TargetConfig:
    similarity_threshold: float
    blend_weight: float
    min_clean_oof_delta: float
    ridge_alpha: float
    knn_k: int
    knn_power: float


CONFIGS = {
    "tg": TargetConfig(
        similarity_threshold=0.42,
        blend_weight=0.30,
        min_clean_oof_delta=0.004,
        ridge_alpha=50.0,
        knn_k=24,
        knn_power=5.0,
    ),
    "egc": TargetConfig(
        similarity_threshold=0.36,
        blend_weight=0.35,
        min_clean_oof_delta=0.004,
        ridge_alpha=25.0,
        knn_k=20,
        knn_power=4.0,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_path(path: Path, *, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden path: {path}")
    if not allow_output:
        if "/archive/" in low or low.endswith("/archive") or "with_archive" in low:
            raise RuntimeError(f"Refusing archive/cross-branch input path for no-archive C323: {path}")


def load_base(path: Path, ids: np.ndarray) -> pd.DataFrame:
    guard_path(path)
    if "without_archive" not in str(path):
        raise RuntimeError(f"C323 base must be branch-local without_archive: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), ids):
        raise RuntimeError(f"Invalid base ID order: {path}")
    values = frame["target"].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Base contains non-finite predictions: {path}")
    return frame


def load_current_only(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    guard_path(train_path)
    guard_path(test_path)
    inputs = {
        "train.csv": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "bytes": train_path.stat().st_size,
        },
        "test.csv": {
            "path": str(test_path),
            "sha256": sha256_file(test_path),
            "bytes": test_path.stat().st_size,
        },
    }
    if inputs["train.csv"]["sha256"] != reference.EXPECTED_HASHES["train.csv"]:
        raise RuntimeError("train.csv hash mismatch")
    if inputs["test.csv"]["sha256"] != reference.EXPECTED_HASHES["test.csv"]:
        raise RuntimeError("test.csv hash mismatch")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if list(train.columns) != ["smiles", "target", "target_type"] or len(train) != 7409:
        raise RuntimeError("Unexpected train.csv schema/count")
    if list(test.columns) != ["id", "smiles", "target_type"] or len(test) != 4940:
        raise RuntimeError("Unexpected test.csv schema/count")
    for frame in (train, test):
        frame["target_type"] = frame["target_type"].astype(str).str.lower()
        frame["canonical"] = [reference.canonicalize(value) for value in frame["smiles"]]
    if set(train["target_type"]) != set(TARGETS) or set(test["target_type"]) != set(TARGETS):
        raise RuntimeError("Unexpected target set")
    if test["id"].duplicated().any() or not np.array_equal(test["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("Unexpected test IDs")
    return train, test, inputs


def build_feature_space(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    keys = sorted(set(train["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    dense[~np.isfinite(dense)] = np.nan
    sparse_matrix = sparse.hstack(
        [
            reference.morgan_count_matrix(molecules, radius=2, bits=4096),
            reference.morgan_count_matrix(molecules, radius=3, bits=4096),
            reference.text_matrix(keys, 32768),
        ],
        format="csr",
        dtype=np.float64,
    )
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
    return {
        "keys": keys,
        "key_to_index": {key: index for index, key in enumerate(keys)},
        "dense": dense,
        "sparse": sparse_matrix,
        "fingerprints": fingerprints,
        "feature_report": {
            "rdkit_descriptors": int(len(descriptor_names)),
            "physical_features": int(len(physical_names)),
            "morgan_count_bits": 8192,
            "text_hash_features": 32768,
            "total_sparse_features": int(sparse_matrix.shape[1]),
        },
    }


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    if (folds < 0).any():
        raise RuntimeError("Fold assignment failed")
    return folds


def dense_fold(
    dense: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.asarray(dense[train_indices], dtype=np.float64).copy()
    validation_x = np.asarray(dense[validation_indices], dtype=np.float64).copy()
    limit = float(reference.DEFAULT_CONFIG["dense_abs_limit"])
    train_x[(~np.isfinite(train_x)) | (np.abs(train_x) > limit)] = np.nan
    validation_x[(~np.isfinite(validation_x)) | (np.abs(validation_x) > limit)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(train_x)
    validation_x = imputer.transform(validation_x)
    keep = np.ptp(train_x, axis=0) > 1.0e-12
    if not np.any(keep):
        raise RuntimeError("No nonconstant dense features remained")
    return train_x[:, keep], validation_x[:, keep]


def clip_bounds(y: np.ndarray) -> tuple[float, float]:
    q01, q99 = np.quantile(y, [0.01, 0.99])
    q25, q75 = np.quantile(y, [0.25, 0.75])
    margin = max(float(q75 - q25), float(np.std(y)), 1.0)
    return float(q01 - 2.0 * margin), float(q99 + 2.0 * margin)


def weighted_knn_predict(
    query_fps: list[Any],
    train_fps: list[Any],
    y_train: np.ndarray,
    *,
    k: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    preds = np.empty(len(query_fps), dtype=np.float64)
    max_sims = np.empty(len(query_fps), dtype=np.float64)
    fallback = float(np.median(y_train))
    for row, fp in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=np.float64)
        if sims.size == 0:
            preds[row] = fallback
            max_sims[row] = 0.0
            continue
        max_sims[row] = float(np.max(sims))
        top = np.argsort(sims)[-min(k, len(sims)) :]
        weights = np.power(np.maximum(sims[top], 0.0), power)
        if float(np.sum(weights)) <= 1.0e-12:
            preds[row] = fallback
        else:
            preds[row] = float(np.average(y_train[top], weights=weights))
    return preds, max_sims


def fit_full_dense_models(
    target: str,
    dense: np.ndarray,
    train_indices: np.ndarray,
    y: np.ndarray,
) -> tuple[Any, Any, np.ndarray]:
    train_x = np.asarray(dense[train_indices], dtype=np.float64).copy()
    limit = float(reference.DEFAULT_CONFIG["dense_abs_limit"])
    train_x[(~np.isfinite(train_x)) | (np.abs(train_x) > limit)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(train_x)
    keep = np.ptp(train_x, axis=0) > 1.0e-12
    if not np.any(keep):
        raise RuntimeError(f"No dense features remained for {target}")
    train_x = train_x[:, keep]
    hgb = HistGradientBoostingRegressor(
        max_iter=260 if target == "tg" else 220,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=16 if target == "tg" else 10,
        l2_regularization=0.15,
        random_state=20260808,
    )
    et = ExtraTreesRegressor(
        n_estimators=260,
        min_samples_leaf=4 if target == "tg" else 3,
        max_features=0.60,
        random_state=20260808,
        n_jobs=2,
    )
    hgb.fit(train_x, y)
    et.fit(train_x, y)
    return hgb, et, keep


def predict_full_dense(model: Any, dense: np.ndarray, train_indices: np.ndarray, query_indices: np.ndarray, keep: np.ndarray) -> np.ndarray:
    train_x = np.asarray(dense[train_indices], dtype=np.float64).copy()
    query_x = np.asarray(dense[query_indices], dtype=np.float64).copy()
    limit = float(reference.DEFAULT_CONFIG["dense_abs_limit"])
    train_x[(~np.isfinite(train_x)) | (np.abs(train_x) > limit)] = np.nan
    query_x[(~np.isfinite(query_x)) | (np.abs(query_x) > limit)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(train_x)
    query_x = imputer.transform(query_x)[:, keep]
    return np.asarray(model.predict(query_x), dtype=np.float64)


def evaluate_target(
    target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    base: pd.DataFrame,
    feature_space: dict[str, Any],
    baseline_oof_r2: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = CONFIGS[target]
    key_to_index = feature_space["key_to_index"]
    dense = np.asarray(feature_space["dense"], dtype=np.float64)
    sparse_matrix = feature_space["sparse"]
    fingerprints = feature_space["fingerprints"]

    target_train = train[train["target_type"] == target].reset_index(drop=True)
    y = target_train["target"].to_numpy(float)
    groups = target_train["canonical"].to_numpy(str)
    train_indices = np.asarray([key_to_index[key] for key in target_train["canonical"]], dtype=np.int64)
    folds = grouped_folds(groups)
    low, high = clip_bounds(y)
    oof: dict[str, np.ndarray] = {
        "knn_tanimoto": np.full(len(y), np.nan, dtype=np.float64),
        "sparse_ridge": np.full(len(y), np.nan, dtype=np.float64),
        "dense_hgb": np.full(len(y), np.nan, dtype=np.float64),
        "dense_extra_trees": np.full(len(y), np.nan, dtype=np.float64),
    }
    nearest = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(set(folds)):
        tr = np.flatnonzero(folds != fold)
        va = np.flatnonzero(folds == fold)
        tr_idx = train_indices[tr]
        va_idx = train_indices[va]
        knn_pred, max_sim = weighted_knn_predict(
            [fingerprints[int(index)] for index in va_idx],
            [fingerprints[int(index)] for index in tr_idx],
            y[tr],
            k=cfg.knn_k,
            power=cfg.knn_power,
        )
        oof["knn_tanimoto"][va] = knn_pred
        nearest[va] = max_sim

        ridge = Ridge(alpha=cfg.ridge_alpha, solver="lsqr")
        ridge.fit(sparse_matrix[tr_idx], y[tr])
        oof["sparse_ridge"][va] = np.asarray(ridge.predict(sparse_matrix[va_idx]), dtype=np.float64)

        train_x, validation_x = dense_fold(dense, tr_idx, va_idx)
        hgb = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=31,
            min_samples_leaf=16 if target == "tg" else 10,
            l2_regularization=0.15,
            random_state=20260808 + int(fold),
        )
        et = ExtraTreesRegressor(
            n_estimators=220,
            min_samples_leaf=4 if target == "tg" else 3,
            max_features=0.60,
            random_state=20260818 + int(fold),
            n_jobs=2,
        )
        hgb.fit(train_x, y[tr])
        et.fit(train_x, y[tr])
        oof["dense_hgb"][va] = np.asarray(hgb.predict(validation_x), dtype=np.float64)
        oof["dense_extra_trees"][va] = np.asarray(et.predict(validation_x), dtype=np.float64)
        fold_rows.append({"fold": int(fold), "train_rows": int(len(tr)), "validation_rows": int(len(va))})

    for name, values in oof.items():
        values[:] = np.clip(values, low, high)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite OOF values for {target}/{name}")
    oof["mean_knn_ridge"] = np.clip(0.5 * oof["knn_tanimoto"] + 0.5 * oof["sparse_ridge"], low, high)
    oof["median_all"] = np.clip(
        np.median(
            np.column_stack(
                [oof["knn_tanimoto"], oof["sparse_ridge"], oof["dense_hgb"], oof["dense_extra_trees"]]
            ),
            axis=1,
        ),
        low,
        high,
    )
    arms = {
        name: {
            "oof_r2": float(r2_score(y, values)),
            "delta_vs_current_only_reference_oof": float(r2_score(y, values) - baseline_oof_r2),
            "support_oof_r2": None,
            "support_rows": int(np.sum(nearest >= cfg.similarity_threshold)),
        }
        for name, values in oof.items()
    }
    support = nearest >= cfg.similarity_threshold
    if int(np.sum(support)) >= 20:
        for name, values in oof.items():
            arms[name]["support_oof_r2"] = float(r2_score(y[support], values[support]))
            arms[name]["support_delta_vs_all_reference"] = float(r2_score(y[support], values[support]) - baseline_oof_r2)

    selected_arm = max(arms, key=lambda name: arms[name]["oof_r2"])
    pass_gate = bool(arms[selected_arm]["delta_vs_current_only_reference_oof"] >= cfg.min_clean_oof_delta)

    target_test = test[test["target_type"] == target].copy()
    target_test_indices = np.asarray([key_to_index[key] for key in target_test["canonical"]], dtype=np.int64)
    base_values = base.loc[target_test.index, "target"].to_numpy(float)

    all_train_fps = [fingerprints[int(index)] for index in train_indices]
    query_fps = [fingerprints[int(index)] for index in target_test_indices]
    test_predictions: dict[str, np.ndarray] = {}
    test_predictions["knn_tanimoto"], test_nearest = weighted_knn_predict(
        query_fps,
        all_train_fps,
        y,
        k=cfg.knn_k,
        power=cfg.knn_power,
    )
    ridge = Ridge(alpha=cfg.ridge_alpha, solver="lsqr")
    ridge.fit(sparse_matrix[train_indices], y)
    test_predictions["sparse_ridge"] = np.asarray(ridge.predict(sparse_matrix[target_test_indices]), dtype=np.float64)
    hgb, et, keep = fit_full_dense_models(target, dense, train_indices, y)
    test_predictions["dense_hgb"] = predict_full_dense(hgb, dense, train_indices, target_test_indices, keep)
    test_predictions["dense_extra_trees"] = predict_full_dense(et, dense, train_indices, target_test_indices, keep)
    test_predictions["mean_knn_ridge"] = 0.5 * test_predictions["knn_tanimoto"] + 0.5 * test_predictions["sparse_ridge"]
    test_predictions["median_all"] = np.median(
        np.column_stack(
            [
                test_predictions["knn_tanimoto"],
                test_predictions["sparse_ridge"],
                test_predictions["dense_hgb"],
                test_predictions["dense_extra_trees"],
            ]
        ),
        axis=1,
    )
    raw_selected = np.clip(test_predictions[selected_arm], low, high)
    support_test = test_nearest >= cfg.similarity_threshold
    changed = pass_gate & support_test
    overlay = base_values.copy()
    overlay[changed] = (1.0 - cfg.blend_weight) * base_values[changed] + cfg.blend_weight * raw_selected[changed]
    report = {
        "target": target,
        "train_rows": int(len(target_train)),
        "test_rows": int(len(target_test)),
        "baseline_oof_source": "R2-C282 current-only reference selected_oof_r2",
        "baseline_oof_r2": float(baseline_oof_r2),
        "arms": arms,
        "folds": fold_rows,
        "selected_arm": selected_arm,
        "selected_oof_r2": float(arms[selected_arm]["oof_r2"]),
        "selected_delta_vs_current_only_reference_oof": float(arms[selected_arm]["delta_vs_current_only_reference_oof"]),
        "clean_oof_gate_pass": pass_gate,
        "clean_oof_min_delta": cfg.min_clean_oof_delta,
        "support_threshold": cfg.similarity_threshold,
        "blend_weight": cfg.blend_weight,
        "changed_rows": int(np.sum(changed)),
        "support_rows_test": int(np.sum(support_test)),
        "changed_ids_sha256": hashlib.sha256(
            json.dumps(target_test.loc[changed, "id"].astype(int).tolist(), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "base_target_mean": float(np.mean(base_values)),
        "overlay_target_mean": float(np.mean(overlay)),
    }
    return overlay, report


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    invalid = [target for target in targets if target not in ACTIVE_TARGETS]
    if invalid:
        raise RuntimeError(f"C323 only supports {ACTIVE_TARGETS}, got {invalid}")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--base-csv", default=DEFAULT_BASE)
    parser.add_argument("--baseline-report", default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--targets", default="tg,egc")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    base_path = Path(args.base_csv).resolve()
    baseline_report_path = Path(args.baseline_report).resolve()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for path in (base_path, baseline_report_path):
        guard_path(path)
    for path in (output, manifest):
        guard_path(path, allow_output=True)
    if output.exists() or manifest.exists():
        raise RuntimeError("Refusing overwrite")
    if "without_archive" not in str(output):
        raise RuntimeError(f"C323 output must live in without_archive namespace: {output}")
    if "Polymer Prediction Challenge Round 2" not in str(output):
        raise RuntimeError(f"Output outside Round 2 boundary: {output}")

    train, test, inputs = load_current_only(data_dir)
    ids = test["id"].to_numpy(int)
    base = load_base(base_path, ids)
    result = base["target"].to_numpy(float).copy()
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    feature_space = build_feature_space(train, test)
    target_reports: dict[str, Any] = {}
    for target in parse_targets(args.targets):
        baseline_oof = float(baseline_report["validation"]["target_reports"][target]["selected_oof_r2"])
        overlay_values, report = evaluate_target(target, train, test, base, feature_space, baseline_oof)
        mask = test["target_type"].to_numpy(str) == target
        if int(np.sum(mask)) != len(overlay_values):
            raise RuntimeError(f"Target row alignment failed for {target}")
        result[mask] = overlay_values
        target_reports[target] = report

    if not np.isfinite(result).all():
        raise RuntimeError("Output has non-finite predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, "target": result})
    out.to_csv(output, index=False)
    record = {
        "schema_version": "ppp.round2.c323.noarchive-tg-egc-readacross-overlay.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "official_only_training": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "method": "support-gated Tg/Egc structural read-across overlay over C312 fallback",
        "targets_requested": list(parse_targets(args.targets)),
        "target_reports": target_reports,
        "inputs": inputs,
        "baseline_report": {
            "path": str(baseline_report_path),
            "sha256": sha256_file(baseline_report_path),
            "role": "clean OOF gate reference only, not prediction input",
        },
        "base_candidate": {
            "path": str(base_path),
            "sha256": sha256_file(base_path),
            "bytes": base_path.stat().st_size,
            "role": "branch-local fallback/splice carrier only",
        },
        "features": feature_space["feature_report"],
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": int(len(out)),
            "bytes": output.stat().st_size,
        },
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": record["output"],
                "target_reports": {
                    target: {
                        "selected_arm": report["selected_arm"],
                        "selected_oof_r2": report["selected_oof_r2"],
                        "clean_oof_gate_pass": report["clean_oof_gate_pass"],
                        "changed_rows": report["changed_rows"],
                    }
                    for target, report in target_reports.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
