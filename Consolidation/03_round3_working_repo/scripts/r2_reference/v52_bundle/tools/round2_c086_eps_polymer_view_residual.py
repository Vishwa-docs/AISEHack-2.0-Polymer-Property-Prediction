#!/usr/bin/env python3
"""Official-only polymer-view residual screen for the frozen v7 EPS carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as panel_tools
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_polymer_views_v3 as polymer_views


TARGET = "eps"
SEED = 2026
ARMS = ("histgb_residual", "ridge_residual")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    return result


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if float(np.var(y[rows])) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def fit_view_model(name: str, x_train: np.ndarray, residual: np.ndarray) -> object:
    if name == "histgb_residual":
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=12,
            l2_regularization=0.10,
            random_state=SEED,
        )
        model.fit(x_train, residual)
        return model
    if name == "ridge_residual":
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=30.0),
        )
        model.fit(x_train, residual)
        return model
    raise ValueError(name)


def prepare_dense(features: np.ndarray, train_rows: np.ndarray, prediction_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.asarray(features[train_rows], dtype=np.float64).copy()
    prediction_x = np.asarray(features[prediction_rows], dtype=np.float64).copy()
    train_x[~np.isfinite(train_x)] = np.nan
    prediction_x[~np.isfinite(prediction_x)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(train_x), imputer.transform(prediction_x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created run directory with protocol.json only is required")
    started = time.time()

    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}

    rich_dense, view_report = polymer_views.feature_views(root, keys, molecules)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)

    detail, parent_oof_frame, _ = reference.fit_targets(
        pooled, test, keys, np.hstack([reference.descriptor_matrix(molecules)[0], reference.physical_matrix(molecules, keys)[0]]),
        cross_values, cross_available, sparse_parts, fingerprints, reference.DEFAULT_CONFIG,
    )
    detail_replay, parent_oof_replay_frame, _ = reference.fit_targets(
        pooled, test, keys, np.hstack([reference.descriptor_matrix(molecules)[0], reference.physical_matrix(molecules, keys)[0]]),
        cross_values, cross_available, sparse_parts, fingerprints, reference.DEFAULT_CONFIG,
    )

    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET].reset_index(drop=True)
    parent_rows_replay = parent_oof_replay_frame[parent_oof_replay_frame["target_type"] == TARGET].reset_index(drop=True)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), parent_rows["canonical"].astype(str).to_numpy()):
        raise RuntimeError("frozen v7 EPS parent OOF alignment failed")
    y = frame["target"].to_numpy(float)
    parent_oof = parent_rows["prediction"].to_numpy(float)
    parent_oof_replay = parent_rows_replay["prediction"].to_numpy(float)
    parent_replay_oof_max_abs = float(np.max(np.abs(parent_oof - parent_oof_replay)))
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True)
    parent_test = detail[detail["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(float)
    parent_test_replay = detail_replay[detail_replay["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(float)
    parent_replay_test_max_abs = float(np.max(np.abs(parent_test - parent_test_replay)))

    feature_row = {key: index for index, key in enumerate(keys)}
    rows = np.asarray([feature_row[key] for key in frame["canonical"]], dtype=np.int64)
    test_rows = np.asarray([feature_row[key] for key in test_frame["canonical"]], dtype=np.int64)
    groups = np.asarray([plumbing.no_stereo(key) for key in frame["canonical"]], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(key) for key in frame["canonical"]], dtype=object)
    folds = folds_for(groups)
    predictions = {arm: np.full(len(y), np.nan, dtype=np.float64) for arm in ARMS}
    fold_reports: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
    similarity = np.full(len(y), np.nan, dtype=np.float64)

    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        x_train, x_validation = prepare_dense(rich_dense, rows[training], rows[validation])
        residual = y[training] - parent_oof[training]
        train_fps = [fingerprints[index] for index in rows[training]]
        global_validation = np.asarray([key_to_index[key] for key in frame.iloc[validation]["canonical"]], dtype=np.int64)
        similarity[validation] = panel_tools.nearest_similarity(fingerprints, global_validation, np.asarray([key_to_index[key] for key in frame.iloc[training]["canonical"]], dtype=np.int64))
        for arm in ARMS:
            model = fit_view_model(arm, x_train, residual)
            correction = np.asarray(model.predict(x_validation), dtype=np.float64)
            candidate = parent_oof[validation] + correction
            predictions[arm][validation] = candidate
            fold_reports[arm].append({
                "fold": fold,
                "rows": int(len(validation)),
                "parent_r2": float(r2_score(y[validation], parent_oof[validation])),
                "candidate_r2": float(r2_score(y[validation], candidate)),
                "delta_r2": float(r2_score(y[validation], candidate) - r2_score(y[validation], parent_oof[validation])),
            })

    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("non-finite polymer-view OOF prediction")
    panels_by_arm: dict[str, dict[str, object]] = {}
    reports: dict[str, object] = {}
    for arm in ARMS:
        candidate = predictions[arm]
        lower = bootstrap_lower(y, parent_oof, candidate, groups)
        panels, minimum_panel = panel_tools.panel_report(y, parent_oof, candidate, scaffolds, similarity)
        candidate_rows = []
        x_full, x_test = prepare_dense(rich_dense, rows, test_rows)
        model = fit_view_model(arm, x_full, y - parent_oof)
        test_candidate = parent_test + np.asarray(model.predict(x_test), dtype=np.float64)
        component = pd.DataFrame({
            "id": test_frame["id"].astype(int),
            "target_type": TARGET,
            "parent_prediction": parent_test,
            "candidate_prediction": test_candidate,
        })
        if len(component) != 153 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(test_candidate).all():
            raise RuntimeError(f"{arm} component output contract failed")
        component.to_csv(run_dir / f"eps_component_{arm}.csv", index=False)
        parent_r2 = float(r2_score(y, parent_oof))
        candidate_r2 = float(r2_score(y, candidate))
        delta = candidate_r2 - parent_r2
        positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_reports[arm]))
        gates = {
            "gain_pass": delta >= 0.01,
            "fold_pass": positive_folds >= 4,
            "bootstrap_pass": lower > 0.0,
            "panel_pass": minimum_panel is not None and minimum_panel >= 0.0,
            "test_rows_pass": len(component) == 153,
            "parent_replay_oof_pass": parent_replay_oof_max_abs <= 1.0e-12,
            "parent_replay_test_pass": parent_replay_test_max_abs <= 1.0e-12,
        }
        reports[arm] = {
            "parent_r2": parent_r2,
            "candidate_r2": candidate_r2,
            "delta_r2": delta,
            "positive_folds": positive_folds,
            "folds": fold_reports[arm],
            "group_bootstrap_lower": lower,
            "panels": panels,
            "minimum_panel_delta": minimum_panel,
            "gates": gates,
            "decision": "pass_component_gate" if all(gates.values()) else "rejected_component_gate",
            "test_rows": int(len(component)),
        }
        panels_by_arm[arm] = panels

    oof = pd.DataFrame({
        "canonical": frame["canonical"].astype(str),
        "target": y,
        "parent": parent_oof,
        "histgb_candidate": predictions["histgb_residual"],
        "ridge_candidate": predictions["ridge_residual"],
        "group": groups,
        "scaffold": scaffolds,
        "outer_fold": folds,
        "nearest_similarity": similarity,
    })
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    source_names = (
        "round2_c086_eps_polymer_view_residual.py",
        "round2_polymer_views_v3.py",
        "../Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py",
        "initial_reference_pipeline.py",
        "round2_c063_egb_endpoint_conjugation_residual.py",
        "round2_eea_cross_target_oof_residual_stack.py",
    )
    source_hashes: dict[str, str] = {}
    for name in source_names:
        source_path = root / "tools" / name if not name.startswith("..") else (root / name).resolve()
        source_hashes[name] = sha256_file(source_path)
    report = {
        "schema_version": "ppp.round2.c086.eps-polymer-view-residual.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 EPS parent regenerated twice",
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "target": TARGET,
        "feature_view_report": view_report,
        "arms": reports,
        "parent_replay_oof_max_abs": parent_replay_oof_max_abs,
        "parent_replay_test_max_abs": parent_replay_test_max_abs,
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "oof": int(len(y)), "component_test": 153},
        "source_hashes": source_hashes,
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "arms": list(ARMS), "view_report": view_report, "bootstrap_resamples": 2000, "external_label_file_read": False, "local_eval_read": False, "no_hyperparameter_sweep": True})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    selected = max(ARMS, key=lambda arm: reports[arm]["candidate_r2"])
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{reports[selected]['decision']}** for best fixed arm `{selected}`. Official-only; no local_eval, external_label file, Kaggle action, or submission.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in source_hashes.items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "selected_arm": selected, "arms": reports}, sort_keys=True))


if __name__ == "__main__":
    main()
