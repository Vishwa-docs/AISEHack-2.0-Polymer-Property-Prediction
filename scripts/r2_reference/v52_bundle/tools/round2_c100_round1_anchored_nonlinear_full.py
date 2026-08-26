#!/usr/bin/env python3
"""C100: fixed target-routed CatBoost/XGBoost residual candidate."""

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

import catboost as cb
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import RDLogger
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c076_eps_paired_charge_polarizability_residual as paired_features
import round2_c098_target_routed_qspr_full as c098
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
CHANGED = {"nc", "eps"}
RESIDUAL_WEIGHT = 0.10
SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes(order="C")).hexdigest()


def model_for(target: str) -> Any:
    if target == "eps":
        return cb.CatBoostRegressor(iterations=200, depth=4, learning_rate=0.03, l2_leaf_reg=10.0, loss_function="RMSE", random_seed=SEED, verbose=False, allow_writing_files=False, thread_count=1)
    return xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, max_depth=2, learning_rate=0.03, min_child_weight=8.0, reg_lambda=10.0, subsample=0.9, colsample_bytree=0.8, random_state=SEED, n_jobs=1, tree_method="hist", verbosity=0)


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def feature_matrix(bundle: dict[str, Any], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    info = bundle["target_info"][target]
    keys = sorted(set(info["canonical"]) | set(bundle["test"].loc[bundle["test"]["target_type"] == target, "canonical"]))
    global_indices = np.asarray([bundle["key_to_index"][value] for value in keys], dtype=np.int64)
    fixed, fixed_names = fixed_features.fixed_features(bundle["molecules"], global_indices.tolist())
    physical, physical_names = paired_features.physics_features(bundle["molecules"], global_indices.tolist())
    charge, charge_names = paired_features.charge_features(bundle["molecules"], global_indices.tolist())
    morgan2 = reference.morgan_count_matrix(bundle["molecules"], 2, 512).toarray()[global_indices]
    morgan3 = reference.morgan_count_matrix(bundle["molecules"], 3, 512).toarray()[global_indices]
    matrix = np.hstack([fixed, physical, charge, morgan2, morgan3]).astype(np.float64, copy=False)
    feature_row = {value: row for row, value in enumerate(keys)}
    train_rows = np.asarray([feature_row[value] for value in info["canonical"]], dtype=np.int64)
    test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    test_rows = np.asarray([feature_row[value] for value in test_frame["canonical"]], dtype=np.int64)
    names = fixed_names + physical_names + charge_names + [f"morgan2_{i}" for i in range(512)] + [f"morgan3_{i}" for i in range(512)]
    return matrix, train_rows, test_rows, names


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, Any], float]:
    panels: dict[str, Any] = {}
    values: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected))
        item: dict[str, Any] = {"rows": count, "delta_r2": 0.0, "status": "inapplicable"}
        if count >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"delta_r2": delta, "status": "evaluable"})
            values.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    for name, selected in {"similarity_lt_0.30": similarity < 0.30, "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50), "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70), "similarity_ge_0.70": similarity >= 0.70}.items():
        add(name, selected)
    for name in sorted(set(scaffolds)):
        add(f"scaffold_{name}", scaffolds == name, minimum=10)
    return panels, float(min(values)) if values else 0.0


def compare_replay(bundle: dict[str, Any], replay_path: Path) -> float:
    recorded = json.loads(replay_path.read_text(encoding="utf-8"))
    max_error = 0.0
    replay_dir = replay_path.parent / "replay_parent"
    for target in TARGETS:
        info = bundle["target_info"][target]
        ids = bundle["test_detail"].loc[bundle["test_detail"]["target_type"] == target, "id"].to_numpy()
        values = bundle["test_detail"].set_index("id").loc[ids, "model_prediction"].to_numpy(float)
        replay_oof = np.load(replay_dir / f"{target}_oof.npy")
        replay_test = np.load(replay_dir / f"{target}_test.npy")
        oof_error = float(np.max(np.abs(np.asarray(info["parent"]) - replay_oof)))
        test_error = float(np.max(np.abs(values - replay_test)))
        max_error = max(max_error, oof_error, test_error)
        if oof_error > 1.0e-12 or test_error > 1.0e-12:
            raise RuntimeError(f"independent parent numerical replay mismatch for {target}: oof={oof_error} test={test_error}")
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json", "pre_run_parent_replay.json", "replay_parent"}:
        raise RuntimeError("protocol plus independent replay artifact required")
    started = time.time()
    bundle = c098.parent_bundle(root, (root / args.data_dir).resolve())
    replay_max = compare_replay(bundle, run_dir / "pre_run_parent_replay.json")
    raw_test = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    components: list[pd.DataFrame] = []
    for target in TARGETS:
        info = bundle["target_info"][target]
        y, parent, folds = info["y"], info["parent"], info["folds"]
        candidate = parent.copy()
        if target in CHANGED:
            matrix, train_rows, test_rows, names = feature_matrix(bundle, target)
            clean_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            train_matrix = clean_imputer.fit_transform(matrix[train_rows])
            candidate = parent.copy()
            similarity = np.full(len(y), np.nan, dtype=np.float64)
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                x_train = imputer.fit_transform(matrix[train_rows[training]])
                x_valid = imputer.transform(matrix[train_rows[validation]])
                fitted = model_for(target)
                fitted.fit(x_train, y[training] - parent[training])
                candidate[validation] = parent[validation] + RESIDUAL_WEIGHT * fitted.predict(x_valid)
                global_validation = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][validation]], dtype=np.int64)
                global_training = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][training]], dtype=np.int64)
                similarity[validation] = fixed_features.nearest_similarity(bundle["fingerprints"], global_validation, global_training)
            full_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            x_full = full_imputer.fit_transform(matrix[train_rows])
            x_test = full_imputer.transform(matrix[test_rows])
            fitted = model_for(target)
            fitted.fit(x_full, y - parent)
            test_correction = RESIDUAL_WEIGHT * fitted.predict(x_test)
            test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            test_parent = raw_test.loc[raw_test["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float).copy()
            test_candidate = test_parent + test_correction
            raw_test.loc[raw_test["target_type"] == target, "model_prediction"] = test_candidate
            panels, minimum_panel = panel_report(y, parent, candidate, info["scaffolds"], similarity)
            components.append(pd.DataFrame({"id": test_frame["id"], "target_type": target, "parent_prediction": test_parent, "candidate_prediction": test_candidate}))
            feature_count = int(matrix.shape[1])
        else:
            similarity = np.full(len(y), np.nan, dtype=np.float64)
            panels, minimum_panel, feature_count = {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}}, 0.0, 0
        fold_rows = []
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation]))})
        delta = float(r2_score(y, candidate) - r2_score(y, parent))
        lower = bootstrap_lower(y, parent, candidate, info["groups"]) if target in CHANGED else 0.0
        positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        reports[target] = {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "folds": fold_rows, "panels": panels, "feature_count": feature_count, "pass": bool(target not in CHANGED or (delta >= 0.01 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0)), "unchanged_parent": target not in CHANGED}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "group": info["groups"], "scaffold": info["scaffolds"], "fold": folds}))
    final_detail, override_report = reference.apply_official_overrides(raw_test, bundle["test"], bundle["raw_labels"])
    submission = final_detail[["id", "target"]].copy()
    if len(submission) != 4940 or not submission["id"].equals(bundle["test"]["id"]) or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C100 output contract failed")
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(reports[target]["delta_r2"] for target in TARGETS))
    full_pass = bool(replay_max <= 1.0e-12 and mean_candidate > 0.8748045537286532 and all(reports[target]["pass"] for target in CHANGED) and max_loss >= -0.003)
    submission.to_csv(run_dir / "predictions.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    if components:
        pd.concat(components, ignore_index=True).to_csv(run_dir / "component_predictions.csv", index=False)
    source_paths = {"script": root / "tools" / "round2_c100_round1_anchored_nonlinear_full.py", "replay_script": root / "tools" / "round2_c100_parent_replay.py", "reference": root / "tools" / "initial_reference_pipeline.py", "parent_route": root / "tools" / "round2_c098_target_routed_qspr_full.py", "fixed_features": root / "tools" / "round2_c063_egb_endpoint_conjugation_residual.py", "paired_features": root / "tools" / "round2_c076_eps_paired_charge_polarizability_residual.py"}
    report = {"schema_version": "ppp.round2.c100.round1-anchored-nonlinear-full.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "C050 source rebuild verified against independent parent hash artifact", "official_inputs": bundle["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "pretrained_weights": False, "target_reports": reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": max_loss, "complete_output_rows": int(len(submission)), "complete_output_order_pass": True, "independent_parent_replay_pass": True, "independent_parent_replay_max_abs": replay_max, "official_override_report": override_report, "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()}, "elapsed_seconds": float(time.time() - started), "decision": "candidate_pending_group_audit" if full_pass else "rejected_full_candidate_gate"}
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "changed_targets": sorted(CHANGED), "eps_model": "catboost iterations=200 depth=4 learning_rate=0.03 l2=10", "nc_model": "xgboost n_estimators=200 max_depth=2 learning_rate=0.03 min_child_weight=8 l2=10", "feature_views": "fixed endpoint/physical/capped-charge + Morgan count radius 2/3 512 bins", "residual_weight": RESIDUAL_WEIGHT, "no_sweep": True, "counterpart_labels": False, "independent_parent_hash_replay": True})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\ncatboost={cb.__version__}\nxgboost={xgb.__version__}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent `{mean_parent:.12f}`; candidate `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`; independent parent replay passed. No local_eval or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{digest}  SOURCE tools/{name}.py" for name, digest in report["source_hashes"].items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "decision": report["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
