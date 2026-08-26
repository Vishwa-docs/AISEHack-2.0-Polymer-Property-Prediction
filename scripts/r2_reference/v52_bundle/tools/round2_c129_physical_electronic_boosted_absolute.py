#!/usr/bin/env python3
"""C129: transformed, non-paired physical/electronic absolute carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as endpoint_features
import round2_c076_eps_paired_charge_polarizability_residual as physical_features
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as c127


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("eps", "nc", "ei", "eea")
SEED = 2026
N_FOLDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "time": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), **payload}, sort_keys=True, allow_nan=False) + "\n")


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    return result


def feature_matrix(parent: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    molecules = parent["molecules"]
    indices = list(range(len(molecules)))
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, parent["keys"])
    endpoint, endpoint_names = endpoint_features.fixed_features(molecules, indices)
    normalized, normalized_names = physical_features.physics_features(molecules, indices)
    charges, charge_names = physical_features.charge_features(molecules, indices)
    grammar = parent_builder.grammar_features(molecules)
    matrix = np.hstack([descriptor, physical, endpoint, normalized, charges, grammar]).astype(np.float64)
    report = {
        "shape": [int(value) for value in matrix.shape],
        "blocks": {"rdkit": len(descriptor_names), "physical": len(physical_names), "endpoint": len(endpoint_names), "normalized_physical": len(normalized_names), "charge": len(charge_names), "fragment_grammar": int(grammar.shape[1])},
        "paired_labels_used": False,
    }
    return matrix, report


def transform_target(y_train: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, PowerTransformer]:
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    transformed = transformer.fit_transform(y_train.reshape(-1, 1)).ravel()
    return transformer.inverse_transform(y_values.reshape(-1, 1)).ravel(), transformer


def fit_arms(
    matrix: np.ndarray,
    y: np.ndarray,
    train_matrix_rows: np.ndarray,
    prediction_matrix_rows: np.ndarray,
    train_y_rows: np.ndarray,
    seed: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(matrix[train_matrix_rows])
    prediction_x = imputer.transform(matrix[prediction_matrix_rows])
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    train_z = transformer.fit_transform(y[train_y_rows].reshape(-1, 1)).ravel()
    hist = HistGradientBoostingRegressor(
        max_iter=180, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=10, l2_regularization=3.0, random_state=seed
    )
    hist.fit(train_x, train_z)
    hist_prediction = transformer.inverse_transform(hist.predict(prediction_x).reshape(-1, 1)).ravel()
    cat = CatBoostRegressor(
        iterations=350, depth=6, learning_rate=0.03, l2_leaf_reg=5.0, loss_function="RMSE", random_seed=seed,
        verbose=False, allow_writing_files=False, thread_count=2, random_strength=0.5,
    )
    cat.fit(train_x, train_z)
    cat_prediction = transformer.inverse_transform(cat.predict(prediction_x).reshape(-1, 1)).ravel()
    return np.column_stack([hist_prediction, cat_prediction])


def target_run(parent: dict[str, Any], target: str, matrix: np.ndarray, progress: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    info = dict(parent["target_info"][target])
    info["fingerprints"] = parent["fingerprints"]
    y = np.asarray(info["y"], dtype=float)
    parent_oof = np.asarray(info["parent"], dtype=float)
    groups = np.asarray(info["groups"], dtype=object)
    folds = grouped_folds(groups)
    graph_indices = np.asarray(info["indices"], dtype=np.int64)
    direct_oof = np.full((len(y), 2), np.nan, dtype=float)
    for fold in range(N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        direct_oof[validation] = fit_arms(
            matrix,
            y,
            graph_indices[training],
            graph_indices[validation],
            training,
            SEED + 19 * (TARGETS.index(target) + 1) + fold,
        )
    arms = np.column_stack([parent_oof, direct_oof])
    weights, intercept, blend_name, _ = reference.blend_from_oof(y, arms)
    candidate = arms @ weights + intercept
    test_frame = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    test_indices = np.asarray([parent["key_to_index"][value] for value in test_frame["canonical"]], dtype=np.int64)
    test_arms = fit_arms(
        matrix,
        y,
        graph_indices,
        test_indices,
        np.arange(len(y), dtype=np.int64),
        SEED + 1000 + TARGETS.index(target),
    )
    test_parent = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id")["target"].to_numpy(float)
    test_candidate = np.column_stack([test_parent, test_arms]) @ weights + intercept
    report = c127.evaluate_target(info, {"candidate": candidate})
    report.update({"blend_name": blend_name, "blend_weights": [float(value) for value in weights], "blend_intercept": float(intercept), "hist_r2": float(r2_score(y, direct_oof[:, 0])), "catboost_r2": float(r2_score(y, direct_oof[:, 1])), "feature_count": int(matrix.shape[1])})
    checkpoint(progress, f"target_{target}_complete", target=target, delta_r2=report["delta_r2"], candidate_r2=report["candidate_r2"], pass_gate=report["pass"])
    return report, candidate, test_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    progress = run_dir / "progress.jsonl"
    checkpoint(progress, "started", experiment_id=run_dir.name)
    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = c127.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)
    matrix, feature_report = feature_matrix(parent)
    checkpoint(progress, "features_constructed", **feature_report)
    target_reports: dict[str, Any] = {}
    candidate_oof: dict[str, np.ndarray] = {}
    candidate_test: dict[str, np.ndarray] = {}
    for target in ACTIVE_TARGETS:
        report, oof, test_prediction = target_run(parent, target, matrix, progress)
        target_reports[target] = report
        candidate_oof[target] = oof
        candidate_test[target] = test_prediction
    for target in TARGETS:
        if target not in target_reports:
            info = parent["target_info"][target]
            y = np.asarray(info["y"], dtype=float)
            target_reports[target] = {"parent_r2": float(r2_score(y, info["parent"])), "candidate_r2": float(r2_score(y, info["parent"])), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}}, "folds": [], "pass": True, "unchanged_parent": True}
            candidate_oof[target] = np.asarray(info["parent"], dtype=float)
            candidate_test[target] = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id")["target"].to_numpy(float)
    banked = [target for target in ACTIVE_TARGETS if target_reports[target]["pass"]]
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = parent["target_info"][target]
        assembled = candidate_oof[target] if target in banked else np.asarray(info["parent"], dtype=float)
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": info["y"], "parent": info["parent"], "candidate": candidate_oof[target], "assembled": assembled, "group": info["groups"], "scaffold": info["scaffolds"], "outer_fold": grouped_folds(np.asarray(info["groups"], dtype=object))}))
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(mean_candidate >= mean_parent + 0.002 and max_loss >= -0.003 and len(banked) > 0)
    parent_detail = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    parts: list[pd.DataFrame] = []
    for target in TARGETS:
        frame = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        values = candidate_test[target] if target in banked else parent_detail.loc[parent_detail["target_type"] == target].sort_values("id")["target"].to_numpy(float)
        parts.append(pd.DataFrame({"id": frame["id"].astype(int), "target_type": target, "model_prediction": values}))
    raw = pd.concat(parts, ignore_index=True).sort_values("id")
    raw_labels, _ = reference.build_label_pool(parent["train"], parent["archive"])
    detail, override_report = reference.apply_official_overrides(raw, parent["test"], raw_labels)
    predictions = detail[["id", "target"]].copy()
    if len(predictions) != 4940 or not predictions["id"].equals(parent["test"]["id"]) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C129 complete output contract failed")
    source_paths = {
        "runner": Path(__file__),
        "parent_builder": root / "tools/round2_c097_graph_grammar_hgb_full.py",
        "reference": root / "tools/initial_reference_pipeline.py",
        "endpoint_features": root / "tools/round2_c063_egb_endpoint_conjugation_residual.py",
        "physical_features": root / "tools/round2_c076_eps_paired_charge_polarizability_residual.py",
    }
    report = {"schema_version": "ppp.round2.c129.physical-electronic-boosted-absolute.run.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "parent": "C050 source rebuild; no C127/C128 artifact input", "official_inputs": parent["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "pretrained_weights": False, "parent_replay_parity": parity, "feature_report": feature_report, "active_targets": list(ACTIVE_TARGETS), "banked_targets": banked, "target_reports": target_reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": max_loss, "complete_output_rows": int(len(predictions)), "complete_output_order_pass": True, "full_candidate_gate_pass": full_pass, "official_override_report": override_report, "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()}, "elapsed_seconds": float(time.time() - started), "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate"}
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c129.physical-electronic-boosted-absolute.v1", "active_targets": list(ACTIVE_TARGETS), "seed": SEED, "transform": "Yeo-Johnson", "models": ["HistGradientBoosting", "CatBoost"], "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"catboost={__import__('catboost').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{mean_parent:.12f}`; assembled `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. No local_eval read.\n", encoding="utf-8")
    checkpoint(progress, "metrics_written", decision=report["decision"], mean_candidate_r2=mean_candidate, banked_targets=banked)
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, path in source_paths.items():
        manifest.append(f"{sha256_file(path)}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "banked_targets": banked, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
