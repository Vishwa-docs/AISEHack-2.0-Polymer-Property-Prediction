#!/usr/bin/env python3
"""C120: nested two-arm bridge over the strongest official EPS/Nc near-miss."""

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
from rdkit import RDLogger
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c098_target_routed_qspr_full as qspr
import round2_c112_c050_parent_parity_control as parent_control
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE = ("eps", "nc")
SEED = 2026
OUTER_FOLDS = 5
INNER_FOLDS = 4
RESIDUAL_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 1e-15 else float("nan")


def make_arm(kind: str) -> Any:
    if kind == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=30.0),
        )
    if kind == "histgb":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.03,
                max_leaf_nodes=7,
                min_samples_leaf=8,
                l2_regularization=0.5,
                random_state=SEED,
            ),
        )
    raise ValueError(kind)


def fit_arms(x_train: np.ndarray, residual: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    predictions = []
    for kind in ("ridge", "histgb"):
        model = make_arm(kind)
        model.fit(x_train, residual)
        predictions.append(model.predict(x_pred))
    return np.column_stack(predictions)


def blend_weights(y_residual: np.ndarray, predictions: np.ndarray, support: np.ndarray) -> np.ndarray:
    usable = support & np.isfinite(y_residual) & np.isfinite(predictions).all(axis=1)
    if int(np.sum(usable)) < 8:
        return np.asarray([0.5, 0.5], dtype=float)
    weights, _ = nnls(predictions[usable], y_residual[usable])
    if float(weights.sum()) <= 1e-12:
        return np.asarray([0.5, 0.5], dtype=float)
    return weights / float(weights.sum())


def grouped_bootstrap(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1e-15:
            values.append(score(y[rows], candidate[rows]) - score(y[rows], parent[rows]))
    return {"lower_2_5": float(np.quantile(values, 0.025)), "median": float(np.quantile(values, 0.5)), "replicates": len(values)}


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray, pair: np.ndarray) -> tuple[dict[str, Any], float | None]:
    panels: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, mask: np.ndarray, minimum: int = 5, required: bool = True) -> None:
        rows = int(np.sum(mask))
        item: dict[str, Any] = {"rows": rows, "delta_r2": None, "status": "inapplicable"}
        if rows >= minimum and np.var(y[mask]) > 1e-15:
            delta = score(y[mask], candidate[mask]) - score(y[mask], parent[mask])
            item.update({"delta_r2": delta, "status": "evaluable"})
            if required:
                deltas.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    add("counterpart_available", pair)
    add("counterpart_missing", ~pair, required=False)
    for name, mask in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        add(name, mask)
    for scaffold in sorted(set(scaffolds)):
        add(f"scaffold_{scaffold}", scaffolds == scaffold, minimum=10)
    return panels, (float(min(deltas)) if deltas else None)


def parent_parity(root: Path, data_dir: Path, run_dir: Path, bundle: dict[str, Any]) -> dict[str, float]:
    replay_predictions, replay_oof, _ = parent_control.rebuild_parent(root, data_dir, run_dir)
    replay_oof = replay_oof.sort_values(["target_type", "canonical", "target"], kind="mergesort").reset_index(drop=True)
    bundle_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = bundle["target_info"][target]
        bundle_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": info["y"], "parent_prediction": info["parent"]}))
    bundle_oof = pd.concat(bundle_parts, ignore_index=True).sort_values(["target_type", "canonical", "target"], kind="mergesort").reset_index(drop=True)
    if not bundle_oof[["canonical", "target_type", "target"]].astype(str).equals(replay_oof[["canonical", "target_type", "target"]].astype(str)):
        raise RuntimeError("C120 parent OOF row identity differs from independent C050 replay")
    oof_delta = float(np.max(np.abs(bundle_oof["parent_prediction"].to_numpy(float) - replay_oof["parent_prediction"].to_numpy(float))))
    raw_detail = bundle["test_detail"].copy()
    final_detail, _ = reference.apply_official_overrides(raw_detail, bundle["test"], bundle["raw_labels"])
    test_delta = float(np.max(np.abs(final_detail["target"].to_numpy(float) - replay_predictions["target"].to_numpy(float))))
    return {"oof_max_abs": oof_delta, "test_max_abs": test_delta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    data_dir = (root / args.data_dir).resolve()
    run_dir = (root / args.run_dir).resolve()
    if {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("C120 requires a fresh protocol-only run directory")
    started = time.time()
    bundle = qspr.parent_bundle(root, data_dir)
    parity = parent_parity(root, data_dir, run_dir, bundle)
    write_json(run_dir / "parent_parity.json", parity)
    if parity["oof_max_abs"] > 1e-12 or parity["test_max_abs"] > 1e-12:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    target_weights: dict[str, list[list[float]]] = {}
    test_models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for target in TARGETS:
        info = bundle["target_info"][target]
        y = np.asarray(info["y"], dtype=float)
        parent = np.asarray(info["parent"], dtype=float)
        if target not in ACTIVE:
            target_reports[target] = {"parent_r2": score(y, parent), "candidate_r2": score(y, parent), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "pass": True, "unchanged_parent": True}
            oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": parent, "outer_fold": -1}))
            continue
        matrix, train_rows, test_rows, names, pair_train, pair_test = qspr.target_features(bundle, target)
        residual = y - parent
        groups = np.asarray([plumbing.no_stereo(value) for value in info["canonical"]], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in info["canonical"]], dtype=object)
        outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        candidate = np.full(len(y), np.nan, dtype=float)
        similarity = np.full(len(y), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        weights_rows: list[list[float]] = []
        for fold, (outer_train, outer_valid) in enumerate(outer.split(np.arange(len(y)))):
            inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold + 1)
            inner_predictions = np.full((len(outer_train), 2), np.nan, dtype=float)
            for inner_train_rel, inner_valid_rel in inner.split(outer_train):
                inner_train = outer_train[inner_train_rel]
                inner_valid = outer_train[inner_valid_rel]
                inner_predictions[inner_valid_rel] = fit_arms(matrix[train_rows[inner_train]], residual[inner_train], matrix[train_rows[inner_valid]])
            inner_pair = np.isfinite(pair_train[outer_train])
            weights = blend_weights(residual[outer_train], inner_predictions, inner_pair)
            weights_rows.append(weights.tolist())
            arm_predictions = fit_arms(matrix[train_rows[outer_train]], residual[outer_train], matrix[train_rows[outer_valid]])
            correction = RESIDUAL_WEIGHT * (arm_predictions @ weights)
            candidate[outer_valid] = parent[outer_valid]
            supported = np.isfinite(pair_train[outer_valid])
            candidate[outer_valid[supported]] += correction[supported]
            train_fps = [bundle["fingerprints"][int(train_rows[index])] for index in outer_train]
            for row, global_index in enumerate(train_rows[outer_valid]):
                similarities = reference.DataStructs.BulkTanimotoSimilarity(bundle["fingerprints"][int(global_index)], train_fps)
                similarity[outer_valid[row]] = max(similarities) if similarities else 0.0
            fold_rows.append({"fold": fold, "rows": int(len(outer_valid)), "parent_r2": score(y[outer_valid], parent[outer_valid]), "candidate_r2": score(y[outer_valid], candidate[outer_valid]), "delta_r2": score(y[outer_valid], candidate[outer_valid]) - score(y[outer_valid], parent[outer_valid]), "weights": weights.tolist()})
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"non-finite {target} OOF candidate")
        bootstrap = grouped_bootstrap(y, parent, candidate, groups)
        panels, minimum_panel = panel_report(y, parent, candidate, groups, scaffolds, similarity, np.isfinite(pair_train))
        delta = score(y, candidate) - score(y, parent)
        positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        target_pass = bool(delta >= 0.01 and positive >= 4 and bootstrap["lower_2_5"] > 0.0 and minimum_panel is not None and minimum_panel >= 0.0 and (panels["counterpart_missing"]["delta_r2"] is None or abs(float(panels["counterpart_missing"]["delta_r2"])) <= 1e-12))
        target_reports[target] = {"parent_r2": score(y, parent), "candidate_r2": score(y, candidate), "delta_r2": delta, "positive_folds": positive, "folds": fold_rows, "group_bootstrap_lower": bootstrap["lower_2_5"], "minimum_panel_delta": minimum_panel, "panels": panels, "feature_count": int(matrix.shape[1]), "pair_rows": int(np.sum(np.isfinite(pair_train))), "pass": target_pass, "unchanged_parent": False}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "outer_fold": np.asarray([row for row in range(len(y))])}))
        target_weights[target] = weights_rows
        test_models[target] = (matrix, train_rows, test_rows, pair_test, names)

    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(target_reports[target]["delta_r2"] for target in TARGETS))
    clean_pass = bool(mean_candidate - mean_parent >= 0.002 and max_loss >= -0.003 and all(target_reports[target]["pass"] for target in ACTIVE))
    report: dict[str, Any] = {
        "schema_version": "ppp.round2.c120.nearmiss-bridge-mixed.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7",
        "parent_parity": parity,
        "target_reports": target_reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": max_loss,
        "clean_gate_pass": clean_pass,
        "decision": "clean_gate_pass_pending_full_data_fit" if clean_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "blend_weights.json", target_weights)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c120.nearmiss-bridge-mixed.v1", "active_targets": ACTIVE, "residual_weight": RESIDUAL_WEIGHT, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "arms": ["ridge_alpha_30", "histgb_fixed"], "no_sweep": True, "official_only": True, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent R2 `{mean_parent:.12f}`; candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. No full-data test fit or local_eval action was opened before the clean gates.\n", encoding="utf-8")
    source_paths = [Path(__file__), root / "tools/initial_reference_pipeline.py", root / "tools/round2_c098_target_routed_qspr_full.py", root / "tools/round2_c112_c050_parent_parity_control.py", root / "tools/round2_c063_egb_endpoint_conjugation_residual.py", root / "tools/round2_eea_cross_target_oof_residual_stack.py"]
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for path in source_paths:
        manifest.append(f"{sha256_file(path)}  SOURCE {path.relative_to(root)}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "eps_delta": target_reports["eps"]["delta_r2"], "nc_delta": target_reports["nc"]["delta_r2"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
