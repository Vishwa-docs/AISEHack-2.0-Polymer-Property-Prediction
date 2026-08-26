#!/usr/bin/env python3
"""C126: grouped-fold Clausius-Mossotti/Lorentz-Lorenz EPS/Nc audit."""

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
from rdkit import DataStructs, RDLogger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c098_target_routed_qspr_full as qspr
import round2_c112_c050_parent_parity_control as parent_control
import round2_c120_nearmiss_bridge_mixed as diagnostics
import round2_c121_nearmiss_bridge_one_replay as replay
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE = ("eps", "nc")
OUTER_FOLDS = 5
RESIDUAL_WEIGHT = 0.20
RIDGE_ALPHA = 30.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, name: str, **fields: Any) -> None:
    record = {"checkpoint": name, "at": datetime.now().astimezone().isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 1.0e-15 else float("nan")


def forward(target: str, values: np.ndarray) -> np.ndarray:
    if target == "eps":
        return (values - 1.0) / (values + 2.0)
    if target == "nc":
        squared = np.square(values)
        return (squared - 1.0) / (squared + 2.0)
    raise ValueError(target)


def inverse(target: str, values: np.ndarray) -> np.ndarray:
    fraction = (1.0 + 2.0 * values) / (1.0 - values)
    if target == "eps":
        return fraction
    if target == "nc":
        return np.sqrt(np.maximum(fraction, 0.0))
    raise ValueError(target)


def model() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def target_context(bundle: dict[str, Any], target: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    matrix, train_rows, test_rows, names, pair_train, pair_test = qspr.target_features(bundle, target)
    info = bundle["target_info"][target]
    feature_keys = sorted(set(info["canonical"]) | set(bundle["test"].loc[bundle["test"]["target_type"] == target, "canonical"]))
    return matrix, train_rows, names, np.asarray(feature_keys, dtype=object), pair_train, pair_test


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray, pair: np.ndarray) -> tuple[dict[str, Any], float | None]:
    panels: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, mask: np.ndarray, minimum: int = 5, required: bool = True) -> None:
        rows = int(np.sum(mask))
        item: dict[str, Any] = {"rows": rows, "delta_r2": None, "status": "inapplicable"}
        if rows >= minimum and np.var(y[mask]) > 1.0e-15:
            delta = score(y[mask], candidate[mask]) - score(y[mask], parent[mask])
            item.update({"delta_r2": delta, "status": "evaluable"})
            if required:
                deltas.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    add("counterpart_available", pair)
    add("counterpart_missing", ~pair, required=False)
    for name, mask in {"similarity_lt_0.30": similarity < 0.30, "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50), "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70), "similarity_ge_0.70": similarity >= 0.70}.items():
        add(name, mask)
    for scaffold in sorted(set(scaffolds)):
        add(f"scaffold_{scaffold}", scaffolds == scaffold, minimum=10)
    return panels, (float(min(deltas)) if deltas else None)


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
        raise RuntimeError("C126 requires a fresh protocol-only run directory")
    started = time.time()
    progress_path = run_dir / "progress.jsonl"
    bundle, parity = replay.build_one_replay_bundle(root, data_dir, run_dir)
    checkpoint(progress_path, "parent_replay", oof_rows=9851, test_rows=4940)
    checkpoint(progress_path, "parent_parity", **parity)
    if parity["oof_max_abs"] > 1.0e-12 or parity["test_max_abs"] > 1.0e-12:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = bundle["target_info"][target]
        y = np.asarray(info["y"], dtype=float)
        parent = np.asarray(info["parent"], dtype=float)
        if target not in ACTIVE:
            target_reports[target] = {"parent_r2": score(y, parent), "candidate_r2": score(y, parent), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "pass": True, "unchanged_parent": True}
            oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": parent, "outer_fold": -1}))
            continue
        matrix, train_rows, feature_names, feature_keys, pair_train, _ = target_context(bundle, target)
        feature_groups = np.asarray([plumbing.no_stereo(value) for value in feature_keys], dtype=object)
        groups = np.asarray([plumbing.no_stereo(value) for value in info["canonical"]], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in info["canonical"]], dtype=object)
        folds = plumbing.folds_for(groups, OUTER_FOLDS)
        transformed_y = forward(target, y)
        transformed_parent = forward(target, parent)
        transformed_residual = transformed_y - transformed_parent
        candidate = parent.copy()
        similarity = np.full(len(y), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(OUTER_FOLDS):
            outer_train = np.flatnonzero(folds != fold)
            outer_valid = np.flatnonzero(folds == fold)
            allowed_groups = set(groups[outer_train])
            fold_matrix = matrix.copy()
            fold_matrix[~np.isin(feature_groups, list(allowed_groups)), -3:] = np.nan
            train_supported = np.isfinite(pair_train[outer_train]) & np.isin(groups[outer_train], list(allowed_groups))
            valid_supported = np.isfinite(pair_train[outer_valid]) & np.isin(groups[outer_valid], list(allowed_groups))
            if int(np.sum(train_supported)) >= 8 and int(np.sum(valid_supported)) > 0:
                fitted = model()
                fitted.fit(fold_matrix[train_rows[outer_train[train_supported]]], transformed_residual[outer_train[train_supported]])
                correction = fitted.predict(fold_matrix[train_rows[outer_valid[valid_supported]]])
                transformed_candidate = transformed_parent[outer_valid[valid_supported]] + RESIDUAL_WEIGHT * correction
                candidate[outer_valid[valid_supported]] = inverse(target, transformed_candidate)
            train_fps = [bundle["fingerprints"][int(index)] for index in train_rows[outer_train]]
            for row, global_index in enumerate(train_rows[outer_valid]):
                similarities = DataStructs.BulkTanimotoSimilarity(bundle["fingerprints"][int(global_index)], train_fps)
                similarity[outer_valid[row]] = max(similarities) if similarities else 0.0
            fold_parent = score(y[outer_valid], parent[outer_valid])
            fold_candidate = score(y[outer_valid], candidate[outer_valid])
            fold_rows.append({"fold": fold, "rows": int(len(outer_valid)), "train_supported_rows": int(np.sum(train_supported)), "valid_supported_rows": int(np.sum(valid_supported)), "parent_r2": fold_parent, "candidate_r2": fold_candidate, "delta_r2": fold_candidate - fold_parent})
            checkpoint(progress_path, f"{target}_fold_{fold}", delta_r2=fold_candidate - fold_parent, train_supported_rows=int(np.sum(train_supported)), valid_supported_rows=int(np.sum(valid_supported)))
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"non-finite {target} candidate")
        pair_mask = np.isfinite(pair_train)
        no_op_error = float(np.max(np.abs(candidate[~pair_mask] - parent[~pair_mask]))) if np.any(~pair_mask) else 0.0
        bootstrap = diagnostics.grouped_bootstrap(y, parent, candidate, groups)
        panels, minimum_panel = panel_report(y, parent, candidate, groups, scaffolds, similarity, pair_mask)
        delta = score(y, candidate) - score(y, parent)
        positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        target_pass = bool(delta >= 0.01 and positive >= 4 and bootstrap["lower_2_5"] > 0.0 and minimum_panel is not None and minimum_panel >= 0.0 and no_op_error <= 1.0e-12)
        target_reports[target] = {"parent_r2": score(y, parent), "candidate_r2": score(y, candidate), "delta_r2": delta, "positive_folds": positive, "folds": fold_rows, "group_bootstrap_lower": bootstrap["lower_2_5"], "minimum_panel_delta": minimum_panel, "panels": panels, "feature_count": int(matrix.shape[1]), "pair_rows": int(np.sum(pair_mask)), "missing_noop_max_abs": no_op_error, "pass": target_pass, "unchanged_parent": False}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "outer_fold": folds}))
        checkpoint(progress_path, f"{target}_oof", delta_r2=delta, positive_folds=positive, minimum_panel_delta=minimum_panel, component_pass=target_pass)

    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(target_reports[target]["delta_r2"] for target in TARGETS))
    clean_pass = bool(mean_candidate - mean_parent >= 0.002 and max_loss >= -0.003 and all(target_reports[target]["pass"] for target in ACTIVE))
    report = {"schema_version": "ppp.round2.c126.cm-ll-supported-eps-nc.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "official_only": True, "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7", "parent_parity": parity, "target_reports": target_reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": max_loss, "clean_gate_pass": clean_pass, "decision": "clean_gate_pass_pending_full_data_fit" if clean_pass else "rejected_component_or_full_gate", "elapsed_seconds": float(time.time() - started)}
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c126.cm-ll-supported-eps-nc.v1", "active_targets": ACTIVE, "ridge_alpha": RIDGE_ALPHA, "residual_weight": RESIDUAL_WEIGHT, "outer_folds": OUTER_FOLDS, "fold_assignment": "plumbing.folds_for(no_stereo,5)", "transforms": {"eps": "(y-1)/(y+2)", "nc": "(y^2-1)/(y^2+2)"}, "counterpart_mask": "outer-training no-stereo groups only", "official_only": True, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent R2 `{mean_parent:.12f}`; candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. The physical-coordinate route remains research-only unless both active component gates and the complete clean candidate gate pass.\n", encoding="utf-8")
    checkpoint(progress_path, "metrics_written", decision=report["decision"], mean_gain=report["mean_gain"])
    source_paths = [Path(__file__), root / "tools/round2_c098_target_routed_qspr_full.py", root / "tools/round2_c121_nearmiss_bridge_one_replay.py", root / "tools/round2_c120_nearmiss_bridge_mixed.py", root / "tools/round2_c112_c050_parent_parity_control.py", root / "tools/round2_eea_cross_target_oof_residual_stack.py", root / "tools/initial_reference_pipeline.py"]
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
