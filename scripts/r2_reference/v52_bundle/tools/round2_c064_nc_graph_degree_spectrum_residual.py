#!/usr/bin/env python3
"""Official-only Nc graph-distance/degree-spectrum residual screen."""

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
from rdkit import DataStructs, RDLogger
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c062_tg_topological_shape_free_volume_residual as proxy
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
SEED = 2026
TARGET = "nc"
RESIDUAL_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def augment_degree_features(molecules: list[object], indices: list[int], base: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names = list(proxy.graph_features(molecules, indices)[1]["feature_names"])
    degree_names = [f"degree_fraction_{degree}" for degree in range(7)]
    degree = np.zeros((len(indices), len(degree_names)), dtype=np.float64)
    for row, index in enumerate(indices):
        atoms = list(molecules[index].GetAtoms()); denominator = max(len(atoms), 1)
        for atom in atoms:
            if atom.GetDegree() < 7: degree[row, atom.GetDegree()] += 1.0 / denominator
    return np.hstack([base, degree]), names + degree_names


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups); members = {group: np.flatnonzero(groups == group) for group in unique}; rng = np.random.default_rng(SEED); values = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True); rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15: values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--data-dir", default="ppp-round-2"); parser.add_argument("--run-dir", required=True); args = parser.parse_args()
    root = Path(args.root).resolve(); run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}: raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time(); train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve()); _, pooled = reference.build_label_pool(train, archive); keys = sorted(set(pooled["canonical"]) | set(test["canonical"])); key_to_index = {key: index for index, key in enumerate(keys)}; molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules); physical, _ = reference.physical_matrix(molecules, keys); dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False); cross_values, cross_available = reference.cross_property_arrays(pooled, keys); sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]; fingerprints = reference.morgan_bits(molecules, 2, 4096)
    detail, parent_oof_frame, _ = reference.fit_targets(pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, reference.DEFAULT_CONFIG); frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True); parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET].reset_index(drop=True)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), parent_rows["canonical"].astype(str).to_numpy()): raise RuntimeError("exact v7 Nc parent row alignment failed")
    y = frame["target"].to_numpy(float); parent_oof = parent_rows["prediction"].to_numpy(float); target_keys = sorted(set(frame["canonical"]) | set(test.loc[test["target_type"] == TARGET, "canonical"])); target_indices = [key_to_index[value] for value in target_keys]; base, feature_report = proxy.graph_features(molecules, target_indices); features, feature_names = augment_degree_features(molecules, target_indices, base); feature_row = {value: row for row, value in enumerate(target_keys)}; rows = np.asarray([feature_row[value] for value in frame["canonical"]], dtype=np.int64); test_rows = np.asarray([feature_row[value] for value in test.loc[test["target_type"] == TARGET, "canonical"]], dtype=np.int64)
    groups = np.asarray([plumbing.no_stereo(value) for value in frame["canonical"]], dtype=object); scaffolds = np.asarray([plumbing.scaffold(value) for value in frame["canonical"]], dtype=object); folds = plumbing.folds_for(groups, 5); residual = y - parent_oof; candidate = np.full(len(y), np.nan); similarity = np.full(len(y), np.nan); fold_rows = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold); training = np.flatnonzero(folds != fold); fitted = proxy.model(); fitted.fit(features[rows[training]], residual[training]); candidate[validation] = parent_oof[validation] + RESIDUAL_WEIGHT * fitted.predict(features[rows[validation]]); global_validation = np.asarray([key_to_index[value] for value in frame.iloc[validation]["canonical"]], dtype=np.int64); global_training = np.asarray([key_to_index[value] for value in frame.iloc[training]["canonical"]], dtype=np.int64); similarity[validation] = proxy.nearest_similarity(fingerprints, global_validation, global_training); fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent_oof[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent_oof[validation]))})
    parent_r2 = float(r2_score(y, parent_oof)); candidate_r2 = float(r2_score(y, candidate)); delta = candidate_r2 - parent_r2; lower = bootstrap_lower(y, parent_oof, candidate, groups); panel, minimum_panel = proxy.panel_report(y, parent_oof, candidate, scaffolds, similarity)
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True); test_parent = detail[detail["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(float); fitted = proxy.model(); fitted.fit(features[rows], residual); test_candidate = test_parent + RESIDUAL_WEIGHT * fitted.predict(features[test_rows]); component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": test_parent, "candidate_prediction": test_candidate})
    if len(component) != 153 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all(): raise RuntimeError("Nc component output contract failed")
    component.to_csv(run_dir / "nc_component_predictions.csv", index=False); pd.DataFrame({"canonical": frame["canonical"].astype(str), "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel >= 0.0, "strict_no_regression": delta >= -0.003, "component_rows_pass": len(component) == 153}; passed = bool(all(gates.values())); source_names = ("round2_c064_nc_graph_degree_spectrum_residual.py", "round2_c062_tg_topological_shape_free_volume_residual.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py")
    report = {"schema_version": "ppp.round2.c064.nc-graph-degree-spectrum-residual.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 Nc regenerated in-process", "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False, "target": TARGET, "feature_names": feature_names, "feature_report": feature_report, "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "group_bootstrap_lower": lower, "panels": panel, "minimum_panel_delta": minimum_panel, "gates": gates, "decision": "pass_component_gate" if passed else "rejected_component_gate", "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))}, "source_hashes": {name: sha256_file(root / "tools" / name) for name in source_names}, "elapsed_seconds": time.time() - started}
    write_json(run_dir / "metrics.json", report); write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "feature_family": "graph-distance moments/eigen spectrum plus degree spectrum and free-volume proxies", "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": 10.0, "outer": "canonical_no_stereo GroupKFold(5)", "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False}); (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8"); (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8"); (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Nc parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file(): manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items(): manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8"); print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel}, sort_keys=True))


if __name__ == "__main__":
    main()
