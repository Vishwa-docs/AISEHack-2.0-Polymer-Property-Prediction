#!/usr/bin/env python3
"""Fixed official-only Gaussian-process residual screen for Nc."""

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as panels
import round2_eea_cross_target_oof_residual_stack as plumbing


TARGET = "nc"
SEED = 2026
EXPECTED_PARENT_R2 = 0.8397322432486007


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    return result


def finite_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, prediction))


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups); members = {group: np.flatnonzero(groups == group) for group in unique}; rng = np.random.default_rng(SEED); values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True); rows = np.concatenate([members[group] for group in selected])
        if not np.isclose(np.var(y[rows]), 0.0):
            values.append(finite_r2(y[rows], candidate[rows]) - finite_r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025))


def model() -> object:
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=5.0, length_scale_bounds="fixed") + WhiteKernel(noise_level=0.10, noise_level_bounds="fixed")
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=1.0e-6, normalize_y=True, optimizer=None, random_state=SEED, n_restarts_optimizer=0))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--data-dir", default="ppp-round-2"); parser.add_argument("--run-dir", required=True); args = parser.parse_args()
    root = Path(args.root).resolve(); run_dir = Path(args.run_dir); run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time(); train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve()); _, pooled = reference.build_label_pool(train, archive); pooled = pooled.reset_index(drop=True); keys = sorted(set(pooled["canonical"]) | set(test["canonical"])); key_to_index = {key: index for index, key in enumerate(keys)}; molecules = reference.build_molecules(keys); descriptor, descriptor_names = reference.descriptor_matrix(molecules); physical, physical_names = reference.physical_matrix(molecules, keys); features = np.hstack([descriptor, physical]).astype(np.float64, copy=False); cross_values, cross_available = reference.cross_property_arrays(pooled, keys); sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]; fingerprints = reference.morgan_bits(molecules, 2, 4096); config = dict(reference.DEFAULT_CONFIG); config.update({"seed": SEED, "folds": 5, "mixed_candidate": True, "special_targets": []})
    parent_detail, parent_oof_frame, _ = reference.fit_targets(pooled, test, keys, features, cross_values, cross_available, sparse_parts, fingerprints, config); replay_detail, replay_oof_frame, _ = reference.fit_targets(pooled, test, keys, features, cross_values, cross_available, sparse_parts, fingerprints, config)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True); y = frame["target"].to_numpy(float); canonical = frame["canonical"].astype(str).to_numpy(object); groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object); scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object); folds = folds_for(groups); rows = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET][["canonical", "target", "prediction"]].rename(columns={"prediction": "parent"}); replay_rows = replay_oof_frame[replay_oof_frame["target_type"] == TARGET][["canonical", "target", "prediction"]].rename(columns={"prediction": "replay"}); aligned = frame[["canonical", "target"]].merge(parent_rows, on=["canonical", "target"], how="left", validate="one_to_one").merge(replay_rows, on=["canonical", "target"], how="left", validate="one_to_one"); parent_oof = aligned["parent"].to_numpy(float); replay_oof = aligned["replay"].to_numpy(float); parent_detail_target = parent_detail[parent_detail["target_type"] == TARGET].sort_values("id"); replay_detail_target = replay_detail[replay_detail["target_type"] == TARGET].sort_values("id"); parent_test = parent_detail_target["model_prediction"].to_numpy(float); replay_test = replay_detail_target["model_prediction"].to_numpy(float); parent_replay_oof_max = float(np.max(np.abs(parent_oof - replay_oof))); parent_replay_test_max = float(np.max(np.abs(parent_test - replay_test))); parent_r2 = finite_r2(y, parent_oof)
    if abs(parent_r2 - EXPECTED_PARENT_R2) > 1.0e-10 or parent_replay_oof_max > 1.0e-12 or parent_replay_test_max > 1.0e-12:
        raise RuntimeError("exact v7 Nc parent replay/value gate failed")
    candidate = np.full(len(y), np.nan, dtype=np.float64); similarity = np.full(len(y), np.nan, dtype=np.float64); fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold); training = np.flatnonzero(folds != fold); fitted = model(); fitted.fit(features[rows[training]], y[training] - parent_oof[training]); candidate[validation] = parent_oof[validation] + fitted.predict(features[rows[validation]]); similarity[validation] = panels.nearest_similarity(fingerprints, rows[validation], rows[training]); fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": finite_r2(y[validation], parent_oof[validation]), "candidate_r2": finite_r2(y[validation], candidate[validation]), "delta_r2": finite_r2(y[validation], candidate[validation]) - finite_r2(y[validation], parent_oof[validation])})
    if not np.isfinite(candidate).all(): raise RuntimeError("non-finite Nc GP OOF candidate")
    candidate_r2 = finite_r2(y, candidate); delta = candidate_r2 - parent_r2; lower = bootstrap_lower(y, parent_oof, candidate, groups); panel_report, minimum_panel = panels.panel_report(y, parent_oof, candidate, scaffolds, similarity)
    fitted = model(); fitted.fit(features[rows], y - parent_oof); test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True); test_rows = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64); test_candidate = parent_test + fitted.predict(features[test_rows]); component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": parent_test, "candidate_prediction": test_candidate})
    if len(component) != 153 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(test_candidate).all(): raise RuntimeError("Nc GP component output contract failed")
    component.to_csv(run_dir / "nc_component_predictions.csv", index=False); pd.DataFrame({"canonical": canonical, "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {"parent_value_pass": abs(parent_r2 - EXPECTED_PARENT_R2) <= 1.0e-10, "parent_replay_oof_pass": parent_replay_oof_max <= 1.0e-12, "parent_replay_test_pass": parent_replay_test_max <= 1.0e-12, "gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0.0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel is not None and minimum_panel >= 0.0, "component_rows_pass": len(component) == 153}
    source_names = ("round2_c090_nc_gaussian_process_residual.py", "initial_reference_pipeline.py", "round2_c063_egb_endpoint_conjugation_residual.py", "round2_eea_cross_target_oof_residual_stack.py"); source_hashes = {name: sha256_file(root / "tools" / name) for name in source_names}; report = {"schema_version": "ppp.round2.c090.nc-gaussian-process-residual.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "parent": "C050-v7 Nc regenerated from official inputs", "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False, "target": TARGET, "feature_count": int(features.shape[1]), "feature_names": descriptor_names + physical_names, "kernel": "ConstantKernel(1.0)*RBF(5.0)+WhiteKernel(0.10), fixed optimizer=None", "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "group_bootstrap_lower": lower, "panels": panel_report, "minimum_panel_delta": minimum_panel, "parent_replay_oof_max_abs": parent_replay_oof_max, "parent_replay_test_max_abs": parent_replay_test_max, "gates": gates, "decision": "pass_component_gate" if all(gates.values()) else "rejected_component_gate", "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))}, "source_hashes": source_hashes, "elapsed_seconds": time.time() - started}; write_json(run_dir / "metrics.json", report); write_json(run_dir / "config.json", {"schema_version": report["schema_version"], "seed": SEED, "target": TARGET, "estimator": "GaussianProcessRegressor", "kernel": report["kernel"], "optimizer": None, "normalize_y": True, "outer": "canonical no-stereo GroupKFold(5)", "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False}); (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8"); (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8"); (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Exact-v7 Nc parent `{parent_r2:.12f}`, GP candidate `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8"); manifest_paths = [path for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]; lines = [f"{sha256_file(path)}  {path.relative_to(run_dir)}" for path in manifest_paths]; lines.extend(f"{digest}  SOURCE tools/{name}" for name, digest in source_hashes.items()); (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
