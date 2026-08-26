#!/usr/bin/env python3
"""Parent-compatible bridge of the fixed C088 topology residual to C050-v7 EPS."""

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c088_eps_topo_happy_nested as topo
import round2_eea_cross_target_oof_residual_stack as plumbing


TARGET = "eps"
SEED = 2026
ALPHA = 30.0
WEIGHT = 0.25


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


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--data-dir", default="ppp-round-2"); parser.add_argument("--run-dir", required=True); args = parser.parse_args()
    root = Path(args.root).resolve(); run_dir = Path(args.run_dir); run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time(); data_dir = (root / args.data_dir).resolve(); train, test, archive, inputs = reference.load_inputs(data_dir); _, pooled = reference.build_label_pool(train, archive); pooled = pooled.reset_index(drop=True)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"])); key_to_index = {key: index for index, key in enumerate(keys)}; molecules = reference.build_molecules(keys); descriptor, _ = reference.descriptor_matrix(molecules); physical, _ = reference.physical_matrix(molecules, keys); dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]; fingerprints = reference.morgan_bits(molecules, 2, 4096)
    config = dict(reference.DEFAULT_CONFIG); config.update({"seed": SEED, "folds": 5, "mixed_candidate": True, "special_targets": []})
    parent_detail, parent_oof_frame, _ = reference.fit_targets(pooled, test, keys, dense_base, *reference.cross_property_arrays(pooled, keys), sparse_parts, fingerprints, config)
    replay_detail, replay_oof_frame, _ = reference.fit_targets(pooled, test, keys, dense_base, *reference.cross_property_arrays(pooled, keys), sparse_parts, fingerprints, config)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True); y = frame["target"].to_numpy(float); canonical = frame["canonical"].astype(str).to_numpy(object); groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object); scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object); folds = folds_for(groups)
    parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET][["canonical", "target", "prediction"]].rename(columns={"prediction": "parent"}); parent_rows_replay = replay_oof_frame[replay_oof_frame["target_type"] == TARGET][["canonical", "target", "prediction"]].rename(columns={"prediction": "replay"}); aligned = frame[["canonical", "target"]].merge(parent_rows, on=["canonical", "target"], how="left", validate="one_to_one").merge(parent_rows_replay, on=["canonical", "target"], how="left", validate="one_to_one"); parent_oof = aligned["parent"].to_numpy(float); replay_oof = aligned["replay"].to_numpy(float)
    parent_detail_eps = parent_detail[parent_detail["target_type"] == TARGET].sort_values("id"); replay_detail_eps = replay_detail[replay_detail["target_type"] == TARGET].sort_values("id"); parent_test = parent_detail_eps["model_prediction"].to_numpy(float); replay_test = replay_detail_eps["model_prediction"].to_numpy(float)
    replay_oof_max = float(np.max(np.abs(parent_oof - replay_oof))); replay_test_max = float(np.max(np.abs(parent_test - replay_test))); expected_parent_r2 = 0.7835054389877212; generated_parent_r2 = finite_r2(y, parent_oof)
    if replay_oof_max > 1.0e-12 or replay_test_max > 1.0e-12 or abs(generated_parent_r2 - expected_parent_r2) > 1.0e-10:
        raise RuntimeError("generated C050-v7 EPS parent replay/value gate failed")
    topo_all = topo.topo_matrix(molecules, np.arange(len(keys), dtype=np.int64)); rows = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64); candidate = np.full(len(y), np.nan, dtype=np.float64); similarity = np.full(len(y), np.nan, dtype=np.float64); fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold); training = np.flatnonzero(folds != fold); model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)); model.fit(topo_all[rows[training]], y[training] - parent_oof[training]); candidate[validation] = parent_oof[validation] + WEIGHT * model.predict(topo_all[rows[validation]]); similarity[validation] = topo.similarity_for(fingerprints, rows[validation], rows[training]); fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": finite_r2(y[validation], parent_oof[validation]), "candidate_r2": finite_r2(y[validation], candidate[validation]), "delta_r2": finite_r2(y[validation], candidate[validation]) - finite_r2(y[validation], parent_oof[validation])})
    parent_r2 = finite_r2(y, parent_oof); candidate_r2 = finite_r2(y, candidate); delta = candidate_r2 - parent_r2; lower = bootstrap_lower(y, parent_oof, candidate, groups); panels, minimum_panel = topo.panel_report(y, parent_oof, candidate, groups, scaffolds, similarity)
    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)); model.fit(topo_all[rows], y - parent_oof); test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True); test_rows = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64); test_candidate = parent_test + WEIGHT * model.predict(topo_all[test_rows]); component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": parent_test, "candidate_prediction": test_candidate})
    if len(component) != 153 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(test_candidate).all(): raise RuntimeError("v7 bridge component output contract failed")
    component.to_csv(run_dir / "eps_component_predictions.csv", index=False); pd.DataFrame({"canonical": canonical, "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {"parent_value_pass": abs(generated_parent_r2 - expected_parent_r2) <= 1.0e-10, "parent_replay_oof_pass": replay_oof_max <= 1.0e-12, "parent_replay_test_pass": replay_test_max <= 1.0e-12, "gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0.0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel is not None and minimum_panel >= 0.0, "component_rows_pass": len(component) == 153}
    source_names = ("round2_c089_eps_topo_happy_v7_bridge.py", "round2_c088_eps_topo_happy_nested.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py"); source_hashes = {name: sha256_file(root / "tools" / name) for name in source_names}; report = {"schema_version": "ppp.round2.c089.eps-topo-happy-v7-bridge.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "parent": "C050-v7 regenerated from official inputs", "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False, "target": TARGET, "feature_count": topo.FEATURES, "alpha": ALPHA, "blend_weight": WEIGHT, "generated_parent_r2": generated_parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "group_bootstrap_lower": lower, "panels": panels, "minimum_panel_delta": minimum_panel, "parent_replay_oof_max_abs": replay_oof_max, "parent_replay_test_max_abs": replay_test_max, "gates": gates, "decision": "pass_component_gate" if all(gates.values()) else "rejected_component_gate", "projected_mean_if_assembled": 0.8731493564508485 + delta / 7.0, "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))}, "source_hashes": source_hashes, "elapsed_seconds": time.time() - started}; write_json(run_dir / "metrics.json", report); write_json(run_dir / "config.json", {"schema_version": report["schema_version"], "seed": SEED, "target": TARGET, "feature_count": topo.FEATURES, "feature_family": "fixed C088 Topo-HAPPY-like hashed topology counts", "ridge_alpha": ALPHA, "blend": {"parent": 0.75, "topology_residual": 0.25}, "parent": "C050-v7 regenerated through initial_reference_pipeline", "outer": "canonical no-stereo GroupKFold(5)", "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False}); (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8"); (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8"); (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. C050-v7 EPS parent `{parent_r2:.12f}`, bridge candidate `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Projected seven-target mean if assembled `{report['projected_mean_if_assembled']:.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8"); manifest_paths = [path for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]; lines = [f"{sha256_file(path)}  {path.relative_to(run_dir)}" for path in manifest_paths]; lines.extend(f"{digest}  SOURCE tools/{name}" for name, digest in source_hashes.items()); (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "projected_mean_if_assembled": report["projected_mean_if_assembled"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
