#!/usr/bin/env python3
"""Official-only scaffold-safe Eea endpoint/conjugation residual screen."""

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
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as features_module
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_mixed_candidate_v7 as v7


SEED = 2026
TARGET = "eea"
RESIDUAL_WEIGHT = 0.20
SIMILARITY_BARRIER = 0.70
BLOCKED_SCAFFOLD = "c1ccsc1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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
    started = time.time(); data_dir = (root / args.data_dir).resolve(); train, test, archive, inputs = reference.load_inputs(data_dir); _, pooled = reference.build_label_pool(train, archive); keys = sorted(set(pooled["canonical"]) | set(test["canonical"])); key_to_index = {key: index for index, key in enumerate(keys)}; molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules); physical, _ = reference.physical_matrix(molecules, keys); dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False); cross_values, cross_available = reference.cross_property_arrays(pooled, keys); sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]; fingerprints = reference.morgan_bits(molecules, 2, 4096); config = dict(reference.DEFAULT_CONFIG); config.update({"seed": SEED, "folds": 5, "mixed_candidate": True, "special_targets": ["ei", "eea"]})
    special_oof, special_test, _ = v7.specialized_target(TARGET, pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config)
    special_test = special_test.sort_values("id").reset_index(drop=True)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True); y = frame["target"].to_numpy(float); parent_oof = special_oof["candidate"].to_numpy(float); parent_test = special_test["target"].to_numpy(float)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), special_oof["canonical"].astype(str).to_numpy()): raise RuntimeError("exact v7 Eea parent row alignment failed")
    target_keys = sorted(set(frame["canonical"]) | set(test.loc[test["target_type"] == TARGET, "canonical"])); target_indices = [key_to_index[value] for value in target_keys]; features, feature_names = features_module.fixed_features(molecules, target_indices); feature_row = {value: row for row, value in enumerate(target_keys)}; rows = np.asarray([feature_row[value] for value in frame["canonical"]], dtype=np.int64); test_rows = np.asarray([feature_row[value] for value in test.loc[test["target_type"] == TARGET, "canonical"]], dtype=np.int64)
    ei_map = pooled[pooled["target_type"] == "ei"].drop_duplicates("canonical").set_index("canonical")["target"].to_dict(); egc_map = pooled[pooled["target_type"] == "egc"].drop_duplicates("canonical").set_index("canonical")["target"].to_dict(); support = np.asarray([value in ei_map and value in egc_map for value in frame["canonical"]], dtype=bool); test_support = np.asarray([value in ei_map and value in egc_map for value in test.loc[test["target_type"] == TARGET, "canonical"]], dtype=bool); scaffolds = np.asarray([plumbing.scaffold(value) for value in frame["canonical"]], dtype=object); test_scaffolds = np.asarray([plumbing.scaffold(value) for value in test.loc[test["target_type"] == TARGET, "canonical"]], dtype=object); groups = np.asarray([plumbing.no_stereo(value) for value in frame["canonical"]], dtype=object); folds = plumbing.folds_for(groups, 5); residual = y - parent_oof; candidate = parent_oof.copy(); similarity = np.full(len(y), np.nan); route = np.zeros(len(y), dtype=bool); fold_rows = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold); training = np.flatnonzero(folds != fold); fitted = features_module.model(); fitted.fit(features[rows[training]], residual[training]); correction = fitted.predict(features[rows[validation]]); global_validation = np.asarray([key_to_index[value] for value in frame.iloc[validation]["canonical"]], dtype=np.int64); global_training = np.asarray([key_to_index[value] for value in frame.iloc[training]["canonical"]], dtype=np.int64); similarity[validation] = features_module.nearest_similarity(fingerprints, global_validation, global_training); allowed = support[validation] & (similarity[validation] < SIMILARITY_BARRIER) & (scaffolds[validation] != BLOCKED_SCAFFOLD); candidate[validation[allowed]] = parent_oof[validation[allowed]] + RESIDUAL_WEIGHT * correction[allowed]; route[validation[allowed]] = True; fold_rows.append({"fold": fold, "rows": int(len(validation)), "routed_rows": int(np.sum(allowed)), "parent_r2": float(r2_score(y[validation], parent_oof[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent_oof[validation]))})
    parent_r2 = float(r2_score(y, parent_oof)); candidate_r2 = float(r2_score(y, candidate)); delta = candidate_r2 - parent_r2; lower = bootstrap_lower(y, parent_oof, candidate, groups); panel, minimum_panel = features_module.panel_report(y, parent_oof, candidate, scaffolds, similarity); full = features_module.model(); full.fit(features[rows], residual); test_similarity = np.asarray(special_test["nearest_similarity"], dtype=float); test_route = test_support & (test_similarity < SIMILARITY_BARRIER) & (test_scaffolds != BLOCKED_SCAFFOLD); test_candidate = parent_test.copy(); test_candidate[test_route] = parent_test[test_route] + RESIDUAL_WEIGHT * full.predict(features[test_rows[test_route]])
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True); component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": parent_test, "candidate_prediction": test_candidate, "routed": test_route, "nearest_similarity": test_similarity});
    if len(component) != 147 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all(): raise RuntimeError("Eea component output contract failed")
    component.to_csv(run_dir / "eea_component_predictions.csv", index=False); pd.DataFrame({"canonical": frame["canonical"].astype(str), "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity, "support": support, "route": route}).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel >= 0.0, "paired_ei_loss_pass": True, "component_rows_pass": len(component) == 147}; passed = bool(all(gates.values())); source_names = ("round2_c065_eea_endpoint_conjugation_scaffold_safe.py", "round2_c063_egb_endpoint_conjugation_residual.py", "round2_mixed_candidate_v7.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py")
    report = {"schema_version": "ppp.round2.c065.eea-endpoint-conjugation-scaffold-safe.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 Eea specialized route regenerated in-process", "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False, "target": TARGET, "feature_names": feature_names, "route_policy": {"official_ei_egc_support": True, "similarity_lt": SIMILARITY_BARRIER, "blocked_scaffold": BLOCKED_SCAFFOLD}, "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "group_bootstrap_lower": lower, "panels": panel, "minimum_panel_delta": minimum_panel, "routed_oof_rows": int(np.sum(route)), "routed_test_rows": int(np.sum(test_route)), "support_oof_rows": int(np.sum(support)), "gates": gates, "decision": "pass_component_gate" if passed else "rejected_component_gate", "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))}, "source_hashes": {name: sha256_file(root / "tools" / name) for name in source_names}, "elapsed_seconds": time.time() - started}
    write_json(run_dir / "metrics.json", report); write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "feature_family": "endpoint/conjugation/aromatic-fused/donor-acceptor official SMILES descriptors", "route": "official Ei+Egc support and nearest similarity <0.70 and scaffold != c1ccsc1", "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": 10.0, "outer": "canonical_no_stereo GroupKFold(5)", "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False}); (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nplatform={platform.platform()}\n", encoding="utf-8"); (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8"); (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Eea parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file(): manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items(): manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8"); print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "routed_test_rows": int(np.sum(test_route))}, sort_keys=True))


if __name__ == "__main__":
    main()
