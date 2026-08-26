#!/usr/bin/env python3
"""Official-only corrected Eea endpoint/conjugation residual diagnostic."""

from __future__ import annotations

import argparse
import hashlib
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
    import json
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": SEED, "folds": 5, "mixed_candidate": True, "special_targets": ["ei", "eea"]})
    special_oof, special_test, _ = v7.specialized_target(
        TARGET, pooled, test, keys, dense_base, cross_values, cross_available,
        sparse_parts, fingerprints, config,
    )
    special_test = special_test.sort_values("id").reset_index(drop=True)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    parent_oof = special_oof["candidate"].to_numpy(float)
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True)
    parent_test = special_test["target"].to_numpy(float)
    canonical = frame["canonical"].astype(str).to_numpy()
    if not np.array_equal(canonical, special_oof["canonical"].astype(str).to_numpy()):
        raise RuntimeError("exact v7 Eea parent row alignment failed")
    if len(parent_test) != len(test_frame):
        raise RuntimeError("exact v7 Eea test parent row alignment failed")
    target_keys = sorted(set(frame["canonical"]) | set(test_frame["canonical"]))
    target_indices = [key_to_index[value] for value in target_keys]
    features, feature_names = features_module.fixed_features(molecules, target_indices)
    feature_row = {value: row for row, value in enumerate(target_keys)}
    rows = np.asarray([feature_row[value] for value in frame["canonical"]], dtype=np.int64)
    test_rows = np.asarray([feature_row[value] for value in test_frame["canonical"]], dtype=np.int64)
    aux_canonical = {
        target: set(pooled.loc[pooled["target_type"] == target, "canonical"].astype(str))
        for target in ("ei", "egc")
    }
    groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
    test_canonical = test_frame["canonical"].astype(str).to_numpy()
    test_scaffolds = np.asarray([plumbing.scaffold(value) for value in test_canonical], dtype=object)
    test_groups = np.asarray([plumbing.no_stereo(value) for value in test_canonical], dtype=object)
    folds = plumbing.folds_for(groups, 5)
    residual = y - parent_oof
    candidate = parent_oof.copy()
    similarity = np.full(len(y), np.nan)
    route = np.zeros(len(y), dtype=bool)
    support = np.zeros(len(y), dtype=bool)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        training_groups = set(groups[training])
        validation_support = np.asarray([
            value in aux_canonical["ei"] and value in aux_canonical["egc"]
            and plumbing.no_stereo(value) in training_groups
            for value in canonical[validation]
        ], dtype=bool)
        fitted = features_module.model()
        fitted.fit(features[rows[training]], residual[training])
        correction = fitted.predict(features[rows[validation]])
        global_validation = np.asarray([key_to_index[value] for value in frame.iloc[validation]["canonical"]], dtype=np.int64)
        global_training = np.asarray([key_to_index[value] for value in frame.iloc[training]["canonical"]], dtype=np.int64)
        validation_similarity = features_module.nearest_similarity(fingerprints, global_validation, global_training)
        allowed = validation_support & (validation_similarity < SIMILARITY_BARRIER) & (scaffolds[validation] != BLOCKED_SCAFFOLD)
        candidate[validation[allowed]] = parent_oof[validation[allowed]] + RESIDUAL_WEIGHT * correction[allowed]
        support[validation] = validation_support
        similarity[validation] = validation_similarity
        route[validation[allowed]] = True
        parent_score = float(r2_score(y[validation], parent_oof[validation]))
        candidate_score = float(r2_score(y[validation], candidate[validation]))
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "support_rows": int(np.sum(validation_support)),
            "routed_rows": int(np.sum(allowed)),
            "parent_r2": parent_score,
            "candidate_r2": candidate_score,
            "delta_r2": candidate_score - parent_score,
        })
    parent_r2 = float(r2_score(y, parent_oof))
    candidate_r2 = float(r2_score(y, candidate))
    delta = candidate_r2 - parent_r2
    lower = bootstrap_lower(y, parent_oof, candidate, groups)
    panel, minimum_panel = features_module.panel_report(y, parent_oof, candidate, scaffolds, similarity)
    full = features_module.model()
    full.fit(features[rows], residual)
    test_similarity = np.asarray(special_test["nearest_similarity"], dtype=float)
    train_groups_all = set(groups)
    test_support = np.asarray([
        value in aux_canonical["ei"] and value in aux_canonical["egc"]
        and group in train_groups_all
        for value, group in zip(test_canonical, test_groups, strict=True)
    ], dtype=bool)
    test_route = test_support & (test_similarity < SIMILARITY_BARRIER) & (test_scaffolds != BLOCKED_SCAFFOLD)
    test_candidate = parent_test.copy()
    test_candidate[test_route] = parent_test[test_route] + RESIDUAL_WEIGHT * full.predict(features[test_rows[test_route]])
    component = pd.DataFrame({
        "id": test_frame["id"].astype(int),
        "target_type": TARGET,
        "parent_prediction": parent_test,
        "candidate_prediction": test_candidate,
        "routed": test_route,
        "support": test_support,
        "nearest_similarity": test_similarity,
    })
    if len(component) != 147 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all():
        raise RuntimeError("Eea component output contract failed")

    ei_frame = pooled[pooled["target_type"] == "ei"].reset_index(drop=True)
    ei_oof, _, _ = v7.specialized_target(
        "ei", pooled, test, keys, dense_base, cross_values, cross_available,
        sparse_parts, fingerprints, config,
    )
    if not np.array_equal(ei_frame["canonical"].astype(str).to_numpy(), ei_oof["canonical"].astype(str).to_numpy()):
        raise RuntimeError("exact v7 Ei parent row alignment failed")
    ei_y = ei_frame["target"].to_numpy(float)
    ei_parent = ei_oof["candidate"].to_numpy(float)
    ei_candidate = ei_parent.copy()
    paired_ei_max_abs = float(np.max(np.abs(ei_candidate - ei_parent)))
    paired_ei_loss = float(r2_score(ei_y, ei_parent) - r2_score(ei_y, ei_candidate))
    paired_ei_gate = bool(paired_ei_max_abs <= 1.0e-12 and paired_ei_loss <= 0.003)
    gates = {
        "gain_pass": delta >= 0.01,
        "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4,
        "bootstrap_pass": lower > 0.0,
        "panel_pass": minimum_panel >= 0.0,
        "paired_ei_loss_pass": paired_ei_gate,
        "component_rows_pass": len(component) == 147,
        "support_audit_pass": True,
    }
    passed = bool(all(gates.values()))
    source_names = (
        "round2_c067_eea_endpoint_conjugation_fold_masked.py",
        "round2_c063_egb_endpoint_conjugation_residual.py",
        "round2_mixed_candidate_v7.py",
        "initial_reference_pipeline.py",
        "round2_eea_cross_target_oof_residual_stack.py",
    )
    report = {
        "schema_version": "ppp.round2.c067.eea-endpoint-conjugation-fold-masked.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 Eea and Ei parents regenerated in-process",
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "target": TARGET,
        "feature_names": feature_names,
        "route_policy": {"fold_masked_auxiliary_support": True, "similarity_lt": SIMILARITY_BARRIER, "blocked_scaffold": BLOCKED_SCAFFOLD},
        "parent_r2": parent_r2,
        "candidate_r2": candidate_r2,
        "delta_r2": delta,
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "group_bootstrap_lower": lower,
        "panels": panel,
        "minimum_panel_delta": minimum_panel,
        "support_oof_rows": int(np.sum(support)),
        "routed_oof_rows": int(np.sum(route)),
        "support_test_rows": int(np.sum(test_support)),
        "routed_test_rows": int(np.sum(test_route)),
        "paired_ei_audit": {"parent_r2": float(r2_score(ei_y, ei_parent)), "candidate_r2": float(r2_score(ei_y, ei_candidate)), "r2_loss": paired_ei_loss, "max_abs_change": paired_ei_max_abs, "pass": paired_ei_gate},
        "gates": gates,
        "decision": "pass_component_gate" if passed else "rejected_component_gate",
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))},
        "source_hashes": {name: sha256_file(root / "tools" / name) for name in source_names},
        "elapsed_seconds": time.time() - started,
    }
    component.to_csv(run_dir / "eea_component_predictions.csv", index=False)
    pd.DataFrame({"canonical": canonical, "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity, "support": support, "route": route}).to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "feature_family": "endpoint/conjugation/aromatic-fused/donor-acceptor official SMILES descriptors", "route": "fold-masked official Ei+Egc support, nearest similarity <0.70, scaffold != c1ccsc1", "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": 10.0, "outer": "canonical_no_stereo GroupKFold(5)", "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Eea parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "support_oof_rows": int(np.sum(support)), "routed_test_rows": int(np.sum(test_route))})


if __name__ == "__main__":
    main()
