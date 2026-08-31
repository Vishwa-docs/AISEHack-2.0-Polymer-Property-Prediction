#!/usr/bin/env python3
"""Official-only EPS Topo-HAPPY-like residual with a nested grouped parent."""

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
from rdkit import Chem, DataStructs, RDLogger
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGET = "eps"
SEED = 2026
FEATURES = 2048
ALPHA = 30.0
WEIGHT = 0.25
TARGETS = tuple(reference.TARGETS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray, count: int = 5) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=count).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError("incomplete grouped fold assignment")
    return result


def finite_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, prediction))


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if not np.isclose(np.var(y[rows]), 0.0):
            values.append(finite_r2(y[rows], candidate[rows]) - finite_r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def add_token(features: dict[str, float], token: str, amount: float = 1.0) -> None:
    features[token] = features.get(token, 0.0) + amount


def topo_tokens(molecule: Chem.Mol) -> dict[str, float]:
    """Create compact role/connectivity tokens, with no learned or external state."""
    features: dict[str, float] = {}
    endpoints = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    distances = {index: 10_000 for index in range(molecule.GetNumAtoms())}
    queue: list[int] = []
    for index in endpoints:
        distances[index] = 0
        queue.append(index)
    while queue:
        current = queue.pop(0)
        for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
            index = neighbor.GetIdx()
            if distances[index] > distances[current] + 1:
                distances[index] = distances[current] + 1
                queue.append(index)
    endpoint_pair_distance = None
    if len(endpoints) >= 2:
        try:
            endpoint_pair_distance = int(Chem.GetShortestPath(molecule, endpoints[0], endpoints[1]).__len__() - 1)
        except Exception:
            endpoint_pair_distance = None
    add_token(features, f"META:endpoints:{min(len(endpoints), 3)}")
    add_token(features, f"META:atoms:{min(molecule.GetNumAtoms(), 64)}")
    add_token(features, f"META:bonds:{min(molecule.GetNumBonds(), 96)}")
    if endpoint_pair_distance is not None:
        add_token(features, f"META:endpoint_path:{min(endpoint_pair_distance, 32)}")
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        atomic = atom.GetAtomicNum()
        if atomic == 0:
            element = "STAR"
        else:
            element = atom.GetSymbol()
        distance = min(distances.get(index, 10_000), 6)
        role = "endpoint" if distance == 0 else "mainline" if distance <= 2 else "sideline"
        aromatic = int(atom.GetIsAromatic())
        ring = int(atom.IsInRing())
        degree = min(atom.GetDegree(), 5)
        charge = max(-2, min(2, atom.GetFormalCharge()))
        add_token(features, f"A:{role}:{element}:{aromatic}:{ring}")
        add_token(features, f"A:env:{element}:{degree}:{distance}:{charge}")
        add_token(features, f"A:role_degree:{role}:{degree}")
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtom()
        right = bond.GetEndAtom()
        pair = "-".join(sorted(("STAR" if left.GetAtomicNum() == 0 else left.GetSymbol(), "STAR" if right.GetAtomicNum() == 0 else right.GetSymbol())))
        bond_type = str(bond.GetBondType()).replace(" ", "_")
        role_left = "endpoint" if distances[left.GetIdx()] == 0 else "mainline" if distances[left.GetIdx()] <= 2 else "sideline"
        role_right = "endpoint" if distances[right.GetIdx()] == 0 else "mainline" if distances[right.GetIdx()] <= 2 else "sideline"
        add_token(features, f"B:{pair}:{bond_type}:{int(bond.GetIsAromatic())}")
        add_token(features, f"B:roles:{role_left}-{role_right}:{bond_type}")
        if role_left == "mainline" and role_right == "mainline":
            add_token(features, f"B:connector:{pair}:{bond_type}")
    return features


def topo_matrix(molecules: list[Chem.Mol], indices: np.ndarray) -> np.ndarray:
    rows = [topo_tokens(molecules[int(index)]) for index in indices]
    hasher = FeatureHasher(n_features=FEATURES, input_type="dict", alternate_sign=False)
    matrix = hasher.transform(rows).toarray().astype(np.float64, copy=False)
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix


def masked_target_dense(
    pooled: pd.DataFrame,
    allowed_groups: set[str],
    all_keys: list[str],
    dense_base: np.ndarray,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed = pooled[pooled["group"].isin(allowed_groups)].copy()
    cross_values, cross_available = reference.cross_property_arrays(allowed, all_keys)
    return reference.target_dense_features(dense_base, cross_values, cross_available, target), cross_values, cross_available


def target_global_labels(frame: pd.DataFrame, allowed_groups: set[str], groups: np.ndarray, global_indices: np.ndarray, key_count: int) -> np.ndarray:
    output = np.full(key_count, np.nan, dtype=np.float64)
    allowed = np.isin(groups, list(allowed_groups))
    output[global_indices[allowed]] = frame.loc[allowed, "target"].to_numpy(float)
    return output


def nested_parent_fold(
    frame: pd.DataFrame,
    target_groups: np.ndarray,
    pooled: pd.DataFrame,
    all_keys: list[str],
    dense_base: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    global_indices: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    outer_groups = set(target_groups[outer_train].tolist())
    inner_folds = folds_for(target_groups[outer_train], 4)
    inner_oof = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for inner_fold in range(4):
        valid_pos = np.flatnonzero(inner_folds == inner_fold)
        train_pos = np.flatnonzero(inner_folds != inner_fold)
        inner_train = outer_train[train_pos]
        inner_valid = outer_train[valid_pos]
        inner_groups = set(target_groups[inner_train].tolist())
        inner_dense, _, _ = masked_target_dense(pooled, inner_groups, all_keys, dense_base, TARGET)
        inner_labels = target_global_labels(frame, inner_groups, target_groups, global_indices, len(all_keys))
        inner_oof[valid_pos] = reference.predict_base_models(inner_dense, sparse_parts, fingerprints, inner_labels, global_indices[inner_train], global_indices[inner_valid], config, TARGET)
    weights, intercept, _, _ = reference.blend_from_oof(frame["target"].to_numpy(float)[outer_train], inner_oof)
    inner_parent = reference.clip_prediction(frame["target"].to_numpy(float)[outer_train], inner_oof @ weights + intercept)
    outer_dense, _, _ = masked_target_dense(pooled, outer_groups, all_keys, dense_base, TARGET)
    outer_labels = target_global_labels(frame, outer_groups, target_groups, global_indices, len(all_keys))
    outer_arms = reference.predict_base_models(outer_dense, sparse_parts, fingerprints, outer_labels, global_indices[outer_train], global_indices[outer_validation], config, TARGET)
    outer_parent = reference.clip_prediction(frame["target"].to_numpy(float)[outer_train], outer_arms @ weights + intercept)
    return outer_parent, inner_parent


def nested_parent_test(
    frame: pd.DataFrame,
    target_groups: np.ndarray,
    pooled: pd.DataFrame,
    all_keys: list[str],
    dense_base: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    global_indices: np.ndarray,
    test_global_indices: np.ndarray,
    config: dict[str, object],
) -> np.ndarray:
    all_positions = np.arange(len(frame), dtype=np.int64)
    all_groups = set(target_groups.tolist())
    inner_folds = folds_for(target_groups, 4)
    inner_oof = np.full((len(frame), 4), np.nan, dtype=np.float64)
    for inner_fold in range(4):
        valid = np.flatnonzero(inner_folds == inner_fold)
        train = np.flatnonzero(inner_folds != inner_fold)
        train_groups = set(target_groups[train].tolist())
        dense, _, _ = masked_target_dense(pooled, train_groups, all_keys, dense_base, TARGET)
        labels = target_global_labels(frame, train_groups, target_groups, global_indices, len(all_keys))
        inner_oof[valid] = reference.predict_base_models(dense, sparse_parts, fingerprints, labels, global_indices[train], global_indices[valid], config, TARGET)
    weights, intercept, _, _ = reference.blend_from_oof(frame["target"].to_numpy(float), inner_oof)
    full_dense, _, _ = masked_target_dense(pooled, all_groups, all_keys, dense_base, TARGET)
    full_labels = target_global_labels(frame, all_groups, target_groups, global_indices, len(all_keys))
    test_arms = reference.predict_base_models(full_dense, sparse_parts, fingerprints, full_labels, global_indices, test_global_indices, config, TARGET)
    return reference.clip_prediction(frame["target"].to_numpy(float), test_arms @ weights + intercept)


def similarity_for(fingerprints: list[object], rows: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    result = np.full(len(rows), np.nan, dtype=np.float64)
    train_fps = [fingerprints[int(index)] for index in train_rows]
    for position, row in enumerate(rows):
        result[position] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(row)], train_fps))
    return result


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, Any], float | None]:
    panels: dict[str, Any] = {}
    deltas: list[float] = []
    for name, selected in {"similarity_lt_0.30": similarity < 0.30, "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50), "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70), "similarity_ge_0.70": similarity >= 0.70}.items():
        rows = int(np.sum(selected)); group_count = int(np.unique(groups[selected]).size); eligible = rows >= 20 and group_count >= 5
        delta = finite_r2(y[selected], candidate[selected]) - finite_r2(y[selected], parent[selected]) if eligible else None
        panels[name] = {"rows": rows, "groups": group_count, "eligible": bool(eligible), "delta_r2": delta}
        if delta is not None:
            deltas.append(float(delta))
    scaffold_deltas: list[float] = []
    for value in np.unique(scaffolds):
        selected = scaffolds == value
        if int(np.sum(selected)) >= 10 and int(np.unique(groups[selected]).size) >= 3 and not np.isclose(np.var(y[selected]), 0.0):
            scaffold_deltas.append(finite_r2(y[selected], candidate[selected]) - finite_r2(y[selected], parent[selected]))
    panels["scaffold_groups_ge_3"] = {"evaluated": len(scaffold_deltas), "minimum_delta_r2": min(scaffold_deltas) if scaffold_deltas else None}
    if scaffold_deltas:
        deltas.append(min(scaffold_deltas))
    return panels, (min(deltas) if deltas else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time()
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    pooled = pooled.reset_index(drop=True)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    pooled["group"] = [plumbing.no_stereo(value) for value in pooled["canonical"]]
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    canonical = frame["canonical"].astype(str).to_numpy(object)
    groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    target_folds = folds_for(groups, 5)
    topo_all = topo_matrix(molecules, np.arange(len(keys), dtype=np.int64))
    rows = global_indices
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True)
    test_rows = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
    config = dict(reference.DEFAULT_CONFIG); config.update({"seed": SEED, "folds": 5, "mixed_candidate": True, "special_targets": [TARGET]})

    def generate_parent() -> tuple[np.ndarray, np.ndarray]:
        parent_oof_local = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            validation = np.flatnonzero(target_folds == fold); training = np.flatnonzero(target_folds != fold)
            outer, _ = nested_parent_fold(frame, groups, pooled, keys, dense_base, sparse_parts, fingerprints, global_indices, training, validation, config)
            parent_oof_local[validation] = outer
        test_parent_local = nested_parent_test(frame, groups, pooled, keys, dense_base, sparse_parts, fingerprints, global_indices, test_rows, config)
        return parent_oof_local, test_parent_local

    parent_oof, parent_test = generate_parent()
    replay_oof, replay_test = generate_parent()
    replay_oof_max = float(np.max(np.abs(parent_oof - replay_oof))); replay_test_max = float(np.max(np.abs(parent_test - replay_test)))
    if replay_oof_max > 1.0e-12 or replay_test_max > 1.0e-12:
        raise RuntimeError("nested parent replay is not deterministic")

    candidate = np.full(len(y), np.nan, dtype=np.float64)
    similarity = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation = np.flatnonzero(target_folds == fold); training = np.flatnonzero(target_folds != fold)
        outer_parent, inner_parent = nested_parent_fold(frame, groups, pooled, keys, dense_base, sparse_parts, fingerprints, global_indices, training, validation, config)
        model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        model.fit(topo_all[rows[training]], y[training] - inner_parent)
        correction = model.predict(topo_all[rows[validation]])
        candidate[validation] = outer_parent + WEIGHT * correction
        similarity[validation] = similarity_for(fingerprints, rows[validation], rows[training])
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": finite_r2(y[validation], outer_parent), "candidate_r2": finite_r2(y[validation], candidate[validation]), "delta_r2": finite_r2(y[validation], candidate[validation]) - finite_r2(y[validation], outer_parent)})
    if not np.isfinite(candidate).all():
        raise RuntimeError("non-finite Topo-HAPPY OOF candidate")
    parent_r2 = finite_r2(y, parent_oof); candidate_r2 = finite_r2(y, candidate); delta = candidate_r2 - parent_r2
    lower = bootstrap_lower(y, parent_oof, candidate, groups)
    panels, minimum_panel = panel_report(y, parent_oof, candidate, groups, scaffolds, similarity)
    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)); model.fit(topo_all[rows], y - parent_oof)
    test_candidate = parent_test + WEIGHT * model.predict(topo_all[test_rows])
    component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": parent_test, "candidate_prediction": test_candidate})
    if len(component) != 153 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(test_candidate).all():
        raise RuntimeError("Topo-HAPPY component output contract failed")
    component.to_csv(run_dir / "eps_component_predictions.csv", index=False)
    pd.DataFrame({"canonical": canonical, "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": target_folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0.0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel is not None and minimum_panel >= 0.0, "component_rows_pass": len(component) == 153, "parent_replay_oof_pass": replay_oof_max <= 1.0e-12, "parent_replay_test_pass": replay_test_max <= 1.0e-12}
    source_names = ("round2_c088_eps_topo_happy_nested.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py")
    source_hashes = {name: sha256_file(root / "tools" / name) for name in source_names}
    report = {"schema_version": "ppp.round2.c088.eps-topo-happy-nested.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "parent": "nested grouped official structural reference; external C050-v7 incumbent only for context", "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False, "target": TARGET, "feature_count": FEATURES, "alpha": ALPHA, "blend_weight": WEIGHT, "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "group_bootstrap_lower": lower, "panels": panels, "minimum_panel_delta": minimum_panel, "parent_replay_oof_max_abs": replay_oof_max, "parent_replay_test_max_abs": replay_test_max, "gates": gates, "decision": "pass_component_gate" if all(gates.values()) else "rejected_component_gate", "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))}, "source_hashes": source_hashes, "elapsed_seconds": time.time() - started}
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": report["schema_version"], "seed": SEED, "target": TARGET, "feature_count": FEATURES, "feature_family": "fixed hashed endpoint/mainline/sideline atom, bond, and connector topology tokens", "ridge_alpha": ALPHA, "blend": {"parent": 0.75, "topo_happy_residual": 0.25}, "outer": "canonical no-stereo GroupKFold(5) with nested grouped parent", "inner_parent_folds": 4, "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Nested parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest_paths = [path for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]
    lines = [f"{sha256_file(path)}  {path.relative_to(run_dir)}" for path in manifest_paths]
    lines.extend(f"{digest}  SOURCE tools/{name}" for name, digest in source_hashes.items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "parent_replay_oof_max_abs": replay_oof_max, "parent_replay_test_max_abs": replay_test_max, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
