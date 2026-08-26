#!/usr/bin/env python3
"""Bounded official-only EPS periodic descriptor/log-target tree diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "eps"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def no_stereo(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("official SMILES failed RDKit parsing")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        value = ""
    return value or "ACYCLIC"


POLARIZABILITY = {1: 0.667, 5: 3.03, 6: 1.76, 7: 1.10, 8: 0.802, 9: 0.557, 14: 5.38, 15: 3.63, 16: 2.90, 17: 2.18, 35: 3.05, 53: 5.35}
ATOMIC_VOLUME = {1: 7.24, 5: 20.0, 6: 16.35, 7: 14.39, 8: 12.43, 9: 11.2, 14: 38.4, 15: 35.0, 16: 25.6, 17: 22.7, 35: 32.5, 53: 42.0}
ELEMENTS = [0, 1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]


def periodic_features(molecule: Any) -> np.ndarray:
    atoms = list(molecule.GetAtoms())
    bonds = list(molecule.GetBonds())
    z = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=np.float64)
    pol = np.asarray([POLARIZABILITY.get(int(value), 0.04 * value) for value in z], dtype=np.float64)
    volume = np.asarray([ATOMIC_VOLUME.get(int(value), 1.5 * value) for value in z], dtype=np.float64)
    degree = np.asarray([atom.GetDegree() for atom in atoms], dtype=np.float64)
    aromatic = np.asarray([float(atom.GetIsAromatic()) for atom in atoms], dtype=np.float64)
    hetero = np.asarray([float(atom.GetAtomicNum() not in (0, 1, 6)) for atom in atoms], dtype=np.float64)
    base = np.asarray([
        float(len(atoms)), float(np.sum(z > 1)), float(molecule.GetNumBonds()), float(molecule.GetRingInfo().NumRings()),
        float(np.sum(pol)), float(np.mean(pol)) if len(pol) else 0.0, float(np.std(pol)) if len(pol) else 0.0,
        float(np.sum(volume)), float(np.mean(volume)) if len(volume) else 0.0, float(np.std(volume)) if len(volume) else 0.0,
        float(np.sum(pol * volume)), float(np.sum(z * pol)), float(np.sum(z * volume)),
        float(np.mean(degree)) if len(degree) else 0.0, float(np.std(degree)) if len(degree) else 0.0,
        float(np.max(degree)) if len(degree) else 0.0, float(np.mean(aromatic)) if len(aromatic) else 0.0,
        float(np.mean(hetero)) if len(hetero) else 0.0,
    ], dtype=np.float64)
    elements = np.asarray([float(np.sum(z == value)) for value in ELEMENTS], dtype=np.float64)
    bonds_summary = np.asarray([
        float(sum(bond.GetBondTypeAsDouble() == value for bond in bonds)) for value in (1.0, 2.0, 3.0, 1.5)
    ] + [
        float(sum(bond.GetIsAromatic() for bond in bonds)),
        float(sum(bond.GetIsConjugated() for bond in bonds)),
        float(sum(bond.IsInRing() for bond in bonds)),
        float(sum(bond.GetBondTypeAsDouble() for bond in bonds)),
    ], dtype=np.float64)
    distance_count = np.zeros(8, dtype=np.float64)
    distance_pol = np.zeros(8, dtype=np.float64)
    distance_volume = np.zeros(8, dtype=np.float64)
    distance_pv = np.zeros(8, dtype=np.float64)
    distance_aromatic = np.zeros(8, dtype=np.float64)
    if len(atoms) > 1:
        distance = np.asarray(Chem.GetDistanceMatrix(molecule), dtype=np.float64)
        for left in range(len(atoms)):
            for right in range(left + 1, len(atoms)):
                step = int(round(float(distance[left, right])))
                if not 1 <= step <= 8:
                    continue
                column = step - 1
                distance_count[column] += 1.0
                distance_pol[column] += pol[left] * pol[right]
                distance_volume[column] += volume[left] * volume[right]
                distance_pv[column] += pol[left] * volume[right] + pol[right] * volume[left]
                distance_aromatic[column] += aromatic[left] * aromatic[right]
    norm = max(float(len(atoms)), 1.0)
    interaction = np.concatenate([distance_count, distance_pol / norm, distance_volume / norm, distance_pv / norm, distance_aromatic / norm])
    vector = np.concatenate([base, elements, bonds_summary, interaction])
    vector[~np.isfinite(vector)] = np.nan
    return vector


def folds_for(groups: np.ndarray, n_splits: int) -> np.ndarray:
    if len(np.unique(groups)) < n_splits:
        raise RuntimeError(f"need {n_splits} groups, found {len(np.unique(groups))}")
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def parent_arms(dense: np.ndarray, sparse_parts: list[Any], fingerprints: list[Any], y_global: np.ndarray, train: np.ndarray, validation: np.ndarray) -> np.ndarray:
    return reference.predict_base_models(dense, sparse_parts, fingerprints, y_global, train, validation, reference.DEFAULT_CONFIG, TARGET)


def nested_parent(y: np.ndarray, groups: np.ndarray, outer_train: np.ndarray, outer_validation: np.ndarray, dense: np.ndarray, sparse_parts: list[Any], fingerprints: list[Any], y_global: np.ndarray, global_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    inner_folds = folds_for(groups[outer_train], 4)
    inner_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_arms[inner_folds == fold] = parent_arms(dense, sparse_parts, fingerprints, y_global, global_indices[local_train], global_indices[local_validation])
    outer_arms = parent_arms(dense, sparse_parts, fingerprints, y_global, global_indices[outer_train], global_indices[outer_validation])
    weights, intercept, _, _ = reference.blend_from_oof(y[outer_train], inner_arms)
    parent = reference.clip_prediction(y[outer_train], outer_arms @ weights + intercept)
    return parent, weights, [float(intercept)], float(r2_score(y[outer_train], reference.clip_prediction(y[outer_train], inner_arms @ weights + intercept)))


def fit_candidate(features: np.ndarray, y: np.ndarray, train: np.ndarray, validation: np.ndarray, global_indices: np.ndarray) -> np.ndarray:
    train_x = np.asarray(features[global_indices[train]], dtype=np.float64).copy()
    validation_x = np.asarray(features[global_indices[validation]], dtype=np.float64).copy()
    absolute_limit = float(reference.DEFAULT_CONFIG["dense_abs_limit"])
    train_x[~np.isfinite(train_x) | (np.abs(train_x) > absolute_limit)] = np.nan
    validation_x[~np.isfinite(validation_x) | (np.abs(validation_x) > absolute_limit)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(train_x)
    validation_x = imputer.transform(validation_x)
    keep = np.ptp(train_x, axis=0) > 1.0e-12
    train_x = train_x[:, keep]
    validation_x = validation_x[:, keep]
    model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=2, max_features=0.35, random_state=2030, n_jobs=2)
    model.fit(train_x, np.log1p(y[train]))
    prediction = np.expm1(model.predict(validation_x))
    return reference.clip_prediction(y[train], prediction)


def nearest_to_train(left: list[Any], right: list[Any]) -> np.ndarray:
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, right)) for fp in left], dtype=np.float64)


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(500):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in chosen])
        if len(rows) > 1 and np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"pre-created protocol directory is required: {run_dir}")
    if {path.name for path in run_dir.iterdir()} - {"protocol.json"}:
        raise RuntimeError(f"refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    periodic = np.vstack([periodic_features(molecule) for molecule in molecules])
    features = np.hstack([descriptor, physical, periodic]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    target_dense = reference.target_dense_features(np.hstack([descriptor, physical]), cross_values, cross_available, TARGET)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    isomeric = frame["canonical"].to_numpy(object)
    groups = np.asarray([no_stereo(value) for value in isomeric], dtype=object)
    scaffolds = np.asarray([scaffold(value) for value in isomeric], dtype=object)
    indices = np.asarray([key_to_index[value] for value in isomeric], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[indices] = y
    target_index = reference.TARGETS.index(TARGET)
    availability = np.sum(cross_available, axis=1) - cross_available[:, target_index]

    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan)
    candidate = np.full(len(y), np.nan)
    nearest = np.full(len(y), np.nan)
    fold_rows = []
    blend_rows = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        parent, weights, intercept, inner_r2 = nested_parent(y, groups, training, validation, target_dense, sparse_parts, fingerprints, y_global, indices)
        prediction = fit_candidate(features, y, training, validation, indices)
        baseline[validation] = parent
        candidate[validation] = prediction
        nearest[validation] = nearest_to_train([fingerprints[indices[row]] for row in validation], [fingerprints[indices[row]] for row in training])
        base_r2 = float(r2_score(y[validation], parent))
        cand_r2 = float(r2_score(y[validation], prediction))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "baseline_r2": base_r2, "candidate_r2": cand_r2, "delta_r2": cand_r2 - base_r2})
        blend_rows.append({"fold": fold, "weights": weights.tolist(), "intercept": intercept[0], "inner_parent_r2": inner_r2})

    scaffold_groups = [name for name in sorted(set(scaffolds)) if int(np.sum(scaffolds == name)) >= 10]
    scaffold_holdout = {}
    for name in scaffold_groups:
        validation = np.flatnonzero(scaffolds == name)
        training = np.flatnonzero(scaffolds != name)
        parent, _, _, _ = nested_parent(y, groups, training, validation, target_dense, sparse_parts, fingerprints, y_global, indices)
        prediction = fit_candidate(features, y, training, validation, indices)
        parent_r2 = float(r2_score(y[validation], parent))
        candidate_r2 = float(r2_score(y[validation], prediction))
        scaffold_holdout[name] = {"rows": int(len(validation)), "baseline_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2}

    panels = {}
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (nearest >= lower) & (nearest < upper)
        delta = panel_delta(y, baseline, candidate, selected)
        panels[f"similarity_{name}"] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "insufficient_or_constant"}
    auxiliary = availability[indices] > 0
    for name, selected in (("available_other_property", auxiliary), ("missing_other_property", ~auxiliary)):
        delta = panel_delta(y, baseline, candidate, selected)
        panels[f"availability_{name}"] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "insufficient_or_constant"}
    scaffold_slice = {}
    for name in scaffold_groups:
        selected = scaffolds == name
        delta = panel_delta(y, baseline, candidate, selected)
        scaffold_slice[name] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "insufficient_or_constant"}
    panels["scaffold_slice_canonical_oof"] = scaffold_slice

    baseline_r2 = float(r2_score(y, baseline))
    candidate_r2 = float(r2_score(y, candidate))
    delta_r2 = candidate_r2 - baseline_r2
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    panel_values = []
    panel_incomplete = False
    for value in panels.values():
        items = value.values() if isinstance(value, dict) and "delta_r2" not in value else [value]
        for item in items:
            if item.get("delta_r2") is None:
                panel_incomplete = True
            else:
                panel_values.append(float(item["delta_r2"]))
    for item in scaffold_holdout.values():
        panel_values.append(float(item["delta_r2"]))
    min_panel = min(panel_values) if panel_values else None
    pass_value = bool(delta_r2 >= 0.01 and positive_folds >= 4 and bootstrap > 0.0 and not panel_incomplete and (min_panel is None or min_panel >= -0.003))
    script_path = root / "tools" / "round2_eps_logtarget_extratrees.py"
    reference_path = root / "tools" / "initial_reference_pipeline.py"
    audit = {
        "schema_version": "ppp.round2.eps-periodic-logtarget-extratrees-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C027-20260803-2100-ei-absolute-electronic-topology-v2",
        "official_inputs": inputs,
        "source_hashes": {"script": sha256_file(script_path), "reference_module": sha256_file(reference_path)},
        "target": TARGET,
        "rows": int(len(y)),
        "group_count": int(len(np.unique(groups))),
        "baseline_r2_nested_parent": baseline_r2,
        "candidate_r2_logtarget_extratrees": candidate_r2,
        "delta_r2": delta_r2,
        "positive_outer_folds": positive_folds,
        "group_r2_bootstrap_lower": bootstrap,
        "outer_folds": fold_rows,
        "blend_folds": blend_rows,
        "panels": panels,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": min((float(item["delta_r2"]) for item in scaffold_holdout.values()), default=None),
        "min_panel_delta": min_panel,
        "panel_incomplete": panel_incomplete,
        "pass": pass_value,
        "decision": "component_pass" if pass_value else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({"canonical_no_stereo_group": groups, "fold": main_folds, "scaffold": scaffolds, "nearest_tanimoto": nearest, "has_other_property": auxiliary, "y": y, "nested_parent": baseline, "candidate": candidate}).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.eps-periodic-logtarget-extratrees.v1", "seed": 2030, "target": TARGET, "outer_folds": 5, "inner_folds": 4, "target_transform": "log1p/expm1", "estimator": {"n_estimators": 256, "min_samples_leaf": 2, "max_features": 0.35}, "physical_features": physical_names, "official_inputs": inputs, "source_hashes": audit["source_hashes"]})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C028 EPS periodic/log-target ExtraTrees\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(f"experiment_id={run_dir.name}\ntarget={TARGET}\nnested_parent_r2={baseline_r2:.12f}\ncandidate_r2={candidate_r2:.12f}\ndelta_r2={delta_r2:.12f}\npass={pass_value}\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([f"{audit['source_hashes']['script']}  SOURCE tools/round2_eps_logtarget_extratrees.py", f"{audit['source_hashes']['reference_module']}  SOURCE tools/initial_reference_pipeline.py"])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
