#!/usr/bin/env python3
"""Bounded official-only EPS scaffold-balanced PLS diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cross_decomposition import PLSRegression
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


def folds_for(groups: np.ndarray, n_splits: int) -> np.ndarray:
    if len(np.unique(groups)) < n_splits:
        raise RuntimeError(f"need {n_splits} groups, found {len(np.unique(groups))}")
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def parent_arms(dense: np.ndarray, sparse_parts: list[Any], fingerprints: list[Any], y_global: np.ndarray, train: np.ndarray, validation: np.ndarray) -> np.ndarray:
    values = reference.predict_base_models(dense, sparse_parts, fingerprints, y_global, train, validation, reference.DEFAULT_CONFIG, TARGET)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite parent arm prediction")
    return values


def nested_parent(y: np.ndarray, groups: np.ndarray, outer_train: np.ndarray, outer_validation: np.ndarray, dense: np.ndarray, sparse_parts: list[Any], fingerprints: list[Any], y_global: np.ndarray, global_indices: np.ndarray) -> tuple[np.ndarray, list[float], float, float]:
    inner_folds = folds_for(groups[outer_train], 4)
    inner_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_arms[inner_folds == fold] = parent_arms(dense, sparse_parts, fingerprints, y_global, global_indices[local_train], global_indices[local_validation])
    outer_arms = parent_arms(dense, sparse_parts, fingerprints, y_global, global_indices[outer_train], global_indices[outer_validation])
    weights, intercept, _, _ = reference.blend_from_oof(y[outer_train], inner_arms)
    inner_parent = reference.clip_prediction(y[outer_train], inner_arms @ weights + intercept)
    outer_parent = reference.clip_prediction(y[outer_train], outer_arms @ weights + intercept)
    return outer_parent, weights.tolist(), float(intercept), float(r2_score(y[outer_train], inner_parent))


def fit_candidate(features: np.ndarray, y: np.ndarray, scaffolds: np.ndarray, train: np.ndarray, validation: np.ndarray, global_indices: np.ndarray) -> np.ndarray:
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
    counts = Counter(str(value) for value in scaffolds[train])
    maximum = max(counts.values())
    repetitions = np.asarray([max(1, int(round(maximum / counts[str(value)]))) for value in scaffolds[train]], dtype=np.int64)
    expanded = np.repeat(np.arange(len(train)), repetitions)
    model = PLSRegression(n_components=3, scale=True, max_iter=500, tol=1.0e-6)
    model.fit(train_x[expanded], y[train][expanded])
    prediction = np.asarray(model.predict(validation_x), dtype=np.float64).reshape(-1)
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
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    features = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    target_dense = reference.target_dense_features(features, cross_values, cross_available, TARGET)
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
        prediction = fit_candidate(features, y, scaffolds, training, validation, indices)
        baseline[validation] = parent
        candidate[validation] = prediction
        nearest[validation] = nearest_to_train([fingerprints[indices[row]] for row in validation], [fingerprints[indices[row]] for row in training])
        base_r2 = float(r2_score(y[validation], parent))
        cand_r2 = float(r2_score(y[validation], prediction))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "baseline_r2": base_r2, "candidate_r2": cand_r2, "delta_r2": cand_r2 - base_r2})
        blend_rows.append({"fold": fold, "weights": weights, "intercept": intercept, "inner_parent_r2": inner_r2})

    scaffold_groups = [name for name in sorted(set(scaffolds)) if int(np.sum(scaffolds == name)) >= 10]
    scaffold_holdout = {}
    for name in scaffold_groups:
        validation = np.flatnonzero(scaffolds == name)
        training = np.flatnonzero(scaffolds != name)
        parent, _, _, _ = nested_parent(y, groups, training, validation, target_dense, sparse_parts, fingerprints, y_global, indices)
        prediction = fit_candidate(features, y, scaffolds, training, validation, indices)
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
    panel_values.extend(float(item["delta_r2"]) for item in scaffold_holdout.values())
    min_panel = min(panel_values) if panel_values else None
    passed = bool(delta_r2 >= 0.01 and positive_folds >= 4 and bootstrap > 0.0 and not panel_incomplete and (min_panel is None or min_panel >= -0.003))
    script_path = root / "tools" / "round2_eps_scaffold_balanced_pls.py"
    reference_path = root / "tools" / "initial_reference_pipeline.py"
    audit = {
        "schema_version": "ppp.round2.eps-scaffold-balanced-pls-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C028-20260803-2130-eps-logtarget-extratrees",
        "official_inputs": inputs,
        "source_hashes": {"script": sha256_file(script_path), "reference_module": sha256_file(reference_path)},
        "target": TARGET,
        "rows": int(len(y)),
        "group_count": int(len(np.unique(groups))),
        "baseline_r2_nested_parent": baseline_r2,
        "candidate_r2_scaffold_balanced_pls": candidate_r2,
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
        "pass": passed,
        "decision": "component_pass" if passed else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({"canonical_no_stereo_group": groups, "fold": main_folds, "scaffold": scaffolds, "nearest_tanimoto": nearest, "has_other_property": auxiliary, "y": y, "nested_parent": baseline, "candidate": candidate}).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.eps-scaffold-balanced-pls.v1", "seed": 2026, "target": TARGET, "outer_folds": 5, "inner_folds": 4, "estimator": {"name": "PLSRegression", "n_components": 3, "scale": True, "max_iter": 500}, "scaffold_replication": "round(max_scaffold_count / scaffold_count), minimum 1", "descriptor_features": descriptor_names + physical_names, "official_inputs": inputs, "source_hashes": audit["source_hashes"]})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C029 EPS scaffold-balanced PLS\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(f"experiment_id={run_dir.name}\ntarget={TARGET}\nnested_parent_r2={baseline_r2:.12f}\ncandidate_r2={candidate_r2:.12f}\ndelta_r2={delta_r2:.12f}\npass={passed}\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([f"{audit['source_hashes']['script']}  SOURCE tools/round2_eps_scaffold_balanced_pls.py", f"{audit['source_hashes']['reference_module']}  SOURCE tools/initial_reference_pipeline.py"])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
