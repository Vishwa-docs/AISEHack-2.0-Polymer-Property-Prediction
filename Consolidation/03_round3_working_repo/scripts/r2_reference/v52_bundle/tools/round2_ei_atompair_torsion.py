#!/usr/bin/env python3
"""Bounded official-only Ei AtomPair/TopologicalTorsion specialist."""

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
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "ei"
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"
EXPERIMENT_ID = "R2-C026-20260803-2040-ei-atompair-torsion"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    if np.any(folds < 0):
        raise RuntimeError("incomplete GroupKFold assignment")
    return folds


def count_fingerprint_matrix(molecules: list[Any], generator: Any, bits: int) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, molecule in enumerate(molecules):
        fingerprint = generator.GetCountFingerprint(molecule)
        for column, count in fingerprint.GetNonzeroElements().items():
            rows.append(row)
            columns.append(int(column))
            values.append(float(np.log1p(float(count))))
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(molecules), bits), dtype=np.float64)


def nearest_to_train(left: list[Any], right: list[Any]) -> np.ndarray:
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, right)) for fp in left], dtype=np.float64)


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray, seed: int = 2026) -> float:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(500):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[group] for group in selected])
        if len(rows) < 2 or float(np.var(y[rows])) <= 1.0e-15:
            continue
        values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def make_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        value = ""
    return value or "ACYCLIC"


def fold_dense(matrix: np.ndarray, train_rows: np.ndarray, validation_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_raw = np.asarray(matrix[train_rows], dtype=np.float64).copy()
    validation_raw = np.asarray(matrix[validation_rows], dtype=np.float64).copy()
    train_raw[~np.isfinite(train_raw)] = np.nan
    validation_raw[~np.isfinite(validation_raw)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_raw)
    validation_imputed = imputer.transform(validation_raw)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    validation_scaled = scaler.transform(validation_imputed)
    return train_imputed, validation_imputed, train_scaled, validation_scaled


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
    existing = {path.name for path in run_dir.iterdir()}
    if existing - {"protocol.json"}:
        raise RuntimeError(f"refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()

    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, TARGET)
    atom_pair = count_fingerprint_matrix(molecules, rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048), 2048)
    torsion = count_fingerprint_matrix(molecules, rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048), 2048)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    c001_report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    isomeric_keys = frame["canonical"].to_numpy(object)
    groups = np.asarray([reference.canonicalize(Chem.MolToSmiles(Chem.MolFromSmiles(value), isomericSmiles=False)) for value in isomeric_keys], dtype=object)
    indices = np.asarray([key_to_index[key] for key in isomeric_keys], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[indices] = y
    folds = folds_for(groups)
    baseline = np.full(len(y), np.nan, dtype=np.float64)
    ridge_prediction = np.full(len(y), np.nan, dtype=np.float64)
    tree_prediction = np.full(len(y), np.nan, dtype=np.float64)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    retained_features: list[int] = []
    weights = c001_report["validation"]["target_reports"][TARGET]["blend_weights"]
    blend_weights = np.asarray([weights[name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)
    intercept = float(c001_report["validation"]["target_reports"][TARGET]["blend_intercept"])

    for fold in range(5):
        train_rows = np.flatnonzero(folds != fold)
        validation_rows = np.flatnonzero(folds == fold)
        base = reference.predict_base_models(
            target_dense,
            sparse_parts,
            fingerprints,
            y_global,
            indices[train_rows],
            indices[validation_rows],
            reference.DEFAULT_CONFIG,
            TARGET,
        )
        baseline[validation_rows] = base @ blend_weights + intercept
        dense_train, dense_validation, dense_train_scaled, dense_validation_scaled = fold_dense(
            target_dense, indices[train_rows], indices[validation_rows]
        )
        atom_train = atom_pair[indices[train_rows]]
        atom_validation = atom_pair[indices[validation_rows]]
        torsion_train = torsion[indices[train_rows]]
        torsion_validation = torsion[indices[validation_rows]]
        ridge_train = sparse.hstack([atom_train, torsion_train, sparse.csr_matrix(dense_train_scaled)], format="csr")
        ridge_validation = sparse.hstack([atom_validation, torsion_validation, sparse.csr_matrix(dense_validation_scaled)], format="csr")
        ridge = Ridge(alpha=10.0, solver="lsqr", max_iter=5000, tol=1.0e-4)
        ridge.fit(ridge_train, y[train_rows])
        ridge_values = reference.clip_prediction(y[train_rows], ridge.predict(ridge_validation))
        tree_train = np.hstack([atom_train.toarray(), torsion_train.toarray(), dense_train])
        tree_validation = np.hstack([atom_validation.toarray(), torsion_validation.toarray(), dense_validation])
        tree = ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=2,
            max_features=0.35,
            random_state=2027,
            n_jobs=2,
        )
        tree.fit(tree_train, y[train_rows])
        tree_values = reference.clip_prediction(y[train_rows], tree.predict(tree_validation))
        ridge_prediction[validation_rows] = ridge_values
        tree_prediction[validation_rows] = tree_values
        candidate[validation_rows] = reference.clip_prediction(y[train_rows], 0.5 * ridge_values + 0.5 * tree_values)
        baseline_fold = float(r2_score(y[validation_rows], baseline[validation_rows]))
        ridge_fold = float(r2_score(y[validation_rows], ridge_values))
        tree_fold = float(r2_score(y[validation_rows], tree_values))
        candidate_fold = float(r2_score(y[validation_rows], candidate[validation_rows]))
        retained_features.append(int(ridge_train.shape[1]))
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation_rows)),
            "baseline_r2": baseline_fold,
            "ridge_r2": ridge_fold,
            "tree_r2": tree_fold,
            "candidate_r2": candidate_fold,
            "delta_r2": candidate_fold - baseline_fold,
        })

    nearest = np.empty(len(y), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        nearest[validation] = nearest_to_train(
            [fingerprints[indices[row]] for row in validation],
            [fingerprints[indices[row]] for row in training],
        )
    target_index = reference.TARGETS.index(TARGET)
    auxiliary = np.sum(cross_available[indices], axis=1) - cross_available[indices, target_index] > 0
    panels: dict[str, Any] = {}
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (nearest >= lower) & (nearest < upper)
        value = panel_delta(y, baseline, candidate, selected)
        if value is not None:
            panels[f"similarity_{name}"] = {"rows": int(np.sum(selected)), "delta_r2": value}
    for name, selected in (("available_other_property", auxiliary), ("missing_other_property", ~auxiliary)):
        value = panel_delta(y, baseline, candidate, selected)
        if value is not None:
            panels[f"availability_{name}"] = {"rows": int(np.sum(selected)), "delta_r2": value}
    scaffolds = np.asarray([make_scaffold(value) for value in isomeric_keys], dtype=object)
    scaffold_details: dict[str, Any] = {}
    for scaffold in sorted(set(scaffolds)):
        selected = scaffolds == scaffold
        if int(np.sum(selected)) < 10:
            continue
        value = panel_delta(y, baseline, candidate, selected)
        if value is not None:
            scaffold_details[str(scaffold)] = {"rows": int(np.sum(selected)), "delta_r2": value}
    if scaffold_details:
        panels["scaffold_minimum_10_rows"] = {
            "groups": int(len(scaffold_details)),
            "minimum_delta_r2": float(min(item["delta_r2"] for item in scaffold_details.values())),
            "details": scaffold_details,
        }
    baseline_r2 = float(r2_score(y, baseline))
    candidate_r2 = float(r2_score(y, candidate))
    delta_r2 = candidate_r2 - baseline_r2
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    panel_values = [float(item["delta_r2"]) for name, item in panels.items() if name != "scaffold_minimum_10_rows"]
    if "scaffold_minimum_10_rows" in panels:
        panel_values.append(float(panels["scaffold_minimum_10_rows"]["minimum_delta_r2"]))
    min_panel = min(panel_values) if panel_values else None
    passed = bool(
        delta_r2 >= 0.01
        and positive_folds >= 4
        and bootstrap > 0.0
        and (min_panel is None or min_panel >= -0.003)
    )
    audit = {
        "schema_version": "ppp.round2.ei-atompair-torsion-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C025-20260803-2025-eea-char-ngram-ridge",
        "official_inputs": inputs,
        "target": TARGET,
        "rows": int(len(y)),
        "group_count": int(len(np.unique(groups))),
        "baseline_r2": baseline_r2,
        "candidate_r2": candidate_r2,
        "delta_r2": delta_r2,
        "positive_folds": positive_folds,
        "group_r2_bootstrap_lower": bootstrap,
        "folds": fold_rows,
        "ridge_features_per_fold": retained_features,
        "panels": panels,
        "min_panel_delta": min_panel,
        "pass": passed,
        "decision": "component_pass" if passed else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({
        "canonical_no_stereo_group": groups,
        "fold": folds,
        "scaffold": scaffolds,
        "nearest_tanimoto": nearest,
        "has_other_property": auxiliary,
        "y": y,
        "baseline": baseline,
        "ridge": ridge_prediction,
        "tree": tree_prediction,
        "candidate": candidate,
    }).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.ei-atompair-torsion.v1",
        "seed": 2027,
        "folds": 5,
        "target": TARGET,
        "atom_pair_bits": 2048,
        "topological_torsion_bits": 2048,
        "ridge_alpha": 10.0,
        "extra_trees_estimators": 256,
        "extra_trees_min_samples_leaf": 2,
        "extra_trees_max_features": 0.35,
        "fixed_blend": [0.5, 0.5],
        "official_inputs": inputs,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# R2-C026 Ei AtomPair/TopologicalTorsion\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(
        f"experiment_id={run_dir.name}\n"
        f"target={TARGET}\n"
        f"candidate_r2={candidate_r2:.12f}\n"
        f"delta_r2={delta_r2:.12f}\n"
        f"pass={passed}\n",
        encoding="utf-8",
    )
    manifest_lines = []
    for path in sorted(run_dir.iterdir()):
        if path.name == "artifact_manifest.sha256" or not path.is_file():
            continue
        manifest_lines.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
