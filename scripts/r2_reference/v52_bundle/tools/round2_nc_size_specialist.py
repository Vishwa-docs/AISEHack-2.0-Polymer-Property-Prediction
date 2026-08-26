#!/usr/bin/env python3
"""Evaluate the preregistered official-only Nc size/free-volume specialist."""

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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "nc"
TARGETS = reference.TARGETS
MODEL_NAMES = ("size_ridge_alpha_10", "size_extra_trees_32_min_leaf_4")
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    return {
        "rows": int(len(y)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and not np.isclose(np.var(y), 0.0) else None,
        "mae": float(mean_absolute_error(y, pred)) if len(y) else None,
        "rmse": float(np.sqrt(np.mean(np.square(y - pred)))) if len(y) else None,
    }


def load_weights(root: Path) -> np.ndarray:
    report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    values = report["validation"]["target_reports"][TARGET]["blend_weights"]
    return np.asarray([float(values[name]) for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)


def scaffold_group(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        value = ""
    return value or "ACYCLIC"


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError("Incomplete fold assignment")
    return result


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    output = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in training]
        for index in validation:
            output[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    return output


def bootstrap_lower(y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> float:
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(2000):
        sample = rng.integers(0, len(y), size=len(y))
        if np.isclose(np.var(y[sample]), 0.0):
            continue
        values.append(float(r2_score(y[sample], candidate[sample]) - r2_score(y[sample], baseline[sample])))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def size_block(descriptor: np.ndarray, descriptor_names: list[str], physical: np.ndarray, physical_names: list[str]) -> tuple[np.ndarray, list[str]]:
    wanted_descriptors = ["MolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA", "FractionCSP3", "NumRotatableBonds"]
    wanted_physical = ["smiles_length", "atom_count", "heavy_atom_count", "dummy_atom_count", "ring_count", "aromatic_atom_count", "hetero_atom_count", "rotatable_bonds_approx", "branch_count", "n_count", "o_count", "s_count", "si_count"]
    di = {name: index for index, name in enumerate(descriptor_names)}
    pi = {name: index for index, name in enumerate(physical_names)}
    missing = [name for name in wanted_descriptors if name not in di] + [name for name in wanted_physical if name not in pi]
    if missing:
        raise RuntimeError(f"Size descriptor block missing: {missing}")
    values = np.column_stack([descriptor[:, di[name]] for name in wanted_descriptors] + [physical[:, pi[name]] for name in wanted_physical]).astype(np.float64, copy=False)
    return values, [f"descriptor_{name}" for name in wanted_descriptors] + [f"physical_{name}" for name in wanted_physical]


def specialist_predictions(block: np.ndarray, y: np.ndarray, train_rows: np.ndarray, validation_rows: np.ndarray) -> np.ndarray:
    clean = np.asarray(block, dtype=np.float64).copy()
    clean[(~np.isfinite(clean)) | (np.abs(clean) > 1.0e12)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(clean[train_rows])
    validation_x = imputer.transform(clean[validation_rows])
    scaler = StandardScaler().fit(train_x)
    ridge = Ridge(alpha=10.0).fit(scaler.transform(train_x), y[train_rows])
    trees = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=4, max_features=0.8, random_state=2026, n_jobs=2).fit(train_x, y[train_rows])
    output = np.column_stack([ridge.predict(scaler.transform(validation_x)), trees.predict(validation_x)])
    for column in range(output.shape[1]):
        output[:, column] = reference.clip_prediction(y[train_rows], output[:, column])
    return output


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
        raise RuntimeError(f"Pre-created protocol directory is required: {run_dir}")
    existing = {path.name for path in run_dir.iterdir()}
    if existing - {"protocol.json"}:
        raise RuntimeError(f"Refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    target_rows = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = target_rows["target"].to_numpy(float)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    target_indices = np.asarray([key_to_index[value] for value in target_rows["canonical"]], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[target_indices] = y
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    block, block_names = size_block(descriptor, descriptor_names, physical, physical_names)
    dense_base = np.hstack([descriptor, physical])
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    c001_dense = reference.target_dense_features(dense_base, cross_values, cross_available, TARGET)
    c001_sparse = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    c001_fingerprints = reference.morgan_bits(molecules, 2, 4096)
    weights = load_weights(root)
    reports: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    panel_outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for panel, groups in (("canonical_group", target_rows["canonical"].to_numpy(object)), ("scaffold_family", target_rows["smiles"].map(scaffold_group).to_numpy(object))):
        folds = folds_for(groups)
        baseline = np.full(len(y), np.nan)
        candidate = np.full((len(y), len(MODEL_NAMES)), np.nan)
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            base_parts = reference.predict_base_models(c001_dense, c001_sparse, c001_fingerprints, y_global, target_indices[train_rows], target_indices[validation_rows], reference.DEFAULT_CONFIG, TARGET)
            baseline[validation_rows] = base_parts @ weights
            candidate[validation_rows] = specialist_predictions(block[target_indices], y, train_rows, validation_rows)
            fold_rows.extend({"panel": panel, "fold": fold, "target": TARGET, "index": int(index), "group": groups[index]} for index in validation_rows)
        nearest = nearest_similarity([c001_fingerprints[index] for index in target_indices], folds)
        panel_outputs[panel] = (baseline, candidate, folds)
        report = {"baseline": metric(y, baseline), "models": {name: metric(y, candidate[:, index]) for index, name in enumerate(MODEL_NAMES)}, "folds": [], "slices": {}}
        for fold in range(5):
            selected = folds == fold
            row = {"fold": fold, "baseline": metric(y[selected], baseline[selected])}
            for index, name in enumerate(MODEL_NAMES):
                row[name] = metric(y[selected], candidate[selected, index])
                row[f"{name}_delta_r2"] = float(row[name]["r2"] - row["baseline"]["r2"])
            report["folds"].append(row)
        slices = {
            "smiles_long": target_rows["smiles"].str.len().to_numpy() >= np.quantile(target_rows["smiles"].str.len().to_numpy(), 0.75),
            "heavy_high": physical[target_indices, physical_names.index("heavy_atom_count")] >= np.quantile(physical[target_indices, physical_names.index("heavy_atom_count")], 0.75),
            "nearest_lt_0.30": nearest < 0.30,
            "nearest_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
            "nearest_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
            "nearest_ge_0.70": nearest >= 0.70,
        }
        for slice_name, selected in slices.items():
            if int(np.sum(selected)) < 5:
                continue
            report["slices"][slice_name] = {"rows": int(np.sum(selected)), "baseline": metric(y[selected], baseline[selected]), **{name: metric(y[selected], candidate[selected, index]) for index, name in enumerate(MODEL_NAMES)}}
            for name in MODEL_NAMES:
                base_r2 = report["slices"][slice_name]["baseline"]["r2"]
                arm_r2 = report["slices"][slice_name][name]["r2"]
                report["slices"][slice_name][f"{name}_delta_r2"] = None if base_r2 is None or arm_r2 is None else float(arm_r2 - base_r2)
        reports[panel] = report
        for index, name in enumerate(("frozen_c001_blend", *MODEL_NAMES)):
            prediction = baseline if index == 0 else candidate[:, index - 1]
            metrics_rows.append({"panel": panel, "method": name, **metric(y, prediction)})
        for index, row in target_rows.iterrows():
            prediction_rows.append({"panel": panel, "index": int(index), "target": float(y[index]), "fold": int(folds[index]), "baseline": float(baseline[index]), **{name: float(candidate[index, column]) for column, name in enumerate(MODEL_NAMES)}})
    canonical_baseline, canonical_candidate, canonical_folds = panel_outputs["canonical_group"]
    selected_name = max(MODEL_NAMES, key=lambda name: reports["canonical_group"]["models"][name]["r2"])
    selected_index = MODEL_NAMES.index(selected_name)
    fold_deltas = [float(reports["canonical_group"]["folds"][fold][f"{selected_name}_delta_r2"]) for fold in range(5)]
    long_delta = reports["canonical_group"]["slices"]["smiles_long"][f"{selected_name}_delta_r2"]
    heavy_delta = reports["canonical_group"]["slices"]["heavy_high"][f"{selected_name}_delta_r2"]
    low_deltas = [value[f"{selected_name}_delta_r2"] for name, value in reports["canonical_group"]["slices"].items() if name.startswith("nearest_") and value[f"{selected_name}_delta_r2"] is not None]
    scaffold_delta = float(reports["scaffold_family"]["models"][selected_name]["r2"] - reports["scaffold_family"]["baseline"]["r2"])
    boot = bootstrap_lower(y, canonical_candidate[:, selected_index], canonical_baseline)
    passing = bool(reports["canonical_group"]["models"][selected_name]["r2"] - reports["canonical_group"]["baseline"]["r2"] >= 0.01 and sum(value > 0 for value in fold_deltas) >= 4 and boot > 0.0 and long_delta >= 0.0 and heavy_delta >= 0.0 and scaffold_delta >= 0.0 and low_deltas and min(low_deltas) >= 0.0)
    audit = {
        "schema_version": "ppp.round2.nc-size-specialist-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C008-20260803-1731-residual-diagnosis",
        "official_inputs": inputs,
        "official_hashes_pass": all(inputs[name]["sha256"] == expected for name, expected in reference.EXPECTED_HASHES.items()),
        "feature_names": block_names,
        "models": list(MODEL_NAMES),
        "target": TARGET,
        "panels": reports,
        "selected_model": selected_name,
        "selected_delta_r2": float(reports["canonical_group"]["models"][selected_name]["r2"] - reports["canonical_group"]["baseline"]["r2"]),
        "selected_positive_folds": int(sum(value > 0 for value in fold_deltas)),
        "selected_bootstrap_lower_bound": boot,
        "long_slice_delta": long_delta,
        "heavy_slice_delta": heavy_delta,
        "scaffold_family_delta": scaffold_delta,
        "min_low_similarity_delta": min(low_deltas) if low_deltas else None,
        "passing_component": passing,
        "decision": "component_pass" if passing else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame(metrics_rows).to_csv(run_dir / "panel_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_assignments.csv", index=False)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.nc-size-specialist.v1", "seed": 2026, "folds": 5, "models": list(MODEL_NAMES), "feature_names": block_names, "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", audit)
    (run_dir / "decision.md").write_text(f"# R2-C009 Nc size/free-volume specialist decision\n\nDecision: **{audit['decision']}**.\n\nNo candidate changed unless every preregistered component gate passes; the run produced train-side panels only.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "panel_metrics.csv", "predictions.csv", "fold_assignments.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "selected_model": selected_name, "selected_delta_r2": audit["selected_delta_r2"], "selected_positive_folds": audit["selected_positive_folds"], "bootstrap_lower": boot, "long_delta": long_delta, "heavy_delta": heavy_delta, "scaffold_delta": scaffold_delta, "min_low_similarity_delta": audit["min_low_similarity_delta"], "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
