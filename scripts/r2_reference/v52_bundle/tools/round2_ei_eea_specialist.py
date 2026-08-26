#!/usr/bin/env python3
"""Evaluate the preregistered Ei/Eea electronic specialist comparison."""

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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
SPECIAL_TARGETS = ("ei", "eea")
MODEL_NAMES = ("electronic_ridge", "electronic_extra_trees", "electronic_gradient_boosting")
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    return {
        "rows": int(len(y)),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 and not np.isclose(np.var(y), 0.0) else None,
        "mae": float(mean_absolute_error(y, prediction)) if len(y) else None,
        "rmse": float(np.sqrt(np.mean(np.square(y - prediction)))) if len(y) else None,
    }


def folds_for(rows: pd.DataFrame) -> np.ndarray:
    result = np.full(len(rows), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    groups = rows["canonical"].to_numpy(object)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(rows)), groups=groups)):
        result[validation] = fold
    return result


def load_weights(root: Path, target: str) -> np.ndarray:
    report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    values = report["validation"]["target_reports"][target]["blend_weights"]
    return np.asarray([float(values[name]) for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=np.float64)


def electronic_block(descriptor: np.ndarray, names: list[str], physical: np.ndarray, physical_names: list[str]) -> tuple[np.ndarray, list[str]]:
    keep = [
        "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
        "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "AvgIpc", "BertzCT",
        "HallKierAlpha", "Ipc", "Kappa1", "Kappa2", "Kappa3", "TPSA", "FractionCSP3",
        "NHOHCount", "NOCount", "NumAromaticRings", "NumAromaticHeterocycles", "NumHAcceptors",
        "NumHDonors", "NumHeteroatoms", "NumHeterocycles", "MolLogP", "MolMR", "fr_Ar_N",
        "fr_Ar_NH", "fr_C_O", "fr_C_O_noCOO", "fr_NH0", "fr_NH1", "fr_nitrile", "fr_nitro",
        "fr_sulfone", "fr_halogen",
    ]
    pkeep = ["smiles_length", "heavy_atom_count", "aromatic_atom_count", "hetero_atom_count", "double_bond_count", "triple_bond_count", "branch_count"]
    di = {name: index for index, name in enumerate(names)}
    pi = {name: index for index, name in enumerate(physical_names)}
    missing = [name for name in (*keep, *pkeep) if name not in di and name not in pi]
    if missing:
        raise RuntimeError(f"Electronic descriptors missing: {missing}")
    columns = [descriptor[:, di[name]] for name in keep]
    columns.extend(physical[:, pi[name]] for name in pkeep)
    return np.column_stack(columns).astype(np.float64, copy=False), [f"descriptor_{name}" for name in keep] + [f"physical_{name}" for name in pkeep]


def features_for_target(block: np.ndarray, cross_values: np.ndarray, cross_available: np.ndarray, target: str) -> np.ndarray:
    values = cross_values.copy()
    available = cross_available.copy()
    index = TARGETS.index(target)
    values[:, index] = np.nan
    available[:, index] = 0.0
    return np.hstack([block, values, available]).astype(np.float64, copy=False)


def preprocess(values: np.ndarray, train_indices: np.ndarray, prediction_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(values, dtype=np.float64).copy()
    clean[(~np.isfinite(clean)) | (np.abs(clean) > 1.0e12)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(clean[train_indices]), imputer.transform(clean[prediction_indices])


def specialist_predictions(features: np.ndarray, y_global: np.ndarray, target_indices: np.ndarray, train_rows: np.ndarray, validation_rows: np.ndarray) -> np.ndarray:
    train_indices = target_indices[train_rows]
    validation_indices = target_indices[validation_rows]
    x_train, x_validation = preprocess(features, train_indices, validation_indices)
    scaled_train = StandardScaler().fit_transform(x_train)
    scaler = StandardScaler().fit(x_train)
    ridge = Ridge(alpha=30.0).fit(scaler.transform(x_train), y_global[train_indices])
    ridge_prediction = ridge.predict(scaler.transform(x_validation))
    trees = ExtraTreesRegressor(n_estimators=192, min_samples_leaf=3, max_features=0.7, random_state=2026, n_jobs=2).fit(x_train, y_global[train_indices])
    tree_prediction = trees.predict(x_validation)
    boosted = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.03, loss="huber", random_state=2026).fit(x_train, y_global[train_indices])
    boosted_prediction = boosted.predict(x_validation)
    y_train = y_global[train_indices]
    return np.column_stack([reference.clip_prediction(y_train, ridge_prediction), reference.clip_prediction(y_train, tree_prediction), reference.clip_prediction(y_train, boosted_prediction)])


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    output = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in train]
        for index in validation:
            output[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    return output


def bootstrap_lower(y: np.ndarray, specialist: np.ndarray, baseline: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        sample = rng.integers(0, len(y), size=len(y))
        if np.isclose(np.var(y[sample]), 0.0):
            continue
        values.append(float(r2_score(y[sample], specialist[sample]) - r2_score(y[sample], baseline[sample])))
    return float(np.quantile(values, 0.025))


def add_metric(rows: list[dict[str, Any]], target: str, method: str, stratum: str, fold: int | None, y: np.ndarray, prediction: np.ndarray) -> None:
    item = {"target": target, "method": method, "stratum": stratum, "fold": fold}
    item.update(metric(y, prediction))
    rows.append(item)


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
    if run_dir.exists():
        existing = {path.name for path in run_dir.iterdir()}
        if existing - {"protocol.json"}:
            raise RuntimeError(f"Refusing to reuse non-empty run directory: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    block, block_names = electronic_block(descriptor, descriptor_names, physical, physical_names)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    key_to_index = {key: index for index, key in enumerate(keys)}
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for target in SPECIAL_TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_indices = np.asarray([key_to_index[value] for value in target_rows["canonical"]], dtype=np.int64)
        y = target_rows["target"].to_numpy(float)
        folds = folds_for(target_rows)
        nearest = nearest_similarity([fingerprints[index] for index in target_indices], folds)
        availability_count = np.sum(np.delete(cross_available[target_indices], TARGETS.index(target), axis=1), axis=1).astype(int)
        dense = reference.target_dense_features(np.hstack([descriptor, physical]), cross_values, cross_available, target)
        specialist_features_matrix = features_for_target(block, cross_values, cross_available, target)
        baseline_parts = np.full((len(y), 4), np.nan, dtype=np.float64)
        specialist_parts = np.full((len(y), 3), np.nan, dtype=np.float64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[target_indices] = y
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            baseline_parts[validation_rows] = reference.predict_base_models(dense, sparse_parts, fingerprints, y_global, target_indices[train_rows], target_indices[validation_rows], reference.DEFAULT_CONFIG, target)
            specialist_parts[validation_rows] = specialist_predictions(specialist_features_matrix, y_global, target_indices, train_rows, validation_rows)
            fold_rows.extend({"target": target, "fold": fold, "canonical": target_rows.iloc[index]["canonical"], "availability_count": int(availability_count[index]), "nearest_similarity": float(nearest[index])} for index in validation_rows)
        baseline = baseline_parts @ load_weights(root, target)
        report: dict[str, Any] = {"rows": int(len(y)), "baseline": metric(y, baseline), "models": {}, "folds": [], "availability": {}, "low_similarity": {}}
        for index, name in enumerate(MODEL_NAMES):
            report["models"][name] = metric(y, specialist_parts[:, index])
        for fold in range(5):
            selected = folds == fold
            fold_report = {"fold": fold, "baseline": metric(y[selected], baseline[selected])}
            for index, name in enumerate(MODEL_NAMES):
                fold_report[name] = metric(y[selected], specialist_parts[selected, index])
                fold_report[f"{name}_delta_r2"] = None if fold_report[name]["r2"] is None else float(fold_report[name]["r2"] - fold_report["baseline"]["r2"])
            report["folds"].append(fold_report)
        for count in sorted(np.unique(availability_count)):
            selected = availability_count == count
            report["availability"][f"aux_count_{count}"] = {"rows": int(np.sum(selected)), "baseline": metric(y[selected], baseline[selected]), **{name: metric(y[selected], specialist_parts[selected, index]) for index, name in enumerate(MODEL_NAMES)}}
            for name in MODEL_NAMES:
                report["availability"][f"aux_count_{count}"][f"{name}_delta_r2"] = (None if report["availability"][f"aux_count_{count}"][name]["r2"] is None else float(report["availability"][f"aux_count_{count}"][name]["r2"] - report["availability"][f"aux_count_{count}"]["baseline"]["r2"]))
        bins = (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01))
        for name_bin, lower, upper in bins:
            selected = (nearest >= lower) & (nearest < upper)
            report["low_similarity"][name_bin] = {"rows": int(np.sum(selected)), "baseline": metric(y[selected], baseline[selected]), **{name: metric(y[selected], specialist_parts[selected, index]) for index, name in enumerate(MODEL_NAMES)}}
            for name in MODEL_NAMES:
                report["low_similarity"][name_bin][f"{name}_delta_r2"] = (None if report["low_similarity"][name_bin][name]["r2"] is None else float(report["low_similarity"][name_bin][name]["r2"] - report["low_similarity"][name_bin]["baseline"]["r2"]))
        best_name = max(MODEL_NAMES, key=lambda name: report["models"][name]["r2"])
        best_index = MODEL_NAMES.index(best_name)
        deltas = [report["folds"][fold][f"{best_name}_delta_r2"] for fold in range(5)]
        slice_deltas = [value[f"{best_name}_delta_r2"] for value in report["low_similarity"].values() if value[f"{best_name}_delta_r2"] is not None and value["rows"] >= 5]
        count_zero = report["availability"].get("aux_count_0")
        missing_delta = None if count_zero is None else count_zero[f"{best_name}_delta_r2"]
        report["selected_model"] = best_name
        report["selected_delta_r2"] = float(report["models"][best_name]["r2"] - report["baseline"]["r2"])
        report["selected_positive_folds"] = int(sum(value > 0 for value in deltas))
        report["selected_bootstrap_lower_bound"] = bootstrap_lower(y, specialist_parts[:, best_index], baseline, 2026 + TARGETS.index(target))
        report["selected_missing_auxiliary_delta"] = missing_delta
        report["selected_min_low_similarity_delta"] = min(slice_deltas) if slice_deltas else None
        reports[target] = report
        for name, prediction in [("frozen_c001_blend", baseline), *[(name, specialist_parts[:, index]) for index, name in enumerate(MODEL_NAMES)]]:
            add_metric(rows, target, name, "all", None, y, prediction)
            for fold in range(5):
                selected = folds == fold
                add_metric(rows, target, name, "all", fold, y[selected], prediction[selected])
            for count in sorted(np.unique(availability_count)):
                selected = availability_count == count
                add_metric(rows, target, name, f"aux_count_{count}", None, y[selected], prediction[selected])
            for name_bin, lower, upper in bins:
                selected = (nearest >= lower) & (nearest < upper)
                add_metric(rows, target, name, name_bin, None, y[selected], prediction[selected])
        for index, row in target_rows.iterrows():
            prediction_rows.append({"canonical": row["canonical"], "target_type": target, "target": float(y[index]), "fold": int(folds[index]), "availability_count": int(availability_count[index]), "nearest_similarity": float(nearest[index]), "baseline": float(baseline[index]), **{name: float(specialist_parts[index, column]) for column, name in enumerate(MODEL_NAMES)}})
    passing_targets = [target for target in SPECIAL_TARGETS if reports[target]["selected_delta_r2"] >= 0.01 and reports[target]["selected_positive_folds"] >= 4 and reports[target]["selected_bootstrap_lower_bound"] > 0.0 and reports[target]["selected_missing_auxiliary_delta"] is not None and reports[target]["selected_missing_auxiliary_delta"] >= 0.0 and reports[target]["selected_min_low_similarity_delta"] is not None and reports[target]["selected_min_low_similarity_delta"] >= 0.0]
    audit = {"schema_version": "ppp.round2.ei-eea-specialist-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C003-20260803-1722-eps-nc-specialist-repaired", "official_inputs": inputs, "official_hashes_pass": all(inputs[name]["sha256"] == expected for name, expected in reference.EXPECTED_HASHES.items()), "feature_names": block_names + [f"cross_{target}" for target in TARGETS] + [f"available_{target}" for target in TARGETS], "targets": reports, "passing_targets": passing_targets, "decision": "component_pass" if passing_targets else "rejected_component_gate", "elapsed_seconds": float(time.time() - started)}
    pd.DataFrame(rows).to_csv(run_dir / "panel_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_assignments.csv", index=False)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.ei-eea-specialist.v1", "seed": 2026, "folds": 5, "models": list(MODEL_NAMES), "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", audit)
    (run_dir / "decision.md").write_text(f"# R2-C004 Ei/Eea specialist decision\n\nDecision: **{audit['decision']}**.\n\nNo candidate changed; the three specialist arms were compared on fixed canonical-group folds.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "panel_metrics.csv", "predictions.csv", "fold_assignments.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "passing_targets": passing_targets, "summary": {target: {"baseline": reports[target]["baseline"]["r2"], "selected_model": reports[target]["selected_model"], "selected_r2": reports[target]["models"][reports[target]["selected_model"]]["r2"], "delta": reports[target]["selected_delta_r2"]} for target in SPECIAL_TARGETS}, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
