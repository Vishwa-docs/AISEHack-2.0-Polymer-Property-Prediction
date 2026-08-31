#!/usr/bin/env python3
"""Evaluate the frozen Round 2 EPS/Nc paired-property specialist."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
SPECIAL_TARGETS = ("eps", "nc")
MODEL_NAMES = ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    return {
        "rows": int(len(y)),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 and not np.isclose(np.var(y), 0.0) else None,
        "mae": float(mean_absolute_error(y, prediction)) if len(y) else None,
        "rmse": float(np.sqrt(np.mean(np.square(y - prediction)))) if len(y) else None,
    }


def make_folds(rows: pd.DataFrame, folds: int) -> np.ndarray:
    assignment = np.full(len(rows), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=folds)
    groups = rows["canonical"].to_numpy(object)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(rows)), groups=groups)):
        assignment[validation] = fold
    return assignment


def load_blend_weights(root: Path) -> dict[str, np.ndarray]:
    report_path = root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result: dict[str, np.ndarray] = {}
    for target in SPECIAL_TARGETS:
        values = report["validation"]["target_reports"][target]["blend_weights"]
        result[target] = np.asarray([float(values[name]) for name in MODEL_NAMES], dtype=np.float64)
    return result


def selected_descriptor_matrix(
    descriptor: np.ndarray,
    descriptor_names: list[str],
    physical: np.ndarray,
    physical_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    descriptor_keep = [
        "MolWt", "HeavyAtomMolWt", "ExactMolWt", "LabuteASA", "TPSA",
        "FractionCSP3", "HeavyAtomCount", "NumAliphaticRings", "NumAromaticRings",
        "NumRotatableBonds", "NumSaturatedRings", "RingCount", "MolLogP", "MolMR",
    ]
    physical_keep = ["smiles_length", "atom_count", "heavy_atom_count", "ring_count", "hetero_atom_count", "rotatable_bonds_approx"]
    descriptor_index = {name: index for index, name in enumerate(descriptor_names)}
    physical_index = {name: index for index, name in enumerate(physical_names)}
    missing = [name for name in (*descriptor_keep, *physical_keep) if name not in descriptor_index and name not in physical_index]
    if missing:
        raise RuntimeError(f"Specialist descriptors missing: {missing}")
    columns = [descriptor[:, descriptor_index[name]] for name in descriptor_keep]
    columns.extend(physical[:, physical_index[name]] for name in physical_keep)
    mass = descriptor[:, descriptor_index["MolWt"]]
    asa = descriptor[:, descriptor_index["LabuteASA"]]
    mr = descriptor[:, descriptor_index["MolMR"]]
    tpsa = descriptor[:, descriptor_index["TPSA"]]
    heavy = physical[:, physical_index["heavy_atom_count"]]
    length = physical[:, physical_index["smiles_length"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        columns.extend([mass / asa, mr / mass, tpsa / mass, heavy / length])
    names = [f"descriptor_{name}" for name in descriptor_keep]
    names.extend(f"physical_{name}" for name in physical_keep)
    names.extend(["mass_per_labute_asa", "mr_per_mass", "tpsa_per_mass", "heavy_atoms_per_smiles_length"])
    return np.column_stack(columns).astype(np.float64, copy=False), names


def specialist_features(
    base: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    target: str,
    paired_permutation: np.ndarray | None,
) -> np.ndarray:
    paired = "nc" if target == "eps" else "eps"
    paired_index = TARGETS.index(paired)
    positions = np.arange(len(base)) if paired_permutation is None else paired_permutation
    paired_values = cross_values[positions, paired_index]
    paired_available = cross_available[positions, paired_index]
    return np.column_stack([base, paired_values, paired_available]).astype(np.float64, copy=False)


def fit_specialist(
    features: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64).copy()
    invalid = ~np.isfinite(values) | (np.abs(values) > 1.0e12)
    values[invalid] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(values[train_indices]))
    x_prediction = scaler.transform(imputer.transform(values[prediction_indices]))
    model = Ridge(alpha=alpha)
    model.fit(x_train, y[train_indices])
    return model.predict(x_prediction)


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    result = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(int(np.max(folds)) + 1):
        validation = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_fingerprints = [fingerprints[index] for index in train]
        for index in validation:
            result[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fingerprints))
    return result


def bootstrap_lower_bound(y: np.ndarray, specialist: np.ndarray, baseline: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(2000):
        sample = rng.integers(0, len(y), size=len(y))
        if np.isclose(np.var(y[sample]), 0.0):
            continue
        values.append(float(r2_score(y[sample], specialist[sample]) - r2_score(y[sample], baseline[sample])))
    quantiles = np.quantile(values, [0.025, 0.5, 0.975])
    return float(quantiles[0]), float(quantiles[1]), float(quantiles[2])


def append_row(rows: list[dict[str, Any]], target: str, method: str, stratum: str, fold: int | None, y: np.ndarray, prediction: np.ndarray) -> None:
    item = {"target": target, "method": method, "stratum": stratum, "fold": fold}
    item.update(score(y, prediction))
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
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    specialist_base, specialist_names = selected_descriptor_matrix(descriptor, descriptor_names, physical, physical_names)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=4096),
        reference.morgan_count_matrix(molecules, radius=3, bits=4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
    key_to_index = {key: index for index, key in enumerate(keys)}
    weights = load_blend_weights(root)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "specialist_alpha": 30.0})
    rng_permutation = np.random.default_rng(2026).permutation(len(keys))
    metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    target_reports: dict[str, Any] = {}

    for target in SPECIAL_TARGETS:
        target_rows = pooled[pooled["target_type"] == target].reset_index(drop=True)
        target_indices = np.asarray([key_to_index[value] for value in target_rows["canonical"]], dtype=np.int64)
        y = target_rows["target"].to_numpy(float)
        folds = make_folds(target_rows, 5)
        fps = [fingerprints[index] for index in target_indices]
        nearest = nearest_similarity(fps, folds)
        paired = "nc" if target == "eps" else "eps"
        availability = cross_available[target_indices, TARGETS.index(paired)].astype(int)
        dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
        pair_features = specialist_features(specialist_base, cross_values, cross_available, target, None)
        negative_features = specialist_features(specialist_base, cross_values, cross_available, target, rng_permutation)
        baseline_parts = np.full((len(target_rows), len(MODEL_NAMES)), np.nan, dtype=np.float64)
        specialist_prediction = np.full(len(target_rows), np.nan, dtype=np.float64)
        negative_prediction = np.full(len(target_rows), np.nan, dtype=np.float64)
        for fold in range(5):
            local_train = np.flatnonzero(folds != fold)
            validation = np.flatnonzero(folds == fold)
            y_global = np.full(len(keys), np.nan, dtype=np.float64)
            y_global[target_indices] = y
            base_prediction = reference.predict_base_models(
                dense, sparse_parts, fingerprints, y_global,
                target_indices[local_train], target_indices[validation], config, target,
            )
            baseline_parts[validation] = base_prediction
            specialist_prediction[validation] = fit_specialist(pair_features, y_global, target_indices[local_train], target_indices[validation], 30.0)
            negative_prediction[validation] = fit_specialist(negative_features, y_global, target_indices[local_train], target_indices[validation], 30.0)
            fold_rows.extend({"target": target, "fold": fold, "canonical": target_rows.iloc[index]["canonical"], "availability": int(availability[index]), "nearest_similarity": float(nearest[index])} for index in validation)
        baseline_prediction = baseline_parts @ weights[target]
        target_report: dict[str, Any] = {
            "paired_property": paired,
            "rows": int(len(y)),
            "baseline": score(y, baseline_prediction),
            "specialist": score(y, specialist_prediction),
            "negative_control": score(y, negative_prediction),
            "folds": [],
            "availability": {},
            "low_similarity": {},
        }
        for fold in range(5):
            selected = folds == fold
            base_fold = score(y[selected], baseline_prediction[selected])
            specialist_fold = score(y[selected], specialist_prediction[selected])
            negative_fold = score(y[selected], negative_prediction[selected])
            target_report["folds"].append({"fold": fold, "baseline": base_fold, "specialist": specialist_fold, "negative_control": negative_fold, "delta_r2": None if specialist_fold["r2"] is None else float(specialist_fold["r2"] - base_fold["r2"])})
        for availability_value in (0, 1):
            selected = availability == availability_value
            target_report["availability"][f"paired_available_{availability_value}"] = {
                "rows": int(np.sum(selected)),
                "baseline": score(y[selected], baseline_prediction[selected]),
                "specialist": score(y[selected], specialist_prediction[selected]),
                "delta_r2": float(r2_score(y[selected], specialist_prediction[selected]) - r2_score(y[selected], baseline_prediction[selected])) if np.sum(selected) > 1 and not np.isclose(np.var(y[selected]), 0.0) else None,
            }
        bins = (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01))
        for name, lower, upper in bins:
            selected = (nearest >= lower) & (nearest < upper)
            target_report["low_similarity"][name] = {
                "rows": int(np.sum(selected)),
                "baseline": score(y[selected], baseline_prediction[selected]),
                "specialist": score(y[selected], specialist_prediction[selected]),
                "delta_r2": float(r2_score(y[selected], specialist_prediction[selected]) - r2_score(y[selected], baseline_prediction[selected])) if np.sum(selected) > 1 and not np.isclose(np.var(y[selected]), 0.0) else None,
            }
        lower, median, upper = bootstrap_lower_bound(y, specialist_prediction, baseline_prediction, 2026 + TARGETS.index(target))
        target_report["group_bootstrap_delta_r2"] = {"lower_2_5": lower, "median": median, "upper_97_5": upper, "replicates": 2000}
        for method, prediction in (("frozen_c001_blend", baseline_prediction), ("paired_specialist", specialist_prediction), ("shuffled_pair_negative_control", negative_prediction)):
            append_row(metrics, target, method, "all", None, y, prediction)
            for fold in range(5):
                selected = folds == fold
                append_row(metrics, target, method, "all", fold, y[selected], prediction[selected])
            for availability_value in (0, 1):
                selected = availability == availability_value
                append_row(metrics, target, method, f"paired_available_{availability_value}", None, y[selected], prediction[selected])
            for name, lower_bound, upper_bound in bins:
                selected = (nearest >= lower_bound) & (nearest < upper_bound)
                append_row(metrics, target, method, name, None, y[selected], prediction[selected])
        for index, row in target_rows.iterrows():
            prediction_rows.append({"canonical": row["canonical"], "target_type": target, "target": float(y[index]), "fold": int(folds[index]), "paired_available": int(availability[index]), "nearest_similarity": float(nearest[index]), "baseline": float(baseline_prediction[index]), "specialist": float(specialist_prediction[index]), "negative_control": float(negative_prediction[index])})
        target_reports[target] = target_report

    eps = target_reports["eps"]
    nc = target_reports["nc"]
    eps_fold_deltas = [row["delta_r2"] for row in eps["folds"] if row["delta_r2"] is not None]
    eps_missing_delta = eps["availability"]["paired_available_0"]["delta_r2"]
    nc_missing_delta = nc["availability"]["paired_available_0"]["delta_r2"]
    low_deltas = [value["delta_r2"] for report in (eps, nc) for value in report["low_similarity"].values() if value["delta_r2"] is not None]
    pass_gate = bool(
        eps["specialist"]["r2"] - eps["baseline"]["r2"] >= 0.01
        and sum(value > 0 for value in eps_fold_deltas) >= 4
        and eps["group_bootstrap_delta_r2"]["lower_2_5"] > 0.0
        and nc["specialist"]["r2"] - nc["baseline"]["r2"] >= -0.003
        and eps_missing_delta is not None and eps_missing_delta >= 0.0
        and nc_missing_delta is not None and nc_missing_delta >= 0.0
        and len(low_deltas) > 0 and min(low_deltas) >= 0.0
    )
    audit = {
        "schema_version": "ppp.round2.eps-nc-specialist-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C002-20260803-1708-validation-hardening",
        "official_inputs": inputs,
        "official_hashes_pass": all(inputs[name]["sha256"] == expected for name, expected in reference.EXPECTED_HASHES.items()),
        "feature_names": specialist_names + ["paired_official_value", "paired_official_available"],
        "specialist_alpha": 30.0,
        "targets": target_reports,
        "decision": "component_pass" if pass_gate else "rejected_component_gate",
        "gate_evidence": {
            "eps_grouped_gain": float(eps["specialist"]["r2"] - eps["baseline"]["r2"]),
            "eps_positive_folds": int(sum(value > 0 for value in eps_fold_deltas)),
            "eps_group_bootstrap_lower_bound": float(eps["group_bootstrap_delta_r2"]["lower_2_5"]),
            "nc_delta": float(nc["specialist"]["r2"] - nc["baseline"]["r2"]),
            "eps_missing_pair_delta": eps_missing_delta,
            "nc_missing_pair_delta": nc_missing_delta,
            "minimum_low_similarity_delta": min(low_deltas) if low_deltas else None,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame(metrics).to_csv(run_dir / "panel_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_assignments.csv", index=False)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.eps-nc-specialist.v1", "seed": 2026, "folds": 5, "alpha": 30.0, "parent": C001_ID, "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", audit)
    (run_dir / "decision.md").write_text(
        "# R2-C003 EPS/Nc specialist decision\n\n"
        f"Decision: **{audit['decision']}**.\n\n"
        "The comparison used the preregistered paired-property Ridge specialist and fixed shuffled-pair negative control. "
        "No candidate or post-hoc external_label source was used.\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "panel_metrics.csv", "predictions.csv", "fold_assignments.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "gate_evidence": audit["gate_evidence"], "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))
if __name__ == "__main__":
    main()
