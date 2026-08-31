#!/usr/bin/env python3
"""Evaluate the frozen Round 1 portable Tg carrier on Round 2 official data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "tg"
MODEL_NAMES = ("portable_sparse_ridge", "portable_dense_ridge", "portable_extra_trees", "portable_tanimoto_local")
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


def load_c001_weights(root: Path) -> np.ndarray:
    report = json.loads((root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "report.json").read_text(encoding="utf-8"))
    names = ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")
    values = report["validation"]["target_reports"][TARGET]["blend_weights"]
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def bit_matrix(molecules: list[Any], radius: int, bits: int) -> sparse.csr_matrix:
    generator = reference.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, molecule in enumerate(molecules):
        for column in generator.GetFingerprint(molecule).GetOnBits():
            rows.append(row)
            columns.append(int(column))
            values.append(1.0)
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(molecules), bits), dtype=np.float64)


def portable_features(molecules: list[Any], keys: list[str]) -> tuple[list[sparse.csr_matrix], np.ndarray, list[Any]]:
    sparse_parts = []
    for radius in (1, 2, 3):
        sparse_parts.append(reference.morgan_count_matrix(molecules, radius, 4096))
    for radius in (1, 2, 3):
        sparse_parts.append(bit_matrix(molecules, radius, 4096))
    sparse_parts.append(reference.text_matrix(keys, 65536))
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    return sparse_parts, np.hstack([descriptor, physical]), fingerprints


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError("Incomplete fold assignment")
    return result


def scaffold_group(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        value = ""
    return value or "ACYCLIC"


def portable_predictions(
    dense: np.ndarray,
    sparse_parts: list[sparse.csr_matrix],
    fingerprints: list[Any],
    y_global: np.ndarray,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
) -> np.ndarray:
    clean = np.asarray(dense, dtype=np.float64).copy()
    clean[(~np.isfinite(clean)) | (np.abs(clean) > 1.0e12)] = np.nan
    imputer = reference.SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(imputer.fit_transform(clean[train_indices]))
    prediction_dense = scaler.transform(imputer.transform(clean[prediction_indices]))
    train_sparse = sparse.hstack([part[train_indices] for part in sparse_parts] + [sparse.csr_matrix(train_dense)], format="csr")
    prediction_sparse = sparse.hstack([part[prediction_indices] for part in sparse_parts] + [sparse.csr_matrix(prediction_dense)], format="csr")
    y_train = y_global[train_indices]
    ridge_sparse = Ridge(alpha=10.0, solver="lsqr", max_iter=5000, tol=1e-4).fit(train_sparse, y_train)
    ridge_dense = Ridge(alpha=10.0).fit(train_dense, y_train)
    tree = ExtraTreesRegressor(n_estimators=192, min_samples_leaf=2, max_features=0.75, random_state=2026, n_jobs=2).fit(imputer.transform(clean[train_indices]), y_train)
    local = reference.tanimoto_prediction(fingerprints, y_global, train_indices, prediction_indices, k=15, krr_alpha=0.05)
    output = np.column_stack([ridge_sparse.predict(prediction_sparse), ridge_dense.predict(prediction_dense), tree.predict(imputer.transform(clean[prediction_indices])), local])
    for column in range(output.shape[1]):
        output[:, column] = reference.clip_prediction(y_train, output[:, column])
    return output


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    output = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        reference_fps = [fingerprints[index] for index in training]
        for index in validation:
            output[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], reference_fps))
    return output


def bootstrap_lower(y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> float:
    rng = np.random.default_rng(2026)
    values: list[float] = []
    for _ in range(2000):
        sample = rng.integers(0, len(y), size=len(y))
        if np.isclose(np.var(y[sample]), 0.0):
            continue
        values.append(float(r2_score(y[sample], candidate[sample]) - r2_score(y[sample], baseline[sample])))
    return float(np.quantile(values, 0.025)) if values else float("nan")


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
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    target_indices = np.asarray([key_to_index[value] for value in target_rows["canonical"]], dtype=np.int64)
    y = target_rows["target"].to_numpy(float)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[target_indices] = y
    portable_sparse, portable_dense, portable_fingerprints = portable_features(molecules, keys)
    c001_sparse = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    c001_dense_base = np.hstack([reference.descriptor_matrix(molecules)[0], reference.physical_matrix(molecules, keys)[0]])
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    c001_dense = reference.target_dense_features(c001_dense_base, cross_values, cross_available, TARGET)
    c001_fingerprints = reference.morgan_bits(molecules, 2, 4096)
    weights = load_c001_weights(root)
    reports: dict[str, Any] = {}
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_folds: list[dict[str, Any]] = []
    panel_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for panel, groups in (("canonical_group", target_rows["canonical"].to_numpy(object)), ("scaffold_family", target_rows["smiles"].map(scaffold_group).to_numpy(object))):
        folds = folds_for(groups)
        baseline = np.full(len(y), np.nan, dtype=np.float64)
        candidate = np.full((len(y), len(MODEL_NAMES)), np.nan, dtype=np.float64)
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            base_parts = reference.predict_base_models(c001_dense, c001_sparse, c001_fingerprints, y_global, target_indices[train_rows], target_indices[validation_rows], reference.DEFAULT_CONFIG, TARGET)
            baseline[validation_rows] = base_parts @ weights
            candidate[validation_rows] = portable_predictions(portable_dense, portable_sparse, portable_fingerprints, y_global, target_indices[train_rows], target_indices[validation_rows])
            all_folds.extend({"panel": panel, "fold": fold, "canonical": target_rows.iloc[index]["canonical"], "scaffold": groups[index]} for index in validation_rows)
        panel_predictions[panel] = (baseline, candidate)
        panel_report: dict[str, Any] = {"baseline": metric(y, baseline), "models": {name: metric(y, candidate[:, index]) for index, name in enumerate(MODEL_NAMES)}, "folds": []}
        for fold in range(5):
            selected = folds == fold
            row = {"fold": fold, "baseline": metric(y[selected], baseline[selected])}
            for index, name in enumerate(MODEL_NAMES):
                row[name] = metric(y[selected], candidate[selected, index])
                row[f"{name}_delta_r2"] = float(row[name]["r2"] - row["baseline"]["r2"])
            panel_report["folds"].append(row)
        if panel == "canonical_group":
            nearest = nearest_similarity(portable_fingerprints, folds)
            panel_report["low_similarity"] = {}
            for name_bin, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
                selected = (nearest[target_indices] >= lower) & (nearest[target_indices] < upper)
                panel_report["low_similarity"][name_bin] = {"rows": int(np.sum(selected)), "baseline": metric(y[selected], baseline[selected]), **{name: metric(y[selected], candidate[selected, index]) for index, name in enumerate(MODEL_NAMES)}}
                for name in MODEL_NAMES:
                    base_r2 = panel_report["low_similarity"][name_bin]["baseline"]["r2"]
                    model_r2 = panel_report["low_similarity"][name_bin][name]["r2"]
                    panel_report["low_similarity"][name_bin][f"{name}_delta_r2"] = None if base_r2 is None or model_r2 is None else float(model_r2 - base_r2)
        reports[panel] = panel_report
        for index, name in enumerate(("frozen_c001_blend", *MODEL_NAMES)):
            prediction = baseline if index == 0 else candidate[:, index - 1]
            all_metrics.append({"panel": panel, "method": name, **metric(y, prediction)})
        for index, row in target_rows.iterrows():
            all_predictions.append({"panel": panel, "canonical": row["canonical"], "target": float(y[index]), "fold": int(folds[index]), "baseline": float(baseline[index]), **{name: float(candidate[index, column]) for column, name in enumerate(MODEL_NAMES)}})
    canonical_baseline, canonical_candidate = panel_predictions["canonical_group"]
    selected_name = max(MODEL_NAMES, key=lambda name: reports["canonical_group"]["models"][name]["r2"])
    selected_index = MODEL_NAMES.index(selected_name)
    fold_deltas = [float(reports["canonical_group"]["folds"][fold][f"{selected_name}_delta_r2"]) for fold in range(5)]
    scaffold_delta = float(reports["scaffold_family"]["models"][selected_name]["r2"] - reports["scaffold_family"]["baseline"]["r2"])
    low_deltas = [value[f"{selected_name}_delta_r2"] for value in reports["canonical_group"]["low_similarity"].values() if value["rows"] >= 5 and value[f"{selected_name}_delta_r2"] is not None]
    audit = {
        "schema_version": "ppp.round2.tg-portable-carrier-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C005-20260803-1722-egc-egb-coupled-repaired",
        "official_inputs": inputs,
        "official_hashes_pass": all(inputs[name]["sha256"] == expected for name, expected in reference.EXPECTED_HASHES.items()),
        "models": list(MODEL_NAMES),
        "target": TARGET,
        "panels": reports,
        "selected_model": selected_name,
        "selected_delta_r2": float(reports["canonical_group"]["models"][selected_name]["r2"] - reports["canonical_group"]["baseline"]["r2"]),
        "selected_positive_folds": int(sum(value > 0 for value in fold_deltas)),
        "selected_bootstrap_lower_bound": bootstrap_lower(y, canonical_candidate[:, selected_index], canonical_baseline),
        "scaffold_family_delta": scaffold_delta,
        "selected_min_low_similarity_delta": min(low_deltas) if low_deltas else None,
        "passing_component": bool(
            reports["canonical_group"]["models"][selected_name]["r2"] - reports["canonical_group"]["baseline"]["r2"] >= 0.01
            and sum(value > 0 for value in fold_deltas) >= 4
            and bootstrap_lower(y, canonical_candidate[:, selected_index], canonical_baseline) > 0.0
            and scaffold_delta >= 0.0
            and low_deltas
            and min(low_deltas) >= 0.0
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    audit["decision"] = "component_pass" if audit["passing_component"] else "rejected_component_gate"
    pd.DataFrame(all_metrics).to_csv(run_dir / "panel_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(all_folds).to_csv(run_dir / "fold_assignments.csv", index=False)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.tg-portable-carrier.v1", "seed": 2026, "folds": 5, "models": list(MODEL_NAMES), "official_inputs": inputs, "feature_views": ["rdkit_descriptors", "physical_counts", "morgan_count_r1_r2_r3", "morgan_bits_r1_r2_r3", "smiles_char_ngrams_2_7", "tanimoto_local"]})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", audit)
    (run_dir / "decision.md").write_text(f"# R2-C006 portable Tg carrier decision\n\nDecision: **{audit['decision']}**.\n\nThe expanded official current/archive label pool was evaluated without importing Round 1 weights, predictions, caches, or local_eval artifacts.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "panel_metrics.csv", "predictions.csv", "fold_assignments.csv", "metrics.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "selected_model": selected_name, "selected_delta_r2": audit["selected_delta_r2"], "selected_positive_folds": audit["selected_positive_folds"], "bootstrap_lower": audit["selected_bootstrap_lower_bound"], "scaffold_delta": scaffold_delta, "min_low_similarity_delta": audit["selected_min_low_similarity_delta"], "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
