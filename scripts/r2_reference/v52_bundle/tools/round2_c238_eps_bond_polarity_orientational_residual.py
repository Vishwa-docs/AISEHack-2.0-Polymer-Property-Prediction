#!/usr/bin/env python3
"""C238: EPS bond-polarity/orientational residual over regenerated C214.

This is a queue-safety continuation after C237.  It targets only EPS, whose
current selected clean component is C214.  The selected C214 EPS parent is
regenerated from official inputs and source code inside this runner; no C214
prediction file is read.  The only new factor is an official-SMILES graph block
of formal-charge, hetero-bond polarity, donor/acceptor graph-distance, and
wildcard-backbone/pedant concentration descriptors.

No external group constants, bond-dipole tables, local_eval/public feedback, PI1M,
pretrained assets, Kaggle action, or stored prediction replay are used.
"""

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
from rdkit import Chem, rdBase
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c187_ionic_eps_only as c187
import round2_c208_tg_robust_group_measurement as c208


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "eps"
SCHEMA = "ppp.round2.c238.eps-bond-polarity-orientational-residual.v1"
SEED = 20260805
C214_RUN_ID = "R2-C214-20260805-0440-eps-ionic-full-amplitude-v1"
C214_EXPECTED_EPS_R2 = 0.8500949465048359
C214_MAX_ABS_ERROR = 1e-10
MIN_SELECTED_REFERENCE_DELTA_R2 = 0.010
RIDGE_ALPHA = 80.0
TREE_COUNT = 220
TREE_MIN_LEAF = 3
RESIDUAL_WEIGHT = 0.30
RIDGE_BLEND_WEIGHT = 0.70
TREE_BLEND_WEIGHT = 0.30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": stage, "time": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), **payload},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def stats(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.sum(array)),
        float(np.mean(array)),
        float(np.std(array)),
        float(np.min(array)),
        float(np.max(array)),
        float(np.sum(np.abs(array))),
    ]


def partition_masks(molecule: Chem.Mol) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    atoms = list(molecule.GetAtoms())
    n_atoms = len(atoms)
    real = np.asarray([atom.GetAtomicNum() > 0 for atom in atoms], dtype=bool)
    dummy_indices = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 0]
    backbone = np.zeros(n_atoms, dtype=bool)
    path_length = 0
    if len(dummy_indices) >= 2:
        try:
            path = Chem.GetShortestPath(molecule, int(dummy_indices[0]), int(dummy_indices[1]))
        except Exception:
            path = tuple()
        path_length = max(len(path) - 1, 0)
        for idx in path:
            if idx < n_atoms and real[idx]:
                backbone[idx] = True
    if not np.any(backbone & real):
        backbone = real.copy()
    pendant = real & ~backbone
    bridge_bonds = 0
    for bond in molecule.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if (backbone[a] and pendant[b]) or (backbone[b] and pendant[a]):
            bridge_bonds += 1
    return backbone, pendant, {
        "dummy_atom_count": int(len(dummy_indices)),
        "wildcard_path_length": int(path_length),
        "real_atom_count": int(np.sum(real)),
        "backbone_atom_count": int(np.sum(backbone)),
        "pendant_atom_count": int(np.sum(pendant)),
        "bridge_bond_count": int(bridge_bonds),
    }


def bond_polarity_row(molecule: Chem.Mol) -> tuple[list[float], dict[str, Any]]:
    atoms = list(molecule.GetAtoms())
    n_atoms = len(atoms)
    if n_atoms == 0:
        return [0.0] * 118, {"empty_molecule": True}

    backbone, pendant, meta = partition_masks(molecule)
    real = np.asarray([atom.GetAtomicNum() > 0 for atom in atoms], dtype=bool)
    heavy_count = max(int(np.sum(real)), 1)
    atomic = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=np.float64)
    valence = np.asarray([atom.GetTotalValence() for atom in atoms], dtype=np.float64)
    charge = np.asarray([atom.GetFormalCharge() for atom in atoms], dtype=np.float64)
    degree = np.asarray([atom.GetDegree() for atom in atoms], dtype=np.float64)
    aromatic = np.asarray([atom.GetIsAromatic() for atom in atoms], dtype=np.float64)
    hetero = np.asarray([atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms], dtype=np.float64)
    polar = np.asarray([atom.GetAtomicNum() in (7, 8, 9, 15, 16, 17, 35, 53) for atom in atoms], dtype=np.float64)
    halogen = np.asarray([atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms], dtype=np.float64)
    donor = np.asarray([atom.GetAtomicNum() in (7, 8, 15, 16) and atom.GetFormalCharge() >= 0 for atom in atoms], dtype=bool)
    acceptor = np.asarray([atom.GetAtomicNum() in (7, 8, 9, 15, 16, 17, 35, 53) and atom.GetFormalCharge() <= 0 for atom in atoms], dtype=bool)

    row: list[float] = [
        float(heavy_count),
        float(np.sum(hetero[real]) / heavy_count),
        float(np.sum(polar[real]) / heavy_count),
        float(np.sum(halogen[real]) / heavy_count),
        float(np.sum(donor & real) / heavy_count),
        float(np.sum(acceptor & real) / heavy_count),
        float(np.sum(charge[real] > 0)),
        float(np.sum(charge[real] < 0)),
        float(np.sum(np.abs(charge[real]))),
        float(np.any(charge[real] > 0) and np.any(charge[real] < 0)),
        float(np.sum(aromatic[real]) / heavy_count),
        float(meta["dummy_atom_count"]),
        float(meta["wildcard_path_length"]),
        float(meta["backbone_atom_count"] / heavy_count),
        float(meta["pendant_atom_count"] / heavy_count),
        float(meta["bridge_bond_count"]),
    ]

    for mask in (real, backbone, pendant):
        denom = max(int(np.sum(mask)), 1)
        row.extend(
            [
                float(np.sum(mask)),
                float(np.sum(hetero[mask]) / denom),
                float(np.sum(polar[mask]) / denom),
                float(np.sum(halogen[mask]) / denom),
                float(np.sum(donor & mask) / denom),
                float(np.sum(acceptor & mask) / denom),
                float(np.mean(atomic[mask])) if np.any(mask) else 0.0,
                float(np.mean(valence[mask])) if np.any(mask) else 0.0,
                float(np.mean(degree[mask])) if np.any(mask) else 0.0,
                float(np.sum(np.abs(charge[mask]))),
            ]
        )

    bond_blocks: dict[str, list[float]] = {"all": [], "backbone": [], "pendant": [], "bridge": []}
    charge_blocks: dict[str, list[float]] = {"all": [], "backbone": [], "pendant": [], "bridge": []}
    valence_blocks: dict[str, list[float]] = {"all": [], "backbone": [], "pendant": [], "bridge": []}
    hetero_bond_counts = {"all": 0, "backbone": 0, "pendant": 0, "bridge": 0}
    for bond in molecule.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if not (real[a] and real[b]):
            continue
        order = float(bond.GetBondTypeAsDouble())
        denom = max(float(atomic[a] + atomic[b]), 1.0)
        z_moment = order * abs(float(atomic[a] - atomic[b])) / denom
        charge_moment = order * abs(float(charge[a] - charge[b]))
        valence_moment = order * abs(float(valence[a] - valence[b]))
        memberships = ["all"]
        if backbone[a] and backbone[b]:
            memberships.append("backbone")
        if pendant[a] and pendant[b]:
            memberships.append("pendant")
        if (backbone[a] and pendant[b]) or (backbone[b] and pendant[a]):
            memberships.append("bridge")
        for name in memberships:
            bond_blocks[name].append(z_moment)
            charge_blocks[name].append(charge_moment)
            valence_blocks[name].append(valence_moment)
            if bool(hetero[a]) or bool(hetero[b]):
                hetero_bond_counts[name] += 1
    for name in ("all", "backbone", "pendant", "bridge"):
        row.extend(stats(bond_blocks[name]))
        row.extend(stats(charge_blocks[name]))
        row.extend(stats(valence_blocks[name]))
        row.append(float(hetero_bond_counts[name]))

    try:
        distances = np.asarray(Chem.GetDistanceMatrix(molecule), dtype=np.float64)
    except Exception:
        distances = np.full((n_atoms, n_atoms), np.nan, dtype=np.float64)
    donor_acceptor_distances: list[float] = []
    donor_acceptor_weighted: list[float] = []
    polar_pair_distances: list[float] = []
    for i in range(n_atoms):
        if not real[i]:
            continue
        for j in range(i + 1, n_atoms):
            if not real[j] or not np.isfinite(distances[i, j]) or distances[i, j] <= 0:
                continue
            if (donor[i] and acceptor[j]) or (donor[j] and acceptor[i]):
                donor_acceptor_distances.append(float(distances[i, j]))
                donor_acceptor_weighted.append(float(abs(atomic[i] - atomic[j]) / distances[i, j]))
            if polar[i] and polar[j]:
                polar_pair_distances.append(float(distances[i, j]))
    row.extend(stats(donor_acceptor_distances))
    row.extend(stats(donor_acceptor_weighted))
    row.extend(stats(polar_pair_distances))

    if len(row) != 118:
        raise RuntimeError(f"unexpected C238 feature width {len(row)}")
    return row, meta


def bond_polarity_features(parent: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[list[float]] = []
    failures = 0
    pendant_rows = 0
    zwitterion_like = 0
    for molecule in parent["molecules"]:
        try:
            row, meta = bond_polarity_row(molecule)
        except Exception:
            failures += 1
            row, meta = [0.0] * 118, {"feature_failure": True}
        pendant_rows += int(meta.get("pendant_atom_count", 0) > 0)
        zwitterion_like += int(row[9] > 0.0)
        rows.append(row)
    matrix = np.asarray(rows, dtype=np.float64)
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)] = np.nan
    return matrix, {
        "feature_family": "official_smiles_bond_polarity_orientational_moments",
        "shape": [int(value) for value in matrix.shape],
        "feature_count": int(matrix.shape[1]),
        "feature_failures": int(failures),
        "pendant_rows": int(pendant_rows),
        "zwitterion_like_rows": int(zwitterion_like),
        "uses_external_constants": False,
        "uses_bond_dipole_tables": False,
        "uses_labels": False,
        "uses_cross_property_labels": False,
        "uses_pi1m": False,
        "uses_stored_predictions": False,
        "c214_exact_bond_polarity_block_not_reused": True,
    }


def regenerate_c214_selected_parent(
    root: Path,
    parent: dict[str, Any],
    dense: np.ndarray,
    sparse_matrix: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    eps_info = dict(parent["target_info"][ACTIVE_TARGET])
    nc_info = dict(parent["target_info"]["nc"])
    eps_by_canon = dict(zip(eps_info["canonical"], eps_info["y"], strict=True))
    eps_parent_by_canon = dict(zip(eps_info["canonical"], eps_info["parent"], strict=True))
    nc_by_canon = dict(zip(nc_info["canonical"], nc_info["y"], strict=True))
    pair_canons = sorted(set(eps_by_canon) & set(nc_by_canon))
    if len(pair_canons) < 50:
        raise RuntimeError("insufficient official EPS/Nc support to regenerate C214")
    key_to_index = parent["key_to_index"]
    pair_indices = np.asarray([key_to_index[value] for value in pair_canons], dtype=np.int64)
    eps_y = np.asarray([eps_by_canon[value] for value in pair_canons], dtype=np.float64)
    nc_y = np.asarray([nc_by_canon[value] for value in pair_canons], dtype=np.float64)
    ionic_y = eps_y - nc_y ** 2
    if np.any(ionic_y <= 0):
        raise RuntimeError("non-positive regenerated C214 ionic coordinate")
    log_ionic = np.log(ionic_y)
    group_map = dict(zip(eps_info["canonical"], eps_info["groups"], strict=True))
    pair_groups = np.asarray([group_map[value] for value in pair_canons], dtype=object)
    folds = carrier.grouped_folds(pair_groups)
    pair_oof = {kind: np.full(len(pair_canons), np.nan, dtype=np.float64) for kind in c187.MODEL_KINDS}
    fold_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        x_train, x_validation = c187.fold_matrix(dense, sparse_matrix, pair_indices[training], pair_indices[validation])
        for kind in c187.MODEL_KINDS:
            model = c187.make_model(kind, fold)
            model.fit(x_train, log_ionic[training])
            pair_oof[kind][validation] = np.exp(np.clip(model.predict(x_validation), -8, 4))
        raw_eps = nc_y[validation] ** 2 + np.mean([pair_oof[kind][validation] for kind in c187.MODEL_KINDS], axis=0)
        parent_pair = np.asarray([eps_parent_by_canon[value] for value in np.asarray(pair_canons)[validation]], dtype=np.float64)
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "parent_r2": float(r2_score(eps_y[validation], parent_pair)),
                "candidate_r2": float(r2_score(eps_y[validation], raw_eps)),
                "delta_r2": float(r2_score(eps_y[validation], raw_eps) - r2_score(eps_y[validation], parent_pair)),
            }
        )
    raw_oof = nc_y ** 2 + np.mean(np.column_stack([pair_oof[kind] for kind in c187.MODEL_KINDS]), axis=1)
    selected_parent = np.asarray(eps_info["parent"], dtype=np.float64).copy()
    eps_position = {value: index for index, value in enumerate(eps_info["canonical"])}
    for canon, value in zip(pair_canons, raw_oof, strict=True):
        selected_parent[eps_position[canon]] = value

    test_rows, test_indices, test_c050 = c208.target_test_rows(parent, ACTIVE_TARGET)
    selected_test_parent = np.asarray(test_c050, dtype=np.float64).copy()
    test_pair_mask = np.asarray([value in nc_by_canon for value in test_rows["canonical"]], dtype=bool)
    if np.any(test_pair_mask):
        pred_indices = np.asarray([key_to_index[value] for value in test_rows.loc[test_pair_mask, "canonical"]], dtype=np.int64)
        x_train, x_test = c187.fold_matrix(dense, sparse_matrix, pair_indices, pred_indices)
        full_preds = []
        for kind in c187.MODEL_KINDS:
            model = c187.make_model(kind, SEED)
            model.fit(x_train, log_ionic)
            full_preds.append(np.exp(np.clip(model.predict(x_test), -8, 4)))
        nc_test = np.asarray([nc_by_canon[value] for value in test_rows.loc[test_pair_mask, "canonical"]], dtype=np.float64)
        selected_test_parent[test_pair_mask] = nc_test ** 2 + np.mean(np.column_stack(full_preds), axis=1)

    c050_parent = np.asarray(eps_info["parent"], dtype=np.float64)
    selected_r2 = float(r2_score(eps_info["y"], selected_parent))
    expected_abs_error = float(abs(selected_r2 - C214_EXPECTED_EPS_R2))
    if expected_abs_error > C214_MAX_ABS_ERROR:
        raise RuntimeError(
            "regenerated C214 selected EPS parent failed tolerance: "
            f"selected_r2={selected_r2:.17g}, expected={C214_EXPECTED_EPS_R2:.17g}, "
            f"abs_error={expected_abs_error:.17g}, tolerance={C214_MAX_ABS_ERROR:.17g}"
        )
    return selected_parent, selected_test_parent, {
        "selected_parent_run_id": C214_RUN_ID,
        "selected_parent_regenerated_from_source": True,
        "stored_c214_prediction_files_read": False,
        "selected_parent_uses_official_eps_nc_pair_route": True,
        "new_residual_uses_cross_property_labels": False,
        "pair_rows": int(len(pair_canons)),
        "test_pair_rows": int(np.sum(test_pair_mask)),
        "c050_parent_r2": float(r2_score(eps_info["y"], c050_parent)),
        "selected_parent_r2": selected_r2,
        "selected_parent_delta_vs_c050": float(selected_r2 - r2_score(eps_info["y"], c050_parent)),
        "expected_c214_eps_r2": C214_EXPECTED_EPS_R2,
        "expected_c214_abs_error": expected_abs_error,
        "expected_c214_max_abs_error": C214_MAX_ABS_ERROR,
        "expected_c214_tolerance_pass": True,
        "folds": fold_rows,
        "model_kinds": list(c187.MODEL_KINDS),
        "half_parent_blend": 1.0,
    }


def fit_models(x_train: np.ndarray, y_train: np.ndarray) -> list[Any]:
    return [
        make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA),
        ).fit(x_train, y_train),
        make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=TREE_COUNT,
                random_state=SEED,
                min_samples_leaf=TREE_MIN_LEAF,
                max_features=0.85,
                n_jobs=1,
            ),
        ).fit(x_train, y_train),
    ]


def predict_blend(models: list[Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ridge = np.asarray(models[0].predict(x), dtype=np.float64)
    tree = np.asarray(models[1].predict(x), dtype=np.float64)
    correction = RIDGE_BLEND_WEIGHT * ridge + TREE_BLEND_WEIGHT * tree
    return correction, np.column_stack([ridge, tree])


def fit_eps_residual(
    info: dict[str, Any],
    selected_parent: np.ndarray,
    features: np.ndarray,
    test_indices: np.ndarray,
    selected_test_parent: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    direct_oof = np.full((len(y), 2), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        residual = y[training] - selected_parent[training]
        models = fit_models(features[indices[training]], residual)
        correction, direct = predict_blend(models, features[indices[validation]])
        direct_oof[validation] = direct
        fold_candidate = reference.clip_prediction(y[training], selected_parent[validation] + RESIDUAL_WEIGHT * correction)
        candidate[validation] = fold_candidate
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "train_rows": int(len(training)),
                "selected_parent_r2": float(r2_score(y[validation], selected_parent[validation])),
                "candidate_r2": float(r2_score(y[validation], fold_candidate)),
                "delta_r2": float(r2_score(y[validation], fold_candidate) - r2_score(y[validation], selected_parent[validation])),
                "residual_mean": float(np.mean(residual)),
                "residual_std": float(np.std(residual)),
            }
        )
    if not np.isfinite(candidate).all():
        raise RuntimeError("C238 produced non-finite EPS OOF candidate")
    full_models = fit_models(features[indices], y - selected_parent)
    test_correction, _ = predict_blend(full_models, features[test_indices])
    test_candidate = reference.clip_prediction(y, selected_test_parent + RESIDUAL_WEIGHT * test_correction)
    return {
        "candidate": candidate,
        "test_candidate": test_candidate,
        "direct_oof": direct_oof,
        "blend_name": "fixed_0.30_residual__0.70_ridge_0.30_extratrees",
        "weights": [RIDGE_BLEND_WEIGHT, TREE_BLEND_WEIGHT],
        "intercept": 0.0,
        "blend_r2": float(r2_score(y, candidate)),
        "fold_robust_reports": fold_reports,
        "full_robust_report": {
            "train_rows": int(len(y)),
            "test_rows": int(len(test_indices)),
            "residual_weight": RESIDUAL_WEIGHT,
            "ridge_alpha": RIDGE_ALPHA,
            "tree_count": TREE_COUNT,
            "tree_min_leaf": TREE_MIN_LEAF,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--canonical-run",
        default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7",
    )
    args = parser.parse_args()
    started = time.time()
    root = Path(args.root).resolve()
    round2_root = root / "Polymer Prediction Challenge Round 2"
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    progress = run_dir / "progress.jsonl"
    checkpoint(progress, "started", experiment_id=run_dir.name)

    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)

    rich_dense, rich_sparse, rich_report = c187.rich_builder.build_features(root, parent["keys"])
    rich_dense = np.asarray(rich_dense, dtype=np.float64)
    rich_sparse = rich_sparse.astype(np.float64)
    selected_parent, selected_test_parent, selected_parent_report = regenerate_c214_selected_parent(root, parent, rich_dense, rich_sparse)
    checkpoint(
        progress,
        "selected_parent_regenerated",
        selected_parent_r2=selected_parent_report["selected_parent_r2"],
        expected_c214_abs_error=selected_parent_report["expected_c214_abs_error"],
        pair_rows=selected_parent_report["pair_rows"],
    )

    features, feature_report = bond_polarity_features(parent)
    checkpoint(progress, "features_complete", shape=feature_report["shape"], pendant_rows=feature_report["pendant_rows"])

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    c050_parent = np.asarray(info["parent"], dtype=np.float64)
    eval_info = dict(info)
    eval_info["parent"] = selected_parent
    test_rows, test_indices, _ = c208.target_test_rows(parent, ACTIVE_TARGET)
    result = fit_eps_residual(eval_info, selected_parent, features, test_indices, selected_test_parent)
    active_report = c208.evaluate_tg(eval_info, result)
    c050_parent_r2 = float(r2_score(info["y"], c050_parent))
    active_report.update(
        {
            "changed_factor": "EPS official-SMILES bond-polarity/orientational residual over regenerated C214 selected parent",
            "selected_reference_run_id": C214_RUN_ID,
            "selected_reference_r2": selected_parent_report["selected_parent_r2"],
            "selected_reference_delta_vs_c050": selected_parent_report["selected_parent_delta_vs_c050"],
            "c050_parent_r2": c050_parent_r2,
            "candidate_delta_vs_c050": float(active_report["candidate_r2"] - c050_parent_r2),
            "minimum_selected_reference_delta_r2": MIN_SELECTED_REFERENCE_DELTA_R2,
            "new_residual_uses_cross_property_labels": False,
            "selected_parent_uses_official_eps_nc_pair_route": True,
            "uses_external_constants": False,
            "uses_bond_dipole_tables": False,
            "uses_pi1m": False,
            "uses_stored_prediction_replay": False,
            "c214_exact_bond_polarity_block_not_reused": True,
        }
    )
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "eps_bond_polarity_residual_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        selected_reference_r2=active_report["selected_reference_r2"],
        candidate_delta_vs_c050=active_report["candidate_delta_vs_c050"],
        pass_gate=banked,
        positive_folds=active_report["positive_folds"],
        minimum_panel_delta=active_report["minimum_panel_delta"],
        group_bootstrap_lower=active_report["group_bootstrap_lower"],
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = active_report
            candidate = np.asarray(result["candidate"], dtype=np.float64)
            direct_oof = np.asarray(result["direct_oof"], dtype=np.float64)
            immediate_parent = selected_parent
        else:
            report = c208.unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=np.float64)
            direct_oof = np.full((len(candidate), 2), np.nan, dtype=np.float64)
            immediate_parent = np.asarray(target_info["parent"], dtype=np.float64)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        assembled = candidate if target == ACTIVE_TARGET and banked else immediate_parent
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "selected_reference_parent": immediate_parent,
                    "candidate": candidate,
                    "assembled": assembled,
                    "direct_ridge": direct_oof[:, 0],
                    "direct_tree": direct_oof[:, 1],
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": folds,
                    "banked": bool(target == ACTIVE_TARGET and banked),
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target == ACTIVE_TARGET and banked else 0.0 for target in TARGETS))

    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component_test = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "selected_reference_parent": selected_test_parent,
            "direct_candidate": result["test_candidate"],
        }
    )
    predictions = parent_test.merge(component_test, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(
        (predictions["target_type"] == ACTIVE_TARGET) & banked,
        predictions["direct_candidate"],
        predictions["target"],
    )
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940:
        raise RuntimeError("C238 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C238 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C238 prediction finite check failed")

    report = {
        "schema_version": SCHEMA,
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "stored_prediction_replay": False,
        "stored_c214_prediction_files_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "active_target": ACTIVE_TARGET,
        "parent_replay_parity": parity,
        "selected_parent_report": selected_parent_report,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": bool(banked),
        "goal_0_95_met": bool(banked and assembled_mean >= 0.95),
        "decision": "banked_component_pending_compound_audit" if banked else "rejected_component_or_full_gate",
        "component_diagnostics": {
            "active_target": ACTIVE_TARGET,
            "changed_factor": "EPS bond-polarity/orientational residual over regenerated C214",
            "eps_delta_vs_selected_reference": active_report["delta_r2"],
            "eps_candidate_r2": active_report["candidate_r2"],
            "eps_selected_reference_r2": active_report["selected_reference_r2"],
            "eps_candidate_delta_vs_c050": active_report["candidate_delta_vs_c050"],
            "eps_positive_folds": active_report["positive_folds"],
            "eps_group_bootstrap_lower": active_report["group_bootstrap_lower"],
            "eps_minimum_panel_delta": active_report["minimum_panel_delta"],
            "new_residual_uses_cross_property_labels": False,
            "uses_external_constants": False,
            "uses_pi1m": False,
            "uses_stored_prediction_replay": False,
        },
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "c187_selected_parent_helper": sha256_file(round2_root / "tools/round2_c187_ionic_eps_only.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c208_panel_helper": sha256_file(round2_root / "tools/round2_c208_tg_robust_group_measurement.py"),
            "rich_builder": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    eps_oof = oof.loc[oof["target_type"].astype(str).eq(ACTIVE_TARGET), ["canonical", "target", "parent", "selected_reference_parent", "candidate", "assembled"]].copy()
    eps_oof.to_csv(run_dir / "eps_oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component_test.to_csv(run_dir / "eps_component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "active_target": ACTIVE_TARGET,
            "features": "official-SMILES formal-charge, bond-polarity, graph-distance, and wildcard-backbone/pendant concentration moments",
            "selected_parent": "regenerated C214 EPS ionic full-amplitude component from official inputs",
            "model": "fixed 0.30 residual blend of Ridge and ExtraTrees",
            "ridge_alpha": RIDGE_ALPHA,
            "tree_count": TREE_COUNT,
            "tree_min_leaf": TREE_MIN_LEAF,
            "residual_weight": RESIDUAL_WEIGHT,
            "component_gate": {
                "minimum_delta_vs_selected_reference_r2": MIN_SELECTED_REFERENCE_DELTA_R2,
                "minimum_positive_folds": 4,
                "bootstrap_lower_must_exceed": 0.0,
                "minimum_panel_delta_must_be_at_least": 0.0,
            },
            "official_only": True,
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"sklearn={__import__('sklearn').__version__}",
                f"rdkit={rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: **{report['decision']}**. EPS selected reference R² "
        f"`{active_report['selected_reference_r2']:.12f}`, candidate R² "
        f"`{active_report['candidate_r2']:.12f}`, delta "
        f"`{active_report['delta_r2']:+.12f}`. C214 parent regenerated from "
        "official inputs; no stored C214 predictions, local_eval, Kaggle compute, "
        "upload, submission, or final-notebook action.\n",
        encoding="utf-8",
    )
    manifest_paths = [path for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]
    lines = [f"{sha256_file(path)}  {path.name}" for path in manifest_paths]
    lines.extend(f"{digest}  SOURCE {name}" for name, digest in sorted(report["source_hashes"].items()))
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "decision": report["decision"],
                "eps_selected_reference_r2": active_report["selected_reference_r2"],
                "eps_candidate_r2": active_report["candidate_r2"],
                "eps_delta_vs_selected_reference": active_report["delta_r2"],
                "banked_targets": report["banked_targets"],
                "mean_candidate_r2": report["mean_candidate_r2"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
