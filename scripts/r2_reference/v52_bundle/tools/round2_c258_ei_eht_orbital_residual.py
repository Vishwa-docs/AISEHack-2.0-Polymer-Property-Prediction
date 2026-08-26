#!/usr/bin/env python3
"""C258: Ei RDKit/YAeHMOP extended-Hueckel orbital residual.

This child is allocated after the C257 reflection gate.  It tests one new
official-only physical coordinate: deterministic RDKit rdEHTTools orbital and
charge features computed from H-capped and ring-closed official polymer SMILES.

The active reference is the currently selected C199 Ei component.  C258 may bank
Ei only if the EHT residual improves that selected reference by at least +0.010
R2 under grouped folds, bootstrap, and transfer panels.  C050 is still rebuilt
and replayed exactly before fitting; C199 is regenerated from official inputs,
not read from stored predictions.

No local_eval/test external_labels, public feedback, external data rows, pretrained weights,
stored predictions, Kaggle compute, upload, submission, or final notebook action
are used.
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
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, rdEHTTools
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as c180
import round2_c196_ei_ffox_shrinkage_confirmation as c196
import round2_c199_ei_c196_transfer_guard as c199


RDLogger.DisableLog("rdApp.*")

TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "ei"
SCHEMA = "ppp.round2.c258.ei-eht-orbital-residual.v1"
SEED = 20260805
RIDGE_ALPHA = 60.0
RESIDUAL_WEIGHT = 0.35
MIN_SELECTED_REFERENCE_DELTA_R2 = 0.010
C199_REFERENCE_EI_R2 = 0.8566558157138717
EMBED_MAX_ITERS = 100


VALENCE_ELECTRONS = {
    1: 1,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
    32: 4,
    34: 6,
    35: 7,
    50: 4,
    53: 7,
}


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


def remove_dummy_caps(smiles: str) -> Chem.Mol | None:
    """Remove polymer attachment-point dummies and let RDKit add hydrogens."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        editable = Chem.RWMol(molecule)
        for atom_index in sorted(
            [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0],
            reverse=True,
        ):
            editable.RemoveAtom(atom_index)
        capped = editable.GetMol()
        Chem.SanitizeMol(capped)
        return Chem.AddHs(capped)
    except Exception:
        return None


def ring_close_dummy_caps(smiles: str) -> Chem.Mol | None:
    """Close the two dummy-attachment neighbors into a cyclic repeat surrogate."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        editable = Chem.RWMol(molecule)
        dummy = [atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy) != 2:
            return None
        neighbors: list[int] = []
        for atom_index in dummy:
            atom = editable.GetAtomWithIdx(atom_index)
            heavy_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 0]
            if not heavy_neighbors:
                return None
            neighbors.append(int(heavy_neighbors[0]))
        if neighbors[0] == neighbors[1]:
            return None
        if editable.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is None:
            editable.AddBond(neighbors[0], neighbors[1], Chem.BondType.SINGLE)
        for atom_index in sorted(dummy, reverse=True):
            editable.RemoveAtom(atom_index)
        closed = editable.GetMol()
        Chem.SanitizeMol(closed)
        return Chem.AddHs(closed)
    except Exception:
        return None


def embed_for_eht(molecule: Chem.Mol, seed: int) -> Chem.Mol | None:
    if molecule is None or molecule.GetNumAtoms() < 2:
        return None
    try:
        working = Chem.Mol(molecule)
        conformer_id = AllChem.EmbedMolecule(working, randomSeed=int(seed))
        if conformer_id < 0:
            conformer_id = AllChem.EmbedMolecule(working, randomSeed=int(seed), useRandomCoords=True)
        if conformer_id < 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(working, maxIters=EMBED_MAX_ITERS)
        except Exception:
            pass
        return working
    except Exception:
        return None


def valence_electron_count(molecule: Chem.Mol) -> int:
    total = 0
    for atom in molecule.GetAtoms():
        atomic_number = int(atom.GetAtomicNum())
        total += VALENCE_ELECTRONS.get(atomic_number, min(max(atomic_number, 1), 8))
    return int(total)


def eht_variant_features(molecule: Chem.Mol | None, seed: int) -> tuple[list[float], bool]:
    embedded = embed_for_eht(molecule, seed)
    if embedded is None:
        return [np.nan] * 18, False
    try:
        ok, result = rdEHTTools.RunMol(embedded)
    except Exception:
        return [np.nan] * 18, False
    if not ok:
        return [np.nan] * 18, False
    energies = np.asarray(result.GetOrbitalEnergies(), dtype=np.float64)
    charges = np.asarray(result.GetAtomicCharges(), dtype=np.float64)
    if energies.size < 2 or charges.size == 0 or not np.isfinite(energies).all() or not np.isfinite(charges).all():
        return [np.nan] * 18, False
    electrons = valence_electron_count(embedded)
    homo_index = max(0, min(int(energies.size) - 2, int(electrons // 2) - 1))
    lumo_index = homo_index + 1
    homo = float(energies[homo_index])
    lumo = float(energies[lumo_index])
    below = float(energies[homo_index - 1]) if homo_index > 0 else homo
    above = float(energies[lumo_index + 1]) if lumo_index + 1 < energies.size else lumo
    features = [
        homo,
        lumo,
        float(lumo - homo),
        below,
        above,
        float(homo - below),
        float(above - lumo),
        float(np.min(energies)),
        float(np.max(energies)),
        float(np.mean(energies)),
        float(np.std(energies)),
        float(np.quantile(energies, 0.25)),
        float(np.quantile(energies, 0.75)),
        float(np.min(charges)),
        float(np.max(charges)),
        float(np.mean(charges)),
        float(np.std(charges)),
        float(np.sum(np.abs(charges))),
    ]
    return features, True


def eht_feature_names() -> list[str]:
    per_variant = [
        "homo",
        "lumo",
        "gap",
        "homo_minus_1",
        "lumo_plus_1",
        "homo_spacing",
        "lumo_spacing",
        "energy_min",
        "energy_max",
        "energy_mean",
        "energy_std",
        "energy_q25",
        "energy_q75",
        "charge_min",
        "charge_max",
        "charge_mean",
        "charge_std",
        "charge_abs_sum",
    ]
    names: list[str] = []
    for variant in ("hcap", "ring"):
        names.extend([f"eht_{variant}_{name}" for name in per_variant])
        names.append(f"eht_{variant}_supported")
    names.extend(
        [
            "eht_hcap_minus_ring_homo",
            "eht_hcap_minus_ring_lumo",
            "eht_hcap_minus_ring_gap",
        ]
    )
    return names


def eht_features_for_smiles(smiles: str, row_number: int) -> tuple[np.ndarray, dict[str, Any]]:
    hcap_values, hcap_ok = eht_variant_features(remove_dummy_caps(smiles), SEED + row_number * 2 + 1)
    ring_values, ring_ok = eht_variant_features(ring_close_dummy_caps(smiles), SEED + row_number * 2 + 2)
    diff = [
        hcap_values[0] - ring_values[0] if hcap_ok and ring_ok else np.nan,
        hcap_values[1] - ring_values[1] if hcap_ok and ring_ok else np.nan,
        hcap_values[2] - ring_values[2] if hcap_ok and ring_ok else np.nan,
    ]
    row = np.asarray(hcap_values + [float(hcap_ok)] + ring_values + [float(ring_ok)] + diff, dtype=np.float64)
    return row, {"hcap_supported": bool(hcap_ok), "ring_supported": bool(ring_ok)}


def build_eht_matrix(parent: dict[str, Any], progress: Path) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    hcap_supported = 0
    ring_supported = 0
    started = time.time()
    for row_number, smiles in enumerate(parent["keys"]):
        row, support = eht_features_for_smiles(str(smiles), row_number)
        rows.append(row)
        hcap_supported += int(support["hcap_supported"])
        ring_supported += int(support["ring_supported"])
        if (row_number + 1) % 500 == 0:
            checkpoint(
                progress,
                "eht_features_progress",
                processed=row_number + 1,
                hcap_supported=hcap_supported,
                ring_supported=ring_supported,
                elapsed_seconds=float(time.time() - started),
            )
    matrix = np.vstack(rows).astype(np.float64)
    names = eht_feature_names()
    if matrix.shape[1] != len(names):
        raise RuntimeError("C258 EHT feature-name width mismatch")
    report = {
        "feature_shape": [int(value) for value in matrix.shape],
        "feature_names": names,
        "hcap_supported": int(hcap_supported),
        "ring_supported": int(ring_supported),
        "hcap_support_fraction": float(hcap_supported / max(1, len(parent["keys"]))),
        "ring_support_fraction": float(ring_supported / max(1, len(parent["keys"]))),
        "elapsed_seconds": float(time.time() - started),
        "rdkit_eht_source": "rdkit.Chem.rdEHTTools interface to YAeHMOP extended Hueckel library",
        "external_data_rows": False,
    }
    return matrix, report


def selected_c199_reference(parent: dict[str, Any], root: Path, progress: Path) -> dict[str, Any]:
    dense, sparse_features, feature_report = c180.build_features(root, parent["keys"])
    checkpoint(
        progress,
        "c199_reference_features_complete",
        dense_shape=feature_report["dense_shape"],
        sparse_shape=feature_report["sparse_shape"],
        sparse_nnz=feature_report["sparse_nnz"],
    )
    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = c196.target_test_rows(parent, ACTIVE_TARGET)
    raw_result = carrier.fit_target(info, dense, sparse_features, test_indices, test_parent)
    raw_report = carrier.evaluate_target(info, raw_result)

    c050_oof = np.asarray(info["parent"], dtype=np.float64)
    shrunk_oof = c050_oof + c199.SHRINK_ALPHA * (np.asarray(raw_result["candidate"], dtype=np.float64) - c050_oof)
    nearest = c199.fold_local_nearest(parent, info)
    oof_guard = c199.transfer_guard_mask(np.asarray(info["scaffolds"], dtype=object), nearest)
    selected_oof = shrunk_oof.copy()
    selected_oof[oof_guard] = c050_oof[oof_guard]

    shrunk_test = test_parent + c199.SHRINK_ALPHA * (
        np.asarray(raw_result["test_direct"], dtype=np.float64) - test_parent
    )
    test_nearest = c199.full_train_nearest(parent, info, test_indices)
    test_scaffolds = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"]], dtype=object)
    test_guard = c199.transfer_guard_mask(test_scaffolds, test_nearest)
    selected_test = shrunk_test.copy()
    selected_test[test_guard] = test_parent[test_guard]

    selected_r2 = float(r2_score(info["y"], selected_oof))
    replay_error = abs(selected_r2 - C199_REFERENCE_EI_R2)
    report = {
        "selected_reference": "R2-C199-20260805-0254-ei-c196-transfer-guard-v1 regenerated from official inputs",
        "selected_reference_r2": selected_r2,
        "expected_c199_reference_r2": C199_REFERENCE_EI_R2,
        "selected_reference_r2_abs_error": float(replay_error),
        "selected_reference_replay_pass": bool(replay_error <= 1.0e-10),
        "raw_c180_candidate_r2": raw_report["candidate_r2"],
        "raw_c180_delta_r2": raw_report["delta_r2"],
        "raw_c180_positive_folds": raw_report["positive_folds"],
        "raw_c180_group_bootstrap_lower": raw_report["group_bootstrap_lower"],
        "raw_c180_minimum_panel_delta": raw_report["minimum_panel_delta"],
        "guarded_oof_rows": int(np.sum(oof_guard)),
        "guarded_test_rows": int(np.sum(test_guard)),
        "feature_report": feature_report,
    }
    if not report["selected_reference_replay_pass"]:
        raise RuntimeError(f"C199 selected reference replay failed: {report}")
    checkpoint(progress, "c199_selected_reference_replayed", **report)
    return {
        "oof": selected_oof,
        "test": selected_test,
        "test_rows": test_rows,
        "test_indices": test_indices,
        "report": report,
    }


def fit_fold_residual(train_x: np.ndarray, train_y: np.ndarray) -> Any:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4),
    )
    model.fit(train_x, train_y)
    return model


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, selected: np.ndarray, minimum: int = 8) -> float | None:
    selected = np.asarray(selected, dtype=bool)
    if int(np.sum(selected)) < minimum or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))


def transfer_report(
    parent: dict[str, Any],
    info: dict[str, Any],
    selected_reference: np.ndarray,
    candidate: np.ndarray,
    eht_matrix: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    indices = np.asarray(info["indices"], dtype=np.int64)
    nearest = c199.fold_local_nearest(parent, info)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    hcap_supported = np.isfinite(eht_matrix[indices, 0])
    ring_supported = np.isfinite(eht_matrix[indices, 19])
    panel_specs: dict[str, np.ndarray] = {
        "eht_hcap_supported": hcap_supported,
        "eht_hcap_missing": ~hcap_supported,
        "eht_ring_supported": ring_supported,
        "eht_ring_missing": ~ring_supported,
        "similarity_lt_0.30": nearest < 0.30,
        "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
        "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
        "similarity_ge_0.70": nearest >= 0.70,
        "ei_low_quartile": y <= np.quantile(y, 0.25),
        "ei_high_quartile": y >= np.quantile(y, 0.75),
    }
    for scaffold_name in sorted(set(scaffolds.astype(str))):
        selected = scaffolds.astype(str) == scaffold_name
        if int(np.sum(selected)) >= 10:
            panel_specs[f"scaffold_{scaffold_name}"] = selected
    panels: dict[str, Any] = {}
    panel_values: list[float] = []
    for name, selected in panel_specs.items():
        delta = panel_delta(y, selected_reference, candidate, selected)
        panels[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable_insufficient_support",
        }
        if delta is not None:
            panel_values.append(float(delta))
    fold_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        rows = folds == fold
        fold_parent = float(r2_score(y[rows], selected_reference[rows]))
        fold_candidate = float(r2_score(y[rows], candidate[rows]))
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(np.sum(rows)),
                "selected_reference_r2": fold_parent,
                "candidate_r2": fold_candidate,
                "delta_r2": float(fold_candidate - fold_parent),
            }
        )
    selected_delta = float(r2_score(y, candidate) - r2_score(y, selected_reference))
    lower = float(carrier.bootstrap_lower(y, selected_reference, candidate, groups))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    minimum_panel = float(min(panel_values)) if panel_values else 0.0
    return {
        "selected_reference_delta_r2": selected_delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_panel_delta": minimum_panel,
        "panels": panels,
        "folds": fold_rows,
    }


def unchanged_report(info: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    return {
        "active": False,
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, parent)),
        "delta_r2": 0.0,
        "positive_folds": 0,
        "group_bootstrap_lower": 0.0,
        "minimum_panel_delta": 0.0,
        "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}},
        "folds": [],
        "pass": False,
        "unchanged_parent": True,
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

    selected_reference = selected_c199_reference(parent, root, progress)
    eht_matrix, eht_report = build_eht_matrix(parent, progress)
    checkpoint(progress, "eht_features_complete", **eht_report)

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    y = np.asarray(info["y"], dtype=np.float64)
    c050_parent = np.asarray(info["parent"], dtype=np.float64)
    selected_oof = np.asarray(selected_reference["oof"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    residual_oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        model = fit_fold_residual(eht_matrix[indices[training]], y[training] - selected_oof[training])
        residual_oof[validation] = model.predict(eht_matrix[indices[validation]])
    if not np.isfinite(residual_oof).all():
        raise RuntimeError("C258 residual OOF contains non-finite values")
    candidate_oof = selected_oof + RESIDUAL_WEIGHT * residual_oof
    transfer = transfer_report(parent, info, selected_oof, candidate_oof, eht_matrix)

    selected_reference_r2 = float(r2_score(y, selected_oof))
    candidate_r2 = float(r2_score(y, candidate_oof))
    c050_r2 = float(r2_score(y, c050_parent))
    selected_delta = float(candidate_r2 - selected_reference_r2)
    c050_delta = float(candidate_r2 - c050_r2)
    passed = bool(
        selected_delta >= MIN_SELECTED_REFERENCE_DELTA_R2
        and transfer["positive_folds"] >= 4
        and transfer["group_bootstrap_lower"] > 0.0
        and transfer["minimum_panel_delta"] >= 0.0
    )
    checkpoint(
        progress,
        "ei_eht_residual_complete",
        selected_reference_r2=selected_reference_r2,
        candidate_r2=candidate_r2,
        selected_reference_delta_r2=selected_delta,
        c050_delta_r2=c050_delta,
        positive_folds=transfer["positive_folds"],
        group_bootstrap_lower=transfer["group_bootstrap_lower"],
        minimum_panel_delta=transfer["minimum_panel_delta"],
        pass_gate=passed,
    )

    test_rows = selected_reference["test_rows"]
    test_indices = np.asarray(selected_reference["test_indices"], dtype=np.int64)
    full_model = fit_fold_residual(eht_matrix[indices], y - selected_oof)
    selected_test = np.asarray(selected_reference["test"], dtype=np.float64)
    candidate_test = selected_test + RESIDUAL_WEIGHT * full_model.predict(eht_matrix[test_indices])

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = {
                "active": True,
                "parent_r2": selected_reference_r2,
                "c050_parent_r2": c050_r2,
                "selected_reference_r2": selected_reference_r2,
                "expected_selected_reference_r2": C199_REFERENCE_EI_R2,
                "candidate_r2": candidate_r2,
                "delta_r2": selected_delta,
                "selected_reference_delta_r2": selected_delta,
                "c050_delta_r2": c050_delta,
                "positive_folds": transfer["positive_folds"],
                "group_bootstrap_lower": transfer["group_bootstrap_lower"],
                "minimum_panel_delta": transfer["minimum_panel_delta"],
                "panels": transfer["panels"],
                "folds": transfer["folds"],
                "pass": passed,
                "minimum_selected_reference_delta_r2": MIN_SELECTED_REFERENCE_DELTA_R2,
                "ridge_alpha": RIDGE_ALPHA,
                "residual_weight": RESIDUAL_WEIGHT,
                "feature_family": "rdkit_yaehmop_extended_hueckel_orbital_charge",
                "selected_reference_report": selected_reference["report"],
            }
            assembled = candidate_oof if passed else c050_parent
            candidate = candidate_oof
            parent_for_oof = c050_parent
        else:
            report = unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=np.float64)
            assembled = candidate
            parent_for_oof = candidate
        target_reports[target] = report
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": parent_for_oof,
                    "selected_reference": selected_oof if target == ACTIVE_TARGET else parent_for_oof,
                    "candidate": candidate,
                    "assembled": assembled,
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object)),
                }
            )
        )

    banked = [ACTIVE_TARGET] if passed else []
    parent_mean = float(np.mean([r2_score(part["target"], part["parent"]) for part in oof_parts]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    selected_composite_mean_if_banked = 0.8941972740330625 + (selected_delta / 7.0)

    parent_detail = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if passed:
        component = pd.DataFrame(
            {
                "id": test_rows["id"].astype(int),
                "target_type": ACTIVE_TARGET,
                "candidate": candidate_test,
            }
        )
        predictions_detail = parent_detail.merge(component, on=["id", "target_type"], how="left", validate="one_to_one")
        predictions_detail["target"] = np.where(
            predictions_detail["target_type"].astype(str).eq(ACTIVE_TARGET),
            predictions_detail["candidate"],
            predictions_detail["target"],
        )
    else:
        predictions_detail = parent_detail.copy()
    predictions = predictions_detail[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C258 ID coverage/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("C258 produced non-finite predictions")

    source_paths = {
        "runner": Path(__file__).resolve(),
        "parent_builder": root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py",
        "carrier": root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py",
        "reference": root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py",
        "c180": root / "Polymer Prediction Challenge Round 2/tools/round2_c180_flory_fox_oligomer_carriers.py",
        "c196": root / "Polymer Prediction Challenge Round 2/tools/round2_c196_ei_ffox_shrinkage_confirmation.py",
        "c199": root / "Polymer Prediction Challenge Round 2/tools/round2_c199_ei_c196_transfer_guard.py",
    }
    metrics = {
        "schema_version": SCHEMA,
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "stored_prediction_replay": False,
        "cross_target_labels": False,
        "active_target": ACTIVE_TARGET,
        "parent": "C050 source rebuild plus regenerated C199 selected Ei reference",
        "hypothesis": "RDKit/YAeHMOP extended-Hueckel orbital and charge coordinates add Ei residual signal beyond the selected C199 Ei component.",
        "changed_factor": "Add deterministic RDKit rdEHTTools H-capped/ring-closed orbital-charge residual features over regenerated C199 Ei; fixed Ridge alpha and residual weight.",
        "parent_replay_parity": parity,
        "eht_feature_report": eht_report,
        "selected_reference_report": selected_reference["report"],
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "selected_composite_mean_if_banked": float(selected_composite_mean_if_banked if passed else 0.8941972740330625),
        "gap_to_0_95_if_banked": float(0.95 - selected_composite_mean_if_banked) if passed else 0.05580272596693747,
        "full_candidate_gate_pass": bool(passed),
        "goal_0_95_met": bool(passed and selected_composite_mean_if_banked >= 0.95),
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "prediction_exact_order": True,
        "prediction_unique_ids": True,
        "prediction_finite_targets": True,
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "decision": "completed_banked" if passed else "completed_rejected",
    }
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "selected_reference": selected_test,
            "candidate": candidate_test,
        }
    ).to_csv(run_dir / "ei_component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": SEED,
            "ridge_alpha": RIDGE_ALPHA,
            "residual_weight": RESIDUAL_WEIGHT,
            "minimum_selected_reference_delta_r2": MIN_SELECTED_REFERENCE_DELTA_R2,
            "active_target": ACTIVE_TARGET,
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
        f"Decision: `{metrics['decision']}`. Ei selected-reference R2 "
        f"`{selected_reference_r2:.12f}`; EHT candidate `{candidate_r2:.12f}`; "
        f"selected-reference delta `{selected_delta:+.12f}`; C050 delta `{c050_delta:+.12f}`. "
        f"Positive folds `{transfer['positive_folds']}/5`; bootstrap lower "
        f"`{transfer['group_bootstrap_lower']:+.12f}`; minimum panel delta "
        f"`{transfer['minimum_panel_delta']:+.12f}`. Official-only; no local_eval read; "
        "no Kaggle action; no final-notebook action.\n",
        encoding="utf-8",
    )
    manifest = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in sorted(metrics["source_hashes"].items()):
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    checkpoint(progress, "finished", decision=metrics["decision"], elapsed_seconds=metrics["elapsed_seconds"])


if __name__ == "__main__":
    main()
