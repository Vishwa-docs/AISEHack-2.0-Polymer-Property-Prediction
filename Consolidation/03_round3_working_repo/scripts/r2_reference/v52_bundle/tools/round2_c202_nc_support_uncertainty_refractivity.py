#!/usr/bin/env python3
"""C202: Nc support/uncertainty refractivity residual.

This is a bounded official-only Nc child. It does not reuse the C195/C197 Nc
arms, does not use PI1M, does not use EPS-to-Nc partner predictions, and does
not consume any active/queued component output. It rebuilds exact C050, derives
conformer-free refractivity and graph-distance shell features from official
SMILES, adds fold-local label-free support-density features, and applies one
fixed low-variance Ridge residual only on rows passing a fixed support gate.

Rows outside the support gate are exact C050 fallback. Any failed gate rejects
the component and cools this branch; there is no threshold/alpha/shell retune.
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
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "nc"
SEED = 20260805
RIDGE_ALPHA = 80.0
RESIDUAL_WEIGHT = 0.25
SUPPORT_NEAREST_MIN = 0.30
SUPPORT_TOP3_MEAN_MIN = 0.20
MIN_BANKABLE_DELTA_R2 = 0.01


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


def atom_flags(atom: Chem.Atom) -> tuple[float, float, float, float]:
    atomic = atom.GetAtomicNum()
    heavy = float(atomic > 1)
    hetero = float(atomic not in (0, 1, 6))
    aromatic = float(atom.GetIsAromatic())
    polar = float(atomic in (7, 8, 9, 15, 16, 17, 35, 53))
    return heavy, hetero, aromatic, polar


def safe_stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.sum(values)),
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    ]


def graph_shell_moments(molecule: Chem.Mol, mr: np.ndarray, logp: np.ndarray) -> list[float]:
    atoms = list(molecule.GetAtoms())
    if not atoms:
        return [0.0] * (5 * 8)
    distance = np.asarray(Chem.GetDistanceMatrix(molecule, useBO=False), dtype=np.float64)
    heavy = np.asarray([atom_flags(atom)[0] for atom in atoms], dtype=np.float64)
    hetero = np.asarray([atom_flags(atom)[1] for atom in atoms], dtype=np.float64)
    aromatic = np.asarray([atom_flags(atom)[2] for atom in atoms], dtype=np.float64)
    polar = np.asarray([atom_flags(atom)[3] for atom in atoms], dtype=np.float64)
    values: list[float] = []
    for shell in (1, 2, 3, 4, 5):
        if shell < 5:
            mask = distance == float(shell)
        else:
            mask = distance >= 5.0
        # Use upper-triangle pairs only; the shell features are label-free.
        mask = np.triu(mask, k=1)
        if not np.any(mask):
            values.extend([0.0] * 8)
            continue
        mr_pair = (mr[:, None] * mr[None, :])[mask]
        logp_pair = (logp[:, None] * logp[None, :])[mask]
        hetero_pair = (hetero[:, None] + hetero[None, :])[mask]
        aromatic_pair = (aromatic[:, None] + aromatic[None, :])[mask]
        polar_pair = (polar[:, None] + polar[None, :])[mask]
        heavy_pair = (heavy[:, None] + heavy[None, :])[mask]
        values.extend(
            [
                float(np.sum(mask)),
                float(np.mean(mr_pair)) if mr_pair.size else 0.0,
                float(np.std(mr_pair)) if mr_pair.size else 0.0,
                float(np.mean(logp_pair)) if logp_pair.size else 0.0,
                float(np.mean(hetero_pair)) if hetero_pair.size else 0.0,
                float(np.mean(aromatic_pair)) if aromatic_pair.size else 0.0,
                float(np.mean(polar_pair)) if polar_pair.size else 0.0,
                float(np.mean(heavy_pair)) if heavy_pair.size else 0.0,
            ]
        )
    return values


def refractivity_features(parent: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[list[float]] = []
    failures = 0
    for molecule in parent["molecules"]:
        try:
            contribs = Crippen._GetAtomContribs(molecule)
            logp = np.asarray([item[0] for item in contribs], dtype=np.float64)
            mr = np.asarray([item[1] for item in contribs], dtype=np.float64)
            atoms = list(molecule.GetAtoms())
            heavy = max(int(molecule.GetNumHeavyAtoms()), 1)
            hetero = np.asarray([atom_flags(atom)[1] for atom in atoms], dtype=np.float64)
            aromatic = np.asarray([atom_flags(atom)[2] for atom in atoms], dtype=np.float64)
            polar = np.asarray([atom_flags(atom)[3] for atom in atoms], dtype=np.float64)
            polar_mr = mr * polar
            hetero_mr = mr * hetero
            aromatic_mr = mr * aromatic
            base = [
                float(heavy),
                float(np.log1p(heavy)),
                float(rdMolDescriptors.CalcTPSA(molecule) / heavy),
                float(rdMolDescriptors.CalcNumAromaticRings(molecule)),
                float(rdMolDescriptors.CalcNumAliphaticRings(molecule)),
                float(rdMolDescriptors.CalcNumHBA(molecule) / heavy),
                float(rdMolDescriptors.CalcNumHBD(molecule) / heavy),
                float(Descriptors.NumRotatableBonds(molecule) / heavy),
                float(Crippen.MolMR(molecule)),
                float(Crippen.MolMR(molecule) / heavy),
                float(Crippen.MolLogP(molecule)),
                float(Crippen.MolLogP(molecule) / heavy),
                float(np.sum(hetero) / heavy),
                float(np.sum(aromatic) / heavy),
                float(np.sum(polar) / heavy),
                float(np.sum(polar_mr) / heavy),
                float(np.sum(hetero_mr) / heavy),
                float(np.sum(aromatic_mr) / heavy),
                float((Crippen.MolMR(molecule) ** 2) / heavy),
                float(Crippen.MolMR(molecule) * max(Crippen.MolLogP(molecule), -20.0) / heavy),
            ]
            row = base + safe_stats(mr) + safe_stats(logp) + safe_stats(polar_mr) + safe_stats(hetero_mr) + graph_shell_moments(molecule, mr, logp)
        except Exception:
            failures += 1
            row = [0.0] * (20 + 5 * 4 + 5 * 8)
        rows.append(row)
    matrix = np.asarray(rows, dtype=np.float64)
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)] = np.nan
    return matrix, {
        "feature_family": "conformer_free_refractivity_graph_shells",
        "shape": [int(value) for value in matrix.shape],
        "rdkit_contribution_source": "Crippen._GetAtomContribs per-atom logP/MR",
        "graph_shells": [1, 2, 3, 4, "5_or_more"],
        "failed_feature_rows": int(failures),
        "uses_labels": False,
        "uses_cross_property_labels": False,
        "uses_pi1m": False,
    }


def support_summary_for_rows(
    fingerprints: list[Any],
    prediction_indices: np.ndarray,
    train_indices: np.ndarray,
    *,
    leave_one_out: bool,
) -> np.ndarray:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    prediction_indices = np.asarray(prediction_indices, dtype=np.int64)
    output = np.zeros((len(prediction_indices), 7), dtype=np.float64)
    train_fps = [fingerprints[int(index)] for index in train_indices]
    for row, index in enumerate(prediction_indices):
        if len(train_fps) == 0:
            continue
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps), dtype=np.float64)
        if leave_one_out:
            same = np.flatnonzero(train_indices == int(index))
            if same.size:
                similarities = np.delete(similarities, int(same[0]))
        if similarities.size == 0:
            continue
        ordered = np.sort(similarities)[::-1]
        top3 = ordered[: min(3, len(ordered))]
        top10 = ordered[: min(10, len(ordered))]
        output[row] = [
            float(ordered[0]),
            float(np.mean(top3)),
            float(np.mean(top10)),
            float(np.std(top10)),
            float(np.mean(similarities >= 0.30)),
            float(np.mean(similarities >= 0.50)),
            float(np.mean(similarities >= 0.70)),
        ]
    return output


def support_pass(features: np.ndarray) -> np.ndarray:
    return (features[:, 0] >= SUPPORT_NEAREST_MIN) & (features[:, 1] >= SUPPORT_TOP3_MEAN_MIN)


def fit_preprocessor(train: np.ndarray, predict: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean_train = np.asarray(train, dtype=np.float64).copy()
    clean_predict = np.asarray(predict, dtype=np.float64).copy()
    clean_train[~np.isfinite(clean_train) | (np.abs(clean_train) > 1.0e12)] = np.nan
    clean_predict[~np.isfinite(clean_predict) | (np.abs(clean_predict) > 1.0e12)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(imputer.fit_transform(clean_train))
    predict_scaled = scaler.transform(imputer.transform(clean_predict))
    return train_scaled, predict_scaled


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, mask: np.ndarray, minimum: int = 8) -> float | None:
    if int(np.sum(mask)) < minimum or float(np.var(y[mask])) <= 1.0e-15:
        return None
    return float(r2_score(y[mask], candidate[mask]) - r2_score(y[mask], parent[mask]))


def transfer_panels(
    info: dict[str, Any],
    candidate: np.ndarray,
    fold_support: np.ndarray,
    support_gate: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    nearest = fold_support[:, 0]
    specs: dict[str, np.ndarray] = {
        "support_pass": support_gate,
        "support_fallback": ~support_gate,
        "similarity_lt_0.30": nearest < 0.30,
        "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
        "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
        "similarity_ge_0.70": nearest >= 0.70,
        "quantile_low": y <= np.quantile(y, 0.25),
        "quantile_high": y >= np.quantile(y, 0.75),
    }
    for name in sorted(set(scaffolds)):
        selected = scaffolds == name
        if int(np.sum(selected)) >= 10:
            specs[f"scaffold_{name}"] = selected
    panels: dict[str, Any] = {}
    values: list[float] = []
    for name, mask in specs.items():
        delta = panel_delta(y, parent, candidate, mask)
        panels[name] = {
            "rows": int(np.sum(mask)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable_insufficient_support",
        }
        if delta is not None:
            values.append(delta)
    support_strata = {
        "support_pass": panel_delta(y, parent, candidate, support_gate),
        "support_fallback": panel_delta(y, parent, candidate, ~support_gate),
    }
    stratum_values = [value for value in support_strata.values() if value is not None]
    return {
        "panels": panels,
        "minimum_panel_delta": float(min(values)) if values else 0.0,
        "minimum_support_stratum_delta": float(min(stratum_values)) if stratum_values else 0.0,
        "support_strata": support_strata,
        "nearest_tanimoto": nearest,
    }


def nc_test_rows(parent: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = parent["test"].loc[parent["test"]["target_type"].astype(str).eq(ACTIVE_TARGET)].sort_values("id").reset_index(drop=True)
    detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"].astype(str).eq(ACTIVE_TARGET)].sort_values("id").reset_index(drop=True)
    if not np.array_equal(rows["id"].to_numpy(int), detail["id"].to_numpy(int)):
        raise RuntimeError("C202 Nc test ID alignment failed")
    indices = np.asarray([parent["key_to_index"][value] for value in rows["canonical"]], dtype=np.int64)
    return rows, indices, detail["target"].to_numpy(np.float64)


def fit_nc_component(
    parent: dict[str, Any],
    features: np.ndarray,
    progress: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    info = parent["target_info"][ACTIVE_TARGET]
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    candidate = parent_oof.copy()
    raw_residual = np.zeros(len(y), dtype=np.float64)
    support_oof = np.zeros((len(y), 7), dtype=np.float64)
    gate_oof = np.zeros(len(y), dtype=bool)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_support = support_summary_for_rows(parent["fingerprints"], indices[training], indices[training], leave_one_out=True)
        valid_support = support_summary_for_rows(parent["fingerprints"], indices[validation], indices[training], leave_one_out=False)
        train_x = np.hstack([features[indices[training]], train_support])
        valid_x = np.hstack([features[indices[validation]], valid_support])
        train_scaled, valid_scaled = fit_preprocessor(train_x, valid_x)
        model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
        model.fit(train_scaled, y[training] - parent_oof[training])
        residual = model.predict(valid_scaled)
        raw = reference.clip_prediction(y[training], parent_oof[validation] + RESIDUAL_WEIGHT * residual)
        gate = support_pass(valid_support)
        candidate[validation] = np.where(gate, raw, parent_oof[validation])
        raw_residual[validation] = residual
        support_oof[validation] = valid_support
        gate_oof[validation] = gate
        fold_rows.append({
            "fold": int(fold),
            "rows": int(len(validation)),
            "support_pass_rows": int(np.sum(gate)),
            "support_fallback_rows": int(np.sum(~gate)),
            "parent_r2": float(r2_score(y[validation], parent_oof[validation])),
            "candidate_r2": float(r2_score(y[validation], candidate[validation])),
            "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent_oof[validation])),
        })
    panels = transfer_panels(info, candidate, support_oof, gate_oof)
    delta = float(r2_score(y, candidate) - r2_score(y, parent_oof))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    lower = float(carrier.bootstrap_lower(y, parent_oof, candidate, groups))
    passed = bool(
        delta >= MIN_BANKABLE_DELTA_R2
        and positive >= 4
        and lower > 0.0
        and panels["minimum_panel_delta"] >= 0.0
        and panels["minimum_support_stratum_delta"] >= 0.0
    )
    report = {
        "active": True,
        "parent_r2": float(r2_score(y, parent_oof)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_panel_delta": float(panels["minimum_panel_delta"]),
        "minimum_support_stratum_delta": float(panels["minimum_support_stratum_delta"]),
        "panels": panels["panels"],
        "support_strata": panels["support_strata"],
        "folds": fold_rows,
        "support_pass_rows": int(np.sum(gate_oof)),
        "support_fallback_rows": int(np.sum(~gate_oof)),
        "support_nearest_min": SUPPORT_NEAREST_MIN,
        "support_top3_mean_min": SUPPORT_TOP3_MEAN_MIN,
        "ridge_alpha": RIDGE_ALPHA,
        "residual_weight": RESIDUAL_WEIGHT,
        "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
        "pass": passed,
    }
    diagnostics = pd.DataFrame({
        "canonical": info["canonical"],
        "target": y,
        "parent": parent_oof,
        "candidate": candidate,
        "raw_residual": raw_residual,
        "support_gate": gate_oof,
        "nearest_tanimoto": support_oof[:, 0],
        "top3_mean_tanimoto": support_oof[:, 1],
        "top10_mean_tanimoto": support_oof[:, 2],
        "top10_std_tanimoto": support_oof[:, 3],
        "density_ge_030": support_oof[:, 4],
        "density_ge_050": support_oof[:, 5],
        "density_ge_070": support_oof[:, 6],
        "fold": folds,
    })
    checkpoint(
        progress,
        "nc_support_uncertainty_complete",
        delta_r2=delta,
        positive_folds=positive,
        group_bootstrap_lower=lower,
        minimum_panel_delta=report["minimum_panel_delta"],
        support_pass_rows=report["support_pass_rows"],
        pass_gate=passed,
    )
    return candidate, gate_oof, report, diagnostics


def full_nc_test_predictions(parent: dict[str, Any], features: np.ndarray, bank_active: bool) -> tuple[np.ndarray, pd.DataFrame]:
    rows, test_indices, test_parent = nc_test_rows(parent)
    if not bank_active:
        return test_parent, pd.DataFrame({
            "id": rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "parent": test_parent,
            "candidate": test_parent,
            "support_gate": np.zeros(len(rows), dtype=bool),
        })
    info = parent["target_info"][ACTIVE_TARGET]
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    train_indices = np.asarray(info["indices"], dtype=np.int64)
    train_support = support_summary_for_rows(parent["fingerprints"], train_indices, train_indices, leave_one_out=True)
    test_support = support_summary_for_rows(parent["fingerprints"], test_indices, train_indices, leave_one_out=False)
    train_x = np.hstack([features[train_indices], train_support])
    test_x = np.hstack([features[test_indices], test_support])
    train_scaled, test_scaled = fit_preprocessor(train_x, test_x)
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
    model.fit(train_scaled, y - parent_oof)
    residual = model.predict(test_scaled)
    raw = reference.clip_prediction(y, test_parent + RESIDUAL_WEIGHT * residual)
    gate = support_pass(test_support)
    candidate = np.where(gate, raw, test_parent)
    diagnostics = pd.DataFrame({
        "id": rows["id"].astype(int),
        "target_type": ACTIVE_TARGET,
        "parent": test_parent,
        "raw_candidate": raw,
        "candidate": candidate,
        "raw_residual": residual,
        "support_gate": gate,
        "nearest_tanimoto": test_support[:, 0],
        "top3_mean_tanimoto": test_support[:, 1],
        "top10_mean_tanimoto": test_support[:, 2],
        "top10_std_tanimoto": test_support[:, 3],
        "density_ge_030": test_support[:, 4],
        "density_ge_050": test_support[:, 5],
        "density_ge_070": test_support[:, 6],
    })
    return candidate, diagnostics


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

    features, feature_report = refractivity_features(parent)
    checkpoint(progress, "features_complete", rows=int(features.shape[0]), columns=int(features.shape[1]))

    active_candidate, support_gate, active_report, diagnostics = fit_nc_component(parent, features, progress)
    banked = [ACTIVE_TARGET] if active_report["pass"] else []

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = parent["target_info"][target]
        if target == ACTIVE_TARGET:
            report = active_report
            candidate = active_candidate
            support = support_gate
        else:
            report = unchanged_report(info)
            candidate = np.asarray(info["parent"], dtype=np.float64)
            support = np.zeros(len(candidate), dtype=bool)
        target_reports[target] = report
        assembled = candidate if target in banked else np.asarray(info["parent"], dtype=np.float64)
        oof_parts.append(pd.DataFrame({
            "canonical": info["canonical"],
            "target_type": target,
            "target": info["y"],
            "parent": info["parent"],
            "candidate": candidate,
            "assembled": assembled,
            "group": info["groups"],
            "scaffold": info["scaffolds"],
            "fold": carrier.grouped_folds(np.asarray(info["groups"], dtype=object)),
            "support_gate": support,
        }))

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    candidate_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(candidate_mean - parent_mean >= 0.002 and max_loss >= -0.003 and bool(banked))

    nc_test, test_diagnostics = full_nc_test_predictions(parent, features, bool(banked))
    predictions = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if ACTIVE_TARGET in banked:
        values = test_diagnostics.set_index("id")["candidate"]
        mask = predictions["target_type"].astype(str).eq(ACTIVE_TARGET)
        predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(values).to_numpy(np.float64)
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(nc_test) != int(np.sum(parent["test"]["target_type"].astype(str).eq(ACTIVE_TARGET))):
        raise RuntimeError("C202 Nc test prediction length mismatch")
    if (
        len(predictions) != 4940
        or not np.array_equal(predictions["id"].to_numpy(int), np.arange(1, 4941))
        or not np.isfinite(predictions["target"].to_numpy(np.float64)).all()
    ):
        raise RuntimeError("C202 complete output contract failed")

    report = {
        "schema_version": "ppp.round2.c202.nc-support-uncertainty-refractivity.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "pi1m_used": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "active_target": ACTIVE_TARGET,
        "changed_factor": "Nc-only low-variance Ridge residual over conformer-free refractivity graph-shell features plus fold-local label-free support-density gate; exact C050 fallback outside support.",
        "not_reused_families": {
            "c195_c197_nc_arms": False,
            "pi1m_density": False,
            "predicted_eps_to_nc": False,
            "cross_property_stage2": False,
            "local_eval_or_public_feedback": False,
        },
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_pass,
        "goal_0_95_met": bool(full_pass and candidate_mean >= 0.95),
        "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    oof.loc[oof["target_type"].astype(str).eq(ACTIVE_TARGET), ["canonical", "target", "parent", "candidate"]].to_csv(
        run_dir / "nc_oof_predictions.csv",
        index=False,
    )
    diagnostics.to_csv(run_dir / "component_oof_diagnostics.csv", index=False)
    test_diagnostics.to_csv(run_dir / "component_test_diagnostics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": "ppp.round2.c202.nc-support-uncertainty-refractivity.v1",
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "ridge_alpha": RIDGE_ALPHA,
            "residual_weight": RESIDUAL_WEIGHT,
            "support_nearest_min": SUPPORT_NEAREST_MIN,
            "support_top3_mean_min": SUPPORT_TOP3_MEAN_MIN,
            "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
            "selection": "fixed support gate and fixed residual weight; no threshold, alpha, feature, or shell retune",
            "official_only": True,
            "local_eval_read": False,
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
                f"rdkit={reference.Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: **{report['decision']}**. "
        f"Nc delta `{active_report['delta_r2']:+.12f}` with support-pass rows `{active_report['support_pass_rows']}` "
        f"and support-fallback rows `{active_report['support_fallback_rows']}`. "
        f"Mean parent `{parent_mean:.12f}`; assembled `{candidate_mean:.12f}`. "
        "No local_eval, Kaggle compute, upload, or submission action.\n\n"
        "If any gate fails, this Nc support/uncertainty refractivity branch is cooled without retuning.\n",
        encoding="utf-8",
    )
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "banked_targets": banked,
                "nc_delta_r2": active_report["delta_r2"],
                "nc_positive_folds": active_report["positive_folds"],
                "nc_group_bootstrap_lower": active_report["group_bootstrap_lower"],
                "nc_minimum_panel_delta": active_report["minimum_panel_delta"],
                "support_pass_rows": active_report["support_pass_rows"],
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": candidate_mean,
                "mean_gain": candidate_mean - parent_mean,
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
