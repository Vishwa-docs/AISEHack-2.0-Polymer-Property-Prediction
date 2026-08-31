#!/usr/bin/env python3
"""C220: Ei electro-polar topological autocorrelation residual.

This is a bounded official-only Ei child queued after C219.  It tests one
factor: a fixed low-dimensional atom-channel autocorrelation representation for
Ei residuals.  It does not tune lag depth, alpha, residual weight, folds, model
class, panels, or fallback slices.  C199 is used only as a selected-reference
guard before any assembler may consume C220.
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
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, EState, rdMolDescriptors, rdPartialCharges
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c200_clean_component_compound_audit_v3 as c200


RDLogger.DisableLog("rdApp.*")

TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "ei"
SCHEMA = "ppp.round2.c220.ei-electro-polar-autocorr.v1"
SEED = 20260822
MAX_LAG = 6
RIDGE_ALPHA = 30.0
RESIDUAL_WEIGHT = 0.30
MIN_C050_DELTA = 0.010
MIN_SELECTED_REFERENCE_DELTA = 0.010
C199_REFERENCE_EI_R2 = 0.8566558157138717


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
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
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


def capped_molecule(smiles: str) -> Chem.Mol | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    try:
        capped = Chem.RWMol(Chem.Mol(molecule))
        for atom in capped.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(6)
                atom.SetFormalCharge(0)
                atom.SetNoImplicit(False)
        result = capped.GetMol()
        Chem.SanitizeMol(result)
        return result
    except Exception:
        return None


def atom_channels(molecule: Chem.Mol) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    try:
        rdPartialCharges.ComputeGasteigerCharges(molecule)
        charges = np.asarray([float(atom.GetProp("_GasteigerCharge")) for atom in molecule.GetAtoms()], dtype=np.float64)
        estate = np.asarray(EState.EStateIndices(molecule), dtype=np.float64)
        crippen = Crippen._GetAtomContribs(molecule)
        logp = np.asarray([float(item[0]) for item in crippen], dtype=np.float64)
        mr = np.asarray([float(item[1]) for item in crippen], dtype=np.float64)
        atoms = list(molecule.GetAtoms())
        atomic = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=np.float64)
        degree = np.asarray([atom.GetDegree() for atom in atoms], dtype=np.float64)
        total_valence = np.asarray([atom.GetTotalValence() for atom in atoms], dtype=np.float64)
        formal_charge = np.asarray([atom.GetFormalCharge() for atom in atoms], dtype=np.float64)
        aromatic = np.asarray([float(atom.GetIsAromatic()) for atom in atoms], dtype=np.float64)
        sp2 = np.asarray([float(atom.GetHybridization() == Chem.HybridizationType.SP2) for atom in atoms], dtype=np.float64)
        hetero = np.asarray([float(atom.GetAtomicNum() not in (1, 6)) for atom in atoms], dtype=np.float64)
        ring = np.asarray([float(atom.IsInRing()) for atom in atoms], dtype=np.float64)
        matrix = np.column_stack(
            [
                atomic / 20.0,
                degree / 4.0,
                total_valence / 6.0,
                formal_charge,
                charges,
                np.abs(charges),
                estate / 10.0,
                logp,
                mr / 10.0,
                aromatic,
                sp2,
                hetero,
                ring,
            ]
        )
        names = [
            "atomic_z_scaled",
            "degree_scaled",
            "valence_scaled",
            "formal_charge",
            "gasteiger_charge",
            "abs_gasteiger_charge",
            "estate_scaled",
            "crippen_logp",
            "crippen_mr_scaled",
            "aromatic",
            "sp2",
            "hetero",
            "ring",
        ]
        if not np.isfinite(matrix).all():
            return None
        distances = np.asarray(Chem.GetDistanceMatrix(molecule, useBO=False), dtype=np.float64)
        return matrix, names, distances
    except Exception:
        return None


def autocorr_features_for_smiles(smiles: str) -> tuple[np.ndarray, list[str], bool]:
    molecule = capped_molecule(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        names = autocorr_feature_names()
        return np.full(len(names), np.nan, dtype=np.float64), names, False
    channel_result = atom_channels(molecule)
    names = autocorr_feature_names()
    if channel_result is None:
        return np.full(len(names), np.nan, dtype=np.float64), names, False
    channels, channel_names, distances = channel_result
    values: list[float] = []
    feature_names: list[str] = []
    heavy = max(float(molecule.GetNumHeavyAtoms()), 1.0)
    bonds = max(float(molecule.GetNumBonds()), 1.0)
    rings = float(molecule.GetRingInfo().NumRings())
    values.extend(
        [
            heavy,
            np.log1p(heavy),
            float(molecule.GetNumBonds()),
            float(molecule.GetNumBonds()) / heavy,
            rings,
            rings / heavy,
            float(rdMolDescriptors.CalcTPSA(molecule)) / heavy,
            float(Crippen.MolMR(molecule)) / heavy,
            float(Crippen.MolLogP(molecule)) / heavy,
            float(Descriptors.BertzCT(molecule)) / heavy,
        ]
    )
    feature_names.extend(
        [
            "heavy_atoms",
            "log1p_heavy_atoms",
            "bond_count",
            "bond_density",
            "ring_count",
            "ring_density",
            "tpsa_density",
            "molmr_density",
            "logp_density",
            "bertz_density",
        ]
    )
    for column, channel_name in enumerate(channel_names):
        vector = channels[:, column]
        values.extend([float(np.mean(vector)), float(np.std(vector)), float(np.min(vector)), float(np.max(vector))])
        feature_names.extend(
            [
                f"{channel_name}_mean",
                f"{channel_name}_std",
                f"{channel_name}_min",
                f"{channel_name}_max",
            ]
        )
    centered = channels - np.mean(channels, axis=0, keepdims=True)
    upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
    for lag in range(1, MAX_LAG + 1):
        mask = upper & (distances == float(lag))
        pair_count = int(np.sum(mask))
        values.extend([float(pair_count), float(pair_count / max(bonds, 1.0))])
        feature_names.extend([f"lag{lag}_pair_count", f"lag{lag}_pair_density"])
        rows, cols = np.where(mask)
        for column, channel_name in enumerate(channel_names):
            if pair_count == 0:
                product = 0.0
                absdiff = 0.0
                sqdiff = 0.0
            else:
                left = centered[rows, column]
                right = centered[cols, column]
                product = float(np.mean(left * right))
                diff = channels[rows, column] - channels[cols, column]
                absdiff = float(np.mean(np.abs(diff)))
                sqdiff = float(np.mean(diff * diff))
            values.extend([product, absdiff, sqdiff])
            feature_names.extend(
                [
                    f"lag{lag}_{channel_name}_centered_product",
                    f"lag{lag}_{channel_name}_absdiff",
                    f"lag{lag}_{channel_name}_sqdiff",
                ]
            )
    if feature_names != names:
        raise RuntimeError("C220 autocorrelation feature schema drift")
    return np.asarray(values, dtype=np.float64), names, True


def autocorr_feature_names() -> list[str]:
    channel_names = [
        "atomic_z_scaled",
        "degree_scaled",
        "valence_scaled",
        "formal_charge",
        "gasteiger_charge",
        "abs_gasteiger_charge",
        "estate_scaled",
        "crippen_logp",
        "crippen_mr_scaled",
        "aromatic",
        "sp2",
        "hetero",
        "ring",
    ]
    names = [
        "heavy_atoms",
        "log1p_heavy_atoms",
        "bond_count",
        "bond_density",
        "ring_count",
        "ring_density",
        "tpsa_density",
        "molmr_density",
        "logp_density",
        "bertz_density",
    ]
    for channel_name in channel_names:
        names.extend(
            [
                f"{channel_name}_mean",
                f"{channel_name}_std",
                f"{channel_name}_min",
                f"{channel_name}_max",
            ]
        )
    for lag in range(1, MAX_LAG + 1):
        names.extend([f"lag{lag}_pair_count", f"lag{lag}_pair_density"])
        for channel_name in channel_names:
            names.extend(
                [
                    f"lag{lag}_{channel_name}_centered_product",
                    f"lag{lag}_{channel_name}_absdiff",
                    f"lag{lag}_{channel_name}_sqdiff",
                ]
            )
    return names


def build_autocorr_features(smiles: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    names: list[str] | None = None
    valid = 0
    for value in smiles:
        row, row_names, ok = autocorr_features_for_smiles(value)
        if names is None:
            names = row_names
        elif names != row_names:
            raise RuntimeError("C220 inconsistent feature names")
        rows.append(row)
        valid += int(ok)
    if names is None:
        names = autocorr_feature_names()
    matrix = np.vstack(rows).astype(np.float64)
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)] = np.nan
    return matrix, {
        "feature_family": "fixed RDKit electro-polar topological autocorrelation",
        "feature_count": int(matrix.shape[1]),
        "rows": int(matrix.shape[0]),
        "valid_molecule_rows": int(valid),
        "invalid_molecule_rows": int(matrix.shape[0] - valid),
        "max_lag": MAX_LAG,
        "atom_channels": [
            "atomic_z",
            "degree",
            "valence",
            "formal_charge",
            "gasteiger_charge",
            "abs_gasteiger_charge",
            "estate",
            "crippen_logp",
            "crippen_mr",
            "aromatic",
            "sp2",
            "hetero",
            "ring",
        ],
        "feature_names": names,
    }


def residual_model() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4),
    )


def fit_ei_residual(
    info: dict[str, Any],
    features: np.ndarray,
    test_indices: np.ndarray,
    test_parent: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        model = residual_model()
        model.fit(features[indices[training]], y[training] - parent[training])
        residual = np.asarray(model.predict(features[indices[validation]]), dtype=float)
        lower_clip = float(np.quantile(y[training], 0.005) - 0.10)
        upper_clip = float(np.quantile(y[training], 0.995) + 0.10)
        candidate[validation] = np.clip(parent[validation] + RESIDUAL_WEIGHT * residual, lower_clip, upper_clip)
        fold_parent = float(r2_score(y[validation], parent[validation]))
        fold_candidate = float(r2_score(y[validation], candidate[validation]))
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(len(validation)),
                "parent_r2": fold_parent,
                "candidate_r2": fold_candidate,
                "delta_r2": fold_candidate - fold_parent,
            }
        )
    if not np.isfinite(candidate).all():
        raise RuntimeError("C220 produced non-finite Ei OOF predictions")
    full_model = residual_model()
    full_model.fit(features[indices], y - parent)
    test_residual = np.asarray(full_model.predict(features[test_indices]), dtype=float)
    lower_clip = float(np.quantile(y, 0.005) - 0.10)
    upper_clip = float(np.quantile(y, 0.995) + 0.10)
    test_candidate = np.clip(np.asarray(test_parent, dtype=float) + RESIDUAL_WEIGHT * test_residual, lower_clip, upper_clip)
    return {"candidate": candidate, "test_candidate": test_candidate, "folds": fold_rows}


def selected_ei_reference(root: Path) -> dict[str, Any]:
    for run_id in (
        "R2-C219-20260805-0500-clean-component-compound-audit-v11",
        "R2-C217-20260805-0450-clean-component-compound-audit-v10",
        "R2-C215-20260805-0440-clean-component-compound-audit-v9",
        "R2-C213-20260805-0422-clean-component-compound-audit-v8",
        "R2-C211-20260805-0419-clean-component-compound-audit-v7",
        "R2-C209-20260805-0406-clean-component-compound-audit-v6",
        "R2-C200-20260805-0301-clean-component-compound-audit-v3",
    ):
        metrics = c200.load_json(c200.run_dir(root, run_id) / "metrics.json")
        if not isinstance(metrics, dict):
            continue
        selected = metrics.get("selected_components", {}).get(ACTIVE_TARGET, {})
        if isinstance(selected, dict) and selected.get("candidate_r2") is not None:
            return {
                "reference_source": f"{run_id}_selected_component",
                "run_id": selected.get("run_id"),
                "candidate_r2": float(selected.get("candidate_r2")),
            }
    c199 = c200.load_json(c200.run_dir(root, "R2-C199-20260805-0254-ei-c196-transfer-guard-v1") / "metrics.json")
    if isinstance(c199, dict):
        report = c199.get("target_reports", {}).get(ACTIVE_TARGET, {})
        if isinstance(report, dict) and report.get("candidate_r2") is not None:
            return {
                "reference_source": "c199_target_report_fallback",
                "run_id": "R2-C199-20260805-0254-ei-c196-transfer-guard-v1",
                "candidate_r2": float(report.get("candidate_r2")),
            }
    return {
        "reference_source": "constant_c199_reference_fallback",
        "run_id": "R2-C199-20260805-0254-ei-c196-transfer-guard-v1",
        "candidate_r2": C199_REFERENCE_EI_R2,
    }


def target_test_indices(parent: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_rows = (
        parent["test"]
        .loc[parent["test"]["target_type"].astype(str).eq(ACTIVE_TARGET)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"].astype(str).eq(ACTIVE_TARGET)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
        raise RuntimeError("C220 Ei test ID alignment failed")
    indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    return test_rows, indices, test_detail["target"].to_numpy(float)


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

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    canonical_run = Path(args.canonical_run)
    if not canonical_run.is_absolute():
        canonical_run = (root / canonical_run).resolve()

    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, canonical_run)
    checkpoint(progress, "parent_parity", **parity)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    features, feature_report = build_autocorr_features(list(parent["keys"]))
    checkpoint(progress, "features_complete", feature_count=feature_report["feature_count"], valid_rows=feature_report["valid_molecule_rows"])

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = target_test_indices(parent)
    result = fit_ei_residual(info, features, test_indices, test_parent)
    target_report = carrier.evaluate_target(info, {"candidate": result["candidate"]})
    target_report["folds"] = result["folds"]
    reference_info = selected_ei_reference(root)
    selected_reference_r2 = float(reference_info["candidate_r2"])
    delta_vs_selected = float(target_report["candidate_r2"] - selected_reference_r2)
    replacement_gate = bool(target_report["pass"] and delta_vs_selected >= MIN_SELECTED_REFERENCE_DELTA)
    target_report.update(
        {
            "changed_factor": "fixed Ei electro-polar topological autocorrelation Ridge residual",
            "model_family": "Ridge residual over fixed atom-channel graph-distance autocorrelation features",
            "ridge_alpha": RIDGE_ALPHA,
            "residual_weight": RESIDUAL_WEIGHT,
            "max_lag": MAX_LAG,
            "normal_component_gate_pass": bool(target_report["pass"]),
            "selected_ei_reference": reference_info,
            "delta_vs_selected_ei_reference": delta_vs_selected,
            "beats_selected_ei_reference_gate": bool(delta_vs_selected >= MIN_SELECTED_REFERENCE_DELTA),
            "replacement_gate_pass": replacement_gate,
            "minimum_c050_delta": MIN_C050_DELTA,
            "minimum_selected_reference_delta": MIN_SELECTED_REFERENCE_DELTA,
            "no_lag_grid": True,
            "no_alpha_grid": True,
            "no_residual_weight_grid": True,
            "no_cross_target_labels": True,
        }
    )
    banked = [ACTIVE_TARGET] if replacement_gate else []

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = target_report
            candidate = result["candidate"]
        else:
            report = unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=float)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": candidate,
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": folds,
                    "assembled": candidate if target in banked else target_info["parent"],
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    predictions = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if banked:
        mask = predictions["target_type"].astype(str).eq(ACTIVE_TARGET)
        predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(
            pd.Series(result["test_candidate"], index=test_rows["id"].astype(int))
        ).to_numpy(float)
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C220 ID coverage/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C220 produced non-finite predictions")

    report = {
        "schema_version": SCHEMA,
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "pi1m_used": False,
        "stored_prediction_replay": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "feature_report": {key: value for key, value in feature_report.items() if key != "feature_names"},
        "feature_names_sha256": hashlib.sha256("\n".join(feature_report["feature_names"]).encode("utf-8")).hexdigest(),
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": bool(assembled_mean - parent_mean >= 0.002 and bool(banked)),
        "goal_0_95_met": bool(assembled_mean >= 0.95 and bool(banked)),
        "decision": "candidate_pass_pending_clean_reproduction" if banked else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "compound_audit_helpers": sha256_file(round2_root / "tools/round2_c200_clean_component_compound_audit_v3.py"),
        },
    }
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "canonical": info["canonical"],
            "target": info["y"],
            "parent": info["parent"],
            "candidate": result["candidate"],
        }
    ).to_csv(run_dir / "ei_oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "parent": test_parent,
            "candidate": result["test_candidate"],
        }
    ).to_csv(run_dir / "ei_component_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "max_lag": MAX_LAG,
            "ridge_alpha": RIDGE_ALPHA,
            "residual_weight": RESIDUAL_WEIGHT,
            "minimum_c050_delta": MIN_C050_DELTA,
            "minimum_selected_reference_delta": MIN_SELECTED_REFERENCE_DELTA,
            "selected_reference": reference_info,
            "selection_rule": "one fixed Ei electro-polar autocorrelation residual; no lag/alpha/weight/model grid; no local_eval/public feedback",
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
                f"rdkit={reference.Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: `{report['decision']}`. Ei parent `{target_report['parent_r2']:.12f}`; "
        f"candidate `{target_report['candidate_r2']:.12f}`; C050-relative delta "
        f"`{target_report['delta_r2']:+.12f}`; selected-reference delta "
        f"`{target_report['delta_vs_selected_ei_reference']:+.12f}`. Mean parent "
        f"`{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; gain "
        f"`{assembled_mean - parent_mean:+.12f}`. Official-only; no local_eval read; "
        "no Kaggle action.\n",
        encoding="utf-8",
    )
    manifest = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    checkpoint(
        progress,
        "metrics_written",
        decision=report["decision"],
        ei_delta=target_report["delta_r2"],
        delta_vs_selected=target_report["delta_vs_selected_ei_reference"],
    )
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "banked_targets": banked,
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
                "ei_delta": target_report["delta_r2"],
                "delta_vs_selected_ei_reference": target_report["delta_vs_selected_ei_reference"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
