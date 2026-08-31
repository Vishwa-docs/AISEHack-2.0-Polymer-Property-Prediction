#!/usr/bin/env python3
"""C254: Tg backbone/pendant rigidity residual.

This is a fresh continuation after C253 if the 0.95 objective is still unmet.
It targets the unbanked Tg component with a single label-free structural factor:
wildcard-to-wildcard backbone versus pendant rigidity.  It is deliberately not
a C244/C228/C232 residual-stack retune and does not read stored predictions.

The fixed deployment support is strict: apply the residual only when the repeat
unit has an unambiguous wildcard backbone and the held-out/test row has nearest
same-target training Tanimoto similarity at least 0.30.  C050 is the exact
fallback everywhere else.
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
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c208_tg_robust_group_measurement as c208
import round2_c236_nc_backbone_pendant_polarizability as c236


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "tg"
SCHEMA = "ppp.round2.c254.tg-backbone-pendant-rigidity.v1"
SEED = 20260805
RIDGE_ALPHA = 80.0
TREE_COUNT = 220
TREE_MIN_LEAF = 5
RESIDUAL_WEIGHT = 0.20
RIDGE_BLEND_WEIGHT = 0.70
TREE_BLEND_WEIGHT = 0.30
SUPPORT_SIMILARITY_FLOOR = 0.30
MIN_FULL_MEAN_GAIN = 0.002

MOTIF_SMARTS = {
    "sulfone": "S(=O)(=O)",
    "imide": "C(=O)N(C=O)",
    "amide": "C(=O)N",
    "ether": "[OD2]([#6])[#6]",
    "carbonate": "[OD2][CX3](=[OX1])[OD2]",
    "siloxane": "[Si][OD2][Si]",
    "phenyl": "c1ccccc1",
    "alkene": "C=C",
    "nitrile": "C#N",
}
MOTIFS = {name: Chem.MolFromSmarts(smarts) for name, smarts in MOTIF_SMARTS.items()}


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


def ring_fusion_counts(molecule: Chem.Mol) -> tuple[float, float]:
    rings = [set(ring) for ring in molecule.GetRingInfo().AtomRings()]
    fused = 0
    for i, left in enumerate(rings):
        for right in rings[i + 1 :]:
            if len(left & right) >= 2:
                fused += 1
    hetero_rings = 0
    for ring in rings:
        if any(molecule.GetAtomWithIdx(int(idx)).GetAtomicNum() not in (6, 1) for idx in ring):
            hetero_rings += 1
    return float(fused), float(hetero_rings)


def safe_has_substruct(molecule: Chem.Mol, pattern: Chem.Mol | None) -> float:
    if pattern is None:
        return 0.0
    try:
        return float(molecule.HasSubstructMatch(pattern))
    except Exception:
        return 0.0


def rigidity_feature_row(molecule: Chem.Mol) -> tuple[list[float], dict[str, Any]]:
    backbone, pendant, meta = c236.partition_masks(molecule)
    atoms = list(molecule.GetAtoms())
    n_atoms = len(atoms)
    real = np.asarray([atom.GetAtomicNum() > 0 for atom in atoms], dtype=bool)
    heavy = max(int(np.sum(real)), 1)
    pendant_count = int(np.sum(pendant))
    backbone_count = int(np.sum(backbone))
    aromatic = np.asarray([atom.GetIsAromatic() for atom in atoms], dtype=np.float64)
    sp2 = np.asarray(
        [atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2 for atom in atoms],
        dtype=np.float64,
    )
    hetero = np.asarray([atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms], dtype=np.float64)
    degree = np.asarray([atom.GetDegree() for atom in atoms], dtype=np.float64)
    try:
        contribs = Crippen._GetAtomContribs(molecule)
        logp = np.asarray([item[0] for item in contribs], dtype=np.float64)
        mr = np.asarray([item[1] for item in contribs], dtype=np.float64)
    except Exception:
        logp = np.zeros(n_atoms, dtype=np.float64)
        mr = np.zeros(n_atoms, dtype=np.float64)
    fused, hetero_rings = ring_fusion_counts(molecule)
    try:
        rings = float(rdMolDescriptors.CalcNumRings(molecule))
        aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(molecule))
        aliphatic_rings = float(rdMolDescriptors.CalcNumAliphaticRings(molecule))
        rotatable = float(Descriptors.NumRotatableBonds(molecule))
        bridgeheads = float(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule))
        spiro = float(rdMolDescriptors.CalcNumSpiroAtoms(molecule))
        fraction_csp3 = float(Descriptors.FractionCSP3(molecule))
    except Exception:
        rings = aromatic_rings = aliphatic_rings = rotatable = bridgeheads = spiro = fraction_csp3 = 0.0

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else 0.0

    def masked_sum(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.sum(values[mask])) if np.any(mask) else 0.0

    backbone_mr = masked_sum(mr, backbone)
    pendant_mr = masked_sum(mr, pendant)
    backbone_logp = masked_sum(logp, backbone)
    pendant_logp = masked_sum(logp, pendant)
    total_mr = max(abs(masked_sum(mr, real)), 1.0e-12)
    motif_values = [safe_has_substruct(molecule, MOTIFS[name]) for name in sorted(MOTIFS)]
    row = [
        float(meta["dummy_atom_count"]),
        float(meta["wildcard_path_length"]),
        float(backbone_count),
        float(pendant_count),
        float(backbone_count / heavy),
        float(pendant_count / heavy),
        float(rotatable / heavy),
        float(rings / heavy),
        float(aromatic_rings / heavy),
        float(aliphatic_rings / heavy),
        float(fused / max(rings, 1.0)),
        float(hetero_rings / max(rings, 1.0)),
        float(bridgeheads / heavy),
        float(spiro / heavy),
        float(fraction_csp3),
        masked_mean(aromatic, real),
        masked_mean(aromatic, backbone),
        masked_mean(aromatic, pendant),
        masked_mean(sp2, real),
        masked_mean(sp2, backbone),
        masked_mean(sp2, pendant),
        masked_mean(hetero, backbone),
        masked_mean(hetero, pendant),
        masked_mean(degree, backbone),
        masked_mean(degree, pendant),
        float(backbone_mr / total_mr),
        float(pendant_mr / total_mr),
        float(backbone_mr - pendant_mr),
        float(abs(backbone_mr - pendant_mr)),
        float(backbone_logp - pendant_logp),
        float(abs(backbone_logp - pendant_logp)),
        float((aromatic_rings + fused + bridgeheads + spiro) / max(rotatable + pendant_count, 1.0)),
        float((backbone_count + aromatic_rings + fused) / max(pendant_count + rotatable + 1.0, 1.0)),
        *motif_values,
    ]
    meta.update(
        {
            "unambiguous_backbone": bool(
                meta["dummy_atom_count"] >= 2
                and meta["wildcard_path_length"] > 0
                and backbone_count >= 2
            ),
            "tg_rigidity_feature_count": len(row),
            "crippen_role": "limited backbone_vs_pendant aggregate contrasts only; no C236 full polarizability feature block",
        }
    )
    return [float(value) for value in row], meta


def rigidity_features(parent: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[list[float]] = []
    metas: list[dict[str, Any]] = []
    failures = 0
    for molecule in parent["molecules"]:
        try:
            row, meta = rigidity_feature_row(molecule)
        except Exception:
            failures += 1
            row = [0.0] * 42
            meta = {"feature_failure": True, "unambiguous_backbone": False}
        rows.append(row)
        metas.append(meta)
    width = max(len(row) for row in rows)
    matrix = np.asarray([row + [0.0] * (width - len(row)) for row in rows], dtype=np.float64)
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)] = np.nan
    unambiguous = np.asarray([bool(item.get("unambiguous_backbone", False)) for item in metas], dtype=bool)
    pendant_counts = np.asarray([item.get("pendant_atom_count", 0) for item in metas], dtype=np.float64)
    return matrix, unambiguous, {
        "feature_family": "tg_wildcard_backbone_pendant_rigidity",
        "shape": [int(value) for value in matrix.shape],
        "feature_count": int(matrix.shape[1]),
        "feature_failures": int(failures),
        "unambiguous_backbone_rows": int(np.sum(unambiguous)),
        "support_definition": "unambiguous wildcard backbone and fold-local nearest same-target Tanimoto >= 0.30",
        "pendant_rows": int(np.sum(pendant_counts > 0)),
        "pendant_atom_mean": float(np.mean(pendant_counts)),
        "motifs": sorted(MOTIFS),
        "uses_labels": False,
        "uses_cross_property_labels": False,
        "uses_pi1m": False,
        "uses_stored_predictions": False,
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
                max_features=0.75,
                n_jobs=1,
            ),
        ).fit(x_train, y_train),
    ]


def predict_blend(models: list[Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ridge = np.asarray(models[0].predict(x), dtype=np.float64)
    tree = np.asarray(models[1].predict(x), dtype=np.float64)
    direct = RIDGE_BLEND_WEIGHT * ridge + TREE_BLEND_WEIGHT * tree
    return direct, np.column_stack([ridge, tree])


def nearest_to_training(info: dict[str, Any], query_rows: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    train_fps = [info["fingerprints"][int(info["indices"][row])] for row in train_rows]
    nearest = np.zeros(len(query_rows), dtype=np.float64)
    if not train_fps:
        return nearest
    for out_idx, row in enumerate(query_rows):
        nearest[out_idx] = max(
            reference.DataStructs.BulkTanimotoSimilarity(
                info["fingerprints"][int(info["indices"][row])],
                train_fps,
            )
        )
    return nearest


def fit_tg_rigidity(
    info: dict[str, Any],
    features: np.ndarray,
    unambiguous: np.ndarray,
    test_indices: np.ndarray,
    test_parent: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    candidate = parent.copy()
    direct_oof = np.full((len(y), 2), np.nan, dtype=np.float64)
    support_mask = np.zeros(len(y), dtype=bool)
    nearest_oof = np.zeros(len(y), dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_support = training[unambiguous[indices[training]]]
        if len(train_support) < 50:
            train_support = training
        nearest = nearest_to_training(info, validation, training)
        validation_support = unambiguous[indices[validation]] & (nearest >= SUPPORT_SIMILARITY_FLOOR)
        nearest_oof[validation] = nearest
        support_mask[validation] = validation_support
        residual = y[train_support] - parent[train_support]
        models = fit_models(features[indices[train_support]], residual)
        correction, direct = predict_blend(models, features[indices[validation]])
        direct_oof[validation] = direct
        fold_candidate = parent[validation].copy()
        if np.any(validation_support):
            fold_candidate[validation_support] = reference.clip_prediction(
                y[train_support],
                parent[validation][validation_support] + RESIDUAL_WEIGHT * correction[validation_support],
            )
        candidate[validation] = fold_candidate
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "train_rows": int(len(training)),
                "train_support_rows": int(len(train_support)),
                "validation_support_rows": int(np.sum(validation_support)),
                "validation_fallback_rows": int(np.sum(~validation_support)),
                "nearest_min": float(np.min(nearest)) if len(nearest) else 0.0,
                "nearest_mean": float(np.mean(nearest)) if len(nearest) else 0.0,
                "candidate_delta_r2": float(r2_score(y[validation], fold_candidate) - r2_score(y[validation], parent[validation])),
            }
        )
    if not np.isfinite(candidate).all():
        raise RuntimeError("C254 produced non-finite OOF candidate")
    full_support = np.flatnonzero(unambiguous[indices])
    if len(full_support) < 50:
        full_support = np.arange(len(y), dtype=np.int64)
    full_models = fit_models(features[indices[full_support]], y[full_support] - parent[full_support])
    train_fps = [info["fingerprints"][int(idx)] for idx in indices]
    test_nearest = np.zeros(len(test_indices), dtype=np.float64)
    for out_idx, idx in enumerate(test_indices):
        test_nearest[out_idx] = max(reference.DataStructs.BulkTanimotoSimilarity(info["fingerprints"][int(idx)], train_fps))
    test_support = unambiguous[test_indices] & (test_nearest >= SUPPORT_SIMILARITY_FLOOR)
    test_correction, _ = predict_blend(full_models, features[test_indices])
    test_candidate = test_parent.copy()
    if np.any(test_support):
        test_candidate[test_support] = reference.clip_prediction(
            y[full_support],
            test_parent[test_support] + RESIDUAL_WEIGHT * test_correction[test_support],
        )
    return {
        "candidate": candidate,
        "test_candidate": test_candidate,
        "direct_oof": direct_oof,
        "support_mask": support_mask,
        "nearest_oof": nearest_oof,
        "test_support_mask": test_support,
        "test_nearest": test_nearest,
        "blend_name": "support_gated_fixed_0.20_residual__0.70_ridge_0.30_extratrees",
        "weights": [RIDGE_BLEND_WEIGHT, TREE_BLEND_WEIGHT],
        "intercept": 0.0,
        "blend_r2": float(r2_score(y, candidate)),
        "fold_robust_reports": fold_reports,
        "full_robust_report": {
            "train_rows": int(len(y)),
            "full_support_rows": int(len(full_support)),
            "test_rows": int(len(test_indices)),
            "test_support_rows": int(np.sum(test_support)),
            "test_fallback_rows": int(np.sum(~test_support)),
            "support_similarity_floor": SUPPORT_SIMILARITY_FLOOR,
            "residual_weight": RESIDUAL_WEIGHT,
            "ridge_alpha": RIDGE_ALPHA,
            "tree_count": TREE_COUNT,
            "tree_min_leaf": TREE_MIN_LEAF,
        },
    }


def add_support_panels(info: dict[str, Any], result: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    candidate = np.asarray(result["candidate"], dtype=np.float64)
    support = np.asarray(result["support_mask"], dtype=bool)
    nearest = np.asarray(result["nearest_oof"], dtype=np.float64)
    panel_specs = {
        "support_unambiguous_nearest_ge_0.30": support,
        "fallback_or_unsupported": ~support,
        "support_similarity_0.30_0.50": support & (nearest < 0.50),
        "support_similarity_ge_0.50": support & (nearest >= 0.50),
    }
    support_panel_values: list[float] = []
    report = dict(report)
    panels = dict(report.get("panels", {}))
    for name, selected in panel_specs.items():
        delta = carrier.panel_delta(y, parent, candidate, selected)
        panels[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable",
        }
        if delta is not None:
            support_panel_values.append(delta)
    all_values = [
        value["delta_r2"]
        for value in panels.values()
        if isinstance(value, dict) and value.get("delta_r2") is not None
    ]
    minimum_panel = float(min(all_values)) if all_values else 0.0
    support_minimum = float(min(support_panel_values)) if support_panel_values else 0.0
    report["panels"] = panels
    report["minimum_panel_delta"] = minimum_panel
    report["support_panel_minimum_delta"] = support_minimum
    report["support_rows"] = int(np.sum(support))
    report["fallback_rows"] = int(np.sum(~support))
    report["pass"] = bool(
        report["delta_r2"] >= report["minimum_bankable_delta_r2"]
        and report["positive_folds"] >= 4
        and report["group_bootstrap_lower"] > 0.0
        and minimum_panel >= 0.0
        and support_minimum >= 0.0
    )
    return report


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

    features, unambiguous, feature_report = rigidity_features(parent)
    checkpoint(
        progress,
        "features_complete",
        shape=feature_report["shape"],
        unambiguous_backbone_rows=feature_report["unambiguous_backbone_rows"],
    )

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = c208.target_test_rows(parent, ACTIVE_TARGET)
    result = fit_tg_rigidity(info, features, unambiguous, test_indices, test_parent)
    active_report = add_support_panels(info, result, c208.evaluate_tg(info, result))
    active_report.update(
        {
            "changed_factor": "Tg backbone/pendant rigidity support-gated residual",
            "uses_backbone_pendant_partition": True,
            "uses_tg_rigidity_motifs": True,
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
            "c244_c228_c232_residual_stack_not_reused": True,
        }
    )
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "tg_backbone_pendant_rigidity_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        pass_gate=banked,
        positive_folds=active_report["positive_folds"],
        support_rows=active_report["support_rows"],
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
            support = np.asarray(result["support_mask"], dtype=bool)
            nearest = np.asarray(result["nearest_oof"], dtype=np.float64)
        else:
            report = c208.unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=np.float64)
            direct_oof = np.full((len(candidate), 2), np.nan, dtype=np.float64)
            support = np.zeros(len(candidate), dtype=bool)
            nearest = np.full(len(candidate), np.nan, dtype=np.float64)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        assembled = candidate if target == ACTIVE_TARGET and banked else np.asarray(target_info["parent"], dtype=np.float64)
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": candidate,
                    "assembled": assembled,
                    "direct_ridge_residual": direct_oof[:, 0],
                    "direct_tree_residual": direct_oof[:, 1],
                    "support_mask": support,
                    "nearest_same_target_train": nearest,
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
    full_candidate_gate_pass = bool(banked and assembled_mean - parent_mean >= MIN_FULL_MEAN_GAIN and max_loss >= -0.003)

    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component_test = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "direct_candidate": result["test_candidate"],
            "support_mask": result["test_support_mask"],
            "nearest_same_target_train": result["test_nearest"],
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
        raise RuntimeError("C254 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C254 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C254 prediction finite check failed")

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
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "active_target": ACTIVE_TARGET,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(full_candidate_gate_pass and assembled_mean >= 0.95),
        "decision": "banked_component_pending_compound_audit" if banked else "rejected_component_or_full_gate",
        "component_diagnostics": {
            "active_target": ACTIVE_TARGET,
            "changed_factor": "Tg backbone/pendant rigidity support-gated residual",
            "tg_delta_r2": active_report["delta_r2"],
            "tg_positive_folds": active_report["positive_folds"],
            "tg_group_bootstrap_lower": active_report["group_bootstrap_lower"],
            "tg_minimum_panel_delta": active_report["minimum_panel_delta"],
            "tg_support_panel_minimum_delta": active_report["support_panel_minimum_delta"],
            "tg_support_rows": active_report["support_rows"],
            "tg_fallback_rows": active_report["fallback_rows"],
            "test_support_rows": int(np.sum(result["test_support_mask"])),
            "c244_c228_c232_residual_stack_not_reused": True,
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
        },
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c208_panel_helper": sha256_file(round2_root / "tools/round2_c208_tg_robust_group_measurement.py"),
            "c236_partition_helper": sha256_file(round2_root / "tools/round2_c236_nc_backbone_pendant_polarizability.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component_test.to_csv(run_dir / "tg_component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "active_target": ACTIVE_TARGET,
            "features": "wildcard shortest-path backbone/pendant rigidity, Crippen partition, rings, rotatable density, and motif flags",
            "model": "support-gated fixed 0.20 residual blend of Ridge and ExtraTrees",
            "ridge_alpha": RIDGE_ALPHA,
            "tree_count": TREE_COUNT,
            "tree_min_leaf": TREE_MIN_LEAF,
            "residual_weight": RESIDUAL_WEIGHT,
            "support_similarity_floor": SUPPORT_SIMILARITY_FLOOR,
            "component_gate": {
                "minimum_delta_r2": 0.01,
                "minimum_positive_folds": 4,
                "bootstrap_lower_must_exceed": 0.0,
                "minimum_panel_delta_must_be_at_least": 0.0,
                "support_and_fallback_panels_must_be_nonnegative": True,
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
        f"Decision: **{report['decision']}**. Tg parent R² "
        f"`{active_report['parent_r2']:.12f}`, candidate R² "
        f"`{active_report['candidate_r2']:.12f}`, delta "
        f"`{active_report['delta_r2']:+.12f}`. Support rows "
        f"`{active_report['support_rows']}`; fallback rows `{active_report['fallback_rows']}`. "
        "Official-only; no local_eval, Kaggle compute, upload, submission, or final-notebook action.\n",
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
                "tg_parent_r2": active_report["parent_r2"],
                "tg_candidate_r2": active_report["candidate_r2"],
                "tg_delta_r2": active_report["delta_r2"],
                "support_rows": active_report["support_rows"],
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
