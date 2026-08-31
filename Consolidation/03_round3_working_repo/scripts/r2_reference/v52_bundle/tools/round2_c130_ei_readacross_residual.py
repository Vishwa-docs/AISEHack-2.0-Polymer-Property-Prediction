#!/usr/bin/env python3
"""C130: clean Ei read-across and pi-graph residual specialist.

This is a fresh-source, target-specific Ei experiment.  It uses only structure
derived features and outer-fold training rows: a Tanimoto/Morgan residual
read-across arm and an ExtraTrees residual arm over electronic plus small
pi-graph spectral descriptors.  C050 remains the exact fallback for every
target, including Ei unless the fixed component gates pass.
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
from rdkit import Chem, DataStructs, RDLogger
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c124_electronic_ei_eea_bank as electronic
import round2_c127_round1_carrier_factory as c127


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "ei"
SEED = 2026
N_FOLDS = 5
K_NEIGHBORS = 16
SHRINKAGE = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, stage: str, **payload: Any) -> None:
    record = {"stage": stage, "time": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def spectral_vector(molecule: Chem.Mol) -> np.ndarray:
    """Small deterministic graph-spectrum vector for the capped repeat unit."""
    try:
        rw = Chem.RWMol(Chem.Mol(molecule))
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(6)
                atom.SetFormalCharge(0)
                atom.SetNoImplicit(False)
        capped = rw.GetMol()
        Chem.SanitizeMol(capped)
        atoms = list(capped.GetAtoms())
        n = len(atoms)
        if n == 0:
            return np.zeros(24, dtype=np.float64)
        adjacency = np.zeros((n, n), dtype=np.float64)
        pi_nodes: list[int] = []
        for atom in atoms:
            if atom.GetIsAromatic() or atom.GetHybridization() == Chem.HybridizationType.SP2:
                pi_nodes.append(atom.GetIdx())
        pi_adjacency = np.zeros((len(pi_nodes), len(pi_nodes)), dtype=np.float64)
        pi_position = {idx: row for row, idx in enumerate(pi_nodes)}
        for bond in capped.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            weight = 1.5 if bond.GetIsAromatic() else max(float(bond.GetBondTypeAsDouble()), 1.0)
            adjacency[begin, end] = adjacency[end, begin] = weight
            if begin in pi_position and end in pi_position:
                left, right = pi_position[begin], pi_position[end]
                pi_adjacency[left, right] = pi_adjacency[right, left] = weight

        values = np.linalg.eigvalsh(adjacency) if n > 1 else np.zeros(1, dtype=np.float64)
        pi_values = np.linalg.eigvalsh(pi_adjacency) if len(pi_nodes) > 1 else np.zeros(1, dtype=np.float64)
        vector = np.zeros(24, dtype=np.float64)
        vector[:6] = [
            float(values[-1]),
            float(values[-2]) if len(values) > 1 else 0.0,
            float(values[0]),
            float(np.max(np.abs(values))),
            float(np.sum(np.abs(values))),
            float(np.std(values)),
        ]
        vector[6:12] = [
            float(pi_values[-1]),
            float(pi_values[-2]) if len(pi_values) > 1 else 0.0,
            float(pi_values[0]),
            float(np.max(np.abs(pi_values))),
            float(np.sum(np.abs(pi_values))),
            float(np.std(pi_values)),
        ]
        heavy = max(capped.GetNumHeavyAtoms(), 1)
        vector[12:] = [
            float(len(atoms)), float(heavy), float(len(pi_nodes)), float(len(pi_nodes) / heavy),
            float(capped.GetRingInfo().NumRings()), float(capped.GetNumAtoms() / heavy),
            float(sum(atom.GetAtomicNum() == 6 for atom in atoms) / heavy),
            float(sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms) / heavy),
            float(sum(bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0 for bond in capped.GetBonds())),
            float(len(capped.GetBonds()) / heavy), float(np.trace(adjacency @ adjacency)),
            float(np.trace(pi_adjacency @ pi_adjacency)) if len(pi_nodes) else 0.0,
        ]
        return vector
    except Exception:
        return np.full(24, np.nan, dtype=np.float64)


def spectral_matrix(molecules: list[Chem.Mol]) -> np.ndarray:
    return np.vstack([spectral_vector(molecule) for molecule in molecules]).astype(np.float64, copy=False)


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    return c127.grouped_folds(groups)


def read_across_residual(
    fingerprints: list[Any],
    train_global: np.ndarray,
    train_residual: np.ndarray,
    prediction_global: np.ndarray,
) -> np.ndarray:
    train_fps = [fingerprints[int(index)] for index in train_global]
    take = min(K_NEIGHBORS, len(train_global))
    output = np.zeros(len(prediction_global), dtype=np.float64)
    for row, global_index in enumerate(prediction_global):
        similarity = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[int(global_index)], train_fps), dtype=np.float64)
        selected = np.argpartition(similarity, -take)[-take:]
        weights = np.maximum(similarity[selected], 1.0e-6) ** 4
        output[row] = float(np.dot(weights, train_residual[selected]) / np.sum(weights))
    return output


def extra_residual(
    features: np.ndarray,
    train_global: np.ndarray,
    train_residual: np.ndarray,
    prediction_global: np.ndarray,
) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        ExtraTreesRegressor(
            n_estimators=256,
            max_depth=10,
            min_samples_leaf=2,
            max_features=0.60,
            random_state=SEED,
            n_jobs=2,
        ),
    )
    model.fit(features[train_global], train_residual)
    return np.asarray(model.predict(features[prediction_global]), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    progress = run_dir / "progress.jsonl"
    checkpoint(progress, "started", experiment_id=run_dir.name)
    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = c127.source_parity(root, parent, (root / args.canonical_run).resolve())
    checkpoint(progress, "parent_parity", **parity)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    global_indices = np.arange(len(parent["molecules"]), dtype=np.int64)
    structural, structural_names, continuous_count = electronic.structural_bank(parent["molecules"], global_indices.tolist())
    spectral = spectral_matrix(parent["molecules"])
    features = np.hstack([structural, spectral]).astype(np.float64, copy=False)
    feature_report = {"shape": [int(value) for value in features.shape], "structural_count": int(structural.shape[1]), "continuous_count": int(continuous_count), "spectral_count": int(spectral.shape[1]), "paired_labels_used": False}
    checkpoint(progress, "features_constructed", **feature_report)

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    target_global = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = grouped_folds(groups)
    residual = y - parent_oof
    direct_oof = np.full((len(y), 2), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        knn = read_across_residual(parent["fingerprints"], target_global[training], residual[training], target_global[validation])
        extra = extra_residual(features, target_global[training], residual[training], target_global[validation])
        direct_oof[validation] = np.column_stack([parent_oof[validation] + SHRINKAGE * knn, parent_oof[validation] + SHRINKAGE * extra])
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "knn_delta_r2": float(reference.r2_score(y[validation], direct_oof[validation, 0]) - reference.r2_score(y[validation], parent_oof[validation])), "extra_delta_r2": float(reference.r2_score(y[validation], direct_oof[validation, 1]) - reference.r2_score(y[validation], parent_oof[validation]))})
        checkpoint(progress, f"ei_fold_{fold}", **fold_rows[-1])

    arms = np.column_stack([parent_oof, direct_oof])
    weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, arms)
    candidate = arms @ weights + intercept
    report = c127.evaluate_target(info, {"candidate": candidate})
    report.update({"blend_name": blend_name, "blend_weights": [float(value) for value in weights], "blend_intercept": float(intercept), "blend_r2": float(blend_r2), "folds_direct": fold_rows, "feature_count": int(features.shape[1]), "read_across_k": K_NEIGHBORS, "residual_shrinkage": SHRINKAGE})
    checkpoint(progress, "ei_complete", delta_r2=report["delta_r2"], positive_folds=report["positive_folds"], group_bootstrap_lower=report["group_bootstrap_lower"], minimum_panel_delta=report["minimum_panel_delta"], pass_gate=report["pass"])

    test_frame = parent["test"].loc[parent["test"]["target_type"] == ACTIVE_TARGET].sort_values("id").reset_index(drop=True)
    test_indices = np.asarray([parent["key_to_index"][value] for value in test_frame["canonical"]], dtype=np.int64)
    test_knn = read_across_residual(parent["fingerprints"], target_global, residual, test_indices)
    test_extra = extra_residual(features, target_global, residual, test_indices)
    test_parent = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == ACTIVE_TARGET].sort_values("id")["target"].to_numpy(np.float64)
    test_arms = np.column_stack([test_parent, test_parent + SHRINKAGE * test_knn, test_parent + SHRINKAGE * test_extra])
    test_candidate = test_arms @ weights + intercept

    target_reports: dict[str, Any] = {ACTIVE_TARGET: report}
    candidate_oof: dict[str, np.ndarray] = {ACTIVE_TARGET: candidate}
    candidate_test: dict[str, np.ndarray] = {ACTIVE_TARGET: test_candidate}
    oof_parts: list[pd.DataFrame] = []
    banked = [ACTIVE_TARGET] if report["pass"] else []
    for target in TARGETS:
        if target not in target_reports:
            target_info = parent["target_info"][target]
            target_y = np.asarray(target_info["y"], dtype=np.float64)
            target_parent = np.asarray(target_info["parent"], dtype=np.float64)
            target_reports[target] = {"parent_r2": float(reference.r2_score(target_y, target_parent)), "candidate_r2": float(reference.r2_score(target_y, target_parent)), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "panels": {"unchanged_parent": {"rows": int(len(target_y)), "delta_r2": 0.0, "status": "unchanged"}}, "folds": [], "pass": True, "unchanged_parent": True}
            candidate_oof[target] = target_parent
            candidate_test[target] = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id")["target"].to_numpy(np.float64)
        target_info = parent["target_info"][target]
        assembled = candidate_oof[target] if target in banked else np.asarray(target_info["parent"], dtype=np.float64)
        oof_parts.append(pd.DataFrame({"canonical": target_info["canonical"], "target_type": target, "target": target_info["y"], "parent": target_info["parent"], "candidate": candidate_oof[target], "assembled": assembled, "group": target_info["groups"], "scaffold": target_info["scaffolds"], "outer_fold": grouped_folds(np.asarray(target_info["groups"], dtype=object))}))

    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reference.r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(mean_candidate >= mean_parent + 0.002 and max_loss >= -0.003 and len(banked) > 0)
    parent_detail = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    parts: list[pd.DataFrame] = []
    for target in TARGETS:
        frame = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        values = candidate_test[target] if target in banked else parent_detail.loc[parent_detail["target_type"] == target].sort_values("id")["target"].to_numpy(np.float64)
        parts.append(pd.DataFrame({"id": frame["id"].astype(int), "target_type": target, "model_prediction": values}))
    raw = pd.concat(parts, ignore_index=True).sort_values("id")
    raw_labels, _ = reference.build_label_pool(parent["train"], parent["archive"])
    detail, override_report = reference.apply_official_overrides(raw, parent["test"], raw_labels)
    predictions = detail[["id", "target"]].copy()
    if len(predictions) != 4940 or not predictions["id"].equals(parent["test"]["id"]) or not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C130 complete output contract failed")

    source_paths = {
        "runner": Path(__file__),
        "parent_builder": root / "tools/round2_c097_graph_grammar_hgb_full.py",
        "reference": root / "tools/initial_reference_pipeline.py",
        "c124_electronic_features": root / "tools/round2_c124_electronic_ei_eea_bank.py",
        "c127_evaluation": root / "tools/round2_c127_round1_carrier_factory.py",
    }
    result = {
        "schema_version": "ppp.round2.c130.ei-readacross-residual.run.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "parent": "C050 source rebuild; no C127/C128/C129 artifact input",
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "pretrained_weights": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "active_target": ACTIVE_TARGET,
        "banked_targets": banked,
        "target_reports": target_reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_pass,
        "official_override_report": override_report,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "elapsed_seconds": float(time.time() - started),
        "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate",
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c130.ei-readacross-residual.v1", "active_target": ACTIVE_TARGET, "seed": SEED, "neighbors": K_NEIGHBORS, "shrinkage": SHRINKAGE, "arms": ["Tanimoto/Morgan residual read-across", "ExtraTrees electronic/pi-spectrum residual"], "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{result['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{mean_parent:.12f}`; assembled `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. No local_eval read.\n", encoding="utf-8")
    checkpoint(progress, "metrics_written", decision=result["decision"], mean_candidate_r2=mean_candidate, banked_targets=banked)
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, path in source_paths.items():
        manifest.append(f"{sha256_file(path)}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": result["decision"], "banked_targets": banked, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
