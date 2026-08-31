#!/usr/bin/env python3
"""Nested official-only Eea cross-target predicted-label residual diagnostic."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "eps"
AUXILIARY = ("tg", "egc", "egb", "ei", "eea", "nc")
PARENT_WEIGHTS = np.asarray([0.0, 0.1790654924385139, 0.715171004274314, 0.10576350328717195], dtype=np.float64)
PARENT_INTERCEPT = -0.004633009752527485
RESIDUAL_WEIGHT = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def no_stereo(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("official SMILES failed RDKit parsing")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        value = ""
    return value or "ACYCLIC"


def folds_for(groups: np.ndarray, n_splits: int) -> np.ndarray:
    if len(np.unique(groups)) < n_splits:
        raise RuntimeError(f"need {n_splits} groups, found {len(np.unique(groups))}")
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def parent_arms(dense: np.ndarray, sparse_parts: list[Any], fingerprints: list[Any], y_global: np.ndarray, train: np.ndarray, validation: np.ndarray) -> np.ndarray:
    values = reference.predict_base_models(dense, sparse_parts, fingerprints, y_global, train, validation, reference.DEFAULT_CONFIG, TARGET)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite EPS parent arm prediction")
    return values


def fold_target_dense(
    deterministic: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    target: str,
    key_groups: np.ndarray,
    forbidden_groups: set[str],
) -> np.ndarray:
    values = np.asarray(cross_values, dtype=np.float64).copy()
    available = np.asarray(cross_available, dtype=np.float64).copy()
    blocked = np.isin(key_groups, list(forbidden_groups)) if forbidden_groups else np.zeros(len(key_groups), dtype=bool)
    values[blocked] = np.nan
    available[blocked] = 0.0
    target_index = reference.TARGETS.index(target)
    values[:, target_index] = np.nan
    available[:, target_index] = 0.0
    return np.hstack([deterministic, values, available]).astype(np.float64, copy=False)


def aux_prediction(
    target: str,
    forbidden_groups: set[str],
    prediction_global: np.ndarray,
    aux_info: dict[str, dict[str, Any]],
    deterministic: np.ndarray,
) -> np.ndarray:
    info = aux_info[target]
    keep = np.asarray([group not in forbidden_groups for group in info["groups"]], dtype=bool)
    train_global = info["global_indices"][keep]
    train_y = info["y"][keep]
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=30.0),
    )
    model.fit(deterministic[train_global], train_y)
    return np.asarray(model.predict(deterministic[prediction_global]), dtype=np.float64)


def nested_split(
    y: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    deterministic: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    key_groups: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    y_global: np.ndarray,
    global_indices: np.ndarray,
    aux_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    inner_folds = folds_for(groups[outer_train], 4)
    inner_parent_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    inner_aux = np.full((len(outer_train), len(AUXILIARY)), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_dense = fold_target_dense(deterministic, cross_values, cross_available, TARGET, key_groups, set(groups[local_validation]))
        inner_parent_arms[inner_folds == fold] = parent_arms(
            inner_dense, sparse_parts, fingerprints, y_global, global_indices[local_train], global_indices[local_validation]
        )
        forbidden = set(groups[local_validation])
        for column, target in enumerate(AUXILIARY):
            inner_aux[inner_folds == fold, column] = aux_prediction(
                target, forbidden, global_indices[local_validation], aux_info, deterministic
            )
    outer_dense = fold_target_dense(deterministic, cross_values, cross_available, TARGET, key_groups, set(groups[outer_validation]))
    outer_parent_arms = parent_arms(outer_dense, sparse_parts, fingerprints, y_global, global_indices[outer_train], global_indices[outer_validation])
    forbidden = set(groups[outer_validation])
    outer_aux = np.column_stack([
        aux_prediction(target, forbidden, global_indices[outer_validation], aux_info, deterministic)
        for target in AUXILIARY
    ])
    inner_parent = reference.clip_prediction(y[outer_train], inner_parent_arms @ PARENT_WEIGHTS + PARENT_INTERCEPT)
    outer_parent = reference.clip_prediction(y[outer_train], outer_parent_arms @ PARENT_WEIGHTS + PARENT_INTERCEPT)
    inner_features = np.hstack([inner_parent[:, None], inner_aux])
    outer_features = np.hstack([outer_parent[:, None], outer_aux])
    residual_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=30.0),
    )
    residual_model.fit(inner_features, y[outer_train] - inner_parent)
    correction = residual_model.predict(outer_features)
    candidate = reference.clip_prediction(y[outer_train], outer_parent + RESIDUAL_WEIGHT * correction)
    return {
        "parent": outer_parent,
        "candidate": candidate,
        "inner_blend_r2": float(r2_score(y[outer_train], inner_parent)),
        "blend_name": "frozen_c050_eps_v7_weights",
        "weights": PARENT_WEIGHTS.tolist(),
        "intercept": float(PARENT_INTERCEPT),
    }


def nearest_to_train(left: list[Any], right: list[Any]) -> np.ndarray:
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, right)) for fp in left], dtype=np.float64)


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(2000):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in chosen])
        if len(rows) > 1 and np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


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
        raise RuntimeError(f"pre-created protocol directory is required: {run_dir}")
    if {path.name for path in run_dir.iterdir()} - {"protocol.json"}:
        raise RuntimeError(f"refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    deterministic = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    aux_info: dict[str, dict[str, Any]] = {}
    for target in AUXILIARY:
        aux_frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        aux_info[target] = {
            "global_indices": np.asarray([key_to_index[value] for value in aux_frame["canonical"]], dtype=np.int64),
            "groups": np.asarray([no_stereo(value) for value in aux_frame["canonical"]], dtype=object),
            "y": aux_frame["target"].to_numpy(float),
        }
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    isomeric = frame["canonical"].to_numpy(object)
    groups = np.asarray([no_stereo(value) for value in isomeric], dtype=object)
    scaffolds = np.asarray([scaffold(value) for value in isomeric], dtype=object)
    indices = np.asarray([key_to_index[value] for value in isomeric], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[indices] = y
    key_groups = np.asarray([no_stereo(value) for value in keys], dtype=object)
    target_index = reference.TARGETS.index(TARGET)
    availability = np.sum(cross_available, axis=1) - cross_available[:, target_index]

    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan)
    candidate = np.full(len(y), np.nan)
    nearest = np.full(len(y), np.nan)
    fold_rows = []
    blend_rows = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        result = nested_split(
            y, groups, training, validation, deterministic, cross_values, cross_available,
            key_groups, sparse_parts, fingerprints, y_global, indices, aux_info
        )
        baseline[validation] = result["parent"]
        candidate[validation] = result["candidate"]
        nearest[validation] = nearest_to_train(
            [fingerprints[indices[row]] for row in validation],
            [fingerprints[indices[row]] for row in training],
        )
        base_r2 = float(r2_score(y[validation], result["parent"]))
        cand_r2 = float(r2_score(y[validation], result["candidate"]))
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "baseline_r2": base_r2,
            "candidate_r2": cand_r2,
            "delta_r2": cand_r2 - base_r2,
        })
        blend_rows.append({
            "fold": fold,
            "weights": result["weights"],
            "intercept": result["intercept"],
            "inner_parent_r2": result["inner_blend_r2"],
            "blend_name": result["blend_name"],
        })

    scaffold_groups = [name for name in sorted(set(scaffolds)) if int(np.sum(scaffolds == name)) >= 10]
    scaffold_holdout = {}
    for name in scaffold_groups:
        validation = np.flatnonzero(scaffolds == name)
        training = np.flatnonzero(scaffolds != name)
        result = nested_split(
            y, groups, training, validation, deterministic, cross_values, cross_available,
            key_groups, sparse_parts, fingerprints, y_global, indices, aux_info
        )
        parent_r2 = float(r2_score(y[validation], result["parent"]))
        candidate_r2 = float(r2_score(y[validation], result["candidate"]))
        scaffold_holdout[name] = {
            "rows": int(len(validation)),
            "baseline_r2": parent_r2,
            "candidate_r2": candidate_r2,
            "delta_r2": candidate_r2 - parent_r2,
        }

    panels = {}
    for name, lower, upper in (
        ("lt_0.30", 0.0, 0.30),
        ("0.30_0.50", 0.30, 0.50),
        ("0.50_0.70", 0.50, 0.70),
        ("ge_0.70", 0.70, 1.01),
    ):
        selected = (nearest >= lower) & (nearest < upper)
        delta = panel_delta(y, baseline, candidate, selected)
        panels[f"similarity_{name}"] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "insufficient_or_constant",
        }
    auxiliary = availability[indices] > 0
    for name, selected in (("available_other_property", auxiliary), ("missing_other_property", ~auxiliary)):
        delta = panel_delta(y, baseline, candidate, selected)
        panels[f"availability_{name}"] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "insufficient_or_constant",
        }
    scaffold_slice = {}
    for name in scaffold_groups:
        selected = scaffolds == name
        delta = panel_delta(y, baseline, candidate, selected)
        scaffold_slice[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "insufficient_or_constant",
        }
    panels["scaffold_slice_canonical_oof"] = scaffold_slice

    baseline_r2 = float(r2_score(y, baseline))
    candidate_r2 = float(r2_score(y, candidate))
    delta_r2 = candidate_r2 - baseline_r2
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    panel_values = []
    panel_incomplete = False
    for value in panels.values():
        items = value.values() if isinstance(value, dict) and "delta_r2" not in value else [value]
        for item in items:
            if item.get("delta_r2") is None:
                panel_incomplete = True
            else:
                panel_values.append(float(item["delta_r2"]))
    panel_values.extend(float(item["delta_r2"]) for item in scaffold_holdout.values())
    min_panel = min(panel_values) if panel_values else None
    passed = bool(
        delta_r2 >= 0.01
        and positive_folds >= 4
        and bootstrap > 0.0
        and not panel_incomplete
        and (min_panel is None or min_panel >= -0.003)
    )
    # Deployment-matched EPS component: fit the residual model on cross-fitted
    # predictions for every official EPS training row, then predict the 153
    # official test EPS rows.  This remains source-only and does not consume
    # stored C050 predictions or any local_eval artifact.
    full_inner_folds = folds_for(groups, 5)
    full_parent_arms = np.full((len(y), 4), np.nan, dtype=np.float64)
    full_aux = np.full((len(y), len(AUXILIARY)), np.nan, dtype=np.float64)
    for fold in range(5):
        local_train = np.flatnonzero(full_inner_folds != fold)
        local_validation = np.flatnonzero(full_inner_folds == fold)
        dense = fold_target_dense(deterministic, cross_values, cross_available, TARGET, key_groups, set(groups[local_validation]))
        full_parent_arms[local_validation] = parent_arms(dense, sparse_parts, fingerprints, y_global, indices[local_train], indices[local_validation])
        forbidden = set(groups[local_validation])
        for column, target in enumerate(AUXILIARY):
            full_aux[local_validation, column] = aux_prediction(target, forbidden, indices[local_validation], aux_info, deterministic)
    full_parent = reference.clip_prediction(y, full_parent_arms @ PARENT_WEIGHTS + PARENT_INTERCEPT)
    full_features = np.hstack([full_parent[:, None], full_aux])
    full_residual_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=30.0),
    )
    full_residual_model.fit(full_features, y - full_parent)
    test_frame = test[test["target_type"] == TARGET].reset_index(drop=True)
    test_global_indices = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
    test_dense = fold_target_dense(deterministic, cross_values, cross_available, TARGET, key_groups, set())
    test_parent_arms = parent_arms(test_dense, sparse_parts, fingerprints, y_global, indices, test_global_indices)
    test_parent = reference.clip_prediction(y, test_parent_arms @ PARENT_WEIGHTS + PARENT_INTERCEPT)
    test_aux = np.column_stack([
        aux_prediction(target, set(), test_global_indices, aux_info, deterministic)
        for target in AUXILIARY
    ])
    test_features = np.hstack([test_parent[:, None], test_aux])
    test_correction = full_residual_model.predict(test_features)
    test_candidate = reference.clip_prediction(y, test_parent + RESIDUAL_WEIGHT * test_correction)
    if len(test_candidate) != 153 or not np.isfinite(test_candidate).all():
        raise RuntimeError("EPS test component output contract failed")
    script_path = root / "tools" / "round2_c092_eps_cross_target_oof_residual.py"
    reference_path = root / "tools" / "initial_reference_pipeline.py"
    audit = {
        "schema_version": "ppp.round2.c092.eps-cross-target-oof-residual-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "C050-v7 EPS configuration regenerated with fold-masked grouped parent",
        "official_inputs": inputs,
        "source_hashes": {
            "script": sha256_file(script_path),
            "reference_module": sha256_file(reference_path),
        },
        "target": TARGET,
        "auxiliary_targets": AUXILIARY,
        "rows": int(len(y)),
        "group_count": int(len(np.unique(groups))),
        "baseline_r2_nested_parent": baseline_r2,
        "candidate_r2_cross_target_residual": candidate_r2,
        "delta_r2": delta_r2,
        "residual_weight": RESIDUAL_WEIGHT,
        "parent_weights": PARENT_WEIGHTS.tolist(),
        "parent_intercept": PARENT_INTERCEPT,
        "positive_outer_folds": positive_folds,
        "group_r2_bootstrap_lower": bootstrap,
        "outer_folds": fold_rows,
        "blend_folds": blend_rows,
        "panels": panels,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": min(
            (float(item["delta_r2"]) for item in scaffold_holdout.values()),
            default=None,
        ),
        "min_panel_delta": min_panel,
        "panel_incomplete": panel_incomplete,
        "pass": passed,
        "decision": "component_pass" if passed else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
        "test_component_rows": int(len(test_candidate)),
        "stored_parent_predictions_read": False,
        "external_label_file_read": False,
        "local_eval_read": False,
    }
    pd.DataFrame({
        "canonical_no_stereo_group": groups,
        "fold": main_folds,
        "scaffold": scaffolds,
        "nearest_tanimoto": nearest,
        "has_other_property": auxiliary,
        "y": y,
        "nested_parent": baseline,
        "candidate": candidate,
    }).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame({
        "id": test_frame["id"].astype(int),
        "target": test_candidate.astype(float),
        "target_type": TARGET,
        "parent_prediction": test_parent.astype(float),
        "correction": test_correction.astype(float),
    }).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.c092.eps-cross-target-oof-residual.v1",
        "seed": 2026,
        "target": TARGET,
        "auxiliary_targets": AUXILIARY,
        "outer_folds": 5,
        "inner_folds": 4,
        "auxiliary_model": "fold-local standardized Ridge(alpha=30) on deterministic official descriptors",
        "residual_model": "fold-local standardized Ridge(alpha=30), fixed correction strength 0.25",
        "official_inputs": inputs,
        "source_hashes": audit["source_hashes"],
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
        f"rdkit={Chem.rdBase.rdkitVersion}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# R2-C092 EPS cross-target OOF residual\n\nDecision: **{audit['decision']}**. EPS component only; no full candidate or local_eval diagnostic was created by this bounded component run.\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(
        f"experiment_id={run_dir.name}\n"
        f"target={TARGET}\n"
        f"nested_parent_r2={baseline_r2:.12f}\n"
        f"candidate_r2={candidate_r2:.12f}\n"
        f"delta_r2={delta_r2:.12f}\n"
        f"pass={passed}\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([
        f"{audit['source_hashes']['script']}  SOURCE tools/round2_c092_eps_cross_target_oof_residual.py",
        f"{audit['source_hashes']['reference_module']}  SOURCE tools/initial_reference_pipeline.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
