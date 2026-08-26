#!/usr/bin/env python3
"""Nested official-only Ei disagreement-gated residual stacker diagnostic."""

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
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "ei"
C027_ID = "R2-C027-20260803-2100-ei-absolute-electronic-topology"


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


def make_scaffold(smiles: str) -> str:
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


def nearest_to_train(left: list[Any], right: list[Any]) -> np.ndarray:
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, right)) for fp in left], dtype=np.float64)


def parent_arms(
    dense: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    y_global: np.ndarray,
    global_train: np.ndarray,
    global_validation: np.ndarray,
) -> np.ndarray:
    values = reference.predict_base_models(
        dense,
        sparse_parts,
        fingerprints,
        y_global,
        global_train,
        global_validation,
        reference.DEFAULT_CONFIG,
        TARGET,
    )
    if not np.isfinite(values).all():
        raise RuntimeError("parent arm predictions are not finite")
    return values


def residual_features(
    arms: np.ndarray,
    physical: np.ndarray,
    availability: np.ndarray,
) -> np.ndarray:
    disagreement = np.column_stack([
        np.mean(arms, axis=1),
        np.std(arms, axis=1),
        np.ptp(arms, axis=1),
    ])
    return np.hstack([arms, disagreement, physical, availability[:, None]]).astype(np.float64, copy=False)


def nested_split(
    y: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    dense: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    y_global: np.ndarray,
    global_indices: np.ndarray,
    physical: np.ndarray,
    availability: np.ndarray,
    inner_splits: int = 4,
) -> dict[str, Any]:
    inner_groups = groups[outer_train]
    inner_folds = folds_for(inner_groups, inner_splits)
    inner_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for inner_fold in range(inner_splits):
        inner_train_local = np.flatnonzero(inner_folds != inner_fold)
        inner_validation_local = np.flatnonzero(inner_folds == inner_fold)
        inner_train = outer_train[inner_train_local]
        inner_validation = outer_train[inner_validation_local]
        inner_arms[inner_validation_local] = parent_arms(
            dense, sparse_parts, fingerprints, y_global, global_indices[inner_train], global_indices[inner_validation]
        )
    if not np.isfinite(inner_arms).all():
        raise RuntimeError("inner parent OOF predictions are incomplete")
    outer_arms = parent_arms(
        dense, sparse_parts, fingerprints, y_global, global_indices[outer_train], global_indices[outer_validation]
    )
    weights, intercept, blend_name, inner_blend_r2 = reference.blend_from_oof(y[outer_train], inner_arms)
    inner_parent = reference.clip_prediction(y[outer_train], inner_arms @ weights + intercept)
    outer_parent = reference.clip_prediction(y[outer_train], outer_arms @ weights + intercept)
    threshold = float(np.quantile(np.std(inner_arms, axis=1), 0.75))
    inner_features = residual_features(inner_arms, physical[outer_train], availability[outer_train])
    outer_features = residual_features(outer_arms, physical[outer_validation], availability[outer_validation])
    residual_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500),
    )
    residual_model.fit(inner_features, y[outer_train] - inner_parent)
    residual_prediction = np.asarray(residual_model.predict(outer_features), dtype=np.float64)
    high_disagreement = np.std(outer_arms, axis=1) >= threshold
    outer_candidate = outer_parent.copy()
    outer_candidate[high_disagreement] = outer_candidate[high_disagreement] + 0.5 * residual_prediction[high_disagreement]
    outer_candidate = reference.clip_prediction(y[outer_train], outer_candidate)
    return {
        "parent": outer_parent,
        "candidate": outer_candidate,
        "arms": outer_arms,
        "threshold": threshold,
        "high_disagreement": high_disagreement,
        "weights": weights.tolist(),
        "intercept": float(intercept),
        "blend_name": blend_name,
        "inner_blend_r2": float(inner_blend_r2),
        "inner_rows": int(len(outer_train)),
        "residual_model": "HuberRegressor",
    }


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray, seed: int = 2026) -> float:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(500):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[group] for group in selected])
        if len(rows) < 2 or float(np.var(y[rows])) <= 1.0e-15:
            continue
        values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


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
    existing = {path.name for path in run_dir.iterdir()}
    if existing - {"protocol.json"}:
        raise RuntimeError(f"refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()

    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical_all, physical_names = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical_all]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, TARGET)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    isomeric_keys = frame["canonical"].to_numpy(object)
    groups = np.asarray([no_stereo(value) for value in isomeric_keys], dtype=object)
    scaffolds = np.asarray([make_scaffold(value) for value in isomeric_keys], dtype=object)
    indices = np.asarray([key_to_index[key] for key in isomeric_keys], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[indices] = y
    target_index = reference.TARGETS.index(TARGET)
    availability = np.sum(cross_available[:, :], axis=1) - cross_available[:, target_index]

    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan, dtype=np.float64)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    nearest = np.full(len(y), np.nan, dtype=np.float64)
    high_disagreement = np.zeros(len(y), dtype=bool)
    thresholds: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    blend_rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        result = nested_split(
            y, groups, training, validation, target_dense, sparse_parts, fingerprints,
            y_global, indices, physical_all[indices], availability[indices], inner_splits=4,
        )
        baseline[validation] = result["parent"]
        candidate[validation] = result["candidate"]
        high_disagreement[validation] = result["high_disagreement"]
        thresholds.append(float(result["threshold"]))
        train_fps = [fingerprints[indices[row]] for row in training]
        nearest[validation] = nearest_to_train([fingerprints[indices[row]] for row in validation], train_fps)
        base_r2 = float(r2_score(y[validation], result["parent"]))
        candidate_r2 = float(r2_score(y[validation], result["candidate"]))
        high_delta = panel_delta(y[validation], result["parent"], result["candidate"], result["high_disagreement"])
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "baseline_r2": base_r2,
            "candidate_r2": candidate_r2,
            "delta_r2": candidate_r2 - base_r2,
            "high_disagreement_rows": int(np.sum(result["high_disagreement"])),
            "high_disagreement_delta_r2": high_delta,
            "threshold": float(result["threshold"]),
        })
        blend_rows.append({
            "fold": fold,
            "weights": result["weights"],
            "intercept": result["intercept"],
            "blend_name": result["blend_name"],
            "inner_blend_r2": result["inner_blend_r2"],
            "inner_rows": result["inner_rows"],
        })

    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise RuntimeError("main nested predictions are incomplete")

    scaffold_groups = [name for name in sorted(set(scaffolds)) if int(np.sum(scaffolds == name)) >= 10]
    scaffold_holdout: dict[str, Any] = {}
    for scaffold in scaffold_groups:
        validation = np.flatnonzero(scaffolds == scaffold)
        training = np.flatnonzero(scaffolds != scaffold)
        result = nested_split(
            y, groups, training, validation, target_dense, sparse_parts, fingerprints,
            y_global, indices, physical_all[indices], availability[indices], inner_splits=4,
        )
        scaffold_holdout[scaffold] = {
            "rows": int(len(validation)),
            "baseline_r2": float(r2_score(y[validation], result["parent"])),
            "candidate_r2": float(r2_score(y[validation], result["candidate"])),
            "delta_r2": float(r2_score(y[validation], result["candidate"]) - r2_score(y[validation], result["parent"])),
            "threshold": float(result["threshold"]),
        }

    panels: dict[str, Any] = {}
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (nearest >= lower) & (nearest < upper)
        value = panel_delta(y, baseline, candidate, selected)
        if value is not None:
            panels[f"similarity_{name}"] = {"rows": int(np.sum(selected)), "delta_r2": value}
    auxiliary = availability[indices] > 0
    for name, selected in (("available_other_property", auxiliary), ("missing_other_property", ~auxiliary)):
        value = panel_delta(y, baseline, candidate, selected)
        if value is not None:
            panels[f"availability_{name}"] = {"rows": int(np.sum(selected)), "delta_r2": value}
    scaffold_slice: dict[str, Any] = {}
    for scaffold in scaffold_groups:
        selected = scaffolds == scaffold
        value = panel_delta(y, baseline, candidate, selected)
        if value is not None:
            scaffold_slice[scaffold] = {"rows": int(np.sum(selected)), "delta_r2": value}
    if scaffold_slice:
        panels["scaffold_slice_canonical_oof"] = scaffold_slice
    high_delta = panel_delta(y, baseline, candidate, high_disagreement)
    if high_delta is not None:
        panels["high_disagreement"] = {"rows": int(np.sum(high_disagreement)), "delta_r2": high_delta}

    baseline_r2 = float(r2_score(y, baseline))
    candidate_r2 = float(r2_score(y, candidate))
    delta_r2 = candidate_r2 - baseline_r2
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    panel_values = []
    for name, value in panels.items():
        if name == "scaffold_slice_canonical_oof":
            panel_values.extend(float(item["delta_r2"]) for item in value.values())
        else:
            panel_values.append(float(value["delta_r2"]))
    if scaffold_holdout:
        panel_values.extend(float(item["delta_r2"]) for item in scaffold_holdout.values())
    min_panel = min(panel_values) if panel_values else None
    passed = bool(
        delta_r2 >= 0.01
        and positive_folds >= 4
        and bootstrap > 0.0
        and (min_panel is None or min_panel >= -0.003)
        and high_delta is not None
        and high_delta >= 0.01
    )
    script_path = root / "tools" / "round2_ei_absolute_electronic_topology.py"
    reference_path = root / "tools" / "initial_reference_pipeline.py"
    audit = {
        "schema_version": "ppp.round2.ei-nested-disagreement-residual-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C026-20260803-2040-ei-atompair-torsion",
        "official_inputs": inputs,
        "source_hashes": {
            "script": sha256_file(script_path),
            "reference_module": sha256_file(reference_path),
        },
        "target": TARGET,
        "rows": int(len(y)),
        "group_count": int(len(np.unique(groups))),
        "baseline_r2_nested_parent": baseline_r2,
        "candidate_r2_nested_residual": candidate_r2,
        "delta_r2": delta_r2,
        "positive_outer_folds": positive_folds,
        "group_r2_bootstrap_lower": bootstrap,
        "outer_folds": fold_rows,
        "blend_folds": blend_rows,
        "thresholds": thresholds,
        "panels": panels,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": min((float(item["delta_r2"]) for item in scaffold_holdout.values()), default=None),
        "min_panel_delta": min_panel,
        "pass": passed,
        "decision": "component_pass" if passed else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({
        "canonical_no_stereo_group": groups,
        "fold": main_folds,
        "scaffold": scaffolds,
        "nearest_tanimoto": nearest,
        "has_other_property": auxiliary,
        "high_disagreement": high_disagreement,
        "y": y,
        "nested_parent": baseline,
        "residual_candidate": candidate,
    }).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.ei-nested-disagreement-residual.v1",
        "seed": 2026,
        "target": TARGET,
        "outer_folds": 5,
        "inner_folds": 4,
        "parent_arms": ["sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local"],
        "residual_model": {"name": "HuberRegressor", "epsilon": 1.35, "alpha": 0.0001, "max_iter": 500},
        "threshold_quantile": 0.75,
        "correction_strength": 0.5,
        "physical_features": physical_names,
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
        f"# R2-C027 nested Ei disagreement residual stacker\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n",
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
    manifest_lines = []
    for path in sorted(run_dir.iterdir()):
        if path.name == "artifact_manifest.sha256" or not path.is_file():
            continue
        manifest_lines.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest_lines.extend([
        f"{audit['source_hashes']['script']}  SOURCE tools/round2_ei_absolute_electronic_topology.py",
        f"{audit['source_hashes']['reference_module']}  SOURCE tools/initial_reference_pipeline.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
