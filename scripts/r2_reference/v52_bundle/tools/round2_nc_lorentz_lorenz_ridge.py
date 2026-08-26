#!/usr/bin/env python3
"""Strict nested official-only Lorentz-Lorenz-inspired Nc residual diagnostic."""

from __future__ import annotations

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
from rdkit import DataStructs, RDLogger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGET = "nc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def masked_target_dense(base_dense, cross_values, cross_available, target):
    return np.hstack([
        base_dense,
        np.full_like(cross_values, np.nan),
        np.zeros_like(cross_available),
    ]).astype(np.float64, copy=False)


def selected_physical_features(descriptor, descriptor_names, physical, physical_names):
    di = {name: index for index, name in enumerate(descriptor_names)}
    pi = {name: index for index, name in enumerate(physical_names)}
    descriptor_keep = ["MolMR", "MolWt", "LabuteASA", "TPSA", "HeavyAtomCount", "FractionCSP3", "NumAromaticRings", "NumRotatableBonds"]
    physical_keep = ["atom_count", "heavy_atom_count", "ring_count", "aromatic_atom_count", "hetero_atom_count", "n_count", "o_count", "s_count", "si_count"]
    missing = [name for name in descriptor_keep if name not in di] + [name for name in physical_keep if name not in pi]
    if missing:
        raise RuntimeError(f"Lorentz-Lorenz descriptors missing: {missing}")
    mr = descriptor[:, di["MolMR"]]
    mass = descriptor[:, di["MolWt"]]
    asa = descriptor[:, di["LabuteASA"]]
    tpsa = descriptor[:, di["TPSA"]]
    heavy = descriptor[:, di["HeavyAtomCount"]]
    aromatic = physical[:, pi["aromatic_atom_count"]]
    columns = [descriptor[:, di[name]] for name in descriptor_keep]
    columns.extend(physical[:, pi[name]] for name in physical_keep)
    with np.errstate(divide="ignore", invalid="ignore"):
        columns.extend([mr / mass, mr / asa, tpsa / mass, mass / asa, heavy / mass, aromatic / heavy, mr / heavy])
    names = [f"descriptor_{name}" for name in descriptor_keep]
    names.extend(f"physical_{name}" for name in physical_keep)
    names.extend(["mr_over_molwt", "mr_over_labute_asa", "tpsa_over_molwt", "molwt_over_labute_asa", "heavy_over_molwt", "aromatic_over_heavy", "mr_over_heavy"])
    values = np.column_stack(columns).astype(np.float64, copy=False)
    values[~np.isfinite(values)] = np.nan
    return values, names


def folds_for(groups: np.ndarray, n_splits: int) -> np.ndarray:
    if len(np.unique(groups)) < n_splits:
        raise RuntimeError(f"need {n_splits} groups, found {len(np.unique(groups))}")
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    return result


def parent_arms(dense, sparse_parts, fingerprints, y_global, train_global, validation_global, config):
    return reference.predict_base_models(
        dense, sparse_parts, fingerprints, y_global, train_global,
        validation_global, config, TARGET,
    )


def fit_specialist(features, residual_train, train_global, validation_global):
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    model.fit(features[train_global], residual_train)
    return np.asarray(model.predict(features[validation_global]), dtype=np.float64)


def nested_split(y, groups, outer_train, outer_validation, dense, sparse_parts, fingerprints, y_global, global_indices, features, config):
    inner_folds = folds_for(groups[outer_train], 4)
    inner_parent_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_parent_arms[inner_folds == fold] = parent_arms(
            dense, sparse_parts, fingerprints, y_global,
            global_indices[local_train], global_indices[local_validation], config,
        )
    outer_parent_arms = parent_arms(
        dense, sparse_parts, fingerprints, y_global,
        global_indices[outer_train], global_indices[outer_validation], config,
    )
    weights, intercept, blend_name, inner_blend_r2 = reference.blend_from_oof(y[outer_train], inner_parent_arms)
    # Use fixed mean inner-Oof parent arms for the residual target so each
    # training row's residual is generated without fitting a blend on itself.
    inner_parent = reference.clip_prediction(y[outer_train], np.nanmean(inner_parent_arms, axis=1))
    outer_parent = reference.clip_prediction(y[outer_train], outer_parent_arms @ weights + intercept)
    residual_train = y[outer_train] - inner_parent
    specialist = fit_specialist(features, residual_train, global_indices[outer_train], global_indices[outer_validation])
    candidate = reference.clip_prediction(y[outer_train], outer_parent + specialist)
    return {
        "parent": outer_parent,
        "candidate": candidate,
        "inner_blend_r2": float(inner_blend_r2),
        "blend_name": blend_name,
        "weights": weights.tolist(),
        "intercept": float(intercept),
        "residual_mean": float(np.mean(residual_train)),
        "residual_std": float(np.std(residual_train)),
    }


def nearest_similarity(fingerprints, validation_global, training_global):
    train_fps = [fingerprints[index] for index in training_global]
    return np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
        for index in validation_global
    ], dtype=np.float64)


def panel_delta(y, baseline, candidate, selected):
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def bootstrap_lower(y, baseline, candidate, groups):
    return plumbing.bootstrap_r2_lower(y, baseline, candidate, groups)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} - {"protocol.json"}:
        raise RuntimeError(f"pre-created protocol-only run directory is required: {run_dir}")
    started = time.time()
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    specialist_features = selected_physical_features(descriptor, descriptor_names, physical, physical_names)[0]
    specialist_names = selected_physical_features(descriptor, descriptor_names, physical, physical_names)[1]
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    dense = masked_target_dense(dense_base, cross_values, cross_available, TARGET)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    canonical = frame["canonical"].to_numpy(object)
    groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
    indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[indices] = y
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5})
    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan, dtype=np.float64)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    nearest = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    blend_rows = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        result = nested_split(y, groups, training, validation, dense, sparse_parts, fingerprints, y_global, indices, specialist_features, config)
        baseline[validation] = result["parent"]
        candidate[validation] = result["candidate"]
        nearest[validation] = nearest_similarity(fingerprints, indices[validation], indices[training])
        base_r2 = float(r2_score(y[validation], result["parent"]))
        candidate_r2 = float(r2_score(y[validation], result["candidate"]))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "baseline_r2": base_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - base_r2})
        blend_rows.append({"fold": fold, "weights": result["weights"], "intercept": result["intercept"], "inner_parent_r2": result["inner_blend_r2"], "blend_name": result["blend_name"]})

    scaffold_groups = [name for name in sorted(set(scaffolds)) if int(np.sum(scaffolds == name)) >= 10]
    scaffold_holdout = {}
    for name in scaffold_groups:
        validation = np.flatnonzero(scaffolds == name)
        training = np.flatnonzero(scaffolds != name)
        result = nested_split(y, groups, training, validation, dense, sparse_parts, fingerprints, y_global, indices, specialist_features, config)
        scaffold_holdout[name] = {
            "rows": int(len(validation)),
            "baseline_r2": float(r2_score(y[validation], result["parent"])),
            "candidate_r2": float(r2_score(y[validation], result["candidate"])),
        }
        scaffold_holdout[name]["delta_r2"] = scaffold_holdout[name]["candidate_r2"] - scaffold_holdout[name]["baseline_r2"]

    panels = {}
    panel_values = []
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (nearest >= lower) & (nearest < upper)
        delta = panel_delta(y, baseline, candidate, selected)
        panels[f"similarity_{name}"] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "incomplete"}
        if delta is not None:
            panel_values.append(delta)
    counts = pd.Series(groups).value_counts()
    duplicate_groups = np.asarray([counts[group] >= 2 for group in groups], dtype=bool)
    singleton_groups = ~duplicate_groups
    for name, selected in (("canonical_duplicate_group", duplicate_groups), ("canonical_singleton_group", singleton_groups)):
        delta = panel_delta(y, baseline, candidate, selected)
        panels[name] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "support_incomplete"}
    panels["scaffold_slice_canonical_oof"] = {}
    for name in scaffold_groups:
        selected = scaffolds == name
        delta = panel_delta(y, baseline, candidate, selected)
        panels["scaffold_slice_canonical_oof"][name] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "incomplete"}
        if delta is not None:
            panel_values.append(delta)
    baseline_r2 = float(r2_score(y, baseline))
    candidate_r2 = float(r2_score(y, candidate))
    delta_r2 = candidate_r2 - baseline_r2
    bootstrap = bootstrap_lower(y, baseline, candidate, groups)
    required_incomplete = any(value["delta_r2"] is None for key, value in panels.items() if key.startswith("similarity_")) or any(value["delta_r2"] is None for value in scaffold_holdout.values())
    panel_values.extend(float(value["delta_r2"]) for value in scaffold_holdout.values())
    min_panel = min(panel_values) if panel_values else None
    passed = bool(delta_r2 >= 0.01 and sum(row["delta_r2"] > 0.0 for row in fold_rows) >= 4 and bootstrap > 0.0 and not required_incomplete and (min_panel is None or min_panel >= 0.0))
    audit = {
        "schema_version": "ppp.round2.nc-lorentz-lorenz-ridge-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C045-20260803-2200-eps-compact-qspr-rbf-v2",
        "official_inputs": inputs,
        "target": TARGET,
        "feature_names": specialist_names,
        "feature_count": len(specialist_names),
        "model": "SimpleImputer(median) -> StandardScaler -> Ridge(alpha=10) on fixed Lorentz-Lorenz residual features",
        "baseline_r2_nested_parent": baseline_r2,
        "candidate_r2_lorentz_lorenz_residual": candidate_r2,
        "delta_r2": delta_r2,
        "positive_outer_folds": int(sum(row["delta_r2"] > 0.0 for row in fold_rows)),
        "group_r2_bootstrap_lower": bootstrap,
        "outer_folds": fold_rows,
        "blend_folds": blend_rows,
        "panels": panels,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": min((float(item["delta_r2"]) for item in scaffold_holdout.values()), default=None),
        "min_panel_delta": min_panel,
        "panel_incomplete": required_incomplete,
        "support_counts": {"target_rows": int(len(y)), "canonical_groups": int(len(counts)), "duplicate_group_rows": int(np.sum(duplicate_groups)), "singleton_group_rows": int(np.sum(singleton_groups))},
        "pass": passed,
        "decision": "component_pass" if passed else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({"canonical_no_stereo_group": groups, "fold": main_folds, "scaffold": scaffolds, "nearest_tanimoto": nearest, "y": y, "nested_parent": baseline, "lorentz_lorenz_residual": candidate}).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.nc-lorentz-lorenz-ridge.v1", "seed": 2026, "outer_folds": 5, "inner_folds": 4, "features": specialist_names, "alpha": 10.0, "residual_target": "Nc minus fixed-mean inner-Oof C001 parent", "official_inputs": inputs})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__} ", f"rdkit={__import__('rdkit').rdBase.rdkitVersion}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    source_hashes = {"script": sha256_file(Path(__file__).resolve()), "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py")}
    audit["source_hashes"] = source_hashes
    write_json(run_dir / "metrics.json", audit)
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([f"{source_hashes['script']}  SOURCE tools/round2_nc_lorentz_lorenz_ridge.py", f"{source_hashes['reference_module']}  SOURCE tools/initial_reference_pipeline.py"])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
