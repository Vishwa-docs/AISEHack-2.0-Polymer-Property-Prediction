#!/usr/bin/env python3
"""Direct Egb periodic/electronic absolute specialist."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGET = "egb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def masked_target_dense(base_dense, cross_values, cross_available, target):
    return np.hstack([base_dense, np.full_like(cross_values, np.nan), np.zeros_like(cross_available)]).astype(np.float64, copy=False)


def folds_for(groups, n_splits):
    if len(np.unique(groups)) < n_splits:
        raise RuntimeError(f"need {n_splits} groups, found {len(np.unique(groups))}")
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def parent_arms(dense, sparse_parts, fingerprints, y_global, train_global, validation_global, config):
    return reference.predict_base_models(dense, sparse_parts, fingerprints, y_global, train_global, validation_global, config, TARGET)


def fit_specialist(features, y_train, train_global, validation_global):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = np.asarray(features[train_global], dtype=np.float64).copy()
    validation_x = np.asarray(features[validation_global], dtype=np.float64).copy()
    limit = float(reference.DEFAULT_CONFIG["dense_abs_limit"])
    train_x[~np.isfinite(train_x) | (np.abs(train_x) > limit)] = np.nan
    validation_x[~np.isfinite(validation_x) | (np.abs(validation_x) > limit)] = np.nan
    train_x = imputer.fit_transform(train_x)
    validation_x = imputer.transform(validation_x)
    keep = np.ptp(train_x, axis=0) > 1.0e-12
    if not np.any(keep):
        raise RuntimeError("no nonconstant periodic/electronic features remained")
    model = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=2,
        max_features=0.60,
        bootstrap=False,
        random_state=2026,
        n_jobs=2,
    )
    model.fit(train_x[:, keep], y_train)
    return np.asarray(model.predict(validation_x[:, keep]), dtype=np.float64), int(np.sum(keep))


def nested_split(y, groups, outer_train, outer_validation, dense, sparse_parts, fingerprints, y_global, global_indices, features, config):
    inner_folds = folds_for(groups[outer_train], 4)
    inner_parent_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_parent_arms[inner_folds == fold] = parent_arms(dense, sparse_parts, fingerprints, y_global, global_indices[local_train], global_indices[local_validation], config)
    weights, intercept, blend_name, inner_blend_r2 = reference.blend_from_oof(y[outer_train], inner_parent_arms)
    outer_parent_arms = parent_arms(dense, sparse_parts, fingerprints, y_global, global_indices[outer_train], global_indices[outer_validation], config)
    outer_parent = reference.clip_prediction(y[outer_train], outer_parent_arms @ weights + intercept)
    specialist, kept = fit_specialist(features, y[outer_train], global_indices[outer_train], global_indices[outer_validation])
    return {"parent": outer_parent, "candidate": specialist, "kept_features": kept, "inner_blend_r2": float(inner_blend_r2), "blend_name": blend_name, "weights": weights.tolist(), "intercept": float(intercept)}


def nearest_similarity(fingerprints, validation_global, training_global):
    train_fps = [fingerprints[index] for index in training_global]
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps)) for index in validation_global], dtype=np.float64)


def panel_delta(y, baseline, candidate, selected):
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


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
    base_desc, base_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([base_desc, physical]).astype(np.float64, copy=False)
    tools_dir = root.parent / "Polymer Prediction Challenge" / "tools"
    sys.path.insert(0, str(tools_dir))
    import polymer_official_train_eval_loop as rich

    electronic, electronic_names, electronic_report = rich.electronic_tail_feature_matrix(molecules)
    huckel, huckel_names, huckel_report = rich.huckel_spectrum_feature_matrix(molecules)
    conjugation, conjugation_names, conjugation_report = rich.conjugation_feature_matrix(molecules)
    infinite, infinite_names, infinite_report = rich.infinite_chain_proxy_feature_matrix(keys, molecules)
    rich_dense = np.hstack([base_dense, electronic, huckel, conjugation, infinite]).astype(np.float64, copy=False)
    rich_dense[~np.isfinite(rich_dense)] = np.nan
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    dense = masked_target_dense(rich_dense, cross_values, cross_available, TARGET)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
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
    kept_features = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        result = nested_split(y, groups, training, validation, dense, sparse_parts, fingerprints, y_global, indices, rich_dense, config)
        baseline[validation] = result["parent"]
        candidate[validation] = result["candidate"]
        nearest[validation] = nearest_similarity(fingerprints, indices[validation], indices[training])
        base_r2 = float(r2_score(y[validation], result["parent"]))
        cand_r2 = float(r2_score(y[validation], result["candidate"]))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "baseline_r2": base_r2, "candidate_r2": cand_r2, "delta_r2": cand_r2 - base_r2})
        kept_features.append(result["kept_features"])
    scaffold_groups = [name for name in sorted(set(scaffolds)) if int(np.sum(scaffolds == name)) >= 10]
    scaffold_holdout = {}
    for name in scaffold_groups:
        validation = np.flatnonzero(scaffolds == name)
        training = np.flatnonzero(scaffolds != name)
        result = nested_split(y, groups, training, validation, dense, sparse_parts, fingerprints, y_global, indices, rich_dense, config)
        base_r2 = float(r2_score(y[validation], result["parent"]))
        cand_r2 = float(r2_score(y[validation], result["candidate"]))
        scaffold_holdout[name] = {"rows": int(len(validation)), "baseline_r2": base_r2, "candidate_r2": cand_r2, "delta_r2": cand_r2 - base_r2}
    panels = {}
    panel_values = []
    for name, lower, upper in (("lt_0.30", 0.0, 0.30), ("0.30_0.50", 0.30, 0.50), ("0.50_0.70", 0.50, 0.70), ("ge_0.70", 0.70, 1.01)):
        selected = (nearest >= lower) & (nearest < upper)
        delta = panel_delta(y, baseline, candidate, selected)
        panels[f"similarity_{name}"] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "incomplete"}
        if delta is not None:
            panel_values.append(delta)
    counts = pd.Series(groups).value_counts()
    duplicate = np.asarray([counts[group] >= 2 for group in groups], dtype=bool)
    exact_delta = panel_delta(y, baseline, candidate, duplicate)
    panels["exact_archive_duplicate_group"] = {"rows": int(np.sum(duplicate)), "delta_r2": exact_delta, "status": "evaluable" if exact_delta is not None else "inapplicable_zero_support"}
    singleton = ~duplicate
    singleton_delta = panel_delta(y, baseline, candidate, singleton)
    panels["sparse_singleton_support"] = {"rows": int(np.sum(singleton)), "delta_r2": singleton_delta, "status": "evaluable" if singleton_delta is not None else "support_incomplete"}
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
    bootstrap = plumbing.bootstrap_r2_lower(y, baseline, candidate, groups)
    required_incomplete = any(value["delta_r2"] is None for key, value in panels.items() if key.startswith("similarity_")) or any(value["delta_r2"] is None for value in scaffold_holdout.values())
    panel_values.extend(float(value["delta_r2"]) for value in scaffold_holdout.values())
    min_panel = min(panel_values) if panel_values else None
    passed = bool(delta_r2 >= 0.01 and sum(row["delta_r2"] > 0.0 for row in fold_rows) >= 4 and bootstrap > 0.0 and not required_incomplete and (min_panel is None or min_panel >= -0.003))
    rich_path = tools_dir / "polymer_official_train_eval_loop.py"
    source_hashes = {"script": sha256_file(Path(__file__).resolve()), "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py"), "metric_plumbing": sha256_file(root / "tools" / "round2_eea_cross_target_oof_residual_stack.py"), "official_feature_module": sha256_file(rich_path)}
    audit = {
        "schema_version": "ppp.round2.egb-periodic-electronic-absolute-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C040-MASKED-EGB-PARENT-GENERATED-IN-RUN",
        "official_inputs": inputs,
        "target": TARGET,
        "model": "SimpleImputer(median) -> ExtraTreesRegressor(n_estimators=800, min_samples_leaf=2, max_features=0.60, bootstrap=false, random_state=2026, n_jobs=2)",
        "feature_count": int(rich_dense.shape[1]),
        "features": {"base": len(base_names) + len(physical_names), "electronic": len(electronic_names), "huckel": len(huckel_names), "conjugation": len(conjugation_names), "infinite_chain": len(infinite_names)},
        "feature_reports": {"electronic": electronic_report, "huckel": huckel_report, "conjugation": conjugation_report, "infinite_chain": infinite_report},
        "baseline_r2_nested_parent": baseline_r2,
        "candidate_r2_periodic_electronic": candidate_r2,
        "delta_r2": delta_r2,
        "positive_outer_folds": int(sum(row["delta_r2"] > 0.0 for row in fold_rows)),
        "group_r2_bootstrap_lower": bootstrap,
        "outer_folds": fold_rows,
        "kept_features_per_fold": kept_features,
        "panels": panels,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": min((float(value["delta_r2"]) for value in scaffold_holdout.values()), default=None),
        "min_panel_delta": min_panel,
        "panel_incomplete": required_incomplete,
        "support_counts": {"target_rows": int(len(y)), "canonical_groups": int(len(counts)), "exact_archive_duplicate_rows": int(np.sum(duplicate)), "sparse_singleton_rows": int(np.sum(singleton))},
        "source_hashes": source_hashes,
        "pass": passed,
        "decision": "component_pass" if passed else "rejected_component_gate",
        "elapsed_seconds": float(time.time() - started),
    }
    pd.DataFrame({"canonical_no_stereo_group": groups, "fold": main_folds, "scaffold": scaffolds, "nearest_tanimoto": nearest, "y": y, "nested_parent": baseline, "periodic_electronic": candidate}).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.egb-periodic-electronic-absolute.v1", "seed": 2026, "outer_folds": 5, "inner_folds": 4, "feature_count": int(rich_dense.shape[1]), "model": audit["model"], "official_inputs": inputs, "source_hashes": source_hashes})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C040 Egb periodic/electronic absolute\n\nDecision: **{audit['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([f"{source_hashes['script']}  SOURCE tools/round2_egb_periodic_electronic_absolute.py", f"{source_hashes['reference_module']}  SOURCE tools/initial_reference_pipeline.py", f"{source_hashes['metric_plumbing']}  SOURCE tools/round2_eea_cross_target_oof_residual_stack.py", f"{source_hashes['official_feature_module']}  SOURCE ../Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
