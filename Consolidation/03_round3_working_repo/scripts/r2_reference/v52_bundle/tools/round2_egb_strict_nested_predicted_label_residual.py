#!/usr/bin/env python3
"""Strict official-only Egb predicted-label residual diagnostic."""

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
TARGET = "egb"
AUXILIARY = ("egc", "eea", "nc", "eps", "ei")
ALPHA = 10.0


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
    if np.any(folds < 0):
        raise RuntimeError("incomplete group fold assignment")
    return folds


def nearest_similarity(fingerprints: list[Any], validation: np.ndarray, training: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[index] for index in training]
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps)) for index in validation], dtype=np.float64)


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def grouped_bootstrap_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values: list[float] = []
    for _ in range(1000):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in chosen])
        if len(rows) > 1 and np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def build_auxiliary_tables(pooled: pd.DataFrame, key_to_index: dict[str, int]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for target in AUXILIARY:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        canonical = frame["canonical"].to_numpy(object)
        tables[target] = {
            "y": frame["target"].to_numpy(float),
            "global_indices": np.asarray([key_to_index[value] for value in canonical], dtype=np.int64),
            "groups": np.asarray([no_stereo(value) for value in canonical], dtype=object),
            "scaffolds": np.asarray([scaffold(value) for value in canonical], dtype=object),
            "available_globals": set(int(value) for value in [key_to_index[value] for value in canonical]),
            "rows": int(len(frame)),
        }
    return tables


def fit_auxiliary_prediction(
    table: dict[str, Any],
    features: np.ndarray,
    query_global: np.ndarray,
    excluded_groups: set[str],
    excluded_scaffolds: set[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    if np.any(query_global < 0) or np.any(query_global >= len(features)):
        raise RuntimeError("auxiliary query global index out of bounds")
    keep = np.asarray([
        group not in excluded_groups and scaffold_name not in excluded_scaffolds
        for group, scaffold_name in zip(table["groups"], table["scaffolds"], strict=True)
    ], dtype=bool)
    train_global = table["global_indices"][keep]
    train_y = table["y"][keep]
    if len(train_global) < 5 or len(np.unique(train_global)) < 4:
        return np.full(len(query_global), np.nan, dtype=np.float64), np.zeros(len(query_global), dtype=np.float64), int(np.sum(keep))
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=ALPHA),
    )
    model.fit(features[train_global], train_y)
    prediction = np.asarray(model.predict(features[query_global]), dtype=np.float64)
    available = np.asarray([int(index) in table["available_globals"] for index in query_global], dtype=np.float64)
    prediction[available == 0.0] = np.nan
    return prediction, available, int(np.sum(keep))


def auxiliary_features(
    tables: dict[str, dict[str, Any]],
    features: np.ndarray,
    query_global: np.ndarray,
    query_groups: np.ndarray,
    query_scaffolds: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(query_global) != len(query_groups) or len(query_global) != len(query_scaffolds):
        raise RuntimeError("auxiliary query metadata lengths differ")
    if len(set(int(value) for value in query_global)) != len(query_global):
        raise RuntimeError("duplicate query global indices in an auxiliary prediction batch")
    excluded_groups = set(str(value) for value in query_groups)
    excluded_scaffolds = set(str(value) for value in query_scaffolds)
    columns: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    fit_counts: dict[str, int] = {}
    for target in AUXILIARY:
        prediction, available, fit_count = fit_auxiliary_prediction(tables[target], features, query_global, excluded_groups, excluded_scaffolds)
        columns.append(prediction)
        availability.append(available)
        fit_counts[target] = fit_count
    matrix = np.column_stack(columns + availability).astype(np.float64, copy=False)
    return matrix, {"fit_counts": fit_counts, "availability_counts": {target: int(np.sum(availability[index])) for index, target in enumerate(AUXILIARY)}}


def parent_arms(dense, sparse_parts, fingerprints, y_global, train_global, validation_global, config):
    if np.any(train_global < 0) or np.any(validation_global < 0):
        raise RuntimeError("parent global index below zero")
    if np.any(train_global >= len(dense)) or np.any(validation_global >= len(dense)):
        raise RuntimeError("parent global index out of bounds")
    return reference.predict_base_models(dense, sparse_parts, fingerprints, y_global, train_global, validation_global, config, TARGET)


def nested_outer(
    y: np.ndarray,
    groups: np.ndarray,
    scaffolds: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    parent_dense: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    y_global: np.ndarray,
    global_indices: np.ndarray,
    auxiliary_tables: dict[str, dict[str, Any]],
    features: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    inner_folds = folds_for(groups[outer_train], 4)
    inner_parent_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    inner_aux = np.full((len(outer_train), len(AUXILIARY) * 2), np.nan, dtype=np.float64)
    inner_fit_reports: list[dict[str, Any]] = []
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_parent_arms[inner_folds == fold] = parent_arms(
            parent_dense, sparse_parts, fingerprints, y_global,
            global_indices[local_train], global_indices[local_validation], config,
        )
        aux_matrix, aux_report = auxiliary_features(
            auxiliary_tables, features, global_indices[local_validation],
            groups[local_validation], scaffolds[local_validation],
        )
        inner_aux[inner_folds == fold] = aux_matrix
        inner_fit_reports.append({"fold": fold, "validation_rows": int(len(local_validation)), "excluded_groups": int(len(set(groups[local_validation]))), "excluded_scaffolds": int(len(set(scaffolds[local_validation]))), "auxiliary": aux_report})
    weights, intercept, blend_name, inner_parent_r2 = reference.blend_from_oof(y[outer_train], inner_parent_arms)
    inner_parent = inner_parent_arms @ weights + intercept
    outer_parent_arms = parent_arms(
        parent_dense, sparse_parts, fingerprints, y_global,
        global_indices[outer_train], global_indices[outer_validation], config,
    )
    outer_parent = outer_parent_arms @ weights + intercept
    outer_aux, outer_aux_report = auxiliary_features(
        auxiliary_tables, features, global_indices[outer_validation],
        groups[outer_validation], scaffolds[outer_validation],
    )
    residual_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=ALPHA),
    )
    residual_target = y[outer_train] - inner_parent
    residual_model.fit(inner_aux, residual_target)
    residual_prediction = np.asarray(residual_model.predict(outer_aux), dtype=np.float64)
    support = np.sum(np.isfinite(outer_aux[:, :len(AUXILIARY)]), axis=1) > 0
    candidate = outer_parent.copy()
    candidate[support] = outer_parent[support] + residual_prediction[support]
    if not np.isfinite(candidate).all() or np.any(np.abs(candidate[support] - outer_parent[support]) > 1.0e6):
        raise RuntimeError("non-finite or explosive Egb residual prediction")
    return {
        "parent": outer_parent,
        "candidate": candidate,
        "support": support,
        "inner_aux": inner_aux,
        "outer_aux": outer_aux,
        "weights": [float(value) for value in weights],
        "intercept": float(intercept),
        "blend_name": blend_name,
        "inner_parent_r2": float(inner_parent_r2),
        "inner_folds": inner_folds.tolist(),
        "inner_fit_reports": inner_fit_reports,
        "outer_aux_report": outer_aux_report,
        "residual_mean": float(np.mean(residual_target)),
        "residual_std": float(np.std(residual_target)),
    }


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
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError(f"a protocol-only run directory is required: {run_dir}")
    started = time.time()
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    features = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    features[~np.isfinite(features)] = np.nan
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    parent_dense = reference.target_dense_features(features, cross_values, cross_available, TARGET)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.text_matrix(keys, int(reference.DEFAULT_CONFIG["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"]))
    auxiliary_tables = build_auxiliary_tables(pooled, key_to_index)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    canonical = frame["canonical"].to_numpy(object)
    groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([scaffold(value) for value in canonical], dtype=object)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    if len(set(int(value) for value in global_indices)) != len(global_indices):
        raise RuntimeError("Egb target global indices are not unique")
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[global_indices] = y
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "egb_predicted_label_alpha": ALPHA, "auxiliary_targets": list(AUXILIARY)})
    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan)
    candidate = np.full(len(y), np.nan)
    support = np.zeros(len(y), dtype=bool)
    nearest = np.full(len(y), np.nan)
    fold_rows: list[dict[str, Any]] = []
    fold_aux_reports: list[dict[str, Any]] = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        result = nested_outer(
            y, groups, scaffolds, training, validation, parent_dense, sparse_parts,
            fingerprints, y_global, global_indices, auxiliary_tables, features, config,
        )
        baseline[validation] = result["parent"]
        candidate[validation] = result["candidate"]
        support[validation] = result["support"]
        nearest[validation] = nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
        base_r2 = float(r2_score(y[validation], result["parent"]))
        candidate_r2 = float(r2_score(y[validation], result["candidate"]))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "baseline_r2": base_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - base_r2, "support_rows": int(np.sum(result["support"])), "validation_groups": sorted(set(groups[validation]))})
        fold_aux_reports.append({"fold": fold, "inner_folds": result["inner_folds"], "inner_fit_reports": result["inner_fit_reports"], "outer_aux_report": result["outer_aux_report"], "blend_weights": result["weights"], "blend_intercept": result["intercept"], "blend_name": result["blend_name"], "inner_parent_r2": result["inner_parent_r2"]})
    scaffold_holdout: dict[str, Any] = {}
    for scaffold_name in sorted(set(scaffolds)):
        validation = np.flatnonzero(scaffolds == scaffold_name)
        training = np.flatnonzero(scaffolds != scaffold_name)
        if len(validation) < 10 or len(np.unique(groups[training])) < 4:
            continue
        result = nested_outer(
            y, groups, scaffolds, training, validation, parent_dense, sparse_parts,
            fingerprints, y_global, global_indices, auxiliary_tables, features, config,
        )
        base_r2 = float(r2_score(y[validation], result["parent"]))
        candidate_r2 = float(r2_score(y[validation], result["candidate"]))
        scaffold_holdout[scaffold_name] = {"rows": int(len(validation)), "baseline_r2": base_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - base_r2, "support_rows": int(np.sum(result["support"]))}
    counts = pd.Series(groups).value_counts()
    panels: dict[str, Any] = {}
    panel_values: list[float] = []
    panel_masks = {
        "similarity_lt_0.30": nearest < 0.30,
        "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
        "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
        "similarity_ge_0.70": nearest >= 0.70,
        "auxiliary_support_zero_parent_only": ~support,
        "auxiliary_support_ge_1": support,
    }
    for name, selected in panel_masks.items():
        value = panel_delta(y, baseline, candidate, selected)
        parent_only_control = name == "auxiliary_support_zero_parent_only"
        unchanged = bool(np.max(np.abs(candidate[selected] - baseline[selected])) <= 1.0e-12) if np.any(selected) else True
        status = "inapplicable_zero_support" if value is None else "evaluable"
        if parent_only_control and np.any(selected) and not unchanged:
            status = "failed_parent_only_control"
        panels[name] = {"rows": int(np.sum(selected)), "delta_r2": value, "status": status, "unchanged_parent": unchanged}
        if value is not None:
            panel_values.append(value)
    panels["scaffold_slice_canonical_oof"] = {}
    for name in sorted(set(scaffolds)):
        selected = scaffolds == name
        value = panel_delta(y, baseline, candidate, selected)
        panels["scaffold_slice_canonical_oof"][name] = {"rows": int(np.sum(selected)), "delta_r2": value, "status": "evaluable" if value is not None else "inapplicable_zero_support"}
        if value is not None:
            panel_values.append(value)
    baseline_r2 = float(r2_score(y, baseline))
    candidate_r2 = float(r2_score(y, candidate))
    gain = candidate_r2 - baseline_r2
    bootstrap = grouped_bootstrap_lower(y, baseline, candidate, groups)
    scaffold_min = min((float(value["delta_r2"]) for value in scaffold_holdout.values()), default=None)
    min_panel = min(panel_values) if panel_values else None
    positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    incomplete = any(
        (
            any(item["delta_r2"] is None and item["rows"] >= 5 for item in value.values())
            if name == "scaffold_slice_canonical_oof"
            else value["delta_r2"] is None and value["rows"] >= 5
        )
        for name, value in panels.items()
        if name != "auxiliary_support_zero_parent_only"
    ) or any(value["delta_r2"] is None for value in scaffold_holdout.values())
    parent_control_pass = panels["auxiliary_support_zero_parent_only"]["status"] != "failed_parent_only_control"
    passed = bool(gain >= 0.01 and positive_folds >= 4 and bootstrap > 0.0 and not incomplete and parent_control_pass and (min_panel is None or min_panel >= 0.0) and (scaffold_min is None or scaffold_min >= 0.0))
    report = {
        "rows": int(len(y)), "canonical_groups": int(len(np.unique(groups))), "auxiliary_targets": list(AUXILIARY),
        "baseline_r2_nested_parent": baseline_r2, "candidate_r2_cross_fitted_predicted_label_residual": candidate_r2,
        "delta_r2": gain, "positive_outer_folds": positive_folds, "group_r2_bootstrap_lower": bootstrap,
        "outer_folds": fold_rows, "auxiliary_fold_reports": fold_aux_reports, "panels": panels,
        "scaffold_holdout": scaffold_holdout, "scaffold_holdout_min_delta": scaffold_min,
        "min_panel_delta": min_panel, "panel_incomplete": incomplete,
        "support_counts": {"target_rows": int(len(y)), "support_rows": int(np.sum(support)), "zero_support_rows": int(np.sum(~support)), "canonical_groups": int(len(np.unique(groups)))},
        "route_definition": {"zero_support_rows_parent_only": True, "max_zero_support_change": float(np.max(np.abs(candidate[~support] - baseline[~support]))) if np.any(~support) else 0.0},
        "pass": passed, "decision": "component_pass" if passed else "rejected_component_gate",
    }
    pd.DataFrame({"canonical": canonical, "target_type": TARGET, "target": y, "baseline": baseline, "candidate": candidate, "support": support, "nearest_similarity": nearest, "scaffold": scaffolds, "no_stereo_group": groups, "outer_fold": main_folds}).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.egb-strict-nested-predicted-label-residual.v1", "seed": 2026, "outer_folds": 5, "inner_folds": 4, "alpha": ALPHA, "auxiliary_targets": list(AUXILIARY), "features": "official descriptor_matrix + physical_matrix only for auxiliary predictors", "official_inputs": inputs})
    source_hashes = {"script": sha256_file(Path(__file__).resolve()), "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py")}
    metrics = {"schema_version": "ppp.round2.egb-strict-nested-predicted-label-residual-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C046-20260803-2230-nc-lorentz-lorenz-ridge", "lineage_parent": "R2-C001-20260803-1645-initial-reference-repaired", "official_inputs": inputs, "target": TARGET, "auxiliary_targets": list(AUXILIARY), "model": {"auxiliary_predictor": "SimpleImputer -> StandardScaler -> Ridge(alpha=10)", "residual_model": "SimpleImputer -> StandardScaler -> Ridge(alpha=10)", "residual_target": "Egb minus same outer-fold weighted inner-Oof parent"}, "features": {"auxiliary_predictor_features": int(features.shape[1]), "predicted_label_features": len(AUXILIARY) * 2}, "metrics": report, "source_hashes": source_hashes, "config_sha256": reference.canonical_json_hash(config), "pass": passed, "decision": report["decision"], "elapsed_seconds": float(time.time() - started)}
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "run.log").write_text(f"experiment_id={run_dir.name}\ntarget={TARGET}\nrows={len(y)}\nauxiliary_targets={','.join(AUXILIARY)}\ndelta_r2={gain:.12f}\npositive_outer_folds={positive_folds}\nbootstrap_lower={bootstrap:.12f}\nscaffold_holdout_min={scaffold_min}\npass={passed}\nelapsed_seconds={metrics['elapsed_seconds']:.3f}\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. No candidate or local_eval diagnostic was created by this component run.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([f"{source_hashes['script']}  SOURCE tools/round2_egb_strict_nested_predicted_label_residual.py", f"{source_hashes['reference_module']}  SOURCE tools/initial_reference_pipeline.py"])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "delta_r2": gain, "passing": passed, "elapsed_seconds": metrics["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
