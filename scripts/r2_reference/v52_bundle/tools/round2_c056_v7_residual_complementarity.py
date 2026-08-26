#!/usr/bin/env python3
"""Exact-v7-parent nested residual-complementarity screen for EPS/Nc.

The candidate is a fixed, abstaining meta-router over the four official v7
base-model arms.  Router fitting uses only inner out-of-fold arm predictions;
the outer comparison uses the exact C001/v7 parent weights regenerated from
official inputs.  This is a component screen, not a submission package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
CHANGED = ("eps", "nc")
MODEL_NAMES = ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")
SEED = 2026
INNER_FOLDS = 4
ROUTER_ALPHA = 20.0
ROUTER_WEIGHT = 0.50
SIMILARITY_MIN = 0.35
ARM_STD_MAX_FRACTION = 0.30
PREDICTION_DELTA_MAX_FRACTION = 0.50


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def no_stereo(value: object) -> str:
    molecule = Chem.MolFromSmiles(str(value).replace("[*]", "*"))
    if molecule is None:
        return str(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, prediction))


def bootstrap_lower(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if not np.isclose(np.var(y[rows]), 0.0):
            values.append(r2(y[rows], candidate[rows]) - r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def nearest_similarity(fingerprints: list[Any], query: np.ndarray, train: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[int(index)] for index in train]
    result = np.zeros(len(query), dtype=np.float64)
    for row, index in enumerate(query):
        result[row] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps))
    return result


def router_features(arms: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    mean = np.mean(arms, axis=1)
    std = np.std(arms, axis=1)
    spread = np.max(arms, axis=1) - np.min(arms, axis=1)
    return np.column_stack([arms, mean, std, spread, similarity])


def fit_router(inner_arms: np.ndarray, inner_similarity: np.ndarray, inner_y: np.ndarray) -> tuple[Any, float]:
    features = router_features(inner_arms, inner_similarity)
    model = make_pipeline(StandardScaler(), Ridge(alpha=ROUTER_ALPHA))
    model.fit(features, inner_y)
    spread = max(float(np.subtract(*np.quantile(inner_y, [0.75, 0.25]))), float(np.std(inner_y)), 1.0e-8)
    return model, spread


def apply_router(
    model: Any,
    spread: float,
    arms: np.ndarray,
    similarity: np.ndarray,
    parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = router_features(arms, similarity)
    routed_prediction = np.asarray(model.predict(features), dtype=np.float64)
    route = (
        np.isfinite(routed_prediction)
        & (similarity >= SIMILARITY_MIN)
        & (np.std(arms, axis=1) <= ARM_STD_MAX_FRACTION * spread)
        & (np.abs(routed_prediction - parent) <= PREDICTION_DELTA_MAX_FRACTION * spread)
    )
    candidate = parent.copy()
    candidate[route] = (1.0 - ROUTER_WEIGHT) * parent[route] + ROUTER_WEIGHT * routed_prediction[route]
    return candidate, route, routed_prediction


def panel_report(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    deltas: list[float] = []
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        rows = int(np.sum(selected))
        group_count = int(np.unique(groups[selected]).size)
        eligible = rows >= 20 and group_count >= 5
        delta = r2(y[selected], candidate[selected]) - r2(y[selected], parent[selected]) if eligible else None
        report[name] = {"rows": rows, "groups": group_count, "eligible": bool(eligible), "delta_r2": delta}
        if delta is not None:
            deltas.append(delta)
    scaffold_values: list[float] = []
    for scaffold in sorted(set(scaffolds)):
        selected = scaffolds == scaffold
        if int(np.sum(selected)) >= 10 and int(np.unique(groups[selected]).size) >= 3 and not np.isclose(np.var(y[selected]), 0.0):
            scaffold_values.append(r2(y[selected], candidate[selected]) - r2(y[selected], parent[selected]))
    report["scaffold_groups_ge_3"] = {"evaluated": len(scaffold_values), "minimum_delta_r2": min(scaffold_values) if scaffold_values else None}
    if scaffold_values:
        deltas.append(min(scaffold_values))
    report["minimum_panel_delta"] = min(deltas) if deltas else 0.0
    return report


def masked_cross_arrays(pooled: pd.DataFrame, keys: list[str], excluded_groups: set[str]) -> tuple[np.ndarray, np.ndarray]:
    if not excluded_groups:
        return reference.cross_property_arrays(pooled, keys)
    groups = pooled["canonical"].map(no_stereo)
    return reference.cross_property_arrays(pooled.loc[~groups.isin(excluded_groups)], keys)


def nested_screen(
    target: str,
    pooled: pd.DataFrame,
    keys: list[str],
    key_to_index: dict[str, int],
    base_dense: np.ndarray,
    global_cross_values: np.ndarray,
    global_cross_available: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    y_parent: np.ndarray,
    parent_weights: np.ndarray,
    parent_intercept: float,
    strict: bool,
) -> dict[str, Any]:
    frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
    y = frame["target"].to_numpy(dtype=float)
    canonical = frame["canonical"].astype(str).to_numpy(dtype=object)
    groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[global_indices] = y
    target_dense_global = reference.target_dense_features(base_dense, global_cross_values, global_cross_available, target)
    if strict:
        outer_splitter = GroupKFold(n_splits=5)
        outer_iterator = outer_splitter.split(np.arange(len(y)), groups=groups)
    else:
        outer_splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
        outer_iterator = outer_splitter.split(np.arange(len(y)))
    parent_oof = np.full(len(y), np.nan, dtype=np.float64)
    candidate_oof = np.full(len(y), np.nan, dtype=np.float64)
    router_oof = np.full(len(y), np.nan, dtype=np.float64)
    similarity_oof = np.full(len(y), np.nan, dtype=np.float64)
    route_oof = np.zeros(len(y), dtype=bool)
    fold_rows: list[dict[str, Any]] = []
    for fold, (outer_train, validation) in enumerate(outer_iterator):
        outer_train = np.asarray(outer_train, dtype=np.int64)
        validation = np.asarray(validation, dtype=np.int64)
        excluded_outer = set(groups[validation].tolist()) if strict else set()
        cross_values, cross_available = masked_cross_arrays(pooled, keys, excluded_outer)
        target_dense_outer = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        inner_groups = groups[outer_train]
        if strict:
            inner_splitter = GroupKFold(n_splits=INNER_FOLDS)
            inner_iterator = inner_splitter.split(np.arange(len(outer_train)), groups=inner_groups)
        else:
            inner_splitter = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold + 1)
            inner_iterator = inner_splitter.split(np.arange(len(outer_train)))
        inner_arms = np.full((len(outer_train), len(MODEL_NAMES)), np.nan, dtype=np.float64)
        inner_similarity = np.full(len(outer_train), np.nan, dtype=np.float64)
        for inner_fold, (inner_local_train, inner_local_validation) in enumerate(inner_iterator):
            inner_local_train = np.asarray(inner_local_train, dtype=np.int64)
            inner_local_validation = np.asarray(inner_local_validation, dtype=np.int64)
            inner_train = outer_train[inner_local_train]
            inner_validation = outer_train[inner_local_validation]
            excluded_inner = set(groups[inner_validation].tolist()) if strict else set()
            inner_values, inner_available = masked_cross_arrays(pooled, keys, excluded_outer | excluded_inner)
            inner_dense = reference.target_dense_features(base_dense, inner_values, inner_available, target)
            inner_arms[inner_local_validation] = reference.predict_base_models(
                inner_dense,
                sparse_parts,
                fingerprints,
                y_global,
                global_indices[inner_train],
                global_indices[inner_validation],
                reference.DEFAULT_CONFIG,
                target,
            )
            inner_similarity[inner_local_validation] = nearest_similarity(fingerprints, global_indices[inner_validation], global_indices[inner_train])
        if not np.isfinite(inner_arms).all() or not np.isfinite(inner_similarity).all():
            raise RuntimeError(f"inner OOF incomplete for {target} fold {fold}")
        model, spread = fit_router(inner_arms, inner_similarity, y[outer_train])
        outer_arms = reference.predict_base_models(
            target_dense_outer,
            sparse_parts,
            fingerprints,
            y_global,
            global_indices[outer_train],
            global_indices[validation],
            reference.DEFAULT_CONFIG,
            target,
        )
        outer_similarity = nearest_similarity(fingerprints, global_indices[validation], global_indices[outer_train])
        if strict:
            outer_parent = outer_arms @ parent_weights + parent_intercept
        else:
            outer_parent = y_parent[validation]
        outer_candidate, outer_route, outer_router = apply_router(model, spread, outer_arms, outer_similarity, outer_parent)
        parent_oof[validation] = outer_parent
        candidate_oof[validation] = outer_candidate
        router_oof[validation] = outer_router
        similarity_oof[validation] = outer_similarity
        route_oof[validation] = outer_route
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "parent_r2": r2(y[validation], outer_parent),
            "candidate_r2": r2(y[validation], outer_candidate),
            "delta_r2": r2(y[validation], outer_candidate) - r2(y[validation], outer_parent),
            "routed_rows": int(np.sum(outer_route)),
            "outer_group_count": int(np.unique(groups[validation]).size),
        })
    scaffolds = np.asarray([
        MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=False) or "ACYCLIC"
        for value in canonical
    ], dtype=object)
    panels = panel_report(y, candidate_oof, parent_oof, groups, similarity_oof, scaffolds)
    return {
        "rows": int(len(y)),
        "group_count": int(np.unique(groups).size),
        "parent_r2": r2(y, parent_oof),
        "candidate_r2": r2(y, candidate_oof),
        "delta_r2": r2(y, candidate_oof) - r2(y, parent_oof),
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "group_bootstrap_lower": bootstrap_lower(y, candidate_oof, parent_oof, groups),
        "routed_rows": int(np.sum(route_oof)),
        "routed_groups": int(np.unique(groups[route_oof]).size),
        "folds": fold_rows,
        "panels": panels,
        "oof": {
            "canonical": canonical,
            "groups": groups,
            "y": y,
            "parent": parent_oof,
            "candidate": candidate_oof,
            "router": router_oof,
            "similarity": similarity_oof,
            "routed": route_oof,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} - {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=4096),
        reference.morgan_count_matrix(molecules, radius=3, bits=4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)

    # This is the authoritative v7/C001 parent reproduction.  No prior OOF or
    # prediction file is loaded; all parent arms are regenerated from official
    # inputs in this process.
    detail, parent_oof_frame, parent_report = reference.fit_targets(
        pooled,
        test,
        keys,
        base_dense,
        cross_values,
        cross_available,
        sparse_parts,
        fingerprints,
        reference.DEFAULT_CONFIG,
    )
    reports: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for target in CHANGED:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == target].reset_index(drop=True)
        if not np.array_equal(frame["canonical"].astype(str).to_numpy(), parent_rows["canonical"].astype(str).to_numpy()):
            raise RuntimeError(f"exact parent row alignment failed for {target}")
        parent_y = parent_rows["prediction"].to_numpy(dtype=float)
        weights = np.asarray([parent_report["target_reports"][target]["blend_weights"][name] for name in MODEL_NAMES], dtype=float)
        intercept = float(parent_report["target_reports"][target]["blend_intercept"])
        screen = nested_screen(
            target,
            pooled,
            keys,
            key_to_index,
            base_dense,
            cross_values,
            cross_available,
            sparse_parts,
            fingerprints,
            parent_y,
            weights,
            intercept,
            strict=False,
        )
        strict_screen = nested_screen(
            target,
            pooled,
            keys,
            key_to_index,
            base_dense,
            cross_values,
            cross_available,
            sparse_parts,
            fingerprints,
            parent_y,
            weights,
            intercept,
            strict=True,
        )
        test_parent = detail[detail["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_frame = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        full_inner_splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
        full_inner_arms = np.full((len(frame), len(MODEL_NAMES)), np.nan, dtype=np.float64)
        full_inner_similarity = np.full(len(frame), np.nan, dtype=np.float64)
        target_y = frame["target"].to_numpy(dtype=float)
        canonical = frame["canonical"].astype(str).to_numpy(dtype=object)
        groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
        global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[global_indices] = target_y
        target_dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        for inner_train, inner_validation in full_inner_splitter.split(np.arange(len(frame))):
            full_inner_arms[inner_validation] = reference.predict_base_models(
                target_dense,
                sparse_parts,
                fingerprints,
                y_global,
                global_indices[inner_train],
                global_indices[inner_validation],
                reference.DEFAULT_CONFIG,
                target,
            )
            full_inner_similarity[inner_validation] = nearest_similarity(fingerprints, global_indices[inner_validation], global_indices[inner_train])
        model, spread = fit_router(full_inner_arms, full_inner_similarity, target_y)
        test_indices = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
        test_arms = reference.predict_base_models(
            target_dense,
            sparse_parts,
            fingerprints,
            y_global,
            global_indices,
            test_indices,
            reference.DEFAULT_CONFIG,
            target,
        )
        test_similarity = nearest_similarity(fingerprints, test_indices, global_indices)
        test_parent_values = test_parent["model_prediction"].to_numpy(dtype=float)
        test_candidate, test_route, test_router = apply_router(model, spread, test_arms, test_similarity, test_parent_values)
        for row, parent_value, router_value, candidate_value, similarity, route in zip(
            test_frame.itertuples(index=False), test_parent_values, test_router, test_candidate, test_similarity, test_route, strict=True
        ):
            component_rows.append({
                "id": int(row.id),
                "target_type": target,
                "parent_prediction": float(parent_value),
                "router_prediction": float(router_value),
                "candidate_prediction": float(candidate_value),
                "similarity": float(similarity),
                "routed": bool(route),
            })
        oof = screen["oof"]
        for row, group, y_value, parent_value, router_value, candidate_value, similarity, route in zip(
            frame.itertuples(index=False), oof["groups"], oof["y"], oof["parent"], oof["router"], oof["candidate"], oof["similarity"], oof["routed"], strict=True
        ):
            oof_rows.append({
                "canonical": str(row.canonical),
                "group": str(group),
                "target_type": target,
                "target": float(y_value),
                "parent": float(parent_value),
                "router": float(router_value),
                "candidate": float(candidate_value),
                "similarity": float(similarity),
                "routed": bool(route),
            })
        reports[target] = {
            "entry_masking": {key: value for key, value in screen.items() if key != "oof"},
            "strict_group_masking": {key: value for key, value in strict_screen.items() if key != "oof"},
            "parent_weights": {name: float(value) for name, value in zip(MODEL_NAMES, weights, strict=True)},
            "parent_intercept": intercept,
            "test_rows": int(len(test_frame)),
            "test_routed_rows": int(np.sum(test_route)),
        }
    component = pd.DataFrame(component_rows).sort_values("id").reset_index(drop=True)
    expected = int(sum(np.sum(test["target_type"] == target) for target in CHANGED))
    if len(component) != expected or component["id"].duplicated().any() or not np.isfinite(component["candidate_prediction"].to_numpy(dtype=float)).all():
        raise RuntimeError("component output contract failed")
    component.to_csv(run_dir / "eps_nc_component_predictions.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(run_dir / "oof_predictions.csv", index=False)
    entry_mean_parent = float(np.mean([reports[target]["entry_masking"]["parent_r2"] for target in CHANGED]))
    entry_mean_candidate = float(np.mean([reports[target]["entry_masking"]["candidate_r2"] for target in CHANGED]))
    strict_mean_parent = float(np.mean([reports[target]["strict_group_masking"]["parent_r2"] for target in CHANGED]))
    strict_mean_candidate = float(np.mean([reports[target]["strict_group_masking"]["candidate_r2"] for target in CHANGED]))
    gates = {
        target: {
            "gain_pass": bool(reports[target]["entry_masking"]["delta_r2"] >= 0.010),
            "fold_pass": bool(reports[target]["entry_masking"]["positive_folds"] >= 4),
            "bootstrap_pass": bool(reports[target]["entry_masking"]["group_bootstrap_lower"] > 0.0),
            "panel_pass": bool(reports[target]["entry_masking"]["panels"]["minimum_panel_delta"] >= 0.0),
            "strict_no_regression": bool(reports[target]["strict_group_masking"]["delta_r2"] >= -0.003),
        }
        for target in CHANGED
    }
    passed = bool(all(all(values.values()) for values in gates.values()))
    report = {
        "schema_version": "ppp.round2.c056.v7-parent-residual-complementarity.v1",
        "experiment_id": run_dir.name,
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; EPS/Nc parent regenerated exactly from official inputs and v7 weights",
        "created_at": datetime.now().astimezone().isoformat(),
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "changed_targets": list(CHANGED),
        "targets": reports,
        "entry_masking_mean_parent_r2": entry_mean_parent,
        "entry_masking_mean_candidate_r2": entry_mean_candidate,
        "entry_masking_mean_gain": entry_mean_candidate - entry_mean_parent,
        "strict_group_mean_parent_r2": strict_mean_parent,
        "strict_group_mean_candidate_r2": strict_mean_candidate,
        "strict_group_mean_gain": strict_mean_candidate - strict_mean_parent,
        "gates": gates,
        "decision": "pass_component_gate" if passed else "rejected_component_gate",
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(oof_rows))},
        "source_sha256": sha256_file(root / "tools" / "round2_c056_v7_residual_complementarity.py"),
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {
        "seed": SEED,
        "changed_targets": list(CHANGED),
        "model": "inner-OOF standardized Ridge meta-router over exact v7 four-arm predictions",
        "router_alpha": ROUTER_ALPHA,
        "router_weight": ROUTER_WEIGHT,
        "similarity_min": SIMILARITY_MIN,
        "arm_std_max_fraction": ARM_STD_MAX_FRACTION,
        "prediction_delta_max_fraction": PREDICTION_DELTA_MAX_FRACTION,
        "outer_folds": "KFold(5, shuffle=true, random_state=2026) entry masking plus GroupKFold(5) strict group masking",
        "inner_folds": INNER_FOLDS,
        "no_hyperparameter_sweep": True,
        "prior_prediction_input": False,
        "external_label_file_read": False,
    })
    (run_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        "# C056 decision\n\n"
        f"Entry-mask mean parent: {entry_mean_parent:.12f}\n"
        f"Entry-mask mean candidate: {entry_mean_candidate:.12f}\n"
        f"Entry-mask mean gain: {entry_mean_candidate - entry_mean_parent:+.12f}\n"
        f"Strict-group mean gain: {strict_mean_candidate - strict_mean_parent:+.12f}\n\n"
        f"Decision: {'PASS COMPONENT GATE' if passed else 'REJECT COMPONENT GATE'}\n\n"
        "The parent was regenerated from official inputs using the exact v7/C001 OOF-selected arm weights. The router saw only inner out-of-fold arm predictions; no external_label file, local_eval value, or prior prediction artifact was read.\n",
        encoding="utf-8",
    )
    manifest = [run_dir / name for name in ("protocol.json", "config.json", "metrics.json", "eps_nc_component_predictions.csv", "oof_predictions.csv", "environment.txt", "command.txt", "decision.md")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "entry_masking_mean_parent_r2": entry_mean_parent, "entry_masking_mean_candidate_r2": entry_mean_candidate, "entry_masking_mean_gain": entry_mean_candidate - entry_mean_parent, "strict_group_mean_gain": strict_mean_candidate - strict_mean_parent, "targets": {target: reports[target]["entry_masking"]["delta_r2"] for target in CHANGED}}, sort_keys=True))


if __name__ == "__main__":
    main()
