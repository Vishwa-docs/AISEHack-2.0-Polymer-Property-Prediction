#!/usr/bin/env python3
"""Official-only exact-group low-rank matrix-completion screen for EPS/Nc.

This is a component experiment.  It uses the canonical no-stereo group as an
exact identity kernel and fits a fixed rank-3 factor model to the official
structure-by-property matrix with masked cells.  It never consumes a held-out
target cell and falls back to the freshly regenerated reference carrier when a
group has no observed auxiliary property.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
CHANGED = ("eps", "nc")
SEED = 2026
RANK = 3
ALS_ITERATIONS = 18
ALS_RIDGE = 0.40
MATRIX_WEIGHT = 0.55


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def no_stereo(value: object) -> str:
    molecule = Chem.MolFromSmiles(str(value).replace("[*]", "*"))
    if molecule is None:
        return str(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def finite_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, prediction))


def group_folds(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError("incomplete group folds")
    return result


def bootstrap_lower(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {value: np.flatnonzero(groups == value) for value in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[value] for value in selected])
        if not np.isclose(np.var(y[rows]), 0.0):
            values.append(finite_r2(y[rows], candidate[rows]) - finite_r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def make_group_matrix(pooled: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    groups = sorted({no_stereo(value) for value in pooled["canonical"]})
    positions = {value: index for index, value in enumerate(groups)}
    matrix = np.full((len(groups), len(TARGETS)), np.nan, dtype=np.float64)
    for (group, target), frame in pooled.assign(group=pooled["canonical"].map(no_stereo)).groupby(["group", "target_type"]):
        matrix[positions[group], TARGETS.index(target)] = float(frame["target"].median())
    observed = np.isfinite(matrix)
    return groups, matrix, observed


def low_rank_fit(matrix: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a fixed-rank alternating least-squares model to observed cells."""
    if matrix.shape[1] != len(TARGETS):
        raise RuntimeError("unexpected property matrix width")
    means = np.zeros(matrix.shape[1], dtype=np.float64)
    scales = np.ones(matrix.shape[1], dtype=np.float64)
    standardized = np.zeros_like(matrix, dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[observed[:, column], column]
        means[column] = float(np.mean(values)) if len(values) else 0.0
        scales[column] = float(np.std(values)) if len(values) else 1.0
        if not np.isfinite(scales[column]) or scales[column] < 1.0e-8:
            scales[column] = 1.0
        standardized[observed[:, column], column] = (values - means[column]) / scales[column]
    rank = min(RANK, matrix.shape[0], matrix.shape[1])
    filled = standardized.copy()
    if rank == 0:
        raise RuntimeError("empty factor rank")
    left, singular, right = np.linalg.svd(filled, full_matrices=False)
    singular = np.maximum(singular[:rank], 1.0e-8)
    factors_u = left[:, :rank] * np.sqrt(singular)[None, :]
    factors_v = right[:rank, :].T * np.sqrt(singular)[None, :]
    identity = np.eye(rank, dtype=np.float64)
    for _ in range(ALS_ITERATIONS):
        # Fully vectorized normal equations for all group factors.  The
        # observation mask is retained in every contraction, so a masked cell
        # cannot contribute to either the row factor or the property loading.
        row_gram = np.einsum("ij,jr,js->irs", observed.astype(np.float64), factors_v, factors_v)
        row_gram += ALS_RIDGE * identity[None, :, :]
        row_rhs = np.einsum("ij,ij,jr->ir", observed.astype(np.float64), standardized, factors_v)
        factors_u = np.linalg.solve(row_gram, row_rhs[..., None])[..., 0]
        column_gram = np.einsum("ij,ir,is->jrs", observed.astype(np.float64), factors_u, factors_u)
        column_gram += ALS_RIDGE * identity[None, :, :]
        column_rhs = np.einsum("ij,ir,ij->jr", observed.astype(np.float64), factors_u, standardized)
        factors_v = np.linalg.solve(column_gram, column_rhs[..., None])[..., 0]
    return factors_u, factors_v, means, scales


def predict_group(
    matrix: np.ndarray,
    observed: np.ndarray,
    group_index: int,
    target_index: int,
) -> tuple[float, int]:
    """Mask an entire group-target cell and predict it from other properties."""
    masked = observed.copy()
    masked[group_index, target_index] = False
    support = int(np.sum(masked[group_index]))
    if support == 0:
        return float("nan"), support
    factors_u, factors_v, means, scales = low_rank_fit(matrix, masked)
    value = float(means[target_index] + scales[target_index] * np.dot(factors_u[group_index], factors_v[target_index]))
    if not np.isfinite(value):
        return float("nan"), support
    return value, support


def permuted_matrix(matrix: np.ndarray, observed: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shuffled = matrix.copy()
    shuffled_observed = observed.copy()
    for column in range(matrix.shape[1]):
        rows = np.flatnonzero(observed[:, column])
        if len(rows) > 1:
            shuffled_values = matrix[rng.permutation(rows), column]
            shuffled[rows, column] = shuffled_values
    return shuffled, shuffled_observed


def panel_deltas(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        rows = int(np.sum(selected))
        group_count = int(np.unique(groups[selected]).size)
        eligible = rows >= 20 and group_count >= 5
        report[name] = {
            "rows": rows,
            "groups": group_count,
            "eligible": bool(eligible),
            "delta_r2": finite_r2(y[selected], candidate[selected]) - finite_r2(y[selected], parent[selected]) if eligible else None,
        }
    scaffold_deltas: list[float] = []
    for value in sorted(set(scaffolds)):
        selected = scaffolds == value
        if int(np.sum(selected)) >= 10 and int(np.unique(groups[selected]).size) >= 3 and not np.isclose(np.var(y[selected]), 0.0):
            scaffold_deltas.append(finite_r2(y[selected], candidate[selected]) - finite_r2(y[selected], parent[selected]))
    report["scaffold_groups_ge_3"] = {
        "evaluated": len(scaffold_deltas),
        "minimum_delta_r2": min(scaffold_deltas) if scaffold_deltas else None,
    }
    eligible_deltas = [item["delta_r2"] for item in report.values() if isinstance(item, dict) and item.get("eligible") and item.get("delta_r2") is not None]
    if report["scaffold_groups_ge_3"]["minimum_delta_r2"] is not None:
        eligible_deltas.append(report["scaffold_groups_ge_3"]["minimum_delta_r2"])
    report["minimum_panel_delta"] = min(eligible_deltas) if eligible_deltas else 0.0
    return report


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
    group_names, group_matrix, group_observed = make_group_matrix(pooled)
    group_to_index = {value: index for index, value in enumerate(group_names)}
    target_reports: dict[str, Any] = {}
    component_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    predictions_by_id: dict[int, float] = {}
    rng = np.random.default_rng(SEED)

    for target in CHANGED:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(dtype=float)
        canonical = frame["canonical"].astype(str).to_numpy(dtype=object)
        groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
        target_index = TARGETS.index(target)
        global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[global_indices] = y
        dense = reference.target_dense_features(base_dense, cross_values, cross_available, target)
        kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
        parent_oof = np.full(len(y), np.nan, dtype=np.float64)
        for local_train, local_validation in kfold.split(np.arange(len(y))):
            parent_oof[local_validation] = np.mean(
                reference.predict_base_models(
                    dense,
                    sparse_parts,
                    fingerprints,
                    y_global,
                    global_indices[local_train],
                    global_indices[local_validation],
                    reference.DEFAULT_CONFIG,
                    target,
                ),
                axis=1,
            )
        matrix_oof = np.full(len(y), np.nan, dtype=np.float64)
        support = np.zeros(len(y), dtype=np.int64)
        for group in np.unique(groups):
            group_index = group_to_index[group]
            value, count = predict_group(group_matrix, group_observed, group_index, target_index)
            selected = groups == group
            matrix_oof[selected] = value
            support[selected] = count
        routed = np.isfinite(matrix_oof) & (support > 0)
        candidate_oof = parent_oof.copy()
        candidate_oof[routed] = (1.0 - MATRIX_WEIGHT) * parent_oof[routed] + MATRIX_WEIGHT * matrix_oof[routed]
        folds = np.full(len(y), -1, dtype=np.int64)
        for fold, (_, validation) in enumerate(kfold.split(np.arange(len(y)))):
            folds[validation] = fold
        fold_rows = []
        for fold in range(5):
            selected = folds == fold
            fold_rows.append({
                "fold": fold,
                "rows": int(np.sum(selected)),
                "parent_r2": finite_r2(y[selected], parent_oof[selected]),
                "candidate_r2": finite_r2(y[selected], candidate_oof[selected]),
                "delta_r2": finite_r2(y[selected], candidate_oof[selected]) - finite_r2(y[selected], parent_oof[selected]),
                "routed_rows": int(np.sum(routed[selected])),
            })
        scaffold = np.asarray([
            MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=False) or "ACYCLIC"
            for value in canonical
        ], dtype=object)
        similarity = np.full(len(y), np.nan, dtype=np.float64)
        for local_train, local_validation in kfold.split(np.arange(len(y))):
            train_fps = [fingerprints[global_indices[row]] for row in local_train]
            for row in local_validation:
                similarity[row] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[global_indices[row]], train_fps))
        panels = panel_deltas(y, candidate_oof, parent_oof, groups, similarity, scaffold)
        permuted, permuted_observed = permuted_matrix(group_matrix, group_observed, rng)
        permutation_prediction = np.full(len(y), np.nan, dtype=np.float64)
        permutation_support = np.zeros(len(y), dtype=np.int64)
        for group in np.unique(groups):
            group_index = group_to_index[group]
            value, count = predict_group(permuted, permuted_observed, group_index, target_index)
            selected = groups == group
            permutation_prediction[selected] = value
            permutation_support[selected] = count
        permutation_route = np.isfinite(permutation_prediction) & (permutation_support > 0)
        permutation_candidate = parent_oof.copy()
        permutation_candidate[permutation_route] = (1.0 - MATRIX_WEIGHT) * parent_oof[permutation_route] + MATRIX_WEIGHT * permutation_prediction[permutation_route]
        strict_groups = group_folds(groups)
        strict_parent = np.full(len(y), np.nan, dtype=np.float64)
        strict_candidate = np.full(len(y), np.nan, dtype=np.float64)
        strict_support = np.zeros(len(y), dtype=np.int64)
        for fold in range(5):
            validation = np.flatnonzero(strict_groups == fold)
            training = np.flatnonzero(strict_groups != fold)
            heldout = set(groups[validation].tolist())
            filtered = pooled[~pooled["canonical"].map(no_stereo).isin(heldout)]
            local_values, local_available = reference.cross_property_arrays(filtered, keys)
            local_dense = reference.target_dense_features(base_dense, local_values, local_available, target)
            strict_parent[validation] = np.mean(
                reference.predict_base_models(
                    local_dense,
                    sparse_parts,
                    fingerprints,
                    y_global,
                    global_indices[training],
                    global_indices[validation],
                    reference.DEFAULT_CONFIG,
                    target,
                ),
                axis=1,
            )
            strict_candidate[validation] = strict_parent[validation]
            strict_support[validation] = 0
        test_frame = test[test["target_type"] == target].reset_index(drop=True)
        test_indices = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
        parent_test = np.mean(
            reference.predict_base_models(
                dense,
                sparse_parts,
                fingerprints,
                y_global,
                global_indices,
                test_indices,
                reference.DEFAULT_CONFIG,
                target,
            ),
            axis=1,
        )
        test_groups = np.asarray([no_stereo(value) for value in test_frame["canonical"]], dtype=object)
        matrix_test = np.full(len(test_frame), np.nan, dtype=np.float64)
        test_support = np.zeros(len(test_frame), dtype=np.int64)
        for group in np.unique(test_groups):
            if group not in group_to_index:
                continue
            value, count = predict_group(group_matrix, group_observed, group_to_index[group], target_index)
            selected = test_groups == group
            matrix_test[selected] = value
            test_support[selected] = count
        test_routed = np.isfinite(matrix_test) & (test_support > 0)
        candidate_test = parent_test.copy()
        candidate_test[test_routed] = (1.0 - MATRIX_WEIGHT) * parent_test[test_routed] + MATRIX_WEIGHT * matrix_test[test_routed]
        for row, parent_value, matrix_value, candidate_value, count, route in zip(
            test_frame.itertuples(index=False), parent_test, matrix_test, candidate_test, test_support, test_routed, strict=True
        ):
            record = {
                "id": int(row.id),
                "target_type": target,
                "parent_prediction": float(parent_value),
                "matrix_prediction": float(matrix_value) if np.isfinite(matrix_value) else None,
                "candidate_prediction": float(candidate_value),
                "support_count": int(count),
                "routed": bool(route),
            }
            component_rows.append(record)
            predictions_by_id[int(row.id)] = float(candidate_value)
        for row, group, parent_value, matrix_value, candidate_value, count, route in zip(
            frame.itertuples(index=False), groups, parent_oof, matrix_oof, candidate_oof, support, routed, strict=True
        ):
            oof_rows.append({
                "canonical": str(row.canonical),
                "group": str(group),
                "target_type": target,
                "target": float(row.target),
                "parent": float(parent_value),
                "matrix": float(matrix_value) if np.isfinite(matrix_value) else None,
                "candidate": float(candidate_value),
                "support_count": int(count),
                "routed": bool(route),
            })
        target_reports[target] = {
            "rows": int(len(y)),
            "group_count": int(np.unique(groups).size),
            "entry_support_rows": int(np.sum(routed)),
            "entry_support_groups": int(np.unique(groups[routed]).size),
            "parent_r2": finite_r2(y, parent_oof),
            "candidate_r2": finite_r2(y, candidate_oof),
            "delta_r2": finite_r2(y, candidate_oof) - finite_r2(y, parent_oof),
            "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
            "group_bootstrap_lower": bootstrap_lower(y, candidate_oof, parent_oof, groups),
            "folds": fold_rows,
            "panels": panels,
            "permutation_control": {
                "routed_rows": int(np.sum(permutation_route)),
                "delta_r2": finite_r2(y, permutation_candidate) - finite_r2(y, parent_oof),
            },
            "strict_whole_group_mask": {
                "parent_r2": finite_r2(y, strict_parent),
                "candidate_r2": finite_r2(y, strict_candidate),
                "delta_r2": finite_r2(y, strict_candidate) - finite_r2(y, strict_parent),
                "support_rows": int(np.sum(strict_support > 0)),
            },
            "test_rows": int(len(test_frame)),
            "test_routed_rows": int(np.sum(test_routed)),
        }

    component = pd.DataFrame(component_rows).sort_values("id").reset_index(drop=True)
    if len(component) != sum(int(np.sum(test["target_type"] == target)) for target in CHANGED) or component["id"].duplicated().any() or not np.isfinite(component["candidate_prediction"].to_numpy(float)).all():
        raise RuntimeError("component output contract failed")
    component.to_csv(run_dir / "eps_nc_component_predictions.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(run_dir / "oof_predictions.csv", index=False)
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in CHANGED]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in CHANGED]))
    gates = {
        target: {
            "gain_pass": bool(target_reports[target]["delta_r2"] >= 0.010),
            "fold_pass": bool(target_reports[target]["positive_folds"] >= 4),
            "bootstrap_pass": bool(target_reports[target]["group_bootstrap_lower"] > 0.0),
            "panel_pass": bool(target_reports[target]["panels"]["minimum_panel_delta"] >= 0.0),
            "strict_fallback_pass": bool(abs(target_reports[target]["strict_whole_group_mask"]["delta_r2"]) <= 1.0e-12),
        }
        for target in CHANGED
    }
    passed = bool(all(all(values.values()) for values in gates.values()))
    report = {
        "schema_version": "ppp.round2.c055.structure-kernel-multitask.v1",
        "experiment_id": run_dir.name,
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7 (EPS/Nc component control regenerated from official inputs)",
        "created_at": datetime.now().astimezone().isoformat(),
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "changed_targets": list(CHANGED),
        "groups": int(len(group_names)),
        "observed_cells": int(np.sum(group_observed)),
        "exact_eps_nc_pairs": int(np.sum(group_observed[:, TARGETS.index("eps")] & group_observed[:, TARGETS.index("nc")])),
        "targets": target_reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "gates": gates,
        "decision": "pass_component_gate" if passed else "rejected_component_gate",
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "oof": int(len(oof_rows))},
        "source_sha256": sha256_file(root / "tools" / "round2_c055_structure_kernel_multitask.py"),
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {
        "seed": SEED,
        "rank": RANK,
        "als_iterations": ALS_ITERATIONS,
        "als_ridge": ALS_RIDGE,
        "matrix_weight": MATRIX_WEIGHT,
        "group_key": "canonical_no_stereo_exact_identity_kernel",
        "model": "masked rank-3 alternating-least-squares multi-output matrix completion",
        "changed_targets": list(CHANGED),
        "no_hyperparameter_sweep": True,
        "strict_group_fallback": True,
        "external_label_file_read": False,
    })
    (run_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        "# C055 decision\n\n"
        f"Entry-masking parent mean: {mean_parent:.12f}\n"
        f"Entry-masking candidate mean: {mean_candidate:.12f}\n"
        f"Entry-masking gain: {mean_candidate - mean_parent:+.12f}\n\n"
        f"Decision: {'PASS COMPONENT GATE' if passed else 'REJECT COMPONENT GATE'}\n\n"
        "The strict whole-group panel intentionally masks every label in each held-out no-stereo group; it must fall back exactly to the structural parent.\n"
        "No external_label file, local_eval value, prior prediction artifact, or external target was read.\n",
        encoding="utf-8",
    )
    manifest = [run_dir / name for name in ("protocol.json", "config.json", "metrics.json", "eps_nc_component_predictions.csv", "oof_predictions.csv", "environment.txt", "command.txt", "decision.md")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "targets": {target: target_reports[target]["delta_r2"] for target in CHANGED}}, sort_keys=True))


if __name__ == "__main__":
    main()
