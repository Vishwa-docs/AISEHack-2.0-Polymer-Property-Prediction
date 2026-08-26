#!/usr/bin/env python3
"""Strict nested EPS residual diagnostic using fold-local Nc predictions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import round2_eea_cross_target_oof_residual_stack as implementation


CURRENT_OUTER_GROUPS: set[str] = set()
CURRENT_OUTER_SCAFFOLDS: set[str] = set()
SCAFFOLD_BY_GLOBAL = np.asarray([], dtype=object)
CALL_INDEX = 0
MAIN_ROUTE_ROWS: list[dict[str, np.ndarray]] = []
ORIGINAL_BUILD_MOLECULES = implementation.reference.build_molecules
ORIGINAL_NESTED_SPLIT = implementation.nested_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_target_dense(base_dense, cross_values, cross_available, target):
    return np.hstack([
        base_dense,
        np.full_like(cross_values, np.nan),
        np.zeros_like(cross_available),
    ]).astype(np.float64, copy=False)


def capture_molecules(keys):
    global SCAFFOLD_BY_GLOBAL
    molecules = ORIGINAL_BUILD_MOLECULES(keys)
    SCAFFOLD_BY_GLOBAL = np.asarray([implementation.scaffold(value) for value in keys], dtype=object)
    return molecules


def strict_aux_prediction(target, forbidden_groups, prediction_global, aux_info, deterministic):
    info = aux_info[target]
    global_forbidden = set(forbidden_groups) | CURRENT_OUTER_GROUPS
    candidate_global = info["global_indices"]
    keep = np.asarray([
        group not in global_forbidden
        and str(SCAFFOLD_BY_GLOBAL[index]) not in CURRENT_OUTER_SCAFFOLDS
        for group, index in zip(info["groups"], candidate_global)
    ], dtype=bool)
    train_global = candidate_global[keep]
    train_y = info["y"][keep]
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=30.0),
    )
    model.fit(deterministic[train_global], train_y)
    return np.asarray(model.predict(deterministic[prediction_global]), dtype=np.float64)


def residual_features(parent: np.ndarray, auxiliary: np.ndarray, available: np.ndarray) -> np.ndarray:
    nc_prediction = np.asarray(auxiliary[:, 0], dtype=np.float64).copy()
    nc_prediction[available <= 0] = np.nan
    with np.errstate(invalid="ignore", over="ignore"):
        return np.column_stack([
            parent,
            nc_prediction,
            np.square(nc_prediction),
            parent * nc_prediction,
            nc_prediction - parent,
            available,
        ]).astype(np.float64, copy=False)


def strict_nested_split(
    y, groups, outer_train, outer_validation, dense, sparse_parts, fingerprints,
    y_global, global_indices, cross_available, aux_info, deterministic,
):
    global CURRENT_OUTER_GROUPS, CURRENT_OUTER_SCAFFOLDS, CALL_INDEX
    validation_global = global_indices[outer_validation]
    CURRENT_OUTER_GROUPS = set(groups[outer_validation])
    CURRENT_OUTER_SCAFFOLDS = set(str(SCAFFOLD_BY_GLOBAL[index]) for index in validation_global)
    implementation.aux_prediction = strict_aux_prediction
    try:
        inner_folds = implementation.folds_for(groups[outer_train], 4)
        inner_parent_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
        inner_aux = np.full((len(outer_train), 1), np.nan, dtype=np.float64)
        for fold in range(4):
            local_train = outer_train[np.flatnonzero(inner_folds != fold)]
            local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
            inner_parent_arms[inner_folds == fold] = implementation.parent_arms(
                dense, sparse_parts, fingerprints, y_global,
                global_indices[local_train], global_indices[local_validation],
            )
            forbidden = set(groups[local_validation])
            inner_aux[inner_folds == fold, 0] = strict_aux_prediction(
                "nc", forbidden, global_indices[local_validation], aux_info, deterministic,
            )
        outer_parent_arms = implementation.parent_arms(
            dense, sparse_parts, fingerprints, y_global,
            global_indices[outer_train], global_indices[outer_validation],
        )
        outer_aux = np.column_stack([
            strict_aux_prediction("nc", CURRENT_OUTER_GROUPS, validation_global, aux_info, deterministic)
        ])
    finally:
        CURRENT_OUTER_GROUPS = set()
        CURRENT_OUTER_SCAFFOLDS = set()

    weights, intercept, blend_name, inner_blend_r2 = implementation.reference.blend_from_oof(y[outer_train], inner_parent_arms)
    inner_parent = implementation.reference.clip_prediction(y[outer_train], inner_parent_arms @ weights + intercept)
    outer_parent = implementation.reference.clip_prediction(y[outer_train], outer_parent_arms @ weights + intercept)
    aux_columns = [implementation.reference.TARGETS.index("nc")]
    available_inner = cross_available[global_indices[outer_train]][:, aux_columns]
    available_outer = cross_available[validation_global][:, aux_columns]
    inner_features = residual_features(inner_parent, inner_aux, available_inner[:, 0])
    outer_features = residual_features(outer_parent, outer_aux, available_outer[:, 0])
    residual_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=30.0),
    )
    residual_model.fit(inner_features, y[outer_train] - inner_parent)
    correction = residual_model.predict(outer_features)
    candidate = implementation.reference.clip_prediction(y[outer_train], outer_parent + 0.5 * correction)
    eligible = (available_outer[:, 0] > 0)
    training_global = global_indices[outer_train]
    nearest = implementation.nearest_to_train(
        [fingerprints[index] for index in validation_global],
        [fingerprints[index] for index in training_global],
    )
    eligible &= nearest >= 0.30
    candidate[~eligible] = outer_parent[~eligible]
    result = {
        "parent": outer_parent,
        "candidate": candidate,
        "inner_blend_r2": float(inner_blend_r2),
        "blend_name": blend_name,
        "weights": weights.tolist(),
        "intercept": float(intercept),
    }
    if CALL_INDEX < 5:
        MAIN_ROUTE_ROWS.append({
            "y": np.asarray(y[outer_validation], dtype=np.float64),
            "parent": outer_parent.copy(),
            "candidate": candidate.copy(),
            "available": np.asarray(available_outer[:, 0], dtype=np.int64),
            "nearest": np.asarray(nearest, dtype=np.float64),
        })
    CALL_INDEX += 1
    return result


def panel_delta(rows, selected):
    if int(np.sum(selected)) < 5 or float(np.var(rows["y"][selected])) <= 1.0e-15:
        return None
    return float(r2_score(rows["y"][selected], rows["candidate"][selected]) - r2_score(rows["y"][selected], rows["parent"][selected]))


def main() -> None:
    implementation.TARGET = "eps"
    implementation.AUXILIARY = ("nc",)
    implementation.reference.target_dense_features = masked_target_dense
    implementation.reference.build_molecules = capture_molecules
    implementation.nested_split = strict_nested_split
    implementation.main()

    arguments = sys.argv[1:]
    run_dir = None
    for index, argument in enumerate(arguments):
        if argument == "--run-dir" and index + 1 < len(arguments):
            run_dir = Path(arguments[index + 1]).resolve()
            break
    if run_dir is None:
        raise RuntimeError("--run-dir is required")

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = {key: np.concatenate([item[key] for item in MAIN_ROUTE_ROWS]) for key in ("y", "parent", "candidate", "available", "nearest")}
    low_similarity = rows["nearest"] < 0.30
    missing_nc = rows["available"] <= 0
    eligible = ~low_similarity & ~missing_nc
    route_definition = {
        "minimum_available_auxiliary_count": 1,
        "minimum_nearest_tanimoto": 0.30,
        "correction_strength": 0.5,
        "main_rows": int(len(rows["y"])),
        "route_eligible_rows": int(np.sum(eligible)),
        "low_similarity_rows": int(np.sum(low_similarity)),
        "missing_nc_rows": int(np.sum(missing_nc)),
        "low_similarity_changed_rows": int(np.sum(low_similarity & (np.abs(rows["candidate"] - rows["parent"]) > 1.0e-12))),
        "missing_nc_changed_rows": int(np.sum(missing_nc & (np.abs(rows["candidate"] - rows["parent"]) > 1.0e-12))),
        "route_eligible_delta": panel_delta(rows, eligible),
        "low_similarity_control_delta": 0.0 if np.sum(low_similarity & (np.abs(rows["candidate"] - rows["parent"]) > 1.0e-12)) == 0 else None,
        "missing_nc_control_delta": 0.0 if np.sum(missing_nc & (np.abs(rows["candidate"] - rows["parent"]) > 1.0e-12)) == 0 else None,
    }
    metrics["schema_version"] = "ppp.round2.eps-nc-strict-cross-target-residual-run.v1"
    metrics["parent"] = "R2-C036-MASKED-EPS-PARENT-GENERATED-IN-RUN"
    metrics["target"] = "eps"
    metrics["auxiliary_targets"] = ["nc"]
    metrics["route_definition"] = route_definition
    metrics["panels"]["availability_declared_nc"] = {"rows": int(np.sum(~missing_nc)), "delta_r2": panel_delta(rows, ~missing_nc), "status": "evaluable" if panel_delta(rows, ~missing_nc) is not None else "insufficient_or_constant"}
    metrics["panels"]["availability_missing_nc"] = {"rows": int(np.sum(missing_nc)), "delta_r2": route_definition["missing_nc_control_delta"], "status": "unchanged_control"}
    metrics["panels"]["similarity_lt_0.30"]["delta_r2"] = route_definition["low_similarity_control_delta"]
    metrics["panels"]["similarity_lt_0.30"]["status"] = "unchanged_control"
    panel_values = []
    for value in metrics.get("panels", {}).values():
        items = value.values() if isinstance(value, dict) and "delta_r2" not in value else [value]
        for item in items:
            if item.get("delta_r2") is not None:
                panel_values.append(float(item["delta_r2"]))
    panel_values.extend(float(item["delta_r2"]) for item in metrics.get("scaffold_holdout", {}).values())
    metrics["min_panel_delta"] = min(panel_values) if panel_values else None
    metrics["panel_incomplete"] = False
    metrics["pass"] = bool(
        metrics["delta_r2"] >= 0.01
        and metrics["positive_outer_folds"] >= 4
        and metrics["group_r2_bootstrap_lower"] > 0.0
        and (metrics["min_panel_delta"] is None or metrics["min_panel_delta"] >= -0.003)
        and route_definition["route_eligible_delta"] is not None
        and route_definition["route_eligible_delta"] >= -0.003
        and route_definition["low_similarity_changed_rows"] == 0
        and route_definition["missing_nc_changed_rows"] == 0
    )
    metrics["decision"] = "component_pass" if metrics["pass"] else "rejected_component_gate"
    wrapper_path = Path(__file__).resolve()
    implementation_path = Path(implementation.__file__).resolve()
    metrics["source_hashes"]["wrapper"] = sha256_file(wrapper_path)
    metrics["source_hashes"]["shared_implementation"] = sha256_file(implementation_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"schema_version": "ppp.round2.eps-nc-strict-cross-target-residual.v1", "target": "eps", "auxiliary_targets": ["nc"], "strict_nested_boundary": True, "route": {"minimum_available_auxiliary_count": 1, "minimum_nearest_tanimoto": 0.30}, "source_hashes": metrics["source_hashes"]})
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C036 EPS-Nc strict cross-target residual\n\nDecision: **{metrics['decision']}**. No candidate or local_eval diagnostic was created by this bounded component run.\n", encoding="utf-8")
    (run_dir / "run.log").write_text("\n".join([
        f"experiment_id={run_dir.name}", "target=eps",
        f"nested_parent_r2={metrics['baseline_r2_nested_parent']:.12f}",
        f"candidate_r2={metrics['candidate_r2_cross_target_residual']:.12f}",
        f"delta_r2={metrics['delta_r2']:.12f}",
        f"route_eligible_rows={route_definition['route_eligible_rows']}",
        f"low_similarity_changed_rows={route_definition['low_similarity_changed_rows']}",
        f"missing_nc_changed_rows={route_definition['missing_nc_changed_rows']}",
        f"pass={metrics['pass']}",
    ]) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([
        f"{metrics['source_hashes']['wrapper']}  SOURCE tools/round2_eps_nc_strict_cross_target_residual.py",
        f"{metrics['source_hashes']['shared_implementation']}  SOURCE tools/round2_eea_cross_target_oof_residual_stack.py",
        f"{metrics['source_hashes']['reference_module']}  SOURCE tools/initial_reference_pipeline.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
