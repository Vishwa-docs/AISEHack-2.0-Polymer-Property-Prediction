#!/usr/bin/env python3
"""Strictly nested official-only Eea scaffold-conditioned residual component."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "eea"
AUXILIARY = ("egb", "egc", "nc", "eps", "ei")
CORRECTION_STRENGTH = 0.5
ROUTE_MIN_AUXILIARY = 2
ROUTE_MIN_SIMILARITY = 0.30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
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
    from sklearn.model_selection import GroupKFold

    if len(np.unique(groups)) < n_splits:
        raise RuntimeError(f"need {n_splits} groups, found {len(np.unique(groups))}")
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(500):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in chosen])
        if len(rows) > 1 and np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def nearest_similarity(fingerprints: list[object], validation: np.ndarray, training: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[index] for index in training]
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps)) for index in validation], dtype=float)


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def masked_target_dense(base_dense: np.ndarray, cross_values: np.ndarray, cross_available: np.ndarray) -> np.ndarray:
    return np.hstack([
        base_dense,
        np.full_like(cross_values, np.nan),
        np.zeros_like(cross_available),
    ]).astype(np.float64, copy=False)


def parent_arms(
    target_dense: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    y_global: np.ndarray,
    train_global: np.ndarray,
    prediction_global: np.ndarray,
    config: dict[str, object],
) -> np.ndarray:
    return reference.predict_base_models(
        target_dense, sparse_parts, fingerprints, y_global,
        train_global, prediction_global, config, TARGET,
    )


def aux_prediction(
    target: str,
    forbidden_groups: set[str],
    forbidden_scaffolds: set[str],
    prediction_global: np.ndarray,
    aux_info: dict[str, dict[str, object]],
    deterministic: np.ndarray,
) -> np.ndarray:
    info = aux_info[target]
    keep = np.asarray([
        group not in forbidden_groups and str(scaffold_name) not in forbidden_scaffolds
        for group, scaffold_name in zip(info["groups"], info["scaffolds"], strict=True)
    ], dtype=bool)
    train_global = info["global_indices"][keep]
    train_y = info["y"][keep]
    if len(train_global) < 8:
        return np.full(len(prediction_global), float(np.mean(train_y)), dtype=np.float64)
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=30.0),
    )
    model.fit(deterministic[train_global], train_y)
    prediction = np.asarray(model.predict(deterministic[prediction_global]), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"non-finite auxiliary prediction for {target}")
    return prediction


def scaffold_features(vectorizer: DictVectorizer, values: np.ndarray) -> np.ndarray:
    return vectorizer.transform([{"scaffold": str(value)} for value in values])


def nested_split(
    y: np.ndarray,
    groups: np.ndarray,
    scaffolds: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    target_dense: np.ndarray,
    deterministic: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    y_global: np.ndarray,
    global_indices: np.ndarray,
    cross_available: np.ndarray,
    aux_info: dict[str, dict[str, object]],
    config: dict[str, object],
    forbidden_scaffolds: set[str],
) -> dict[str, object]:
    inner_folds = folds_for(groups[outer_train], 4)
    inner_parent_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    inner_aux = np.full((len(outer_train), len(AUXILIARY)), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_parent_arms[inner_folds == fold] = parent_arms(
            target_dense, sparse_parts, fingerprints, y_global,
            global_indices[local_train], global_indices[local_validation], config,
        )
        fold_forbidden_groups = set(groups[local_validation]) | set(groups[outer_validation])
        fold_forbidden_scaffolds = set(forbidden_scaffolds) | set(scaffolds[local_validation]) | set(scaffolds[outer_validation])
        for column, auxiliary in enumerate(AUXILIARY):
            inner_aux[inner_folds == fold, column] = aux_prediction(
                auxiliary, fold_forbidden_groups, fold_forbidden_scaffolds,
                global_indices[local_validation], aux_info, deterministic,
            )
    weights, intercept, blend_name, inner_parent_r2 = reference.blend_from_oof(y[outer_train], inner_parent_arms)
    inner_parent = reference.clip_prediction(y[outer_train], inner_parent_arms @ weights + intercept)
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit([{"scaffold": str(value)} for value in scaffolds[outer_train]])
    aux_columns = [reference.TARGETS.index(value) for value in AUXILIARY]
    inner_available = cross_available[global_indices[outer_train]][:, aux_columns]
    inner_features = np.hstack([
        inner_parent[:, None],
        inner_aux,
        inner_available,
        scaffold_features(vectorizer, scaffolds[outer_train]),
    ])
    residual_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=30.0),
    )
    residual_model.fit(inner_features, y[outer_train] - inner_parent)
    outer_parent_arms = parent_arms(
        target_dense, sparse_parts, fingerprints, y_global,
        global_indices[outer_train], global_indices[outer_validation], config,
    )
    outer_parent = reference.clip_prediction(y[outer_train], outer_parent_arms @ weights + intercept)
    outer_forbidden_groups = set(groups[outer_validation])
    outer_forbidden_scaffolds = set(forbidden_scaffolds) | set(scaffolds[outer_validation])
    outer_aux = np.column_stack([
        aux_prediction(
            auxiliary, outer_forbidden_groups, outer_forbidden_scaffolds,
            global_indices[outer_validation], aux_info, deterministic,
        )
        for auxiliary in AUXILIARY
    ])
    outer_available = cross_available[global_indices[outer_validation]][:, aux_columns]
    outer_features = np.hstack([
        outer_parent[:, None],
        outer_aux,
        outer_available,
        scaffold_features(vectorizer, scaffolds[outer_validation]),
    ])
    correction = residual_model.predict(outer_features)
    raw_candidate = outer_parent + CORRECTION_STRENGTH * correction
    candidate = reference.clip_prediction(y[outer_train], raw_candidate)
    nearest = nearest_similarity(fingerprints, global_indices[outer_validation], global_indices[outer_train])
    available_count = np.sum(outer_available > 0.5, axis=1)
    route = (available_count >= ROUTE_MIN_AUXILIARY) & (nearest >= ROUTE_MIN_SIMILARITY)
    routed_candidate = outer_parent.copy()
    routed_candidate[route] = candidate[route]
    return {
        "parent": outer_parent,
        "candidate": routed_candidate,
        "nearest": nearest,
        "available_count": available_count,
        "route": route,
        "blend_name": blend_name,
        "blend_weights": [float(value) for value in weights],
        "blend_intercept": float(intercept),
        "inner_parent_r2": float(inner_parent_r2),
        "inner_folds": inner_folds,
    }


def build_panels(y, baseline, candidate, nearest, available_count, scaffolds, measurements, route, groups):
    panels: dict[str, object] = {}
    values: list[float] = []

    def add(name: str, selected: np.ndarray, control: bool = False) -> None:
        delta = panel_delta(y, baseline, candidate, selected)
        rows = int(np.sum(selected))
        if rows < 5:
            status = "inapplicable_zero_support"
        elif delta is None:
            status = "incomplete_constant_support"
        elif control and float(np.max(np.abs(candidate[selected] - baseline[selected]))) > 1.0e-12:
            status = "failed_parent_only_control"
        else:
            status = "evaluable"
        panels[name] = {"rows": rows, "delta_r2": delta, "status": status}
        if delta is not None:
            values.append(delta)

    add("similarity_lt_0.30", nearest < 0.30, control=True)
    add("similarity_0.30_0.50", (nearest >= 0.30) & (nearest < 0.50))
    add("similarity_0.50_0.70", (nearest >= 0.50) & (nearest < 0.70))
    add("similarity_ge_0.70", nearest >= 0.70)
    add("exact_archive_measurements_ge_2", measurements >= 2)
    add("sparse_singleton_measurements_eq_1", measurements == 1)
    add("availability_fewer_than_two_parent_only", available_count < ROUTE_MIN_AUXILIARY, control=True)
    add("availability_at_least_two_route_eligible", route)
    for name in sorted(set(scaffolds)):
        selected = scaffolds == name
        if int(np.sum(selected)) >= 10:
            add(f"scaffold_slice_{name}", selected)
    incomplete = any(isinstance(value, dict) and value["status"] in {"incomplete_constant_support", "failed_parent_only_control"} for value in panels.values())
    return panels, (min(values) if values else None), incomplete


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
    start_time = datetime.now().astimezone()
    started = time.time()
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    deterministic = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    target_dense = masked_target_dense(deterministic, cross_values, cross_available)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.text_matrix(keys, int(reference.DEFAULT_CONFIG["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"]))
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    canonical = frame["canonical"].to_numpy(object)
    groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([scaffold(value) for value in canonical], dtype=object)
    measurements = frame["measurements"].to_numpy(int)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[global_indices] = y
    aux_info: dict[str, dict[str, object]] = {}
    for auxiliary in AUXILIARY:
        aux_frame = pooled[pooled["target_type"] == auxiliary].reset_index(drop=True)
        aux_info[auxiliary] = {
            "global_indices": np.asarray([key_to_index[value] for value in aux_frame["canonical"]], dtype=np.int64),
            "groups": np.asarray([no_stereo(value) for value in aux_frame["canonical"]], dtype=object),
            "scaffolds": np.asarray([scaffold(value) for value in aux_frame["canonical"]], dtype=object),
            "y": aux_frame["target"].to_numpy(float),
        }
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "auxiliary_targets": list(AUXILIARY), "correction_strength": CORRECTION_STRENGTH, "route_min_auxiliary": ROUTE_MIN_AUXILIARY, "route_min_similarity": ROUTE_MIN_SIMILARITY})
    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan, dtype=float)
    candidate = np.full(len(y), np.nan, dtype=float)
    nearest = np.full(len(y), np.nan, dtype=float)
    available_count = np.full(len(y), -1, dtype=int)
    route = np.zeros(len(y), dtype=bool)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        result = nested_split(
            y, groups, scaffolds, training, validation, target_dense, deterministic,
            sparse_parts, fingerprints, y_global, global_indices, cross_available,
            aux_info, config, set(),
        )
        baseline[validation] = result["parent"]
        candidate[validation] = result["candidate"]
        nearest[validation] = result["nearest"]
        available_count[validation] = result["available_count"]
        route[validation] = result["route"]
        parent_score = float(r2_score(y[validation], result["parent"]))
        candidate_score = float(r2_score(y[validation], result["candidate"]))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "baseline_r2": parent_score, "candidate_r2": candidate_score, "delta_r2": candidate_score - parent_score, "routed_rows": int(np.sum(result["route"])), "outer_fold_validation_groups": sorted(set(groups[validation])), "parent_blend": {"name": result["blend_name"], "weights": result["blend_weights"], "intercept": result["blend_intercept"], "inner_r2": result["inner_parent_r2"]}})

    scaffold_holdout: dict[str, object] = {}
    for scaffold_name in sorted(set(scaffolds)):
        validation = np.flatnonzero(scaffolds == scaffold_name)
        if len(validation) < 10:
            continue
        training = np.flatnonzero(scaffolds != scaffold_name)
        result = nested_split(
            y, groups, scaffolds, training, validation, target_dense, deterministic,
            sparse_parts, fingerprints, y_global, global_indices, cross_available,
            aux_info, config, {str(scaffold_name)},
        )
        base_score = float(r2_score(y[validation], result["parent"]))
        cand_score = float(r2_score(y[validation], result["candidate"]))
        scaffold_holdout[str(scaffold_name)] = {"rows": int(len(validation)), "baseline_r2": base_score, "candidate_r2": cand_score, "delta_r2": cand_score - base_score, "routed_rows": int(np.sum(result["route"]))}

    panels, min_panel, panel_incomplete = build_panels(y, baseline, candidate, nearest, available_count, scaffolds, measurements, route, groups)
    baseline_score = float(r2_score(y, baseline))
    candidate_score = float(r2_score(y, candidate))
    delta = candidate_score - baseline_score
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    scaffold_min = min((float(value["delta_r2"]) for value in scaffold_holdout.values()), default=None)
    positive_folds = int(sum(float(row["delta_r2"]) > 0.0 for row in fold_rows))
    route_changes = np.abs(candidate - baseline) > 1.0e-12
    low_similarity = nearest < ROUTE_MIN_SIMILARITY
    low_auxiliary = available_count < ROUTE_MIN_AUXILIARY
    report = {
        "rows": int(len(y)),
        "canonical_groups": int(len(np.unique(groups))),
        "baseline_r2_nested_parent": baseline_score,
        "candidate_r2_scaffold_conditioned_residual": candidate_score,
        "delta_r2": delta,
        "positive_outer_folds": positive_folds,
        "group_r2_bootstrap_lower": float(bootstrap),
        "outer_folds": fold_rows,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": scaffold_min,
        "min_panel_delta": min_panel,
        "panel_incomplete": panel_incomplete,
        "panels": panels,
        "route_definition": {"minimum_available_auxiliary_count": ROUTE_MIN_AUXILIARY, "minimum_nearest_similarity": ROUTE_MIN_SIMILARITY, "routed_rows": int(np.sum(route)), "low_similarity_rows": int(np.sum(low_similarity)), "low_auxiliary_rows": int(np.sum(low_auxiliary)), "low_similarity_changed_rows": int(np.sum(route_changes & low_similarity)), "low_auxiliary_changed_rows": int(np.sum(route_changes & low_auxiliary)), "low_similarity_max_change": float(np.max(np.abs(candidate[low_similarity] - baseline[low_similarity]))) if np.any(low_similarity) else 0.0, "low_auxiliary_max_change": float(np.max(np.abs(candidate[low_auxiliary] - baseline[low_auxiliary]))) if np.any(low_auxiliary) else 0.0},
        "support_counts": {"exact_archive_measurement_rows": int(np.sum(measurements >= 2)), "singleton_rows": int(np.sum(measurements == 1)), "availability_at_least_two_rows": int(np.sum(available_count >= ROUTE_MIN_AUXILIARY))},
    }
    passed = bool(delta >= 0.01 and positive_folds >= 4 and bootstrap > 0.0 and (min_panel is None or min_panel >= -0.003) and (scaffold_min is None or scaffold_min >= -0.003) and not panel_incomplete and report["route_definition"]["low_similarity_changed_rows"] == 0 and report["route_definition"]["low_auxiliary_changed_rows"] == 0 and report["route_definition"]["low_similarity_max_change"] <= 1.0e-12 and report["route_definition"]["low_auxiliary_max_change"] <= 1.0e-12)
    report["pass"] = passed
    report["decision"] = "component_pass" if passed else "rejected_component_gate"
    source_hashes = {"script": sha256_file(Path(__file__).resolve()), "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py")}
    config_hash = reference.canonical_json_hash(config)
    oof = pd.DataFrame({"canonical": canonical, "target_type": TARGET, "target": y, "baseline": baseline, "candidate": candidate, "route": route, "available_auxiliary_count": available_count, "nearest_similarity": nearest, "scaffold": scaffolds, "no_stereo_group": groups, "measurements": measurements, "outer_fold": main_folds})
    oof.to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "config.json", config)
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    metrics = {"schema_version": "ppp.round2.eea-scaffold-conditioned-residual-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C043-20260803-2100-ei-directed-message-passing-guard-v2", "baseline_reference": "C001 official clean incumbent, freshly regenerated as nested masked parent", "official_inputs": inputs, "target": TARGET, "auxiliary_targets": list(AUXILIARY), "metrics": report, "source_hashes": source_hashes, "config_sha256": config_hash, "pass": passed, "decision": report["decision"], "elapsed_seconds": float(time.time() - started)}
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. No candidate or local_eval diagnostic was created by this component run.\n", encoding="utf-8")
    finish_time = datetime.now().astimezone()
    (run_dir / "run.log").write_text(f"experiment_id={run_dir.name}\nstarted_at={start_time.isoformat()}\nfinished_at={finish_time.isoformat()}\ndecision={report['decision']}\nelapsed_seconds={metrics['elapsed_seconds']:.3f}\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([f"{source_hashes['script']}  SOURCE tools/round2_eea_scaffold_conditioned_residual.py", f"{source_hashes['reference_module']}  SOURCE tools/initial_reference_pipeline.py"])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "delta_r2": delta, "passing": passed, "elapsed_seconds": metrics["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
