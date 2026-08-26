#!/usr/bin/env python3
"""Fixed, deployment-like monotonic EPS/Nc counterpart calibration screen.

The only changed input is the officially observed counterpart property.  The
target cell in each outer validation row is excluded from calibration fitting;
the counterpart remains available in entry masking and is removed in the
strict whole-group safety control.  This is a component diagnostic, not a
submission candidate.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
CHANGED = ("eps", "nc")
COUNTERPART = {"eps": "nc", "nc": "eps"}
SEED = 2026
BLEND_WEIGHT = 0.5
BOOTSTRAPS = 2000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def no_stereo(value: object) -> str:
    molecule = Chem.MolFromSmiles(str(value).replace("[*]", "*"))
    if molecule is None:
        return str(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def score(y: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return None
    return float(r2_score(y, prediction))


def delta_score(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray) -> float | None:
    candidate_score = score(y, candidate)
    parent_score = score(y, parent)
    if candidate_score is None or parent_score is None:
        return None
    return candidate_score - parent_score


def bootstrap_lower(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray) -> float | None:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAPS):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        value = delta_score(y[rows], candidate[rows], parent[rows])
        if value is not None:
            values.append(value)
    return float(np.quantile(values, 0.025)) if values else None


def nearest_similarity(fingerprints: list[Any], query: np.ndarray, train: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[int(index)] for index in train]
    if not train_fps:
        return np.full(len(query), np.nan, dtype=np.float64)
    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps)) for index in query],
        dtype=np.float64,
    )


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    usable = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(usable)) < 8 or np.unique(x[usable]).size < 2:
        return None
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(x[usable], y[usable])
    return model


def apply_calibration(
    model: IsotonicRegression | None,
    counterpart: np.ndarray,
    parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = parent.copy()
    mapped = np.full(len(parent), np.nan, dtype=np.float64)
    route = np.isfinite(counterpart) & (model is not None)
    if model is not None and np.any(route):
        mapped[route] = np.asarray(model.predict(counterpart[route]), dtype=np.float64)
        candidate[route] = (1.0 - BLEND_WEIGHT) * parent[route] + BLEND_WEIGHT * mapped[route]
    return candidate, route, mapped


def panel_report(
    y: np.ndarray,
    candidate: np.ndarray,
    parent: np.ndarray,
    groups: np.ndarray,
    similarity: np.ndarray,
    scaffolds: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    panel_deltas: list[float] = []

    def add(name: str, selected: np.ndarray, unchanged: bool = False) -> None:
        rows = int(np.sum(selected))
        group_count = int(np.unique(groups[selected]).size)
        value = delta_score(y[selected], candidate[selected], parent[selected]) if rows >= 5 else None
        if value is not None:
            panel_deltas.append(value)
        if unchanged and rows and float(np.max(np.abs(candidate[selected] - parent[selected]))) > 1.0e-12:
            raise RuntimeError(f"parent-only panel changed: {name}")
        report[name] = {
            "rows": rows,
            "groups": group_count,
            "eligible": bool(value is not None),
            "delta_r2": value,
            "parent_only": unchanged,
        }

    add("counterpart_supported", support)
    add("counterpart_missing_parent_only", ~support, unchanged=True)
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        add(name, selected)
    scaffold_values: list[float] = []
    for scaffold in sorted(set(scaffolds)):
        selected = scaffolds == scaffold
        value = delta_score(y[selected], candidate[selected], parent[selected]) if int(np.sum(selected)) >= 10 else None
        if value is not None and int(np.unique(groups[selected]).size) >= 3:
            scaffold_values.append(value)
    report["scaffold_groups_ge_3"] = {
        "evaluated": len(scaffold_values),
        "minimum_delta_r2": min(scaffold_values) if scaffold_values else None,
    }
    panel_deltas.extend(scaffold_values)
    report["minimum_panel_delta"] = min(panel_deltas) if panel_deltas else 0.0
    return report


def masked_cross_arrays(pooled: pd.DataFrame, keys: list[str], excluded_groups: set[str]) -> tuple[np.ndarray, np.ndarray]:
    if not excluded_groups:
        return reference.cross_property_arrays(pooled, keys)
    groups = pooled["canonical"].map(no_stereo)
    return reference.cross_property_arrays(pooled.loc[~groups.isin(excluded_groups)], keys)


def target_screen(
    target: str,
    pooled: pd.DataFrame,
    keys: list[str],
    key_to_index: dict[str, int],
    base_dense: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    exact_parent: np.ndarray,
    parent_weights: np.ndarray,
    parent_intercept: float,
    strict: bool,
) -> dict[str, Any]:
    frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
    y = frame["target"].to_numpy(dtype=float)
    canonical = frame["canonical"].astype(str).to_numpy(dtype=object)
    groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([
        MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=False) or "ACYCLIC"
        for value in canonical
    ], dtype=object)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    counterpart_target = COUNTERPART[target]
    counterpart_map = (
        pooled[pooled["target_type"] == counterpart_target]
        .drop_duplicates("canonical")
        .set_index("canonical")["target"]
        .to_dict()
    )
    counterpart = np.asarray([counterpart_map.get(value, np.nan) for value in canonical], dtype=np.float64)
    if strict:
        splitter = GroupKFold(n_splits=5)
        iterator = splitter.split(np.arange(len(y)), groups=groups)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
        iterator = splitter.split(np.arange(len(y)))
    parent_oof = np.full(len(y), np.nan, dtype=np.float64)
    candidate_oof = np.full(len(y), np.nan, dtype=np.float64)
    mapped_oof = np.full(len(y), np.nan, dtype=np.float64)
    support_oof = np.zeros(len(y), dtype=bool)
    similarity_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (outer_train, validation) in enumerate(iterator):
        outer_train = np.asarray(outer_train, dtype=np.int64)
        validation = np.asarray(validation, dtype=np.int64)
        excluded = set(groups[validation].tolist()) if strict else set()
        values, available = masked_cross_arrays(pooled, keys, excluded)
        dense = reference.target_dense_features(base_dense, values, available, target)
        target_y_global = np.full(len(keys), np.nan, dtype=np.float64)
        target_y_global[global_indices] = y
        if strict:
            outer_arms = reference.predict_base_models(
                dense,
                sparse_parts,
                fingerprints,
                target_y_global,
                global_indices[outer_train],
                global_indices[validation],
                reference.DEFAULT_CONFIG,
                target,
            )
            outer_parent = outer_arms @ parent_weights + parent_intercept
        else:
            outer_parent = exact_parent[validation]
        train_x = counterpart[outer_train].copy()
        if strict:
            train_x[np.isin(groups[outer_train], list(excluded))] = np.nan
        model = fit_isotonic(train_x, y[outer_train])
        validation_x = counterpart[validation].copy()
        if strict:
            validation_x[:] = np.nan
        candidate, route, mapped = apply_calibration(model, validation_x, outer_parent)
        parent_oof[validation] = outer_parent
        candidate_oof[validation] = candidate
        mapped_oof[validation] = mapped
        support_oof[validation] = route
        similarity_oof[validation] = nearest_similarity(fingerprints, global_indices[validation], global_indices[outer_train])
        fold_delta = delta_score(y[validation], candidate, outer_parent)
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "outer_group_count": int(np.unique(groups[validation]).size),
            "parent_r2": score(y[validation], outer_parent),
            "candidate_r2": score(y[validation], candidate),
            "delta_r2": fold_delta,
            "routed_rows": int(np.sum(route)),
            "calibration_training_pairs": int(np.sum(np.isfinite(train_x))),
        })
    if not np.isfinite(parent_oof).all() or not np.isfinite(candidate_oof).all():
        raise RuntimeError(f"incomplete OOF for {target}")
    report = {
        "rows": int(len(y)),
        "group_count": int(np.unique(groups).size),
        "parent_r2": score(y, parent_oof),
        "candidate_r2": score(y, candidate_oof),
        "delta_r2": delta_score(y, candidate_oof, parent_oof),
        "positive_folds": int(sum((row["delta_r2"] or 0.0) > 0.0 for row in fold_rows)),
        "group_bootstrap_lower": bootstrap_lower(y, candidate_oof, parent_oof, groups),
        "routed_rows": int(np.sum(support_oof)),
        "routed_groups": int(np.unique(groups[support_oof]).size),
        "support_rows": int(np.sum(np.isfinite(counterpart))),
        "folds": fold_rows,
        "panels": panel_report(y, candidate_oof, parent_oof, groups, similarity_oof, scaffolds, support_oof),
        "oof": {
            "canonical": canonical,
            "group": groups,
            "target": y,
            "parent": parent_oof,
            "candidate": candidate_oof,
            "mapped": mapped_oof,
            "counterpart": counterpart,
            "supported": support_oof,
        },
    }
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
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
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
        exact_parent = parent_rows["prediction"].to_numpy(dtype=float)
        weights = np.asarray([
            parent_report["target_reports"][target]["blend_weights"][name]
            for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")
        ], dtype=float)
        intercept = float(parent_report["target_reports"][target]["blend_intercept"])
        entry = target_screen(
            target, pooled, keys, key_to_index, base_dense, cross_values, cross_available,
            sparse_parts, fingerprints, exact_parent, weights, intercept, strict=False,
        )
        strict = target_screen(
            target, pooled, keys, key_to_index, base_dense, cross_values, cross_available,
            sparse_parts, fingerprints, exact_parent, weights, intercept, strict=True,
        )
        test_frame = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_parent = detail[detail["target_type"] == target].sort_values("id").reset_index(drop=True)
        counterpart_target = COUNTERPART[target]
        counterpart_map = (
            pooled[pooled["target_type"] == counterpart_target]
            .drop_duplicates("canonical")
            .set_index("canonical")["target"]
            .to_dict()
        )
        test_counterpart = np.asarray([counterpart_map.get(value, np.nan) for value in test_frame["canonical"]], dtype=float)
        model = fit_isotonic(
            np.asarray([counterpart_map.get(value, np.nan) for value in frame["canonical"]], dtype=float),
            frame["target"].to_numpy(dtype=float),
        )
        test_parent_values = test_parent["model_prediction"].to_numpy(dtype=float)
        test_candidate, test_route, test_mapped = apply_calibration(model, test_counterpart, test_parent_values)
        if not np.isfinite(test_candidate).all():
            raise RuntimeError(f"non-finite test candidate for {target}")
        for row, parent_value, counterpart_value, mapped_value, candidate_value, route in zip(
            test_frame.itertuples(index=False), test_parent_values, test_counterpart, test_mapped, test_candidate, test_route, strict=True
        ):
            component_rows.append({
                "id": int(row.id),
                "target_type": target,
                "parent_prediction": float(parent_value),
                "counterpart": None if not np.isfinite(counterpart_value) else float(counterpart_value),
                "mapped_prediction": None if not np.isfinite(mapped_value) else float(mapped_value),
                "candidate_prediction": float(candidate_value),
                "routed": bool(route),
            })
        oof = entry["oof"]
        for row, group, target_value, parent_value, candidate_value, mapped_value, counterpart_value, route in zip(
            frame.itertuples(index=False), oof["group"], oof["target"], oof["parent"], oof["candidate"], oof["mapped"], oof["counterpart"], oof["supported"], strict=True
        ):
            oof_rows.append({
                "canonical": str(row.canonical),
                "group": str(group),
                "target_type": target,
                "target": float(target_value),
                "parent": float(parent_value),
                "candidate": float(candidate_value),
                "mapped": None if not np.isfinite(mapped_value) else float(mapped_value),
                "counterpart": None if not np.isfinite(counterpart_value) else float(counterpart_value),
                "supported": bool(route),
            })
        reports[target] = {
            "entry_masking": {key: value for key, value in entry.items() if key != "oof"},
            "strict_group_masking": {key: value for key, value in strict.items() if key != "oof"},
            "counterpart_target": counterpart_target,
            "test_rows": int(len(test_frame)),
            "test_supported_rows": int(np.sum(test_route)),
            "test_model_training_pairs": int(np.sum(np.isfinite([counterpart_map.get(value, np.nan) for value in frame["canonical"]]))),
        }
    component = pd.DataFrame(component_rows).sort_values("id").reset_index(drop=True)
    expected = int(sum(np.sum(test["target_type"] == target) for target in CHANGED))
    if len(component) != expected or component["id"].duplicated().any() or not np.isfinite(component["candidate_prediction"].to_numpy()).all():
        raise RuntimeError("component output contract failed")
    for target in CHANGED:
        expected_ids = test.loc[test["target_type"] == target, "id"].to_numpy()
        actual_ids = component.loc[component["target_type"] == target, "id"].to_numpy()
        if not np.array_equal(actual_ids, expected_ids):
            raise RuntimeError(f"test order failed for {target}")
    component.to_csv(run_dir / "eps_nc_component_predictions.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {
        target: {
            "gain_pass": bool(reports[target]["entry_masking"]["delta_r2"] >= 0.010),
            "fold_pass": bool(reports[target]["entry_masking"]["positive_folds"] >= 4),
            "bootstrap_pass": bool(reports[target]["entry_masking"]["group_bootstrap_lower"] is not None and reports[target]["entry_masking"]["group_bootstrap_lower"] > 0.0),
            "panel_pass": bool(reports[target]["entry_masking"]["panels"]["minimum_panel_delta"] >= 0.0),
            "strict_no_regression": bool(reports[target]["strict_group_masking"]["delta_r2"] >= -0.003),
        }
        for target in CHANGED
    }
    passed = bool(all(all(values.values()) for values in gates.values()))
    entry_parent = float(np.mean([reports[target]["entry_masking"]["parent_r2"] for target in CHANGED]))
    entry_candidate = float(np.mean([reports[target]["entry_masking"]["candidate_r2"] for target in CHANGED]))
    strict_parent = float(np.mean([reports[target]["strict_group_masking"]["parent_r2"] for target in CHANGED]))
    strict_candidate = float(np.mean([reports[target]["strict_group_masking"]["candidate_r2"] for target in CHANGED]))
    report = {
        "schema_version": "ppp.round2.c057.monotonic-counterpart-calibration.v1",
        "experiment_id": run_dir.name,
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; EPS/Nc parent regenerated from official inputs",
        "created_at": datetime.now().astimezone().isoformat(),
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "changed_targets": list(CHANGED),
        "targets": reports,
        "entry_masking_mean_parent_r2": entry_parent,
        "entry_masking_mean_candidate_r2": entry_candidate,
        "entry_masking_mean_gain": entry_candidate - entry_parent,
        "strict_group_mean_parent_r2": strict_parent,
        "strict_group_mean_candidate_r2": strict_candidate,
        "strict_group_mean_gain": strict_candidate - strict_parent,
        "gates": gates,
        "decision": "pass_component_gate" if passed else "rejected_component_gate",
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(oof_rows))},
        "source_sha256": sha256_file(root / "tools" / "round2_c057_monotonic_counterpart_calibration.py"),
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {
        "seed": SEED,
        "changed_targets": list(CHANGED),
        "counterpart": COUNTERPART,
        "model": "increasing IsotonicRegression on outer-training paired cells",
        "blend_weight": BLEND_WEIGHT,
        "outer_folds": "KFold(5, shuffle=true, random_state=2026) entry masking plus GroupKFold(5) strict group masking",
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
        "# C057 decision\n\n"
        f"Entry-mask mean parent: {entry_parent:.12f}\n"
        f"Entry-mask mean candidate: {entry_candidate:.12f}\n"
        f"Entry-mask mean gain: {entry_candidate - entry_parent:+.12f}\n"
        f"Strict-group mean gain: {strict_candidate - strict_parent:+.12f}\n\n"
        f"Decision: {'PASS COMPONENT GATE' if passed else 'REJECT COMPONENT GATE'}\n\n"
        "The calibration used only official counterpart labels, withheld each validation target cell from fitting, and fell back to the exact parent when counterpart support was absent. No external_label file, local_eval value, or prior prediction artifact was read.\n",
        encoding="utf-8",
    )
    manifest = [run_dir / name for name in ("protocol.json", "config.json", "metrics.json", "eps_nc_component_predictions.csv", "oof_predictions.csv", "environment.txt", "command.txt", "decision.md")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment_id": run_dir.name,
        "decision": report["decision"],
        "entry_masking_mean_parent_r2": entry_parent,
        "entry_masking_mean_candidate_r2": entry_candidate,
        "entry_masking_mean_gain": entry_candidate - entry_parent,
        "strict_group_mean_gain": strict_candidate - strict_parent,
        "targets": {target: reports[target]["entry_masking"]["delta_r2"] for target in CHANGED},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
