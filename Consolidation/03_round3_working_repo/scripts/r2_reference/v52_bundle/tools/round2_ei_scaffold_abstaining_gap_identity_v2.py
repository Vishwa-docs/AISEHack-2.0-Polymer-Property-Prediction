#!/usr/bin/env python3
"""Strictly nested official-only scaffold-abstaining Ei gap-identity diagnostic."""

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
from rdkit import DataStructs, RDLogger
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = ("ei",)
GAP_ALPHA = 10.0
SIMILARITY_BARRIER = 0.70
BLOCKED_SCAFFOLDS = {"c1ccsc1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray, n_splits: int) -> np.ndarray:
    return plumbing.folds_for(groups, n_splits)


def nearest_similarity(fingerprints: list[object], validation: np.ndarray, training: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[index] for index in training]
    if not train_fps:
        return np.full(len(validation), np.nan, dtype=np.float64)
    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps)) for index in validation],
        dtype=np.float64,
    )


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def make_gap_table(pooled: pd.DataFrame) -> pd.DataFrame:
    pivot = pooled.pivot(index="canonical", columns="target_type", values="target").reset_index()
    for target in ("ei", "eea", "egc"):
        if target not in pivot:
            pivot[target] = np.nan
    pivot["group"] = [plumbing.no_stereo(value) for value in pivot["canonical"]]
    pivot["scaffold"] = [plumbing.scaffold(value) for value in pivot["canonical"]]
    pivot["gap"] = pivot["ei"] - pivot["eea"]
    return pivot


def fit_gap_model(gap_table: pd.DataFrame, forbidden_groups: set[str], forbidden_scaffolds: set[str] | None = None):
    forbidden_scaffolds = forbidden_scaffolds or set()
    usable = (
        gap_table["ei"].notna()
        & gap_table["eea"].notna()
        & gap_table["egc"].notna()
        & ~gap_table["group"].isin(forbidden_groups)
        & ~gap_table["scaffold"].isin(forbidden_scaffolds)
    )
    if int(np.sum(usable)) < 8:
        return None, int(np.sum(usable))
    model = make_pipeline(StandardScaler(), Ridge(alpha=GAP_ALPHA))
    model.fit(gap_table.loc[usable, ["egc"]].to_numpy(float), gap_table.loc[usable, "gap"].to_numpy(float))
    return model, int(np.sum(usable))


def nested_parent(
    target: str,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    target_dense: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    global_indices: np.ndarray,
    y_global: np.ndarray,
    config: dict[str, object],
    canonical: np.ndarray,
    scaffolds: np.ndarray,
    assignment_scope: str,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    inner_folds = folds_for(groups[outer_train], 4)
    inner_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_arms[inner_folds == fold] = reference.predict_base_models(
            target_dense,
            sparse_parts,
            fingerprints,
            y_global,
            global_indices[local_train],
            global_indices[local_validation],
            config,
            target,
        )
    weights, intercept, blend_name, inner_r2 = reference.blend_from_oof(y[outer_train], inner_arms)
    outer_arms = reference.predict_base_models(
        target_dense,
        sparse_parts,
        fingerprints,
        y_global,
        global_indices[outer_train],
        global_indices[outer_validation],
        config,
        target,
    )
    parent = reference.clip_prediction(y[outer_train], outer_arms @ weights + intercept)
    inner_assignments = [
        {
            "record_type": "inner_parent_oof_assignment",
            "assignment_scope": assignment_scope,
            "outer_row_index": int(index),
            "inner_fold": int(inner_folds[position]),
            "canonical": str(canonical[index]),
            "group": str(groups[index]),
            "scaffold": str(scaffolds[index]),
        }
        for position, index in enumerate(outer_train)
    ]
    return parent, {
        "blend_name": blend_name,
        "blend_weights": [float(value) for value in weights],
        "blend_intercept": float(intercept),
        "inner_parent_r2": float(inner_r2),
    }, inner_assignments


def route_predictions(
    target: str,
    canonical: np.ndarray,
    parent: np.ndarray,
    y_train: np.ndarray,
    gap_model,
    maps: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    partner_target = "eea" if target == "ei" else "ei"
    partner = np.asarray([maps[partner_target].get(value, np.nan) for value in canonical], dtype=np.float64)
    egc = np.asarray([maps["egc"].get(value, np.nan) for value in canonical], dtype=np.float64)
    available = np.isfinite(partner) & np.isfinite(egc) & (gap_model is not None)
    candidate = np.asarray(parent, dtype=np.float64).copy()
    predicted_gap = np.full(len(canonical), np.nan, dtype=np.float64)
    if gap_model is not None and np.any(available):
        predicted_gap[available] = gap_model.predict(egc[available, None])
        raw = partner[available] + predicted_gap[available] if target == "ei" else partner[available] - predicted_gap[available]
        candidate[available] = reference.clip_prediction(y_train, raw)
    return candidate, available, partner, predicted_gap


def target_panels(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    nearest: np.ndarray,
    scaffolds: np.ndarray,
    measurements: np.ndarray,
    routed: np.ndarray,
    raw_partner_egc_support: np.ndarray,
    supported_but_abstained: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, object], float | None, bool]:
    panels: dict[str, object] = {}
    panel_values: list[float] = []

    def add(name: str, selected: np.ndarray, unchanged_control: bool = False) -> None:
        delta = panel_delta(y, baseline, candidate, selected)
        rows = int(np.sum(selected))
        if rows < 5:
            status = "inapplicable_zero_support"
        elif delta is None:
            status = "incomplete_constant_support"
        elif unchanged_control and float(np.max(np.abs(candidate[selected] - baseline[selected]))) > 1.0e-12:
            status = "failed_parent_only_control"
        else:
            status = "evaluable"
        panels[name] = {"rows": rows, "delta_r2": delta, "status": status}
        if delta is not None:
            panel_values.append(delta)

    add("similarity_lt_0.30", nearest < 0.30)
    add("similarity_0.30_0.50", (nearest >= 0.30) & (nearest < 0.50))
    add("similarity_0.50_0.70", (nearest >= 0.50) & (nearest < 0.70))
    add("similarity_ge_0.70", nearest >= 0.70)
    add("exact_archive_measurements_ge_2", measurements >= 2)
    add("sparse_singleton_measurements_eq_1", measurements == 1)
    add("routed_support", routed)
    add("raw_partner_and_egc_available", raw_partner_egc_support)
    add("raw_partner_or_egc_missing_parent_only", ~raw_partner_egc_support, unchanged_control=True)
    add("supported_but_abstained_parent_only", supported_but_abstained, unchanged_control=True)
    for scaffold in sorted(set(scaffolds)):
        selected = scaffolds == scaffold
        if int(np.sum(selected)) >= 10:
            add(f"scaffold_slice_{scaffold}", selected)
    bootstrap = plumbing.bootstrap_r2_lower(y, baseline, candidate, groups)
    incomplete = any(
        isinstance(value, dict) and value["status"] == "incomplete_constant_support"
        for value in panels.values()
    )
    return panels, (min(panel_values) if panel_values else None), incomplete


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
    start_time = datetime.now().astimezone()
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.text_matrix(keys, int(reference.DEFAULT_CONFIG["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"]))
    gap_table = make_gap_table(pooled)
    maps = {
        target: dict(zip(frame["canonical"], frame["target"].astype(float), strict=True))
        for target in ("ei", "eea", "egc")
        for frame in [pooled[pooled["target_type"] == target].reset_index(drop=True)]
    }
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "gap_alpha": GAP_ALPHA, "similarity_barrier": SIMILARITY_BARRIER, "blocked_scaffolds": sorted(BLOCKED_SCAFFOLDS), "parent_cross_property_covariates": "clean_baseline_except_own_target"})
    target_reports: dict[str, object] = {}
    all_oof: list[pd.DataFrame] = []
    all_fold_rows: list[dict[str, object]] = []
    test_route_rows: list[dict[str, object]] = []
    all_inner_assignments: list[dict[str, object]] = []

    for target in TARGETS:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        canonical = frame["canonical"].to_numpy(object)
        groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
        measurements = frame["measurements"].to_numpy(int)
        global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        y_global = np.full(len(keys), np.nan, dtype=np.float64)
        y_global[global_indices] = y
        target_dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
        main_folds = folds_for(groups, 5)
        baseline = np.full(len(y), np.nan, dtype=np.float64)
        candidate = np.full(len(y), np.nan, dtype=np.float64)
        routed = np.zeros(len(y), dtype=bool)
        raw_partner_egc_support = np.zeros(len(y), dtype=bool)
        supported_but_abstained = np.zeros(len(y), dtype=bool)
        partner = np.full(len(y), np.nan, dtype=np.float64)
        predicted_gap = np.full(len(y), np.nan, dtype=np.float64)
        nearest = np.full(len(y), np.nan, dtype=np.float64)
        fold_rows: list[dict[str, object]] = []
        gap_rows: list[dict[str, object]] = []
        for fold in range(5):
            validation = np.flatnonzero(main_folds == fold)
            training = np.flatnonzero(main_folds != fold)
            parent, parent_meta, inner_assignments = nested_parent(
                target, y, groups, training, validation, target_dense, sparse_parts,
                fingerprints, global_indices, y_global, config, canonical, scaffolds,
                f"main_outer_fold_{fold}",
            )
            all_inner_assignments.extend(inner_assignments)
            gap_model, gap_count = fit_gap_model(gap_table, set(groups[validation]))
            nearest_validation = nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
            routed_candidate, route, route_partner, route_gap = route_predictions(
                target, canonical[validation], parent, y[training], gap_model, maps,
            )
            allowed = (nearest_validation < SIMILARITY_BARRIER) & ~np.isin(scaffolds[validation], list(BLOCKED_SCAFFOLDS))
            routed_candidate[~allowed] = parent[~allowed]
            route[~allowed] = False
            egc_validation = np.asarray([maps["egc"].get(value, np.nan) for value in canonical[validation]], dtype=np.float64)
            raw_support = np.isfinite(route_partner) & np.isfinite(egc_validation)
            baseline[validation] = parent
            candidate[validation] = routed_candidate
            routed[validation] = route
            raw_partner_egc_support[validation] = raw_support
            supported_but_abstained[validation] = raw_support & ~route
            partner[validation] = route_partner
            predicted_gap[validation] = route_gap
            nearest[validation] = nearest_validation
            parent_score = float(r2_score(y[validation], parent))
            candidate_score = float(r2_score(y[validation], routed_candidate))
            fold_rows.append({
                "record_type": "outer_fold_summary",
                "target": target,
                "fold": fold,
                "rows": int(len(validation)),
                "baseline_r2": parent_score,
                "candidate_r2": candidate_score,
                "delta_r2": candidate_score - parent_score,
                "routed_rows": int(np.sum(route)),
                "gap_training_rows": gap_count,
                "parent_blend": parent_meta,
            })
            all_fold_rows.extend({"record_type": "outer_oof_assignment", "target": target, "row_index": int(index), "fold": fold, "canonical": str(canonical[index]), "group": str(groups[index]), "scaffold": str(scaffolds[index])} for index in validation)
            gap_rows.append({"target": target, "fold": fold, "gap_training_rows": gap_count, "routed_rows": int(np.sum(route))})

        scaffold_holdout: dict[str, object] = {}
        for scaffold_name in sorted(set(scaffolds)):
            validation = np.flatnonzero(scaffolds == scaffold_name)
            if len(validation) < 10:
                continue
            training = np.flatnonzero(scaffolds != scaffold_name)
            parent_holdout, _, inner_assignments = nested_parent(
                target, y, groups, training, validation, target_dense, sparse_parts,
                fingerprints, global_indices, y_global, config, canonical, scaffolds,
                f"scaffold_holdout_{scaffold_name}",
            )
            all_inner_assignments.extend(inner_assignments)
            gap_model, gap_count = fit_gap_model(gap_table, set(groups[validation]), {str(scaffold_name)})
            holdout_candidate, route, _, _ = route_predictions(
                target, canonical[validation], parent_holdout, y[training], gap_model, maps,
            )
            nearest_holdout = nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
            allowed_holdout = (nearest_holdout < SIMILARITY_BARRIER) & ~np.isin(scaffolds[validation], list(BLOCKED_SCAFFOLDS))
            holdout_candidate[~allowed_holdout] = parent_holdout[~allowed_holdout]
            route[~allowed_holdout] = False
            base_score = float(r2_score(y[validation], parent_holdout))
            cand_score = float(r2_score(y[validation], holdout_candidate))
            scaffold_holdout[str(scaffold_name)] = {
                "rows": int(len(validation)),
                "baseline_r2": base_score,
                "candidate_r2": cand_score,
                "delta_r2": cand_score - base_score,
                "routed_rows": int(np.sum(route)),
                "gap_training_rows": gap_count,
            }

        missing = ~raw_partner_egc_support
        panels, min_panel, panel_incomplete = target_panels(
            y, baseline, candidate, nearest, scaffolds, measurements, routed, raw_partner_egc_support, supported_but_abstained, groups,
        )
        baseline_score = float(r2_score(y, baseline))
        candidate_score = float(r2_score(y, candidate))
        delta = candidate_score - baseline_score
        bootstrap = float(plumbing.bootstrap_r2_lower(y, baseline, candidate, groups))
        scaffold_min = min((float(value["delta_r2"]) for value in scaffold_holdout.values()), default=None)
        positive_folds = int(sum(float(row["delta_r2"]) > 0.0 for row in fold_rows))
        target_reports[target] = {
            "rows": int(len(y)),
            "canonical_groups": int(len(np.unique(groups))),
            "baseline_r2_nested_parent": baseline_score,
            "candidate_r2_gap_identity": candidate_score,
            "delta_r2": delta,
            "positive_outer_folds": positive_folds,
            "group_r2_bootstrap_lower": bootstrap,
            "outer_folds": fold_rows,
            "scaffold_holdout": scaffold_holdout,
            "scaffold_holdout_min_delta": scaffold_min,
            "min_panel_delta": min_panel,
            "panel_incomplete": panel_incomplete,
            "panels": panels,
            "route_definition": {
                "partner_target": "eea" if target == "ei" else "ei",
                "gap_feature": "egc",
                "similarity_barrier": SIMILARITY_BARRIER,
                "blocked_scaffolds": sorted(BLOCKED_SCAFFOLDS),
                "routed_rows": int(np.sum(routed)),
                "raw_partner_egc_support_rows": int(np.sum(raw_partner_egc_support)),
                "missing_partner_or_egc_rows": int(np.sum(missing)),
                "supported_but_abstained_rows": int(np.sum(supported_but_abstained)),
                "route_changed_rows": int(np.sum(np.abs(candidate - baseline) > 1.0e-12)),
                "missing_rows_max_change": float(np.max(np.abs(candidate[missing] - baseline[missing]))) if np.any(missing) else 0.0,
                "abstained_rows_max_change": float(np.max(np.abs(candidate[supported_but_abstained] - baseline[supported_but_abstained]))) if np.any(supported_but_abstained) else 0.0,
            },
            "support_counts": {
                "exact_archive_measurement_rows": int(np.sum(measurements >= 2)),
                "singleton_rows": int(np.sum(measurements == 1)),
                "paired_gap_groups": int(np.sum(gap_table["ei"].notna() & gap_table["eea"].notna() & gap_table["egc"].notna())),
            },
        }
        paired_target = "eea"
        paired_target_rows = int(sum(len(frame) for frame in all_oof if "target_type" in frame and (frame["target_type"] == paired_target).any()))
        paired_gate_pass = paired_target_rows == 0
        target_reports[target]["paired_target_gate"] = {
            "target": paired_target,
            "oof_rows": paired_target_rows,
            "r2_loss": 0.0 if paired_gate_pass else None,
            "maximum_allowed_loss": 0.003,
            "pass": paired_gate_pass,
            "basis": "Structural no-op derived from the emitted OOF target scope: this Ei-only component emits no Eea predictions and does not modify Eea rows.",
        }
        target_pass = bool(
            delta >= 0.01
            and positive_folds >= 4
            and bootstrap > 0.0
            and (min_panel is None or min_panel >= 0.0)
            and (scaffold_min is None or scaffold_min >= 0.0)
            and not panel_incomplete
            and float(target_reports[target]["route_definition"]["missing_rows_max_change"]) <= 1.0e-12
            and float(target_reports[target]["route_definition"]["abstained_rows_max_change"]) <= 1.0e-12
            and bool(target_reports[target]["paired_target_gate"]["pass"])
        )
        target_reports[target]["pass"] = target_pass
        target_reports[target]["decision"] = "component_pass" if target_pass else "rejected_component_gate"
        for row in fold_rows:
            all_fold_rows.append(row)
        all_oof.append(pd.DataFrame({
            "canonical": canonical,
            "target_type": target,
            "target": y,
            "baseline": baseline,
            "candidate": candidate,
            "route": routed,
            "raw_partner_egc_support": raw_partner_egc_support,
            "supported_but_abstained": supported_but_abstained,
            "partner": partner,
            "predicted_gap": predicted_gap,
            "nearest_similarity": nearest,
            "scaffold": scaffolds,
            "no_stereo_group": groups,
            "measurements": measurements,
            "outer_fold": main_folds,
        }))

        test_frame = test[test["target_type"] == target].reset_index(drop=True)
        test_canonical = test_frame["canonical"].to_numpy(object)
        full_gap_model, full_gap_count = fit_gap_model(gap_table, set())
        test_global_indices = np.asarray([key_to_index[value] for value in test_canonical], dtype=np.int64)
        test_nearest = nearest_similarity(fingerprints, test_global_indices, global_indices)
        test_scaffolds = np.asarray([plumbing.scaffold(value) for value in test_canonical], dtype=object)
        test_partner = np.asarray([maps["eea" if target == "ei" else "ei"].get(value, np.nan) for value in test_canonical], dtype=float)
        test_egc = np.asarray([maps["egc"].get(value, np.nan) for value in test_canonical], dtype=float)
        test_route = np.isfinite(test_partner) & np.isfinite(test_egc) & (full_gap_model is not None) & (test_nearest < SIMILARITY_BARRIER) & ~np.isin(test_scaffolds, list(BLOCKED_SCAFFOLDS))
        test_pred_gap = np.full(len(test_frame), np.nan, dtype=float)
        test_raw_support = np.isfinite(test_partner) & np.isfinite(test_egc) & (full_gap_model is not None)
        if full_gap_model is not None and np.any(test_raw_support):
            test_pred_gap[test_raw_support] = full_gap_model.predict(test_egc[test_raw_support, None])
        test_reconstruction = np.full(len(test_frame), np.nan, dtype=float)
        if np.any(test_route):
            raw_test = test_partner[test_route] + test_pred_gap[test_route] if target == "ei" else test_partner[test_route] - test_pred_gap[test_route]
            test_reconstruction[test_route] = reference.clip_prediction(y, raw_test)
        test_route_rows.extend(pd.DataFrame({
            "id": test_frame["id"].astype(int),
            "target_type": target,
            "partner_available": np.isfinite(test_partner),
            "egc_available": np.isfinite(test_egc),
            "route_available": test_route,
            "raw_partner_egc_support": test_raw_support,
            "supported_but_abstained": test_raw_support & ~test_route,
            "nearest_similarity": test_nearest,
            "scaffold": test_scaffolds,
            "partner_value": test_partner,
            "predicted_gap": test_pred_gap,
            "reconstructed_value": test_reconstruction,
            "gap_training_rows": full_gap_count,
        }).to_dict("records"))

    pass_targets = [target for target in TARGETS if bool(target_reports[target]["pass"])]
    source_hashes = {
        "script": sha256_file(Path(__file__).resolve()),
        "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py"),
        "metric_plumbing": sha256_file(root / "tools" / "round2_eea_cross_target_oof_residual_stack.py"),
    }
    metrics = {
        "schema_version": "ppp.round2.ei-scaffold-abstaining-gap-identity-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": protocol["parent"],
        "lineage_parent": protocol.get("lineage_parent", protocol["parent"]),
        "correction_of": protocol.get("correction_of"),
        "official_inputs": inputs,
        "route": {"similarity_barrier": SIMILARITY_BARRIER, "blocked_scaffolds": sorted(BLOCKED_SCAFFOLDS), "fallback": "nested C001 parent"},
        "targets": target_reports,
        "passing_targets": pass_targets,
        "pass": bool(pass_targets),
        "decision": "component_pass" if pass_targets else "rejected_component_gate",
        "source_hashes": source_hashes,
        "config_sha256": reference.canonical_json_hash(config),
        "elapsed_seconds": float(time.time() - started),
    }
    oof = pd.concat(all_oof, ignore_index=True)
    oof.to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(run_dir / "folds.csv", index=False)
    pd.DataFrame([row for row in all_fold_rows if "row_index" in row]).to_csv(run_dir / "fold_assignments.csv", index=False)
    pd.DataFrame(all_inner_assignments).to_csv(run_dir / "inner_fold_assignments.csv", index=False)
    pd.DataFrame(test_route_rows).to_csv(run_dir / "test_route_diagnostic.csv", index=False)
    write_json(run_dir / "config.json", config)
    (run_dir / "environment.txt").write_text(
        "\n".join([
            f"python={platform.python_version()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"sklearn={__import__('sklearn').__version__}",
            f"scipy={__import__('scipy').__version__}",
            f"rdkit={reference.Chem.rdBase.rdkitVersion}",
            f"platform={platform.platform()}",
        ]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: **{metrics['decision']}**. Passing targets: `{', '.join(pass_targets) or 'none'}`. No candidate or local_eval diagnostic was created by this component run.\n",
        encoding="utf-8",
    )
    finish_time = datetime.now().astimezone()
    (run_dir / "run.log").write_text(
        f"experiment_id={run_dir.name}\n"
        f"started_at={start_time.isoformat()}\n"
        f"finished_at={finish_time.isoformat()}\n"
        f"decision={metrics['decision']}\n"
        f"passing_targets={','.join(pass_targets)}\n"
        f"elapsed_seconds={metrics['elapsed_seconds']:.3f}\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([
        f"{source_hashes['script']}  SOURCE tools/round2_ei_scaffold_abstaining_gap_identity_v2.py",
        f"{source_hashes['reference_module']}  SOURCE tools/initial_reference_pipeline.py",
        f"{source_hashes['metric_plumbing']}  SOURCE tools/round2_eea_cross_target_oof_residual_stack.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": metrics["decision"], "passing_targets": pass_targets, "elapsed_seconds": metrics["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
