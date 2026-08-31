#!/usr/bin/env python3
"""C1365/C1366 branch-local OOF-stability residual overlay.

This runner regenerates previously positive residual mechanisms from official
Round 2 inputs, applies the standard OOF stability gates, and overlays only
accepted target components onto a branch-local frozen base CSV.

It is deliberately branch-aware:

* ``without_archive`` reads only official current train/test files.
* ``with_archive`` reads official current train/test plus official archive/train.

No local_eval/external_label/nonofficial file is read by this builder. LocalEval scoring happens
only after the output CSV is frozen by the separate scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as c180
import round2_c195_nc_nearmiss_residual_diversity as c195
import round2_c199_ei_c196_transfer_guard as c199
import round2_c207_egc_c180_transfer_guard as c207
import round2_c208_tg_robust_group_measurement as c208
import round2_c220_ei_electro_polar_autocorr as c220
import round2_c228_tg_c208_transfer_guard as c228
import round2_c232_tg_replicate_reliability_feature as c232
import round2_c242_nc_nearmiss_stability_ensemble as c242
import round2_c244_tg_median_residual_stack as c244
import round2_c282_current_only_reference as c282


TARGETS = tuple(reference.TARGETS)
SCHEMA = "ppp.round2.c1365.branch-oof-stability-overlay.v1"
SEED = 20260808
DEFAULT_TARGETS = "tg,egc,ei,nc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def guard_path(path: Path, *, role: str, branch: str | None = None, require_output_branch: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels", "nonofficial"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if branch == "without_archive" and "with_archive" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for without_archive: {path}")
    if branch == "with_archive" and "without_archive" in low:
        raise RuntimeError(f"Refusing cross-branch {role} path for with_archive: {path}")
    if require_output_branch and branch is not None and f"/{branch}/" not in low:
        raise RuntimeError(f"{role} path must stay in /{branch}/ namespace: {path}")


def parse_targets(value: str) -> tuple[str, ...]:
    targets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not targets:
        raise RuntimeError("No targets requested")
    bad = [target for target in targets if target not in TARGETS]
    if bad:
        raise RuntimeError(f"Invalid targets: {bad}")
    return targets


def build_branch_parent(data_dir: Path, branch: str) -> dict[str, Any]:
    if branch == "with_archive":
        train, test, archive, inputs = reference.load_inputs(data_dir)
    elif branch == "without_archive":
        train, test, inputs = c282.load_current_only_inputs(data_dir)
        archive = train.iloc[0:0].copy()
    else:
        raise RuntimeError(f"Unknown branch: {branch}")

    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "branch_parent": branch})
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, int(config["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(config["morgan_bits"])),
        reference.text_matrix(keys, int(config["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(config["morgan_bits"]))
    parent_detail, parent_oof, model_report = reference.fit_targets(
        pooled,
        test,
        keys,
        dense_base,
        cross_values,
        cross_available,
        sparse_parts,
        fingerprints,
        config,
    )
    test_model_detail = parent_detail[["id", "target_type", "model_prediction"]].copy()
    final_detail, override_report = reference.apply_official_overrides(test_model_detail, test, raw_labels)
    target_info: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        rows = parent_oof[parent_oof["target_type"].astype(str).eq(target)].reset_index(drop=True)
        canonical = rows["canonical"].to_numpy(object)
        indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        groups = np.asarray([parent_builder.no_stereo(value) for value in canonical], dtype=object)
        scaffolds = np.asarray([parent_builder.scaffold(value) for value in canonical], dtype=object)
        target_info[target] = {
            "canonical": canonical,
            "y": rows["target"].to_numpy(np.float64),
            "parent": rows["prediction"].to_numpy(np.float64),
            "folds": carrier.grouped_folds(groups),
            "indices": indices,
            "groups": groups,
            "scaffolds": scaffolds,
        }
    return {
        "branch": branch,
        "train": train,
        "test": test,
        "archive": archive,
        "raw_labels": raw_labels,
        "pooled": pooled,
        "inputs": inputs,
        "keys": keys,
        "key_to_index": key_to_index,
        "molecules": molecules,
        "fingerprints": fingerprints,
        "target_info": target_info,
        "test_parent_detail": final_detail[["id", "target_type", "target"]].copy(),
        "override_report": override_report,
        "model_report": model_report,
    }


def target_test_rows(parent: dict[str, Any], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = parent["test"].loc[parent["test"]["target_type"].astype(str).eq(target)].sort_values("id").reset_index(drop=True)
    detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"].astype(str).eq(target)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(rows["id"].to_numpy(np.int64), detail["id"].to_numpy(np.int64)):
        raise RuntimeError(f"Test ID alignment failed for {target}")
    indices = np.asarray([parent["key_to_index"][value] for value in rows["canonical"]], dtype=np.int64)
    return rows, indices, detail["target"].to_numpy(np.float64)


def unchanged_report(info: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    return {
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, parent)),
        "delta_r2": 0.0,
        "positive_folds": 0,
        "group_bootstrap_lower": 0.0,
        "minimum_panel_delta": 0.0,
        "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}},
        "folds": [],
        "pass": False,
        "unchanged_parent": True,
    }


def run_egc(parent: dict[str, Any], dense: np.ndarray, sparse_features: Any) -> dict[str, Any]:
    target = "egc"
    info = dict(parent["target_info"][target])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = target_test_rows(parent, target)
    raw = carrier.fit_target(info, dense, sparse_features, test_indices, test_parent)
    raw_report = carrier.evaluate_target(info, raw)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    raw_candidate = np.asarray(raw["candidate"], dtype=np.float64)
    nearest = c207.fold_local_nearest(parent, info)
    oof_guard, oof_guard_summary = c207.guard_mask_from_scaffold_similarity(np.asarray(info["scaffolds"], dtype=object), nearest)
    guarded_candidate = np.where(oof_guard, parent_oof, raw_candidate)
    guarded_report = carrier.evaluate_target(info, {"candidate": guarded_candidate})
    test_nearest = c207.full_train_nearest(parent, np.asarray(info["indices"], dtype=np.int64), test_indices)
    test_scaffolds = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"].astype(str)], dtype=object)
    test_guard, test_guard_summary = c207.guard_mask_from_scaffold_similarity(test_scaffolds, test_nearest)
    test_candidate = np.where(test_guard, test_parent, np.asarray(raw["test_direct"], dtype=np.float64))
    guarded_report.update(
        {
            "component": "egc_c180_transfer_guard",
            "raw_report": raw_report,
            "oof_guard_summary": oof_guard_summary,
            "test_guard_summary": test_guard_summary,
            "blend_name": raw["blend_name"],
            "blend_weights": [float(value) for value in raw["weights"]],
            "blend_intercept": float(raw["intercept"]),
        }
    )
    return {"target": target, "report": guarded_report, "oof": guarded_candidate, "test": test_candidate, "test_rows": test_rows}


def run_ei(parent: dict[str, Any], dense: np.ndarray, sparse_features: Any) -> dict[str, Any]:
    target = "ei"
    info = dict(parent["target_info"][target])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = target_test_rows(parent, target)
    raw = carrier.fit_target(info, dense, sparse_features, test_indices, test_parent)
    raw_report = carrier.evaluate_target(info, raw)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    raw_candidate = np.asarray(raw["candidate"], dtype=np.float64)
    shrunk = parent_oof + c199.SHRINK_ALPHA * (raw_candidate - parent_oof)
    nearest = c199.fold_local_nearest(parent, info)
    oof_guard = c199.transfer_guard_mask(np.asarray(info["scaffolds"], dtype=object), nearest)
    guarded = shrunk.copy()
    guarded[oof_guard] = parent_oof[oof_guard]
    report = carrier.evaluate_target(info, {"candidate": guarded})
    test_nearest = c199.full_train_nearest(parent, info, test_indices)
    test_scaffold = np.asarray([parent_builder.scaffold(value) for value in test_rows["canonical"].astype(str)], dtype=object)
    test_guard = c199.transfer_guard_mask(test_scaffold, test_nearest)
    shrunk_test = test_parent + c199.SHRINK_ALPHA * (np.asarray(raw["test_direct"], dtype=np.float64) - test_parent)
    test_candidate = shrunk_test.copy()
    test_candidate[test_guard] = test_parent[test_guard]
    report.update(
        {
            "component": "ei_c196_transfer_guard",
            "posthoc_repair_from_c196_failure_slices": True,
            "independent_confirmation_required_before_final_notebook": True,
            "raw_report": raw_report,
            "guarded_oof_rows": int(np.sum(oof_guard)),
            "guarded_test_rows": int(np.sum(test_guard)),
            "shrink_alpha": float(c199.SHRINK_ALPHA),
            "blend_name": raw["blend_name"],
            "blend_weights": [float(value) for value in raw["weights"]],
            "blend_intercept": float(raw["intercept"]),
        }
    )
    return {"target": target, "report": report, "oof": guarded, "test": test_candidate, "test_rows": test_rows}


def run_tg(repo_root: Path, parent: dict[str, Any], round1_dense: np.ndarray, round1_sparse: Any) -> dict[str, Any]:
    target = "tg"
    info = dict(parent["target_info"][target])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = target_test_rows(parent, target)
    c208_result = c208.fit_tg_robust(info, round1_dense, round1_sparse, test_indices, test_parent)
    guarded_oof, guarded_test, guarded_report, oof_guard, test_guard = c244.guarded_c208_candidate(
        parent,
        info,
        c208_result,
        test_rows,
        test_indices,
        test_parent,
    )
    reliability_result = c232.fit_tg_reliability(info, round1_dense, round1_sparse, test_indices, test_parent)
    reliability_report = c208.evaluate_tg(info, reliability_result)
    arms = {
        "c228_style_guarded_c208": np.asarray(guarded_oof, dtype=np.float64),
        "c232_style_replicate_reliability": np.asarray(reliability_result["candidate"], dtype=np.float64),
    }
    test_arms = {
        "c228_style_guarded_c208": np.asarray(guarded_test, dtype=np.float64),
        "c232_style_replicate_reliability": np.asarray(reliability_result["test_direct"], dtype=np.float64),
    }
    candidate, agreement = c244.agreement_median_residual_stack(np.asarray(info["parent"], dtype=np.float64), arms)
    test_candidate, test_agreement = c244.agreement_median_residual_stack(np.asarray(test_parent, dtype=np.float64), test_arms)
    report = c208.evaluate_tg(info, {"candidate": candidate})
    report.update(
        {
            "component": "tg_signed_agreement_median_residual",
            "guarded_c208_report": guarded_report,
            "reliability_report": reliability_report,
            "agreement_oof_rows": int(np.sum(agreement)),
            "agreement_test_rows": int(np.sum(test_agreement)),
            "guarded_c208_oof_rows": int(np.sum(oof_guard)),
            "guarded_c208_test_rows": int(np.sum(test_guard)),
            "stack_arms": list(c244.STACK_ARMS),
            "round1_feature_source": str(repo_root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        }
    )
    return {"target": target, "report": report, "oof": candidate, "test": test_candidate, "test_rows": test_rows}


def run_nc(repo_root: Path, parent: dict[str, Any], c180_dense: np.ndarray, c180_sparse: Any, round1_dense: np.ndarray, round1_sparse: Any, progress_path: Path) -> dict[str, Any]:
    target = "nc"
    info = dict(parent["target_info"][target])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = target_test_rows(parent, target)

    c180_result = carrier.fit_target(info, c180_dense, c180_sparse, test_indices, test_parent)
    c180_report = carrier.evaluate_target(info, c180_result)
    guarded_oof, guarded_test, guarded_report = c242.guarded_c180_candidate(parent, info, c180_result, test_rows, test_indices, test_parent)

    physical_matrix, physical_feature_report = c195.physical_feature_matrix(parent)
    physical_report, physical_oof, physical_test = c195.physical_nc_run(parent, physical_matrix, progress_path)
    c195_fixed_oof = 0.5 * np.asarray(c180_result["candidate"], dtype=np.float64) + 0.5 * np.asarray(physical_oof, dtype=np.float64)
    c195_fixed_test = 0.5 * np.asarray(c180_result["test_direct"], dtype=np.float64) + 0.5 * np.asarray(physical_test, dtype=np.float64)
    c195_fixed_report = carrier.evaluate_target(info, {"candidate": c195_fixed_oof})

    reliability_result = c232.fit_tg_reliability(info, round1_dense, round1_sparse, test_indices, test_parent)
    reliability_report = c208.evaluate_tg(info, reliability_result)

    c220.ACTIVE_TARGET = target
    c220.SEED = c242.SEED
    autocorr_features, autocorr_feature_report = c220.build_autocorr_features(list(parent["keys"]))
    electro_result = c220.fit_ei_residual(info, autocorr_features, test_indices, test_parent)
    electro_report = carrier.evaluate_target(info, {"candidate": electro_result["candidate"]})

    arms = {
        "c195_fixed_nearmiss_diversity": np.asarray(c195_fixed_oof, dtype=np.float64),
        "c226_style_guarded_c180": np.asarray(guarded_oof, dtype=np.float64),
        "c234_style_replicate_reliability": np.asarray(reliability_result["candidate"], dtype=np.float64),
        "c240_style_electro_polar_autocorr": np.asarray(electro_result["candidate"], dtype=np.float64),
    }
    test_arms = {
        "c195_fixed_nearmiss_diversity": np.asarray(c195_fixed_test, dtype=np.float64),
        "c226_style_guarded_c180": np.asarray(guarded_test, dtype=np.float64),
        "c234_style_replicate_reliability": np.asarray(reliability_result["test_direct"], dtype=np.float64),
        "c240_style_electro_polar_autocorr": np.asarray(electro_result["test_candidate"], dtype=np.float64),
    }
    candidate = sum(c242.WEIGHTS[name] * values for name, values in arms.items())
    test_candidate = sum(c242.WEIGHTS[name] * values for name, values in test_arms.items())
    report = carrier.evaluate_target(info, {"candidate": candidate})
    report.update(
        {
            "component": "nc_fixed_nearmiss_stability_ensemble",
            "fixed_weights": c242.WEIGHTS,
            "arm_reports": {
                "c180_raw": c180_report,
                "c195_fixed_nearmiss_diversity": c195_fixed_report,
                "c226_style_guarded_c180": guarded_report,
                "c234_style_replicate_reliability": reliability_report,
                "c240_style_electro_polar_autocorr": electro_report,
                "physical_electronic": physical_report,
            },
            "feature_reports": {
                "physical_electronic": physical_feature_report,
                "autocorr": autocorr_feature_report,
                "round1_feature_source": str(repo_root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
            },
        }
    )
    return {"target": target, "report": report, "oof": candidate, "test": test_candidate, "test_rows": test_rows}


def load_base(path: Path, ids: np.ndarray, branch: str) -> pd.DataFrame:
    guard_path(path, role="base candidate", branch=branch, require_output_branch=True)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"] or len(frame) != len(ids):
        raise RuntimeError(f"Invalid base candidate schema: {path}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(np.int64), ids):
        raise RuntimeError(f"Invalid base candidate IDs/order: {path}")
    if not np.isfinite(frame["target"].to_numpy(np.float64)).all():
        raise RuntimeError(f"Base candidate has non-finite values: {path}")
    return frame


def apply_branch_overrides(parent: dict[str, Any], predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    detail = parent["test"][["id", "target_type"]].copy()
    detail["model_prediction"] = predictions["target"].to_numpy(np.float64)
    final_detail, override_report = reference.apply_official_overrides(detail, parent["test"], parent["raw_labels"])
    return final_detail[["id", "target"]].copy(), override_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="..")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--branch", choices=("with_archive", "without_archive"), required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    started = time.time()
    round2_root = Path(__file__).resolve().parents[1]
    repo_root = Path(args.repo_root).resolve()
    if repo_root.name == "Polymer Prediction Challenge Round 2":
        repo_root = repo_root.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (round2_root / data_dir).resolve()
    base_path = Path(args.base_csv)
    if not base_path.is_absolute():
        base_path = (round2_root / base_path).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (round2_root / output).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (round2_root / run_dir).resolve()

    for path, role in ((data_dir, "data dir"), (base_path, "base candidate")):
        guard_path(path, role=role, branch=args.branch if role == "base candidate" else None)
    guard_path(output, role="output", branch=args.branch, require_output_branch=True)
    guard_path(run_dir, role="run dir")
    if output.exists() or run_dir.exists():
        raise RuntimeError("Refusing to overwrite/reuse output or run dir")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    append_jsonl(progress_path, {"stage": "started", "created_at": datetime.now().astimezone().isoformat(), "branch": args.branch})

    active_targets = parse_targets(args.targets)
    parent = build_branch_parent(data_dir, args.branch)
    ids = parent["test"]["id"].to_numpy(np.int64)
    base = load_base(base_path, ids, args.branch)
    final_values = base["target"].to_numpy(np.float64).copy()
    append_jsonl(
        progress_path,
        {
            "stage": "parent_ready",
            "branch": args.branch,
            "keys": len(parent["keys"]),
            "train_rows": int(len(parent["train"])),
            "archive_rows_used": int(len(parent["archive"])),
            "base_sha256": sha256_file(base_path),
        },
    )

    c180_dense = c180_sparse = round1_dense = round1_sparse = None
    c180_feature_report: dict[str, Any] | None = None
    round1_feature_report: dict[str, Any] | None = None
    if any(target in active_targets for target in ("egc", "ei", "nc")):
        c180_dense, c180_sparse, c180_feature_report = c180.build_features(repo_root, parent["keys"])
        append_jsonl(progress_path, {"stage": "c180_features_ready", "dense_shape": c180_feature_report["dense_shape"], "sparse_shape": c180_feature_report["sparse_shape"]})
    if any(target in active_targets for target in ("tg", "nc")):
        round1_dense, round1_sparse, round1_feature_report = carrier.build_round1_features(repo_root, parent["keys"])
        append_jsonl(progress_path, {"stage": "round1_features_ready", "dense_shape": round1_feature_report["dense_shape"], "sparse_shape": round1_feature_report["sparse_shape"]})

    components: dict[str, Any] = {}
    for target in active_targets:
        target_started = time.time()
        append_jsonl(progress_path, {"stage": "target_started", "target": target})
        if target == "egc":
            if c180_dense is None or c180_sparse is None:
                raise RuntimeError("C180 features not initialized")
            component = run_egc(parent, c180_dense, c180_sparse)
        elif target == "ei":
            if c180_dense is None or c180_sparse is None:
                raise RuntimeError("C180 features not initialized")
            component = run_ei(parent, c180_dense, c180_sparse)
        elif target == "tg":
            if round1_dense is None or round1_sparse is None:
                raise RuntimeError("Round1 features not initialized")
            component = run_tg(repo_root, parent, round1_dense, round1_sparse)
        elif target == "nc":
            if c180_dense is None or c180_sparse is None or round1_dense is None or round1_sparse is None:
                raise RuntimeError("Required NC features not initialized")
            component = run_nc(repo_root, parent, c180_dense, c180_sparse, round1_dense, round1_sparse, progress_path)
        else:
            raise RuntimeError(f"Target {target} is not implemented in this runner")
        report = component["report"]
        accepted = bool(report.get("pass", False))
        components[target] = {
            "accepted": accepted,
            "report": report,
            "elapsed_seconds": float(time.time() - target_started),
        }
        if accepted:
            test_rows = component["test_rows"]
            values = np.asarray(component["test"], dtype=np.float64)
            id_to_value = dict(zip(test_rows["id"].astype(int), values, strict=True))
            mask = parent["test"]["target_type"].astype(str).eq(target).to_numpy()
            final_values[mask] = parent["test"].loc[mask, "id"].astype(int).map(id_to_value).to_numpy(np.float64)
        append_jsonl(
            progress_path,
            {
                "stage": "target_finished",
                "target": target,
                "accepted": accepted,
                "delta_r2": float(report.get("delta_r2", 0.0)),
                "positive_folds": int(report.get("positive_folds", 0)),
                "bootstrap_lower": float(report.get("group_bootstrap_lower", 0.0)),
                "minimum_panel_delta": float(report.get("minimum_panel_delta", 0.0)),
                "elapsed_seconds": float(time.time() - target_started),
            },
        )

    assembled = pd.DataFrame({"id": ids, "target": final_values})
    assembled, final_override_report = apply_branch_overrides(parent, assembled)
    if len(assembled) != 4940 or assembled["id"].duplicated().any() or not np.array_equal(assembled["id"].to_numpy(np.int64), np.arange(1, 4941)):
        raise RuntimeError("Final output row/order contract failed")
    if not np.isfinite(assembled["target"].to_numpy(np.float64)).all():
        raise RuntimeError("Final output contains non-finite values")
    assembled.to_csv(output, index=False)

    report = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": args.branch,
        "classification": "CLEAN_OFFICIAL_ONLY",
        "official_current_train_used": True,
        "archive_labels_used": bool(args.branch == "with_archive"),
        "archive_rows_used": int(len(parent["archive"])),
        "local_eval_read_by_builder": False,
        "external_label_file_read_by_builder": False,
        "nonofficial_file_read_by_builder": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "config": vars(args),
        "inputs": parent["inputs"],
        "base_candidate": {"path": str(base_path), "sha256": sha256_file(base_path), "rows": int(len(base))},
        "active_targets": list(active_targets),
        "accepted_targets": [target for target, value in components.items() if value["accepted"]],
        "components": components,
        "parent_model_report": parent["model_report"],
        "parent_override_report": parent["override_report"],
        "final_override_report": final_override_report,
        "feature_reports": {
            "c180": c180_feature_report,
            "round1": round1_feature_report,
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": int(len(assembled)),
            "bytes": output.stat().st_size,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "rdkit": reference.Chem.rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "initial_reference_pipeline": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c180": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "c199": sha256_file(round2_root / "tools/round2_c199_ei_c196_transfer_guard.py"),
            "c207": sha256_file(round2_root / "tools/round2_c207_egc_c180_transfer_guard.py"),
            "c242": sha256_file(round2_root / "tools/round2_c242_nc_nearmiss_stability_ensemble.py"),
            "c244": sha256_file(round2_root / "tools/round2_c244_tg_median_residual_stack.py"),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "config.json", vars(args))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "target": target,
                "accepted": value["accepted"],
                "delta_r2": value["report"].get("delta_r2"),
                "positive_folds": value["report"].get("positive_folds"),
                "group_bootstrap_lower": value["report"].get("group_bootstrap_lower"),
                "minimum_panel_delta": value["report"].get("minimum_panel_delta"),
                "component": value["report"].get("component"),
            }
            for target, value in components.items()
        ]
    ).to_csv(run_dir / "component_summary.csv", index=False)
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{sha256_file(path)}  {path.name}")
    manifest.append(f"{sha256_file(output)}  OUTPUT {output}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    append_jsonl(progress_path, {"stage": "finished", "output_sha256": report["output"]["sha256"], "accepted_targets": report["accepted_targets"], "elapsed_seconds": report["elapsed_seconds"]})
    print(json.dumps({"output": report["output"], "accepted_targets": report["accepted_targets"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
