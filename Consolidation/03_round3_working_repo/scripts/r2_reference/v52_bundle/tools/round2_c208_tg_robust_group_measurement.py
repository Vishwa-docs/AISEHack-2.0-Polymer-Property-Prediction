#!/usr/bin/env python3
"""C208: Tg robust canonical-group measurement-noise component.

This child tests one bounded official-only factor: a Tg-only direct carrier
trained on the existing C127 official-SMILES/RDKit/Morgan feature basis, but
with fold-local canonical-group median targets and fixed downweighting for
high-dispersion duplicate Tg groups.  The validation row's own group labels are
never used to construct its training target or sample weight.

It reads no local_eval/test-external_label file, no public feedback, no stored prediction
arrays, no PI1M representation, and performs no Kaggle/upload/submission/final
notebook action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "tg"
MIN_BANKABLE_DELTA_R2 = 0.01
MIN_FULL_MEAN_GAIN = 0.002


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": stage, "time": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), **payload},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def unchanged_report(info: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    return {
        "active": False,
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


def target_test_rows(parent: dict[str, Any], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    test_detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"] == target]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(test_rows["id"].to_numpy(np.int64), test_detail["id"].to_numpy(np.int64)):
        raise RuntimeError(f"test ID alignment failed for {target}")
    indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    return test_rows, indices, test_detail["target"].to_numpy(np.float64)


def robust_targets_and_weights(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute deterministic train-only robust Tg targets and weights.

    For duplicate canonical-no-stereo Tg groups inside the current training
    split, each row is trained toward that split's group median.  The group is
    downweighted by a fixed smooth function of its median absolute deviation
    relative to the split-wide Tg MAD.  Singletons retain their own target and
    unit weight.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    robust_y = y.copy()
    sample_weight = np.ones(len(y), dtype=np.float64)
    global_median = float(np.median(y))
    global_mad = float(np.median(np.abs(y - global_median)))
    scale = max(global_mad, 1.0e-12)
    duplicate_groups = 0
    duplicate_rows = 0
    high_dispersion_groups = 0
    group_mads: list[float] = []
    unique_groups = np.unique(groups)
    for group in unique_groups:
        rows = np.flatnonzero(groups == group)
        if len(rows) < 2:
            continue
        values = y[rows]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        weight = float(1.0 / (1.0 + mad / scale))
        robust_y[rows] = median
        sample_weight[rows] = weight
        duplicate_groups += 1
        duplicate_rows += int(len(rows))
        group_mads.append(mad)
        if mad > scale:
            high_dispersion_groups += 1
    return robust_y, sample_weight, {
        "rows": int(len(y)),
        "unique_groups": int(len(unique_groups)),
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "global_mad": global_mad,
        "median_duplicate_group_mad": float(np.median(group_mads)) if group_mads else 0.0,
        "max_duplicate_group_mad": float(np.max(group_mads)) if group_mads else 0.0,
        "high_dispersion_groups_mad_gt_global_mad": high_dispersion_groups,
        "min_sample_weight": float(np.min(sample_weight)),
        "mean_sample_weight": float(np.mean(sample_weight)),
        "target_transform": "train-split canonical-no-stereo duplicate group median",
        "weight_formula": "1 / (1 + duplicate_group_mad / train_split_global_mad)",
    }


def fit_tg_robust(
    info: dict[str, Any],
    dense: np.ndarray,
    sparse_features: sparse.csr_matrix,
    test_indices: np.ndarray,
    test_parent: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    direct_oof = np.full((len(y), 2), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    fold_robust_reports: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        robust_y, sample_weight, robust_report = robust_targets_and_weights(y[training], groups[training])
        train_dense, validation_dense = carrier.dense_pair(dense, indices[training], indices[validation])
        train_matrix = sparse.hstack([sparse_features[indices[training]], sparse.csr_matrix(train_dense)], format="csr")
        validation_matrix = sparse.hstack([sparse_features[indices[validation]], sparse.csr_matrix(validation_dense)], format="csr")
        ridge = Ridge(alpha=carrier.RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
        ridge.fit(train_matrix, robust_y, sample_weight=sample_weight)
        direct_oof[validation, 0] = ridge.predict(validation_matrix)
        tree = ExtraTreesRegressor(
            n_estimators=carrier.TREE_ESTIMATORS,
            min_samples_leaf=carrier.TREE_LEAF,
            max_features=0.65,
            random_state=carrier.SEED,
            n_jobs=2,
        )
        tree.fit(train_dense, robust_y, sample_weight=sample_weight)
        direct_oof[validation, 1] = tree.predict(validation_dense)
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(len(validation)),
                "parent_r2": float(r2_score(y[validation], parent[validation])),
                "ridge_r2": float(r2_score(y[validation], direct_oof[validation, 0])),
                "tree_r2": float(r2_score(y[validation], direct_oof[validation, 1])),
            }
        )
        fold_robust_reports.append({"fold": fold, **robust_report})
    if not np.isfinite(direct_oof).all():
        raise RuntimeError("non-finite C208 OOF arm")
    arms = np.column_stack([parent, direct_oof])
    weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, arms)
    candidate = arms @ weights + intercept

    robust_y, sample_weight, full_robust_report = robust_targets_and_weights(y, groups)
    full_dense, test_dense = carrier.dense_pair(dense, indices, test_indices)
    full_matrix = sparse.hstack([sparse_features[indices], sparse.csr_matrix(full_dense)], format="csr")
    test_matrix = sparse.hstack([sparse_features[test_indices], sparse.csr_matrix(test_dense)], format="csr")
    full_ridge = Ridge(alpha=carrier.RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
    full_ridge.fit(full_matrix, robust_y, sample_weight=sample_weight)
    full_tree = ExtraTreesRegressor(
        n_estimators=carrier.TREE_ESTIMATORS,
        min_samples_leaf=carrier.TREE_LEAF,
        max_features=0.65,
        random_state=carrier.SEED,
        n_jobs=2,
    )
    full_tree.fit(full_dense, robust_y, sample_weight=sample_weight)
    test_arms = np.column_stack([test_parent, full_ridge.predict(test_matrix), full_tree.predict(test_dense)])
    return {
        "candidate": candidate,
        "test_direct": test_arms @ weights + intercept,
        "folds": fold_rows,
        "weights": weights,
        "intercept": float(intercept),
        "blend_name": blend_name,
        "blend_r2": float(blend_r2),
        "direct_oof": direct_oof,
        "fold_robust_reports": fold_robust_reports,
        "full_robust_report": full_robust_report,
    }


def evaluate_tg(info: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    candidate = np.asarray(result["candidate"], dtype=np.float64)
    groups = np.asarray(info["groups"], dtype=object)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    folds = carrier.grouped_folds(groups)
    nearest = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [info["fingerprints"][int(info["indices"][row])] for row in training]
        for row in validation:
            nearest[row] = max(reference.DataStructs.BulkTanimotoSimilarity(info["fingerprints"][int(info["indices"][row])], train_fps))

    counts = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy(dtype=np.int64)
    group_mad_by_group: dict[Any, float] = {}
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        values = y[rows]
        median = float(np.median(values))
        group_mad_by_group[group] = float(np.median(np.abs(values - median)))
    group_mads = np.asarray([group_mad_by_group[group] for group in groups], dtype=np.float64)
    global_mad = max(float(np.median(np.abs(y - np.median(y)))), 1.0e-12)

    panel_specs: dict[str, np.ndarray] = {
        "similarity_lt_0.30": nearest < 0.30,
        "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
        "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
        "similarity_ge_0.70": nearest >= 0.70,
        "quantile_low": y <= np.quantile(y, 0.25),
        "quantile_high": y >= np.quantile(y, 0.75),
        "duplicate_group_count_ge2": counts >= 2,
        "duplicate_group_count_ge3": counts >= 3,
        "duplicate_group_mad_gt_global_mad": (counts >= 2) & (group_mads > global_mad),
    }
    for name in sorted(set(scaffolds)):
        selected = scaffolds == name
        if int(np.sum(selected)) >= 10:
            panel_specs[f"scaffold_{name}"] = selected
    panels: dict[str, Any] = {}
    panel_values: list[float] = []
    for name, selected in panel_specs.items():
        delta = carrier.panel_delta(y, parent, candidate, selected)
        panels[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable",
        }
        if delta is not None:
            panel_values.append(delta)
    fold_rows = []
    for fold in range(carrier.N_FOLDS):
        selected = folds == fold
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(np.sum(selected)),
                "parent_r2": float(r2_score(y[selected], parent[selected])),
                "candidate_r2": float(r2_score(y[selected], candidate[selected])),
                "delta_r2": float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected])),
            }
        )
    delta = float(r2_score(y, candidate) - r2_score(y, parent))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    lower = carrier.bootstrap_lower(y, parent, candidate, groups)
    minimum_panel = float(min(panel_values)) if panel_values else 0.0
    passed = bool(delta >= MIN_BANKABLE_DELTA_R2 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0)
    return {
        "active": True,
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_panel_delta": minimum_panel,
        "panels": panels,
        "folds": fold_rows,
        "pass": passed,
        "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
        "blend_name": result["blend_name"],
        "blend_weights": [float(value) for value in result["weights"]],
        "blend_intercept": float(result["intercept"]),
        "blend_r2": float(result["blend_r2"]),
        "fold_robust_reports": result["fold_robust_reports"],
        "full_robust_report": result["full_robust_report"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--canonical-run",
        default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7",
    )
    args = parser.parse_args()
    started = time.time()
    root = Path(args.root).resolve()
    round2_root = root / "Polymer Prediction Challenge Round 2"
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    progress = run_dir / "progress.jsonl"
    checkpoint(progress, "started", experiment_id=run_dir.name)

    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)

    dense, sparse_features, feature_report = carrier.build_round1_features(root, parent["keys"])
    feature_report = dict(feature_report)
    feature_report.update(
        {
            "active_feature_route": "C127 official-SMILES/RDKit/Morgan carrier",
            "active_target": ACTIVE_TARGET,
            "changed_factor": "fold-local duplicate canonical-group Tg median targets and fixed dispersion downweighting",
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
        }
    )
    checkpoint(progress, "features_complete", dense_shape=feature_report["dense_shape"], sparse_shape=feature_report["sparse_shape"])

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = target_test_rows(parent, ACTIVE_TARGET)
    result = fit_tg_robust(info, dense, sparse_features, test_indices, test_parent)
    active_report = evaluate_tg(info, result)
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "tg_robust_group_measurement_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        pass_gate=banked,
        minimum_panel_delta=active_report["minimum_panel_delta"],
        group_bootstrap_lower=active_report["group_bootstrap_lower"],
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = active_report
            candidate = np.asarray(result["candidate"], dtype=np.float64)
            direct_oof = np.asarray(result["direct_oof"], dtype=np.float64)
        else:
            report = unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=np.float64)
            direct_oof = np.full((len(candidate), 2), np.nan, dtype=np.float64)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        assembled = candidate if target == ACTIVE_TARGET and banked else np.asarray(target_info["parent"], dtype=np.float64)
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": candidate,
                    "assembled": assembled,
                    "direct_ridge": direct_oof[:, 0],
                    "direct_tree": direct_oof[:, 1],
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": folds,
                    "banked": bool(target == ACTIVE_TARGET and banked),
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target == ACTIVE_TARGET and banked else 0.0 for target in TARGETS))
    full_candidate_gate_pass = bool(banked and assembled_mean - parent_mean >= MIN_FULL_MEAN_GAIN and max_loss >= -0.003)

    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component_test = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "direct_candidate": result["test_direct"],
        }
    )
    predictions = parent_test.merge(component_test, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(
        (predictions["target_type"] == ACTIVE_TARGET) & banked,
        predictions["direct_candidate"],
        predictions["target"],
    )
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940:
        raise RuntimeError("C208 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C208 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C208 prediction finite check failed")

    report = {
        "schema_version": "ppp.round2.c208.tg-robust-group-measurement.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "stored_prediction_replay": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "active_target": ACTIVE_TARGET,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(full_candidate_gate_pass and assembled_mean >= 0.95),
        "decision": "banked_component_pending_compound_audit" if banked else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component_test.to_csv(run_dir / "tg_component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": report["schema_version"],
            "seed": carrier.SEED,
            "target": ACTIVE_TARGET,
            "feature_basis": "C127 official-SMILES/RDKit/Morgan carrier",
            "changed_factor": "fold-local canonical-no-stereo duplicate Tg group median labels plus fixed MAD downweighting",
            "normal_component_gate": "delta >= 0.01, positive folds >= 4/5, grouped-bootstrap lower > 0, all explicit panel minima >= 0",
            "no_hyperparameter_sweep": True,
            "local_eval_read": False,
            "pi1m_used": False,
        },
    )
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"sklearn={__import__('sklearn').__version__}",
                f"rdkit={reference.Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: **{report['decision']}**. Tg parent `{active_report['parent_r2']:.12f}`; "
        f"candidate `{active_report['candidate_r2']:.12f}`; delta `{active_report['delta_r2']:+.12f}`. "
        f"Positive folds `{active_report['positive_folds']}/5`; bootstrap lower `{active_report['group_bootstrap_lower']:.12f}`; "
        f"minimum panel delta `{active_report['minimum_panel_delta']:.12f}`. "
        "No local_eval/Kaggle/submission/final-notebook action.\n",
        encoding="utf-8",
    )
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "banked_targets": report["banked_targets"],
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
                "tg_delta_r2": active_report["delta_r2"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
