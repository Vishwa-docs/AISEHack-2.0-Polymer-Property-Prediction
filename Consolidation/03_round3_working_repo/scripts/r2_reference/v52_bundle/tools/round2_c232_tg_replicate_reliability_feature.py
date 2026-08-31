#!/usr/bin/env python3
"""C232: Tg fold-local predicted replicate-reliability feature.

C228 narrowly missed the Tg component gate, but the C208/C228 robust-median and
fixed-panel guard branch is cooled.  This child tests one distinct factor:
append fold-local *predicted* canonical-group replicate/reliability scalars to
the unchanged C127 official-SMILES/RDKit/Morgan carrier, while training the Tg
heads on the original official Tg labels.

The reliability scalars are learned only from outer-training Tg groups:
duplicate count, within-group range/MAD, and high-dispersion indicators are
computed for training groups, then predicted for validation/test structures from
official molecular features.  A validation row's own group labels are never used
to create its reliability feature.

No local_eval external_labels, public feedback, PI1M, cross-target labels, stored
predictions, pretrained weights, Kaggle compute, upload, submission, or final
notebook action is used.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c208_tg_robust_group_measurement as c208


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "tg"
SCHEMA = "ppp.round2.c232.tg-replicate-reliability-feature.v1"
RELIABILITY_TREES = 120
RELIABILITY_LEAF = 3
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


def reliability_target_table(y: np.ndarray, groups: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    """Build group-level reliability targets from a training-only Tg split."""

    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    indices = np.asarray(indices, dtype=np.int64)
    global_mad = max(float(np.median(np.abs(y - np.median(y)))), 1.0e-12)

    rep_indices: list[int] = []
    targets: list[list[float]] = []
    duplicate_groups = 0
    duplicate_rows = 0
    high_dispersion_groups = 0
    range_values: list[float] = []
    mad_values: list[float] = []
    counts: list[int] = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        values = y[rows]
        count = int(len(rows))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) if count >= 2 else 0.0
        value_range = float(np.max(values) - np.min(values)) if count >= 2 else 0.0
        high_dispersion = float(count >= 2 and mad > global_mad)
        rep_indices.append(int(indices[rows[0]]))
        targets.append(
            [
                float(np.log1p(max(count - 1, 0))),
                float(np.log1p(max(value_range, 0.0))),
                float(np.log1p(max(mad, 0.0))),
                high_dispersion,
            ]
        )
        counts.append(count)
        if count >= 2:
            duplicate_groups += 1
            duplicate_rows += count
            range_values.append(value_range)
            mad_values.append(mad)
            if high_dispersion:
                high_dispersion_groups += 1

    return {
        "rep_indices": np.asarray(rep_indices, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.float64),
        "report": {
            "training_rows": int(len(y)),
            "unique_groups": int(len(rep_indices)),
            "duplicate_groups": int(duplicate_groups),
            "duplicate_rows": int(duplicate_rows),
            "max_group_count": int(max(counts) if counts else 0),
            "global_mad": float(global_mad),
            "median_duplicate_range": float(np.median(range_values)) if range_values else 0.0,
            "median_duplicate_mad": float(np.median(mad_values)) if mad_values else 0.0,
            "max_duplicate_range": float(np.max(range_values)) if range_values else 0.0,
            "max_duplicate_mad": float(np.max(mad_values)) if mad_values else 0.0,
            "high_dispersion_groups_mad_gt_global_mad": int(high_dispersion_groups),
            "reliability_targets": [
                "log1p(count_minus_one)",
                "log1p(group_range)",
                "log1p(group_mad)",
                "mad_gt_split_global_mad_indicator",
            ],
        },
    }


def predict_reliability_features(dense: np.ndarray, table: dict[str, Any], prediction_indices: np.ndarray) -> np.ndarray:
    """Predict reliability scalars from official molecular dense features."""

    rep_indices = np.asarray(table["rep_indices"], dtype=np.int64)
    targets = np.asarray(table["targets"], dtype=np.float64)
    prediction_indices = np.asarray(prediction_indices, dtype=np.int64)
    if len(rep_indices) < carrier.N_FOLDS:
        raise RuntimeError("not enough groups for reliability feature model")
    if targets.shape[1] != 4:
        raise RuntimeError("unexpected reliability target width")

    train_dense, prediction_dense = carrier.dense_pair(dense, rep_indices, prediction_indices)
    columns: list[np.ndarray] = []
    for column in range(targets.shape[1]):
        model = ExtraTreesRegressor(
            n_estimators=RELIABILITY_TREES,
            min_samples_leaf=RELIABILITY_LEAF,
            max_features=0.75,
            random_state=carrier.SEED + 100 + column,
            n_jobs=2,
        )
        model.fit(train_dense, targets[:, column])
        columns.append(model.predict(prediction_dense))
    predicted = np.column_stack(columns).astype(np.float64)
    predicted[:, :3] = np.maximum(predicted[:, :3], 0.0)
    predicted[:, 3] = np.clip(predicted[:, 3], 0.0, 1.0)
    predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(predicted).all():
        raise RuntimeError("non-finite predicted reliability feature")
    return predicted


def dense_pair_with_reliability(
    dense: np.ndarray,
    train_rows: np.ndarray,
    prediction_rows: np.ndarray,
    train_reliability: np.ndarray,
    prediction_reliability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_train, base_prediction = carrier.dense_pair(dense, train_rows, prediction_rows)
    train_reliability = np.asarray(train_reliability, dtype=np.float64)
    prediction_reliability = np.asarray(prediction_reliability, dtype=np.float64)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_rel = scaler.fit_transform(imputer.fit_transform(train_reliability))
    prediction_rel = scaler.transform(imputer.transform(prediction_reliability))
    return np.hstack([base_train, train_rel]), np.hstack([base_prediction, prediction_rel])


def outer_reliability_features(
    dense: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create fold-local reliability features for one outer split."""

    rel_train = np.full((len(training), 4), np.nan, dtype=np.float64)
    local_groups = groups[training]
    inner_folds = carrier.grouped_folds(local_groups)
    inner_reports: list[dict[str, Any]] = []
    for inner_fold in range(carrier.N_FOLDS):
        inner_validation_pos = np.flatnonzero(inner_folds == inner_fold)
        inner_training_pos = np.flatnonzero(inner_folds != inner_fold)
        inner_training = training[inner_training_pos]
        table = reliability_target_table(y[inner_training], groups[inner_training], indices[inner_training])
        rel_train[inner_validation_pos] = predict_reliability_features(dense, table, indices[training[inner_validation_pos]])
        inner_reports.append({"inner_fold": int(inner_fold), **table["report"]})
    if not np.isfinite(rel_train).all():
        raise RuntimeError("non-finite inner cross-fitted reliability feature")

    outer_table = reliability_target_table(y[training], groups[training], indices[training])
    rel_validation = predict_reliability_features(dense, outer_table, indices[validation])
    return rel_train, rel_validation, {"outer_table": outer_table["report"], "inner_tables": inner_reports}


def fit_tg_reliability(
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
    fold_reliability_reports: list[dict[str, Any]] = []

    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        rel_train, rel_validation, rel_report = outer_reliability_features(dense, y, groups, indices, training, validation)
        train_dense, validation_dense = dense_pair_with_reliability(
            dense,
            indices[training],
            indices[validation],
            rel_train,
            rel_validation,
        )
        train_matrix = sparse.hstack([sparse_features[indices[training]], sparse.csr_matrix(train_dense)], format="csr")
        validation_matrix = sparse.hstack([sparse_features[indices[validation]], sparse.csr_matrix(validation_dense)], format="csr")

        ridge = Ridge(alpha=carrier.RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
        ridge.fit(train_matrix, y[training])
        direct_oof[validation, 0] = ridge.predict(validation_matrix)

        tree = ExtraTreesRegressor(
            n_estimators=carrier.TREE_ESTIMATORS,
            min_samples_leaf=carrier.TREE_LEAF,
            max_features=0.65,
            random_state=carrier.SEED,
            n_jobs=2,
        )
        tree.fit(train_dense, y[training])
        direct_oof[validation, 1] = tree.predict(validation_dense)
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "parent_r2": float(r2_score(y[validation], parent[validation])),
                "ridge_r2": float(r2_score(y[validation], direct_oof[validation, 0])),
                "tree_r2": float(r2_score(y[validation], direct_oof[validation, 1])),
                "mean_predicted_log_count_minus_one": float(np.mean(rel_validation[:, 0])),
                "mean_predicted_log_group_range": float(np.mean(rel_validation[:, 1])),
                "mean_predicted_log_group_mad": float(np.mean(rel_validation[:, 2])),
                "mean_predicted_high_dispersion": float(np.mean(rel_validation[:, 3])),
            }
        )
        fold_reliability_reports.append({"fold": int(fold), **rel_report})

    if not np.isfinite(direct_oof).all():
        raise RuntimeError("non-finite C232 OOF arm")

    arms = np.column_stack([parent, direct_oof])
    weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, arms)
    candidate = arms @ weights + intercept

    full_table = reliability_target_table(y, groups, indices)
    full_train_rel = predict_reliability_features(dense, full_table, indices)
    test_rel = predict_reliability_features(dense, full_table, test_indices)
    full_dense, test_dense = dense_pair_with_reliability(dense, indices, test_indices, full_train_rel, test_rel)
    full_matrix = sparse.hstack([sparse_features[indices], sparse.csr_matrix(full_dense)], format="csr")
    test_matrix = sparse.hstack([sparse_features[test_indices], sparse.csr_matrix(test_dense)], format="csr")

    full_ridge = Ridge(alpha=carrier.RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
    full_ridge.fit(full_matrix, y)
    full_tree = ExtraTreesRegressor(
        n_estimators=carrier.TREE_ESTIMATORS,
        min_samples_leaf=carrier.TREE_LEAF,
        max_features=0.65,
        random_state=carrier.SEED,
        n_jobs=2,
    )
    full_tree.fit(full_dense, y)
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
        "fold_reliability_reports": fold_reliability_reports,
        "full_reliability_report": full_table["report"],
        # c208.evaluate_tg expects these keys; keep aliases for panel reuse.
        "fold_robust_reports": fold_reliability_reports,
        "full_robust_report": full_table["report"],
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
            "active_feature_route": "C127 official-SMILES/RDKit/Morgan carrier plus fold-local predicted Tg replicate-reliability scalars",
            "active_target": ACTIVE_TARGET,
            "changed_factor": "append fold-local predicted replicate count/range/MAD/high-dispersion scalars; train Tg heads on original labels",
            "reliability_target_count": 4,
            "reliability_trees": RELIABILITY_TREES,
            "reliability_leaf": RELIABILITY_LEAF,
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
        }
    )
    checkpoint(progress, "features_complete", dense_shape=feature_report["dense_shape"], sparse_shape=feature_report["sparse_shape"])

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = c208.target_test_rows(parent, ACTIVE_TARGET)
    result = fit_tg_reliability(info, dense, sparse_features, test_indices, test_parent)
    active_report = c208.evaluate_tg(info, result)
    active_report.update(
        {
            "changed_factor": "fold-local predicted Tg replicate-reliability feature appended to C127 carrier",
            "uses_original_tg_labels": True,
            "uses_robust_target_median_or_mad_downweighting": False,
            "c208_c228_guard_or_panel_retune": False,
            "reliability_feature_names": [
                "predicted_log1p_count_minus_one",
                "predicted_log1p_group_range",
                "predicted_log1p_group_mad",
                "predicted_high_dispersion_probability",
            ],
            "fold_reliability_reports": result["fold_reliability_reports"],
            "full_reliability_report": result["full_reliability_report"],
        }
    )
    active_report.pop("fold_robust_reports", None)
    active_report.pop("full_robust_report", None)
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "tg_replicate_reliability_feature_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        pass_gate=banked,
        positive_folds=active_report["positive_folds"],
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
            report = c208.unchanged_report(target_info)
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
        raise RuntimeError("C232 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C232 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C232 prediction finite check failed")

    report = {
        "schema_version": SCHEMA,
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
        "component_diagnostics": {
            "active_target": ACTIVE_TARGET,
            "changed_factor": "fold-local predicted Tg replicate-reliability feature",
            "tg_delta_r2": active_report["delta_r2"],
            "tg_positive_folds": active_report["positive_folds"],
            "tg_group_bootstrap_lower": active_report["group_bootstrap_lower"],
            "tg_minimum_panel_delta": active_report["minimum_panel_delta"],
            "c208_c228_branch_not_reused": True,
            "uses_original_tg_labels": True,
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
        },
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c208_panel_helper": sha256_file(round2_root / "tools/round2_c208_tg_robust_group_measurement.py"),
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
            "schema_version": SCHEMA,
            "seed": carrier.SEED,
            "target": ACTIVE_TARGET,
            "feature_basis": "C127 official-SMILES/RDKit/Morgan carrier",
            "changed_factor": "append fold-local predicted Tg replicate count/range/MAD/high-dispersion scalars",
            "reliability_model": {
                "estimator": "ExtraTreesRegressor",
                "n_estimators": RELIABILITY_TREES,
                "min_samples_leaf": RELIABILITY_LEAF,
                "max_features": 0.75,
                "targets": [
                    "log1p(count_minus_one)",
                    "log1p(group_range)",
                    "log1p(group_mad)",
                    "mad_gt_split_global_mad_indicator",
                ],
            },
            "main_models": {
                "ridge_alpha": carrier.RIDGE_ALPHA,
                "extra_trees": {
                    "n_estimators": carrier.TREE_ESTIMATORS,
                    "min_samples_leaf": carrier.TREE_LEAF,
                    "max_features": 0.65,
                },
            },
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
