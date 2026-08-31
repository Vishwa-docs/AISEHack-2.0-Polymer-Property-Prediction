#!/usr/bin/env python3
"""C210: Nc optical-dispersion gap residual.

This is a bounded official-only Nc child. It rebuilds exact C050 from source,
then tests one new factor: fold-nested structure-only predictions of Egc/Egb
are converted into fixed optical-dispersion gap coordinates and used by one
low-variance Ridge residual on Nc - C050_Nc.

It does not replay stored C093/C195/C197 predictions, does not use EPS labels or
predictions, does not use PI1M, does not use local_eval/public feedback, and does
not perform any Kaggle/upload/submission/final-notebook action.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "nc"
AUXILIARY_TARGETS = ("egc", "egb")
SEED = 20260805
AUX_RIDGE_ALPHA = 30.0
RESIDUAL_RIDGE_ALPHA = 50.0
RESIDUAL_WEIGHT = 0.25
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


def deterministic_feature_matrix(parent: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    descriptor, descriptor_names = reference.descriptor_matrix(parent["molecules"])
    physical, physical_names = reference.physical_matrix(parent["molecules"], parent["keys"])
    grammar = parent_builder.grammar_features(parent["molecules"])
    matrix = np.hstack([descriptor, physical, grammar]).astype(np.float64, copy=False)
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)] = np.nan
    return matrix, {
        "shape": [int(value) for value in matrix.shape],
        "blocks": {
            "rdkit_descriptor": len(descriptor_names),
            "physical": len(physical_names),
            "graph_grammar": int(grammar.shape[1]),
        },
        "uses_labels": False,
        "uses_pi1m": False,
        "uses_stored_predictions": False,
    }


def fit_auxiliary_prediction(
    target: str,
    parent: dict[str, Any],
    matrix: np.ndarray,
    prediction_indices: np.ndarray,
    forbidden_groups: set[str],
) -> np.ndarray:
    info = parent["target_info"][target]
    groups = np.asarray(info["groups"], dtype=object)
    keep = np.asarray([group not in forbidden_groups for group in groups], dtype=bool)
    if int(np.sum(keep)) < 25:
        raise RuntimeError(f"insufficient {target} auxiliary rows after group exclusion")
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=AUX_RIDGE_ALPHA),
    )
    model.fit(matrix[np.asarray(info["indices"], dtype=np.int64)[keep]], np.asarray(info["y"], dtype=np.float64)[keep])
    prediction = np.asarray(model.predict(matrix[prediction_indices]), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"non-finite {target} auxiliary prediction")
    return prediction


def optical_gap_features(egc: np.ndarray, egb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    egc = np.asarray(egc, dtype=np.float64)
    egb = np.asarray(egb, dtype=np.float64)
    safe_egc = np.clip(egc, 0.05, 20.0)
    safe_egb = np.clip(egb, 0.05, 20.0)
    mean_gap = 0.5 * (safe_egc + safe_egb)
    min_gap = np.minimum(safe_egc, safe_egb)
    max_gap = np.maximum(safe_egc, safe_egb)
    diff_gap = safe_egc - safe_egb
    ratio_gap = safe_egc / np.maximum(safe_egb, 0.05)
    gaps = [safe_egc, safe_egb, mean_gap, min_gap, max_gap]
    columns: list[np.ndarray] = [
        safe_egc,
        safe_egb,
        mean_gap,
        min_gap,
        max_gap,
        diff_gap,
        np.abs(diff_gap),
        ratio_gap,
    ]
    for gap in gaps:
        columns.extend(
            [
                np.sqrt(gap),
                np.log1p(gap),
                1.0 / gap,
                1.0 / np.sqrt(gap),
                1.0 / np.square(gap),
                np.power(gap, -0.25),
            ]
        )
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e8)] = np.nan
    return matrix, {
        "feature_family": "fixed_moss_ravindra_penn_style_gap_coordinates",
        "feature_count": int(matrix.shape[1]),
        "raw_columns": ["egc_pred", "egb_pred", "mean_gap", "min_gap", "max_gap", "diff_gap", "abs_diff_gap", "gap_ratio"],
        "transforms_per_gap": ["sqrt", "log1p", "inverse", "inverse_sqrt", "inverse_square", "inverse_quarter_power"],
        "gap_clip": [0.05, 20.0],
        "uses_external_constants": False,
    }


def fold_local_optical_features(
    parent: dict[str, Any],
    matrix: np.ndarray,
    row_indices: np.ndarray,
    row_groups: np.ndarray,
    outer_forbidden_groups: set[str],
    folds: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    row_groups = np.asarray(row_groups, dtype=object)
    features = np.full((len(row_indices), 38), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in sorted(set(int(value) for value in folds)):
        selected = np.flatnonzero(folds == fold)
        forbidden = set(outer_forbidden_groups) | set(row_groups[selected])
        egc = fit_auxiliary_prediction("egc", parent, matrix, row_indices[selected], forbidden)
        egb = fit_auxiliary_prediction("egb", parent, matrix, row_indices[selected], forbidden)
        optical, _ = optical_gap_features(egc, egb)
        features[selected] = optical
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(len(selected)),
                "forbidden_group_count": int(len(forbidden)),
                "egc_pred_mean": float(np.mean(egc)),
                "egb_pred_mean": float(np.mean(egb)),
            }
        )
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite fold-local optical features")
    return features, {"fold_reports": fold_reports}


def predict_optical_features(
    parent: dict[str, Any],
    matrix: np.ndarray,
    row_indices: np.ndarray,
    forbidden_groups: set[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    egc = fit_auxiliary_prediction("egc", parent, matrix, row_indices, forbidden_groups)
    egb = fit_auxiliary_prediction("egb", parent, matrix, row_indices, forbidden_groups)
    optical, feature_report = optical_gap_features(egc, egb)
    feature_report.update(
        {
            "rows": int(len(row_indices)),
            "forbidden_group_count": int(len(forbidden_groups)),
            "egc_pred_mean": float(np.mean(egc)),
            "egc_pred_std": float(np.std(egc)),
            "egb_pred_mean": float(np.mean(egb)),
            "egb_pred_std": float(np.std(egb)),
            "mean_gap_quantiles": [float(value) for value in np.quantile(0.5 * (np.clip(egc, 0.05, 20.0) + np.clip(egb, 0.05, 20.0)), [0.25, 0.5, 0.75])],
        }
    )
    return optical, feature_report


def fit_nc_optical_residual(
    parent: dict[str, Any],
    matrix: np.ndarray,
    progress: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    info = dict(parent["target_info"][ACTIVE_TARGET])
    y = np.asarray(info["y"], dtype=np.float64)
    base = np.asarray(info["parent"], dtype=np.float64)
    groups = np.asarray(info["groups"], dtype=object)
    indices = np.asarray(info["indices"], dtype=np.int64)
    folds = carrier.grouped_folds(groups)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    optical_all = np.full((len(y), 38), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    fold_feature_reports: list[dict[str, Any]] = []

    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        outer_forbidden = set(groups[validation])
        inner_folds = carrier.grouped_folds(groups[training])
        train_optical, train_feature_report = fold_local_optical_features(
            parent,
            matrix,
            indices[training],
            groups[training],
            outer_forbidden,
            inner_folds,
        )
        validation_optical, validation_feature_report = predict_optical_features(
            parent,
            matrix,
            indices[validation],
            outer_forbidden,
        )
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=RESIDUAL_RIDGE_ALPHA),
        )
        model.fit(train_optical, y[training] - base[training])
        correction = np.asarray(model.predict(validation_optical), dtype=np.float64)
        candidate[validation] = reference.clip_prediction(y[training], base[validation] + RESIDUAL_WEIGHT * correction)
        optical_all[validation] = validation_optical
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "parent_r2": float(r2_score(y[validation], base[validation])),
                "candidate_r2": float(r2_score(y[validation], candidate[validation])),
                "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], base[validation])),
                "correction_mean": float(np.mean(correction)),
                "correction_std": float(np.std(correction)),
                "outer_forbidden_group_count": int(len(outer_forbidden)),
            }
        )
        fold_feature_reports.append(
            {
                "fold": int(fold),
                "train": train_feature_report,
                "validation": validation_feature_report,
            }
        )
        checkpoint(progress, "nc_outer_fold_complete", fold=int(fold), delta_r2=fold_rows[-1]["delta_r2"])

    if not np.isfinite(candidate).all() or not np.isfinite(optical_all).all():
        raise RuntimeError("C210 non-finite OOF candidate or optical features")

    active_report = carrier.evaluate_target({**info, "fingerprints": parent["fingerprints"]}, {"candidate": candidate})
    active_report["folds"] = fold_rows
    gap_mean = optical_all[:, 2]
    gap_panels = {
        "gap_low_q25": gap_mean <= np.quantile(gap_mean, 0.25),
        "gap_mid_q25_q75": (gap_mean > np.quantile(gap_mean, 0.25)) & (gap_mean < np.quantile(gap_mean, 0.75)),
        "gap_high_q75": gap_mean >= np.quantile(gap_mean, 0.75),
        "gap_diff_abs_high_q75": np.abs(optical_all[:, 6]) >= np.quantile(np.abs(optical_all[:, 6]), 0.75),
    }
    gap_panel_reports: dict[str, Any] = {}
    gap_panel_values: list[float] = []
    for name, selected in gap_panels.items():
        delta = carrier.panel_delta(y, base, candidate, selected)
        gap_panel_reports[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable",
        }
        if delta is not None:
            gap_panel_values.append(float(delta))
    existing_panel_values = [
        float(item["delta_r2"])
        for item in active_report["panels"].values()
        if item.get("delta_r2") is not None
    ]
    minimum_panel = min(existing_panel_values + gap_panel_values) if existing_panel_values or gap_panel_values else 0.0
    active_report.update(
        {
            "active": True,
            "changed_factor": "fold-nested predicted Egc/Egb optical-dispersion coordinates with fixed Ridge residual",
            "auxiliary_targets": list(AUXILIARY_TARGETS),
            "residual_weight": RESIDUAL_WEIGHT,
            "aux_ridge_alpha": AUX_RIDGE_ALPHA,
            "residual_ridge_alpha": RESIDUAL_RIDGE_ALPHA,
            "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
            "gap_regime_panels": gap_panel_reports,
            "minimum_panel_delta": float(minimum_panel),
            "pass": bool(
                active_report["delta_r2"] >= MIN_BANKABLE_DELTA_R2
                and active_report["positive_folds"] >= 4
                and active_report["group_bootstrap_lower"] > 0.0
                and minimum_panel >= 0.0
            ),
            "feature_reports": fold_feature_reports,
        }
    )

    full_folds = carrier.grouped_folds(groups)
    full_optical, full_train_feature_report = fold_local_optical_features(
        parent,
        matrix,
        indices,
        groups,
        set(),
        full_folds,
    )
    full_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=RESIDUAL_RIDGE_ALPHA),
    )
    full_model.fit(full_optical, y - base)
    test_rows, test_indices, test_parent = target_test_rows(parent, ACTIVE_TARGET)
    test_optical, test_feature_report = predict_optical_features(parent, matrix, test_indices, set())
    test_correction = np.asarray(full_model.predict(test_optical), dtype=np.float64)
    test_candidate = reference.clip_prediction(y, test_parent + RESIDUAL_WEIGHT * test_correction)
    if len(test_candidate) != len(test_rows) or not np.isfinite(test_candidate).all():
        raise RuntimeError("C210 Nc test candidate contract failed")
    diagnostics = {
        "oof_optical_feature_shape": [int(value) for value in optical_all.shape],
        "full_train_feature_report": full_train_feature_report,
        "test_feature_report": test_feature_report,
        "test_correction_mean": float(np.mean(test_correction)),
        "test_correction_std": float(np.std(test_correction)),
    }
    return active_report, candidate, test_candidate, diagnostics


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

    data_dir = (root / args.data_dir).resolve()
    canonical_run = (root / args.canonical_run).resolve()
    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, canonical_run)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)

    matrix, matrix_report = deterministic_feature_matrix(parent)
    checkpoint(progress, "deterministic_features_complete", shape=matrix_report["shape"])
    active_report, nc_candidate, nc_test_candidate, component_diagnostics = fit_nc_optical_residual(parent, matrix, progress)
    checkpoint(
        progress,
        "nc_optical_residual_complete",
        delta_r2=active_report["delta_r2"],
        positive_folds=active_report["positive_folds"],
        bootstrap_lower=active_report["group_bootstrap_lower"],
        minimum_panel_delta=active_report["minimum_panel_delta"],
        pass_gate=active_report["pass"],
    )

    banked = [ACTIVE_TARGET] if active_report["pass"] else []
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = parent["target_info"][target]
        if target == ACTIVE_TARGET:
            report = active_report
            candidate = nc_candidate
        else:
            report = unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=np.float64)
        target_reports[target] = report
        assembled = candidate if target in banked else np.asarray(target_info["parent"], dtype=np.float64)
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": candidate,
                    "assembled": assembled,
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "outer_fold": carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object)),
                }
            )
        )

    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    maximum_target_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_candidate_gate_pass = bool(mean_candidate >= mean_parent + MIN_FULL_MEAN_GAIN and maximum_target_loss >= -0.003 and bool(banked))

    parent_detail = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    raw_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        frame = parent["test"].loc[parent["test"]["target_type"].astype(str).eq(target)].sort_values("id").reset_index(drop=True)
        if target == ACTIVE_TARGET and ACTIVE_TARGET in banked:
            values = nc_test_candidate
        else:
            values = (
                parent_detail.loc[parent_detail["target_type"].astype(str).eq(target)]
                .sort_values("id")["target"]
                .to_numpy(np.float64)
            )
        raw_parts.append(pd.DataFrame({"id": frame["id"].astype(int), "target_type": target, "model_prediction": values}))
    raw = pd.concat(raw_parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    raw_labels, _ = reference.build_label_pool(parent["train"], parent["archive"])
    detail, override_report = reference.apply_official_overrides(raw, parent["test"], raw_labels)
    predictions = detail[["id", "target"]].copy()
    expected_ids = parent["test"]["id"].astype(int).to_numpy()
    if (
        len(predictions) != 4940
        or not np.array_equal(predictions["id"].astype(int).to_numpy(), expected_ids)
        or predictions["id"].duplicated().any()
        or not np.isfinite(predictions["target"].to_numpy(np.float64)).all()
    ):
        raise RuntimeError("C210 complete output contract failed")

    report = {
        "schema_version": "ppp.round2.c210.nc-optical-dispersion-gap.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "parent": "C050 source rebuild; no stored C093/C195/C197 prediction input",
        "hypothesis": "Nested structure-only Egc/Egb predictions converted to fixed optical-dispersion gap coordinates add stable Nc residual signal beyond exact C050.",
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "pi1m_used": False,
        "stored_prediction_replay": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "active_targets": [ACTIVE_TARGET],
        "feature_report": {
            "deterministic_structure_matrix": matrix_report,
            "optical_coordinates": component_diagnostics,
        },
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": maximum_target_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(mean_candidate >= 0.95 and full_candidate_gate_pass),
        "official_override_report": override_report,
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
        },
        "decision": "candidate_pass_pending_clean_reproduction" if full_candidate_gate_pass else "rejected_component_or_full_gate",
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    oof.loc[oof["target_type"].astype(str).eq(ACTIVE_TARGET), ["canonical", "target", "parent", "candidate", "assembled"]].to_csv(
        run_dir / "nc_oof_predictions.csv",
        index=False,
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": "ppp.round2.c210.nc-optical-dispersion-gap.v1",
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "auxiliary_targets": list(AUXILIARY_TARGETS),
            "auxiliary_model": f"fold-nested Ridge(alpha={AUX_RIDGE_ALPHA}) on official structure descriptors only",
            "residual_model": f"Ridge(alpha={RESIDUAL_RIDGE_ALPHA}) on fixed optical gap transforms",
            "residual_weight": RESIDUAL_WEIGHT,
            "selection": "no tuning; bank only if standard component gate plus gap-regime panels pass",
            "official_only": True,
            "local_eval_read": False,
            "kaggle_submission": False,
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
        f"# {run_dir.name}\n\nDecision: **{report['decision']}**. "
        f"Nc delta `{active_report['delta_r2']:+.12f}`; banked targets: `{','.join(banked) or 'none'}`. "
        f"Mean parent `{mean_parent:.12f}`; assembled `{mean_candidate:.12f}`; "
        f"gain `{mean_candidate - mean_parent:+.12f}`. No local_eval/Kaggle/upload/submission action.\n",
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
                "decision": report["decision"],
                "banked_targets": banked,
                "nc_delta_r2": active_report["delta_r2"],
                "mean_parent_r2": mean_parent,
                "mean_candidate_r2": mean_candidate,
                "mean_gain": mean_candidate - mean_parent,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
