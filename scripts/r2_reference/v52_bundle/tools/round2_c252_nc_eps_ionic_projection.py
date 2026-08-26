#!/usr/bin/env python3
"""C252: Nc projection from regenerated selected EPS and fold-nested ionic coordinate.

This child is the fresh post-C249 queue continuation recommended by the
read-only sidecar.  It targets only unbanked Nc.

The single changed factor is a fold-local projection:

  1. Regenerate the selected C214 EPS ionic component from official inputs.
  2. Fit log(EPS - Nc^2) only on official EPS/Nc paired structures that are
     outside the active Nc validation fold's canonical groups.
  3. For Nc rows with an EPS counterpart, derive Nc from the selected EPS
     prediction and the fold-local ionic prediction; otherwise fall back to C050.

No stored C214 predictions, local_eval/test external_labels, public feedback, PI1M,
pretrained assets, external data, Kaggle compute, upload, submission, or final
notebook artifact are used.
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
from rdkit import rdBase
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c187_ionic_eps_only as c187
import round2_c208_tg_robust_group_measurement as c208
import round2_c238_eps_bond_polarity_orientational_residual as c238


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "nc"
SCHEMA = "ppp.round2.c252.nc-eps-ionic-projection.v1"
SEED = 20260805
SELECTED_EPS_RUN_ID = "R2-C214-20260805-0440-eps-ionic-full-amplitude-v1"
PROJECTION_WEIGHT = 0.50
MIN_BANKABLE_DELTA_R2 = 0.010
NC_FLOOR = 0.05


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


def selected_eps_maps(
    root: Path,
    parent: dict[str, Any],
    dense: np.ndarray,
    sparse_matrix: Any,
) -> tuple[np.ndarray, dict[str, float], dict[str, float], dict[str, Any]]:
    selected_eps_oof, selected_eps_test, report = c238.regenerate_c214_selected_parent(
        root,
        parent,
        dense,
        sparse_matrix,
    )
    eps_info = dict(parent["target_info"]["eps"])
    selected_eps_by_canon = {
        str(canon): float(value) for canon, value in zip(eps_info["canonical"], selected_eps_oof, strict=True)
    }
    eps_test_rows, _, _ = c208.target_test_rows(parent, "eps")
    selected_eps_test_by_canon = {
        str(canon): float(value) for canon, value in zip(eps_test_rows["canonical"], selected_eps_test, strict=True)
    }
    return selected_eps_oof, selected_eps_by_canon, selected_eps_test_by_canon, report


def fit_ionic_models(x_train: np.ndarray, y_log: np.ndarray, fold: int) -> list[Any]:
    models: list[Any] = []
    for kind in c187.MODEL_KINDS:
        model = c187.make_model(kind, fold)
        model.fit(x_train, y_log)
        models.append(model)
    return models


def predict_ionic(models: list[Any], x_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    by_model = [np.exp(np.clip(model.predict(x_pred), -8, 4)) for model in models]
    stacked = np.column_stack(by_model)
    return np.mean(stacked, axis=1), stacked


def safe_nc_from_eps_ionic(selected_eps: np.ndarray, ionic: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(selected_eps, dtype=np.float64) - np.asarray(ionic, dtype=np.float64), NC_FLOOR**2))


def nearest_similarity(parent: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    nearest = np.zeros(len(indices), dtype=np.float64)
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [parent["fingerprints"][int(indices[row])] for row in training]
        for row in validation:
            sims = reference.DataStructs.BulkTanimotoSimilarity(parent["fingerprints"][int(indices[row])], train_fps)
            nearest[row] = float(max(sims)) if sims else 0.0
    return nearest


def evaluate_nc(
    parent: dict[str, Any],
    info: dict[str, Any],
    candidate: np.ndarray,
    pair_available: np.ndarray,
    projection_raw: np.ndarray,
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    groups = np.asarray(info["groups"], dtype=object)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    nearest = nearest_similarity(parent, info)
    panel_specs: dict[str, np.ndarray] = {
        "eps_counterpart_available_projection_rows": pair_available,
        "eps_counterpart_missing_c050_fallback_rows": ~pair_available,
        "projection_raw_above_parent": pair_available & (projection_raw > parent_oof),
        "projection_raw_below_parent": pair_available & (projection_raw < parent_oof),
        "similarity_lt_0.30": nearest < 0.30,
        "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
        "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
        "similarity_ge_0.70": nearest >= 0.70,
        "nc_low_quartile": y <= np.quantile(y, 0.25),
        "nc_high_quartile": y >= np.quantile(y, 0.75),
    }
    for name in sorted(set(scaffolds.astype(str))):
        selected = scaffolds.astype(str) == name
        if int(np.sum(selected)) >= 10:
            panel_specs[f"scaffold_{name}"] = selected
    panels: dict[str, Any] = {}
    panel_values: list[float] = []
    for name, selected in panel_specs.items():
        delta = carrier.panel_delta(y, parent_oof, candidate, selected)
        panels[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable_insufficient_support",
        }
        if delta is not None:
            panel_values.append(float(delta))
    delta = float(r2_score(y, candidate) - r2_score(y, parent_oof))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    lower = float(carrier.bootstrap_lower(y, parent_oof, candidate, groups))
    minimum_panel = float(min(panel_values)) if panel_values else 0.0
    passed = bool(delta >= MIN_BANKABLE_DELTA_R2 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0)
    return {
        "active": True,
        "parent_r2": float(r2_score(y, parent_oof)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_panel_delta": minimum_panel,
        "panels": panels,
        "folds": fold_rows,
        "pass": passed,
        "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
        "projection_weight": PROJECTION_WEIGHT,
        "pair_available_rows": int(np.sum(pair_available)),
        "fallback_rows": int(np.sum(~pair_available)),
    }


def fit_nc_projection(
    parent: dict[str, Any],
    dense: np.ndarray,
    sparse_matrix: Any,
    selected_eps_by_canon: dict[str, float],
    selected_eps_test_by_canon: dict[str, float],
) -> dict[str, Any]:
    eps_info = dict(parent["target_info"]["eps"])
    nc_info = dict(parent["target_info"][ACTIVE_TARGET])
    eps_y_by_canon = {str(canon): float(value) for canon, value in zip(eps_info["canonical"], eps_info["y"], strict=True)}
    nc_y = np.asarray(nc_info["y"], dtype=np.float64)
    nc_parent = np.asarray(nc_info["parent"], dtype=np.float64)
    nc_canons = np.asarray(nc_info["canonical"], dtype=object)
    nc_groups = np.asarray(nc_info["groups"], dtype=object)
    key_to_index = parent["key_to_index"]
    pair_available = np.asarray([str(canon) in selected_eps_by_canon for canon in nc_canons], dtype=bool)
    pair_positions = np.flatnonzero(pair_available)
    if len(pair_positions) < 50:
        raise RuntimeError("insufficient official EPS/Nc pair rows for C252")
    pair_canons = np.asarray([str(nc_canons[pos]) for pos in pair_positions], dtype=object)
    pair_indices = np.asarray([key_to_index[str(canon)] for canon in pair_canons], dtype=np.int64)
    pair_eps_y = np.asarray([eps_y_by_canon[str(canon)] for canon in pair_canons], dtype=np.float64)
    pair_nc_y = nc_y[pair_positions]
    ionic_y = pair_eps_y - pair_nc_y**2
    if np.any(ionic_y <= 0.0):
        raise RuntimeError("non-positive official EPS-Nc^2 ionic coordinate in C252 pair rows")
    log_ionic = np.log(ionic_y)

    folds = carrier.grouped_folds(nc_groups)
    candidate = nc_parent.copy()
    projection_raw = np.full(len(nc_y), np.nan, dtype=np.float64)
    ionic_oof = np.full(len(nc_y), np.nan, dtype=np.float64)
    direct_models_oof = np.full((len(nc_y), len(c187.MODEL_KINDS)), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    pair_position_lookup = {int(pos): index for index, pos in enumerate(pair_positions)}
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        validation_groups = {str(value) for value in nc_groups[validation]}
        validation_pair_positions = [int(pos) for pos in validation if int(pos) in pair_position_lookup]
        training_pair_mask = np.asarray(
            [str(nc_groups[int(pos)]) not in validation_groups for pos in pair_positions],
            dtype=bool,
        )
        training_pair_indices = np.flatnonzero(training_pair_mask)
        validation_pair_indices = np.asarray([pair_position_lookup[pos] for pos in validation_pair_positions], dtype=np.int64)
        if len(validation_pair_indices) and len(training_pair_indices) < 20:
            raise RuntimeError(f"C252 fold {fold} has insufficient train pair rows")
        if len(validation_pair_indices):
            x_train, x_validation = c187.fold_matrix(
                dense,
                sparse_matrix,
                pair_indices[training_pair_indices],
                pair_indices[validation_pair_indices],
            )
            models = fit_ionic_models(x_train, log_ionic[training_pair_indices], fold)
            ionic_pred, model_pred = predict_ionic(models, x_validation)
            selected_eps = np.asarray([selected_eps_by_canon[str(pair_canons[index])] for index in validation_pair_indices], dtype=np.float64)
            raw = safe_nc_from_eps_ionic(selected_eps, ionic_pred)
            full_positions = np.asarray(validation_pair_positions, dtype=np.int64)
            projection_raw[full_positions] = raw
            ionic_oof[full_positions] = ionic_pred
            direct_models_oof[full_positions, :] = model_pred
            candidate[full_positions] = (1.0 - PROJECTION_WEIGHT) * nc_parent[full_positions] + PROJECTION_WEIGHT * raw
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "projection_rows": int(len(validation_pair_indices)),
                "train_pair_rows": int(len(training_pair_indices)),
                "parent_r2": float(r2_score(nc_y[validation], nc_parent[validation])),
                "candidate_r2": float(r2_score(nc_y[validation], candidate[validation])),
                "delta_r2": float(r2_score(nc_y[validation], candidate[validation]) - r2_score(nc_y[validation], nc_parent[validation])),
            }
        )
    if not np.isfinite(candidate).all():
        raise RuntimeError("C252 produced non-finite NC OOF candidate")

    test_rows, test_indices, test_parent = c208.target_test_rows(parent, ACTIVE_TARGET)
    test_candidate = np.asarray(test_parent, dtype=np.float64).copy()
    test_pair_mask = np.asarray([str(canon) in selected_eps_test_by_canon for canon in test_rows["canonical"]], dtype=bool)
    if np.any(test_pair_mask):
        x_train, x_test = c187.fold_matrix(dense, sparse_matrix, pair_indices, test_indices[test_pair_mask])
        full_models = fit_ionic_models(x_train, log_ionic, SEED)
        ionic_test, model_test = predict_ionic(full_models, x_test)
        selected_eps_test = np.asarray(
            [selected_eps_test_by_canon[str(canon)] for canon in test_rows.loc[test_pair_mask, "canonical"]],
            dtype=np.float64,
        )
        raw_test = safe_nc_from_eps_ionic(selected_eps_test, ionic_test)
        test_candidate[test_pair_mask] = (1.0 - PROJECTION_WEIGHT) * test_parent[test_pair_mask] + PROJECTION_WEIGHT * raw_test
    else:
        model_test = np.empty((0, len(c187.MODEL_KINDS)), dtype=np.float64)
        ionic_test = np.empty(0, dtype=np.float64)
        raw_test = np.empty(0, dtype=np.float64)
    return {
        "candidate": candidate,
        "test_candidate": test_candidate,
        "projection_raw": projection_raw,
        "ionic_oof": ionic_oof,
        "direct_models_oof": direct_models_oof,
        "folds": fold_rows,
        "pair_available": pair_available,
        "pair_rows": int(len(pair_positions)),
        "test_pair_rows": int(np.sum(test_pair_mask)),
        "projection_weight": PROJECTION_WEIGHT,
        "model_kinds": list(c187.MODEL_KINDS),
        "full_test_ionic_summary": {
            "rows": int(len(ionic_test)),
            "raw_nc_min": float(np.min(raw_test)) if len(raw_test) else None,
            "raw_nc_max": float(np.max(raw_test)) if len(raw_test) else None,
            "ionic_min": float(np.min(ionic_test)) if len(ionic_test) else None,
            "ionic_max": float(np.max(ionic_test)) if len(ionic_test) else None,
            "model_columns": int(model_test.shape[1]) if len(model_test) else len(c187.MODEL_KINDS),
        },
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

    dense, sparse_matrix, feature_report = c187.rich_builder.build_features(root, parent["keys"])
    dense = np.asarray(dense, dtype=np.float64)
    sparse_matrix = sparse_matrix.astype(np.float64)
    selected_eps_oof, selected_eps_by_canon, selected_eps_test_by_canon, selected_eps_report = selected_eps_maps(
        root,
        parent,
        dense,
        sparse_matrix,
    )
    checkpoint(
        progress,
        "selected_eps_regenerated",
        selected_eps_r2=selected_eps_report["selected_parent_r2"],
        expected_c214_abs_error=selected_eps_report["expected_c214_abs_error"],
        pair_rows=selected_eps_report["pair_rows"],
    )

    result = fit_nc_projection(parent, dense, sparse_matrix, selected_eps_by_canon, selected_eps_test_by_canon)
    nc_info = dict(parent["target_info"][ACTIVE_TARGET])
    active_report = evaluate_nc(
        parent,
        nc_info,
        np.asarray(result["candidate"], dtype=np.float64),
        np.asarray(result["pair_available"], dtype=bool),
        np.asarray(result["projection_raw"], dtype=np.float64),
        list(result["folds"]),
    )
    active_report.update(
        {
            "changed_factor": "NC projection from regenerated selected C214 EPS plus fold-nested official EPS-Nc ionic coordinate",
            "selected_eps_run_id": SELECTED_EPS_RUN_ID,
            "selected_eps_regenerated_from_source": True,
            "stored_c214_prediction_files_read": False,
            "uses_cross_target_labels": True,
            "cross_target_label_use": "official EPS labels are used only to fit fold-nested ionic=EPS-Nc^2 on training pairs and selected EPS predictions are generated from official inputs",
            "uses_pi1m": False,
            "uses_stored_prediction_replay": False,
            "uses_external_data": False,
            "test_pair_rows": int(result["test_pair_rows"]),
        }
    )
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "nc_ionic_projection_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        pass_gate=banked,
        positive_folds=active_report["positive_folds"],
        minimum_panel_delta=active_report["minimum_panel_delta"],
        group_bootstrap_lower=active_report["group_bootstrap_lower"],
        pair_rows=result["pair_rows"],
        test_pair_rows=result["test_pair_rows"],
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = active_report
            candidate = np.asarray(result["candidate"], dtype=np.float64)
            assembled = candidate if banked else np.asarray(info["parent"], dtype=np.float64)
            projection_raw = np.asarray(result["projection_raw"], dtype=np.float64)
            ionic_oof = np.asarray(result["ionic_oof"], dtype=np.float64)
            pair_available = np.asarray(result["pair_available"], dtype=bool)
        else:
            report = unchanged_report(info)
            candidate = np.asarray(info["parent"], dtype=np.float64)
            assembled = candidate
            projection_raw = np.full(len(candidate), np.nan, dtype=np.float64)
            ionic_oof = np.full(len(candidate), np.nan, dtype=np.float64)
            pair_available = np.zeros(len(candidate), dtype=bool)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(info["groups"], dtype=object))
        selected_eps_column = np.asarray(
            [selected_eps_by_canon.get(str(canon), np.nan) for canon in info["canonical"]],
            dtype=np.float64,
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": info["canonical"],
                    "target_type": target,
                    "target": info["y"],
                    "parent": info["parent"],
                    "selected_eps": selected_eps_column,
                    "projection_raw": projection_raw,
                    "ionic_oof": ionic_oof,
                    "candidate": candidate,
                    "assembled": assembled,
                    "group": info["groups"],
                    "scaffold": info["scaffolds"],
                    "fold": folds,
                    "pair_available": pair_available,
                    "banked": bool(target == ACTIVE_TARGET and banked),
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    maximum_target_loss = float(min(target_reports[target]["delta_r2"] if target == ACTIVE_TARGET and banked else 0.0 for target in TARGETS))

    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    nc_test_rows, _, nc_test_parent = c208.target_test_rows(parent, ACTIVE_TARGET)
    component_test = pd.DataFrame(
        {
            "id": nc_test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "selected_eps_available": [str(canon) in selected_eps_test_by_canon for canon in nc_test_rows["canonical"]],
            "parent": nc_test_parent,
            "candidate": result["test_candidate"],
        }
    )
    predictions = parent_test.merge(component_test[["id", "target_type", "candidate"]], on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(
        (predictions["target_type"] == ACTIVE_TARGET) & banked,
        predictions["candidate"],
        predictions["target"],
    )
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940:
        raise RuntimeError("C252 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C252 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C252 prediction finite check failed")

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
        "stored_c214_prediction_files_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "active_target": ACTIVE_TARGET,
        "parent_replay_parity": parity,
        "selected_eps_report": selected_eps_report,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "normal_component_gate_pass": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": maximum_target_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "prediction_exact_order": True,
        "prediction_unique_ids": True,
        "prediction_finite_targets": True,
        "full_candidate_gate_pass": bool(banked),
        "goal_0_95_met": bool(banked and assembled_mean >= 0.95),
        "component_diagnostics": {
            "active_target": ACTIVE_TARGET,
            "selected_eps_run_id": SELECTED_EPS_RUN_ID,
            "pair_rows": int(result["pair_rows"]),
            "test_pair_rows": int(result["test_pair_rows"]),
            "projection_weight": PROJECTION_WEIGHT,
            "nc_delta_r2": active_report["delta_r2"],
            "nc_candidate_r2": active_report["candidate_r2"],
            "nc_parent_r2": active_report["parent_r2"],
            "nc_positive_folds": active_report["positive_folds"],
            "nc_group_bootstrap_lower": active_report["group_bootstrap_lower"],
            "nc_minimum_panel_delta": active_report["minimum_panel_delta"],
            "uses_cross_target_labels": True,
            "uses_external_data": False,
            "uses_pi1m": False,
            "uses_stored_prediction_replay": False,
        },
        "decision": "banked_component_pending_compound_audit" if banked else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "c238_selected_eps_regenerator": sha256_file(round2_root / "tools/round2_c238_eps_bond_polarity_orientational_residual.py"),
            "c187_ionic_helper": sha256_file(round2_root / "tools/round2_c187_ionic_eps_only.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c208_alignment_helper": sha256_file(round2_root / "tools/round2_c208_tg_robust_group_measurement.py"),
            "rich_builder": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    oof.loc[oof["target_type"].astype(str).eq(ACTIVE_TARGET)].to_csv(run_dir / "nc_oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component_test.to_csv(run_dir / "nc_component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "active_target": ACTIVE_TARGET,
            "selected_eps_run_id": SELECTED_EPS_RUN_ID,
            "selected_eps_regenerated_from_source": True,
            "model": "fold-local log(EPS-Nc^2) Ridge/ExtraTrees/HistGradientBoosting ensemble",
            "projection": "Nc = 0.5*C050_Nc + 0.5*sqrt(max(selected_EPS - ionic, 0.05^2)) on EPS/Nc paired rows; C050 fallback elsewhere",
            "projection_weight": PROJECTION_WEIGHT,
            "model_kinds": list(c187.MODEL_KINDS),
            "component_gate": {
                "minimum_delta_r2": MIN_BANKABLE_DELTA_R2,
                "minimum_positive_folds": 4,
                "bootstrap_lower_must_exceed": 0.0,
                "minimum_panel_delta_must_be_at_least": 0.0,
            },
            "official_only": True,
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
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
                f"rdkit={rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: **{report['decision']}**. Nc parent R² "
        f"`{active_report['parent_r2']:.12f}`, candidate R² "
        f"`{active_report['candidate_r2']:.12f}`, delta "
        f"`{active_report['delta_r2']:+.12f}`. Selected EPS was regenerated from "
        "official inputs; no stored C214 predictions, local_eval, Kaggle compute, "
        "upload, submission, or final-notebook action.\n",
        encoding="utf-8",
    )
    manifest_paths = [path for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]
    lines = [f"{sha256_file(path)}  {path.name}" for path in manifest_paths]
    lines.extend(f"{digest}  SOURCE {name}" for name, digest in sorted(report["source_hashes"].items()))
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "decision": report["decision"],
                "nc_parent_r2": active_report["parent_r2"],
                "nc_candidate_r2": active_report["candidate_r2"],
                "nc_delta_r2": active_report["delta_r2"],
                "banked_targets": report["banked_targets"],
                "mean_candidate_r2": report["mean_candidate_r2"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
