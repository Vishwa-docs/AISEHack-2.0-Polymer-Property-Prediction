#!/usr/bin/env python3
"""C256: Egb current-domain residual.

This child tests one bounded Egb hypothesis after C255:

  C050's Egb parent may be diluted by a current/archive source-domain mismatch.
  A single low-variance residual head trained only on official current Round 2
  Egb labels may correct current-domain rows without changing the C050 archive
  replay or using source as a model covariate.

The residual head regenerates C050 and all features from official Round 2 inputs.
It does not read prior prediction artifacts, local_eval/test external_labels, public feedback,
PI1M, pretrained weights, or any Kaggle resource.
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
from rdkit import DataStructs
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as rich_builder
import round2_c201_safe_egb_cross_property_stage2 as c201


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "egb"
SCHEMA = "ppp.round2.c256.egb-current-domain-residual.v1"
SEED = 20260805
MIN_BANKABLE_DELTA_R2 = 0.01
RIDGE_ALPHA = 160.0
RESIDUAL_WEIGHT = 0.35
MIN_CURRENT_TRAIN_ROWS = 32


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


def current_egb_label_map(parent: dict[str, Any]) -> dict[str, float]:
    current = parent["train"].loc[parent["train"]["target_type"].astype(str).eq(ACTIVE_TARGET)].copy()
    if current.empty:
        raise RuntimeError("C256 found no current Round 2 Egb labels")
    grouped = current.groupby("canonical", dropna=False)["target"].median()
    return {str(canonical): float(value) for canonical, value in grouped.items()}


def source_vectors(parent: dict[str, Any], canonicals: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mapping = current_egb_label_map(parent)
    current_y = np.full(len(canonicals), np.nan, dtype=np.float64)
    current_source = np.zeros(len(canonicals), dtype=bool)
    for row, canonical in enumerate(canonicals):
        key = str(canonical)
        if key in mapping:
            current_source[row] = True
            current_y[row] = mapping[key]
    return current_source, current_y, mapping


def zero_extra(rows: int) -> np.ndarray:
    return np.zeros((int(rows), 0), dtype=np.float64)


def fit_residual_model(train_x: Any, residual: np.ndarray) -> Ridge:
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
    model.fit(train_x, residual)
    return model


def fold_local_nearest(parent: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    nearest = np.zeros(len(indices), dtype=np.float64)
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [parent["fingerprints"][int(indices[row])] for row in training]
        for row in validation:
            sims = DataStructs.BulkTanimotoSimilarity(parent["fingerprints"][int(indices[row])], train_fps)
            nearest[row] = float(max(sims)) if sims else 0.0
    return nearest


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, selected: np.ndarray, minimum: int = 8) -> float | None:
    selected = np.asarray(selected, dtype=bool)
    if int(np.sum(selected)) < minimum or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))


def transfer_panels(
    parent: dict[str, Any],
    info: dict[str, Any],
    candidate: np.ndarray,
    residual_prediction: np.ndarray,
    current_source: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    groups = np.asarray(info["groups"], dtype=object)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    nearest = fold_local_nearest(parent, info)
    duplicated_groups = pd.Series(groups.astype(str)).duplicated(keep=False).to_numpy(bool)
    residual_positive = residual_prediction > 0.0
    panel_specs: dict[str, np.ndarray] = {
        "source_current_train": current_source,
        "source_archive_only": ~current_source,
        "duplicate_or_conflict_group": duplicated_groups,
        "nonduplicate_group": ~duplicated_groups,
        "similarity_lt_0.30": nearest < 0.30,
        "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
        "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
        "similarity_ge_0.70": nearest >= 0.70,
        "target_low_quartile": y <= np.quantile(y, 0.25),
        "target_high_quartile": y >= np.quantile(y, 0.75),
        "parent_low_quartile": parent_oof <= np.quantile(parent_oof, 0.25),
        "parent_high_quartile": parent_oof >= np.quantile(parent_oof, 0.75),
        "residual_prediction_positive": residual_positive,
        "residual_prediction_nonpositive": ~residual_positive,
    }
    for name in sorted(set(scaffolds.astype(str))):
        selected = scaffolds.astype(str) == name
        if int(np.sum(selected)) >= 10:
            panel_specs[f"scaffold_{name}"] = selected
    panels: dict[str, Any] = {}
    values: list[float] = []
    for name, selected in panel_specs.items():
        delta = panel_delta(y, parent_oof, candidate, selected)
        panels[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable_insufficient_support",
        }
        if delta is not None:
            values.append(delta)
    return {
        "panels": panels,
        "minimum_panel_delta": float(min(values)) if values else 0.0,
        "nearest_similarity_summary": {
            "min": float(np.min(nearest)),
            "p25": float(np.quantile(nearest, 0.25)),
            "median": float(np.median(nearest)),
            "p75": float(np.quantile(nearest, 0.75)),
            "max": float(np.max(nearest)),
        },
    }


def fit_active_target(
    parent: dict[str, Any],
    dense: np.ndarray,
    sparse_matrix: Any,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    info = parent["target_info"][ACTIVE_TARGET]
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    key_indices = np.asarray(info["indices"], dtype=np.int64)
    canonicals = np.asarray(info["canonical"], dtype=object)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    current_source, current_y, current_map = source_vectors(parent, canonicals)
    if int(np.sum(current_source)) < MIN_CURRENT_TRAIN_ROWS:
        raise RuntimeError(f"C256 has only {int(np.sum(current_source))} current-source Egb rows")
    candidate = parent_oof.copy()
    residual_prediction = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []

    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero((folds != fold) & current_source)
        if len(training) < MIN_CURRENT_TRAIN_ROWS:
            raise RuntimeError(f"C256 fold {fold} has insufficient current-source training rows")
        if set(groups[training].astype(str)) & set(groups[validation].astype(str)):
            raise RuntimeError(f"C256 fold {fold} leaked a validation group into residual training")
        train_x, valid_x = c201.combine_dense_sparse(
            dense,
            sparse_matrix,
            key_indices[training],
            key_indices[validation],
            zero_extra(len(training)),
            zero_extra(len(validation)),
        )
        residual = current_y[training] - parent_oof[training]
        if not np.isfinite(residual).all():
            raise RuntimeError(f"C256 fold {fold} residual target is non-finite")
        model = fit_residual_model(train_x, residual)
        pred_residual = np.asarray(model.predict(valid_x), dtype=np.float64)
        if not np.isfinite(pred_residual).all():
            raise RuntimeError(f"C256 fold {fold} residual prediction is non-finite")
        residual_prediction[validation] = pred_residual
        candidate[validation] = reference.clip_prediction(current_y[training], parent_oof[validation] + RESIDUAL_WEIGHT * pred_residual)
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "current_source_validation_rows": int(np.sum(current_source[validation])),
                "archive_only_validation_rows": int(np.sum(~current_source[validation])),
                "current_source_training_rows": int(len(training)),
                "parent_r2": float(r2_score(y[validation], parent_oof[validation])),
                "candidate_r2": float(r2_score(y[validation], candidate[validation])),
                "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent_oof[validation])),
            }
        )
    if not np.isfinite(candidate).all() or not np.isfinite(residual_prediction).all():
        raise RuntimeError("C256 produced non-finite Egb OOF predictions")

    delta = float(r2_score(y, candidate) - r2_score(y, parent_oof))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    lower = float(carrier.bootstrap_lower(y, parent_oof, candidate, groups))
    panel_report = transfer_panels(parent, info, candidate, residual_prediction, current_source)
    passed = bool(
        delta >= MIN_BANKABLE_DELTA_R2
        and positive >= 4
        and lower > 0.0
        and panel_report["minimum_panel_delta"] >= 0.0
    )
    report = {
        "active": True,
        "parent_r2": float(r2_score(y, parent_oof)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_panel_delta": float(panel_report["minimum_panel_delta"]),
        "panels": panel_report["panels"],
        "nearest_similarity_summary": panel_report["nearest_similarity_summary"],
        "folds": fold_rows,
        "current_source_rows": int(np.sum(current_source)),
        "archive_only_rows": int(np.sum(~current_source)),
        "current_source_unique_canonicals": int(len(current_map)),
        "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
        "pass": passed,
        "model": {
            "residual_head": "Ridge on regenerated official structure features only",
            "ridge_alpha": float(RIDGE_ALPHA),
            "residual_weight": float(RESIDUAL_WEIGHT),
            "training_rows": "current_train Egb canonical medians only; archive labels only in C050 parent replay/fallback",
            "no_source_flag_covariate": True,
            "cross_target_labels": False,
        },
    }
    return candidate, report, residual_prediction, current_source


def full_test_predictions(
    parent: dict[str, Any],
    dense: np.ndarray,
    sparse_matrix: Any,
    bank_active: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    test_rows = parent["test"].loc[parent["test"]["target_type"].astype(str).eq(ACTIVE_TARGET)].sort_values("id").reset_index(drop=True)
    test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"].astype(str).eq(ACTIVE_TARGET)].sort_values("id").reset_index(drop=True)
    if not np.array_equal(test_rows["id"].to_numpy(np.int64), test_detail["id"].to_numpy(np.int64)):
        raise RuntimeError("C256 Egb test ID alignment failed")
    test_parent = test_detail["target"].to_numpy(np.float64)
    if not bank_active:
        component = pd.DataFrame(
            {
                "id": test_rows["id"].astype(int),
                "target_type": ACTIVE_TARGET,
                "parent": test_parent,
                "residual_prediction": np.nan,
                "candidate": test_parent,
                "current_domain_training_only": True,
                "banked": False,
            }
        )
        return parent_test[["id", "target"]].sort_values("id")["target"].to_numpy(np.float64), component

    info = parent["target_info"][ACTIVE_TARGET]
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    train_indices = np.asarray(info["indices"], dtype=np.int64)
    train_canonicals = np.asarray(info["canonical"], dtype=object)
    current_source, current_y, _ = source_vectors(parent, train_canonicals)
    training = np.flatnonzero(current_source)
    if len(training) < MIN_CURRENT_TRAIN_ROWS:
        raise RuntimeError("C256 full-data residual fit has insufficient current-source rows")
    test_indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    train_x, test_x = c201.combine_dense_sparse(
        dense,
        sparse_matrix,
        train_indices[training],
        test_indices,
        zero_extra(len(training)),
        zero_extra(len(test_indices)),
    )
    residual = current_y[training] - parent_oof[training]
    model = fit_residual_model(train_x, residual)
    residual_test = np.asarray(model.predict(test_x), dtype=np.float64)
    test_candidate = reference.clip_prediction(current_y[training], test_parent + RESIDUAL_WEIGHT * residual_test)
    prediction_detail = parent_test.copy()
    mask = prediction_detail["target_type"].astype(str).eq(ACTIVE_TARGET)
    prediction_detail.loc[mask, "target"] = test_candidate
    component = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "parent": test_parent,
            "residual_prediction": residual_test,
            "candidate": test_candidate,
            "current_domain_training_only": True,
            "banked": True,
        }
    )
    return prediction_detail[["id", "target"]].sort_values("id")["target"].to_numpy(np.float64), component


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

    dense, sparse_features, feature_report = rich_builder.build_features(root, parent["keys"])
    dense = np.asarray(dense, dtype=np.float64)
    sparse_features = sparse_features.astype(np.float64).tocsr()
    checkpoint(progress, "features_complete", dense_shape=feature_report["dense_shape"], sparse_shape=feature_report["sparse_shape"])

    active_candidate, active_report, residual_oof, current_source = fit_active_target(parent, dense, sparse_features)
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "egb_current_domain_residual_complete",
        delta_r2=active_report["delta_r2"],
        positive_folds=active_report["positive_folds"],
        bootstrap_lower=active_report["group_bootstrap_lower"],
        minimum_panel_delta=active_report["minimum_panel_delta"],
        current_source_rows=active_report["current_source_rows"],
        archive_only_rows=active_report["archive_only_rows"],
        pass_gate=banked,
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = parent["target_info"][target]
        y = np.asarray(info["y"], dtype=np.float64)
        parent_oof = np.asarray(info["parent"], dtype=np.float64)
        folds = carrier.grouped_folds(np.asarray(info["groups"], dtype=object))
        if target == ACTIVE_TARGET:
            candidate = active_candidate
            report = active_report
            residual = residual_oof
            source_current = current_source
        else:
            candidate = parent_oof.copy()
            report = unchanged_report(info)
            residual = np.full(len(parent_oof), np.nan, dtype=np.float64)
            source_current = np.zeros(len(parent_oof), dtype=bool)
        assembled = candidate if target == ACTIVE_TARGET and banked else parent_oof
        target_reports[target] = report
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": info["canonical"],
                    "target_type": target,
                    "target": y,
                    "parent": parent_oof,
                    "residual_prediction": residual,
                    "candidate": candidate,
                    "assembled": assembled,
                    "current_source": source_current,
                    "group": info["groups"],
                    "scaffold": info["scaffolds"],
                    "fold": folds,
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(
        np.mean(
            [
                r2_score(part["target"].to_numpy(np.float64), part["assembled"].to_numpy(np.float64))
                for part in oof_parts
            ]
        )
    )
    max_loss = float(min(active_report["delta_r2"] if banked else 0.0, 0.0))
    full_candidate_gate_pass = bool(banked and assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003)
    prediction_values, component_test = full_test_predictions(parent, dense, sparse_features, banked)
    predictions = parent["test_parent_detail"][["id", "target_type"]].copy().sort_values("id").reset_index(drop=True)
    predictions["target"] = prediction_values
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941)):
        raise RuntimeError("C256 ID coverage/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C256 produced non-finite full predictions")

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
        "stored_prediction_replay": False,
        "cross_target_labels": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "active_target": ACTIVE_TARGET,
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
        "selection_rule": "Train one Ridge residual head only on current Round 2 Egb labels; apply fixed residual weight; exact C050 fallback unless the component gate passes.",
        "decision": "candidate_pass_pending_clean_reproduction" if full_candidate_gate_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "feature_builder": sha256_file(round2_root / "tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "combine_helper": sha256_file(round2_root / "tools/round2_c201_safe_egb_cross_property_stage2.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
        },
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    component_test.to_csv(run_dir / "egb_component_predictions.csv", index=False)
    predictions[["id", "target"]].to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": SEED,
            "active_target": ACTIVE_TARGET,
            "route": "current-domain Egb Ridge residual",
            "ridge_alpha": RIDGE_ALPHA,
            "residual_weight": RESIDUAL_WEIGHT,
            "training_source": "official current_train Egb labels only",
            "archive_labels_role": "C050 parent replay/fallback only; not residual-head training labels",
            "source_flag_as_covariate": False,
            "component_gate": {
                "minimum_delta_r2": MIN_BANKABLE_DELTA_R2,
                "minimum_positive_folds": 4,
                "bootstrap_lower_must_exceed": 0.0,
                "minimum_panel_delta_must_be_at_least": 0.0,
            },
            "official_only": True,
            "local_eval_read": False,
            "pi1m_used": False,
            "stored_prediction_replay": False,
            "cross_target_labels": False,
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
        f"Decision: **{report['decision']}**. Egb parent `{active_report['parent_r2']:.12f}`; "
        f"candidate `{active_report['candidate_r2']:.12f}`; delta `{active_report['delta_r2']:+.12f}`. "
        f"Positive folds `{active_report['positive_folds']}/5`; bootstrap lower "
        f"`{active_report['group_bootstrap_lower']:.12f}`; minimum panel delta "
        f"`{active_report['minimum_panel_delta']:.12f}`. Current-source rows "
        f"`{active_report['current_source_rows']}`; archive-only rows `{active_report['archive_only_rows']}`. "
        "Official-only; no local_eval/Kaggle/submission/final-notebook action.\n",
        encoding="utf-8",
    )
    manifest = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
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
                "egb_delta_r2": active_report["delta_r2"],
                "current_source_rows": active_report["current_source_rows"],
                "archive_only_rows": active_report["archive_only_rows"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
