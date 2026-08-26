#!/usr/bin/env python3
"""C194: safe Ei cross-property Stage-2 residual.

This is a cleaned, bounded version of the Claude Stage-2 idea for the Ei target.
It tests one scientific factor: whether fold-available Egc/Eea identity features
with a structure-only fallback can improve Ei beyond the exact C050 parent.

Safety properties:
- official Round 2 inputs only;
- no local_eval, public-score, Kaggle compute, upload, or submission action;
- exact C050 source replay before fitting;
- active target Ei is never used as a feature;
- every validation fold excludes the outer Ei validation no-stereo groups from
  Egc/Eea availability and fallback partner fits;
- no Stage-4 in-sample routing or post-hoc row selection.
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
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as rich_builder


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "ei"
PARTNERS = ("egc", "eea")
SEED = 20260805
PARTNER_RIDGE_ALPHA = 25.0
EI_RIDGE_ALPHA = 30.0
RESIDUAL_WEIGHT = 0.35
MIN_BANKABLE_DELTA_R2 = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def combine_dense_sparse(
    dense: np.ndarray,
    sparse_matrix: sparse.csr_matrix,
    train_key_indices: np.ndarray,
    pred_key_indices: np.ndarray,
    train_extra: np.ndarray,
    pred_extra: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    dense = np.asarray(dense, dtype=np.float64)
    train_extra = np.asarray(train_extra, dtype=np.float64)
    pred_extra = np.asarray(pred_extra, dtype=np.float64)
    base = dense.copy()
    base[~np.isfinite(base) | (np.abs(base) > 1.0e12)] = np.nan
    train_dense = np.hstack([base[train_key_indices], train_extra])
    pred_dense = np.hstack([base[pred_key_indices], pred_extra])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(imputer.fit_transform(train_dense))
    pred_scaled = scaler.transform(imputer.transform(pred_dense))
    return (
        sparse.hstack([sparse_matrix[train_key_indices], sparse.csr_matrix(train_scaled)], format="csr"),
        sparse.hstack([sparse_matrix[pred_key_indices], sparse.csr_matrix(pred_scaled)], format="csr"),
    )


def observed_label_maps(parent: dict[str, Any], excluded_groups: set[str] | None = None) -> dict[str, dict[str, float]]:
    excluded_groups = set() if excluded_groups is None else {str(value) for value in excluded_groups}
    result: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        info = parent["target_info"][target]
        bucket: dict[str, list[float]] = {}
        for canonical, group, value in zip(info["canonical"], info["groups"], info["y"], strict=True):
            canonical_key = str(canonical)
            group_key = str(group)
            if canonical_key in excluded_groups or group_key in excluded_groups:
                continue
            bucket.setdefault(canonical_key, []).append(float(value))
        result[target] = {key: float(np.mean(values)) for key, values in bucket.items()}
    return result


def fit_partner_predictor(
    dense: np.ndarray,
    sparse_matrix: sparse.csr_matrix,
    partner_info: dict[str, Any],
    pred_key_indices: np.ndarray,
    excluded_groups: set[str],
) -> np.ndarray:
    groups = np.asarray(partner_info["groups"], dtype=object)
    keep = np.asarray([str(value) not in excluded_groups for value in groups], dtype=bool)
    if int(np.sum(keep)) < 20:
        raise RuntimeError("insufficient partner labels after outer group exclusion")
    train_indices = np.asarray(partner_info["indices"], dtype=np.int64)[keep]
    y = np.asarray(partner_info["y"], dtype=np.float64)[keep]
    zeros_train = np.zeros((len(train_indices), 0), dtype=np.float64)
    zeros_pred = np.zeros((len(pred_key_indices), 0), dtype=np.float64)
    train_x, pred_x = combine_dense_sparse(
        dense,
        sparse_matrix,
        train_indices,
        pred_key_indices,
        zeros_train,
        zeros_pred,
    )
    # combine_dense_sparse receives zero-column extras by row number; keep the
    # explicit arrays above as a guard against accidental target covariates.
    if zeros_train.shape[1] != 0 or zeros_pred.shape[1] != 0:
        raise RuntimeError("unexpected partner extra feature columns")
    model = Ridge(alpha=PARTNER_RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
    model.fit(train_x, y)
    return model.predict(pred_x)


def cross_block(
    parent: dict[str, Any],
    dense: np.ndarray,
    sparse_matrix: sparse.csr_matrix,
    canonicals: np.ndarray,
    key_indices: np.ndarray,
    active_parent: np.ndarray,
    excluded_groups: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maps = observed_label_maps(parent, excluded_groups)
    partner_values: dict[str, np.ndarray] = {}
    observed: dict[str, np.ndarray] = {}
    for partner in PARTNERS:
        fallback = fit_partner_predictor(dense, sparse_matrix, parent["target_info"][partner], key_indices, excluded_groups)
        values = fallback.copy()
        flags = np.zeros(len(canonicals), dtype=bool)
        partner_map = maps[partner]
        for row, canonical in enumerate(canonicals):
            key = str(canonical)
            if key in partner_map:
                values[row] = partner_map[key]
                flags[row] = True
        partner_values[partner] = values
        observed[partner] = flags
    egc = partner_values["egc"]
    eea = partner_values["eea"]
    egc_obs = observed["egc"].astype(np.float64)
    eea_obs = observed["eea"].astype(np.float64)
    identity = egc + eea
    diff = identity - active_parent
    block = np.column_stack([
        egc,
        eea,
        egc_obs,
        eea_obs,
        egc_obs * eea_obs,
        np.maximum(egc_obs, eea_obs),
        identity,
        diff,
        np.abs(diff),
        egc - eea,
        egc * eea,
        active_parent,
    ]).astype(np.float64)
    block[~np.isfinite(block)] = 0.0
    return block, observed["egc"], observed["eea"]


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, selected: np.ndarray, minimum: int = 8) -> float | None:
    if int(np.sum(selected)) < minimum or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))


def nearest_similarity(fingerprints: list[Any], indices: np.ndarray, folds: np.ndarray) -> np.ndarray:
    result = np.full(len(indices), np.nan, dtype=np.float64)
    for fold in sorted(set(int(value) for value in folds)):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[int(indices[row])] for row in training]
        for row in validation:
            result[row] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(indices[row])], train_fps))
    return result


def transfer_panels(
    info: dict[str, Any],
    fingerprints: list[Any],
    candidate: np.ndarray,
    egc_obs: np.ndarray,
    eea_obs: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    folds = carrier.grouped_folds(groups)
    nearest = nearest_similarity(fingerprints, indices, folds)
    both = egc_obs & eea_obs
    one = egc_obs ^ eea_obs
    none = ~(egc_obs | eea_obs)
    panel_specs: dict[str, np.ndarray] = {
        "egc_and_eea_observed": both,
        "exactly_one_partner_observed": one,
        "no_partner_observed": none,
        "similarity_lt_0.30": nearest < 0.30,
        "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
        "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
        "similarity_ge_0.70": nearest >= 0.70,
        "quantile_low": y <= np.quantile(y, 0.25),
        "quantile_high": y >= np.quantile(y, 0.75),
    }
    for name in sorted(set(scaffolds)):
        selected = scaffolds == name
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
    strata = {
        "egc_and_eea_observed": panel_delta(y, parent_oof, candidate, both),
        "exactly_one_partner_observed": panel_delta(y, parent_oof, candidate, one),
        "no_partner_observed": panel_delta(y, parent_oof, candidate, none),
    }
    evaluable_strata = [value for value in strata.values() if value is not None]
    return {
        "panels": panels,
        "strata": strata,
        "minimum_transfer_panel_delta": float(min(values)) if values else 0.0,
        "minimum_stratum_delta": float(min(evaluable_strata)) if evaluable_strata else 0.0,
        "partner_observed_rows": {
            "egc_and_eea_observed": int(np.sum(both)),
            "exactly_one_partner_observed": int(np.sum(one)),
            "no_partner_observed": int(np.sum(none)),
        },
    }


def fit_active_target(
    parent: dict[str, Any],
    dense: np.ndarray,
    sparse_matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    info = parent["target_info"][ACTIVE_TARGET]
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    key_indices = np.asarray(info["indices"], dtype=np.int64)
    canonicals = np.asarray(info["canonical"], dtype=object)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    egc_obs_oof = np.zeros(len(y), dtype=bool)
    eea_obs_oof = np.zeros(len(y), dtype=bool)
    fold_rows: list[dict[str, Any]] = []

    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        excluded = {str(value) for value in groups[validation]} | {str(value) for value in canonicals[validation]}
        combined = np.concatenate([training, validation])
        cross, egc_obs, eea_obs = cross_block(
            parent,
            dense,
            sparse_matrix,
            canonicals[combined],
            key_indices[combined],
            parent_oof[combined],
            excluded,
        )
        train_rows = np.arange(len(training), dtype=np.int64)
        valid_rows = np.arange(len(training), len(combined), dtype=np.int64)
        train_x, valid_x = combine_dense_sparse(
            dense,
            sparse_matrix,
            key_indices[training],
            key_indices[validation],
            cross[train_rows],
            cross[valid_rows],
        )
        model = Ridge(alpha=EI_RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
        model.fit(train_x, y[training] - parent_oof[training])
        residual = model.predict(valid_x)
        candidate[validation] = reference.clip_prediction(y[training], parent_oof[validation] + RESIDUAL_WEIGHT * residual)
        egc_obs_oof[validation] = egc_obs[valid_rows]
        eea_obs_oof[validation] = eea_obs[valid_rows]
        fold_rows.append({
            "fold": int(fold),
            "rows": int(len(validation)),
            "egc_and_eea_observed_rows": int(np.sum(egc_obs[valid_rows] & eea_obs[valid_rows])),
            "one_partner_observed_rows": int(np.sum(egc_obs[valid_rows] ^ eea_obs[valid_rows])),
            "no_partner_observed_rows": int(np.sum(~(egc_obs[valid_rows] | eea_obs[valid_rows]))),
            "parent_r2": float(r2_score(y[validation], parent_oof[validation])),
            "candidate_r2": float(r2_score(y[validation], candidate[validation])),
            "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent_oof[validation])),
        })
    if not np.isfinite(candidate).all():
        raise RuntimeError("C194 produced non-finite Ei OOF predictions")
    delta = float(r2_score(y, candidate) - r2_score(y, parent_oof))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    lower = float(carrier.bootstrap_lower(y, parent_oof, candidate, groups))
    panels = transfer_panels(info, parent["fingerprints"], candidate, egc_obs_oof, eea_obs_oof)
    passed = bool(
        delta >= MIN_BANKABLE_DELTA_R2
        and positive >= 4
        and lower > 0.0
        and panels["minimum_transfer_panel_delta"] >= 0.0
        and panels["minimum_stratum_delta"] >= 0.0
    )
    report = {
        "active": True,
        "parent_r2": float(r2_score(y, parent_oof)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_transfer_panel_delta": float(panels["minimum_transfer_panel_delta"]),
        "minimum_stratum_delta": float(panels["minimum_stratum_delta"]),
        "panels": panels["panels"],
        "strata": panels["strata"],
        "partner_observed_rows": panels["partner_observed_rows"],
        "folds": fold_rows,
        "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
        "pass": passed,
    }
    return candidate, report, egc_obs_oof, eea_obs_oof


def full_test_predictions(
    parent: dict[str, Any],
    dense: np.ndarray,
    sparse_matrix: sparse.csr_matrix,
    bank_ei: bool,
) -> np.ndarray:
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if not bank_ei:
        return parent_test[["id", "target"]].sort_values("id")["target"].to_numpy(np.float64)

    info = parent["target_info"][ACTIVE_TARGET]
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    train_indices = np.asarray(info["indices"], dtype=np.int64)
    train_canonicals = np.asarray(info["canonical"], dtype=object)
    train_cross, _, _ = cross_block(parent, dense, sparse_matrix, train_canonicals, train_indices, parent_oof, set())

    test_rows = parent["test"].loc[parent["test"]["target_type"] == ACTIVE_TARGET].sort_values("id").reset_index(drop=True)
    test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == ACTIVE_TARGET].sort_values("id").reset_index(drop=True)
    if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
        raise RuntimeError("C194 Ei test ID alignment failed")
    test_indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    test_parent = test_detail["target"].to_numpy(np.float64)
    test_cross, _, _ = cross_block(
        parent,
        dense,
        sparse_matrix,
        test_rows["canonical"].to_numpy(object),
        test_indices,
        test_parent,
        set(),
    )
    train_x, test_x = combine_dense_sparse(
        dense,
        sparse_matrix,
        train_indices,
        test_indices,
        train_cross,
        test_cross,
    )
    model = Ridge(alpha=EI_RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
    model.fit(train_x, y - parent_oof)
    residual = model.predict(test_x)
    candidate_test = test_parent + RESIDUAL_WEIGHT * residual
    prediction_detail = parent_test.copy()
    mask = prediction_detail["target_type"].astype(str).eq(ACTIVE_TARGET)
    prediction_detail.loc[mask, "target"] = candidate_test
    return prediction_detail[["id", "target"]].sort_values("id")["target"].to_numpy(np.float64)


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
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")

    data_dir = (root / args.data_dir).resolve()
    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    dense, sparse_features, feature_report = rich_builder.build_features(root, parent["keys"])
    dense = np.asarray(dense, dtype=np.float64)
    sparse_features = sparse_features.astype(np.float64).tocsr()
    ei_candidate, ei_report, egc_obs, eea_obs = fit_active_target(parent, dense, sparse_features)

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    banked = [ACTIVE_TARGET] if ei_report["pass"] else []
    for target in TARGETS:
        info = parent["target_info"][target]
        y = np.asarray(info["y"], dtype=np.float64)
        parent_oof = np.asarray(info["parent"], dtype=np.float64)
        if target == ACTIVE_TARGET:
            candidate = ei_candidate
            target_reports[target] = ei_report
        else:
            candidate = parent_oof.copy()
            target_reports[target] = {
                "active": False,
                "parent_r2": float(r2_score(y, parent_oof)),
                "candidate_r2": float(r2_score(y, candidate)),
                "delta_r2": 0.0,
                "pass": False,
            }
        assembled = candidate if target in banked else parent_oof
        oof_parts.append(pd.DataFrame({
            "canonical": info["canonical"],
            "target_type": target,
            "target": y,
            "parent": parent_oof,
            "candidate": candidate,
            "assembled": assembled,
            "group": info["groups"],
            "scaffold": info["scaffolds"],
            "fold": carrier.grouped_folds(np.asarray(info["groups"], dtype=object)),
        }))

    parent_mean = float(np.mean([r2_score(part["target"], part["parent"]) for part in oof_parts]))
    candidate_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    prediction_values = full_test_predictions(parent, dense, sparse_features, bool(banked))
    predictions = parent["test_parent_detail"][["id", "target_type"]].copy().sort_values("id").reset_index(drop=True)
    predictions["target"] = prediction_values
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C194 ID coverage/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C194 produced non-finite full predictions")

    report = {
        "schema_version": "ppp.round2.c194.safe-ei-cross-property-stage2.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "active_target": ACTIVE_TARGET,
        "partner_targets": list(PARTNERS),
        "stage2_safety": {
            "active_target_feature_excluded": True,
            "outer_validation_groups_excluded_from_partner_availability": True,
            "unavailable_partner_fallback": "fold-local structure-only Ridge predictor",
            "stage4_in_sample_routing": False,
            "same_oof_max_selection": False,
        },
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": bool(banked and candidate_mean - parent_mean >= 0.002),
        "goal_0_95_met": False,
        "decision": "candidate_pass_pending_clean_reproduction" if banked and candidate_mean - parent_mean >= 0.002 else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"),
            "rich_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "reference": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py"),
        },
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions[["id", "target"]].to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.c194.safe-ei-cross-property-stage2.v1",
        "seed": SEED,
        "active_target": ACTIVE_TARGET,
        "partner_targets": list(PARTNERS),
        "partner_ridge_alpha": PARTNER_RIDGE_ALPHA,
        "ei_ridge_alpha": EI_RIDGE_ALPHA,
        "residual_weight": RESIDUAL_WEIGHT,
        "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
        "feature_basis": "C180 rich official-SMILES features plus fixed safe cross-property identity block",
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
        f"rdkit={reference.Chem.rdBase.rdkitVersion}",
        f"platform={platform.platform()}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Active target: `{ACTIVE_TARGET}`. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{parent_mean:.12f}`; assembled `{candidate_mean:.12f}`; gain `{candidate_mean - parent_mean:+.12f}`. Official-only; no local_eval, Kaggle compute, upload, or submission.\n",
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
    print(json.dumps({
        "experiment_id": run_dir.name,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "decision": report["decision"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
