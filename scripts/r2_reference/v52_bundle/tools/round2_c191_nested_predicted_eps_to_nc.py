#!/usr/bin/env python3
"""C191: nested predicted-EPS-to-Nc official-only component.

This run tests whether a deployable EPS estimate can improve refractive index
without using observed same-row EPS labels for validation or test rows.  Every
auxiliary EPS fit excludes the outer validation canonical groups before the Nc
residual model is scored.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as rich_builder


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
ACTIVE_TARGET = "nc"
SEED = 20260805
RESIDUAL_WEIGHT = 0.30
EPS_RIDGE_ALPHA = 30.0
NC_RIDGE_ALPHA = 40.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def dense_sparse_pair(
    dense: np.ndarray,
    sparse_matrix: sparse.csr_matrix,
    train_rows: np.ndarray,
    pred_rows: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    clean = np.asarray(dense, dtype=np.float64).copy()
    clean[~np.isfinite(clean) | (np.abs(clean) > 1.0e12)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(imputer.fit_transform(clean[train_rows]))
    pred_dense = scaler.transform(imputer.transform(clean[pred_rows]))
    train_x = sparse.hstack([sparse_matrix[train_rows], sparse.csr_matrix(train_dense)], format="csr")
    pred_x = sparse.hstack([sparse_matrix[pred_rows], sparse.csr_matrix(pred_dense)], format="csr")
    return train_x, pred_x


def append_physical_columns(
    base: sparse.csr_matrix,
    eps_hat: np.ndarray,
    parent_nc: np.ndarray,
) -> sparse.csr_matrix:
    eps_hat = np.asarray(eps_hat, dtype=np.float64)
    parent_nc = np.asarray(parent_nc, dtype=np.float64)
    ionic = eps_hat - parent_nc ** 2
    extra = np.column_stack([
        eps_hat,
        np.clip(eps_hat, 0.0, 20.0),
        ionic,
        np.sign(ionic) * np.log1p(np.abs(ionic)),
        parent_nc,
        parent_nc ** 2,
    ])
    extra[~np.isfinite(extra)] = 0.0
    return sparse.hstack([base, sparse.csr_matrix(extra)], format="csr")


def fit_eps_predictor(
    dense: np.ndarray,
    sparse_matrix: sparse.csr_matrix,
    eps_info: dict[str, Any],
    pred_indices: np.ndarray,
    excluded_groups: set[str],
) -> np.ndarray:
    eps_groups = np.asarray(eps_info["groups"], dtype=object)
    keep = np.asarray([str(group) not in excluded_groups for group in eps_groups], dtype=bool)
    if int(np.sum(keep)) < 20:
        raise RuntimeError("insufficient EPS labels after group exclusion")
    train_rows = np.asarray(eps_info["indices"], dtype=np.int64)[keep]
    y = np.asarray(eps_info["y"], dtype=float)[keep]
    train_x, pred_x = dense_sparse_pair(dense, sparse_matrix, train_rows, pred_indices)
    model = Ridge(alpha=EPS_RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
    model.fit(train_x, y)
    return model.predict(pred_x)


def nested_eps_hat_for_training(
    dense: np.ndarray,
    sparse_matrix: sparse.csr_matrix,
    eps_info: dict[str, Any],
    nc_indices: np.ndarray,
    nc_groups: np.ndarray,
    outer_excluded_groups: set[str] | None = None,
) -> np.ndarray:
    outer_excluded_groups = set() if outer_excluded_groups is None else {str(value) for value in outer_excluded_groups}
    folds = carrier.grouped_folds(nc_groups)
    eps_hat = np.full(len(nc_indices), np.nan, dtype=np.float64)
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        excluded = {str(value) for value in nc_groups[validation]} | outer_excluded_groups
        if not outer_excluded_groups.issubset(excluded):
            raise RuntimeError("outer validation groups were not excluded from nested EPS auxiliary fit")
        eps_hat[validation] = fit_eps_predictor(dense, sparse_matrix, eps_info, nc_indices[validation], excluded)
    if not np.isfinite(eps_hat).all():
        raise RuntimeError("nested EPS predictor produced non-finite training values")
    return eps_hat


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, selected: np.ndarray, minimum: int = 5) -> float | None:
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


def evaluate_panels(
    info: dict[str, Any],
    fingerprints: list[Any],
    eps_canonicals: set[str],
    candidate: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    folds = carrier.grouped_folds(groups)
    nearest = nearest_similarity(fingerprints, indices, folds)
    has_eps = np.asarray([str(value) in eps_canonicals for value in info["canonical"]], dtype=bool)
    panel_specs: dict[str, np.ndarray] = {
        "official_eps_counterpart_present": has_eps,
        "official_eps_counterpart_missing": ~has_eps,
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
        delta = panel_delta(y, parent, candidate, selected)
        panels[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable_insufficient_support",
        }
        if delta is not None:
            values.append(delta)
    return {"panels": panels, "minimum_panel_delta": min(values) if values else 0.0, "nearest_tanimoto": nearest}


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
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")

    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    dense, sparse_matrix, feature_report = rich_builder.build_features(root, parent["keys"])
    dense = np.asarray(dense, dtype=np.float64)
    sparse_matrix = sparse_matrix.astype(np.float64).tocsr()
    eps_info = dict(parent["target_info"]["eps"])
    nc_info = dict(parent["target_info"]["nc"])
    eps_canonicals = {str(value) for value in eps_info["canonical"]}
    nc_y = np.asarray(nc_info["y"], dtype=float)
    nc_parent = np.asarray(nc_info["parent"], dtype=float)
    nc_indices = np.asarray(nc_info["indices"], dtype=np.int64)
    nc_groups = np.asarray(nc_info["groups"], dtype=object)
    folds = carrier.grouped_folds(nc_groups)
    candidate = np.full(len(nc_y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []

    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        excluded = {str(value) for value in nc_groups[validation]}
        train_eps_hat = nested_eps_hat_for_training(
            dense,
            sparse_matrix,
            eps_info,
            nc_indices[training],
            nc_groups[training],
            outer_excluded_groups=excluded,
        )
        validation_eps_hat = fit_eps_predictor(dense, sparse_matrix, eps_info, nc_indices[validation], excluded)
        train_x_base, validation_x_base = dense_sparse_pair(dense, sparse_matrix, nc_indices[training], nc_indices[validation])
        train_x = append_physical_columns(train_x_base, train_eps_hat, nc_parent[training])
        validation_x = append_physical_columns(validation_x_base, validation_eps_hat, nc_parent[validation])
        ridge = Ridge(alpha=NC_RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
        ridge.fit(train_x, nc_y[training] - nc_parent[training])
        residual = ridge.predict(validation_x)
        candidate[validation] = reference.clip_prediction(nc_y[training], nc_parent[validation] + RESIDUAL_WEIGHT * residual)
        fold_parent = float(r2_score(nc_y[validation], nc_parent[validation]))
        fold_candidate = float(r2_score(nc_y[validation], candidate[validation]))
        fold_rows.append({
            "fold": int(fold),
            "rows": int(len(validation)),
            "parent_r2": fold_parent,
            "candidate_r2": fold_candidate,
            "delta_r2": fold_candidate - fold_parent,
        })

    if not np.isfinite(candidate).all():
        raise RuntimeError("C191 produced non-finite OOF candidate")
    nc_delta = float(r2_score(nc_y, candidate) - r2_score(nc_y, nc_parent))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    lower = float(carrier.bootstrap_lower(nc_y, nc_parent, candidate, nc_groups))
    panel_report = evaluate_panels(nc_info, parent["fingerprints"], eps_canonicals, candidate)
    passed = bool(nc_delta >= 0.010 and positive >= 4 and lower > 0.0 and panel_report["minimum_panel_delta"] >= 0.0)

    target_reports: dict[str, Any] = {
        target: {
            "active": False,
            "changed": False,
            "parent_r2": float(r2_score(parent["target_info"][target]["y"], parent["target_info"][target]["parent"])),
            "candidate_r2": float(r2_score(parent["target_info"][target]["y"], parent["target_info"][target]["parent"])),
            "delta_r2": 0.0,
        }
        for target in TARGETS
        if target != ACTIVE_TARGET
    }
    target_reports[ACTIVE_TARGET] = {
        "active": True,
        "parent_r2": float(r2_score(nc_y, nc_parent)),
        "candidate_r2": float(r2_score(nc_y, candidate)),
        "delta_r2": nc_delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_panel_delta": float(panel_report["minimum_panel_delta"]),
        "panels": panel_report["panels"],
        "folds": fold_rows,
        "pass": passed,
        "eps_labels_used": int(len(eps_info["y"])),
    }
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    candidate_mean = float(np.mean([
        target_reports[target]["candidate_r2"] if target == ACTIVE_TARGET and passed else target_reports[target]["parent_r2"]
        for target in TARGETS
    ]))

    predictions = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    direct_test = pd.DataFrame(columns=["id", "target_type", "direct_candidate"])
    if passed:
        full_eps_hat = nested_eps_hat_for_training(dense, sparse_matrix, eps_info, nc_indices, nc_groups)
        test_rows = parent["test"].loc[parent["test"]["target_type"] == ACTIVE_TARGET].sort_values("id").reset_index(drop=True)
        test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == ACTIVE_TARGET].sort_values("id").reset_index(drop=True)
        if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
            raise RuntimeError("C191 NC test ID alignment failed")
        test_indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
        test_eps_hat = fit_eps_predictor(dense, sparse_matrix, eps_info, test_indices, set())
        train_x_base, test_x_base = dense_sparse_pair(dense, sparse_matrix, nc_indices, test_indices)
        train_x = append_physical_columns(train_x_base, full_eps_hat, nc_parent)
        test_parent = test_detail["target"].to_numpy(float)
        test_x = append_physical_columns(test_x_base, test_eps_hat, test_parent)
        full_model = Ridge(alpha=NC_RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
        full_model.fit(train_x, nc_y - nc_parent)
        direct_values = reference.clip_prediction(nc_y, test_parent + RESIDUAL_WEIGHT * full_model.predict(test_x))
        direct_test = pd.DataFrame({
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "direct_candidate": direct_values,
        })
        nc_id_to_value = dict(zip(direct_test["id"].astype(int), direct_test["direct_candidate"].astype(float), strict=True))
        mask = predictions["target_type"].eq(ACTIVE_TARGET)
        predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(nc_id_to_value).astype(float).to_numpy()

    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C191 complete output contract failed")

    report = {
        "schema_version": "ppp.round2.c191.nested-predicted-eps-to-nc.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "pi1m_used": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "active_targets": [ACTIVE_TARGET],
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if passed else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": bool(passed and candidate_mean - parent_mean >= 0.002),
        "decision": "candidate_pass_pending_clean_reproduction" if passed and candidate_mean - parent_mean >= 0.002 else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"),
            "rich_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c180_flory_fox_oligomer_carriers.py"),
            "reference": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py"),
        },
    }
    pd.DataFrame({
        "canonical": nc_info["canonical"],
        "target_type": ACTIVE_TARGET,
        "target": nc_y,
        "parent": nc_parent,
        "candidate": candidate,
        "group": nc_groups,
        "fold": folds,
        "nearest_tanimoto": panel_report["nearest_tanimoto"],
    }).to_csv(run_dir / "nc_oof_predictions.csv", index=False)
    direct_test.to_csv(run_dir / "component_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.c191.nested-predicted-eps-to-nc.v1",
        "seed": SEED,
        "active_targets": [ACTIVE_TARGET],
        "residual_weight": RESIDUAL_WEIGHT,
        "eps_ridge_alpha": EPS_RIDGE_ALPHA,
        "nc_ridge_alpha": NC_RIDGE_ALPHA,
        "folds": "outer grouped no-stereo; EPS auxiliary excludes each outer validation group",
        "local_eval_read": False,
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
        f"# {run_dir.name}\n\nDecision: {report['decision']}. Nc banked: {passed}. Mean parent {parent_mean:.12f}; assembled {candidate_mean:.12f}; gain {candidate_mean - parent_mean:+.12f}. Official-only; no local_eval read.\n",
        encoding="utf-8",
    )
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.name != "artifact_manifest.sha256" and path.is_file()]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment_id": run_dir.name,
        "banked_targets": report["banked_targets"],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "decision": report["decision"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
