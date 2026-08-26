#!/usr/bin/env python3
"""C187: formal clean reproduction of the ionic-coordinate EPS-only arm."""

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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c180_flory_fox_oligomer_carriers as rich_builder


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
SEED = 20260804
N_COMPONENTS = 64
HALF_PARENT = 0.50
MODEL_KINDS = ("ridge", "et", "hgb")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def make_model(kind: str, fold: int):
    if kind == "ridge":
        return Ridge(alpha=50.0)
    if kind == "et":
        return ExtraTreesRegressor(n_estimators=300, max_features=0.55, min_samples_leaf=2, random_state=SEED + fold, n_jobs=4)
    return HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=1.0, random_state=SEED + fold)


def fold_matrix(dense: np.ndarray, sparse_matrix, train_rows: np.ndarray, pred_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_dense = imputer.fit_transform(dense[train_rows])
    pred_dense = imputer.transform(dense[pred_rows])
    qt = QuantileTransformer(n_quantiles=min(100, max(10, len(train_rows))), output_distribution="normal", random_state=SEED)
    train_dense = qt.fit_transform(train_dense)
    pred_dense = qt.transform(pred_dense)
    if sparse_matrix.shape[1] > 1 and len(train_rows) > 3:
        n_components = min(N_COMPONENTS, sparse_matrix.shape[1] - 1, len(train_rows) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        train_sparse = svd.fit_transform(sparse_matrix[train_rows])
        pred_sparse = svd.transform(sparse_matrix[pred_rows])
        return np.hstack([train_dense, train_sparse]), np.hstack([pred_dense, pred_sparse])
    return train_dense, pred_dense


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in selected])
        if np.var(y[rows]) > 1e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7")
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
    sparse_matrix = sparse_matrix.astype(np.float64)
    eps_info = dict(parent["target_info"]["eps"])
    nc_info = dict(parent["target_info"]["nc"])
    eps_by_canon = dict(zip(eps_info["canonical"], eps_info["y"], strict=True))
    eps_parent_by_canon = dict(zip(eps_info["canonical"], eps_info["parent"], strict=True))
    nc_by_canon = dict(zip(nc_info["canonical"], nc_info["y"], strict=True))
    pair_canons = sorted(set(eps_by_canon) & set(nc_by_canon))
    if len(pair_canons) < 50:
        raise RuntimeError("insufficient official EPS/Nc pair support")
    key_to_index = parent["key_to_index"]
    pair_indices = np.asarray([key_to_index[value] for value in pair_canons], dtype=np.int64)
    eps_y = np.asarray([eps_by_canon[value] for value in pair_canons], dtype=float)
    nc_y = np.asarray([nc_by_canon[value] for value in pair_canons], dtype=float)
    ionic_y = eps_y - nc_y ** 2
    if np.any(ionic_y <= 0):
        raise RuntimeError("non-positive ionic coordinate in official pair rows")
    log_ionic = np.log(ionic_y)
    group_map = dict(zip(eps_info["canonical"], eps_info["groups"], strict=True))
    pair_groups = np.asarray([group_map[value] for value in pair_canons], dtype=object)
    folds = carrier.grouped_folds(pair_groups)
    pair_oof = {kind: np.full(len(pair_canons), np.nan, dtype=float) for kind in MODEL_KINDS}
    fold_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        va = np.flatnonzero(folds == fold)
        tr = np.flatnonzero(folds != fold)
        Xtr, Xva = fold_matrix(dense, sparse_matrix, pair_indices[tr], pair_indices[va])
        for kind in MODEL_KINDS:
            model = make_model(kind, fold)
            model.fit(Xtr, log_ionic[tr])
            pair_oof[kind][va] = np.exp(np.clip(model.predict(Xva), -8, 4))
        raw_eps = nc_y[va] ** 2 + np.mean([pair_oof[kind][va] for kind in MODEL_KINDS], axis=0)
        parent_pair = np.asarray([eps_parent_by_canon[value] for value in np.asarray(pair_canons)[va]], dtype=float)
        candidate_pair = (1.0 - HALF_PARENT) * parent_pair + HALF_PARENT * raw_eps
        fold_rows.append({"fold": fold, "rows": int(len(va)), "parent_r2": float(r2_score(eps_y[va], parent_pair)), "candidate_r2": float(r2_score(eps_y[va], candidate_pair)), "delta_r2": float(r2_score(eps_y[va], candidate_pair) - r2_score(eps_y[va], parent_pair))})
    raw_oof = nc_y ** 2 + np.mean(np.column_stack([pair_oof[kind] for kind in MODEL_KINDS]), axis=1)
    pair_parent = np.asarray([eps_parent_by_canon[value] for value in pair_canons], dtype=float)
    pair_candidate = (1.0 - HALF_PARENT) * pair_parent + HALF_PARENT * raw_oof
    pair_delta = float(r2_score(eps_y, pair_candidate) - r2_score(eps_y, pair_parent))
    pair_positive = int(sum(row["delta_r2"] > 0 for row in fold_rows))
    all_eps_candidate = np.asarray(eps_info["parent"], dtype=float).copy()
    eps_position = {value: index for index, value in enumerate(eps_info["canonical"])}
    for canon, value in zip(pair_canons, pair_candidate, strict=True):
        all_eps_candidate[eps_position[canon]] = value
    eps_delta = float(r2_score(eps_info["y"], all_eps_candidate) - r2_score(eps_info["y"], eps_info["parent"]))
    all_eps_groups = np.asarray(eps_info["groups"], dtype=object)
    eps_bootstrap = bootstrap_lower(np.asarray(eps_info["y"], dtype=float), np.asarray(eps_info["parent"], dtype=float), all_eps_candidate, all_eps_groups)
    passed = bool(eps_delta >= 0.010 and pair_positive >= 4 and eps_bootstrap > 0.0)
    target_reports = {
        "eps": {"parent_r2": float(r2_score(eps_info["y"], eps_info["parent"])), "candidate_r2": float(r2_score(eps_info["y"], all_eps_candidate)), "delta_r2": eps_delta, "pair_parent_r2": float(r2_score(eps_y, pair_parent)), "pair_candidate_r2": float(r2_score(eps_y, pair_candidate)), "pair_delta_r2": pair_delta, "positive_folds": pair_positive, "group_bootstrap_lower": float(eps_bootstrap), "pass": passed, "folds": fold_rows, "pair_rows": int(len(pair_canons))},
        "nc": {"active": False, "changed": False, "parent_r2": float(r2_score(nc_info["y"], nc_info["parent"])), "candidate_r2": float(r2_score(nc_info["y"], nc_info["parent"])), "delta_r2": 0.0},
    }
    target_reports.update({target: {"active": False, "changed": False, "parent_r2": float(r2_score(parent["target_info"][target]["y"], parent["target_info"][target]["parent"])), "candidate_r2": float(r2_score(parent["target_info"][target]["y"], parent["target_info"][target]["parent"])), "delta_r2": 0.0} for target in ("tg", "egc", "egb", "ei", "eea")})
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    candidate_mean = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    predictions = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    if passed:
        train_pair_indices = pair_indices
        Xtr, _ = fold_matrix(dense, sparse_matrix, train_pair_indices, train_pair_indices)
        test_eps = predictions[predictions["target_type"] == "eps"].sort_values("id").reset_index(drop=True)
        test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == "eps"].sort_values("id").reset_index(drop=True)
        test_rows = parent["test"].loc[parent["test"]["target_type"] == "eps"].sort_values("id").reset_index(drop=True)
        if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)) or not np.array_equal(test_eps["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
            raise RuntimeError("C187 EPS test ID alignment failed")
        test_pair_mask = np.asarray([value in nc_by_canon for value in test_rows["canonical"]], dtype=bool)
        if np.any(test_pair_mask):
            pred_indices = np.asarray([key_to_index[value] for value in test_rows.loc[test_pair_mask, "canonical"]], dtype=np.int64)
            _, Xt = fold_matrix(dense, sparse_matrix, train_pair_indices, pred_indices)
            full_preds = []
            for kind in MODEL_KINDS:
                model = make_model(kind, SEED)
                model.fit(Xtr, log_ionic)
                full_preds.append(np.exp(np.clip(model.predict(Xt), -8, 4)))
            nc_test = np.asarray([nc_by_canon[value] for value in test_rows.loc[test_pair_mask, "canonical"]], dtype=float)
            raw = nc_test ** 2 + np.mean(np.column_stack(full_preds), axis=1)
            replacement = (1.0 - HALF_PARENT) * test_detail.loc[test_pair_mask, "target"].to_numpy(float) + HALF_PARENT * raw
            predictions.loc[predictions["target_type"].eq("eps") & predictions["id"].isin(test_detail.loc[test_pair_mask, "id"]), "target"] = replacement
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C187 complete output contract failed")
    report = {"schema_version": "ppp.round2.c187.ionic-eps-only.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "official_inputs": parent["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False, "parent_replay_parity": parity, "feature_report": feature_report, "pair_rows": int(len(pair_canons)), "model_kinds": list(MODEL_KINDS), "half_parent_blend": HALF_PARENT, "target_reports": target_reports, "banked_targets": ["eps"] if passed else [], "mean_parent_r2": parent_mean, "mean_candidate_r2": candidate_mean, "mean_gain": candidate_mean - parent_mean, "complete_output_rows": int(len(predictions)), "complete_output_order_pass": True, "full_candidate_gate_pass": bool(passed and candidate_mean - parent_mean >= 0.002), "decision": "candidate_pass_pending_clean_reproduction" if passed and candidate_mean - parent_mean >= 0.002 else "rejected_component_or_full_gate", "elapsed_seconds": float(time.time() - started), "source_hashes": {"runner": sha256_file(Path(__file__)), "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"), "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"), "rich_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c180_flory_fox_oligomer_carriers.py")}}
    pd.DataFrame({"canonical": eps_info["canonical"], "target": eps_info["y"], "parent": eps_info["parent"], "candidate": all_eps_candidate}).to_csv(run_dir / "eps_oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c187.ionic-eps-only.v1", "seed": SEED, "pair_rows": int(len(pair_canons)), "model_kinds": list(MODEL_KINDS), "half_parent_blend": HALF_PARENT, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: {report['decision']}. EPS banked: {passed}. Mean parent {parent_mean:.12f}; assembled {candidate_mean:.12f}; gain {candidate_mean - parent_mean:+.12f}. Official-only; Nc unchanged; no local_eval read.\n", encoding="utf-8")
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.name != "artifact_manifest.sha256" and path.is_file()]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "banked_targets": report["banked_targets"], "mean_parent_r2": parent_mean, "mean_candidate_r2": candidate_mean, "mean_gain": candidate_mean - parent_mean, "decision": report["decision"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
