#!/usr/bin/env python3
"""C114: fixed fold-local spline/GAM residual portfolio."""

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
from rdkit import RDLogger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c076_eps_paired_charge_polarizability_residual as compact_features
import round2_c112_c050_parent_parity_control as parent_control
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_mixed_candidate_v7 as mixed


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE = ("eps", "nc", "ei", "tg")
SEEDS = (2026, 2027, 2028)
RESIDUAL_WEIGHT = 0.15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def model_for(target: str) -> Any:
    estimator: Any
    if target == "ei":
        estimator = HuberRegressor(epsilon=1.35, alpha=1.0e-4, max_iter=2000)
    else:
        alpha = {"eps": 25.0, "nc": 50.0, "tg": 50.0}[target]
        estimator = Ridge(alpha=alpha)
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        SplineTransformer(n_knots=4, degree=2, knots="quantile", include_bias=False, extrapolation="linear"),
        estimator,
    )


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025)) if values else 0.0


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> tuple[dict[str, Any], float]:
    report: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        rows = int(np.sum(selected))
        item: dict[str, Any] = {"rows": rows, "status": "inapplicable", "delta_r2": 0.0}
        if rows >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"status": "evaluable", "delta_r2": delta})
            deltas.append(delta)
        report[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        add(name, selected)
    for value in sorted(set(scaffolds)):
        add(f"scaffold_{value}", scaffolds == value, minimum=10)
    return report, float(min(deltas)) if deltas else 0.0


def target_feature_matrix(bundle: dict[str, Any], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    test = bundle["test"]
    info = bundle["parent_oof"][bundle["parent_oof"]["target_type"] == target]
    test_rows = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
    keys = sorted(set(info["canonical"]) | set(test_rows["canonical"]))
    global_indices = [bundle["key_to_index"][value] for value in keys]
    endpoint, endpoint_names = fixed_features.fixed_features(bundle["molecules"], global_indices)
    physics, physics_names = compact_features.physics_features(bundle["molecules"], global_indices)
    charge, charge_names = compact_features.charge_features(bundle["molecules"], global_indices)
    features = np.hstack([endpoint, physics, charge]).astype(np.float64, copy=False)
    row_for_key = {value: row for row, value in enumerate(keys)}
    train_rows = np.asarray([row_for_key[value] for value in info["canonical"]], dtype=np.int64)
    test_rows_idx = np.asarray([row_for_key[value] for value in test_rows["canonical"]], dtype=np.int64)
    return features, train_rows, test_rows_idx, endpoint_names + physics_names + charge_names


def run_oof_replica(seed: int, bundle: dict[str, Any], feature_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]], fold_maps: dict[str, np.ndarray]) -> tuple[dict[str, Any], pd.DataFrame]:
    reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        rows = bundle["parent_oof"][bundle["parent_oof"]["target_type"] == target].reset_index(drop=True)
        y = rows["target"].to_numpy(float)
        parent = rows["parent_prediction"].to_numpy(float)
        canonical = rows["canonical"].astype(str).to_numpy(object)
        groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
        folds = fold_maps[target]
        candidate = parent.copy()
        similarity = np.full(len(rows), np.nan, dtype=np.float64)
        feature_names: list[str] = []
        if target in ACTIVE:
            matrix, train_rows, _, feature_names = feature_cache[target]
            residual = y - parent
            global_train = np.asarray([bundle["key_to_index"][value] for value in canonical], dtype=np.int64)
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                fitted = model_for(target)
                fitted.fit(matrix[train_rows[training]], residual[training])
                correction = RESIDUAL_WEIGHT * np.asarray(fitted.predict(matrix[train_rows[validation]])).reshape(-1)
                candidate[validation] = parent[validation] + correction
                train_fps = [bundle["fingerprints"][int(global_train[index])] for index in training]
                validation_fps = [bundle["fingerprints"][int(global_train[index])] for index in validation]
                from rdkit import DataStructs
                similarity[validation] = np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in validation_fps], dtype=np.float64)
        fold_rows = []
        for fold in range(5):
            selected = folds == fold
            fold_rows.append({"fold": fold, "rows": int(np.sum(selected)), "parent_r2": float(r2_score(y[selected], parent[selected])), "candidate_r2": float(r2_score(y[selected], candidate[selected])), "delta_r2": float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))})
        parent_r2 = float(r2_score(y, parent))
        candidate_r2 = float(r2_score(y, candidate))
        delta = candidate_r2 - parent_r2
        lower = bootstrap_lower(y, parent, candidate, groups) if target in ACTIVE else 0.0
        panel, minimum_panel = panel_report(y, parent, candidate, similarity, scaffolds) if target in ACTIVE else ({"unchanged_parent": {"rows": int(len(y)), "status": "unchanged", "delta_r2": 0.0}}, 0.0)
        positive = int(sum(item["delta_r2"] > 0 for item in fold_rows))
        gates = {"gain_pass": target not in ACTIVE or delta >= 0.01, "fold_pass": target not in ACTIVE or positive >= 4, "bootstrap_pass": target not in ACTIVE or lower > 0.0, "panel_pass": target not in ACTIVE or minimum_panel >= 0.0, "strict_no_regression": delta >= -0.003}
        reports[target] = {"parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": float(delta), "positive_folds": positive, "group_bootstrap_lower": float(lower), "minimum_panel_delta": float(minimum_panel), "folds": fold_rows, "panels": panel, "feature_count": len(feature_names), "feature_names": feature_names, "pass": bool(all(gates.values())), "gates": gates, "unchanged_parent": target not in ACTIVE}
        oof_parts.append(pd.DataFrame({"canonical": canonical, "target_type": target, "target": y, "parent": parent, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}))
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in TARGETS]))
    maximum_loss = float(min(reports[target]["delta_r2"] for target in TARGETS))
    gate_pass = bool(mean_candidate - mean_parent >= 0.002 and maximum_loss >= -0.003 and all(reports[target]["pass"] for target in ACTIVE))
    return {"seed": seed, "target_reports": reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": maximum_loss, "clean_gate_pass": gate_pass}, pd.concat(oof_parts, ignore_index=True)


def fit_full_predictions(bundle: dict[str, Any], feature_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]], replica_oof: pd.DataFrame) -> pd.DataFrame:
    detail = bundle["parent_predictions"].merge(bundle["test"][["id", "target_type"]], on="id", how="left", validate="one_to_one")[["id", "target_type", "target"]].rename(columns={"target": "model_prediction"})
    for target in ACTIVE:
        rows = replica_oof[replica_oof["target_type"] == target].sort_values("canonical").reset_index(drop=True)
        info = bundle["parent_oof"][bundle["parent_oof"]["target_type"] == target].sort_values("canonical").reset_index(drop=True)
        matrix, train_rows, test_rows, _ = feature_cache[target]
        residual = info["target"].to_numpy(float) - info["parent_prediction"].to_numpy(float)
        fitted = model_for(target)
        fitted.fit(matrix[train_rows], residual)
        test_frame = bundle["test"][bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        parent_test = detail[detail["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float)
        candidate = parent_test + RESIDUAL_WEIGHT * np.asarray(fitted.predict(matrix[test_rows])).reshape(-1)
        mask = detail["target_type"].to_numpy(object) == target
        replacement = pd.Series(candidate, index=test_frame["id"].to_numpy())
        detail.loc[mask, "model_prediction"] = detail.loc[mask, "id"].map(replacement).astype(float).to_numpy()
    final_detail, _ = reference.apply_official_overrides(detail, bundle["test"], bundle["raw_labels"])
    return final_detail[["id", "target"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only C114 directory required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    parent_predictions, parent_oof, context = parent_control.rebuild_parent(root, data_dir, run_dir)
    canonical_dir = root / "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7"
    canonical_predictions = pd.read_csv(canonical_dir / "predictions.csv")
    canonical_oof = pd.read_csv(canonical_dir / "oof_predictions.csv")
    replay = parent_oof.sort_values(["target_type", "canonical"]).reset_index(drop=True)
    reference_oof = canonical_oof[["canonical", "target_type", "target", "candidate_prediction"]].rename(columns={"candidate_prediction": "canonical_prediction"}).sort_values(["target_type", "canonical"]).reset_index(drop=True)
    oof_max = float(np.max(np.abs(replay["parent_prediction"].to_numpy(float) - reference_oof["canonical_prediction"].to_numpy(float))))
    test_max = float(np.max(np.abs(parent_predictions["target"].to_numpy(float) - canonical_predictions["target"].to_numpy(float))))
    if oof_max > 1.0e-12 or test_max > 1.0e-12:
        raise RuntimeError(f"C114 parent parity failed: oof={oof_max} test={test_max}")
    train, test, archive = context["train"], context["test"], context["archive"]
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "mixed_candidate": True, "special_targets": list(mixed.SPECIAL_TARGETS)})
    sparse_parts = [reference.morgan_count_matrix(molecules, radius=2, bits=4096), reference.morgan_count_matrix(molecules, radius=3, bits=4096), reference.text_matrix(keys, 65536)]
    ei_oof, _, _ = mixed.specialized_target("ei", pooled, test, keys, dense, cross_values, cross_available, sparse_parts, fingerprints, config)
    ei_map = {str(key): int(fold) for key, fold in zip(ei_oof["canonical"], ei_oof["outer_fold"], strict=True)}
    fold_maps: dict[str, np.ndarray] = {}
    for target in TARGETS:
        rows = parent_oof[parent_oof["target_type"] == target].reset_index(drop=True)
        if target == "ei":
            fold_maps[target] = np.asarray([ei_map[value] for value in rows["canonical"]], dtype=np.int64)
        else:
            folds = np.full(len(rows), -1, dtype=np.int64)
            for fold, (_, validation) in enumerate(KFold(n_splits=5, shuffle=True, random_state=2026).split(np.arange(len(rows)))):
                folds[validation] = fold
            fold_maps[target] = folds
    bundle = {"parent_oof": parent_oof, "parent_predictions": parent_predictions, "test": test, "raw_labels": raw_labels, "molecules": molecules, "fingerprints": fingerprints, "key_to_index": key_to_index}
    feature_cache = {target: target_feature_matrix(bundle, target) for target in ACTIVE}
    replica_reports: list[dict[str, Any]] = []
    replica_oofs: list[pd.DataFrame] = []
    for seed in SEEDS:
        report, oof = run_oof_replica(seed, bundle, feature_cache, fold_maps)
        replica_reports.append(report)
        replica_oofs.append(oof)
        write_json(run_dir / f"replica_{seed}_metrics.json", report)
    clean_pass = bool(all(item["clean_gate_pass"] for item in replica_reports))
    final_predictions: pd.DataFrame | None = None
    if clean_pass:
        final_predictions = fit_full_predictions(bundle, feature_cache, replica_oofs[0])
        if len(final_predictions) != 4940 or not final_predictions["id"].equals(test["id"]):
            raise RuntimeError("C114 full-data output contract failed")
        final_predictions.to_csv(run_dir / "predictions.csv", index=False)
    mean_parent = float(np.mean([item["mean_parent_r2"] for item in replica_reports]))
    mean_candidate = float(np.mean([item["mean_candidate_r2"] for item in replica_reports]))
    report: dict[str, Any] = {"schema_version": "ppp.round2.c114.fixed-spline-gam-residual-portfolio.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "C050 exact source replay; C112 parity control path", "official_only": True, "official_inputs": context["inputs"], "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "submission": False, "parent_replay_oof_max_abs": oof_max, "parent_replay_test_max_abs": test_max, "replicas": replica_reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "all_replica_clean_gate_pass": clean_pass, "full_data_fit": clean_pass, "local_eval_eligible": False, "complete_output_rows": int(len(final_predictions)) if final_predictions is not None else 0, "elapsed_seconds": float(time.time() - started)}
    source_paths = {"runner": root / "tools/round2_c114_fixed_spline_gam_residual_portfolio.py", "parent_control": root / "tools/round2_c112_c050_parent_parity_control.py", "reference": root / "tools/initial_reference_pipeline.py", "mixed_parent": root / "tools/round2_mixed_candidate_v7.py", "fixed_features": root / "tools/round2_c063_egb_endpoint_conjugation_residual.py", "compact_features": root / "tools/round2_c076_eps_paired_charge_polarizability_residual.py", "metric_plumbing": root / "tools/round2_eea_cross_target_oof_residual_stack.py", "ei_route": root / "tools/round2_ei_scaffold_abstaining_gap_identity_v4_portable.py", "eea_route": root / "tools/round2_eea_scaffold_abstaining_gap_identity_v7_portable.py"}
    report["source_hashes"] = {name: sha256_file(path) for name, path in source_paths.items()}
    if replica_oofs:
        pd.concat(replica_oofs[:1], ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    decision = "candidate_pending_notebook_parity" if clean_pass else "rejected_full_candidate_gate"
    report["decision"] = decision
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"active_targets": list(ACTIVE), "spline_degree": 2, "spline_knots": 4, "residual_weight": RESIDUAL_WEIGHT, "replica_seeds": list(SEEDS), "model": {"eps": "Ridge(alpha=25)", "nc": "Ridge(alpha=50)", "ei": "HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000)", "tg": "Ridge(alpha=50)"}, "features": "official SMILES endpoint, physicochemical, and charge descriptors", "full_data_fit_only_after_clean_gate": True, "local_eval_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{decision}**. Mean parent R2 `{mean_parent:.12f}`; mean candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. Parent parity maxima are `{oof_max:.16g}`/`{test_max:.16g}`. Full-data fit occurred only if all clean OOF gates passed; local_eval was not read.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{digest}  SOURCE tools/{name}.py" for name, digest in report["source_hashes"].items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": decision, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "parent_oof_max_abs": oof_max, "parent_test_max_abs": test_max, "clean_gate_pass": clean_pass, "complete_output_rows": report["complete_output_rows"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
