#!/usr/bin/env python3
"""C098: exact-C050 target-routed physical QSPR residual candidate.

This is a clean official-only diagnostic.  The C050 parent and its special Ei/
Eea routes are rebuilt from source.  Only Nc and EPS receive the new residual;
the other five target predictions remain the rebuilt parent predictions.
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
from rdkit import RDLogger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c076_eps_paired_charge_polarizability_residual as paired_features
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_mixed_candidate_v7 as mixed


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
CHANGED = {"nc", "eps"}
COUNTERPART = {"nc": "eps", "eps": "nc"}
RESIDUAL_WEIGHT = 0.20
RIDGE_ALPHA = 30.0
SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def model() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def parent_bundle(root: Path, data_dir: Path) -> dict[str, Any]:
    train, test, archive, inputs = reference.load_inputs(data_dir)
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": SEED, "folds": 5, "mixed_candidate": True, "special_targets": ["ei", "eea"]})
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, int(config["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(config["morgan_bits"])),
        reference.text_matrix(keys, int(config["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(config["morgan_bits"]))
    parent_detail, parent_oof, parent_report = reference.fit_targets(
        pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
    )
    test_detail = parent_detail[["id", "target_type", "model_prediction"]].copy()
    target_info: dict[str, dict[str, Any]] = {}
    special_reports: dict[str, Any] = {}
    for target in TARGETS:
        if target in ("ei", "eea"):
            special_oof, special_test, special_report = mixed.specialized_target(
                target, pooled, test, keys, dense_base, cross_values, cross_available,
                sparse_parts, fingerprints, config,
            )
            canonical = special_oof["canonical"].to_numpy(object)
            y = special_oof["target"].to_numpy(float)
            parent = special_oof["candidate"].to_numpy(float)
            folds = special_oof["outer_fold"].to_numpy(int)
            replacement = special_test.set_index("id")["target"]
            mask = test_detail["target_type"].to_numpy(object) == target
            test_detail.loc[mask, "model_prediction"] = test_detail.loc[mask, "id"].map(replacement).astype(float).to_numpy()
            special_reports[target] = special_report
        else:
            rows = parent_oof[parent_oof["target_type"] == target].reset_index(drop=True)
            canonical = rows["canonical"].to_numpy(object)
            y = rows["target"].to_numpy(float)
            parent = rows["prediction"].to_numpy(float)
            folds = np.full(len(rows), -1, dtype=np.int64)
            for fold, (_, validation) in enumerate(
                __import__("sklearn.model_selection", fromlist=["KFold"]).KFold(
                    n_splits=5, shuffle=True, random_state=SEED
                ).split(np.arange(len(rows)))
            ):
                folds[validation] = fold
        indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
        target_info[target] = {
            "canonical": canonical,
            "y": y,
            "parent": parent,
            "folds": folds,
            "indices": indices,
            "groups": groups,
            "scaffolds": scaffolds,
        }
    return {
        "train": train,
        "test": test,
        "archive": archive,
        "raw_labels": raw_labels,
        "pooled": pooled,
        "inputs": inputs,
        "keys": keys,
        "key_to_index": key_to_index,
        "molecules": molecules,
        "cross_values": cross_values,
        "cross_available": cross_available,
        "fingerprints": fingerprints,
        "target_info": target_info,
        "test_detail": test_detail,
        "parent_report": parent_report,
        "special_reports": special_reports,
    }


def target_features(bundle: dict[str, Any], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    info = bundle["target_info"][target]
    counterpart = COUNTERPART[target]
    target_keys = sorted(set(info["canonical"]) | set(bundle["test"].loc[bundle["test"]["target_type"] == target, "canonical"]))
    global_indices = np.asarray([bundle["key_to_index"][value] for value in target_keys], dtype=np.int64)
    base, base_names = fixed_features.fixed_features(bundle["molecules"], global_indices.tolist())
    physics, physics_names = paired_features.physics_features(bundle["molecules"], global_indices.tolist())
    charge, charge_names = paired_features.charge_features(bundle["molecules"], global_indices.tolist())
    counterpart_index = reference.TARGETS.index(counterpart)
    values = bundle["cross_values"][global_indices, counterpart_index]
    available = bundle["cross_available"][global_indices, counterpart_index]
    paired = np.column_stack([values, available, np.square(values)])
    paired[~np.isfinite(paired)] = np.nan
    names = base_names + physics_names + charge_names + [f"official_{counterpart}_value", f"official_{counterpart}_available", f"official_{counterpart}_value_squared"]
    matrix = np.hstack([base, physics, charge, paired]).astype(np.float64, copy=False)
    feature_row = {value: row for row, value in enumerate(target_keys)}
    train_rows = np.asarray([feature_row[value] for value in info["canonical"]], dtype=np.int64)
    test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    test_rows = np.asarray([feature_row[value] for value in test_frame["canonical"]], dtype=np.int64)
    pair_values_train = values[train_rows]
    pair_values_test = values[test_rows]
    return matrix, train_rows, test_rows, names, pair_values_train, pair_values_test


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, Any], float]:
    panels: dict[str, Any] = {}
    values: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected))
        item: dict[str, Any] = {"rows": count, "delta_r2": 0.0, "status": "inapplicable"}
        if count >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"delta_r2": delta, "status": "evaluable"})
            values.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        add(name, selected)
    for name in sorted(set(scaffolds)):
        add(f"scaffold_{name}", scaffolds == name, minimum=10)
    add("counterpart_available", np.isfinite(bundle_pair_placeholder), minimum=5)
    return panels, float(min(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    bundle = parent_bundle(root, (root / args.data_dir).resolve())
    target_reports: dict[str, Any] = {}
    raw_test_predictions = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    oof_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    feature_counts: dict[str, int] = {}
    for target in TARGETS:
        info = bundle["target_info"][target]
        y = info["y"]
        parent = info["parent"]
        folds = info["folds"]
        candidate = parent.copy()
        similarity = np.full(len(y), np.nan, dtype=np.float64)
        pair_train = np.full(len(y), np.nan, dtype=np.float64)
        pair_test = np.array([], dtype=np.float64)
        test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        if target in CHANGED:
            matrix, train_rows, test_rows, names, pair_train, pair_test = target_features(bundle, target)
            feature_counts[target] = int(matrix.shape[1])
            residual = y - parent
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                fitted = model()
                fitted.fit(matrix[train_rows[training]], residual[training])
                correction = RESIDUAL_WEIGHT * fitted.predict(matrix[train_rows[validation]])
                supported = np.isfinite(pair_train[validation])
                candidate[validation[supported]] = parent[validation[supported]] + correction[supported]
                global_validation = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][validation]], dtype=np.int64)
                global_training = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][training]], dtype=np.int64)
                similarity[validation] = fixed_features.nearest_similarity(bundle["fingerprints"], global_validation, global_training)
            fitted = model()
            fitted.fit(matrix[train_rows], residual)
            test_correction = RESIDUAL_WEIGHT * fitted.predict(matrix[test_rows])
            test_candidate = bundle["test_detail"].loc[bundle["test_detail"]["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float).copy()
            supported_test = np.isfinite(pair_test)
            test_candidate[supported_test] += test_correction[supported_test]
            replacement = pd.Series(test_candidate, index=test_frame["id"].to_numpy())
            mask = raw_test_predictions["target_type"].to_numpy(object) == target
            raw_test_predictions.loc[mask, "model_prediction"] = raw_test_predictions.loc[mask, "id"].map(replacement).astype(float).to_numpy()
        else:
            feature_counts[target] = 0
            similarity[:] = np.nan
        groups = info["groups"]
        scaffolds = info["scaffolds"]
        if target in CHANGED:
            panels, minimum_panel = panel_report_with_pair(y, parent, candidate, groups, scaffolds, similarity, np.isfinite(pair_train))
        else:
            panels, minimum_panel = {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}}, 0.0
        fold_rows = []
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation]))})
        delta = float(r2_score(y, candidate) - r2_score(y, parent))
        lower = bootstrap_lower(y, parent, candidate, groups) if target in CHANGED else 0.0
        positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        target_pass = bool(target not in CHANGED or (delta >= 0.005 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0))
        target_reports[target] = {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "folds": fold_rows, "panels": panels, "feature_count": feature_counts[target], "pair_rows": int(np.sum(np.isfinite(pair_train))) if target in CHANGED else 0, "pass": target_pass, "unchanged_parent": target not in CHANGED}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "group": groups, "scaffold": scaffolds, "fold": folds, "pair_available": np.isfinite(pair_train)}))
        if target in CHANGED:
            parent_test = bundle["test_detail"].loc[bundle["test_detail"]["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float)
            candidate_test = raw_test_predictions.loc[raw_test_predictions["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float)
            component_parts.append(pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": target, "parent_prediction": parent_test, "candidate_prediction": candidate_test, "pair_available": np.isfinite(pair_test)}))
    raw_detail, override_report = reference.apply_official_overrides(raw_test_predictions, bundle["test"], bundle["raw_labels"])
    submission = raw_detail[["id", "target"]].copy()
    if len(submission) != len(bundle["test"]) or not submission["id"].equals(bundle["test"]["id"]):
        raise RuntimeError("C098 output IDs/order mismatch")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C098 output contains duplicate or non-finite values")
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(target_reports[target]["delta_r2"] for target in TARGETS))
    complete_pass = bool(mean_candidate > mean_parent and max_loss >= -0.003 and all(target_reports[target]["pass"] for target in CHANGED))
    oof = pd.concat(oof_parts, ignore_index=True)
    components = pd.concat(component_parts, ignore_index=True)
    submission.to_csv(run_dir / "predictions.csv", index=False)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    components.to_csv(run_dir / "component_predictions.csv", index=False)
    script_path = root / "tools" / "round2_c098_target_routed_qspr_full.py"
    source_paths = {
        "script": script_path,
        "reference": root / "tools" / "initial_reference_pipeline.py",
        "mixed_parent_route": root / "tools" / "round2_mixed_candidate_v7.py",
        "fixed_features": root / "tools" / "round2_c063_egb_endpoint_conjugation_residual.py",
        "paired_features": root / "tools" / "round2_c076_eps_paired_charge_polarizability_residual.py",
        "metric_plumbing": root / "tools" / "round2_eea_cross_target_oof_residual_stack.py",
    }
    report = {
        "schema_version": "ppp.round2.c098.target-routed-qspr-full.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "C050 rebuilt from official sources in memory; Ei/Eea use C050 special routes",
        "official_inputs": bundle["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "prior_prediction_input": False,
        "pretrained_weights": False,
        "target_reports": target_reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(submission)),
        "complete_output_order_pass": True,
        "complete_candidate_gate_pass": complete_pass,
        "parent_replay_status": "fresh_process_required_before_promotion_or_local_eval",
        "official_override_report": override_report,
        "feature_counts": feature_counts,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "elapsed_seconds": float(time.time() - started),
        "decision": "candidate_pending_fresh_replay" if complete_pass else "rejected_full_candidate_gate",
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "changed_targets": sorted(CHANGED), "counterparts": COUNTERPART, "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": RIDGE_ALPHA, "model": "median-impute-standardize-ridge", "parent": "C050 exact source rebuild", "fresh_process_parent_replay": "required"})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent R2 `{mean_parent:.12f}`; candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. The candidate is official-only and has not been local_eval-scored.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{digest}  SOURCE tools/{name}.py" for name, digest in report["source_hashes"].items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "decision": report["decision"]}, sort_keys=True), flush=True)


def panel_report_with_pair(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray, pair_available: np.ndarray) -> tuple[dict[str, Any], float]:
    panels: dict[str, Any] = {}
    values: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected))
        item: dict[str, Any] = {"rows": count, "delta_r2": 0.0, "status": "inapplicable"}
        if count >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"delta_r2": delta, "status": "evaluable"})
            values.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    add("counterpart_available", pair_available)
    add("counterpart_missing", ~pair_available)
    for name, selected in {"similarity_lt_0.30": similarity < 0.30, "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50), "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70), "similarity_ge_0.70": similarity >= 0.70}.items():
        add(name, selected)
    for name in sorted(set(scaffolds)):
        add(f"scaffold_{name}", scaffolds == name, minimum=10)
    return panels, float(min(values)) if values else 0.0


if __name__ == "__main__":
    main()
