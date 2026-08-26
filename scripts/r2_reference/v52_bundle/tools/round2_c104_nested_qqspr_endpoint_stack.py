#!/usr/bin/env python3
"""C104: regenerated C098 paired-QSPR plus endpoint-path residual stack."""

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
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c098_target_routed_qspr_full as c098
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
CHANGED = ("eps", "nc", "tg")
COUNTERPART = {"eps": "nc", "nc": "eps"}
SEED = 2026
QSPR_WEIGHT = 0.20
PATH_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def model() -> Any:
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(with_mean=False), Ridge(alpha=30.0))


def folds_for(groups: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import GroupKFold
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def panels(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, Any], float]:
    report: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        rows = int(np.sum(selected))
        item: dict[str, Any] = {"rows": rows, "delta_r2": 0.0, "status": "inapplicable"}
        if rows >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"delta_r2": delta, "status": "evaluable"})
            deltas.append(delta)
        report[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    add("similarity_lt_0.30", similarity < 0.30)
    add("similarity_0.30_0.50", (similarity >= 0.30) & (similarity < 0.50))
    add("similarity_0.50_0.70", (similarity >= 0.50) & (similarity < 0.70))
    add("similarity_ge_0.70", similarity >= 0.70)
    for scaffold in sorted(set(scaffolds)):
        add(f"scaffold_{scaffold}", scaffolds == scaffold, minimum=10)
    return report, float(min(deltas)) if deltas else 0.0


def endpoint_matrix(root: Path, bundle: dict[str, Any], feature_keys: list[str]) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    round1_tools = root.parent / "Polymer Prediction Challenge" / "tools"
    sys.path.insert(0, str(round1_tools))
    import polymer_official_train_eval_loop as round1_engine
    indices = [bundle["key_to_index"][key] for key in feature_keys]
    molecules = [bundle["molecules"][index] for index in indices]
    path, path_report = round1_engine.endpoint_path_ngram_matrix(molecules, n_features=4096, max_bonds=8)
    fixed, _ = fixed_features.fixed_features(molecules, list(range(len(molecules))))
    physical, _ = c098.paired_features.physics_features(molecules, list(range(len(molecules))))
    fixed[~np.isfinite(fixed)] = 0.0
    physical[~np.isfinite(physical)] = 0.0
    combined = sparse.hstack([path, sparse.csr_matrix(np.hstack([fixed, physical]), dtype=np.float32)], format="csr")
    return combined, {"path": path_report, "total_features": int(combined.shape[1])}


def counterpart_groups(bundle: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for target in TARGETS:
        output[target] = {}
        frame = bundle["pooled"][bundle["pooled"]["target_type"] == target]
        for value in frame["canonical"].astype(str).unique():
            output[target][value] = plumbing.no_stereo(value)
    return output


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
    bundle = c098.parent_bundle(root, (root / args.data_dir).resolve())
    feature_keys = sorted({value for target in CHANGED for value in bundle["target_info"][target]["canonical"]} | set(bundle["test"].loc[bundle["test"]["target_type"].isin(CHANGED), "canonical"]))
    path_features, path_report = endpoint_matrix(root, bundle, feature_keys)
    feature_row = {value: row for row, value in enumerate(feature_keys)}
    cp_groups = counterpart_groups(bundle)
    raw_test = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = bundle["target_info"][target]
        y = info["y"]
        parent = info["parent"]
        groups = info["groups"]
        scaffolds = info["scaffolds"]
        candidate = parent.copy()
        similarity = np.full(len(y), np.nan, dtype=np.float64)
        folds = np.full(len(y), -1, dtype=np.int64)
        pair_supported = np.zeros(len(y), dtype=bool)
        qspr_correction = np.zeros(len(y), dtype=np.float64)
        path_correction = np.zeros(len(y), dtype=np.float64)
        if target in CHANGED:
            folds = folds_for(groups)
            path_rows = np.asarray([feature_row[value] for value in info["canonical"]], dtype=np.int64)
            if target in COUNTERPART:
                qspr_matrix, qspr_train_rows, qspr_test_rows, _, pair_train, pair_test = c098.target_features(bundle, target)
                qspr_matrix = np.asarray(qspr_matrix, dtype=np.float64)
                pair_cols = np.arange(qspr_matrix.shape[1] - 3, qspr_matrix.shape[1])
                counterpart_target = COUNTERPART[target]
                counterpart_group = cp_groups[counterpart_target]
            else:
                qspr_matrix = None
                qspr_train_rows = qspr_test_rows = pair_train = pair_test = None
                pair_cols = np.asarray([], dtype=np.int64)
                counterpart_group = {}
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                forbidden = set(groups[validation])
                path_model = model()
                path_model.fit(path_features[path_rows[training]], (y - parent)[training])
                path_correction[validation] = path_model.predict(path_features[path_rows[validation]])
                if target in COUNTERPART:
                    train_canonical = info["canonical"][training]
                    val_canonical = info["canonical"][validation]
                    fit_matrix = qspr_matrix[qspr_train_rows[training]].copy()
                    val_matrix = qspr_matrix[qspr_train_rows[validation]].copy()
                    fit_bad = np.asarray([counterpart_group.get(str(value)) in forbidden for value in train_canonical], dtype=bool)
                    val_good = np.asarray([np.isfinite(pair_train[index]) and counterpart_group.get(str(value)) not in forbidden for index, value in zip(validation, val_canonical, strict=True)], dtype=bool)
                    if np.any(fit_bad):
                        fit_matrix[np.ix_(fit_bad, pair_cols)] = np.nan
                    q_model = model()
                    q_model.fit(fit_matrix, (y - parent)[training])
                    qspr_correction[validation] = q_model.predict(val_matrix)
                    pair_supported[validation] = val_good
                else:
                    pair_supported[validation] = True
                candidate[validation] = parent[validation] + PATH_WEIGHT * path_correction[validation]
                if target in COUNTERPART:
                    candidate[validation[pair_supported[validation]]] += QSPR_WEIGHT * qspr_correction[validation[pair_supported[validation]]]
                global_validation = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][validation]], dtype=np.int64)
                global_training = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][training]], dtype=np.int64)
                similarity[validation] = fixed_features.nearest_similarity(bundle["fingerprints"], global_validation, global_training)
            test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            test_path_rows = np.asarray([feature_row[value] for value in test_frame["canonical"]], dtype=np.int64)
            path_model = model()
            path_model.fit(path_features[path_rows], y - parent)
            test_prediction = raw_test.loc[raw_test["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float) + PATH_WEIGHT * path_model.predict(path_features[test_path_rows])
            if target in COUNTERPART:
                q_model = model()
                q_model.fit(qspr_matrix[qspr_train_rows], y - parent)
                q_prediction = q_model.predict(qspr_matrix[qspr_test_rows])
                supported_test = np.isfinite(pair_test)
                test_prediction[supported_test] += QSPR_WEIGHT * q_prediction[supported_test]
            mask = raw_test["target_type"].to_numpy(object) == target
            raw_test.loc[mask, "model_prediction"] = test_prediction
            component_parts.append(pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": target, "candidate_prediction": test_prediction, "pair_supported": np.isfinite(pair_test) if target in COUNTERPART else np.ones(len(test_prediction), dtype=bool)}))
            fold_rows = []
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation]))})
            delta = float(r2_score(y, candidate) - r2_score(y, parent))
            panel_report, minimum_panel = panels(y, parent, candidate, groups, scaffolds, similarity)
            lower = bootstrap_lower(y, parent, candidate, groups)
            positive = int(sum(row["delta_r2"] > 0 for row in fold_rows))
            target_reports[target] = {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "folds": fold_rows, "panels": panel_report, "pair_supported_rows": int(np.sum(pair_supported)), "pass": bool(delta >= 0.01 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0), "unchanged_parent": False}
        else:
            target_reports[target] = {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "folds": [], "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}}, "pair_supported_rows": 0, "pass": True, "unchanged_parent": True}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "pair_supported": pair_supported}))
    detail, override_report = reference.apply_official_overrides(raw_test, bundle["test"], bundle["raw_labels"])
    submission = detail[["id", "target"]].copy()
    if len(submission) != len(bundle["test"]) or not submission["id"].equals(bundle["test"]["id"]):
        raise RuntimeError("C104 output ID/order mismatch")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C104 output non-finite or duplicate")
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(target_reports[target]["delta_r2"] for target in TARGETS))
    complete_pass = bool(mean_candidate > mean_parent and max_loss >= -0.003 and all(target_reports[target]["pass"] for target in CHANGED))
    submission.to_csv(run_dir / "predictions.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.concat(component_parts, ignore_index=True).to_csv(run_dir / "component_predictions.csv", index=False)
    source_paths = {"script": root / "tools" / "round2_c104_nested_qqspr_endpoint_stack.py", "c098": root / "tools" / "round2_c098_target_routed_qspr_full.py", "reference": root / "tools" / "initial_reference_pipeline.py", "fixed_features": root / "tools" / "round2_c063_egb_endpoint_conjugation_residual.py", "plumbing": root / "tools" / "round2_eea_cross_target_oof_residual_stack.py", "round1_engine": root.parent / "Polymer Prediction Challenge" / "tools" / "polymer_official_train_eval_loop.py"}
    report = {"schema_version": "ppp.round2.c104.nested-qspr-endpoint-stack.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "C050 source rebuild via C098 parent bundle; no saved predictions", "official_inputs": bundle["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "prior_prediction_input": False, "pretrained_weights": False, "target_reports": target_reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": max_loss, "complete_output_rows": int(len(submission)), "complete_output_order_pass": True, "complete_candidate_gate_pass": complete_pass, "parent_replay_required": True, "parent_replay_oof_max_abs": None, "parent_replay_test_max_abs": None, "official_override_report": override_report, "feature_report": path_report, "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()}, "elapsed_seconds": float(time.time() - started), "decision": "candidate_pending_fresh_replay" if complete_pass else "rejected_full_candidate_gate"}
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "changed_targets": list(CHANGED), "qspr_weight": QSPR_WEIGHT, "endpoint_path_weight": PATH_WEIGHT, "outer": "canonical_no_stereo GroupKFold(5)", "heldout_counterpart_group_mask": True, "external_label_file_read": False, "local_eval_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent R2 `{mean_parent:.12f}`; candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, path in source_paths.items():
        manifest.append(f"{sha256_file(path)}  ../../../tools/{path.name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "target_deltas": {target: target_reports[target]["delta_r2"] for target in TARGETS}}, sort_keys=True))


if __name__ == "__main__":
    main()
