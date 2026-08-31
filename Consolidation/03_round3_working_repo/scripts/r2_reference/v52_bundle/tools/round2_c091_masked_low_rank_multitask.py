#!/usr/bin/env python3
"""Research-only masked low-rank multi-output residual diagnostic."""

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
from rdkit import DataStructs, RDLogger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as panels
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")


TARGETS = tuple(reference.TARGETS)
SEED = 2026
ALPHA = 30.0
RANK = 3
WEIGHT = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    return result


def finite_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, prediction))


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups); members = {group: np.flatnonzero(groups == group) for group in unique}; rng = np.random.default_rng(SEED); values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True); rows = np.concatenate([members[group] for group in selected])
        if not np.isclose(np.var(y[rows]), 0.0): values.append(finite_r2(y[rows], candidate[rows]) - finite_r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def fit_low_rank_heads(x: np.ndarray, pooled: pd.DataFrame, parent: np.ndarray, target_codes: np.ndarray, row_indices: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.zeros((len(TARGETS), x.shape[1]), dtype=np.float64)
    intercepts = np.zeros(len(TARGETS), dtype=np.float64)
    for code in range(len(TARGETS)):
        selected = train_mask & (target_codes == code)
        if int(np.sum(selected)) < 8:
            continue
        residual = pooled["target"].to_numpy(float)[selected] - parent[selected]
        mean = float(np.mean(residual)); std = float(np.std(residual)); std = std if std > 1.0e-8 else 1.0
        fitted = Ridge(alpha=ALPHA).fit(x[row_indices[selected]], (residual - mean) / std)
        coefficients[code] = fitted.coef_ * std
        intercepts[code] = float(fitted.intercept_ * std + mean)
    rank = min(RANK, coefficients.shape[0], coefficients.shape[1])
    left, singular, right = np.linalg.svd(coefficients, full_matrices=False)
    low_rank = (left[:, :rank] * singular[:rank]) @ right[:rank, :]
    return low_rank, intercepts


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--data-dir", default="ppp-round-2"); parser.add_argument("--run-dir", required=True); args = parser.parse_args()
    root = Path(args.root).resolve(); run_dir = Path(args.run_dir); run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time(); train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve()); _, pooled = reference.build_label_pool(train, archive); pooled = pooled.reset_index(drop=True); keys = sorted(set(pooled["canonical"]) | set(test["canonical"])); key_to_index = {key: index for index, key in enumerate(keys)}; molecules = reference.build_molecules(keys); descriptor, descriptor_names = reference.descriptor_matrix(molecules); physical, physical_names = reference.physical_matrix(molecules, keys); raw_features = np.hstack([descriptor, physical]).astype(np.float64, copy=False); imputer = SimpleImputer(strategy="median", keep_empty_features=True); scaler = StandardScaler(); features = scaler.fit_transform(imputer.fit_transform(raw_features)); pooled_targets = pooled["target_type"].astype(str).to_numpy(object); target_codes = np.asarray([TARGETS.index(value) for value in pooled_targets], dtype=np.int64); rows = np.asarray([key_to_index[value] for value in pooled["canonical"]], dtype=np.int64); groups = np.asarray([plumbing.no_stereo(value) for value in pooled["canonical"]], dtype=object); scaffolds_all = np.asarray([plumbing.scaffold(value) for value in pooled["canonical"]], dtype=object); folds_all = folds_for(groups)
    parent_dir = root / "experiments" / "CLEAN_OFFICIAL_ONLY" / "R2-C050-20260803-2130-mixed-c001-gap-components-v7"; parent_oof = pd.read_csv(parent_dir / "oof_predictions.csv"); parent_oof = parent_oof[["target_type", "canonical", "target", "candidate_prediction"]].rename(columns={"candidate_prediction": "parent_prediction"}); pooled = pooled.merge(parent_oof, on=["target_type", "canonical", "target"], how="left", validate="one_to_one"); parent_all = pooled["parent_prediction"].to_numpy(float); fingerprints = [reference.morgan_bits(molecules, 2, 4096)[index] for index in range(len(molecules))]
    if not np.isfinite(parent_all).all(): raise RuntimeError("frozen incumbent OOF alignment failed")
    parent_test = pd.read_csv(parent_dir / "predictions.csv");
    if not np.array_equal(parent_test["id"].to_numpy(int), test["id"].to_numpy(int)): raise RuntimeError("frozen incumbent test order mismatch")
    candidate_all = np.full(len(pooled), np.nan, dtype=np.float64); fold_meta: list[dict[str, Any]] = []
    for fold in range(5):
        validation = folds_all == fold; training = ~validation; low_rank, intercepts = fit_low_rank_heads(features, pooled, parent_all, target_codes, rows, training)
        for code, target in enumerate(TARGETS):
            selected = validation & (target_codes == code)
            candidate_all[selected] = parent_all[selected] + WEIGHT * (intercepts[code] + features[rows[selected]] @ low_rank[code])
        fold_meta.append({"fold": fold, "train_rows": int(np.sum(training)), "validation_rows": int(np.sum(validation)), "rank": RANK})
    if not np.isfinite(candidate_all).all(): raise RuntimeError("non-finite low-rank OOF candidate")
    all_train = np.ones(len(pooled), dtype=bool); low_rank, intercepts = fit_low_rank_heads(features, pooled, parent_all, target_codes, rows, all_train); test_targets = test["target_type"].astype(str).to_numpy(object); test_codes = np.asarray([TARGETS.index(value) for value in test_targets], dtype=np.int64); test_rows = np.asarray([key_to_index[value] for value in test["canonical"]], dtype=np.int64); test_candidates = parent_test["target"].to_numpy(float).copy();
    for code in range(len(TARGETS)):
        selected = test_codes == code; test_candidates[selected] += WEIGHT * (intercepts[code] + features[test_rows[selected]] @ low_rank[code])
    if len(test_candidates) != 4940 or not np.isfinite(test_candidates).all(): raise RuntimeError("full test output contract failed")
    target_reports: dict[str, Any] = {}; component_rows: list[dict[str, Any]] = []
    for code, target in enumerate(TARGETS):
        selected = target_codes == code; y = pooled.loc[selected, "target"].to_numpy(float); parent = parent_all[selected]; candidate = candidate_all[selected]; local_groups = groups[selected]; local_scaffolds = scaffolds_all[selected]; local_folds = folds_all[selected]; similarity = np.full(len(y), np.nan, dtype=np.float64); local_rows = rows[selected]
        for fold in range(5):
            valid = np.flatnonzero(local_folds == fold); train_rows = np.flatnonzero(local_folds != fold); train_fps = [fingerprints[int(local_rows[index])] for index in train_rows];
            for pos in valid: similarity[pos] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(local_rows[pos])], train_fps))
        fold_rows = []
        for fold in range(5):
            part = local_folds == fold; fold_rows.append({"fold": fold, "rows": int(np.sum(part)), "parent_r2": finite_r2(y[part], parent[part]), "candidate_r2": finite_r2(y[part], candidate[part]), "delta_r2": finite_r2(y[part], candidate[part]) - finite_r2(y[part], parent[part])})
        panel_report, minimum_panel = panels.panel_report(y, parent, candidate, local_scaffolds, similarity); delta = finite_r2(y, candidate) - finite_r2(y, parent); lower = bootstrap_lower(y, parent, candidate, local_groups); gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0.0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel is not None and minimum_panel >= 0.0}; target_reports[target] = {"rows": int(len(y)), "parent_r2": finite_r2(y, parent), "candidate_r2": finite_r2(y, candidate), "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0.0 for row in fold_rows)), "folds": fold_rows, "group_bootstrap_lower": lower, "panels": panel_report, "minimum_panel_delta": minimum_panel, "gates": gates, "decision": "pass_component_gate" if all(gates.values()) else "rejected_component_gate"}; component_rows.extend({"target": target, "canonical": str(value), "parent": float(parent[index]), "candidate": float(candidate[index]), "fold": int(local_folds[index])} for index, value in enumerate(pooled.loc[selected, "canonical"].astype(str)))
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS])); candidate_mean = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS])); mean_delta = candidate_mean - parent_mean; max_loss = min(float(target_reports[target]["delta_r2"]) for target in TARGETS); passing = [target for target in TARGETS if all(target_reports[target]["gates"].values())]; full_gates = {"mean_gain_pass": mean_delta >= 0.002, "maximum_target_loss_pass": max_loss >= -0.003, "rows_pass": len(test_candidates) == 4940, "finite_pass": bool(np.isfinite(test_candidates).all())}; decision = "pass_research_gate" if passing and all(full_gates.values()) else "rejected_component_gate"
    pd.DataFrame({"id": test["id"].astype(int), "target": test_candidates, "target_type": test_targets, "parent_prediction": parent_test["target"].to_numpy(float)}).to_csv(run_dir / "candidate_predictions.csv", index=False); pd.DataFrame({"canonical": pooled["canonical"].astype(str), "target_type": pooled_targets, "target": pooled["target"].to_numpy(float), "parent": parent_all, "candidate": candidate_all, "group": groups, "outer_fold": folds_all}).to_csv(run_dir / "oof_predictions.csv", index=False)
    source_names = ("round2_c091_masked_low_rank_multitask.py", "initial_reference_pipeline.py", "round2_c063_egb_endpoint_conjugation_residual.py", "round2_eea_cross_target_oof_residual_stack.py"); source_hashes = {name: sha256_file(root / "tools" / name) for name in source_names}; report = {"schema_version": "ppp.round2.c091.masked-low-rank-multitask.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "parent": "frozen C050-v7 OOF/test read-only incumbent", "official_inputs": inputs, "official_only": True, "research_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "same_row_label_lookup": False, "parent_prediction_input": True, "targets": target_reports, "folds": fold_meta, "rank": RANK, "alpha": ALPHA, "residual_weight": WEIGHT, "parent_mean_r2": parent_mean, "candidate_mean_r2": candidate_mean, "mean_delta_r2": mean_delta, "minimum_target_delta_r2": max_loss, "passing_targets": passing, "full_gates": full_gates, "decision": decision, "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "oof": int(len(pooled))}, "source_hashes": source_hashes, "elapsed_seconds": time.time() - started}; write_json(run_dir / "metrics.json", report); write_json(run_dir / "config.json", {"schema_version": report["schema_version"], "seed": SEED, "targets": TARGETS, "rank": RANK, "ridge_alpha": ALPHA, "residual_weight": WEIGHT, "feature_count": int(features.shape[1]), "features": "official descriptor + physical matrix, global unsupervised scaling", "outer": "shared canonical no-stereo GroupKFold(5)", "research_only": True, "parent_prediction_input": True, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False}); (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8"); (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8"); (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{decision}**. Frozen incumbent mean `{parent_mean:.12f}`, low-rank mean `{candidate_mean:.12f}`, delta `{mean_delta:+.12f}`. Research-only diagnostic; no local_eval, upload, submission, or final selection.\n", encoding="utf-8"); manifest_paths = [path for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]; lines = [f"{sha256_file(path)}  {path.relative_to(run_dir)}" for path in manifest_paths]; lines.extend(f"{digest}  SOURCE tools/{name}" for name, digest in source_hashes.items()); (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(json.dumps({"experiment_id": run_dir.name, "decision": decision, "parent_mean_r2": parent_mean, "candidate_mean_r2": candidate_mean, "mean_delta_r2": mean_delta, "minimum_target_delta_r2": max_loss, "passing_targets": passing, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
