#!/usr/bin/env python3
"""C113: fixed Tanimoto-landmark residual portfolio.

The exact C050 parent is rebuilt through the passing C112 parity control.  A
fold-local Morgan/Tanimoto landmark ridge is then evaluated only for EPS, Nc,
and Ei; all other properties retain the parent.
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
from rdkit import DataStructs, RDLogger
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c112_c050_parent_parity_control as parent_control
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_mixed_candidate_v7 as mixed


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE = ("eps", "nc", "ei")
SEEDS = (2026, 2027, 2028)
LANDMARKS = 256
ALPHA = 10.0
RESIDUAL_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def kernel(left: list[Any], right: list[Any]) -> np.ndarray:
    return np.asarray([DataStructs.BulkTanimotoSimilarity(fp, right) for fp in left], dtype=np.float64)


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


def panels(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> tuple[dict[str, Any], float]:
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


def special_ei_folds(
    pooled: pd.DataFrame,
    test: pd.DataFrame,
    keys: list[str],
    dense: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    config: dict[str, Any],
) -> dict[str, int]:
    special_oof, _, _ = mixed.specialized_target(
        "ei", pooled, test, keys, dense, cross_values, cross_available, sparse_parts, fingerprints, config
    )
    return {str(key): int(fold) for key, fold in zip(special_oof["canonical"], special_oof["outer_fold"], strict=True)}


def run_replica(
    seed: int,
    parent_predictions: pd.DataFrame,
    parent_oof: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    pooled: pd.DataFrame,
    molecules: list[Any],
    fingerprints: list[Any],
    key_to_index: dict[str, int],
    ei_fold_map: dict[str, int],
    raw_labels: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    candidate_detail = parent_predictions.merge(test[["id", "target_type"]], on="id", how="left", validate="one_to_one")
    candidate_detail = candidate_detail[["id", "target_type", "target"]].rename(columns={"target": "model_prediction"})
    oof_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}

    for target in TARGETS:
        rows = parent_oof[parent_oof["target_type"] == target].reset_index(drop=True)
        y = rows["target"].to_numpy(float)
        parent = rows["parent_prediction"].to_numpy(float)
        canonical = rows["canonical"].astype(str).to_numpy(object)
        groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
        if target == "ei":
            folds = np.asarray([ei_fold_map[value] for value in canonical], dtype=np.int64)
        else:
            folds = np.full(len(rows), -1, dtype=np.int64)
            for fold, (_, validation) in enumerate(KFold(n_splits=5, shuffle=True, random_state=2026).split(np.arange(len(rows)))):
                folds[validation] = fold
        candidate = parent.copy()
        similarity = np.full(len(rows), np.nan, dtype=np.float64)
        test_rows = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        target_parent_test = parent_predictions[parent_predictions["id"].isin(test_rows["id"])].set_index("id").loc[test_rows["id"], "target"].to_numpy(float)
        target_candidate_test = target_parent_test.copy()
        feature_count = 0
        if target in ACTIVE:
            feature_count = LANDMARKS
            residual = y - parent
            global_train = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                rng = np.random.default_rng(seed + fold)
                landmark = np.sort(rng.choice(training, size=min(LANDMARKS, len(training)), replace=False))
                landmark_fps = [fingerprints[int(global_train[index])] for index in landmark]
                fit = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
                fit.fit(kernel([fingerprints[int(global_train[index])] for index in training], landmark_fps), residual[training])
                correction = RESIDUAL_WEIGHT * fit.predict(kernel([fingerprints[int(global_train[index])] for index in validation], landmark_fps))
                candidate[validation] = parent[validation] + correction
                train_fps = [fingerprints[int(global_train[index])] for index in training]
                similarity[validation] = np.max(kernel([fingerprints[int(global_train[index])] for index in validation], train_fps), axis=1)
            test_global = np.asarray([key_to_index[value] for value in test_rows["canonical"]], dtype=np.int64)
            rng = np.random.default_rng(seed)
            landmark = np.sort(rng.choice(np.arange(len(rows)), size=min(LANDMARKS, len(rows)), replace=False))
            landmark_fps = [fingerprints[int(global_train[index])] for index in landmark]
            fit = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
            fit.fit(kernel([fingerprints[int(index)] for index in global_train], landmark_fps), residual)
            target_candidate_test += RESIDUAL_WEIGHT * fit.predict(kernel([fingerprints[int(index)] for index in test_global], landmark_fps))
            mask = candidate_detail["target_type"].to_numpy(object) == target
            replacement = pd.Series(target_candidate_test, index=test_rows["id"].to_numpy())
            candidate_detail.loc[mask, "model_prediction"] = candidate_detail.loc[mask, "id"].map(replacement).astype(float).to_numpy()
        parent_r2 = float(r2_score(y, parent))
        candidate_r2 = float(r2_score(y, candidate))
        delta = candidate_r2 - parent_r2
        fold_rows = []
        for fold in range(5):
            selected = folds == fold
            fold_rows.append({"fold": fold, "rows": int(np.sum(selected)), "parent_r2": float(r2_score(y[selected], parent[selected])), "candidate_r2": float(r2_score(y[selected], candidate[selected])), "delta_r2": float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))})
        lower = bootstrap_lower(y, parent, candidate, groups) if target in ACTIVE else 0.0
        panel_report, minimum_panel = panels(y, parent, candidate, similarity, scaffolds) if target in ACTIVE else ({"unchanged_parent": {"rows": int(len(y)), "status": "unchanged", "delta_r2": 0.0}}, 0.0)
        gates = {"gain_pass": target not in ACTIVE or delta >= 0.01, "fold_pass": target not in ACTIVE or sum(item["delta_r2"] > 0 for item in fold_rows) >= 4, "bootstrap_pass": target not in ACTIVE or lower > 0.0, "panel_pass": target not in ACTIVE or minimum_panel >= 0.0, "strict_no_regression": delta >= -0.003}
        reports[target] = {"parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": float(delta), "positive_folds": int(sum(item["delta_r2"] > 0 for item in fold_rows)), "group_bootstrap_lower": float(lower), "minimum_panel_delta": float(minimum_panel), "folds": fold_rows, "panels": panel_report, "feature_count": feature_count, "pass": bool(all(gates.values())), "gates": gates, "unchanged_parent": target not in ACTIVE}
        oof_parts.append(pd.DataFrame({"canonical": canonical, "target_type": target, "target": y, "parent": parent, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}))
        component_parts.append(pd.DataFrame({"id": test_rows["id"].astype(int), "target_type": target, "parent_prediction": target_parent_test, "candidate_prediction": target_candidate_test, "changed": target in ACTIVE}))

    final_detail, _ = reference.apply_official_overrides(candidate_detail, test, raw_labels)
    predictions = final_detail[["id", "target"]].copy()
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in TARGETS]))
    maximum_loss = float(min(reports[target]["delta_r2"] for target in TARGETS))
    gate_pass = bool(mean_candidate - mean_parent >= 0.002 and maximum_loss >= -0.003 and all(reports[target]["pass"] for target in ACTIVE))
    return {"seed": seed, "target_reports": reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": maximum_loss, "complete_candidate_gate_pass": gate_pass, "complete_output_rows": int(len(predictions))}, pd.concat(oof_parts, ignore_index=True), pd.concat(component_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only C113 directory required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    parent_predictions, parent_oof, context = parent_control.rebuild_parent(root, data_dir, run_dir)
    canonical_dir = root / "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7"
    canonical_predictions = pd.read_csv(canonical_dir / "predictions.csv")
    canonical_oof = pd.read_csv(canonical_dir / "oof_predictions.csv")
    replay = parent_oof.sort_values(["target_type", "canonical"]).reset_index(drop=True)
    reference_oof = canonical_oof[["canonical", "target_type", "target", "candidate_prediction"]].rename(columns={"candidate_prediction": "canonical_prediction"}).sort_values(["target_type", "canonical"]).reset_index(drop=True)
    oof_delta = np.abs(replay["parent_prediction"].to_numpy(float) - reference_oof["canonical_prediction"].to_numpy(float))
    test_delta = np.abs(parent_predictions["target"].to_numpy(float) - canonical_predictions["target"].to_numpy(float))
    parent_oof_max = float(np.max(oof_delta))
    parent_test_max = float(np.max(test_delta))
    if parent_oof_max > 1.0e-12 or parent_test_max > 1.0e-12:
        raise RuntimeError(f"C113 parent parity failed: oof={parent_oof_max} test={parent_test_max}")
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
    ei_fold_map = special_ei_folds(pooled, test, keys, dense, cross_values, cross_available, sparse_parts, fingerprints, config)
    replica_reports: list[dict[str, Any]] = []
    final_oof: pd.DataFrame | None = None
    final_components: pd.DataFrame | None = None
    final_predictions: pd.DataFrame | None = None
    for seed in SEEDS:
        report, oof, components = run_replica(seed, parent_predictions, parent_oof, train, test, pooled, molecules, fingerprints, key_to_index, ei_fold_map, raw_labels)
        replica_reports.append(report)
        write_json(run_dir / f"replica_{seed}_metrics.json", report)
        if final_oof is None:
            final_oof, final_components, final_predictions = oof, components, parent_predictions.copy()
            for target in ACTIVE:
                rows = components[components["target_type"] == target].sort_values("id")
                mask = final_predictions["id"].isin(rows["id"])
                replacement = rows.set_index("id")["candidate_prediction"]
                final_predictions.loc[mask, "target"] = final_predictions.loc[mask, "id"].map(replacement).astype(float).to_numpy()
            detail = final_predictions.merge(test[["id", "target_type"]], on="id", how="left", validate="one_to_one")[["id", "target_type", "target"]].rename(columns={"target": "model_prediction"})
            final_detail, _ = reference.apply_official_overrides(detail, test, raw_labels)
            final_predictions = final_detail[["id", "target"]].copy()
    if final_oof is None or final_components is None or final_predictions is None:
        raise RuntimeError("C113 produced no replica")
    replica_means = [float(item["mean_candidate_r2"]) for item in replica_reports]
    parent_means = [float(item["mean_parent_r2"]) for item in replica_reports]
    report = {"schema_version": "ppp.round2.c113.tanimoto-landmark-residual-portfolio.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "C050 exact source replay; C112 parity control passed before modeling", "official_only": True, "official_inputs": context["inputs"], "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "submission": False, "parent_replay_oof_max_abs": parent_oof_max, "parent_replay_test_max_abs": parent_test_max, "replicas": replica_reports, "replica_mean_r2": replica_means, "replica_parent_mean_r2": parent_means, "mean_candidate_r2": float(np.mean(replica_means)), "mean_parent_r2": float(np.mean(parent_means)), "mean_gain": float(np.mean(replica_means) - np.mean(parent_means)), "all_replica_gate_pass": bool(all(item["complete_candidate_gate_pass"] for item in replica_reports)), "complete_output_rows": int(len(final_predictions)), "complete_output_order_pass": bool(final_predictions["id"].equals(test["id"])), "full_data_fit": "active residual models refit on all official labeled rows for the frozen test prediction path; local_eval values were not read", "local_eval_eligible": False, "elapsed_seconds": float(time.time() - started)}
    source_paths = {"runner": root / "tools/round2_c113_tanimoto_landmark_residual_portfolio.py", "parent_control": root / "tools/round2_c112_c050_parent_parity_control.py", "reference": root / "tools/initial_reference_pipeline.py", "mixed_parent": root / "tools/round2_mixed_candidate_v7.py", "fixed_features": root / "tools/round2_c063_egb_endpoint_conjugation_residual.py", "metric_plumbing": root / "tools/round2_eea_cross_target_oof_residual_stack.py"}
    report["source_hashes"] = {name: sha256_file(path) for name, path in source_paths.items()}
    final_predictions.to_csv(run_dir / "predictions.csv", index=False)
    final_oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    final_components.to_csv(run_dir / "component_predictions.csv", index=False)
    decision = "candidate_pending_notebook_parity" if report["all_replica_gate_pass"] else "rejected_full_candidate_gate"
    report["decision"] = decision
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"active_targets": list(ACTIVE), "landmarks": LANDMARKS, "alpha": ALPHA, "residual_weight": RESIDUAL_WEIGHT, "replica_seeds": list(SEEDS), "fingerprint": "Morgan radius 2, 4096 bits", "fold_local": True, "full_data_fit": True, "local_eval_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{decision}**. Mean parent R2 `{report['mean_parent_r2']:.12f}`; mean candidate R2 `{report['mean_candidate_r2']:.12f}`; gain `{report['mean_gain']:+.12f}`. Parent parity OOF/test maxima are `{parent_oof_max:.16g}`/`{parent_test_max:.16g}`. LocalEval was not read.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{digest}  SOURCE tools/{name}.py" for name, digest in report["source_hashes"].items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": decision, "mean_parent_r2": report["mean_parent_r2"], "mean_candidate_r2": report["mean_candidate_r2"], "mean_gain": report["mean_gain"], "parent_oof_max_abs": parent_oof_max, "parent_test_max_abs": parent_test_max, "all_replica_gate_pass": report["all_replica_gate_pass"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
