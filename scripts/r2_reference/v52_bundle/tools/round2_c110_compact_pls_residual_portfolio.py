#!/usr/bin/env python3
"""C110: compact official-only mixed residual portfolio.

The parent is rebuilt from the current official files through the C050/C098
source path.  Only four target heads receive a fixed, fold-local residual
correction; the other targets remain the rebuilt parent.  This runner writes
clean OOF evidence and full-data test predictions, but never reads the local_eval
or the validation external_label file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c076_eps_paired_charge_polarizability_residual as compact_features
import round2_c098_target_routed_qspr_full as parent_source
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE = ("eps", "nc", "ei", "tg")
RESIDUAL_WEIGHT = 0.20
REPLICA_SEEDS = (2026, 2027, 2028)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def residual_model(target: str) -> Any:
    if target in {"eps", "nc"}:
        estimator: Any = PLSRegression(n_components=3, max_iter=500)
    elif target == "ei":
        estimator = Ridge(alpha=30.0)
    else:
        estimator = Ridge(alpha=100.0)
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
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


def panel_report(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    scaffolds: np.ndarray,
    similarity: np.ndarray,
) -> tuple[dict[str, Any], float]:
    panels: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        rows = int(np.sum(selected))
        item: dict[str, Any] = {"rows": rows, "status": "inapplicable", "delta_r2": 0.0}
        if rows >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"status": "evaluable", "delta_r2": delta})
            deltas.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        add(name, selected)
    for scaffold in sorted(set(scaffolds)):
        add(f"scaffold_{scaffold}", scaffolds == scaffold, minimum=10)
    return panels, float(min(deltas)) if deltas else 0.0


def compact_matrix(
    bundle: dict[str, Any], target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    info = bundle["target_info"][target]
    test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    target_keys = sorted(set(info["canonical"]) | set(test_frame["canonical"]))
    global_indices = [bundle["key_to_index"][value] for value in target_keys]
    endpoint, endpoint_names = fixed_features.fixed_features(bundle["molecules"], global_indices)
    physics, physics_names = compact_features.physics_features(bundle["molecules"], global_indices)
    charge, charge_names = compact_features.charge_features(bundle["molecules"], global_indices)
    matrix = np.hstack([endpoint, physics, charge]).astype(np.float64, copy=False)
    row_for_key = {value: row for row, value in enumerate(target_keys)}
    train_rows = np.asarray([row_for_key[value] for value in info["canonical"]], dtype=np.int64)
    test_rows = np.asarray([row_for_key[value] for value in test_frame["canonical"]], dtype=np.int64)
    names = endpoint_names + physics_names + charge_names
    if matrix.shape[1] != len(names):
        raise RuntimeError(f"{target} compact feature name/column mismatch")
    return matrix, train_rows, test_rows, names


def run_replica(bundle: dict[str, Any], seed: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    raw_test = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []

    for target in TARGETS:
        info = bundle["target_info"][target]
        y = np.asarray(info["y"], dtype=np.float64)
        parent = np.asarray(info["parent"], dtype=np.float64)
        folds = np.asarray(info["folds"], dtype=np.int64)
        candidate = parent.copy()
        similarity = np.full(len(y), np.nan, dtype=np.float64)
        feature_names: list[str] = []
        test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_parent = bundle["test_detail"].loc[bundle["test_detail"]["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float)
        test_candidate = test_parent.copy()

        if target in ACTIVE:
            matrix, train_rows, test_rows, feature_names = compact_matrix(bundle, target)
            residual = y - parent
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                if len(validation) == 0 or len(training) == 0:
                    raise RuntimeError(f"{target} fold {fold} is empty")
                fitted = residual_model(target)
                fitted.fit(matrix[train_rows[training]], residual[training])
                correction = RESIDUAL_WEIGHT * np.asarray(fitted.predict(matrix[train_rows[validation]])).reshape(-1)
                candidate[validation] = parent[validation] + correction
                global_validation = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][validation]], dtype=np.int64)
                global_training = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][training]], dtype=np.int64)
                similarity[validation] = fixed_features.nearest_similarity(bundle["fingerprints"], global_validation, global_training)
            fitted = residual_model(target)
            fitted.fit(matrix[train_rows], residual)
            test_correction = RESIDUAL_WEIGHT * np.asarray(fitted.predict(matrix[test_rows])).reshape(-1)
            test_candidate += test_correction
            mask = raw_test["target_type"].to_numpy(object) == target
            replacement = pd.Series(test_candidate, index=test_frame["id"].to_numpy())
            raw_test.loc[mask, "model_prediction"] = raw_test.loc[mask, "id"].map(replacement).astype(float).to_numpy()

        parent_r2 = float(r2_score(y, parent))
        candidate_r2 = float(r2_score(y, candidate))
        delta = candidate_r2 - parent_r2
        fold_rows = []
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            fold_rows.append({
                "fold": fold,
                "rows": int(len(validation)),
                "parent_r2": float(r2_score(y[validation], parent[validation])),
                "candidate_r2": float(r2_score(y[validation], candidate[validation])),
                "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation])),
            })
        positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        lower = bootstrap_lower(y, parent, candidate, info["groups"]) if target in ACTIVE else 0.0
        if target in ACTIVE:
            panels, minimum_panel = panel_report(y, parent, candidate, info["scaffolds"], similarity)
            gates = {
                "gain_pass": delta >= 0.01,
                "fold_pass": positive_folds >= 4,
                "bootstrap_pass": lower > 0.0,
                "panel_pass": minimum_panel >= 0.0,
                "strict_no_regression": delta >= -0.003,
            }
        else:
            panels, minimum_panel = {"unchanged_parent": {"rows": int(len(y)), "status": "unchanged", "delta_r2": 0.0}}, 0.0
            gates = {"unchanged_parent": True}
        target_reports[target] = {
            "parent_r2": parent_r2,
            "candidate_r2": candidate_r2,
            "delta_r2": float(delta),
            "positive_folds": positive_folds,
            "group_bootstrap_lower": float(lower),
            "minimum_panel_delta": float(minimum_panel),
            "folds": fold_rows,
            "panels": panels,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "pass": bool(all(gates.values())),
            "gates": gates,
            "unchanged_parent": target not in ACTIVE,
        }
        oof_parts.append(pd.DataFrame({
            "canonical": info["canonical"],
            "target_type": target,
            "target": y,
            "parent": parent,
            "candidate": candidate,
            "group": info["groups"],
            "scaffold": info["scaffolds"],
            "outer_fold": folds,
            "nearest_similarity": similarity,
        }))
        component_parts.append(pd.DataFrame({
            "id": test_frame["id"].astype(int),
            "target_type": target,
            "parent_prediction": test_parent,
            "candidate_prediction": test_candidate,
            "changed": target in ACTIVE,
        }))

    raw_detail, override_report = reference.apply_official_overrides(raw_test, bundle["test"], bundle["raw_labels"])
    predictions = raw_detail[["id", "target"]].copy()
    if len(predictions) != 4940 or predictions["id"].duplicated().any() or not predictions["id"].equals(bundle["test"]["id"]):
        raise RuntimeError("C110 prediction IDs/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C110 produced non-finite predictions")
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    maximum_loss = float(min(target_reports[target]["delta_r2"] for target in TARGETS))
    complete_gate = bool(
        mean_candidate > mean_parent
        and mean_candidate - mean_parent >= 0.002
        and maximum_loss >= -0.003
        and all(target_reports[target]["pass"] for target in ACTIVE)
    )
    report = {
        "seed": seed,
        "target_reports": target_reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": maximum_loss,
        "complete_candidate_gate_pass": complete_gate,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "official_override_report": override_report,
    }
    return report, pd.concat(oof_parts, ignore_index=True), pd.concat(component_parts, ignore_index=True).assign(_predictions=predictions["target"].to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only C110 run directory required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    bundle = parent_source.parent_bundle(root, data_dir)
    replica_reports: list[dict[str, Any]] = []
    final_oof: pd.DataFrame | None = None
    final_components: pd.DataFrame | None = None
    final_predictions: pd.DataFrame | None = None
    for seed in REPLICA_SEEDS:
        report, oof, components = run_replica(bundle, seed)
        replica_reports.append(report)
        if final_oof is None:
            final_oof = oof
            final_components = components
            raw = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
            for target in ACTIVE:
                rows = components[components["target_type"] == target].sort_values("id")
                mask = raw["target_type"].to_numpy(object) == target
                raw.loc[mask, "model_prediction"] = rows["candidate_prediction"].to_numpy(float)
            raw_detail, _ = reference.apply_official_overrides(raw, bundle["test"], bundle["raw_labels"])
            final_predictions = raw_detail[["id", "target"]].copy()
        write_json(run_dir / f"replica_{seed}_metrics.json", report)
    if final_oof is None or final_components is None or final_predictions is None:
        raise RuntimeError("no C110 replica completed")
    if not final_predictions["id"].equals(bundle["test"]["id"]):
        raise RuntimeError("C110 final prediction order mismatch")
    mean_candidates = [float(item["mean_candidate_r2"]) for item in replica_reports]
    mean_parents = [float(item["mean_parent_r2"]) for item in replica_reports]
    final_report = {
        "schema_version": "ppp.round2.c110.compact-pls-residual-portfolio.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "C050 rebuilt through C098 parent_source from official current/archive inputs; no saved predictions loaded",
        "official_only": True,
        "official_inputs": bundle["inputs"],
        "external_label_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "submission": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "replicas": replica_reports,
        "replica_mean_r2": mean_candidates,
        "replica_parent_mean_r2": mean_parents,
        "mean_candidate_r2": float(np.mean(mean_candidates)),
        "mean_parent_r2": float(np.mean(mean_parents)),
        "mean_gain": float(np.mean(mean_candidates) - np.mean(mean_parents)),
        "all_replica_gate_pass": bool(all(item["complete_candidate_gate_pass"] for item in replica_reports)),
        "complete_output_rows": int(len(final_predictions)),
        "complete_output_order_pass": True,
        "full_data_fit": "active residual heads refit on every official labeled row before test predictions; no local_eval values used",
        "local_eval_eligible": False,
        "elapsed_seconds": float(time.time() - started),
    }
    final_predictions.to_csv(run_dir / "predictions.csv", index=False)
    final_oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    final_components.to_csv(run_dir / "component_predictions.csv", index=False)
    source_paths = {
        "runner": root / "tools" / "round2_c110_compact_pls_residual_portfolio.py",
        "parent_source": root / "tools" / "round2_c098_target_routed_qspr_full.py",
        "reference": root / "tools" / "initial_reference_pipeline.py",
        "fixed_features": root / "tools" / "round2_c063_egb_endpoint_conjugation_residual.py",
        "compact_features": root / "tools" / "round2_c076_eps_paired_charge_polarizability_residual.py",
        "metric_plumbing": root / "tools" / "round2_eea_cross_target_oof_residual_stack.py",
        "mixed_parent_route": root / "tools" / "round2_mixed_candidate_v7.py",
        "ei_parent_route": root / "tools" / "round2_ei_scaffold_abstaining_gap_identity_v4_portable.py",
        "eea_parent_route": root / "tools" / "round2_eea_scaffold_abstaining_gap_identity_v7_portable.py",
    }
    final_report["source_hashes"] = {name: sha256_file(path) for name, path in source_paths.items()}
    write_json(run_dir / "metrics.json", final_report)
    write_json(run_dir / "config.json", {
        "active_targets": list(ACTIVE),
        "replica_seeds": list(REPLICA_SEEDS),
        "residual_weight": RESIDUAL_WEIGHT,
        "models": {"eps": "PLSRegression(n_components=3)", "nc": "PLSRegression(n_components=3)", "ei": "Ridge(alpha=30)", "tg": "Ridge(alpha=100)"},
        "features": "fixed endpoint/topology plus physicochemical and charge descriptors from official SMILES",
        "folds": "parent-provided exact C050 target-specific five-fold maps",
        "bootstrap_resamples": 2000,
        "full_data_fit": True,
        "external_label_file_read": False,
        "local_eval_read": False,
    })
    (run_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    decision = "candidate_pending_fresh_notebook_parity_and_local_eval" if final_report["all_replica_gate_pass"] else "rejected_full_candidate_gate"
    final_report["decision"] = decision
    write_json(run_dir / "metrics.json", final_report)
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: **{decision}**. Mean parent R2 `{final_report['mean_parent_r2']:.12f}`; mean candidate R2 `{final_report['mean_candidate_r2']:.12f}`; gain `{final_report['mean_gain']:+.12f}`. Full-data fit was used only to generate the frozen candidate predictions; local_eval was not read.\n",
        encoding="utf-8",
    )
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{digest}  SOURCE tools/{name}.py" for name, digest in final_report["source_hashes"].items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": decision, "mean_parent_r2": final_report["mean_parent_r2"], "mean_candidate_r2": final_report["mean_candidate_r2"], "mean_gain": final_report["mean_gain"], "all_replica_gate_pass": final_report["all_replica_gate_pass"], "elapsed_seconds": final_report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
