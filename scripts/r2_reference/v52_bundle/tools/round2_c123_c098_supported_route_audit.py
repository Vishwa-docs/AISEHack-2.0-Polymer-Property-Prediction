#!/usr/bin/env python3
"""C123: exact-replay audit of the supported-only C098 EPS/Nc route."""

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

import initial_reference_pipeline as reference
import round2_c098_target_routed_qspr_full as qspr
import round2_c120_nearmiss_bridge_mixed as c120
import round2_c121_nearmiss_bridge_one_replay as c121
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE = ("eps", "nc")
SEED = 2026
RESIDUAL_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, name: str, **fields: Any) -> None:
    record = {"checkpoint": name, "at": datetime.now().astimezone().isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    data_dir = (root / args.data_dir).resolve()
    run_dir = (root / args.run_dir).resolve()
    if {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("C123 requires a fresh protocol-only run directory")
    started = time.time()
    progress_path = run_dir / "progress.jsonl"
    bundle, parity = c121.build_one_replay_bundle(root, data_dir, run_dir)
    checkpoint(progress_path, "parent_replay", oof_rows=9851, test_rows=4940)
    checkpoint(progress_path, "parent_parity", **parity)
    if parity["oof_max_abs"] > 1e-12 or parity["test_max_abs"] > 1e-12:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = bundle["target_info"][target]
        y = info["y"]
        parent = info["parent"]
        if target not in ACTIVE:
            reports[target] = {"parent_r2": c120.score(y, parent), "candidate_r2": c120.score(y, parent), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "pass": True, "unchanged_parent": True}
            oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": parent, "outer_fold": -1}))
            continue
        matrix, train_rows, _, names, pair_train, _ = qspr.target_features(bundle, target)
        residual = y - parent
        groups = np.asarray([plumbing.no_stereo(value) for value in info["canonical"]], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in info["canonical"]], dtype=object)
        folds = plumbing.folds_for(groups, 5)
        candidate = np.full(len(y), np.nan, dtype=float)
        similarity = np.full(len(y), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            training = np.flatnonzero(folds != fold)
            model = c120.make_arm("ridge")
            model.fit(matrix[train_rows[training]], residual[training])
            correction = RESIDUAL_WEIGHT * model.predict(matrix[train_rows[validation]])
            candidate[validation] = parent[validation]
            supported = np.isfinite(pair_train[validation])
            candidate[validation[supported]] += correction[supported]
            train_fps = [bundle["fingerprints"][int(train_rows[index])] for index in training]
            for row, global_index in enumerate(train_rows[validation]):
                similarities = reference.DataStructs.BulkTanimotoSimilarity(bundle["fingerprints"][int(global_index)], train_fps)
                similarity[validation[row]] = max(similarities) if similarities else 0.0
            fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": c120.score(y[validation], parent[validation]), "candidate_r2": c120.score(y[validation], candidate[validation]), "delta_r2": c120.score(y[validation], candidate[validation]) - c120.score(y[validation], parent[validation])})
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"non-finite {target} candidate")
        pair_mask = np.isfinite(pair_train)
        bootstrap = c120.grouped_bootstrap(y, parent, candidate, groups)
        panels, minimum_panel = c120.panel_report(y, parent, candidate, groups, scaffolds, similarity, pair_mask)
        delta = c120.score(y, candidate) - c120.score(y, parent)
        supported_delta = c120.score(y[pair_mask], candidate[pair_mask]) - c120.score(y[pair_mask], parent[pair_mask]) if np.sum(pair_mask) > 1 else None
        missing_delta = c120.score(y[~pair_mask], candidate[~pair_mask]) - c120.score(y[~pair_mask], parent[~pair_mask]) if np.sum(~pair_mask) > 1 else None
        positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        target_pass = bool(delta >= 0.01 and positive >= 4 and bootstrap["lower_2_5"] > 0.0 and minimum_panel is not None and minimum_panel >= 0.0 and (missing_delta is None or abs(missing_delta) <= 1e-12))
        reports[target] = {"parent_r2": c120.score(y, parent), "candidate_r2": c120.score(y, candidate), "delta_r2": delta, "positive_folds": positive, "folds": fold_rows, "group_bootstrap_lower": bootstrap["lower_2_5"], "minimum_panel_delta": minimum_panel, "supported_delta_r2": supported_delta, "missing_delta_r2": missing_delta, "panels": panels, "feature_count": int(matrix.shape[1]), "pair_rows": int(np.sum(pair_mask)), "pass": target_pass, "unchanged_parent": False}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "outer_fold": folds, "counterpart_available": pair_mask}))
        checkpoint(progress_path, f"{target}_oof", delta_r2=delta, supported_delta_r2=supported_delta, missing_delta_r2=missing_delta, positive_folds=positive, minimum_panel_delta=minimum_panel)
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(reports[target]["delta_r2"] for target in TARGETS))
    clean_pass = bool(mean_candidate - mean_parent >= 0.002 and max_loss >= -0.003 and all(reports[target]["pass"] for target in ACTIVE))
    report = {"schema_version": "ppp.round2.c123.c098-supported-route-audit.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "official_only": True, "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7", "parent_parity": parity, "target_reports": reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": max_loss, "clean_gate_pass": clean_pass, "decision": "clean_gate_pass_pending_full_data_fit" if clean_pass else "rejected_component_or_full_gate", "elapsed_seconds": float(time.time() - started)}
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c123.c098-supported-route-audit.v1", "active_targets": ACTIVE, "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": 30.0, "outer_folds": "folds_for(groups,5)", "missing_counterpart": "exact_noop", "no_sweep": True, "official_only": True, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent R2 `{mean_parent:.12f}`; candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. No full-data test fit or local_eval action was opened before the clean gates.\n", encoding="utf-8")
    source_paths = [Path(__file__), root / "tools/round2_c121_nearmiss_bridge_one_replay.py", root / "tools/round2_c120_nearmiss_bridge_mixed.py", root / "tools/round2_c098_target_routed_qspr_full.py", root / "tools/round2_c112_c050_parent_parity_control.py", root / "tools/initial_reference_pipeline.py", root / "tools/round2_eea_cross_target_oof_residual_stack.py"]
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for path in source_paths:
        manifest.append(f"{sha256_file(path)}  SOURCE {path.relative_to(root)}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    checkpoint(progress_path, "metrics_written", decision=report["decision"], mean_gain=report["mean_gain"])
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "eps_delta": reports["eps"]["delta_r2"], "nc_delta": reports["nc"]["delta_r2"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
