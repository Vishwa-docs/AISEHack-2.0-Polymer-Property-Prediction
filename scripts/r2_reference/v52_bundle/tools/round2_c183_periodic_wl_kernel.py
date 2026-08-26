#!/usr/bin/env python3
"""C183: compact periodic versus non-periodic WL sparse-kernel residuals."""

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
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = ("ei", "eps", "nc", "eea", "egc", "egb", "tg")
ACTIVE = ("ei", "eps", "nc", "eea")
SEED = 2026
WL_STEPS = 3
HASH_WIDTH = 8192
ALPHA = 30.0
RESIDUAL_WEIGHT = 0.30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def graph_tokens(mol: Any, periodic: bool) -> list[dict[int, float]]:
    n = mol.GetNumAtoms()
    adjacency = [set(int(nb.GetIdx()) for nb in atom.GetNeighbors()) for atom in mol.GetAtoms()]
    if periodic:
        dummy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy) == 2 and all(atom.GetDegree() == 1 for atom in dummy):
            a = int(dummy[0].GetNeighbors()[0].GetIdx())
            b = int(dummy[1].GetNeighbors()[0].GetIdx())
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)
    labels: list[str] = []
    for atom, neighbors in zip(mol.GetAtoms(), adjacency, strict=True):
        labels.append("|".join([
            str(atom.GetAtomicNum()), str(len(neighbors)), str(int(atom.GetIsAromatic())),
            str(atom.GetFormalCharge()), str(atom.GetHybridization()), str(int(atom.GetIsotope())),
        ]))
    rows: list[dict[int, float]] = []
    for step in range(WL_STEPS + 1):
        counts: dict[int, float] = {}
        for label in labels:
            bucket = int.from_bytes(hashlib.blake2b(f"{step}|{label}".encode(), digest_size=8).digest(), "little") % HASH_WIDTH
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        rows.append(counts)
        if step < WL_STEPS:
            labels = [hashlib.blake2b((label + "||" + "|".join(sorted(labels[nb] for nb in neighbors))).encode(), digest_size=12).hexdigest() for label, neighbors in zip(labels, adjacency, strict=True)]
    return rows


def wl_matrix(molecules: list[Any], periodic: bool) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    for row, mol in enumerate(molecules):
        for counts in graph_tokens(mol, periodic):
            for col, value in counts.items():
                row_ids.append(row)
                col_ids.append(col)
                values.append(value)
    matrix = sparse.csr_matrix((values, (row_ids, col_ids)), shape=(len(molecules), HASH_WIDTH), dtype=np.float64)
    return matrix, {"periodic": periodic, "wl_steps": WL_STEPS, "hash_width": HASH_WIDTH, "shape": [int(x) for x in matrix.shape], "nnz": int(matrix.nnz)}


def fit_arm(info: dict[str, Any], matrix: sparse.csr_matrix, test_indices: np.ndarray) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    residual_oof = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        model = Ridge(alpha=ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
        model.fit(matrix[indices[training]], y[training] - parent[training])
        residual_oof[validation] = model.predict(matrix[indices[validation]])
        candidate_fold = parent[validation] + RESIDUAL_WEIGHT * residual_oof[validation]
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent[validation])), "candidate_r2": float(r2_score(y[validation], candidate_fold)), "delta_r2": float(r2_score(y[validation], candidate_fold) - r2_score(y[validation], parent[validation]))})
    candidate = parent + RESIDUAL_WEIGHT * residual_oof
    delta = float(r2_score(y, candidate) - r2_score(y, parent))
    lower = carrier.bootstrap_lower(y, parent, candidate, groups)
    positive = int(sum(row["delta_r2"] > 0 for row in fold_rows))
    full = Ridge(alpha=ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
    full.fit(matrix[indices], y - parent)
    test_residual = full.predict(matrix[test_indices])
    return {"candidate": candidate, "test_candidate": np.asarray(test_residual, dtype=float), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "folds": fold_rows}


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
    nonperiodic, non_report = wl_matrix(parent["molecules"], False)
    periodic, periodic_report = wl_matrix(parent["molecules"], True)
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        test_indices = np.asarray([parent["key_to_index"][value] for value in parent["test"].loc[parent["test"]["target_type"] == target, "canonical"]], dtype=np.int64)
        if target not in ACTIVE:
            target_reports[target] = {"active": False, "parent_r2": float(r2_score(info["y"], info["parent"])), "pass": False}
            # Keep the full seven-target parent baseline in the arithmetic
            # mean even when this family only tests a weak-target residual.
            oof_parts.append(pd.DataFrame({"target": info["y"], "parent": info["parent"], "assembled": info["parent"], "target_type": target}))
            continue
        p = fit_arm(info, periodic, test_indices)
        n = fit_arm(info, nonperiodic, test_indices)
        p_gate = bool(p["delta_r2"] >= 0.005 and p["positive_folds"] >= 4 and p["group_bootstrap_lower"] > 0.0 and p["delta_r2"] >= n["delta_r2"] + 0.002)
        target_reports[target] = {"active": True, "parent_r2": float(r2_score(info["y"], info["parent"])), "periodic_r2": float(r2_score(info["y"], p["candidate"])), "periodic_delta_r2": p["delta_r2"], "periodic_positive_folds": p["positive_folds"], "periodic_group_bootstrap_lower": p["group_bootstrap_lower"], "nonperiodic_delta_r2": n["delta_r2"], "periodic_pass": p_gate, "periodic_folds": p["folds"], "nonperiodic_folds": n["folds"]}
        chosen = p_gate
        oof_parts.append(pd.DataFrame({"target": info["y"], "parent": info["parent"], "candidate": p["candidate"], "assembled": np.where(chosen, p["candidate"], info["parent"]), "target_type": target, "fold": carrier.grouped_folds(np.asarray(info["groups"], dtype=object))}))
        test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        parent_test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        if not np.array_equal(test_rows["id"].to_numpy(int), parent_test_detail["id"].to_numpy(int)):
            raise RuntimeError(f"C183 test ID alignment failed for {target}")
        parent_test_target = parent_test_detail["target"].to_numpy(float)
        test_parts.append(pd.DataFrame({"id": test_rows["id"].astype(int), "target_type": target, "periodic_candidate": parent_test_target + RESIDUAL_WEIGHT * p["test_candidate"]}))
    banked = [target for target in ACTIVE if target_reports[target].get("periodic_pass")]
    oof = pd.concat(oof_parts, ignore_index=True)
    parent_mean = float(np.mean([r2_score(part["target"], part["parent"]) for part in oof_parts if "parent" in part]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component = pd.concat(test_parts, ignore_index=True)
    predictions = parent_test.merge(component, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(predictions["target_type"].isin(banked), predictions["periodic_candidate"], predictions["target"])
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C183 complete output contract failed")
    full_pass = bool(assembled_mean - parent_mean >= 0.002 and len(banked) > 0)
    report = {"schema_version": "ppp.round2.c183.periodic-wl-kernel.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "official_inputs": parent["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "kaggle_compute": False, "kaggle_upload": False, "parent_replay_parity": parity, "nonperiodic_feature_report": non_report, "periodic_feature_report": periodic_report, "target_reports": target_reports, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": assembled_mean, "mean_gain": assembled_mean - parent_mean, "complete_output_rows": int(len(predictions)), "complete_output_order_pass": True, "full_candidate_gate_pass": full_pass, "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate", "elapsed_seconds": float(time.time() - started), "source_hashes": {"runner": sha256_file(Path(__file__)), "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"), "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"), "reference": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py")}}
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component.to_csv(run_dir / "component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c183.periodic-wl-kernel.v1", "seed": SEED, "active_targets": list(ACTIVE), "wl_steps": WL_STEPS, "hash_width": HASH_WIDTH, "ridge_alpha": ALPHA, "residual_weight": RESIDUAL_WEIGHT, "periodic_ablation_required": True, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; gain `{assembled_mean - parent_mean:+.12f}`. Official-only; no local_eval read.\n", encoding="utf-8")
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.name != "artifact_manifest.sha256" and path.is_file()]
    for name, digest in report["source_hashes"].items(): manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": assembled_mean, "mean_gain": assembled_mean - parent_mean, "decision": report["decision"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
