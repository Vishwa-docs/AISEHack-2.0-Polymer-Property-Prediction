#!/usr/bin/env python3
"""C188: typed BRICS-fragment and short atom-path sparse residual kernel."""

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
from rdkit import Chem
from rdkit.Chem import BRICS
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
ACTIVE = ("ei", "eea", "nc", "eps")
SEED = 20260804
HASH_WIDTH = 16384
RESIDUAL_WEIGHT = 0.35


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def bucket(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little") % HASH_WIDTH


def molecule_tokens(mol: Chem.Mol) -> dict[int, float]:
    counts: dict[int, float] = {}
    def add(token: str, value: float = 1.0) -> None:
        key = bucket(token)
        counts[key] = counts.get(key, 0.0) + value
    try:
        fragments = BRICS.BRICSDecompose(mol)
    except Exception:
        fragments = set()
    for fragment in fragments:
        add("BRICS|" + str(fragment))
    for atom in mol.GetAtoms():
        add(f"A|{atom.GetAtomicNum()}|{int(atom.GetIsAromatic())}|{atom.GetHybridization()}|{atom.GetFormalCharge()}")
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        left = f"{a.GetAtomicNum()}:{int(a.GetIsAromatic())}"
        right = f"{b.GetAtomicNum()}:{int(b.GetIsAromatic())}"
        add("B|" + "|".join(sorted((left, right))) + f"|{bond.GetBondType()}")
    for atom in mol.GetAtoms():
        for nb in atom.GetNeighbors():
            if atom.GetIdx() >= nb.GetIdx():
                continue
            for nb2 in nb.GetNeighbors():
                if nb2.GetIdx() in (atom.GetIdx(), nb.GetIdx()):
                    continue
                path = [atom, nb, nb2]
                labels = [f"{x.GetAtomicNum()}:{int(x.GetIsAromatic())}" for x in path]
                add("P2|" + "|".join(labels))
    for atom in mol.GetAtoms():
        for first in atom.GetNeighbors():
            for second in first.GetNeighbors():
                if second.GetIdx() == atom.GetIdx():
                    continue
                for third in second.GetNeighbors():
                    if third.GetIdx() in (atom.GetIdx(), first.GetIdx()):
                        continue
                    labels = [f"{x.GetAtomicNum()}:{int(x.GetIsAromatic())}" for x in (atom, first, second, third)]
                    add("P3|" + "|".join(labels))
    return counts


def token_matrix(molecules: list[Chem.Mol]) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row, mol in enumerate(molecules):
        for col, value in molecule_tokens(mol).items():
            rows.append(row)
            cols.append(col)
            vals.append(value)
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(molecules), HASH_WIDTH), dtype=np.float64)
    matrix = matrix.multiply(1.0 / np.maximum(1.0, np.asarray(matrix.sum(axis=1)).ravel())[:, None]).tocsr()
    return matrix, {"shape": [int(x) for x in matrix.shape], "nnz": int(matrix.nnz), "hash_width": HASH_WIDTH, "representation": "BRICS fragments, typed atoms/bonds, length-2 and length-3 atom paths"}


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
    matrix, feature_report = token_matrix(parent["molecules"])
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        if target not in ACTIVE:
            target_reports[target] = {"active": False, "parent_r2": float(r2_score(info["y"], info["parent"])), "pass": False}
            oof_parts.append(pd.DataFrame({"target": info["y"], "parent": info["parent"], "candidate": info["parent"], "assembled": info["parent"], "target_type": target}))
            continue
        indices = np.asarray(info["indices"], dtype=np.int64)
        y = np.asarray(info["y"], dtype=float)
        parent_pred = np.asarray(info["parent"], dtype=float)
        groups = np.asarray(info["groups"], dtype=object)
        folds = carrier.grouped_folds(groups)
        residual_oof = np.full(len(y), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(carrier.N_FOLDS):
            va = np.flatnonzero(folds == fold)
            tr = np.flatnonzero(folds != fold)
            model = Ridge(alpha=40.0, solver="lsqr", max_iter=5000, tol=1e-4)
            model.fit(matrix[indices[tr]], y[tr] - parent_pred[tr])
            residual_oof[va] = model.predict(matrix[indices[va]])
            fold_parent = r2_score(y[va], parent_pred[va])
            fold_candidate = parent_pred[va] + RESIDUAL_WEIGHT * residual_oof[va]
            fold_rows.append({"fold": fold, "parent_r2": float(fold_parent), "candidate_r2": float(r2_score(y[va], fold_candidate)), "delta_r2": float(r2_score(y[va], fold_candidate) - fold_parent), "rows": int(len(va))})
        candidate = parent_pred + RESIDUAL_WEIGHT * residual_oof
        delta = float(r2_score(y, candidate) - r2_score(y, parent_pred))
        lower = float(carrier.bootstrap_lower(y, parent_pred, candidate, groups))
        positive = int(sum(row["delta_r2"] > 0 for row in fold_rows))
        passed = bool(delta >= 0.006 and positive >= 4 and lower > 0.0)
        target_reports[target] = {"active": True, "parent_r2": float(r2_score(y, parent_pred)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "pass": passed, "folds": fold_rows}
        oof_parts.append(pd.DataFrame({"target": y, "parent": parent_pred, "candidate": candidate, "assembled": candidate if passed else parent_pred, "target_type": target, "group": groups, "fold": folds}))
        test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
            raise RuntimeError(f"C188 test ID alignment failed for {target}")
        test_indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
        model = Ridge(alpha=40.0, solver="lsqr", max_iter=5000, tol=1e-4)
        model.fit(matrix[indices], y - parent_pred)
        test_parts.append(pd.DataFrame({"id": test_rows["id"].astype(int), "target_type": target, "candidate": test_detail["target"].to_numpy(float) + RESIDUAL_WEIGHT * model.predict(matrix[test_indices])}))
    banked = [target for target in ACTIVE if target_reports[target].get("pass")]
    oof = pd.concat(oof_parts, ignore_index=True)
    parent_mean = float(np.mean([r2_score(part["target"], part["parent"]) for part in oof_parts]))
    candidate_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    direct = pd.concat(test_parts, ignore_index=True)
    predictions = parent_test.merge(direct, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(predictions["target_type"].isin(banked), predictions["candidate"], predictions["target"])
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C188 complete output contract failed")
    report = {"schema_version": "ppp.round2.c188.fragment-path-kernel.v1", "experiment_id": run_dir.name, "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "official_inputs": parent["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "kaggle_compute": False, "kaggle_upload": False, "parent_replay_parity": parity, "feature_report": feature_report, "target_reports": target_reports, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": candidate_mean, "mean_gain": candidate_mean - parent_mean, "complete_output_rows": int(len(predictions)), "complete_output_order_pass": True, "full_candidate_gate_pass": bool(banked and candidate_mean - parent_mean >= 0.002), "decision": "candidate_pass_pending_clean_reproduction" if banked and candidate_mean - parent_mean >= 0.002 else "rejected_component_or_full_gate", "elapsed_seconds": float(time.time() - started), "source_hashes": {"runner": sha256_file(Path(__file__)), "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"), "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py")}}
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    direct.to_csv(run_dir / "component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c188.fragment-path-kernel.v1", "seed": SEED, "hash_width": HASH_WIDTH, "residual_weight": RESIDUAL_WEIGHT, "active_targets": list(ACTIVE), "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: {report['decision']}. Banked targets: {','.join(banked) or 'none'}. Mean parent {parent_mean:.12f}; assembled {candidate_mean:.12f}; gain {candidate_mean - parent_mean:+.12f}. Official-only; no local_eval read.\n", encoding="utf-8")
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.name != "artifact_manifest.sha256" and path.is_file()]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": candidate_mean, "mean_gain": candidate_mean - parent_mean, "decision": report["decision"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
