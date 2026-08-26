#!/usr/bin/env python3
"""Local, train-only residual diagnosis for the frozen C001 carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
C001_ID = "R2-C001-20260803-1645-initial-reference-repaired"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        value = ""
    return value or "ACYCLIC"


def stats(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    return {
        "rows": int(len(y)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and not np.isclose(np.var(y), 0.0) else None,
        "mae": float(mean_absolute_error(y, pred)) if len(y) else None,
        "bias": float(np.mean(pred - y)) if len(y) else None,
        "mean_abs_residual": float(np.mean(np.abs(pred - y))) if len(y) else None,
    }


def nearest_similarity(fingerprints: list[Any], folds: np.ndarray) -> np.ndarray:
    output = np.empty(len(fingerprints), dtype=np.float64)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[index] for index in training]
        for index in validation:
            output[index] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"Pre-created protocol directory is required: {run_dir}")
    existing = {path.name for path in run_dir.iterdir()}
    if existing - {"protocol.json"}:
        raise RuntimeError(f"Refusing to reuse non-empty run directory: {run_dir}")
    started = time.time()
    data_dir = (root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    oof_path = root / "experiments" / "CLEAN_OFFICIAL_ONLY" / C001_ID / "oof_predictions.csv"
    oof = pd.read_csv(oof_path)
    expected = {"canonical", "target_type", "target", "prediction", "sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local"}
    if set(oof.columns) != expected:
        raise RuntimeError("C001 OOF schema mismatch")
    merged = pooled[["canonical", "target_type", "target", "smiles"]].merge(oof, on=["canonical", "target_type", "target"], how="left", validate="one_to_one")
    if merged["prediction"].isna().any():
        raise RuntimeError("C001 OOF does not cover the pooled official label table")
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    molecules = reference.build_molecules(keys)
    key_to_index = {key: index for index, key in enumerate(keys)}
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    _, physical_names = reference.physical_matrix(molecules, keys)
    physical_index = {name: index for index, name in enumerate(physical_names)}
    physical_all, _ = reference.physical_matrix(molecules, keys)
    all_rows: list[dict[str, Any]] = []
    target_summaries: dict[str, Any] = {}
    branch_candidates: list[dict[str, Any]] = []
    for target in TARGETS:
        frame = merged[merged["target_type"] == target].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        pred = frame["prediction"].to_numpy(float)
        target_indices = np.asarray([key_to_index[value] for value in frame["canonical"]], dtype=np.int64)
        fold_ids = np.full(len(frame), -1, dtype=np.int64)
        splitter = KFold(n_splits=5, shuffle=True, random_state=2026)
        for fold, (_, validation) in enumerate(splitter.split(np.arange(len(frame)))):
            fold_ids[validation] = fold
        fps = reference.morgan_bits([molecules[index] for index in target_indices], 2, 4096)
        nearest = nearest_similarity(fps, fold_ids)
        physical = physical_all[target_indices]
        aux_count = np.sum(np.delete(cross_available[target_indices], TARGETS.index(target), axis=1), axis=1).astype(int)
        aromatic = physical[:, physical_index["aromatic_atom_count"]]
        hetero = physical[:, physical_index["hetero_atom_count"]]
        heavy = physical[:, physical_index["heavy_atom_count"]]
        length = physical[:, physical_index["smiles_length"]]
        disagreement = np.abs(frame["extra_trees"].to_numpy(float) - frame["sparse_ridge"].to_numpy(float))
        q_heavy = np.quantile(heavy, [0.25, 0.75])
        q_length = np.quantile(length, [0.25, 0.75])
        q_disagreement = np.quantile(disagreement, 0.75)
        slice_specs = [
            ("all", np.ones(len(frame), dtype=bool)),
            ("nearest_lt_0.30", nearest < 0.30),
            ("nearest_0.30_0.50", (nearest >= 0.30) & (nearest < 0.50)),
            ("nearest_0.50_0.70", (nearest >= 0.50) & (nearest < 0.70)),
            ("nearest_ge_0.70", nearest >= 0.70),
            ("aux_count_0", aux_count == 0),
            ("aux_count_1_2", (aux_count >= 1) & (aux_count <= 2)),
            ("aux_count_ge_3", aux_count >= 3),
            ("aromatic_0", aromatic == 0),
            ("aromatic_1_4", (aromatic >= 1) & (aromatic <= 4)),
            ("aromatic_ge_5", aromatic >= 5),
            ("hetero_0", hetero == 0),
            ("hetero_1_2", (hetero >= 1) & (hetero <= 2)),
            ("hetero_ge_3", hetero >= 3),
            ("heavy_low", heavy <= q_heavy[0]),
            ("heavy_high", heavy >= q_heavy[1]),
            ("smiles_short", length <= q_length[0]),
            ("smiles_long", length >= q_length[1]),
            ("model_disagreement_high", disagreement >= q_disagreement),
        ]
        summary = {"overall": stats(y, pred), "rows": int(len(frame)), "slices": {}}
        for name, mask in slice_specs:
            if int(np.sum(mask)) < 10:
                continue
            record = {"target": target, "slice": name, **stats(y[mask], pred[mask])}
            fold_values = []
            for fold in range(5):
                fold_mask = mask & (fold_ids == fold)
                if int(np.sum(fold_mask)) >= 5:
                    fold_values.append(float(np.mean(np.abs(pred[fold_mask] - y[fold_mask]))))
            record["fold_mean_abs_residual"] = float(np.mean(fold_values)) if fold_values else None
            record["fold_std_abs_residual"] = float(np.std(fold_values)) if fold_values else None
            summary["slices"][name] = record
            all_rows.append(record)
        for name in ("nearest_lt_0.30", "aux_count_0", "hetero_ge_3", "aromatic_ge_5", "heavy_high", "smiles_long", "model_disagreement_high"):
            item = summary["slices"].get(name)
            if item is not None and item["rows"] >= 20 and item["mean_abs_residual"] is not None:
                ratio = float(item["mean_abs_residual"] / summary["overall"]["mean_abs_residual"]) if summary["overall"]["mean_abs_residual"] else 0.0
                branch_candidates.append({"target": target, "slice": name, "rows": item["rows"], "residual_ratio": ratio, "fold_std_abs_residual": item["fold_std_abs_residual"]})
        target_summaries[target] = summary
    if not branch_candidates:
        branch = {"status": "none", "reason": "No sufficiently supported stable residual slice."}
    else:
        branch_candidates.sort(key=lambda item: (-item["residual_ratio"], item["target"], item["slice"]))
        winner = branch_candidates[0]
        mechanism = {
            "nearest_lt_0.30": "low-support extrapolation and domain shift",
            "aux_count_0": "missing official cross-property support",
            "hetero_ge_3": "heteroatom/polarity and hydrogen-bonding grammar",
            "aromatic_ge_5": "rigidity/aromatic backbone grammar",
            "heavy_high": "size/free-volume scaling",
            "smiles_long": "repeat-unit size and mobility scaling",
            "model_disagreement_high": "uncertainty-gated residual correction",
        }[winner["slice"]]
        branch = {
            "status": "proposed",
            "target": winner["target"],
            "slice": winner["slice"],
            "mechanism": mechanism,
            "changed_representation": "one bounded target-specific feature block matching the selected mechanism, with train-only OOF abstention",
            "expected_signal": "non-negative residual improvement in the selected slice and no worse than 0.003 grouped loss elsewhere",
            "stop_condition": "reject if the selected slice does not improve by at least 0.01 grouped R2 or any low-similarity/scaffold transfer panel is negative",
            "support": winner,
        }
    audit = {
        "schema_version": "ppp.round2.residual-diagnosis.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C006-20260803-1724-tg-portable-carrier",
        "official_inputs": inputs,
        "official_hashes_pass": all(inputs[name]["sha256"] == expected_hash for name, expected_hash in reference.EXPECTED_HASHES.items()),
        "c001_oof_sha256": sha256_file(oof_path),
        "targets": target_summaries,
        "branch_candidates": branch_candidates,
        "next_branch": branch,
        "candidate_changed": False,
        "elapsed_seconds": float(time.time() - started),
    }
    audit["decision"] = "diagnosis_complete" if branch["status"] == "proposed" else "pivot_without_branch"
    pd.DataFrame(all_rows).to_csv(run_dir / "slice_metrics.csv", index=False)
    write_json(run_dir / "diagnosis.json", audit)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.residual-diagnosis.v1", "seed": 2026, "folds": 5, "source": str(oof_path.relative_to(root)), "slices": [row["slice"] for row in all_rows]})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# R2-C008 residual diagnosis\n\nDecision: **{audit['decision']}**.\n\nThe frozen C001 candidate was not changed. The next branch proposal is recorded in diagnosis.json and must be preregistered separately before execution.\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    manifest_paths = [run_dir / name for name in ("config.json", "environment.txt", "slice_metrics.csv", "diagnosis.json", "decision.md", "command.txt", "protocol.json")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": audit["decision"], "next_branch": branch, "elapsed_seconds": audit["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
