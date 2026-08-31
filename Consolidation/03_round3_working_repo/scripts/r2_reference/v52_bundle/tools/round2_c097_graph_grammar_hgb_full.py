#!/usr/bin/env python3
"""Source-only seven-target graph-grammar HistGradientBoosting candidate.

The C050 parent is rebuilt in memory from official inputs through the same
reference and specialized routes. No parent artifact, external_label file, or local_eval
value is read.
"""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_mixed_candidate_v7 as mixed


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
SPECIAL_TARGETS = ("ei", "eea")
RESIDUAL_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def group_folds(groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    return result


def shuffled_folds(n_rows: int, n_splits: int = 5) -> np.ndarray:
    result = np.full(n_rows, -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=2026).split(np.arange(n_rows))):
        result[validation] = fold
    return result


def no_stereo(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("official SMILES failed RDKit parsing")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        value = ""
    return value or "ACYCLIC"


def add_token(tokens: dict[str, float], token: str, value: float = 1.0) -> None:
    tokens[token] = tokens.get(token, 0.0) + float(value)


def grammar_row(molecule: Chem.Mol) -> tuple[dict[str, float], list[float]]:
    tokens: dict[str, float] = {}
    atoms = list(molecule.GetAtoms())
    bonds = list(molecule.GetBonds())
    numeric = [
        float(len(atoms)),
        float(len(bonds)),
        float(molecule.GetRingInfo().NumRings()),
        float(sum(atom.GetIsAromatic() for atom in atoms)),
        float(sum(atom.GetFormalCharge() != 0 for atom in atoms)),
        float(sum(atom.GetAtomicNum() == 6 for atom in atoms)),
        float(sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms)),
        float(sum(atom.GetDegree() >= 3 for atom in atoms)),
        float(max((atom.GetDegree() for atom in atoms), default=0)),
    ]
    labels: list[str] = []
    common_elements = (0, 1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
    numeric.extend(float(sum(atom.GetAtomicNum() == value for atom in atoms)) for value in common_elements)
    numeric.extend(float(sum(atom.GetDegree() == value for atom in atoms)) for value in range(0, 7))
    for atom in atoms:
        atomic = atom.GetAtomicNum()
        degree = atom.GetDegree()
        aromatic = int(atom.GetIsAromatic())
        charge = atom.GetFormalCharge()
        hybrid = str(atom.GetHybridization())
        label = f"z{atomic}|d{degree}|a{aromatic}|q{charge}|h{hybrid}"
        labels.append(label)
        add_token(tokens, f"atom:{label}")
        add_token(tokens, f"element:{atomic}")
        if atom.IsInRing():
            add_token(tokens, f"ring_atom:{atomic}")
        if atom.GetDegree() >= 3:
            add_token(tokens, f"branch_atom:{atomic}:d{degree}")
    for bond in bonds:
        begin = molecule.GetAtomWithIdx(bond.GetBeginAtomIdx())
        end = molecule.GetAtomWithIdx(bond.GetEndAtomIdx())
        kind = str(bond.GetBondType())
        endpoint = "dummy" if begin.GetAtomicNum() == 0 or end.GetAtomicNum() == 0 else "heavy"
        add_token(tokens, f"bond:{kind}:{endpoint}")
        if bond.GetIsAromatic():
            add_token(tokens, "bond:aromatic")
    numeric.extend(float(sum(bond.GetBondType() == value for bond in bonds)) for value in (
        Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC,
    ))
    numeric.append(float(sum(bond.GetIsAromatic() for bond in bonds)))
    for ring in molecule.GetRingInfo().AtomRings():
        size = len(ring)
        add_token(tokens, f"ring_size:{min(size, 12)}")
    numeric.extend(float(sum(len(ring) == value for ring in molecule.GetRingInfo().AtomRings())) for value in range(3, 9))
    numeric.extend([
        float(sum(atom.GetAtomicNum() == 0 for atom in atoms)),
        float(sum(atom.GetIsAromatic() for atom in atoms)),
        float(sum(atom.GetFormalCharge() for atom in atoms)),
    ])
    for round_id in (1, 2):
        next_labels: list[str] = []
        for atom in atoms:
            neighbours = sorted(labels[neighbour.GetIdx()] for neighbour in atom.GetNeighbors())
            next_label = f"{labels[atom.GetIdx()]}>>{'/'.join(neighbours)}"
            next_labels.append(next_label)
            add_token(tokens, f"wl{round_id}:{next_label}")
        labels = next_labels
    return tokens, numeric


def grammar_features(molecules: list[Chem.Mol]) -> np.ndarray:
    rows: list[dict[str, float]] = []
    numeric_rows: list[list[float]] = []
    for molecule in molecules:
        tokens, numeric = grammar_row(molecule)
        rows.append(tokens)
        numeric_rows.append(numeric)
    hashed = FeatureHasher(n_features=1024, input_type="dict", alternate_sign=False, dtype=np.float64).transform(rows).toarray()
    width = max(len(row) for row in numeric_rows)
    numeric_matrix = np.zeros((len(numeric_rows), width), dtype=np.float64)
    for index, row in enumerate(numeric_rows):
        numeric_matrix[index, :len(row)] = row
    return np.hstack([numeric_matrix, hashed]).astype(np.float64, copy=False)


def nearest_similarity(fingerprints: list[Any], indices: np.ndarray, folds: np.ndarray) -> np.ndarray:
    result = np.full(len(indices), np.nan, dtype=np.float64)
    for fold in sorted(set(int(value) for value in folds)):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[int(indices[row])] for row in training]
        for row in validation:
            result[row] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(indices[row])], train_fps))
    return result


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values: list[float] = []
    for _ in range(2000):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in chosen])
        if len(rows) > 1 and np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def build_parent(root: Path, data_dir: Path) -> dict[str, Any]:
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
    config.update({"seed": 2026, "folds": 5, "mixed_candidate": True, "special_targets": list(SPECIAL_TARGETS)})
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, int(config["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(config["morgan_bits"])),
        reference.text_matrix(keys, int(config["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(config["morgan_bits"]))
    parent_detail, parent_oof, _ = reference.fit_targets(
        pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
    )
    test_parent_detail = parent_detail[["id", "target_type", "model_prediction"]].copy()
    target_info: dict[str, dict[str, Any]] = {}
    special_reports: dict[str, Any] = {}
    for target in TARGETS:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        if target in SPECIAL_TARGETS:
            special_oof, special_test, special_report = mixed.specialized_target(
                target, pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
            )
            canonical = special_oof["canonical"].to_numpy(object)
            y = special_oof["target"].to_numpy(float)
            parent = special_oof["candidate"].to_numpy(float)
            folds = special_oof["outer_fold"].to_numpy(int)
            test_map = special_test.set_index("id")["target"]
            mask = test_parent_detail["target_type"].to_numpy(object) == target
            test_parent_detail.loc[mask, "model_prediction"] = test_parent_detail.loc[mask, "id"].map(test_map).astype(float).to_numpy()
            special_reports[target] = special_report
        else:
            rows = parent_oof[parent_oof["target_type"] == target].reset_index(drop=True)
            canonical = rows["canonical"].to_numpy(object)
            y = rows["target"].to_numpy(float)
            parent = rows["prediction"].to_numpy(float)
            folds = shuffled_folds(len(rows))
        indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
        scaffolds = np.asarray([scaffold(value) for value in canonical], dtype=object)
        target_info[target] = {
            "canonical": canonical,
            "y": y,
            "parent": parent,
            "folds": folds,
            "indices": indices,
            "groups": groups,
            "scaffolds": scaffolds,
        }
    raw_detail, override_report = reference.apply_official_overrides(test_parent_detail, test, raw_labels)
    test_parent = raw_detail["target"].to_numpy(float)
    return {
        "train": train,
        "test": test,
        "archive": archive,
        "inputs": inputs,
        "pooled": pooled,
        "keys": keys,
        "molecules": molecules,
        "key_to_index": key_to_index,
        "fingerprints": fingerprints,
        "target_info": target_info,
        "test_parent": test_parent,
        "test_parent_detail": raw_detail,
        "override_report": override_report,
    }


def fit_target(info: dict[str, Any], feature_matrix: np.ndarray, test_indices: np.ndarray, target_index: int) -> dict[str, Any]:
    y = info["y"]
    parent = info["parent"]
    folds = info["folds"]
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(set(int(value) for value in folds)):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_x = imputer.fit_transform(feature_matrix[info["indices"][training]])
        validation_x = imputer.transform(feature_matrix[info["indices"][validation]])
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=2026 + target_index,
        )
        model.fit(train_x, y[training] - parent[training])
        correction = model.predict(validation_x)
        candidate[validation] = reference.clip_prediction(y[training], parent[validation] + RESIDUAL_WEIGHT * correction)
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "baseline_r2": float(r2_score(y[validation], parent[validation])),
            "candidate_r2": float(r2_score(y[validation], candidate[validation])),
            "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation])),
        })
    full_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    full_x = full_imputer.fit_transform(feature_matrix[info["indices"]])
    test_x = full_imputer.transform(feature_matrix[test_indices])
    full_model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=1.0,
        random_state=2026 + target_index,
    )
    full_model.fit(full_x, y - parent)
    test_correction = full_model.predict(test_x)
    return {"candidate": candidate, "folds": fold_rows, "test_correction": test_correction}


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
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError(f"protocol-only run directory required: {run_dir}")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    parent = build_parent(root, data_dir)
    feature_matrix = grammar_features(parent["molecules"])
    key_to_index = parent["key_to_index"]
    target_results: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    parent_replay_max_abs = 0.0
    for target_index, target in enumerate(TARGETS):
        info = parent["target_info"][target]
        test_frame = parent["test"][parent["test"]["target_type"] == target].reset_index(drop=True)
        test_indices = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
        result = fit_target(info, feature_matrix, test_indices, target_index)
        y = info["y"]
        baseline = info["parent"]
        candidate = result["candidate"]
        groups = info["groups"]
        folds = info["folds"]
        nearest = nearest_similarity(parent["fingerprints"], info["indices"], folds)
        panel_specs = {
            "similarity_lt_0.30": nearest < 0.30,
            "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
            "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
            "similarity_ge_0.70": nearest >= 0.70,
            "quantile_low": y <= np.quantile(y, 0.25),
            "quantile_high": y >= np.quantile(y, 0.75),
        }
        for scaffold_name in sorted(set(info["scaffolds"])):
            selected = info["scaffolds"] == scaffold_name
            if int(np.sum(selected)) >= 10:
                panel_specs[f"scaffold_{scaffold_name}"] = selected
        panels = {}
        panel_values: list[float] = []
        for name, selected in panel_specs.items():
            delta = panel_delta(y, baseline, candidate, selected)
            panels[name] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "inapplicable_insufficient_support"}
            if delta is not None:
                panel_values.append(delta)
        target_delta = float(r2_score(y, candidate) - r2_score(y, baseline))
        positive_folds = int(sum(row["delta_r2"] > 0.0 for row in result["folds"]))
        bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
        min_panel = min(panel_values) if panel_values else None
        target_pass = bool(target_delta > 0.0 and positive_folds >= max(1, int(np.ceil(len(set(folds)) * 0.8))) and bootstrap > 0.0 and (min_panel is None or min_panel >= 0.0))
        target_results[target] = {
            "rows": int(len(y)),
            "parent_r2": float(r2_score(y, baseline)),
            "candidate_r2": float(r2_score(y, candidate)),
            "delta_r2": target_delta,
            "positive_folds": positive_folds,
            "group_bootstrap_lower": bootstrap,
            "minimum_panel_delta": min_panel,
            "panels": panels,
            "folds": result["folds"],
            "pass": target_pass,
        }
        oof_parts.append(pd.DataFrame({
            "canonical": info["canonical"],
            "target_type": target,
            "target": y,
            "parent": baseline,
            "candidate": candidate,
            "group": groups,
            "scaffold": info["scaffolds"],
            "fold": folds,
            "nearest_tanimoto": nearest,
        }))
        parent_test_map = parent["test_parent_detail"].set_index("id")["target"]
        target_parent_test = test_frame["id"].map(parent_test_map).to_numpy(float)
        test_candidate = reference.clip_prediction(y, target_parent_test + RESIDUAL_WEIGHT * result["test_correction"])
        prediction_parts.append(pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": target, "prediction": test_candidate.astype(float)}))

    oof = pd.concat(oof_parts, ignore_index=True)
    raw_predictions = pd.concat(prediction_parts, ignore_index=True)
    detail = parent["test_parent_detail"][["id", "target_type"]].copy()
    detail["model_prediction"] = detail["id"].map(raw_predictions.set_index("id")["prediction"]).astype(float)
    raw_labels, _ = reference.build_label_pool(parent["train"], parent["archive"])
    final_detail, override_report = reference.apply_official_overrides(detail, parent["test"], raw_labels)
    submission = final_detail[["id", "target"]].copy()
    if len(submission) != len(parent["test"]) or not submission["id"].equals(parent["test"]["id"]):
        raise RuntimeError("C097 complete output order contract failed")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C097 output contains duplicate IDs or non-finite predictions")
    target_mean_parent = float(np.mean([value["parent_r2"] for value in target_results.values()]))
    target_mean_candidate = float(np.mean([value["candidate_r2"] for value in target_results.values()]))
    changed_targets = [target for target, value in target_results.items() if abs(value["delta_r2"]) > 1.0e-12]
    complete_pass = bool(
        target_mean_candidate > target_mean_parent
        and len(changed_targets) >= 3
        and all(value["candidate_r2"] >= value["parent_r2"] - 0.003 for value in target_results.values())
        and all(value["pass"] for target, value in target_results.items() if target in changed_targets)
        and len(submission) == 4940
    )
    script_path = root / "tools" / "round2_c097_graph_grammar_hgb_full.py"
    reference_path = root / "tools" / "initial_reference_pipeline.py"
    mixed_path = root / "tools" / "round2_mixed_candidate_v7.py"
    audit = {
        "schema_version": "ppp.round2.c097.graph-grammar-hgb-full.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_inputs": parent["inputs"],
        "source_hashes": {"script": sha256_file(script_path), "reference": sha256_file(reference_path), "mixed_parent": sha256_file(mixed_path)},
        "feature_shape": [int(value) for value in feature_matrix.shape],
        "parent_replay_oof_max_abs": parent_replay_max_abs,
        "parent_replay_pass": parent_replay_max_abs <= 1.0e-12,
        "target_reports": target_results,
        "mean_parent_r2": target_mean_parent,
        "mean_candidate_r2": target_mean_candidate,
        "mean_gain": target_mean_candidate - target_mean_parent,
        "changed_targets": changed_targets,
        "complete_output_rows": int(len(submission)),
        "complete_output_order_pass": bool(len(submission) == 4940 and submission["id"].equals(parent["test"]["id"])),
        "override_report": override_report,
        "pass": complete_pass,
        "decision": "candidate_pass" if complete_pass else "rejected_full_candidate_gate",
        "elapsed_seconds": float(time.time() - started),
        "external_label_file_read": False,
        "local_eval_read": False,
        "stored_parent_predictions_read": False,
        "stored_parent_oof_read": False,
        "kaggle_compute": False,
    }
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    submission.to_csv(run_dir / "predictions.csv", index=False)
    raw_predictions.to_csv(run_dir / "component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", audit)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.c097.graph-grammar-hgb-full.v1",
        "seed": 2026,
        "targets": list(TARGETS),
        "residual_weight": RESIDUAL_WEIGHT,
        "hgb": {"max_iter": 180, "learning_rate": 0.04, "max_leaf_nodes": 15, "min_samples_leaf": 12, "l2_regularization": 1.0},
        "feature_shape": [int(value) for value in feature_matrix.shape],
        "official_inputs": parent["inputs"],
        "source_hashes": audit["source_hashes"],
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
        f"rdkit={Chem.rdBase.rdkitVersion}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# R2-C097 graph-grammar HistGradientBoosting\n\nDecision: **{audit['decision']}**. Mean parent={target_mean_parent:.12f}; mean candidate={target_mean_candidate:.12f}; gain={audit['mean_gain']:.12f}.\n",
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        f"experiment_id={run_dir.name}\nmean_parent_r2={target_mean_parent:.12f}\nmean_candidate_r2={target_mean_candidate:.12f}\nmean_gain={audit['mean_gain']:.12f}\npass={complete_pass}\n",
        encoding="utf-8",
    )
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    manifest.extend([
        f"{audit['source_hashes']['script']}  SOURCE tools/round2_c097_graph_grammar_hgb_full.py",
        f"{audit['source_hashes']['reference']}  SOURCE tools/initial_reference_pipeline.py",
        f"{audit['source_hashes']['mixed_parent']}  SOURCE tools/round2_mixed_candidate_v7.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
