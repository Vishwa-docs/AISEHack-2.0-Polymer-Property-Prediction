#!/usr/bin/env python3
"""C101: clean target-specific sparse fingerprint refresh.

The parent is rebuilt from the official Round 2 sources.  EPS, Nc, and Tg get
one fixed residual Ridge over official-SMILES sparse fingerprint families that
were not present in the C050 carrier; the other four targets remain unchanged.
No counterpart labels, external_label files, pretrained assets, or prior predictions
are used by the new route.
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, rdFingerprintGenerator, rdMolDescriptors
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c076_eps_paired_charge_polarizability_residual as paired_features
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_mixed_candidate_v7 as mixed


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
CHANGED = ("eps", "nc", "tg")
SEED = 2026
RESIDUAL_WEIGHT = 0.20
RIDGE_ALPHA = 30.0
FP_BITS = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def folds_for(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def sparse_bit_matrix(fingerprints: list[Any], bits: int) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for row, fingerprint in enumerate(fingerprints):
        array = np.zeros(bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        active = np.flatnonzero(array)
        rows.extend([row] * len(active))
        columns.extend(active.tolist())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(len(fingerprints), bits))


def sparse_count_matrix(molecules: list[Chem.Mol], kind: str, bits: int, radius: int | None = None) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    if kind == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=bits)
        get_values = lambda mol: generator.GetCountFingerprint(mol).GetNonzeroElements()
    elif kind == "feature_morgan":
        get_values = lambda mol: rdMolDescriptors.GetHashedMorganFingerprint(mol, int(radius), nBits=bits, useFeatures=True).GetNonzeroElements()
    elif kind == "atom_pair":
        get_values = lambda mol: rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=bits).GetNonzeroElements()
    elif kind == "torsion":
        get_values = lambda mol: rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol, nBits=bits).GetNonzeroElements()
    else:
        raise ValueError(kind)
    for row, molecule in enumerate(molecules):
        for column, count in get_values(molecule).items():
            rows.append(row)
            columns.append(int(column))
            values.append(float(np.log1p(float(count))))
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(molecules), bits), dtype=np.float32)


def rich_fingerprint_matrix(molecules: list[Chem.Mol], include_slow_families: bool = True, minimal: bool = False) -> tuple[sparse.csr_matrix, dict[str, int]]:
    blocks: list[sparse.csr_matrix] = []
    counts: dict[str, int] = {}
    for radius in (1, 4, 5):
        block = sparse_count_matrix(molecules, "morgan", FP_BITS, radius)
        blocks.append(block)
        counts[f"morgan_count_r{radius}"] = int(block.shape[1])
    if not minimal:
        for kind, radius in (("feature_morgan_bit", 2), ("feature_morgan_count", 2)):
            if kind.endswith("bit"):
                fingerprints = [AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=FP_BITS, useFeatures=True) for molecule in molecules]
                block = sparse_bit_matrix(fingerprints, FP_BITS)
            else:
                block = sparse_count_matrix(molecules, "feature_morgan", FP_BITS, radius)
            blocks.append(block)
            counts[kind] = int(block.shape[1])
    if include_slow_families:
        for kind in ("atom_pair", "torsion"):
            block = sparse_count_matrix(molecules, kind, FP_BITS)
            blocks.append(block)
            counts[kind] = int(block.shape[1])
    if not minimal:
        rdk_fingerprints = [RDKFingerprint(molecule, fpSize=FP_BITS) for molecule in molecules]
        rdk = sparse_bit_matrix(rdk_fingerprints, FP_BITS)
        blocks.append(rdk)
        counts["rdk_bits"] = int(rdk.shape[1])
    maccs_fingerprints = [MACCSkeys.GenMACCSKeys(molecule) for molecule in molecules]
    maccs = sparse_bit_matrix(maccs_fingerprints, 167)
    blocks.append(maccs)
    counts["maccs_bits"] = int(maccs.shape[1])
    return sparse.hstack(blocks, format="csr", dtype=np.float32), counts


def model() -> Any:
    return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=RIDGE_ALPHA))


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


def target_panel(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, Any], float]:
    panels: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        rows = int(np.sum(selected))
        item: dict[str, Any] = {"rows": rows, "delta_r2": 0.0, "status": "inapplicable"}
        if rows >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"delta_r2": delta, "status": "evaluable"})
            deltas.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    add("similarity_lt_0.30", similarity < 0.30)
    add("similarity_0.30_0.50", (similarity >= 0.30) & (similarity < 0.50))
    add("similarity_0.50_0.70", (similarity >= 0.50) & (similarity < 0.70))
    add("similarity_ge_0.70", similarity >= 0.70)
    for name in sorted(set(scaffolds)):
        add(f"scaffold_{name}", scaffolds == name, minimum=10)
    return panels, float(min(deltas)) if deltas else 0.0


def build_parent(root: Path, data_dir: Path, replay: bool = True) -> dict[str, Any]:
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
    if replay:
        replay_detail, replay_oof, _ = reference.fit_targets(
            pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
        )
        normal_oof_diff = float(np.max(np.abs(parent_oof["prediction"].to_numpy(float) - replay_oof["prediction"].to_numpy(float))))
        parent_test_diff = float(np.max(np.abs(parent_detail["model_prediction"].to_numpy(float) - replay_detail["model_prediction"].to_numpy(float))))
    else:
        normal_oof_diff = None
        parent_test_diff = None
    test_detail = parent_detail[["id", "target_type", "model_prediction"]].copy()
    target_info: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        if target in ("ei", "eea"):
            special_oof, special_test, _ = mixed.specialized_target(
                target, pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, config
            )
            canonical = special_oof["canonical"].to_numpy(object)
            y = special_oof["target"].to_numpy(float)
            parent = special_oof["candidate"].to_numpy(float)
            replacement = special_test.set_index("id")["target"]
            mask = test_detail["target_type"].to_numpy(object) == target
            test_detail.loc[mask, "model_prediction"] = test_detail.loc[mask, "id"].map(replacement).astype(float).to_numpy()
        else:
            rows = parent_oof[parent_oof["target_type"] == target].reset_index(drop=True)
            canonical = rows["canonical"].to_numpy(object)
            y = rows["target"].to_numpy(float)
            parent = rows["prediction"].to_numpy(float)
        groups = np.asarray([plumbing.no_stereo(value) for value in canonical], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in canonical], dtype=object)
        target_info[target] = {"canonical": canonical, "y": y, "parent": parent, "groups": groups, "scaffolds": scaffolds}
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
        "fingerprints": fingerprints,
        "target_info": target_info,
        "test_detail": test_detail,
        "parent_report": parent_report,
        "parent_replay_oof_max_abs": normal_oof_diff,
        "parent_replay_test_max_abs": parent_test_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fast", action="store_true", help="omit the slow atom-pair/torsion bank for a bounded screen")
    parser.add_argument("--defer-parent-replay", action="store_true", help="defer independent parent replay until a candidate passes")
    parser.add_argument("--minimal", action="store_true", help="use only Morgan count and MACCS blocks")
    parser.add_argument("--endpoint-path", action="store_true", help="use the Round 1 endpoint-path n-gram matrix instead of fingerprint blocks")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    bundle = build_parent(root, (root / args.data_dir).resolve(), replay=not args.defer_parent_replay)
    feature_keys = sorted(
        {value for target in CHANGED for value in bundle["target_info"][target]["canonical"]}
        | set(bundle["test"].loc[bundle["test"]["target_type"].isin(CHANGED), "canonical"])
    )
    feature_indices = [bundle["key_to_index"][value] for value in feature_keys]
    feature_molecules = [bundle["molecules"][index] for index in feature_indices]
    if args.endpoint_path:
        round1_tools = root.parent / "Polymer Prediction Challenge" / "tools"
        if str(round1_tools) not in sys.path:
            sys.path.insert(0, str(round1_tools))
        import polymer_official_train_eval_loop as round1_engine
        rich_features, endpoint_report = round1_engine.endpoint_path_ngram_matrix(feature_molecules, n_features=4096, max_bonds=8)
        feature_counts = {"endpoint_path": int(rich_features.shape[1]), "endpoint_path_nnz": int(rich_features.nnz), "endpoint_path_tokens": int(endpoint_report["token_count"])}
    else:
        rich_features, feature_counts = rich_fingerprint_matrix(feature_molecules, include_slow_families=not args.fast, minimal=args.minimal)
    dense_parts: list[sparse.csr_matrix] = []
    dense_names: list[str] = []
    endpoint, endpoint_names = fixed_features.fixed_features(feature_molecules, list(range(len(feature_molecules))))
    physics, physics_names = paired_features.physics_features(feature_molecules, list(range(len(feature_molecules))))
    dense = np.hstack([endpoint, physics]).astype(np.float64, copy=False)
    dense[~np.isfinite(dense)] = 0.0
    full_features = sparse.hstack([rich_features, sparse.csr_matrix(dense, dtype=np.float32)], format="csr")
    feature_counts["endpoint_physics_dense"] = int(dense.shape[1])
    feature_counts["total"] = int(full_features.shape[1])
    feature_counts["feature_pool_rows"] = int(len(feature_keys))
    key_to_index = bundle["key_to_index"]
    feature_key_to_row = {key: row for row, key in enumerate(feature_keys)}
    raw_test_predictions = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    components: list[pd.DataFrame] = []
    for target in TARGETS:
        info = bundle["target_info"][target]
        y = info["y"]
        parent = info["parent"]
        groups = info["groups"]
        scaffolds = info["scaffolds"]
        candidate = parent.copy()
        similarity = np.full(len(y), np.nan, dtype=np.float64)
        pair_rows = 0
        if target in CHANGED:
            folds = folds_for(groups)
            global_rows = np.asarray([feature_key_to_row[value] for value in info["canonical"]], dtype=np.int64)
            for fold in range(5):
                validation = np.flatnonzero(folds == fold)
                training = np.flatnonzero(folds != fold)
                fitted = model()
                fitted.fit(full_features[global_rows[training]], (y - parent)[training])
                candidate[validation] = parent[validation] + RESIDUAL_WEIGHT * fitted.predict(full_features[global_rows[validation]])
                global_validation = global_rows[validation]
                global_training = global_rows[training]
                similarity[validation] = fixed_features.nearest_similarity(bundle["fingerprints"], global_validation, global_training)
            test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            test_rows = np.asarray([feature_key_to_row[value] for value in test_frame["canonical"]], dtype=np.int64)
            fitted = model()
            fitted.fit(full_features[global_rows], y - parent)
            test_parent = raw_test_predictions.loc[raw_test_predictions["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float)
            test_candidate = test_parent + RESIDUAL_WEIGHT * fitted.predict(full_features[test_rows])
            mask = raw_test_predictions["target_type"].to_numpy(object) == target
            raw_test_predictions.loc[mask, "model_prediction"] = test_candidate
            components.append(pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": target, "parent_prediction": test_parent, "candidate_prediction": test_candidate}))
            pair_rows = int(len(y))
            folds_for_report = folds
        else:
            folds_for_report = np.full(len(y), -1, dtype=np.int64)
            target_reports[target] = {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "folds": [], "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}}, "feature_count": 0, "pass": True, "unchanged_parent": True}
        if target in CHANGED:
            fold_rows = []
            for fold in range(5):
                validation = np.flatnonzero(folds_for_report == fold)
                parent_score = float(r2_score(y[validation], parent[validation]))
                candidate_score = float(r2_score(y[validation], candidate[validation]))
                fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": parent_score, "candidate_r2": candidate_score, "delta_r2": candidate_score - parent_score})
            panels, minimum_panel = target_panel(y, parent, candidate, groups, scaffolds, similarity)
            delta = float(r2_score(y, candidate) - r2_score(y, parent))
            lower = bootstrap_lower(y, parent, candidate, groups)
            positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
            passed = bool(delta >= 0.01 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0)
            target_reports[target] = {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "folds": fold_rows, "panels": panels, "feature_count": int(full_features.shape[1]), "pair_rows": pair_rows, "pass": passed, "unchanged_parent": False}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds_for_report, "changed": target in CHANGED}))
    raw_detail, override_report = reference.apply_official_overrides(raw_test_predictions, bundle["test"], bundle["raw_labels"])
    submission = raw_detail[["id", "target"]].copy()
    if len(submission) != len(bundle["test"]) or not submission["id"].equals(bundle["test"]["id"]):
        raise RuntimeError("C101 output ID/order mismatch")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C101 output contains duplicate or non-finite values")
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(target_reports[target]["delta_r2"] for target in TARGETS))
    complete_pass = bool(mean_candidate > mean_parent and max_loss >= -0.003 and all(target_reports[target]["pass"] for target in CHANGED))
    oof = pd.concat(oof_parts, ignore_index=True)
    component = pd.concat(components, ignore_index=True)
    submission.to_csv(run_dir / "predictions.csv", index=False)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    component.to_csv(run_dir / "component_predictions.csv", index=False)
    source_paths = {
        "script": root / "tools" / "round2_c101_rich_sparse_fingerprint_refresh.py",
        "reference": root / "tools" / "initial_reference_pipeline.py",
        "mixed_parent_route": root / "tools" / "round2_mixed_candidate_v7.py",
        "fixed_features": root / "tools" / "round2_c063_egb_endpoint_conjugation_residual.py",
        "paired_features": root / "tools" / "round2_c076_eps_paired_charge_polarizability_residual.py",
        "metric_plumbing": root / "tools" / "round2_eea_cross_target_oof_residual_stack.py",
    }
    report = {
        "schema_version": "ppp.round2.c101.rich-sparse-fingerprint-refresh.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "C050 rebuilt twice from official sources; Ei/Eea use C050 special routes",
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
        "parent_replay_oof_max_abs": bundle["parent_replay_oof_max_abs"],
        "parent_replay_test_max_abs": bundle["parent_replay_test_max_abs"],
        "official_override_report": override_report,
        "feature_counts": feature_counts,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "elapsed_seconds": float(time.time() - started),
        "decision": "candidate_pending_fresh_replay" if complete_pass else "rejected_full_candidate_gate",
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "changed_targets": list(CHANGED), "features": feature_counts, "fingerprint_bits": FP_BITS, "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": RIDGE_ALPHA, "outer": "canonical_no_stereo GroupKFold(5)", "external_label_file_read": False, "local_eval_read": False, "parent_replay_oof_max_abs": bundle["parent_replay_oof_max_abs"], "parent_replay_test_max_abs": bundle["parent_replay_test_max_abs"]})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nscipy={sparse.__version__ if hasattr(sparse, '__version__') else 'installed'}\nrdkit={Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent R2 `{mean_parent:.12f}`; candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  ../../../tools/{source_paths[name].name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "target_deltas": {target: target_reports[target]["delta_r2"] for target in TARGETS}, "parent_replay_oof_max_abs": bundle["parent_replay_oof_max_abs"], "parent_replay_test_max_abs": bundle["parent_replay_test_max_abs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
