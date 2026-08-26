#!/usr/bin/env python3
"""Fixed official-only Ei QSPR interaction screen against the exact v7 route."""

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

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_mixed_candidate_v7 as v7


RDLogger.DisableLog("rdApp.*")
SEED = 2026
TARGET = "ei"
ALPHA = 10.0
DESCRIPTOR_NAMES = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "FractionCSP3", "HeavyAtomCount",
    "NumAliphaticRings", "NumAromaticRings", "BalabanJ", "BertzCT",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
)
PHYSICAL_NAMES = (
    "smiles_length", "atom_count", "heavy_atom_count", "ring_count",
    "aromatic_atom_count", "hetero_atom_count", "halogen_count",
    "rotatable_bonds_approx", "double_bond_count", "branch_count",
    "n_count", "o_count", "s_count", "si_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def feature_matrix(descriptor: np.ndarray, descriptor_names: list[str], physical: np.ndarray, physical_names: list[str]) -> tuple[np.ndarray, list[str]]:
    descriptor_index = {name: index for index, name in enumerate(descriptor_names)}
    physical_index = {name: index for index, name in enumerate(physical_names)}
    missing = [name for name in DESCRIPTOR_NAMES if name not in descriptor_index]
    missing.extend(name for name in PHYSICAL_NAMES if name not in physical_index)
    if missing:
        raise RuntimeError(f"fixed QSPR feature list missing: {missing}")
    indices = [descriptor_index[name] for name in DESCRIPTOR_NAMES]
    p_indices = [physical_index[name] for name in PHYSICAL_NAMES]
    values = np.hstack([descriptor[:, indices], physical[:, p_indices]]).astype(np.float64, copy=False)
    names = list(DESCRIPTOR_NAMES) + list(PHYSICAL_NAMES)
    values[~np.isfinite(values)] = np.nan
    return values, names


def model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        Ridge(alpha=ALPHA),
    )


def panels(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, object], float]:
    values: dict[str, object] = {}
    deltas: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected))
        item: dict[str, object] = {"rows": count, "eligible_for_r2": False, "delta_r2": 0.0}
        if count >= minimum and float(np.var(y[selected])) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"eligible_for_r2": True, "delta_r2": delta})
            deltas.append(delta)
        values[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
        "similarity_unavailable": ~np.isfinite(similarity),
    }.items():
        add(name, selected)
    for scaffold in sorted(set(scaffolds)):
        add(f"scaffold_{scaffold}", scaffolds == scaffold, minimum=10)
    return values, (float(min(deltas)) if deltas else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    features, feature_names = feature_matrix(descriptor, descriptor_names, physical, physical_names)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": SEED, "folds": 5, "mixed_candidate": True, "special_targets": ["ei", "eea"]})
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=int(config["morgan_bits"])),
        reference.morgan_count_matrix(molecules, radius=3, bits=int(config["morgan_bits"])),
        reference.text_matrix(keys, int(config["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(config["morgan_bits"]))

    # This is the exact v7 Ei route, regenerated in this process from official inputs.
    special_oof, special_test, parent_report = v7.specialized_target(
        TARGET, pooled, test, keys, dense_base, cross_values, cross_available,
        sparse_parts, fingerprints, config,
    )
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), special_oof["canonical"].astype(str).to_numpy()):
        raise RuntimeError("exact v7 Ei OOF row alignment failed")
    y = frame["target"].to_numpy(float)
    groups = np.asarray([plumbing.no_stereo(value) for value in frame["canonical"]], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in frame["canonical"]], dtype=object)
    row_indices = np.asarray([key_to_index[value] for value in frame["canonical"]], dtype=np.int64)
    parent = special_oof["candidate"].to_numpy(float)
    if not np.isfinite(parent).all():
        raise RuntimeError("exact v7 Ei parent has non-finite OOF predictions")
    folds = plumbing.folds_for(groups, 5)
    candidate = np.full(len(y), np.nan, dtype=float)
    similarity = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        fitted = model()
        fitted.fit(features[row_indices[training]], y[training])
        candidate[validation] = np.asarray(fitted.predict(features[row_indices[validation]]), dtype=float)
        train_fps = [fingerprints[int(index)] for index in row_indices[training]]
        similarity[validation] = np.asarray(
            [max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps)) for index in row_indices[validation]],
            dtype=float,
        )
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "parent_r2": float(r2_score(y[validation], parent[validation])),
            "candidate_r2": float(r2_score(y[validation], candidate[validation])),
            "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation])),
        })
    if not np.isfinite(candidate).all():
        raise RuntimeError("non-finite Ei QSPR OOF predictions")
    panels_report, minimum_panel = panels(y, parent, candidate, groups, scaffolds, similarity)
    baseline_r2 = float(r2_score(y, parent))
    candidate_r2 = float(r2_score(y, candidate))
    bootstrap_lower = float(plumbing.bootstrap_r2_lower(y, parent, candidate, groups))
    test_frame = test[test["target_type"] == TARGET].reset_index(drop=True)
    test_indices = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
    full_model = model()
    full_model.fit(features[row_indices], y)
    test_prediction = np.asarray(full_model.predict(features[test_indices]), dtype=float)
    if not np.isfinite(test_prediction).all() or len(test_prediction) != 148:
        raise RuntimeError("Ei test component contract failed")
    component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": special_test["target"].to_numpy(float), "candidate_prediction": test_prediction})
    if not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or component["id"].duplicated().any():
        raise RuntimeError("Ei test component order failed")
    component.to_csv(run_dir / "ei_component_predictions.csv", index=False)
    pd.DataFrame({"canonical": frame["canonical"], "target": y, "parent": parent, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    report = {
        "schema_version": "ppp.round2.c059.ei-symbolic-qspr-interactions.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact Ei specialized route regenerated in-process",
        "official_inputs": inputs,
        "official_only": True, "external_label_file_read": False, "local_eval_read": False,
        "target": TARGET, "descriptor_names": list(feature_names), "polynomial": {"degree": 2, "interaction_only": True, "alpha": ALPHA},
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))},
        "parent_r2": baseline_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - baseline_r2,
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows,
        "group_bootstrap_lower": bootstrap_lower, "panels": panels_report, "minimum_panel_delta": minimum_panel,
        "gates": {"gain_pass": bool(candidate_r2 - baseline_r2 >= 0.01), "fold_pass": bool(sum(row["delta_r2"] > 0 for row in fold_rows) >= 4), "bootstrap_pass": bool(bootstrap_lower > 0.0), "panel_pass": bool(minimum_panel >= 0.0), "component_rows_pass": bool(len(component) == 148)},
        "decision": "pass_component_gate" if (candidate_r2 - baseline_r2 >= 0.01 and sum(row["delta_r2"] > 0 for row in fold_rows) >= 4 and bootstrap_lower > 0.0 and minimum_panel >= 0.0) else "rejected_component_gate",
        "source_hashes": {name: sha256_file(root / "tools" / name) for name in ("round2_c059_ei_symbolic_qspr_interactions.py", "round2_mixed_candidate_v7.py", "initial_reference_pipeline.py", "round2_ei_scaffold_abstaining_gap_identity_v4_portable.py", "round2_eea_scaffold_abstaining_gap_identity_v7_portable.py", "round2_eea_cross_target_oof_residual_stack.py")},
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "descriptor_names": list(feature_names), "polynomial_degree": 2, "interaction_only": True, "ridge_alpha": ALPHA, "outer_folds": "canonical_no_stereo GroupKFold(5)", "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Ei parent R2 `{baseline_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{candidate_r2 - baseline_r2:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": baseline_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - baseline_r2, "positive_folds": report["positive_folds"], "group_bootstrap_lower": bootstrap_lower, "minimum_panel_delta": minimum_panel}, sort_keys=True))


if __name__ == "__main__":
    main()
