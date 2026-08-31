#!/usr/bin/env python3
"""Official-only deterministic 3D Tg residual screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
SEED = 2026
TARGET = "tg"
ALPHA = 10.0
RESIDUAL_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def shape_features(molecules: list[Chem.Mol], indices: list[int]) -> tuple[np.ndarray, dict[str, int]]:
    names = [
        "pmi1", "pmi2", "pmi3", "npr1", "npr2", "asphericity", "eccentricity",
        "inertial_shape_factor", "radius_of_gyration", "spherocity", "pbf",
        "labute_asa", "exact_mol_wt", "tpsa", "fraction_csp3",
    ]
    out = np.full((len(indices), len(names)), np.nan, dtype=np.float64)
    success = 0
    for row, source_index in enumerate(indices):
        original = molecules[source_index]
        molecule = Chem.AddHs(Chem.Mol(original))
        params = AllChem.ETKDGv3()
        params.randomSeed = int(SEED + source_index)
        params.numThreads = 1
        params.pruneRmsThresh = 0.1
        try:
            embedded = int(AllChem.EmbedMolecule(molecule, params))
            if embedded < 0:
                continue
            try:
                AllChem.UFFOptimizeMolecule(molecule, maxIters=200, confId=embedded)
            except Exception:
                pass
            values = [
                Descriptors3D.PMI1(molecule), Descriptors3D.PMI2(molecule), Descriptors3D.PMI3(molecule),
                Descriptors3D.NPR1(molecule), Descriptors3D.NPR2(molecule), Descriptors3D.Asphericity(molecule),
                Descriptors3D.Eccentricity(molecule), Descriptors3D.InertialShapeFactor(molecule),
                Descriptors3D.RadiusOfGyration(molecule), Descriptors3D.SpherocityIndex(molecule),
                Descriptors3D.PBF(molecule), rdMolDescriptors.CalcLabuteASA(original),
                rdMolDescriptors.CalcExactMolWt(original), rdMolDescriptors.CalcTPSA(original),
                rdMolDescriptors.CalcFractionCSP3(original),
            ]
            values = np.asarray(values, dtype=np.float64)
            values[~np.isfinite(values)] = np.nan
            out[row] = values
            success += 1
        except Exception:
            continue
    return out, {"attempted": len(indices), "conformer_success": success, "conformer_failure": len(indices) - success, "feature_names": names}


def fit_model() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ALPHA))


def nearest_similarity(fingerprints: list[object], query: np.ndarray, training: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[int(index)] for index in training]
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps)) for index in query], dtype=np.float64)


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, object], float]:
    report: dict[str, object] = {}
    values: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected))
        item: dict[str, object] = {"rows": count, "eligible_for_r2": False, "delta_r2": 0.0}
        if count >= minimum and float(np.var(y[selected])) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"eligible_for_r2": True, "delta_r2": delta})
            values.append(delta)
        report[name] = item

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
    return report, (float(min(values)) if values else 0.0)


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
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    detail, parent_oof_frame, parent_report = reference.fit_targets(pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, reference.DEFAULT_CONFIG)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET].reset_index(drop=True)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), parent_rows["canonical"].astype(str).to_numpy()):
        raise RuntimeError("exact v7 Tg parent row alignment failed")
    y = frame["target"].to_numpy(dtype=float)
    parent_oof = parent_rows["prediction"].to_numpy(dtype=float)
    target_keys = sorted(set(frame["canonical"]) | set(test.loc[test["target_type"] == TARGET, "canonical"]))
    target_indices = [key_to_index[value] for value in target_keys]
    three_d, descriptor_report = shape_features(molecules, target_indices)
    target_key_to_row = {value: row for row, value in enumerate(target_keys)}
    feature_rows = np.asarray([target_key_to_row[value] for value in frame["canonical"]], dtype=np.int64)
    groups = np.asarray([plumbing.no_stereo(value) for value in frame["canonical"]], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in frame["canonical"]], dtype=object)
    folds = plumbing.folds_for(groups, 5)
    candidate = np.full(len(y), np.nan, dtype=float)
    similarity = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []
    residual = y - parent_oof
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        fitted = fit_model()
        fitted.fit(three_d[feature_rows[training]], residual[training])
        correction = np.asarray(fitted.predict(three_d[feature_rows[validation]]), dtype=float)
        candidate[validation] = parent_oof[validation] + RESIDUAL_WEIGHT * correction
        global_validation = np.asarray([key_to_index[value] for value in frame.iloc[validation]["canonical"]], dtype=np.int64)
        global_training = np.asarray([key_to_index[value] for value in frame.iloc[training]["canonical"]], dtype=np.int64)
        similarity[validation] = nearest_similarity(fingerprints, global_validation, global_training)
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent_oof[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent_oof[validation]))})
    if not np.isfinite(candidate).all():
        raise RuntimeError("non-finite Tg residual candidate")
    panel, minimum_panel = panel_report(y, parent_oof, candidate, groups, scaffolds, similarity)
    parent_r2 = float(r2_score(y, parent_oof)); candidate_r2 = float(r2_score(y, candidate)); delta = candidate_r2 - parent_r2
    bootstrap_lower = float(plumbing.bootstrap_r2_lower(y, parent_oof, candidate, groups))
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True)
    test_feature_rows = np.asarray([target_key_to_row[value] for value in test_frame["canonical"]], dtype=np.int64)
    full_model = fit_model(); full_model.fit(three_d[feature_rows], residual)
    test_parent = detail[detail["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(dtype=float)
    test_candidate = test_parent + RESIDUAL_WEIGHT * np.asarray(full_model.predict(three_d[test_feature_rows]), dtype=float)
    component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": test_parent, "candidate_prediction": test_candidate})
    if len(component) != 2763 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all():
        raise RuntimeError("Tg component output contract failed")
    component.to_csv(run_dir / "tg_component_predictions.csv", index=False)
    pd.DataFrame({"canonical": frame["canonical"].astype(str), "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    strict_candidate = np.full(len(y), np.nan, dtype=float)
    strict_folds = plumbing.folds_for(groups, 5)
    for fold in range(5):
        validation = np.flatnonzero(strict_folds == fold); training = np.flatnonzero(strict_folds != fold)
        fitted = fit_model(); fitted.fit(three_d[feature_rows[training]], residual[training])
        strict_candidate[validation] = parent_oof[validation] + RESIDUAL_WEIGHT * fitted.predict(three_d[feature_rows[validation]])
    strict_delta = float(r2_score(y, strict_candidate) - r2_score(y, parent_oof))
    gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4, "bootstrap_pass": bootstrap_lower > 0.0, "panel_pass": minimum_panel >= 0.0, "strict_no_regression": strict_delta >= -0.003, "component_rows_pass": len(component) == 2763}
    passed = bool(all(gates.values()))
    source_names = ("round2_c061_tg_3d_shape_free_volume_residual.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py")
    report = {"schema_version": "ppp.round2.c061.tg-3d-shape-free-volume-residual.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 Tg regenerated in-process", "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False, "target": TARGET, "descriptor_report": descriptor_report, "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "group_bootstrap_lower": bootstrap_lower, "panels": panel, "minimum_panel_delta": minimum_panel, "strict_group_delta": strict_delta, "gates": gates, "decision": "pass_component_gate" if passed else "rejected_component_gate", "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))}, "source_hashes": {name: sha256_file(root / "tools" / name) for name in source_names}, "elapsed_seconds": time.time() - started}
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "etkdg": "ETKDGv3 one conformer randomSeed=2026+index maxAttempts=20", "uff_max_iters": 200, "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": ALPHA, "outer": "canonical_no_stereo GroupKFold(5)", "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Tg parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file(): manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items(): manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": bootstrap_lower, "minimum_panel_delta": minimum_panel, "conformer_success": descriptor_report["conformer_success"]}, sort_keys=True))


if __name__ == "__main__":
    main()
