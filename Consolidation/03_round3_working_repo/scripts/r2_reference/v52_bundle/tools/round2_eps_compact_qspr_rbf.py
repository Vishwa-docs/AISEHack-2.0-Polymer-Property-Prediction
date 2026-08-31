#!/usr/bin/env python3
"""Strictly nested official-only compact-QSPR RBF EPS component."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGET = "eps"
FEATURE_COUNT = 28
KRR_ALPHA = 10.0
KRR_GAMMA = 1.0 / FEATURE_COUNT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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


def folds_for(groups: np.ndarray, n_splits: int) -> np.ndarray:
    if len(np.unique(groups)) < n_splits:
        raise RuntimeError(f"need {n_splits} groups, found {len(np.unique(groups))}")
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(500):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in chosen])
        if len(rows) > 1 and np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def nearest_similarity(fingerprints: list[object], validation: np.ndarray, training: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[index] for index in training]
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps)) for index in validation], dtype=float)


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def compact_qspr(molecule: Chem.Mol) -> np.ndarray:
    atoms = list(molecule.GetAtoms())
    bonds = list(molecule.GetBonds())
    counts = {number: sum(atom.GetAtomicNum() == number for atom in atoms) for number in (6, 7, 8, 15, 16)}
    values = [
        Descriptors.MolWt(molecule),
        Crippen.MolLogP(molecule),
        Crippen.MolMR(molecule),
        rdMolDescriptors.CalcTPSA(molecule),
        Descriptors.ExactMolWt(molecule),
        rdMolDescriptors.CalcLabuteASA(molecule),
        Descriptors.HeavyAtomCount(molecule),
        molecule.GetNumAtoms(),
        rdMolDescriptors.CalcNumRings(molecule),
        rdMolDescriptors.CalcNumAromaticRings(molecule),
        rdMolDescriptors.CalcNumAliphaticRings(molecule),
        rdMolDescriptors.CalcNumSaturatedRings(molecule),
        rdMolDescriptors.CalcNumHeterocycles(molecule),
        rdMolDescriptors.CalcNumHeteroatoms(molecule),
        Lipinski.NumHAcceptors(molecule),
        Lipinski.NumHDonors(molecule),
        Lipinski.NumRotatableBonds(molecule),
        rdMolDescriptors.CalcFractionCSP3(molecule),
        sum(atom.GetFormalCharge() for atom in atoms),
        sum(bond.GetBondTypeAsDouble() == 2.0 for bond in bonds),
        sum(bond.GetBondTypeAsDouble() == 3.0 for bond in bonds),
        sum(atom.GetIsAromatic() for atom in atoms),
        sum(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms),
        counts[6],
        counts[7],
        counts[8],
        counts[16],
        counts[15],
    ]
    if len(values) != FEATURE_COUNT:
        raise RuntimeError(f"compact QSPR feature count changed: {len(values)}")
    return np.asarray(values, dtype=np.float64)


def nested_parent(
    y: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    target_dense: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    global_indices: np.ndarray,
    y_global: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    inner_folds = folds_for(groups[outer_train], 4)
    inner_arms = np.full((len(outer_train), 4), np.nan, dtype=float)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_arms[inner_folds == fold] = reference.predict_base_models(
            target_dense, sparse_parts, fingerprints, y_global,
            global_indices[local_train], global_indices[local_validation], config, TARGET,
        )
    weights, intercept, blend_name, inner_r2 = reference.blend_from_oof(y[outer_train], inner_arms)
    outer_arms = reference.predict_base_models(
        target_dense, sparse_parts, fingerprints, y_global,
        global_indices[outer_train], global_indices[outer_validation], config, TARGET,
    )
    parent = reference.clip_prediction(y[outer_train], outer_arms @ weights + intercept)
    return parent, {"blend_name": blend_name, "blend_weights": [float(v) for v in weights], "blend_intercept": float(intercept), "inner_parent_r2": float(inner_r2), "inner_folds": inner_folds.tolist()}


def fit_qspr(features: np.ndarray, y: np.ndarray, train_local: np.ndarray, train_global: np.ndarray, prediction_global: np.ndarray):
    if len(train_local) != len(train_global):
        raise RuntimeError("local/global training index lengths differ")
    if np.any(train_global < 0) or np.any(train_global >= len(features)):
        raise RuntimeError("global QSPR feature index out of bounds")
    if not np.isfinite(y[train_local]).all():
        raise RuntimeError("non-finite EPS training labels")
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        KernelRidge(kernel="rbf", alpha=KRR_ALPHA, gamma=KRR_GAMMA),
    )
    model.fit(features[train_global], y[train_local])
    prediction = np.asarray(model.predict(features[prediction_global]), dtype=np.float64)
    return prediction


def build_panels(y, baseline, candidate, nearest, scaffolds, measurements, fallback):
    panels: dict[str, object] = {}
    values: list[float] = []

    def add(name: str, selected: np.ndarray, control: bool = False) -> None:
        delta = panel_delta(y, baseline, candidate, selected)
        rows = int(np.sum(selected))
        if rows < 5:
            status = "inapplicable_zero_support"
        elif delta is None:
            status = "incomplete_constant_support"
        elif control and float(np.max(np.abs(candidate[selected] - baseline[selected]))) > 1.0e-12:
            status = "failed_parent_only_control"
        else:
            status = "evaluable"
        panels[name] = {"rows": rows, "delta_r2": delta, "status": status}
        if delta is not None:
            values.append(delta)

    add("similarity_lt_0.30", nearest < 0.30)
    add("similarity_0.30_0.50", (nearest >= 0.30) & (nearest < 0.50))
    add("similarity_0.50_0.70", (nearest >= 0.50) & (nearest < 0.70))
    add("similarity_ge_0.70", nearest >= 0.70)
    add("exact_archive_measurements_ge_2", measurements >= 2)
    add("sparse_singleton_measurements_eq_1", measurements == 1)
    add("invalid_or_nonfinite_qspr_parent_only", fallback, control=True)
    for name in sorted(set(scaffolds)):
        selected = scaffolds == name
        if int(np.sum(selected)) >= 10:
            add(f"scaffold_slice_{name}", selected)
    incomplete = any(isinstance(value, dict) and value["status"] in {"incomplete_constant_support", "failed_parent_only_control"} for value in panels.values())
    return panels, (min(values) if values else None), incomplete


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
        raise RuntimeError(f"a protocol-only run directory is required: {run_dir}")
    start_time = datetime.now().astimezone()
    started = time.time()
    train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve())
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    features = np.vstack([compact_qspr(molecule) for molecule in molecules])
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    target_dense = reference.target_dense_features(dense_base, cross_values, cross_available, TARGET)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.text_matrix(keys, int(reference.DEFAULT_CONFIG["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"]))
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    canonical = frame["canonical"].to_numpy(object)
    groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([scaffold(value) for value in canonical], dtype=object)
    measurements = frame["measurements"].to_numpy(int)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=float)
    y_global[global_indices] = y
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "compact_qspr_features": FEATURE_COUNT, "krr_alpha": KRR_ALPHA, "krr_gamma": KRR_GAMMA})
    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan, dtype=float)
    candidate = np.full(len(y), np.nan, dtype=float)
    nearest = np.full(len(y), np.nan, dtype=float)
    fallback = np.zeros(len(y), dtype=bool)
    qspr_prediction = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        parent, parent_meta = nested_parent(y, groups, training, validation, target_dense, sparse_parts, fingerprints, global_indices, y_global, config)
        prediction = fit_qspr(features, y, training, global_indices[training], global_indices[validation])
        valid = np.isfinite(prediction)
        routed_candidate = parent.copy()
        routed_candidate[valid] = reference.clip_prediction(y[training], prediction[valid])
        baseline[validation] = parent
        candidate[validation] = routed_candidate
        fallback[validation] = ~valid
        qspr_prediction[validation] = prediction
        nearest[validation] = nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
        parent_score = float(r2_score(y[validation], parent))
        candidate_score = float(r2_score(y[validation], routed_candidate))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "baseline_r2": parent_score, "candidate_r2": candidate_score, "delta_r2": candidate_score - parent_score, "fallback_rows": int(np.sum(~valid)), "outer_fold_validation_groups": sorted(set(groups[validation])), "parent_blend": parent_meta})

    scaffold_holdout: dict[str, object] = {}
    for scaffold_name in sorted(set(scaffolds)):
        validation = np.flatnonzero(scaffolds == scaffold_name)
        if len(validation) < 10:
            continue
        training = np.flatnonzero(scaffolds != scaffold_name)
        parent_holdout, _ = nested_parent(y, groups, training, validation, target_dense, sparse_parts, fingerprints, global_indices, y_global, config)
        prediction = fit_qspr(features, y, training, global_indices[training], global_indices[validation])
        valid = np.isfinite(prediction)
        holdout_candidate = parent_holdout.copy()
        holdout_candidate[valid] = reference.clip_prediction(y[training], prediction[valid])
        base_score = float(r2_score(y[validation], parent_holdout))
        cand_score = float(r2_score(y[validation], holdout_candidate))
        scaffold_holdout[str(scaffold_name)] = {"rows": int(len(validation)), "baseline_r2": base_score, "candidate_r2": cand_score, "delta_r2": cand_score - base_score, "fallback_rows": int(np.sum(~valid))}

    panels, min_panel, panel_incomplete = build_panels(y, baseline, candidate, nearest, scaffolds, measurements, fallback)
    baseline_score = float(r2_score(y, baseline))
    candidate_score = float(r2_score(y, candidate))
    delta = candidate_score - baseline_score
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    scaffold_min = min((float(value["delta_r2"]) for value in scaffold_holdout.values()), default=None)
    positive_folds = int(sum(float(row["delta_r2"]) > 0.0 for row in fold_rows))
    report = {
        "rows": int(len(y)),
        "canonical_groups": int(len(np.unique(groups))),
        "baseline_r2_nested_parent": baseline_score,
        "candidate_r2_compact_qspr_rbf": candidate_score,
        "delta_r2": delta,
        "positive_outer_folds": positive_folds,
        "group_r2_bootstrap_lower": float(bootstrap),
        "outer_folds": fold_rows,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": scaffold_min,
        "min_panel_delta": min_panel,
        "panel_incomplete": panel_incomplete,
        "panels": panels,
        "route_definition": {"fallback_rows": int(np.sum(fallback)), "fallback_max_change": float(np.max(np.abs(candidate[fallback] - baseline[fallback]))) if np.any(fallback) else 0.0},
        "support_counts": {"exact_archive_measurement_rows": int(np.sum(measurements >= 2)), "singleton_rows": int(np.sum(measurements == 1))},
    }
    passed = bool(delta >= 0.01 and positive_folds >= 4 and bootstrap > 0.0 and (min_panel is None or min_panel >= -0.003) and (scaffold_min is None or scaffold_min >= -0.003) and not panel_incomplete and report["route_definition"]["fallback_max_change"] <= 1.0e-12)
    report["pass"] = passed
    report["decision"] = "component_pass" if passed else "rejected_component_gate"
    source_hashes = {"script": sha256_file(Path(__file__).resolve()), "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py")}
    config_hash = reference.canonical_json_hash(config)
    oof = pd.DataFrame({"canonical": canonical, "target_type": TARGET, "target": y, "baseline": baseline, "candidate": candidate, "fallback": fallback, "qspr_prediction": qspr_prediction, "nearest_similarity": nearest, "scaffold": scaffolds, "no_stereo_group": groups, "measurements": measurements, "outer_fold": main_folds})
    oof.to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "config.json", config)
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    metrics = {"schema_version": "ppp.round2.eps-compact-qspr-rbf-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C044-20260803-2130-eea-scaffold-conditioned-residual", "lineage_parent": "R2-C044-20260803-2130-eea-scaffold-conditioned-residual", "baseline_reference": "C001 official clean incumbent, freshly regenerated as nested parent", "official_inputs": inputs, "target": TARGET, "features": {"count": FEATURE_COUNT}, "model": {"kernel": "rbf", "alpha": KRR_ALPHA, "gamma": KRR_GAMMA}, "metrics": report, "source_hashes": source_hashes, "config_sha256": config_hash, "pass": passed, "decision": report["decision"], "elapsed_seconds": float(time.time() - started)}
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. No candidate or local_eval diagnostic was created by this component run.\n", encoding="utf-8")
    finish_time = datetime.now().astimezone()
    (run_dir / "run.log").write_text(f"experiment_id={run_dir.name}\nstarted_at={start_time.isoformat()}\nfinished_at={finish_time.isoformat()}\ndecision={report['decision']}\nelapsed_seconds={metrics['elapsed_seconds']:.3f}\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([f"{source_hashes['script']}  SOURCE tools/round2_eps_compact_qspr_rbf.py", f"{source_hashes['reference_module']}  SOURCE tools/initial_reference_pipeline.py"])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "delta_r2": delta, "passing": passed, "elapsed_seconds": metrics["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
