#!/usr/bin/env python3
"""Official-only Egb endpoint/conjugation residual screen."""

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
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
SEED = 2026
TARGET = "egb"
ALPHA = 10.0
RESIDUAL_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fixed_features(molecules: list[Chem.Mol], indices: list[int]) -> tuple[np.ndarray, list[str]]:
    names = ["endpoint_count", "endpoint_bond_order_mean", "endpoint_neighbor_z_mean", "endpoint_neighbor_aromatic_fraction", "endpoint_neighbor_degree_mean", "endpoint_path_length", "atom_count", "heavy_atom_count", "ring_count", "aromatic_atom_count", "aromatic_ring_count", "hetero_atom_count", "halogen_count", "rotatable_bonds", "double_bond_count", "triple_bond_count", "conjugated_bond_count", "conjugated_fraction", "graph_diameter", "graph_mean_distance", "logp", "mol_mr", "exact_mol_wt", "tpsa", "fraction_csp3", "bertz_ct", "balaban_j", "hba", "hbd"]
    output = np.full((len(indices), len(names)), np.nan, dtype=np.float64)
    for row, source_index in enumerate(indices):
        molecule = molecules[source_index]
        try:
            atoms = list(molecule.GetAtoms()); bonds = list(molecule.GetBonds()); endpoints = [atom for atom in atoms if atom.GetAtomicNum() == 0]
            endpoint_bonds = [bond for bond in bonds if bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0]
            endpoint_neighbors = [bond.GetOtherAtom(endpoint) for endpoint in endpoints for bond in endpoint_bonds if bond.GetBeginAtomIdx() == endpoint.GetIdx() or bond.GetEndAtomIdx() == endpoint.GetIdx()]
            distance = np.asarray(Chem.GetDistanceMatrix(molecule, useBO=False), dtype=np.float64)
            upper = distance[np.triu_indices_from(distance, k=1)]
            conjugated = [bond for bond in bonds if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0]
            endpoint_path = float(Chem.GetShortestPath(molecule, endpoints[0].GetIdx(), endpoints[-1].GetIdx()).__len__() - 1) if len(endpoints) >= 2 else 0.0
            output[row] = [
                float(len(endpoints)), float(np.mean([bond.GetBondTypeAsDouble() for bond in endpoint_bonds])) if endpoint_bonds else 0.0,
                float(np.mean([atom.GetAtomicNum() for atom in endpoint_neighbors])) if endpoint_neighbors else 0.0,
                float(np.mean([atom.GetIsAromatic() for atom in endpoint_neighbors])) if endpoint_neighbors else 0.0,
                float(np.mean([atom.GetDegree() for atom in endpoint_neighbors])) if endpoint_neighbors else 0.0, endpoint_path,
                float(molecule.GetNumAtoms()), float(molecule.GetNumHeavyAtoms()), float(molecule.GetRingInfo().NumRings()),
                float(sum(atom.GetIsAromatic() for atom in atoms)), float(rdMolDescriptors.CalcNumAromaticRings(molecule)),
                float(sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)), float(sum(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)),
                float(Descriptors.NumRotatableBonds(molecule)), float(sum(bond.GetBondTypeAsDouble() == 2.0 for bond in bonds)), float(sum(bond.GetBondTypeAsDouble() == 3.0 for bond in bonds)),
                float(len(conjugated)), float(len(conjugated) / max(len(bonds), 1)), float(np.max(distance)) if distance.size else 0.0, float(np.mean(upper)) if upper.size else 0.0,
                float(Crippen.MolLogP(molecule)), float(Crippen.MolMR(molecule)), float(rdMolDescriptors.CalcExactMolWt(molecule)), float(rdMolDescriptors.CalcTPSA(molecule)), float(rdMolDescriptors.CalcFractionCSP3(molecule)), float(Descriptors.BertzCT(molecule)), float(Descriptors.BalabanJ(molecule)), float(rdMolDescriptors.CalcNumHBA(molecule)), float(rdMolDescriptors.CalcNumHBD(molecule)),
            ]
        except Exception:
            continue
    output[~np.isfinite(output)] = np.nan
    return output, names


def model() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ALPHA))


def nearest_similarity(fingerprints: list[object], query: np.ndarray, training: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[int(index)] for index in training]
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps)) for index in query], dtype=np.float64)


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups); members = {group: np.flatnonzero(groups == group) for group in unique}; rng = np.random.default_rng(SEED); values = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True); rows = np.concatenate([members[group] for group in selected]);
        if np.var(y[rows]) > 1.0e-15: values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, object], float]:
    report: dict[str, object] = {}; values: list[float] = []
    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected)); item = {"rows": count, "eligible_for_r2": False, "delta_r2": 0.0}
        if count >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected])); item.update({"eligible_for_r2": True, "delta_r2": delta}); values.append(delta)
        report[name] = item
    add("all_rows", np.ones(len(y), dtype=bool))
    for name, selected in {"similarity_lt_0.30": similarity < 0.30, "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50), "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70), "similarity_ge_0.70": similarity >= 0.70}.items(): add(name, selected)
    for scaffold in sorted(set(scaffolds)): add(f"scaffold_{scaffold}", scaffolds == scaffold, minimum=10)
    return report, (float(min(values)) if values else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--data-dir", default="ppp-round-2"); parser.add_argument("--run-dir", required=True); args = parser.parse_args()
    root = Path(args.root).resolve(); run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}: raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time(); train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve()); _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"])); key_to_index = {key: index for index, key in enumerate(keys)}; molecules = reference.build_molecules(keys); descriptor, _ = reference.descriptor_matrix(molecules); physical, _ = reference.physical_matrix(molecules, keys); dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False); cross_values, cross_available = reference.cross_property_arrays(pooled, keys); sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]; fingerprints = reference.morgan_bits(molecules, 2, 4096)
    detail, parent_oof_frame, _ = reference.fit_targets(pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, reference.DEFAULT_CONFIG); frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True); parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET].reset_index(drop=True)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), parent_rows["canonical"].astype(str).to_numpy()): raise RuntimeError("exact v7 Egb parent row alignment failed")
    y = frame["target"].to_numpy(float); parent_oof = parent_rows["prediction"].to_numpy(float); target_keys = sorted(set(frame["canonical"]) | set(test.loc[test["target_type"] == TARGET, "canonical"])); target_indices = [key_to_index[value] for value in target_keys]; features, feature_names = fixed_features(molecules, target_indices); feature_row = {value: row for row, value in enumerate(target_keys)}; rows = np.asarray([feature_row[value] for value in frame["canonical"]], dtype=np.int64); test_rows = np.asarray([feature_row[value] for value in test.loc[test["target_type"] == TARGET, "canonical"]], dtype=np.int64)
    groups = np.asarray([plumbing.no_stereo(value) for value in frame["canonical"]], dtype=object); scaffolds = np.asarray([plumbing.scaffold(value) for value in frame["canonical"]], dtype=object); folds = plumbing.folds_for(groups, 5); residual = y - parent_oof; candidate = np.full(len(y), np.nan); similarity = np.full(len(y), np.nan); fold_rows = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold); training = np.flatnonzero(folds != fold); fitted = model(); fitted.fit(features[rows[training]], residual[training]); candidate[validation] = parent_oof[validation] + RESIDUAL_WEIGHT * fitted.predict(features[rows[validation]]); global_validation = np.asarray([key_to_index[value] for value in frame.iloc[validation]["canonical"]], dtype=np.int64); global_training = np.asarray([key_to_index[value] for value in frame.iloc[training]["canonical"]], dtype=np.int64); similarity[validation] = nearest_similarity(fingerprints, global_validation, global_training); fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent_oof[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent_oof[validation]))})
    parent_r2 = float(r2_score(y, parent_oof)); candidate_r2 = float(r2_score(y, candidate)); delta = candidate_r2 - parent_r2; lower = bootstrap_lower(y, parent_oof, candidate, groups); panel, minimum_panel = panel_report(y, parent_oof, candidate, scaffolds, similarity)
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True); test_parent = detail[detail["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(float); fitted = model(); fitted.fit(features[rows], residual); test_candidate = test_parent + RESIDUAL_WEIGHT * fitted.predict(features[test_rows]); component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": test_parent, "candidate_prediction": test_candidate})
    if len(component) != 224 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all(): raise RuntimeError("Egb component output contract failed")
    component.to_csv(run_dir / "egb_component_predictions.csv", index=False); pd.DataFrame({"canonical": frame["canonical"].astype(str), "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel >= 0.0, "strict_no_regression": delta >= -0.003, "component_rows_pass": len(component) == 224}; passed = bool(all(gates.values())); source_names = ("round2_c063_egb_endpoint_conjugation_residual.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py")
    report = {"schema_version": "ppp.round2.c063.egb-endpoint-conjugation-residual.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 Egb regenerated in-process", "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False, "target": TARGET, "feature_names": feature_names, "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "group_bootstrap_lower": lower, "panels": panel, "minimum_panel_delta": minimum_panel, "gates": gates, "decision": "pass_component_gate" if passed else "rejected_component_gate", "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))}, "source_hashes": {name: sha256_file(root / "tools" / name) for name in source_names}, "elapsed_seconds": time.time() - started}
    write_json(run_dir / "metrics.json", report); write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "feature_family": "endpoint-path/conjugation/aromatic-fused/donor-acceptor official SMILES descriptors", "residual_weight": RESIDUAL_WEIGHT, "ridge_alpha": ALPHA, "outer": "canonical_no_stereo GroupKFold(5)", "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False}); (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8"); (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8"); (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Egb parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file(): manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items(): manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8"); print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel}, sort_keys=True))


if __name__ == "__main__":
    main()
