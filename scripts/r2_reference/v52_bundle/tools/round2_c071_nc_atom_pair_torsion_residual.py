#!/usr/bin/env python3
"""Official-only Nc atom-pair/topological-torsion residual screen."""

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
import round2_c063_egb_endpoint_conjugation_residual as fixed_features_module
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGET = "nc"
SEED = 2026
RESIDUAL_WEIGHT = 0.20
ALPHA = 30.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def physics_features(molecules: list[Chem.Mol], indices: list[int]) -> tuple[np.ndarray, list[str]]:
    names = [
        "hetero_density", "polar_group_density", "tpsa_density", "mr_density", "logp_density",
        "hba_density", "hbd_density", "aromatic_fraction", "aromatic_ring_fraction",
        "rotatable_density", "ring_density", "double_bond_density", "conjugated_density",
        "halogen_density", "nitrogen_density", "oxygen_density", "sulfur_density",
        "formal_charge", "absolute_formal_charge", "charge_density", "size_log_heavy",
    ]
    output = np.full((len(indices), len(names)), np.nan, dtype=np.float64)
    for row, source_index in enumerate(indices):
        molecule = molecules[source_index]
        try:
            atoms = list(molecule.GetAtoms())
            bonds = list(molecule.GetBonds())
            heavy = max(molecule.GetNumHeavyAtoms(), 1)
            bonds_count = max(len(bonds), 1)
            hetero = sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)
            hba = float(rdMolDescriptors.CalcNumHBA(molecule))
            hbd = float(rdMolDescriptors.CalcNumHBD(molecule))
            tpsa = float(rdMolDescriptors.CalcTPSA(molecule))
            mol_mr = float(Crippen.MolMR(molecule))
            logp = float(Crippen.MolLogP(molecule))
            aromatic_atoms = sum(atom.GetIsAromatic() for atom in atoms)
            aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(molecule))
            rings = max(molecule.GetRingInfo().NumRings(), 1)
            rotatable = float(Descriptors.NumRotatableBonds(molecule))
            double_bonds = sum(bond.GetBondTypeAsDouble() == 2.0 for bond in bonds)
            conjugated = sum(bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0 for bond in bonds)
            halogens = sum(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms)
            nitrogens = sum(atom.GetAtomicNum() == 7 for atom in atoms)
            oxygens = sum(atom.GetAtomicNum() == 8 for atom in atoms)
            sulfurs = sum(atom.GetAtomicNum() == 16 for atom in atoms)
            charge = float(sum(atom.GetFormalCharge() for atom in atoms))
            output[row] = [
                hetero / heavy,
                (hetero + hba + hbd) / heavy,
                tpsa / heavy,
                mol_mr / heavy,
                logp / heavy,
                hba / heavy,
                hbd / heavy,
                aromatic_atoms / heavy,
                aromatic_rings / rings,
                rotatable / heavy,
                molecule.GetRingInfo().NumRings() / heavy,
                double_bonds / bonds_count,
                conjugated / bonds_count,
                halogens / heavy,
                nitrogens / heavy,
                oxygens / heavy,
                sulfurs / heavy,
                charge,
                abs(charge),
                abs(charge) / heavy,
                np.log1p(heavy),
            ]
        except Exception:
            continue
    output[~np.isfinite(output)] = np.nan
    return output, names


def fingerprint_features(molecules: list[Chem.Mol], indices: list[int], bits: int = 1024) -> tuple[np.ndarray, list[str]]:
    output = np.zeros((len(indices), bits * 2), dtype=np.float64)
    for row, source_index in enumerate(indices):
        molecule = molecules[source_index]
        atom_pair = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(molecule, nBits=bits)
        torsion = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(molecule, nBits=bits)
        DataStructs.ConvertToNumpyArray(atom_pair, output[row, :bits])
        DataStructs.ConvertToNumpyArray(torsion, output[row, bits:])
    names = [f"atom_pair_{index}" for index in range(bits)] + [f"torsion_{index}" for index in range(bits)]
    return output, names


def model() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ALPHA))


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


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
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, 4096),
        reference.morgan_count_matrix(molecules, 3, 4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, 4096)
    detail, parent_oof_frame, _ = reference.fit_targets(
        pooled, test, keys, dense_base, cross_values, cross_available,
        sparse_parts, fingerprints, reference.DEFAULT_CONFIG,
    )
    replay_detail, replay_parent_oof_frame, _ = reference.fit_targets(
        pooled, test, keys, dense_base, cross_values, cross_available,
        sparse_parts, fingerprints, reference.DEFAULT_CONFIG,
    )
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET].reset_index(drop=True)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), parent_rows["canonical"].astype(str).to_numpy()):
        raise RuntimeError("exact v7 Nc parent row alignment failed")
    replay_parent_rows = replay_parent_oof_frame[replay_parent_oof_frame["target_type"] == TARGET].reset_index(drop=True)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), replay_parent_rows["canonical"].astype(str).to_numpy()):
        raise RuntimeError("exact v7 Nc replay row alignment failed")
    y = frame["target"].to_numpy(float)
    parent_oof = parent_rows["prediction"].to_numpy(float)
    target_keys = sorted(set(frame["canonical"]) | set(test.loc[test["target_type"] == TARGET, "canonical"]))
    target_indices = [key_to_index[value] for value in target_keys]
    base_features, base_names = fixed_features_module.fixed_features(molecules, target_indices)
    extra_features, extra_names = physics_features(molecules, target_indices)
    pair_features, pair_names = fingerprint_features(molecules, target_indices)
    features = np.hstack([base_features, extra_features, pair_features])
    feature_names = base_names + extra_names + pair_names
    feature_row = {value: row for row, value in enumerate(target_keys)}
    rows = np.asarray([feature_row[value] for value in frame["canonical"]], dtype=np.int64)
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True)
    test_rows = np.asarray([feature_row[value] for value in test_frame["canonical"]], dtype=np.int64)
    groups = np.asarray([plumbing.no_stereo(value) for value in frame["canonical"]], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in frame["canonical"]], dtype=object)
    folds = plumbing.folds_for(groups, 5)
    residual = y - parent_oof
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    similarity = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        fitted = model()
        fitted.fit(features[rows[training]], residual[training])
        candidate[validation] = parent_oof[validation] + RESIDUAL_WEIGHT * fitted.predict(features[rows[validation]])
        global_validation = np.asarray([key_to_index[value] for value in frame.iloc[validation]["canonical"]], dtype=np.int64)
        global_training = np.asarray([key_to_index[value] for value in frame.iloc[training]["canonical"]], dtype=np.int64)
        similarity[validation] = fixed_features_module.nearest_similarity(fingerprints, global_validation, global_training)
        parent_score = float(r2_score(y[validation], parent_oof[validation]))
        candidate_score = float(r2_score(y[validation], candidate[validation]))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": parent_score, "candidate_r2": candidate_score, "delta_r2": candidate_score - parent_score})
    parent_oof_replay = replay_parent_rows["prediction"].to_numpy(float)
    parent_oof_replay_max_abs = float(np.max(np.abs(parent_oof - parent_oof_replay)))
    if not np.isfinite(candidate).all():
        raise RuntimeError("non-finite Nc OOF candidate")
    parent_r2 = float(r2_score(y, parent_oof))
    candidate_r2 = float(r2_score(y, candidate))
    delta = candidate_r2 - parent_r2
    lower = bootstrap_lower(y, parent_oof, candidate, groups)
    panels, minimum_panel = fixed_features_module.panel_report(y, parent_oof, candidate, scaffolds, similarity)
    fitted = model()
    fitted.fit(features[rows], residual)
    test_parent = detail[detail["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(float)
    test_replay_parent = replay_detail[replay_detail["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(float)
    parent_test_replay_max_abs = float(np.max(np.abs(test_parent - test_replay_parent)))
    test_candidate = test_parent + RESIDUAL_WEIGHT * fitted.predict(features[test_rows])
    component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": test_parent, "candidate_prediction": test_candidate})
    if len(component) != 153 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all():
        raise RuntimeError("Nc component output contract failed")
    component.to_csv(run_dir / "nc_component_predictions.csv", index=False)
    pd.DataFrame({"canonical": frame["canonical"].astype(str), "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel >= 0.0, "component_rows_pass": len(component) == 153, "parent_replay_oof_pass": parent_oof_replay_max_abs <= 1.0e-12, "parent_replay_test_pass": parent_test_replay_max_abs <= 1.0e-12}
    passed = bool(all(gates.values()))
    source_names = ("round2_c071_nc_atom_pair_torsion_residual.py", "round2_c063_egb_endpoint_conjugation_residual.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py")
    report = {
        "schema_version": "ppp.round2.c071.nc-atom-pair-torsion-residual.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 Nc parent regenerated twice in-process",
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "target": TARGET,
        "feature_names": feature_names,
        "residual_weight": RESIDUAL_WEIGHT,
        "parent_r2": parent_r2,
        "candidate_r2": candidate_r2,
        "delta_r2": delta,
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "group_bootstrap_lower": lower,
        "panels": panels,
        "minimum_panel_delta": minimum_panel,
        "parent_replay_oof_max_abs": parent_oof_replay_max_abs,
        "parent_replay_test_max_abs": parent_test_replay_max_abs,
        "gates": gates,
        "decision": "pass_component_gate" if passed else "rejected_component_gate",
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))},
        "source_hashes": {name: sha256_file(root / "tools" / name) for name in source_names},
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "feature_family": "official-SMILES atom-pair/topological-torsion counts plus deterministic size/aromatic/electronic descriptors", "model": "Ridge", "ridge_alpha": ALPHA, "residual_weight": RESIDUAL_WEIGHT, "atom_pair_bits": 1024, "topological_torsion_bits": 1024, "outer": "canonical_no_stereo GroupKFold(5)", "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False, "parent_replay_oof_max_abs": parent_oof_replay_max_abs, "parent_replay_test_max_abs": parent_test_replay_max_abs})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Nc parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel}, sort_keys=True))


if __name__ == "__main__":
    main()
