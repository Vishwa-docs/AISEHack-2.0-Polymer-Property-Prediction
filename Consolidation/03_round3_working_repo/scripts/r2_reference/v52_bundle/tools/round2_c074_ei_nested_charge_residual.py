#!/usr/bin/env python3
"""Official-only strictly nested Ei electronic residual screen."""

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
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdPartialCharges
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features_module
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_ei_scaffold_abstaining_gap_identity_v4_portable as ei_route


RDLogger.DisableLog("rdApp.*")
TARGET = "ei"
SEED = 2026
RESIDUAL_WEIGHT = 0.20
HUBER_EPSILON = 1.35
HUBER_ALPHA = 0.0001
HUBER_MAX_ITER = 500


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


def model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        HuberRegressor(epsilon=HUBER_EPSILON, alpha=HUBER_ALPHA, max_iter=HUBER_MAX_ITER),
    )


def electronic_features(molecules: list[Chem.Mol], indices: list[int]) -> tuple[np.ndarray, list[str]]:
    """Compute charge features after capping polymer dummy atoms as carbon.

    Gasteiger charges on the official repeat-unit SMILES otherwise return ``-nan``
    for every row because the ``*`` endpoints are unsatisfied dummy atoms. The
    cap is a deterministic representation-only operation and never changes a
    target or training row.
    """
    names = [
        "gasteiger_mean", "gasteiger_std", "gasteiger_min", "gasteiger_max",
        "gasteiger_range", "abs_gasteiger_mean", "abs_gasteiger_density",
        "hetero_abs_gasteiger_mean", "charge_weighted_distance",
        "charge_weighted_hetero_distance",
    ]
    output = np.full((len(indices), len(names)), np.nan, dtype=np.float64)
    for row, source_index in enumerate(indices):
        try:
            capped = Chem.RWMol(Chem.Mol(molecules[source_index]))
            for atom in capped.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(6)
                    atom.SetFormalCharge(0)
                    atom.SetNoImplicit(False)
            molecule = capped.GetMol()
            Chem.SanitizeMol(molecule)
            rdPartialCharges.ComputeGasteigerCharges(molecule)
            charges = np.asarray([float(atom.GetProp("_GasteigerCharge")) for atom in molecule.GetAtoms()], dtype=np.float64)
            if not np.isfinite(charges).all():
                continue
            distances = np.asarray(Chem.GetDistanceMatrix(molecule, useBO=False), dtype=np.float64)
            atoms = list(molecule.GetAtoms())
            hetero = np.asarray([atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms], dtype=bool)
            weights = np.abs(charges[:, None] * charges[None, :])
            total_weight = max(float(weights.sum()), 1.0e-12)
            hetero_weights = weights * (hetero[:, None] | hetero[None, :])
            hetero_total = max(float(hetero_weights.sum()), 1.0e-12)
            heavy = max(molecule.GetNumHeavyAtoms(), 1)
            output[row] = [
                float(np.mean(charges)), float(np.std(charges)), float(np.min(charges)), float(np.max(charges)),
                float(np.ptp(charges)), float(np.mean(np.abs(charges))), float(np.sum(np.abs(charges)) / heavy),
                float(np.mean(np.abs(charges[hetero]))) if np.any(hetero) else 0.0,
                float((weights * distances).sum() / total_weight),
                float((hetero_weights * distances).sum() / hetero_total),
            ]
        except Exception:
            continue
    output[~np.isfinite(output)] = np.nan
    return output, names


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


def strict_specialized_split(
    y: np.ndarray,
    groups: np.ndarray,
    scaffolds: np.ndarray,
    canonical: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    target_dense: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    global_indices: np.ndarray,
    y_global: np.ndarray,
    config: dict[str, object],
    gap_table: pd.DataFrame,
    maps: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return outer validation and outer-training parent predictions.

    The inner parent predictions are generated only from outer-training rows;
    the outer validation groups and scaffolds are excluded from the gap model.
    This closes the parent/residual nesting defect identified after C073.
    """
    inner_folds = ei_route.folds_for(groups[outer_train], 4)
    inner_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        local_positions = np.flatnonzero(inner_folds == fold)
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[local_positions]
        inner_arms[local_positions] = reference.predict_base_models(
            target_dense, sparse_parts, fingerprints, y_global,
            global_indices[local_train], global_indices[local_validation], config, TARGET,
        )
    weights, intercept, blend_name, inner_r2 = reference.blend_from_oof(y[outer_train], inner_arms)
    inner_base = reference.clip_prediction(y[outer_train], inner_arms @ weights + intercept)
    outer_arms = reference.predict_base_models(
        target_dense, sparse_parts, fingerprints, y_global,
        global_indices[outer_train], global_indices[outer_validation], config, TARGET,
    )
    outer_base = reference.clip_prediction(y[outer_train], outer_arms @ weights + intercept)
    inner_parent = np.full(len(outer_train), np.nan, dtype=np.float64)
    gap_groups = set(gap_table["group"].astype(str))
    gap_scaffolds = set(gap_table["scaffold"].astype(str))
    for fold in range(4):
        local_positions = np.flatnonzero(inner_folds == fold)
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[local_positions]
        forbidden_groups = gap_groups - set(groups[local_train].astype(str))
        forbidden_scaffolds = gap_scaffolds - set(scaffolds[local_train].astype(str))
        gap_model, _ = ei_route.fit_gap_model(gap_table, forbidden_groups, forbidden_scaffolds)
        raw, _, _, _ = ei_route.route_predictions(
            TARGET, canonical[local_validation], inner_base[local_positions], y[local_train], gap_model, maps,
        )
        global_validation = global_indices[local_validation]
        global_training = global_indices[local_train]
        nearest = fixed_features_module.nearest_similarity(fingerprints, global_validation, global_training)
        allowed = (nearest < 0.70) & ~np.isin(scaffolds[local_validation], ["c1ccsc1"])
        raw[~allowed] = inner_base[local_positions][~allowed]
        inner_parent[local_positions] = raw
    forbidden_groups = gap_groups - set(groups[outer_train].astype(str))
    forbidden_scaffolds = gap_scaffolds - set(scaffolds[outer_train].astype(str))
    gap_model, gap_rows = ei_route.fit_gap_model(gap_table, forbidden_groups, forbidden_scaffolds)
    raw, _, _, _ = ei_route.route_predictions(
        TARGET, canonical[outer_validation], outer_base, y[outer_train], gap_model, maps,
    )
    nearest = fixed_features_module.nearest_similarity(
        fingerprints, global_indices[outer_validation], global_indices[outer_train],
    )
    allowed = (nearest < 0.70) & ~np.isin(scaffolds[outer_validation], ["c1ccsc1"])
    raw[~allowed] = outer_base[~allowed]
    return raw, inner_parent, {"blend_name": blend_name, "inner_parent_r2": float(inner_r2), "gap_training_rows": gap_rows}


def strict_specialized_test(
    y: np.ndarray,
    groups: np.ndarray,
    scaffolds: np.ndarray,
    canonical: np.ndarray,
    test_canonical: np.ndarray,
    target_dense: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    global_indices: np.ndarray,
    test_global_indices: np.ndarray,
    y_global: np.ndarray,
    config: dict[str, object],
    gap_table: pd.DataFrame,
    maps: dict[str, dict[str, float]],
) -> np.ndarray:
    inner_folds = ei_route.folds_for(groups, 4)
    inner_arms = np.full((len(y), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        validation = np.flatnonzero(inner_folds == fold)
        training = np.flatnonzero(inner_folds != fold)
        inner_arms[validation] = reference.predict_base_models(
            target_dense, sparse_parts, fingerprints, y_global,
            global_indices[training], global_indices[validation], config, TARGET,
        )
    weights, intercept, _, _ = reference.blend_from_oof(y, inner_arms)
    test_arms = reference.predict_base_models(
        target_dense, sparse_parts, fingerprints, y_global,
        global_indices, test_global_indices, config, TARGET,
    )
    parent = reference.clip_prediction(y, test_arms @ weights + intercept)
    gap_model, _ = ei_route.fit_gap_model(gap_table, set(), set())
    raw, _, _, _ = ei_route.route_predictions(TARGET, test_canonical, parent, y, gap_model, maps)
    test_scaffolds = np.asarray([plumbing.scaffold(value) for value in test_canonical], dtype=object)
    test_nearest = fixed_features_module.nearest_similarity(fingerprints, test_global_indices, global_indices)
    allowed = (test_nearest < 0.70) & ~np.isin(test_scaffolds, ["c1ccsc1"])
    raw[~allowed] = parent[~allowed]
    return raw


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
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": SEED, "folds": 5, "mixed_candidate": True, "special_targets": ["ei", "eea"]})
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    target_keys = sorted(set(frame["canonical"]) | set(test.loc[test["target_type"] == TARGET, "canonical"]))
    target_indices = [key_to_index[value] for value in target_keys]
    base_features, base_names = fixed_features_module.fixed_features(molecules, target_indices)
    extra_features, extra_names = physics_features(molecules, target_indices)
    electronic, electronic_names = electronic_features(molecules, target_indices)
    electronic_finite_rows = int(np.sum(np.isfinite(electronic).any(axis=1)))
    features = np.hstack([base_features, extra_features, electronic])
    feature_names = base_names + extra_names + electronic_names
    feature_row = {value: row for row, value in enumerate(target_keys)}
    rows = np.asarray([feature_row[value] for value in frame["canonical"]], dtype=np.int64)
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True)
    test_rows = np.asarray([feature_row[value] for value in test_frame["canonical"]], dtype=np.int64)
    groups = np.asarray([plumbing.no_stereo(value) for value in frame["canonical"]], dtype=object)
    scaffolds = np.asarray([plumbing.scaffold(value) for value in frame["canonical"]], dtype=object)
    folds = plumbing.folds_for(groups, 5)
    target_dense = reference.target_dense_features(dense_base, cross_values, cross_available, TARGET)
    target_global_indices = np.asarray([key_to_index[value] for value in frame["canonical"]], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[target_global_indices] = y
    gap_table = ei_route.make_gap_table(pooled)
    maps = {
        name: dict(zip(part["canonical"], part["target"].astype(float), strict=True))
        for name in ("ei", "eea", "egc")
        for part in [pooled[pooled["target_type"] == name].reset_index(drop=True)]
    }

    def generate_parent() -> tuple[np.ndarray, np.ndarray]:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            training = np.flatnonzero(folds != fold)
            outer_parent, _, _ = strict_specialized_split(
                y, groups, scaffolds, frame["canonical"].astype(str).to_numpy(), training, validation,
                target_dense, sparse_parts, fingerprints, target_global_indices, y_global,
                config, gap_table, maps,
            )
            oof[validation] = outer_parent
        test_parent = strict_specialized_test(
            y, groups, scaffolds, frame["canonical"].astype(str).to_numpy(), test_frame["canonical"].astype(str).to_numpy(),
            target_dense, sparse_parts, fingerprints, target_global_indices, test_rows, y_global,
            config, gap_table, maps,
        )
        return oof, test_parent

    parent_oof, parent_test = generate_parent()
    replay_oof, replay_test = generate_parent()
    parent_oof_replay_max_abs = float(np.max(np.abs(parent_oof - replay_oof)))
    parent_test_replay_max_abs = float(np.max(np.abs(parent_test - replay_test)))
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    similarity = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        strict_parent, inner_parent, parent_meta = strict_specialized_split(
            y, groups, scaffolds, frame["canonical"].astype(str).to_numpy(), training, validation,
            target_dense, sparse_parts, fingerprints, target_global_indices, y_global, config, gap_table, maps,
        )
        if float(np.max(np.abs(strict_parent - parent_oof[validation]))) > 1.0e-12:
            raise RuntimeError("strict nested Ei parent replay against OOF construction failed")
        fitted = model()
        fitted.fit(features[rows[training]], y[training] - inner_parent)
        candidate[validation] = strict_parent + RESIDUAL_WEIGHT * fitted.predict(features[rows[validation]])
        global_validation = np.asarray([key_to_index[value] for value in frame.iloc[validation]["canonical"]], dtype=np.int64)
        global_training = np.asarray([key_to_index[value] for value in frame.iloc[training]["canonical"]], dtype=np.int64)
        similarity[validation] = fixed_features_module.nearest_similarity(fingerprints, global_validation, global_training)
        parent_score = float(r2_score(y[validation], strict_parent))
        candidate_score = float(r2_score(y[validation], candidate[validation]))
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": parent_score, "candidate_r2": candidate_score, "delta_r2": candidate_score - parent_score})
    if not np.isfinite(candidate).all():
        raise RuntimeError("non-finite Ei OOF candidate")
    parent_r2 = float(r2_score(y, parent_oof))
    candidate_r2 = float(r2_score(y, candidate))
    delta = candidate_r2 - parent_r2
    lower = bootstrap_lower(y, parent_oof, candidate, groups)
    panels, minimum_panel = fixed_features_module.panel_report(y, parent_oof, candidate, scaffolds, similarity)
    fitted = model()
    fitted.fit(features[rows], y - parent_oof)
    test_parent = parent_test
    test_replay_parent = replay_test
    test_candidate = test_parent + RESIDUAL_WEIGHT * fitted.predict(features[test_rows])
    component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": test_parent, "candidate_prediction": test_candidate})
    if len(component) != 148 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all():
        raise RuntimeError("Ei component output contract failed")
    component.to_csv(run_dir / "ei_component_predictions.csv", index=False)
    pd.DataFrame({"canonical": frame["canonical"].astype(str), "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": minimum_panel >= 0.0, "component_rows_pass": len(component) == 148, "parent_replay_oof_pass": parent_oof_replay_max_abs <= 1.0e-12, "parent_replay_test_pass": parent_test_replay_max_abs <= 1.0e-12, "electronic_feature_support_pass": electronic_finite_rows == len(y)}
    passed = bool(all(gates.values()))
    source_names = ("round2_c074_ei_nested_charge_residual.py", "round2_c063_egb_endpoint_conjugation_residual.py", "round2_ei_scaffold_abstaining_gap_identity_v4_portable.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py")
    report = {
        "schema_version": "ppp.round2.c074.ei-nested-charge-residual.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; strict v7-compatible Ei parent regenerated twice with outer-training-only parent/gap labels",
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "target": TARGET,
        "feature_names": feature_names,
        "electronic_feature_finite_rows": electronic_finite_rows,
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
    write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "feature_family": "official-SMILES endpoint/conjugation/dielectric density plus dummy-capped Gasteiger charge and charge-distance descriptors", "model": "HuberRegressor", "huber_epsilon": HUBER_EPSILON, "huber_alpha": HUBER_ALPHA, "huber_max_iter": HUBER_MAX_ITER, "residual_weight": RESIDUAL_WEIGHT, "outer": "canonical_no_stereo GroupKFold(5) with outer-training-only inner parent/gap labels", "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False, "parent_replay_oof_max_abs": parent_oof_replay_max_abs, "parent_replay_test_max_abs": parent_test_replay_max_abs, "electronic_feature_finite_rows": electronic_finite_rows})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Ei parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
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
