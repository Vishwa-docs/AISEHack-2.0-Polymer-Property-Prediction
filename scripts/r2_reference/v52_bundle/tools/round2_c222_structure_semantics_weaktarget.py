#!/usr/bin/env python3
"""C222: official-only structure-semantics residual for weak targets.

This child tests one bounded factor after the C218/C220 weak-target failures:
whether deterministic RDKit interpretation deltas for the same official SMILES
string carry residual signal.  The feature family is deliberately fixed:
raw/capped/neutralized/kekulized parse summaries plus their descriptor deltas.

No local_eval external_labels, public feedback, PI1M rows, pretrained weights, stored
prediction replay, model grid, target grid, or fallback slice search is used.
Targets already selected by the compound audit must also beat the current
selected component by at least +0.010 R2 before C222 can bank that target.
"""

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
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c200_clean_component_compound_audit_v3 as c200


RDLogger.DisableLog("rdApp.*")

TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("ei", "nc", "eps")
SCHEMA = "ppp.round2.c222.structure-semantics-weaktarget.v1"
SEED = 20260824
RIDGE_ALPHA = 60.0
RESIDUAL_WEIGHT = 0.50
MIN_C050_DELTA = 0.010
MIN_SELECTED_REFERENCE_DELTA = 0.010
CURRENT_AUDITS = (
    "R2-C221-20260805-0510-clean-component-compound-audit-v12",
    "R2-C219-20260805-0500-clean-component-compound-audit-v11",
    "R2-C217-20260805-0450-clean-component-compound-audit-v10",
    "R2-C215-20260805-0440-clean-component-compound-audit-v9",
)

DESC_NAMES = (
    "heavy_atoms",
    "atom_count",
    "bond_count",
    "ring_count",
    "aromatic_atom_count",
    "aromatic_bond_count",
    "hetero_atom_count",
    "formal_charge_sum",
    "formal_charge_abs_sum",
    "dummy_atom_count",
    "fraction_csp3",
    "rotatable_bonds",
    "h_acceptors",
    "h_donors",
    "tpsa",
    "mol_logp",
    "mol_mr",
    "bertz_ct",
    "exact_mol_wt",
    "valence_electrons",
    "spiro_atoms",
    "bridgehead_atoms",
    "aliphatic_rings",
    "aromatic_rings",
    "saturated_rings",
    "amide_bonds",
    "atom_C",
    "atom_N",
    "atom_O",
    "atom_S",
    "atom_F",
    "atom_Cl",
    "atom_Br",
    "atom_I",
    "atom_P",
    "atom_Si",
    "atom_B",
)

STRING_NAMES = (
    "length",
    "star_count",
    "bracket_count",
    "plus_count",
    "minus_count",
    "dot_count",
    "branch_open_count",
    "branch_close_count",
    "double_bond_count",
    "triple_bond_count",
    "aromatic_lowercase_count",
    "ring_digit_count",
    "slash_count",
    "at_count",
    "char_C",
    "char_N",
    "char_O",
    "char_S",
    "char_F",
    "char_P",
    "char_B",
    "char_I",
)

VARIANTS = ("raw", "capped", "neutralized", "kekule")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, stage: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"stage": stage, "time": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), **payload},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def unchanged_report(info: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    return {
        "active": False,
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, parent)),
        "delta_r2": 0.0,
        "positive_folds": 0,
        "group_bootstrap_lower": 0.0,
        "minimum_panel_delta": 0.0,
        "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}},
        "folds": [],
        "pass": False,
        "unchanged_parent": True,
    }


def capped_molecule(molecule: Chem.Mol | None) -> Chem.Mol | None:
    if molecule is None:
        return None
    try:
        capped = Chem.RWMol(Chem.Mol(molecule))
        for atom in capped.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(6)
                atom.SetFormalCharge(0)
                atom.SetNoImplicit(False)
        result = capped.GetMol()
        Chem.SanitizeMol(result)
        return result
    except Exception:
        return None


def neutralized_molecule(molecule: Chem.Mol | None) -> Chem.Mol | None:
    if molecule is None:
        return None
    try:
        neutral = Chem.RWMol(Chem.Mol(molecule))
        for atom in neutral.GetAtoms():
            if atom.GetFormalCharge() != 0:
                atom.SetFormalCharge(0)
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
        result = neutral.GetMol()
        Chem.SanitizeMol(result)
        return result
    except Exception:
        return None


def kekulized_molecule(molecule: Chem.Mol | None) -> Chem.Mol | None:
    if molecule is None:
        return None
    try:
        result = Chem.Mol(molecule)
        Chem.Kekulize(result, clearAromaticFlags=True)
        Chem.SanitizeMol(result)
        return result
    except Exception:
        return None


def molecule_descriptors(molecule: Chem.Mol | None) -> np.ndarray:
    if molecule is None:
        return np.full(len(DESC_NAMES), np.nan, dtype=np.float64)
    try:
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        atomic_nums = [atom.GetAtomicNum() for atom in atoms]
        values = [
            float(molecule.GetNumHeavyAtoms()),
            float(molecule.GetNumAtoms()),
            float(molecule.GetNumBonds()),
            float(molecule.GetRingInfo().NumRings()),
            float(sum(atom.GetIsAromatic() for atom in atoms)),
            float(sum(bond.GetIsAromatic() for bond in bonds)),
            float(sum(num not in (0, 1, 6) for num in atomic_nums)),
            float(sum(atom.GetFormalCharge() for atom in atoms)),
            float(sum(abs(atom.GetFormalCharge()) for atom in atoms)),
            float(sum(num == 0 for num in atomic_nums)),
            float(rdMolDescriptors.CalcFractionCSP3(molecule)),
            float(Descriptors.NumRotatableBonds(molecule)),
            float(Descriptors.NumHAcceptors(molecule)),
            float(Descriptors.NumHDonors(molecule)),
            float(rdMolDescriptors.CalcTPSA(molecule)),
            float(Crippen.MolLogP(molecule)),
            float(Crippen.MolMR(molecule)),
            float(Descriptors.BertzCT(molecule)),
            float(Descriptors.ExactMolWt(molecule)),
            float(Descriptors.NumValenceElectrons(molecule)),
            float(rdMolDescriptors.CalcNumSpiroAtoms(molecule)),
            float(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)),
            float(rdMolDescriptors.CalcNumAliphaticRings(molecule)),
            float(rdMolDescriptors.CalcNumAromaticRings(molecule)),
            float(rdMolDescriptors.CalcNumSaturatedRings(molecule)),
            float(rdMolDescriptors.CalcNumAmideBonds(molecule)),
            float(sum(num == 6 for num in atomic_nums)),
            float(sum(num == 7 for num in atomic_nums)),
            float(sum(num == 8 for num in atomic_nums)),
            float(sum(num == 16 for num in atomic_nums)),
            float(sum(num == 9 for num in atomic_nums)),
            float(sum(num == 17 for num in atomic_nums)),
            float(sum(num == 35 for num in atomic_nums)),
            float(sum(num == 53 for num in atomic_nums)),
            float(sum(num == 15 for num in atomic_nums)),
            float(sum(num == 14 for num in atomic_nums)),
            float(sum(num == 5 for num in atomic_nums)),
        ]
        result = np.asarray(values, dtype=np.float64)
        result[~np.isfinite(result) | (np.abs(result) > 1.0e12)] = np.nan
        return result
    except Exception:
        return np.full(len(DESC_NAMES), np.nan, dtype=np.float64)


def string_features(text: str | None) -> np.ndarray:
    value = "" if text is None else str(text)
    aromatic = sum(1 for char in value if char in "bcnops")
    digits = sum(1 for char in value if char.isdigit())
    items = [
        len(value),
        value.count("*"),
        value.count("[") + value.count("]"),
        value.count("+"),
        value.count("-"),
        value.count("."),
        value.count("("),
        value.count(")"),
        value.count("="),
        value.count("#"),
        aromatic,
        digits,
        value.count("/") + value.count("\\"),
        value.count("@"),
        value.count("C"),
        value.count("N"),
        value.count("O"),
        value.count("S"),
        value.count("F"),
        value.count("P"),
        value.count("B"),
        value.count("I"),
    ]
    return np.asarray(items, dtype=np.float64)


def canonical_smiles(molecule: Chem.Mol | None, kekule: bool = False) -> str | None:
    if molecule is None:
        return None
    try:
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False, kekuleSmiles=kekule)
    except Exception:
        return None


def feature_names() -> list[str]:
    names: list[str] = []
    for variant in VARIANTS:
        names.extend([f"{variant}_{name}" for name in DESC_NAMES])
    for left, right in (("capped", "raw"), ("neutralized", "capped"), ("kekule", "capped")):
        names.extend([f"delta_{left}_minus_{right}_{name}" for name in DESC_NAMES])
    for variant in ("input", "raw_canonical", "capped_canonical", "neutralized_canonical", "kekule_canonical"):
        names.extend([f"{variant}_string_{name}" for name in STRING_NAMES])
    names.extend(
        [
            "raw_parse_ok",
            "capped_parse_ok",
            "neutralized_parse_ok",
            "kekule_parse_ok",
            "capped_changes_raw_canonical",
            "neutralized_changes_capped_canonical",
            "kekule_changes_capped_canonical",
        ]
    )
    return names


def structure_semantics_features_for_smiles(smiles: str) -> tuple[np.ndarray, list[str], dict[str, bool]]:
    raw = Chem.MolFromSmiles(str(smiles))
    capped = capped_molecule(raw)
    neutral = neutralized_molecule(capped)
    kekule = kekulized_molecule(capped)
    molecules = {"raw": raw, "capped": capped, "neutralized": neutral, "kekule": kekule}
    desc = {name: molecule_descriptors(molecule) for name, molecule in molecules.items()}
    raw_smiles = canonical_smiles(raw)
    capped_smiles = canonical_smiles(capped)
    neutral_smiles = canonical_smiles(neutral)
    kekule_smiles = canonical_smiles(kekule, kekule=True)
    values: list[np.ndarray] = [desc[variant] for variant in VARIANTS]
    values.extend(
        [
            desc["capped"] - desc["raw"],
            desc["neutralized"] - desc["capped"],
            desc["kekule"] - desc["capped"],
        ]
    )
    values.extend(
        [
            string_features(str(smiles)),
            string_features(raw_smiles),
            string_features(capped_smiles),
            string_features(neutral_smiles),
            string_features(kekule_smiles),
        ]
    )
    flags = {
        "raw_parse_ok": raw is not None,
        "capped_parse_ok": capped is not None,
        "neutralized_parse_ok": neutral is not None,
        "kekule_parse_ok": kekule is not None,
        "capped_changes_raw_canonical": bool(raw_smiles is not None and capped_smiles is not None and raw_smiles != capped_smiles),
        "neutralized_changes_capped_canonical": bool(
            neutral_smiles is not None and capped_smiles is not None and neutral_smiles != capped_smiles
        ),
        "kekule_changes_capped_canonical": bool(kekule_smiles is not None and capped_smiles is not None and kekule_smiles != capped_smiles),
    }
    values.append(np.asarray([float(flags[name]) for name in flags], dtype=np.float64))
    row = np.concatenate(values).astype(np.float64)
    row[~np.isfinite(row) | (np.abs(row) > 1.0e12)] = np.nan
    return row, feature_names(), flags


def build_structure_semantics_features(smiles_values: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    names: list[str] | None = None
    flag_counts: dict[str, int] = {}
    for value in smiles_values:
        row, row_names, flags = structure_semantics_features_for_smiles(value)
        if names is None:
            names = row_names
        elif names != row_names:
            raise RuntimeError("C222 structure-semantics feature schema drift")
        rows.append(row)
        for key, flag in flags.items():
            flag_counts[key] = flag_counts.get(key, 0) + int(flag)
    if names is None:
        names = feature_names()
    matrix = np.vstack(rows).astype(np.float64)
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)] = np.nan
    return matrix, {
        "feature_family": "fixed RDKit structure-semantics variant descriptors",
        "feature_count": int(matrix.shape[1]),
        "rows": int(matrix.shape[0]),
        "flag_counts": flag_counts,
        "variants": list(VARIANTS),
        "descriptor_count_per_variant": len(DESC_NAMES),
        "string_feature_count_per_variant": len(STRING_NAMES),
        "feature_names": names,
    }


def residual_model() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4),
    )


def fit_target_residual(
    info: dict[str, Any],
    features: np.ndarray,
    test_indices: np.ndarray,
    test_parent: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        model = residual_model()
        model.fit(features[indices[training]], y[training] - parent[training])
        residual = np.asarray(model.predict(features[indices[validation]]), dtype=float)
        lower_clip = float(np.quantile(y[training], 0.005) - 0.10)
        upper_clip = float(np.quantile(y[training], 0.995) + 0.10)
        candidate[validation] = np.clip(parent[validation] + RESIDUAL_WEIGHT * residual, lower_clip, upper_clip)
        fold_parent = float(r2_score(y[validation], parent[validation]))
        fold_candidate = float(r2_score(y[validation], candidate[validation]))
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(len(validation)),
                "parent_r2": fold_parent,
                "candidate_r2": fold_candidate,
                "delta_r2": fold_candidate - fold_parent,
            }
        )
    if not np.isfinite(candidate).all():
        raise RuntimeError("C222 produced non-finite OOF predictions")
    full_model = residual_model()
    full_model.fit(features[indices], y - parent)
    test_residual = np.asarray(full_model.predict(features[test_indices]), dtype=float)
    lower_clip = float(np.quantile(y, 0.005) - 0.10)
    upper_clip = float(np.quantile(y, 0.995) + 0.10)
    test_candidate = np.clip(np.asarray(test_parent, dtype=float) + RESIDUAL_WEIGHT * test_residual, lower_clip, upper_clip)
    return {"candidate": candidate, "test_candidate": test_candidate, "folds": fold_rows}


def selected_reference(root: Path, target: str) -> dict[str, Any]:
    for run_id in CURRENT_AUDITS:
        metrics = c200.load_json(c200.run_dir(root, run_id) / "metrics.json")
        if not isinstance(metrics, dict):
            continue
        selected = metrics.get("selected_components", {}).get(target, {})
        if isinstance(selected, dict) and selected.get("candidate_r2") is not None:
            return {
                "reference_source": f"{run_id}_selected_component",
                "run_id": selected.get("run_id"),
                "source": selected.get("source"),
                "candidate_r2": float(selected.get("candidate_r2")),
            }
    return {
        "reference_source": "C050_parent_fallback",
        "run_id": None,
        "source": "C050_parent",
        "candidate_r2": None,
    }


def target_test_indices(parent: dict[str, Any], target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_rows = (
        parent["test"]
        .loc[parent["test"]["target_type"].astype(str).eq(target)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    test_detail = (
        parent["test_parent_detail"]
        .loc[parent["test_parent_detail"]["target_type"].astype(str).eq(target)]
        .sort_values("id")
        .reset_index(drop=True)
    )
    if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
        raise RuntimeError(f"C222 {target} test ID alignment failed")
    indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
    return test_rows, indices, test_detail["target"].to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--canonical-run",
        default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7",
    )
    args = parser.parse_args()
    started = time.time()
    root = Path(args.root).resolve()
    round2_root = root / "Polymer Prediction Challenge Round 2"
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    progress = run_dir / "progress.jsonl"
    checkpoint(progress, "started", experiment_id=run_dir.name)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    canonical_run = Path(args.canonical_run)
    if not canonical_run.is_absolute():
        canonical_run = (root / canonical_run).resolve()

    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, canonical_run)
    checkpoint(progress, "parent_parity", **parity)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    features, feature_report = build_structure_semantics_features(list(parent["keys"]))
    checkpoint(progress, "features_complete", feature_count=feature_report["feature_count"], rows=feature_report["rows"])

    target_reports: dict[str, Any] = {}
    result_by_target: dict[str, dict[str, Any]] = {}
    test_rows_by_target: dict[str, pd.DataFrame] = {}
    banked: list[str] = []
    for target in ACTIVE_TARGETS:
        info = dict(parent["target_info"][target])
        info["fingerprints"] = parent["fingerprints"]
        test_rows, test_indices, test_parent = target_test_indices(parent, target)
        result = fit_target_residual(info, features, test_indices, test_parent)
        report = carrier.evaluate_target(info, {"candidate": result["candidate"]})
        report["folds"] = result["folds"]
        ref = selected_reference(root, target)
        selected_r2 = float(ref["candidate_r2"]) if ref.get("candidate_r2") is not None else float(report["parent_r2"])
        delta_vs_selected = float(report["candidate_r2"] - selected_r2)
        replacement_gate = bool(report["pass"] and delta_vs_selected >= MIN_SELECTED_REFERENCE_DELTA)
        report.update(
            {
                "active": True,
                "changed_factor": "fixed official-SMILES structure-semantics variant residual",
                "model_family": "Ridge residual over raw/capped/neutralized/kekulized RDKit interpretation-delta descriptors",
                "ridge_alpha": RIDGE_ALPHA,
                "residual_weight": RESIDUAL_WEIGHT,
                "minimum_c050_delta": MIN_C050_DELTA,
                "selected_current_reference": ref,
                "delta_vs_selected_current_reference": delta_vs_selected,
                "minimum_selected_reference_delta": MIN_SELECTED_REFERENCE_DELTA,
                "beats_selected_reference_gate": bool(delta_vs_selected >= MIN_SELECTED_REFERENCE_DELTA),
                "replacement_gate_pass": replacement_gate,
                "no_hyperparameter_grid": True,
                "no_target_grid": True,
                "no_cross_target_labels": True,
            }
        )
        target_reports[target] = report
        result_by_target[target] = result
        test_rows_by_target[target] = test_rows
        if replacement_gate:
            banked.append(target)

    oof_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        if target in ACTIVE_TARGETS:
            report = target_reports[target]
            result = result_by_target[target]
            candidate = result["candidate"]
        else:
            report = unchanged_report(info)
            candidate = np.asarray(info["parent"], dtype=float)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(info["groups"], dtype=object))
        assembled = candidate if target in banked else np.asarray(info["parent"], dtype=float)
        frame = pd.DataFrame(
            {
                "canonical": info["canonical"],
                "target_type": target,
                "target": info["y"],
                "parent": info["parent"],
                "candidate": candidate,
                "group": info["groups"],
                "scaffold": info["scaffolds"],
                "fold": folds,
                "assembled": assembled,
            }
        )
        oof_parts.append(frame)
        if target in ACTIVE_TARGETS:
            frame[["canonical", "target", "parent", "candidate"]].to_csv(run_dir / f"{target}_oof_predictions.csv", index=False)
            test_rows = test_rows_by_target[target]
            component_parts.append(
                pd.DataFrame(
                    {
                        "id": test_rows["id"].astype(int),
                        "target_type": target,
                        "parent": target_test_indices(parent, target)[2],
                        "candidate": result_by_target[target]["test_candidate"],
                    }
                )
            )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    predictions = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    for target in banked:
        test_rows = test_rows_by_target[target]
        values = pd.Series(result_by_target[target]["test_candidate"], index=test_rows["id"].astype(int))
        mask = predictions["target_type"].astype(str).eq(target)
        predictions.loc[mask, "target"] = predictions.loc[mask, "id"].astype(int).map(values).to_numpy(float)
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C222 ID coverage/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C222 produced non-finite predictions")

    report = {
        "schema_version": SCHEMA,
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "pi1m_used": False,
        "stored_prediction_replay": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "active_targets": list(ACTIVE_TARGETS),
        "feature_report": {key: value for key, value in feature_report.items() if key != "feature_names"},
        "feature_names_sha256": hashlib.sha256("\n".join(feature_report["feature_names"]).encode("utf-8")).hexdigest(),
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": bool(assembled_mean - parent_mean >= 0.002 and bool(banked)),
        "goal_0_95_met": bool(assembled_mean >= 0.95 and bool(banked)),
        "decision": "candidate_pass_pending_clean_reproduction" if banked else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "compound_audit_helpers": sha256_file(round2_root / "tools/round2_c200_clean_component_compound_audit_v3.py"),
        },
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.concat(component_parts, ignore_index=True).to_csv(run_dir / "component_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "seed": SEED,
            "active_targets": list(ACTIVE_TARGETS),
            "ridge_alpha": RIDGE_ALPHA,
            "residual_weight": RESIDUAL_WEIGHT,
            "minimum_c050_delta": MIN_C050_DELTA,
            "minimum_selected_reference_delta": MIN_SELECTED_REFERENCE_DELTA,
            "feature_family": "raw/capped/neutralized/kekulized RDKit structure-semantics descriptor deltas",
            "selection_rule": "one fixed weak-target structure-semantics residual; bank only targets clearing normal and selected-reference gates",
            "official_only": True,
            "local_eval_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "kaggle_submission": False,
        },
    )
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"sklearn={__import__('sklearn').__version__}",
                f"rdkit={Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: `{report['decision']}`. Banked targets: `{','.join(banked) or 'none'}`. "
        f"Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; "
        f"gain `{assembled_mean - parent_mean:+.12f}`. "
        "Official-only; no local_eval read; no Kaggle action; no final-notebook action.\n",
        encoding="utf-8",
    )
    checkpoint(
        progress,
        "metrics_written",
        decision=report["decision"],
        banked_targets=banked,
        mean_candidate_r2=assembled_mean,
    )
    manifest = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "banked_targets": banked,
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": assembled_mean,
                "mean_gain": assembled_mean - parent_mean,
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
