#!/usr/bin/env python3
"""C124: official-only target-specific electronic heads for Ei and Eea.

The parent is rebuilt once and checked against the frozen C050 artifacts.  The
new heads only use structure-derived features computed in this process; the
other five properties remain exact C050 fallback heads.  This is a clean OOF
screen.  Full-test fitting is deliberately opened only if every preregistered
clean gate passes.
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
from rdkit.Chem import EState, MACCSkeys, Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c075_ei_cross_target_masked_charge_ridge as electronic_base
import round2_c112_c050_parent_parity_control as parent_control
import round2_c121_nearmiss_bridge_one_replay as replay
import round2_eea_cross_target_oof_residual_stack as plumbing
import round2_c120_nearmiss_bridge_mixed as diagnostics


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE = ("ei", "eea")
SEED = 2026
OUTER_FOLDS = 5
RESIDUAL_WEIGHT = 0.20
BLEND_WEIGHTS = np.asarray([0.5, 0.5], dtype=float)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, name: str, **fields: Any) -> None:
    record = {"checkpoint": name, "at": datetime.now().astimezone().isoformat(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 1.0e-15 else float("nan")


def capped_molecule(molecule: Chem.Mol) -> Chem.Mol:
    capped = Chem.RWMol(Chem.Mol(molecule))
    for atom in capped.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
    result = capped.GetMol()
    Chem.SanitizeMol(result)
    return result


def structural_bank(molecules: list[Chem.Mol], indices: list[int]) -> tuple[np.ndarray, list[str], int]:
    """Return continuous electronic/topology features followed by MACCS bits."""
    physics, physics_names = electronic_base.physics_features(molecules, indices)
    charges, charge_names = electronic_base.electronic_features(molecules, indices)
    continuous_names = list(physics_names) + list(charge_names)
    continuous = np.hstack([physics, charges]).astype(np.float64, copy=False)
    extra_names = [
        "estate_min", "estate_max", "estate_mean", "estate_std", "estate_sum_density",
        "aromatic_atom_density", "sp2_carbon_density", "sp3_carbon_density",
        "conjugated_bond_density", "heteroatom_density_again", "ring_density_again",
        "rotatable_density_again", "mw_per_heavy", "heavy_log1p",
    ]
    extra = np.full((len(indices), len(extra_names)), np.nan, dtype=np.float64)
    maccs = np.zeros((len(indices), 167), dtype=np.float64)
    for row, source_index in enumerate(indices):
        try:
            molecule = capped_molecule(molecules[source_index])
            atoms = list(molecule.GetAtoms())
            bonds = list(molecule.GetBonds())
            heavy = max(molecule.GetNumHeavyAtoms(), 1)
            bond_count = max(len(bonds), 1)
            estate = np.asarray(EState.EStateIndices(molecule), dtype=np.float64)
            estate = estate[np.isfinite(estate)]
            if len(estate) == 0:
                raise ValueError("empty EState vector")
            aromatic = sum(atom.GetIsAromatic() for atom in atoms)
            sp2 = sum(atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP2 for atom in atoms)
            sp3 = sum(atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP3 for atom in atoms)
            conjugated = sum(bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0 for bond in bonds)
            hetero = sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)
            rings = molecule.GetRingInfo().NumRings()
            rotatable = Descriptors.NumRotatableBonds(molecule)
            extra[row] = [
                float(np.min(estate)), float(np.max(estate)), float(np.mean(estate)), float(np.std(estate)),
                float(np.sum(estate) / heavy), aromatic / heavy, sp2 / heavy, sp3 / heavy,
                conjugated / bond_count, hetero / heavy, rings / heavy, rotatable / heavy,
                float(Descriptors.MolWt(molecule) / heavy), float(np.log1p(heavy)),
            ]
            bit_vector = MACCSkeys.GenMACCSKeys(molecule)
            DataStructs.ConvertToNumpyArray(bit_vector, maccs[row])
        except Exception:
            continue
    names = continuous_names + extra_names + [f"maccs_{index:03d}" for index in range(maccs.shape[1])]
    matrix = np.hstack([continuous, extra, maccs])
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names, continuous.shape[1] + extra.shape[1]


def ridge_arm(x: np.ndarray, y: np.ndarray, predict: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=30.0),
    )
    model.fit(x, y)
    return np.asarray(model.predict(predict), dtype=np.float64)


def extra_arm(x: np.ndarray, y: np.ndarray, predict: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        ExtraTreesRegressor(
            n_estimators=256,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.7,
            random_state=SEED,
            n_jobs=1,
        ),
    )
    model.fit(x, y)
    return np.asarray(model.predict(predict), dtype=np.float64)


def arm_predictions(x: np.ndarray, continuous_count: int, train_global: np.ndarray, valid_global: np.ndarray, residual_train: np.ndarray) -> np.ndarray:
    ridge = ridge_arm(x[train_global, :continuous_count], residual_train, x[valid_global, :continuous_count])
    extra = extra_arm(x[train_global], residual_train, x[valid_global])
    return np.column_stack([ridge, extra])


def grouped_bootstrap(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return diagnostics.grouped_bootstrap(y, parent, candidate, groups)


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, Any], float | None]:
    panels: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, mask: np.ndarray, minimum: int = 5) -> None:
        rows = int(np.sum(mask))
        item: dict[str, Any] = {"rows": rows, "delta_r2": None, "status": "inapplicable"}
        if rows >= minimum and np.var(y[mask]) > 1.0e-15:
            delta = score(y[mask], candidate[mask]) - score(y[mask], parent[mask])
            item.update({"delta_r2": delta, "status": "evaluable"})
            deltas.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    for name, mask in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        add(name, mask)
    for scaffold in sorted(set(scaffolds)):
        add(f"scaffold_{scaffold}", scaffolds == scaffold, minimum=10)
    return panels, (float(min(deltas)) if deltas else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    data_dir = (root / args.data_dir).resolve()
    run_dir = (root / args.run_dir).resolve()
    if {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("C124 requires a fresh protocol-only run directory")
    started = time.time()
    progress_path = run_dir / "progress.jsonl"
    bundle, parity = replay.build_one_replay_bundle(root, data_dir, run_dir)
    checkpoint(progress_path, "parent_replay", oof_rows=9851, test_rows=4940)
    checkpoint(progress_path, "parent_parity", **parity)
    if parity["oof_max_abs"] > 1.0e-12 or parity["test_max_abs"] > 1.0e-12:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    global_indices = list(range(len(bundle["keys"])))
    matrix, names, continuous_count = structural_bank(bundle["molecules"], global_indices)
    write_json(run_dir / "feature_schema.json", {"feature_count": int(matrix.shape[1]), "continuous_count": int(continuous_count), "names": names})
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    test_predictions: dict[str, np.ndarray] = {}

    for target in TARGETS:
        info = bundle["target_info"][target]
        y = np.asarray(info["y"], dtype=np.float64)
        parent = np.asarray(info["parent"], dtype=np.float64)
        if target not in ACTIVE:
            target_reports[target] = {"parent_r2": score(y, parent), "candidate_r2": score(y, parent), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "pass": True, "unchanged_parent": True}
            oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": parent, "outer_fold": -1}))
            continue
        train_global = np.asarray([bundle["key_to_index"][canonical] for canonical in info["canonical"]], dtype=np.int64)
        groups = np.asarray([plumbing.no_stereo(value) for value in info["canonical"]], dtype=object)
        scaffolds = np.asarray([plumbing.scaffold(value) for value in info["canonical"]], dtype=object)
        folds = plumbing.folds_for(groups, OUTER_FOLDS)
        residual = y - parent
        candidate = np.full(len(y), np.nan, dtype=np.float64)
        similarity = np.full(len(y), np.nan, dtype=np.float64)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(OUTER_FOLDS):
            train = np.flatnonzero(folds != fold)
            valid = np.flatnonzero(folds == fold)
            arms = arm_predictions(matrix, continuous_count, train_global[train], train_global[valid], residual[train])
            correction = RESIDUAL_WEIGHT * (arms @ BLEND_WEIGHTS)
            candidate[valid] = parent[valid] + correction
            train_fps = [bundle["fingerprints"][int(index)] for index in train_global[train]]
            for row, global_index in enumerate(train_global[valid]):
                similarities = DataStructs.BulkTanimotoSimilarity(bundle["fingerprints"][int(global_index)], train_fps)
                similarity[valid[row]] = max(similarities) if similarities else 0.0
            fold_parent = score(y[valid], parent[valid])
            fold_candidate = score(y[valid], candidate[valid])
            fold_rows.append({"fold": fold, "rows": int(len(valid)), "parent_r2": fold_parent, "candidate_r2": fold_candidate, "delta_r2": fold_candidate - fold_parent, "weights": BLEND_WEIGHTS.tolist()})
            checkpoint(progress_path, f"{target}_fold_{fold}", delta_r2=fold_candidate - fold_parent)
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"non-finite {target} OOF candidate")
        bootstrap = grouped_bootstrap(y, parent, candidate, groups)
        panels, minimum_panel = panel_report(y, parent, candidate, groups, scaffolds, similarity)
        delta = score(y, candidate) - score(y, parent)
        positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        target_pass = bool(delta >= 0.01 and positive >= 4 and bootstrap["lower_2_5"] > 0.0 and minimum_panel is not None and minimum_panel >= 0.0)
        target_reports[target] = {"parent_r2": score(y, parent), "candidate_r2": score(y, candidate), "delta_r2": delta, "positive_folds": positive, "folds": fold_rows, "group_bootstrap_lower": bootstrap["lower_2_5"], "minimum_panel_delta": minimum_panel, "panels": panels, "feature_count": int(matrix.shape[1]), "pass": target_pass, "unchanged_parent": False}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "outer_fold": folds}))
        checkpoint(progress_path, f"{target}_oof", delta_r2=delta, positive_folds=positive, minimum_panel_delta=minimum_panel, component_pass=target_pass)

    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(target_reports[target]["delta_r2"] for target in TARGETS))
    clean_pass = bool(mean_candidate - mean_parent >= 0.002 and max_loss >= -0.003 and all(target_reports[target]["pass"] for target in ACTIVE))

    # Full-data test fitting is intentionally unreachable for a failed clean screen.
    # If this branch passes, the test predictions are written only as a research
    # candidate for the separate post-freeze local_eval audit.
    if clean_pass:
        for target in ACTIVE:
            info = bundle["target_info"][target]
            train_global = np.asarray([bundle["key_to_index"][canonical] for canonical in info["canonical"]], dtype=np.int64)
            test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            test_global = np.asarray([bundle["key_to_index"][canonical] for canonical in test_frame["canonical"]], dtype=np.int64)
            residual = np.asarray(info["y"], dtype=np.float64) - np.asarray(info["parent"], dtype=np.float64)
            arms = arm_predictions(matrix, continuous_count, train_global, test_global, residual)
            parent_test = bundle["test_detail"].loc[bundle["test_detail"]["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float)
            test_predictions[target] = parent_test + RESIDUAL_WEIGHT * (arms @ BLEND_WEIGHTS)
        parts = []
        for target in TARGETS:
            frame = bundle["test"].loc[bundle["test"]["target_type"] == target, ["id", "target_type"]].sort_values("id").copy()
            if target in test_predictions:
                frame["target"] = test_predictions[target]
            else:
                frame = frame.merge(bundle["test_detail"][["id", "target_type", "model_prediction"]], on=["id", "target_type"], how="left", validate="one_to_one").rename(columns={"model_prediction": "target"})
            parts.append(frame[["id", "target_type", "target"]])
        pd.concat(parts, ignore_index=True).sort_values("id").to_csv(run_dir / "full_test_candidate.csv", index=False)

    report = {"schema_version": "ppp.round2.c124.electronic-ei-eea-bank.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "official_only": True, "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7", "parent_parity": parity, "target_reports": target_reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": max_loss, "clean_gate_pass": clean_pass, "decision": "clean_gate_pass_pending_local_eval" if clean_pass else "rejected_component_or_full_gate", "elapsed_seconds": float(time.time() - started)}
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c124.electronic-ei-eea-bank.v1", "active_targets": ACTIVE, "residual_weight": RESIDUAL_WEIGHT, "blend_weights": BLEND_WEIGHTS.tolist(), "outer_folds": OUTER_FOLDS, "fold_assignment": "plumbing.folds_for(no_stereo,5)", "arms": ["ridge_alpha_30_continuous", "extratrees_256_depth8_leaf3_maxfeatures0.7_all"], "official_only": True, "local_eval_read": False, "pi1m_used": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent R2 `{mean_parent:.12f}`; candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. The Ei/Eea specialist is bankable only if both active component gates pass; no local_eval file was read.\n", encoding="utf-8")
    source_paths = [Path(__file__), root / "tools/round2_c075_ei_cross_target_masked_charge_ridge.py", root / "tools/round2_c121_nearmiss_bridge_one_replay.py", root / "tools/round2_c120_nearmiss_bridge_mixed.py", root / "tools/round2_c112_c050_parent_parity_control.py", root / "tools/round2_eea_cross_target_oof_residual_stack.py", root / "tools/initial_reference_pipeline.py"]
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for path in source_paths:
        manifest.append(f"{sha256_file(path)}  SOURCE {path.relative_to(root)}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    checkpoint(progress_path, "metrics_written", decision=report["decision"], mean_gain=report["mean_gain"])
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "ei_delta": target_reports["ei"]["delta_r2"], "eea_delta": target_reports["eea"]["delta_r2"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
