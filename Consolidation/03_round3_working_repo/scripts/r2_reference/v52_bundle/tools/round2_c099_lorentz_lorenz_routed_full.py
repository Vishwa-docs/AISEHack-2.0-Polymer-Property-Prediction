#!/usr/bin/env python3
"""C099: fixed Lorentz-Lorenz-inspired Nc route plus frozen C098 EPS route."""

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
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c076_eps_paired_charge_polarizability_residual as paired_features
import round2_c098_target_routed_qspr_full as c098
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
RESIDUAL_WEIGHT = 0.20
RIDGE_ALPHA = 30.0
SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def model() -> Any:
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


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


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-9))


def lorentz_lorenz_features(molecules: list[Chem.Mol], indices: list[int]) -> tuple[np.ndarray, list[str]]:
    names = [
        "mol_mr", "exact_mw", "labute_asa", "vdw_volume", "mr_per_vdw", "mw_per_vdw", "mr_per_mw", "tpsa_per_vdw",
        "heavy_per_vdw", "hetero_per_vdw", "aromatic_per_vdw", "conjugated_per_vdw", "ring_per_vdw", "rotatable_per_vdw",
        "hba_per_vdw", "hbd_per_vdw", "logp", "tpsa", "hba", "hbd", "aromatic_atoms", "hetero_atoms", "halogen_atoms",
        "sulfur_atoms", "ring_count", "rotatable_bonds", "conjugated_bonds", "dummy_atoms", "formal_charge", "endpoint_count",
    ]
    output = np.full((len(indices), len(names)), np.nan, dtype=np.float64)
    table = Chem.GetPeriodicTable()
    for row, source_index in enumerate(indices):
        try:
            molecule = molecules[source_index]
            atoms = list(molecule.GetAtoms())
            bonds = list(molecule.GetBonds())
            heavy = max(sum(atom.GetAtomicNum() > 1 for atom in atoms), 1)
            vdw = sum(float(table.GetRvdw(atom.GetAtomicNum())) for atom in atoms if atom.GetAtomicNum() > 0)
            vdw = max(vdw, 1.0e-9)
            mr = float(Crippen.MolMR(molecule))
            mw = float(rdMolDescriptors.CalcExactMolWt(molecule))
            asa = float(Descriptors.LabuteASA(molecule))
            tpsa = float(rdMolDescriptors.CalcTPSA(molecule))
            hba = float(rdMolDescriptors.CalcNumHBA(molecule))
            hbd = float(rdMolDescriptors.CalcNumHBD(molecule))
            aromatic = float(sum(atom.GetIsAromatic() for atom in atoms))
            hetero = float(sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms))
            halogen = float(sum(atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms))
            sulfur = float(sum(atom.GetAtomicNum() == 16 for atom in atoms))
            rings = float(molecule.GetRingInfo().NumRings())
            rotatable = float(Descriptors.NumRotatableBonds(molecule))
            conjugated = float(sum(bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0 for bond in bonds))
            dummy = float(sum(atom.GetAtomicNum() == 0 for atom in atoms))
            charge = float(sum(atom.GetFormalCharge() for atom in atoms))
            endpoints = dummy
            output[row] = [
                mr, mw, asa, vdw, safe_ratio(mr, vdw), safe_ratio(mw, vdw), safe_ratio(mr, mw), safe_ratio(tpsa, vdw),
                safe_ratio(heavy, vdw), safe_ratio(hetero, vdw), safe_ratio(aromatic, vdw), safe_ratio(conjugated, vdw),
                safe_ratio(rings, vdw), safe_ratio(rotatable, vdw), safe_ratio(hba, vdw), safe_ratio(hbd, vdw),
                float(Crippen.MolLogP(molecule)), tpsa, hba, hbd, aromatic, hetero, halogen, sulfur, rings, rotatable,
                conjugated, dummy, charge, endpoints,
            ]
        except Exception:
            continue
    output[~np.isfinite(output)] = np.nan
    return output, names


def nc_features(bundle: dict[str, Any], target: str = "nc") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    info = bundle["target_info"][target]
    keys = sorted(set(info["canonical"]) | set(bundle["test"].loc[bundle["test"]["target_type"] == target, "canonical"]))
    global_indices = np.asarray([bundle["key_to_index"][value] for value in keys], dtype=np.int64)
    fixed, fixed_names = fixed_features.fixed_features(bundle["molecules"], global_indices.tolist())
    physical, physical_names = paired_features.physics_features(bundle["molecules"], global_indices.tolist())
    charge, charge_names = paired_features.charge_features(bundle["molecules"], global_indices.tolist())
    ll, ll_names = lorentz_lorenz_features(bundle["molecules"], global_indices.tolist())
    matrix = np.hstack([fixed, physical, charge, ll]).astype(np.float64, copy=False)
    feature_row = {value: row for row, value in enumerate(keys)}
    train_rows = np.asarray([feature_row[value] for value in info["canonical"]], dtype=np.int64)
    test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    test_rows = np.asarray([feature_row[value] for value in test_frame["canonical"]], dtype=np.int64)
    return matrix, train_rows, test_rows, fixed_names + physical_names + charge_names + ll_names


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, Any], float]:
    panels: dict[str, Any] = {}
    values: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected))
        item: dict[str, Any] = {"rows": count, "delta_r2": 0.0, "status": "inapplicable"}
        if count >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"delta_r2": delta, "status": "evaluable"})
            values.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    for name, selected in {"similarity_lt_0.30": similarity < 0.30, "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50), "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70), "similarity_ge_0.70": similarity >= 0.70}.items():
        add(name, selected)
    for name in sorted(set(scaffolds)):
        add(f"scaffold_{name}", scaffolds == name, minimum=10)
    return panels, float(min(values)) if values else 0.0


def evaluate_target(bundle: dict[str, Any], target: str, matrix: np.ndarray, train_rows: np.ndarray, test_rows: np.ndarray, feature_names: list[str], pair_train: np.ndarray | None = None, pair_test: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    info = bundle["target_info"][target]
    y, parent, folds = info["y"], info["parent"], info["folds"]
    candidate = parent.copy()
    similarity = np.full(len(y), np.nan, dtype=np.float64)
    residual = y - parent
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        fitted = model()
        fitted.fit(matrix[train_rows[training]], residual[training])
        correction = RESIDUAL_WEIGHT * fitted.predict(matrix[train_rows[validation]])
        if pair_train is None:
            candidate[validation] = parent[validation] + correction
        else:
            supported = np.isfinite(pair_train[validation])
            candidate[validation[supported]] = parent[validation[supported]] + correction[supported]
        global_validation = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][validation]], dtype=np.int64)
        global_training = np.asarray([bundle["key_to_index"][value] for value in info["canonical"][training]], dtype=np.int64)
        similarity[validation] = fixed_features.nearest_similarity(bundle["fingerprints"], global_validation, global_training)
    fitted = model()
    fitted.fit(matrix[train_rows], residual)
    test_correction = RESIDUAL_WEIGHT * fitted.predict(matrix[test_rows])
    if pair_test is not None:
        test_correction = test_correction * np.isfinite(pair_test)
    fold_rows = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation]))})
    panels, minimum_panel = panel_report(y, parent, candidate, info["scaffolds"], similarity)
    delta = float(r2_score(y, candidate) - r2_score(y, parent))
    lower = bootstrap_lower(y, parent, candidate, info["groups"])
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    return {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "folds": fold_rows, "panels": panels, "feature_count": int(matrix.shape[1]), "pair_rows": int(np.sum(np.isfinite(pair_train))) if pair_train is not None else 0, "pass": bool(delta >= 0.005 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0)}, candidate, test_correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    bundle = c098.parent_bundle(root, data_dir)
    replay = c098.parent_bundle(root, data_dir)
    replay_max = 0.0
    for target in TARGETS:
        left, right = bundle["target_info"][target], replay["target_info"][target]
        replay_max = max(replay_max, float(np.max(np.abs(left["parent"] - right["parent"]))))
        ids = bundle["test_detail"][bundle["test_detail"]["target_type"] == target]["id"].to_numpy()
        left_test = bundle["test_detail"].set_index("id").loc[ids, "model_prediction"].to_numpy(float)
        right_test = replay["test_detail"].set_index("id").loc[ids, "model_prediction"].to_numpy(float)
        replay_max = max(replay_max, float(np.max(np.abs(left_test - right_test))))
    raw_test = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    components: list[pd.DataFrame] = []
    for target in TARGETS:
        info = bundle["target_info"][target]
        if target == "eps":
            matrix, train_rows, test_rows, names, pair_train, pair_test = c098.target_features(bundle, target)
            report, candidate, test_correction = evaluate_target(bundle, target, matrix, train_rows, test_rows, names, pair_train, pair_test)
            test_parent = raw_test.loc[raw_test["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float).copy()
            test_candidate = test_parent + test_correction
            test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            raw_test.loc[raw_test["target_type"] == target, "model_prediction"] = test_candidate
            components.append(pd.DataFrame({"id": test_frame["id"], "target_type": target, "parent_prediction": test_parent, "candidate_prediction": test_candidate, "pair_available": np.isfinite(pair_test)}))
        elif target == "nc":
            matrix, train_rows, test_rows, names = nc_features(bundle)
            report, candidate, test_correction = evaluate_target(bundle, target, matrix, train_rows, test_rows, names)
            test_parent = raw_test.loc[raw_test["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float).copy()
            test_candidate = test_parent + test_correction
            test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            raw_test.loc[raw_test["target_type"] == target, "model_prediction"] = test_candidate
            components.append(pd.DataFrame({"id": test_frame["id"], "target_type": target, "parent_prediction": test_parent, "candidate_prediction": test_candidate, "pair_available": True}))
        else:
            y, parent = info["y"], info["parent"]
            report = {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, parent)), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "feature_count": 0, "pair_rows": 0, "pass": True, "unchanged_parent": True}
            candidate = parent.copy()
        report["unchanged_parent"] = target not in {"nc", "eps"}
        reports[target] = report
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": info["y"], "parent": info["parent"], "candidate": candidate, "group": info["groups"], "scaffold": info["scaffolds"], "fold": info["folds"]}))
    final_detail, override_report = reference.apply_official_overrides(raw_test, bundle["test"], bundle["raw_labels"])
    submission = final_detail[["id", "target"]].copy()
    if len(submission) != 4940 or not submission["id"].equals(bundle["test"]["id"]) or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C099 complete output contract failed")
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(reports[target]["delta_r2"] for target in TARGETS))
    changed_pass = all(reports[target]["pass"] for target in ("nc", "eps"))
    full_pass = bool(replay_max <= 1.0e-12 and mean_candidate > 0.8748045537286532 and mean_candidate > mean_parent and changed_pass and max_loss >= -0.003)
    submission.to_csv(run_dir / "predictions.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.concat(components, ignore_index=True).to_csv(run_dir / "component_predictions.csv", index=False)
    source_paths = {"script": root / "tools" / "round2_c099_lorentz_lorenz_routed_full.py", "reference": root / "tools" / "initial_reference_pipeline.py", "c098": root / "tools" / "round2_c098_target_routed_qspr_full.py", "fixed_features": root / "tools" / "round2_c063_egb_endpoint_conjugation_residual.py", "paired_features": root / "tools" / "round2_c076_eps_paired_charge_polarizability_residual.py"}
    report = {"schema_version": "ppp.round2.c099.lorentz-lorenz-routed-full.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "C050 source rebuild; EPS frozen C098 route; Nc new Lorentz-Lorenz structure-only route", "official_inputs": bundle["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "target_reports": reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": max_loss, "complete_output_rows": int(len(submission)), "complete_output_order_pass": True, "parent_replay_oof_test_max_abs": replay_max, "parent_replay_pass": bool(replay_max <= 1.0e-12), "official_override_report": override_report, "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()}, "elapsed_seconds": float(time.time() - started), "decision": "candidate_pending_fresh_process_audit" if full_pass else "rejected_full_candidate_gate"}
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "changed_targets": ["nc", "eps"], "nc_feature_family": "fixed endpoint/physical/capped-charge plus MolMR, LabuteASA, atom-additive van-der-Waals volume and density ratios", "eps_route": "frozen C098 paired Ridge", "ridge_alpha": RIDGE_ALPHA, "residual_weight": RESIDUAL_WEIGHT, "no_sweep": True, "fresh_parent_replay": True})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent `{mean_parent:.12f}`; candidate `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`; replay max `{replay_max:.3e}`. No local_eval or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend(f"{digest}  SOURCE tools/{name}.py" for name, digest in report["source_hashes"].items())
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "parent_replay_max": replay_max, "decision": report["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
