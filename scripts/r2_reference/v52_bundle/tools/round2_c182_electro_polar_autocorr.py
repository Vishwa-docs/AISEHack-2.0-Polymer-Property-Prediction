#!/usr/bin/env python3
"""C182: compact RDKit topological electro-polar autocorrelation residuals.

This experiment is intentionally cheaper and more interpretable than a GNN
or conformer model.  It computes distance-lagged products, centered products,
and squared differences for charge/electronic/topological atom channels on
the official monomer graph and a deterministic endpoint-closed graph.  Fixed
Ridge residual heads challenge the C050 parent on Ei, EPS, and Nc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, EState, rdPartialCharges
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = ("ei", "eps", "nc")
SEED = 2026
MAX_DISTANCE = 8
ALPHA = 30.0
RESIDUAL_WEIGHT = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def atom_channels(mol: Chem.Mol) -> np.ndarray:
    work = Chem.Mol(mol)
    try:
        rdPartialCharges.ComputeGasteigerCharges(work)
    except Exception:
        pass
    try:
        estate = EState.EStateIndices(work)
    except Exception:
        estate = [0.0] * work.GetNumAtoms()
    try:
        crippen = Crippen._GetAtomContribs(work)[0]
    except Exception:
        crippen = [(0.0, 0.0)] * work.GetNumAtoms()
    values = np.zeros((work.GetNumAtoms(), 8), dtype=np.float64)
    for row, atom in enumerate(work.GetAtoms()):
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            charge = 0.0
        try:
            mr = float(crippen[row][1])
        except Exception:
            mr = 0.0
        values[row] = [
            charge,
            abs(charge),
            float(estate[row]) if row < len(estate) else 0.0,
            mr,
            float(atom.GetAtomicNum()),
            float(atom.GetIsAromatic()),
            float(atom.GetHybridization() == Chem.HybridizationType.SP2),
            float(atom.GetAtomicNum() not in (0, 1, 6)),
        ]
    values[~np.isfinite(values)] = 0.0
    return values


def closed_core(mol: Chem.Mol) -> Chem.Mol | None:
    dummy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy) != 2 or any(atom.GetDegree() != 1 for atom in dummy):
        return None
    neighbor_ids = [atom.GetNeighbors()[0].GetIdx() for atom in dummy]
    keep = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0]
    mapping = {old: new for new, old in enumerate(keep)}
    rw = Chem.RWMol(Chem.Mol())
    for old in keep:
        rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old)))
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in mapping and b in mapping:
            rw.AddBond(mapping[a], mapping[b], bond.GetBondType())
    left, right = (mapping.get(value) for value in neighbor_ids)
    if left is None or right is None or left == right or rw.GetBondBetweenAtoms(left, right) is not None:
        return None
    rw.AddBond(left, right, Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def autocorr_for_graph(mol: Chem.Mol) -> np.ndarray:
    channels = atom_channels(mol)
    distance = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.int64)
    rows: list[float] = []
    for lag in range(1, MAX_DISTANCE + 1):
        mask = np.triu(distance == lag, k=1)
        left, right = np.where(mask)
        if len(left) == 0:
            rows.extend([0.0] * (channels.shape[1] * 3))
            continue
        a, b = channels[left], channels[right]
        for col in range(channels.shape[1]):
            x, y = a[:, col], b[:, col]
            rows.extend([
                float(np.mean(x * y)),
                float(np.mean((x - np.mean(channels[:, col])) * (y - np.mean(channels[:, col])))),
                float(np.mean((x - y) ** 2)),
            ])
    return np.asarray(rows, dtype=np.float64)


def feature_matrix(molecules: list[Chem.Mol]) -> tuple[np.ndarray, dict[str, Any]]:
    output: list[np.ndarray] = []
    closed = 0
    for mol in molecules:
        base = autocorr_for_graph(mol)
        core = closed_core(mol)
        if core is None:
            periodic = np.zeros_like(base)
        else:
            closed += 1
            periodic = autocorr_for_graph(core)
        output.append(np.concatenate([base, periodic]))
    values = np.vstack(output).astype(np.float64)
    values[~np.isfinite(values)] = 0.0
    return values, {
        "source": "official SMILES only; RDKit Gasteiger/EState/Crippen and graph distances",
        "max_distance": MAX_DISTANCE,
        "channels": ["charge", "abs_charge", "estate", "molar_refractivity", "atomic_number", "aromatic", "sp2", "hetero"],
        "forms": ["lag_product", "centered_product", "geary_squared_difference"],
        "monomer_feature_count": int(values.shape[1] // 2),
        "total_feature_count": int(values.shape[1]),
        "periodic_supported_rows": int(closed),
        "nonfinite_after_sanitize": int(np.size(values) - np.isfinite(values).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    parent = parent_builder.build_parent(root, data_dir)
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    features, feature_report = feature_matrix(parent["molecules"])
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        y = np.asarray(info["y"], dtype=float)
        parent_oof = np.asarray(info["parent"], dtype=float)
        indices = np.asarray(info["indices"], dtype=np.int64)
        groups = np.asarray(info["groups"], dtype=object)
        folds = carrier.grouped_folds(groups)
        residual_oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(carrier.N_FOLDS):
            validation = np.flatnonzero(folds == fold)
            training = np.flatnonzero(folds != fold)
            imputer_mean = np.nanmedian(features[indices[training]], axis=0)
            train_x = np.nan_to_num(features[indices[training]], nan=imputer_mean)
            valid_x = np.nan_to_num(features[indices[validation]], nan=imputer_mean)
            model = Ridge(alpha=ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
            model.fit(train_x, y[training] - parent_oof[training])
            residual_oof[validation] = model.predict(valid_x)
            candidate_fold = parent_oof[validation] + RESIDUAL_WEIGHT * residual_oof[validation]
            fold_rows.append({
                "fold": fold,
                "rows": int(len(validation)),
                "parent_r2": float(r2_score(y[validation], parent_oof[validation])),
                "candidate_r2": float(r2_score(y[validation], candidate_fold)),
                "delta_r2": float(r2_score(y[validation], candidate_fold) - r2_score(y[validation], parent_oof[validation])),
            })
        candidate = parent_oof + RESIDUAL_WEIGHT * residual_oof
        delta = float(r2_score(y, candidate) - r2_score(y, parent_oof))
        lower = carrier.bootstrap_lower(y, parent_oof, candidate, groups)
        positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        selected = bool(delta >= 0.005 and positive >= 4 and lower > 0.0)
        full_mean = np.nanmedian(features[indices], axis=0)
        full_x = np.nan_to_num(features[indices], nan=full_mean)
        test_indices = np.asarray([parent["key_to_index"][value] for value in parent["test"].loc[parent["test"]["target_type"] == target, "canonical"]], dtype=np.int64)
        test_x = np.nan_to_num(features[test_indices], nan=full_mean)
        model = Ridge(alpha=ALPHA, solver="lsqr", max_iter=5000, tol=1e-4)
        model.fit(full_x, y - parent_oof)
        target_reports[target] = {
            "parent_r2": float(r2_score(y, parent_oof)),
            "candidate_r2": float(r2_score(y, candidate)),
            "delta_r2": delta,
            "positive_folds": positive,
            "group_bootstrap_lower": lower,
            "pass": selected,
            "folds": fold_rows,
        }
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent_oof, "candidate": candidate, "assembled": np.where(selected, candidate, parent_oof), "fold": folds}))
        test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_parts.append(pd.DataFrame({"id": test_rows["id"].astype(int), "target_type": target, "residual_candidate": parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target, "target"].to_numpy(float) + RESIDUAL_WEIGHT * model.predict(test_x)}))
    banked = [target for target in TARGETS if target_reports[target]["pass"]]
    oof = pd.concat(oof_parts, ignore_index=True)
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component = pd.concat(test_parts, ignore_index=True)
    predictions = parent_test.merge(component, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(predictions["target_type"].isin(banked), predictions["residual_candidate"], predictions["target"])
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C182 complete output contract failed")
    full_pass = bool(assembled_mean - parent_mean >= 0.002 and len(banked) > 0)
    report = {
        "schema_version": "ppp.round2.c182.electro-polar-autocorr.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_pass,
        "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {"runner": sha256_file(Path(__file__)), "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"), "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"), "reference": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py")},
    }
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component.to_csv(run_dir / "component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c182.electro-polar-autocorr.v1", "seed": SEED, "max_distance": MAX_DISTANCE, "ridge_alpha": ALPHA, "residual_weight": RESIDUAL_WEIGHT, "active_targets": list(TARGETS), "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; gain `{assembled_mean - parent_mean:+.12f}`. Official-only; no local_eval read.\n", encoding="utf-8")
    manifest = [f"{sha256_file(path)}  {path.name}" for path in sorted(run_dir.iterdir()) if path.name != "artifact_manifest.sha256" and path.is_file()]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": assembled_mean, "mean_gain": assembled_mean - parent_mean, "decision": report["decision"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
