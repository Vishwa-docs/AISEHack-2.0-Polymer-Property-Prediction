#!/usr/bin/env python3
"""C131: from-scratch PI1M denoising functional-group bottleneck.

PI1M is used only as an official unlabeled corpus.  A randomly initialized
denoising autoencoder learns a compact functional-group/descriptor bottleneck
from structure features, then target-local EPS/Nc residual heads are evaluated
under the same grouped clean gates as the incumbent.  No target labels enter
the representation fit and no prior prediction artifact is loaded.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as c127


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("eps", "nc")
SEED = 2026
N_FOLDS = 5
PI1M_ROWS = 200_000
LATENT_WIDTH = 32
RESIDUAL_SHRINKAGE = 0.20

SMARTS_PATTERNS = (
    "c1ccccc1", "C(=O)", "C(=O)O", "C(=O)N", "COC", "C-O-C", "S(=O)(=O)",
    "C#N", "[N+](=O)[O-]", "[OH]", "[NH]", "[nH]", "[Si]", "[P]", "[F,Cl,Br,I]",
    "C=C", "c", "[N,O,S]", "[o,s]", "[C;H1,H2,H3,H4]",
)
PATTERN_MOLS = tuple(Chem.MolFromSmarts(value) for value in SMARTS_PATTERNS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint(path: Path, stage: str, **payload: Any) -> None:
    record = {"stage": stage, "time": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def capped(smiles: str) -> Chem.Mol | None:
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        rw = Chem.RWMol(Chem.Mol(molecule))
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(6)
                atom.SetFormalCharge(0)
                atom.SetNoImplicit(False)
        molecule = rw.GetMol()
        Chem.SanitizeMol(molecule)
        return molecule
    except Exception:
        return None


def structure_vector(smiles: str) -> np.ndarray:
    """Functional-group counts plus normalized physical/electronic descriptors."""
    molecule = capped(smiles)
    values = np.full(48, np.nan, dtype=np.float64)
    if molecule is None:
        return values
    try:
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        heavy = max(molecule.GetNumHeavyAtoms(), 1)
        bond_count = max(len(bonds), 1)
        patterns = [float(molecule.HasSubstructMatch(pattern)) if pattern is not None else 0.0 for pattern in PATTERN_MOLS]
        for index in range(len(patterns)):
            patterns[index] = patterns[index] / max(heavy, 1)
        estate = np.asarray(rdMolDescriptors.CalcNumHBA(molecule), dtype=np.float64)
        values[:20] = patterns
        values[20:] = [
            float(Descriptors.MolWt(molecule) / heavy), float(Descriptors.ExactMolWt(molecule) / heavy),
            float(Crippen.MolLogP(molecule) / heavy), float(Crippen.MolMR(molecule) / heavy),
            float(rdMolDescriptors.CalcTPSA(molecule) / heavy), float(rdMolDescriptors.CalcNumHBA(molecule) / heavy),
            float(rdMolDescriptors.CalcNumHBD(molecule) / heavy), float(molecule.GetRingInfo().NumRings() / heavy),
            float(rdMolDescriptors.CalcNumAromaticRings(molecule) / heavy), float(Descriptors.NumRotatableBonds(molecule) / heavy),
            float(rdMolDescriptors.CalcFractionCSP3(molecule)), float(sum(atom.GetIsAromatic() for atom in atoms) / heavy),
            float(sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms) / heavy),
            float(sum(atom.GetAtomicNum() == 7 for atom in atoms) / heavy), float(sum(atom.GetAtomicNum() == 8 for atom in atoms) / heavy),
            float(sum(atom.GetAtomicNum() == 16 for atom in atoms) / heavy),
            float(sum(bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.0 for bond in bonds) / bond_count),
            float(sum(bond.GetBondTypeAsDouble() == 2.0 for bond in bonds) / bond_count),
            float(sum(atom.GetFormalCharge() for atom in atoms)), float(np.log1p(heavy)),
            float(len(atoms)), float(len(bonds)), float(estate), float(np.std([atom.GetAtomicNum() for atom in atoms])),
            float(np.mean([atom.GetAtomicNum() for atom in atoms])), float(np.max([atom.GetAtomicNum() for atom in atoms])),
            float(np.min([atom.GetAtomicNum() for atom in atoms])), float(sum(atom.GetHybridization() == Chem.HybridizationType.SP2 for atom in atoms) / heavy),
        ]
    except Exception:
        return values
    values[~np.isfinite(values)] = np.nan
    return values


def feature_matrix(smiles: list[str]) -> np.ndarray:
    return np.vstack([structure_vector(value) for value in smiles]).astype(np.float64, copy=False)


def latent_from_mlp(model: MLPRegressor, matrix: np.ndarray) -> np.ndarray:
    first = np.maximum(0.0, matrix @ model.coefs_[0] + model.intercepts_[0])
    second = np.maximum(0.0, first @ model.coefs_[1] + model.intercepts_[1])
    return second.astype(np.float64, copy=False)


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    return c127.grouped_folds(groups)


def fit_residual_arms(features: np.ndarray, latent: np.ndarray, train_global: np.ndarray, valid_global: np.ndarray, residual: np.ndarray) -> np.ndarray:
    ridge_latent = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0))
    ridge_raw = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0))
    ridge_latent.fit(latent[train_global], residual)
    ridge_raw.fit(features[train_global], residual)
    return np.column_stack([ridge_latent.predict(latent[valid_global]), ridge_raw.predict(features[valid_global])]).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    data_dir = (root / args.data_dir).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    progress = run_dir / "progress.jsonl"
    checkpoint(progress, "started", experiment_id=run_dir.name)
    parent = parent_builder.build_parent(root, data_dir)
    parity = c127.source_parity(root, parent, (root / args.canonical_run).resolve())
    checkpoint(progress, "parent_parity", **parity)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    pi1m_path = data_dir / "PI1M.csv"
    pi1m_hash = sha256_file(pi1m_path)
    pi1m = pd.read_csv(pi1m_path, usecols=["SMILES"])
    if len(pi1m) < PI1M_ROWS or pi1m["SMILES"].isna().any():
        raise RuntimeError("PI1M schema/row preflight failed")
    # Deterministic stride selection spreads the unlabeled fit over the full corpus.
    selection = np.linspace(0, len(pi1m) - 1, PI1M_ROWS, dtype=np.int64)
    pi_smiles = pi1m.iloc[selection]["SMILES"].astype(str).tolist()
    pi_features = feature_matrix(pi_smiles)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    pi_clean = scaler.fit_transform(imputer.fit_transform(pi_features))
    rng = np.random.default_rng(SEED)
    noisy = pi_clean + rng.normal(0.0, 0.05, size=pi_clean.shape)
    mask = rng.random(pi_clean.shape) < 0.12
    noisy[mask] = 0.0
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(96, LATENT_WIDTH, 96), activation="relu", solver="adam", batch_size=1024,
        learning_rate_init=0.001, max_iter=30, random_state=SEED, early_stopping=True,
        validation_fraction=0.02, n_iter_no_change=5, tol=1.0e-4,
    )
    autoencoder.fit(noisy, pi_clean)
    checkpoint(progress, "pi1m_denoising_complete", pi1m_rows_used=PI1M_ROWS, pi1m_sha256=pi1m_hash, valid_rows=int(np.sum(np.isfinite(pi_features).all(axis=1))), iterations=int(autoencoder.n_iter_))

    official_features = feature_matrix(parent["keys"])
    official_scaled = scaler.transform(imputer.transform(official_features))
    official_latent = latent_from_mlp(autoencoder, official_scaled)
    feature_report = {"raw_features": int(official_features.shape[1]), "latent_features": int(official_latent.shape[1]), "pi1m_rows_used": PI1M_ROWS, "pi1m_sha256": pi1m_hash, "label_free_representation": True}
    checkpoint(progress, "official_bottleneck_constructed", **feature_report)

    target_reports: dict[str, Any] = {}
    candidate_oof: dict[str, np.ndarray] = {}
    candidate_test: dict[str, np.ndarray] = {}
    for target in ACTIVE_TARGETS:
        info = dict(parent["target_info"][target])
        info["fingerprints"] = parent["fingerprints"]
        y = np.asarray(info["y"], dtype=np.float64)
        parent_oof = np.asarray(info["parent"], dtype=np.float64)
        target_global = np.asarray(info["indices"], dtype=np.int64)
        folds = grouped_folds(np.asarray(info["groups"], dtype=object))
        residual = y - parent_oof
        direct_oof = np.full((len(y), 2), np.nan, dtype=np.float64)
        for fold in range(N_FOLDS):
            validation = np.flatnonzero(folds == fold)
            training = np.flatnonzero(folds != fold)
            direct_oof[validation] = parent_oof[validation, None] + RESIDUAL_SHRINKAGE * fit_residual_arms(official_features, official_latent, target_global[training], target_global[validation], residual[training])
            checkpoint(progress, f"{target}_fold_{fold}", rows=int(len(validation)))
        arms = np.column_stack([parent_oof, direct_oof])
        weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, arms)
        candidate = arms @ weights + intercept
        report = c127.evaluate_target(info, {"candidate": candidate})
        report.update({"blend_name": blend_name, "blend_weights": [float(value) for value in weights], "blend_intercept": float(intercept), "blend_r2": float(blend_r2), "feature_report": feature_report, "residual_shrinkage": RESIDUAL_SHRINKAGE})
        target_reports[target] = report
        candidate_oof[target] = candidate
        test_frame = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_indices = np.asarray([parent["key_to_index"][value] for value in test_frame["canonical"]], dtype=np.int64)
        test_residual_arms = fit_residual_arms(official_features, official_latent, target_global, test_indices, residual)
        test_parent = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id")["target"].to_numpy(np.float64)
        candidate_test[target] = np.column_stack([test_parent, test_parent[:, None] + RESIDUAL_SHRINKAGE * test_residual_arms]) @ weights + intercept
        checkpoint(progress, f"{target}_complete", delta_r2=report["delta_r2"], positive_folds=report["positive_folds"], group_bootstrap_lower=report["group_bootstrap_lower"], minimum_panel_delta=report["minimum_panel_delta"], pass_gate=report["pass"])

    banked = [target for target in ACTIVE_TARGETS if target_reports[target]["pass"]]
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = parent["target_info"][target]
        y = np.asarray(info["y"], dtype=np.float64)
        parent_oof = np.asarray(info["parent"], dtype=np.float64)
        candidate = candidate_oof[target] if target in candidate_oof else parent_oof
        assembled = candidate if target in banked else parent_oof
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent_oof, "candidate": candidate, "assembled": assembled, "group": info["groups"], "scaffold": info["scaffolds"], "outer_fold": grouped_folds(np.asarray(info["groups"], dtype=object))}))
        if target not in target_reports:
            target_reports[target] = {"parent_r2": float(reference.r2_score(y, parent_oof)), "candidate_r2": float(reference.r2_score(y, parent_oof)), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}}, "folds": [], "pass": True, "unchanged_parent": True}
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reference.r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(mean_candidate >= mean_parent + 0.002 and max_loss >= -0.003 and len(banked) > 0)

    parent_detail = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    parts: list[pd.DataFrame] = []
    for target in TARGETS:
        frame = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        values = candidate_test[target] if target in banked else parent_detail.loc[parent_detail["target_type"] == target].sort_values("id")["target"].to_numpy(np.float64)
        parts.append(pd.DataFrame({"id": frame["id"].astype(int), "target_type": target, "model_prediction": values}))
    raw = pd.concat(parts, ignore_index=True).sort_values("id")
    raw_labels, _ = reference.build_label_pool(parent["train"], parent["archive"])
    detail, override_report = reference.apply_official_overrides(raw, parent["test"], raw_labels)
    predictions = detail[["id", "target"]].copy()
    if len(predictions) != 4940 or not predictions["id"].equals(parent["test"]["id"]) or not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C131 complete output contract failed")

    source_paths = {
        "runner": Path(__file__),
        "parent_builder": root / "tools/round2_c097_graph_grammar_hgb_full.py",
        "reference": root / "tools/initial_reference_pipeline.py",
        "c127_evaluation": root / "tools/round2_c127_round1_carrier_factory.py",
    }
    result = {
        "schema_version": "ppp.round2.c131.pi1m-denoising-bottleneck.run.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "parent": "C050 source rebuild; no C127-C130 prediction artifact input",
        "official_inputs": parent["inputs"],
        "pi1m_path": str(pi1m_path.relative_to(root)),
        "pi1m_sha256": pi1m_hash,
        "pi1m_rows_used": PI1M_ROWS,
        "official_only": True,
        "label_free_representation": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "pretrained_weights": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "banked_targets": banked,
        "target_reports": target_reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_pass,
        "official_override_report": override_report,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "elapsed_seconds": float(time.time() - started),
        "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate",
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c131.pi1m-denoising-bottleneck.v1", "active_targets": list(ACTIVE_TARGETS), "seed": SEED, "pi1m_rows": PI1M_ROWS, "latent_width": LATENT_WIDTH, "hidden_layers": [96, 32, 96], "mask_probability": 0.12, "noise_std": 0.05, "residual_shrinkage": RESIDUAL_SHRINKAGE, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{result['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{mean_parent:.12f}`; assembled `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. PI1M was used only for a from-scratch label-free denoising representation. No local_eval read.\n", encoding="utf-8")
    checkpoint(progress, "metrics_written", decision=result["decision"], mean_candidate_r2=mean_candidate, banked_targets=banked)
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, path in source_paths.items():
        manifest.append(f"{sha256_file(path)}  SOURCE {name}")
    manifest.append(f"{pi1m_hash}  SOURCE PI1M.csv")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": result["decision"], "banked_targets": banked, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
