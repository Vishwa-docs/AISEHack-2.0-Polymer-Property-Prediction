#!/usr/bin/env python3
"""C236: Nc backbone/pendant polarizability partition.

This is a queue-safety continuation after C235.  It targets the largest
unbanked bottleneck (Nc) with one distinct, label-free structural factor:
partition per-atom Crippen molar refractivity/logP contributions into the
shortest wildcard-to-wildcard backbone and pendant atoms.  A fixed low-variance
residual model is trained fold-locally on original official Nc labels.

It is not a C180/C226 guard retune, not robust-rank/optical-dispersion, not an
EPS-to-Nc counterpart route, not a PI1M retry, and not stored prediction replay.
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
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c208_tg_robust_group_measurement as c208


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGET = "nc"
SCHEMA = "ppp.round2.c236.nc-backbone-pendant-polarizability.v1"
SEED = 20260805
RIDGE_ALPHA = 60.0
TREE_COUNT = 180
TREE_MIN_LEAF = 3
RESIDUAL_WEIGHT = 0.25
RIDGE_BLEND_WEIGHT = 0.65
TREE_BLEND_WEIGHT = 0.35
MIN_FULL_MEAN_GAIN = 0.002


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


def part_stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.sum(values)),
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
    ]


def partition_masks(molecule: Chem.Mol) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    atoms = list(molecule.GetAtoms())
    n_atoms = len(atoms)
    real = np.asarray([atom.GetAtomicNum() > 0 for atom in atoms], dtype=bool)
    dummy_indices = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 0]
    backbone = np.zeros(n_atoms, dtype=bool)
    path_length = 0
    if len(dummy_indices) >= 2:
        try:
            path = Chem.GetShortestPath(molecule, int(dummy_indices[0]), int(dummy_indices[1]))
        except Exception:
            path = tuple()
        path_length = max(len(path) - 1, 0)
        for idx in path:
            if idx < n_atoms and real[idx]:
                backbone[idx] = True
    if not np.any(backbone & real):
        backbone = real.copy()
    pendant = real & ~backbone
    attachment_bonds = 0
    for bond in molecule.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if (backbone[a] and pendant[b]) or (backbone[b] and pendant[a]):
            attachment_bonds += 1
    return backbone, pendant, {
        "dummy_atom_count": int(len(dummy_indices)),
        "wildcard_path_length": int(path_length),
        "real_atom_count": int(np.sum(real)),
        "backbone_atom_count": int(np.sum(backbone)),
        "pendant_atom_count": int(np.sum(pendant)),
        "attachment_bond_count": int(attachment_bonds),
    }


def partition_feature_row(molecule: Chem.Mol) -> tuple[list[float], dict[str, Any]]:
    atoms = list(molecule.GetAtoms())
    n_atoms = len(atoms)
    if n_atoms == 0:
        return [0.0] * 72, {"empty_molecule": True}
    backbone, pendant, meta = partition_masks(molecule)
    real = np.asarray([atom.GetAtomicNum() > 0 for atom in atoms], dtype=bool)
    heavy_count = max(int(np.sum(real)), 1)
    try:
        contribs = Crippen._GetAtomContribs(molecule)
        logp = np.asarray([item[0] for item in contribs], dtype=np.float64)
        mr = np.asarray([item[1] for item in contribs], dtype=np.float64)
    except Exception:
        logp = np.zeros(n_atoms, dtype=np.float64)
        mr = np.zeros(n_atoms, dtype=np.float64)
        meta["crippen_fallback"] = True
    atomic = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=np.float64)
    degree = np.asarray([atom.GetDegree() for atom in atoms], dtype=np.float64)
    aromatic = np.asarray([atom.GetIsAromatic() for atom in atoms], dtype=np.float64)
    hetero = np.asarray([atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms], dtype=np.float64)
    polar = np.asarray([atom.GetAtomicNum() in (7, 8, 9, 15, 16, 17, 35, 53) for atom in atoms], dtype=np.float64)
    halogen = np.asarray([atom.GetAtomicNum() in (9, 17, 35, 53) for atom in atoms], dtype=np.float64)

    def block(mask: np.ndarray) -> list[float]:
        denom = max(int(np.sum(mask)), 1)
        return [
            float(np.sum(mask)),
            float(np.sum(mask) / heavy_count),
            *part_stats(mr[mask]),
            *part_stats(logp[mask]),
            float(np.sum(hetero[mask]) / denom),
            float(np.sum(polar[mask]) / denom),
            float(np.sum(aromatic[mask]) / denom),
            float(np.sum(halogen[mask]) / denom),
            float(np.mean(degree[mask])) if np.any(mask) else 0.0,
            float(np.mean(atomic[mask])) if np.any(mask) else 0.0,
        ]

    all_real = real
    row: list[float] = []
    row.extend(block(all_real))
    row.extend(block(backbone))
    row.extend(block(pendant))
    backbone_mr = float(np.sum(mr[backbone])) if np.any(backbone) else 0.0
    pendant_mr = float(np.sum(mr[pendant])) if np.any(pendant) else 0.0
    total_mr = float(np.sum(mr[real])) if np.any(real) else 0.0
    backbone_logp = float(np.sum(logp[backbone])) if np.any(backbone) else 0.0
    pendant_logp = float(np.sum(logp[pendant])) if np.any(pendant) else 0.0
    safe_total = max(abs(total_mr), 1.0e-12)
    safe_backbone = max(abs(backbone_mr), 1.0e-12)
    try:
        tpsa = float(rdMolDescriptors.CalcTPSA(molecule))
        rings = float(rdMolDescriptors.CalcNumRings(molecule))
        aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(molecule))
        rotatable = float(Descriptors.NumRotatableBonds(molecule))
    except Exception:
        tpsa = rings = aromatic_rings = rotatable = 0.0
    row.extend(
        [
            float(meta["dummy_atom_count"]),
            float(meta["wildcard_path_length"]),
            float(meta["backbone_atom_count"]),
            float(meta["pendant_atom_count"]),
            float(meta["attachment_bond_count"]),
            float(backbone_mr / safe_total),
            float(pendant_mr / safe_total),
            float(pendant_mr / safe_backbone),
            float(backbone_mr - pendant_mr),
            float(abs(backbone_mr - pendant_mr)),
            float(backbone_logp - pendant_logp),
            float(abs(backbone_logp - pendant_logp)),
            float(tpsa / heavy_count),
            float(rings),
            float(aromatic_rings),
            float(rotatable / heavy_count),
            float((pendant_mr ** 2) / max(meta["pendant_atom_count"], 1)),
            float((backbone_mr ** 2) / max(meta["backbone_atom_count"], 1)),
        ]
    )
    if len(row) != 72:
        raise RuntimeError(f"unexpected C236 feature width {len(row)}")
    return row, meta


def partition_features(parent: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[list[float]] = []
    metas: list[dict[str, Any]] = []
    failures = 0
    for molecule in parent["molecules"]:
        try:
            row, meta = partition_feature_row(molecule)
        except Exception:
            failures += 1
            row, meta = [0.0] * 72, {"feature_failure": True}
        rows.append(row)
        metas.append(meta)
    matrix = np.asarray(rows, dtype=np.float64)
    matrix[~np.isfinite(matrix) | (np.abs(matrix) > 1.0e12)] = np.nan
    pendant_counts = np.asarray([item.get("pendant_atom_count", 0) for item in metas], dtype=np.float64)
    return matrix, {
        "feature_family": "wildcard_shortest_path_backbone_pendant_crippen_partition",
        "shape": [int(value) for value in matrix.shape],
        "feature_count": int(matrix.shape[1]),
        "feature_failures": int(failures),
        "dummy_atoms_source": "RDKit atomic number 0 parsed from official wildcard SMILES",
        "backbone_definition": "shortest path between first two wildcard atoms; fallback to all real atoms if unavailable",
        "pendant_rows": int(np.sum(pendant_counts > 0)),
        "pendant_atom_mean": float(np.mean(pendant_counts)),
        "uses_labels": False,
        "uses_cross_property_labels": False,
        "uses_pi1m": False,
        "uses_stored_predictions": False,
    }


def fit_models(x_train: np.ndarray, y_train: np.ndarray) -> list[Any]:
    return [
        make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA),
        ).fit(x_train, y_train),
        make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=TREE_COUNT,
                random_state=SEED,
                min_samples_leaf=TREE_MIN_LEAF,
                max_features=0.85,
                n_jobs=1,
            ),
        ).fit(x_train, y_train),
    ]


def predict_blend(models: list[Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ridge = np.asarray(models[0].predict(x), dtype=np.float64)
    tree = np.asarray(models[1].predict(x), dtype=np.float64)
    direct = RIDGE_BLEND_WEIGHT * ridge + TREE_BLEND_WEIGHT * tree
    return direct, np.column_stack([ridge, tree])


def fit_nc_partition(
    info: dict[str, Any],
    features: np.ndarray,
    test_indices: np.ndarray,
    test_parent: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent = np.asarray(info["parent"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = carrier.grouped_folds(groups)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    direct_oof = np.full((len(y), 2), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        x_train = features[indices[training]]
        residual = y[training] - parent[training]
        models = fit_models(x_train, residual)
        correction, direct = predict_blend(models, features[indices[validation]])
        direct_oof[validation] = direct
        fold_candidate = reference.clip_prediction(y[training], parent[validation] + RESIDUAL_WEIGHT * correction)
        candidate[validation] = fold_candidate
        fold_reports.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "train_rows": int(len(training)),
                "residual_mean": float(np.mean(residual)),
                "residual_std": float(np.std(residual)),
                "candidate_delta_r2": float(r2_score(y[validation], fold_candidate) - r2_score(y[validation], parent[validation])),
            }
        )
    if not np.isfinite(candidate).all():
        raise RuntimeError("C236 produced non-finite OOF candidate")
    full_models = fit_models(features[indices], y - parent)
    test_correction, _ = predict_blend(full_models, features[test_indices])
    test_candidate = reference.clip_prediction(y, test_parent + RESIDUAL_WEIGHT * test_correction)
    return {
        "candidate": candidate,
        "test_candidate": test_candidate,
        "direct_oof": direct_oof,
        "blend_name": "fixed_0.25_residual__0.65_ridge_0.35_extratrees",
        "weights": [RIDGE_BLEND_WEIGHT, TREE_BLEND_WEIGHT],
        "intercept": 0.0,
        "blend_r2": float(r2_score(y, candidate)),
        "fold_robust_reports": fold_reports,
        "full_robust_report": {
            "train_rows": int(len(y)),
            "test_rows": int(len(test_indices)),
            "residual_weight": RESIDUAL_WEIGHT,
            "ridge_alpha": RIDGE_ALPHA,
            "tree_count": TREE_COUNT,
            "tree_min_leaf": TREE_MIN_LEAF,
        },
    }


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

    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(progress, "parent_parity", **parity)

    features, feature_report = partition_features(parent)
    checkpoint(progress, "features_complete", shape=feature_report["shape"], pendant_rows=feature_report["pendant_rows"])

    info = dict(parent["target_info"][ACTIVE_TARGET])
    info["fingerprints"] = parent["fingerprints"]
    test_rows, test_indices, test_parent = c208.target_test_rows(parent, ACTIVE_TARGET)
    result = fit_nc_partition(info, features, test_indices, test_parent)
    active_report = c208.evaluate_tg(info, result)
    active_report.update(
        {
            "changed_factor": "Nc backbone/pendant polarizability partition residual",
            "uses_backbone_pendant_partition": True,
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
            "c180_guard_rank_optical_eps_partner_routes_not_reused": True,
        }
    )
    banked = bool(active_report["pass"])
    checkpoint(
        progress,
        "nc_backbone_pendant_complete",
        delta_r2=active_report["delta_r2"],
        candidate_r2=active_report["candidate_r2"],
        pass_gate=banked,
        positive_folds=active_report["positive_folds"],
        minimum_panel_delta=active_report["minimum_panel_delta"],
        group_bootstrap_lower=active_report["group_bootstrap_lower"],
    )

    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        target_info = dict(parent["target_info"][target])
        if target == ACTIVE_TARGET:
            report = active_report
            candidate = np.asarray(result["candidate"], dtype=np.float64)
            direct_oof = np.asarray(result["direct_oof"], dtype=np.float64)
        else:
            report = c208.unchanged_report(target_info)
            candidate = np.asarray(target_info["parent"], dtype=np.float64)
            direct_oof = np.full((len(candidate), 2), np.nan, dtype=np.float64)
        target_reports[target] = report
        folds = carrier.grouped_folds(np.asarray(target_info["groups"], dtype=object))
        assembled = candidate if target == ACTIVE_TARGET and banked else np.asarray(target_info["parent"], dtype=np.float64)
        oof_parts.append(
            pd.DataFrame(
                {
                    "canonical": target_info["canonical"],
                    "target_type": target,
                    "target": target_info["y"],
                    "parent": target_info["parent"],
                    "candidate": candidate,
                    "assembled": assembled,
                    "direct_ridge": direct_oof[:, 0],
                    "direct_tree": direct_oof[:, 1],
                    "group": target_info["groups"],
                    "scaffold": target_info["scaffolds"],
                    "fold": folds,
                    "banked": bool(target == ACTIVE_TARGET and banked),
                }
            )
        )

    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target == ACTIVE_TARGET and banked else 0.0 for target in TARGETS))
    full_candidate_gate_pass = bool(banked and assembled_mean - parent_mean >= MIN_FULL_MEAN_GAIN and max_loss >= -0.003)

    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    component_test = pd.DataFrame(
        {
            "id": test_rows["id"].astype(int),
            "target_type": ACTIVE_TARGET,
            "direct_candidate": result["test_candidate"],
        }
    )
    predictions = parent_test.merge(component_test, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(
        (predictions["target_type"] == ACTIVE_TARGET) & banked,
        predictions["direct_candidate"],
        predictions["target"],
    )
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940:
        raise RuntimeError("C236 prediction row count failed")
    if not np.array_equal(predictions["id"].to_numpy(np.int64), np.arange(1, 4941, dtype=np.int64)):
        raise RuntimeError("C236 prediction ID order failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C236 prediction finite check failed")

    report = {
        "schema_version": SCHEMA,
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pi1m_used": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "stored_prediction_replay": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "active_target": ACTIVE_TARGET,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "target_reports": target_reports,
        "banked_targets": [ACTIVE_TARGET] if banked else [],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": bool(full_candidate_gate_pass and assembled_mean >= 0.95),
        "decision": "banked_component_pending_compound_audit" if banked else "rejected_component_or_full_gate",
        "component_diagnostics": {
            "active_target": ACTIVE_TARGET,
            "changed_factor": "Nc backbone/pendant polarizability partition residual",
            "nc_delta_r2": active_report["delta_r2"],
            "nc_positive_folds": active_report["positive_folds"],
            "nc_group_bootstrap_lower": active_report["group_bootstrap_lower"],
            "nc_minimum_panel_delta": active_report["minimum_panel_delta"],
            "c180_guard_rank_optical_eps_partner_routes_not_reused": True,
            "uses_cross_property_labels": False,
            "uses_pi1m": False,
        },
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "carrier": sha256_file(round2_root / "tools/round2_c127_round1_carrier_factory.py"),
            "parent_builder": sha256_file(round2_root / "tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(round2_root / "tools/initial_reference_pipeline.py"),
            "c208_panel_helper": sha256_file(round2_root / "tools/round2_c208_tg_robust_group_measurement.py"),
        },
    }

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    component_test.to_csv(run_dir / "nc_component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA,
            "active_target": ACTIVE_TARGET,
            "features": "wildcard shortest-path backbone/pendant Crippen MR/logP partition",
            "model": "fixed 0.25 residual blend of Ridge and ExtraTrees",
            "ridge_alpha": RIDGE_ALPHA,
            "tree_count": TREE_COUNT,
            "tree_min_leaf": TREE_MIN_LEAF,
            "residual_weight": RESIDUAL_WEIGHT,
            "component_gate": {
                "minimum_delta_r2": 0.01,
                "minimum_positive_folds": 4,
                "bootstrap_lower_must_exceed": 0.0,
                "minimum_panel_delta_must_be_at_least": 0.0,
            },
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
                f"rdkit={rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\n"
        f"Decision: **{report['decision']}**. Nc parent R² "
        f"`{active_report['parent_r2']:.12f}`, candidate R² "
        f"`{active_report['candidate_r2']:.12f}`, delta "
        f"`{active_report['delta_r2']:+.12f}`. Official-only; no local_eval, "
        "Kaggle compute, upload, submission, or final-notebook action.\n",
        encoding="utf-8",
    )
    manifest_paths = [path for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact_manifest.sha256"]
    lines = [f"{sha256_file(path)}  {path.name}" for path in manifest_paths]
    lines.extend(f"{digest}  SOURCE {name}" for name, digest in sorted(report["source_hashes"].items()))
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "decision": report["decision"],
                "nc_parent_r2": active_report["parent_r2"],
                "nc_candidate_r2": active_report["candidate_r2"],
                "nc_delta_r2": active_report["delta_r2"],
                "banked_targets": report["banked_targets"],
                "mean_candidate_r2": report["mean_candidate_r2"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
