#!/usr/bin/env python3
"""C192: support-conditioned PI1M applicability residuals.

This is a clean official-only continuation after C191.  It deliberately differs
from C185: PI1M still contributes only unlabeled Morgan-bit density summaries,
but each weak-target residual head is fitted and evaluated inside a fixed
official-label availability stratum.  This tests whether PI1M applicability
signal was washed out by C185's global heads rather than being absent.
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


RDLogger.DisableLog("rdApp.*")

TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
ACTIVE_TARGETS = ("ei", "eea", "nc", "eps")
SEED = 20260805
MORGAN_BITS = 2048
PI1M_SAMPLE = 50000
RESIDUAL_WEIGHT = 0.25
RIDGE_ALPHA = 35.0
MIN_BANKABLE_DELTA_R2 = 0.01
PARTNERS = {
    "ei": ("eea", "egc"),
    "eea": ("ei", "egc"),
    "nc": ("eps",),
    "eps": ("nc",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def selected_pi1m(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=["SMILES"])
    values = frame["SMILES"].dropna().astype(str).drop_duplicates().tolist()
    values.sort(key=lambda value: hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest())
    return values[:PI1M_SAMPLE]


def bit_vector(molecule: Chem.Mol) -> np.ndarray:
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=MORGAN_BITS)
    return np.asarray(fingerprint, dtype=np.int8)


def build_pi1m_density_features(parent: dict[str, Any], pi1m_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    pi_smiles = selected_pi1m(pi1m_path)
    document_frequency = np.zeros(MORGAN_BITS, dtype=np.int64)
    valid_pi1m = 0
    for smiles in pi_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        document_frequency += bit_vector(molecule)
        valid_pi1m += 1
    if valid_pi1m < 1000:
        raise RuntimeError("too few valid PI1M molecules for density features")
    idf = np.log((valid_pi1m + 1.0) / (document_frequency + 1.0)) + 1.0
    low_cut = float(np.quantile(idf, 0.25))
    high_cut = float(np.quantile(idf, 0.75))
    rows: list[np.ndarray] = []
    for molecule in parent["molecules"]:
        bits = bit_vector(molecule).astype(bool)
        active = idf[bits]
        if len(active) == 0:
            active = np.asarray([float(np.mean(idf))], dtype=np.float64)
        heavy = max(int(molecule.GetNumHeavyAtoms()), 1)
        rows.append(np.asarray([
            float(np.sum(bits)),
            float(np.mean(active)),
            float(np.std(active)),
            float(np.max(active)),
            float(np.min(active)),
            float(np.sum(active) / max(1, np.sum(bits))),
            float(np.mean(active >= high_cut)),
            float(np.mean(active <= low_cut)),
            float(np.log1p(heavy)),
            float(molecule.GetNumAtoms()),
        ], dtype=np.float64))
    features = np.vstack(rows)
    features[~np.isfinite(features)] = 0.0
    return features, {
        "pi1m_source": str(pi1m_path),
        "pi1m_selected_rows": int(len(pi_smiles)),
        "pi1m_valid_molecules": int(valid_pi1m),
        "morgan_bits": int(MORGAN_BITS),
        "feature_shape": [int(value) for value in features.shape],
        "feature_names": [
            "morgan_bit_count",
            "mean_bit_idf",
            "std_bit_idf",
            "max_bit_idf",
            "min_bit_idf",
            "sum_bit_idf_per_bit",
            "rare_bit_fraction",
            "common_bit_fraction",
            "log_heavy_atoms",
            "atom_count",
        ],
        "labels_used_for_pi1m_features": False,
        "pretrained": False,
    }


def availability_sets(parent: dict[str, Any], excluded_groups: set[str] | None = None) -> dict[str, set[str]]:
    excluded_groups = set() if excluded_groups is None else {str(value) for value in excluded_groups}
    available: dict[str, set[str]] = {}
    for target in TARGETS:
        info = parent["target_info"][target]
        canonicals = np.asarray(info["canonical"], dtype=object)
        groups = np.asarray(info.get("groups", canonicals), dtype=object)
        values: set[str] = set()
        for canonical, group in zip(canonicals, groups, strict=True):
            canonical_key = str(canonical)
            group_key = str(group)
            if canonical_key in excluded_groups or group_key in excluded_groups:
                continue
            values.add(canonical_key)
        available[target] = values
    return available


def support_mask(canonicals: np.ndarray, target: str, available: dict[str, set[str]]) -> np.ndarray:
    partners = PARTNERS[target]
    return np.asarray([
        any(str(canonical) in available[partner] for partner in partners)
        for canonical in canonicals
    ], dtype=bool)


def fit_model() -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def stratum_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float | None:
    if int(np.sum(mask)) < 8 or float(np.var(y[mask])) <= 1.0e-15:
        return None
    return float(r2_score(y[mask], candidate[mask]) - r2_score(y[mask], parent[mask]))


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, mask: np.ndarray, minimum: int = 8) -> float | None:
    if int(np.sum(mask)) < minimum or float(np.var(y[mask])) <= 1.0e-15:
        return None
    return float(r2_score(y[mask], candidate[mask]) - r2_score(y[mask], parent[mask]))


def nearest_similarity(fingerprints: list[Any], indices: np.ndarray, folds: np.ndarray) -> np.ndarray:
    result = np.full(len(indices), np.nan, dtype=np.float64)
    for fold in sorted(set(int(value) for value in folds)):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [fingerprints[int(indices[row])] for row in training]
        for row in validation:
            result[row] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(indices[row])], train_fps))
    return result


def transfer_panels(
    info: dict[str, Any],
    fingerprints: list[Any],
    support_oof: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    folds = carrier.grouped_folds(groups)
    nearest = nearest_similarity(fingerprints, indices, folds)
    panel_specs: dict[str, np.ndarray] = {
        "partner_present": support_oof,
        "partner_missing": ~support_oof,
        "similarity_lt_0.30": nearest < 0.30,
        "similarity_0.30_0.50": (nearest >= 0.30) & (nearest < 0.50),
        "similarity_0.50_0.70": (nearest >= 0.50) & (nearest < 0.70),
        "similarity_ge_0.70": nearest >= 0.70,
        "quantile_low": y <= np.quantile(y, 0.25),
        "quantile_high": y >= np.quantile(y, 0.75),
    }
    for name in sorted(set(scaffolds)):
        selected = scaffolds == name
        if int(np.sum(selected)) >= 10:
            panel_specs[f"scaffold_{name}"] = selected
    panels: dict[str, Any] = {}
    values: list[float] = []
    for name, selected in panel_specs.items():
        delta = panel_delta(y, parent_oof, candidate, selected)
        panels[name] = {
            "rows": int(np.sum(selected)),
            "delta_r2": delta,
            "status": "evaluable" if delta is not None else "inapplicable_insufficient_support",
        }
        if delta is not None:
            values.append(delta)
    return {
        "panels": panels,
        "minimum_transfer_panel_delta": float(min(values)) if values else 0.0,
        "nearest_tanimoto": nearest,
    }


def target_candidate(
    x_all: np.ndarray,
    info: dict[str, Any],
    target: str,
    parent: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(info["y"], dtype=np.float64)
    parent_oof = np.asarray(info["parent"], dtype=np.float64)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    canonicals = np.asarray(info["canonical"], dtype=object)
    support_oof = np.zeros(len(y), dtype=bool)
    folds = carrier.grouped_folds(groups)
    residual_oof = np.zeros(len(y), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(carrier.N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        validation_groups = {str(value) for value in groups[validation]}
        fold_available = availability_sets(parent, validation_groups)
        support_train = support_mask(canonicals[training], target, fold_available)
        support_validation = support_mask(canonicals[validation], target, fold_available)
        support_oof[validation] = support_validation
        prediction = np.zeros(len(validation), dtype=np.float64)
        for stratum_value, stratum_name in ((False, "partner_missing"), (True, "partner_present")):
            train_mask = training[support_train == stratum_value]
            valid_mask = validation[support_validation == stratum_value]
            if len(valid_mask) == 0:
                continue
            if len(train_mask) < 12:
                prediction[np.isin(validation, valid_mask)] = 0.0
                continue
            model = fit_model()
            model.fit(x_all[indices[train_mask]], y[train_mask] - parent_oof[train_mask])
            prediction[np.isin(validation, valid_mask)] = model.predict(x_all[indices[valid_mask]])
        residual_oof[validation] = prediction
        fold_candidate = parent_oof[validation] + RESIDUAL_WEIGHT * residual_oof[validation]
        fold_rows.append({
            "fold": int(fold),
            "rows": int(len(validation)),
            "partner_present_rows": int(np.sum(support_validation)),
            "partner_missing_rows": int(np.sum(~support_validation)),
            "delta_r2": float(r2_score(y[validation], fold_candidate) - r2_score(y[validation], parent_oof[validation])),
        })
    candidate = parent_oof + RESIDUAL_WEIGHT * residual_oof
    delta = float(r2_score(y, candidate) - r2_score(y, parent_oof))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    lower = float(carrier.bootstrap_lower(y, parent_oof, candidate, groups))
    present_delta = stratum_delta(y, parent_oof, candidate, support_oof)
    missing_delta = stratum_delta(y, parent_oof, candidate, ~support_oof)
    evaluable = [value for value in (present_delta, missing_delta) if value is not None]
    minimum_stratum_delta = float(min(evaluable)) if evaluable else 0.0
    panel_report = transfer_panels(info, parent["fingerprints"], support_oof, candidate)
    passed = bool(
        delta >= MIN_BANKABLE_DELTA_R2
        and positive >= 4
        and lower > 0.0
        and minimum_stratum_delta >= 0.0
        and panel_report["minimum_transfer_panel_delta"] >= 0.0
    )
    report = {
        "active": True,
        "parent_r2": float(r2_score(y, parent_oof)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "partner_present_delta_r2": present_delta,
        "partner_missing_delta_r2": missing_delta,
        "minimum_stratum_delta": minimum_stratum_delta,
        "minimum_transfer_panel_delta": float(panel_report["minimum_transfer_panel_delta"]),
        "panels": panel_report["panels"],
        "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
        "partner_present_rows": int(np.sum(support_oof)),
        "partner_missing_rows": int(np.sum(~support_oof)),
        "oof_support_availability": "outer_validation_exact_canonical_and_no_stereo_groups_excluded",
        "folds": fold_rows,
        "pass": passed,
    }
    return candidate, report


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

    density, feature_report = build_pi1m_density_features(parent, data_dir / "PI1M.csv")
    available = availability_sets(parent)
    all_support = {
        target: support_mask(np.asarray(parent["target_info"][target]["canonical"], dtype=object), target, available)
        for target in ACTIVE_TARGETS
    }
    # Do not add global target-availability indicators as model features.
    # Some indicators would encode the held-out active target's own label
    # availability during OOF.  Support is used only to select the fixed
    # predeclared stratum, and every residual model sees PI1M density features.
    x_all = density.astype(np.float64)

    target_reports: dict[str, Any] = {}
    candidate_oof: dict[str, np.ndarray] = {}
    candidate_test: dict[str, np.ndarray] = {}
    oof_parts: list[pd.DataFrame] = []
    banked: list[str] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        y = np.asarray(info["y"], dtype=np.float64)
        parent_oof = np.asarray(info["parent"], dtype=np.float64)
        if target in ACTIVE_TARGETS:
            candidate, report = target_candidate(x_all, info, target, parent)
            target_reports[target] = report
            candidate_oof[target] = candidate
            if report["pass"]:
                banked.append(target)
        else:
            target_reports[target] = {
                "active": False,
                "parent_r2": float(r2_score(y, parent_oof)),
                "candidate_r2": float(r2_score(y, parent_oof)),
                "delta_r2": 0.0,
                "pass": False,
            }
            candidate_oof[target] = parent_oof
        assembled = candidate_oof[target] if target in banked else parent_oof
        oof_parts.append(pd.DataFrame({
            "target": y,
            "parent": parent_oof,
            "candidate": candidate_oof[target],
            "assembled": assembled,
            "target_type": target,
        }))

    for target in TARGETS:
        test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
            raise RuntimeError(f"C192 {target} test ID alignment failed")
        parent_test = test_detail["target"].to_numpy(np.float64)
        if target not in banked:
            candidate_test[target] = parent_test
            continue
        info = dict(parent["target_info"][target])
        indices = np.asarray(info["indices"], dtype=np.int64)
        y = np.asarray(info["y"], dtype=np.float64)
        parent_oof = np.asarray(info["parent"], dtype=np.float64)
        train_support = all_support[target]
        test_support = support_mask(test_rows["canonical"].to_numpy(object), target, available)
        test_indices = np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64)
        residual = np.zeros(len(test_rows), dtype=np.float64)
        for stratum_value in (False, True):
            train_rows = np.flatnonzero(train_support == stratum_value)
            predict_rows = np.flatnonzero(test_support == stratum_value)
            if len(train_rows) < 12 or len(predict_rows) == 0:
                continue
            model = fit_model()
            model.fit(x_all[indices[train_rows]], y[train_rows] - parent_oof[train_rows])
            residual[predict_rows] = model.predict(x_all[test_indices[predict_rows]])
        candidate_test[target] = parent_test + RESIDUAL_WEIGHT * residual

    parent_mean = float(np.mean([r2_score(part["target"], part["parent"]) for part in oof_parts]))
    candidate_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    parts: list[pd.DataFrame] = []
    for target in TARGETS:
        frame = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        parts.append(pd.DataFrame({
            "id": frame["id"].astype(int),
            "target_type": target,
            "target": candidate_test[target],
        }))
    predictions = pd.concat(parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)):
        raise RuntimeError("C192 ID coverage/order contract failed")
    if not np.isfinite(predictions["target"].to_numpy(np.float64)).all():
        raise RuntimeError("C192 produced non-finite predictions")

    report = {
        "schema_version": "ppp.round2.c192.pi1m-support-conditioned-residual.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "pi1m_used": True,
        "pi1m_labels_used": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent_replay_parity": parity,
        "feature_report": feature_report,
        "support_indicator_use": "stratum_selection_only_not_model_features",
        "oof_support_availability": "outer_validation_exact_canonical_and_no_stereo_groups_excluded",
        "test_support_availability": "full_official_train_archive_availability",
        "active_targets": list(ACTIVE_TARGETS),
        "target_reports": target_reports,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": bool(banked and candidate_mean - parent_mean >= 0.002),
        "decision": "candidate_pass_pending_clean_reproduction" if banked and candidate_mean - parent_mean >= 0.002 else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"),
        },
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions[["id", "target"]].to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.c192.pi1m-support-conditioned-residual.v1",
        "seed": SEED,
        "morgan_bits": MORGAN_BITS,
        "pi1m_sample": PI1M_SAMPLE,
        "residual_weight": RESIDUAL_WEIGHT,
        "ridge_alpha": RIDGE_ALPHA,
        "minimum_bankable_delta_r2": MIN_BANKABLE_DELTA_R2,
        "active_targets": list(ACTIVE_TARGETS),
        "routing": PARTNERS,
        "local_eval_read": False,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
        f"rdkit={Chem.rdBase.rdkitVersion}",
        f"platform={platform.platform()}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: {report['decision']}. Banked targets: {banked or 'none'}. Mean parent {parent_mean:.12f}; assembled {candidate_mean:.12f}; gain {candidate_mean - parent_mean:+.12f}. Official-only; PI1M used only as unlabeled density features; no local_eval read.\n",
        encoding="utf-8",
    )
    manifest = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiment_id": run_dir.name,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "decision": report["decision"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
