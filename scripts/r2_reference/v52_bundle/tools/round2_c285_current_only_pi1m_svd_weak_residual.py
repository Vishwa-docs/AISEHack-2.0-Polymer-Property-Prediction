#!/usr/bin/env python3
"""C285 current-only PI1M-SVD weak-target residual.

This is a bounded no-archive PI1M experiment. It rebuilds the C282-style
current-only parent from official train/test, learns a label-free character
n-gram SVD representation from official PI1M, and fits residual heads only for
ei/nc/eps. LocalEval files, archive labels, prior predictions, and pretrained
weights are not read.
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
from rdkit import Chem
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c282_current_only_reference as c282


TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("ei", "nc", "eps")
DEFAULT_CONFIG: dict[str, Any] = {
    **reference.DEFAULT_CONFIG,
    "pi1m_limit": 100_000,
    "pi1m_hash_features": 32_768,
    "pi1m_svd_components": 96,
    "residual_weight": 0.50,
    "ridge_alpha": 20.0,
    "extra_trees_estimators": 400,
    "extra_trees_min_leaf": 4,
    "gate_min_oof_delta": 0.005,
    "gate_min_positive_folds": 3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def hash_ranked_pi1m(path: Path, limit: int) -> list[str]:
    frame = pd.read_csv(path, usecols=["SMILES"])
    values = frame["SMILES"].dropna().astype(str).tolist()
    unique = sorted(
        set(values),
        key=lambda value: hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
    )
    return unique[:limit]


def pi1m_char_svd(keys: list[str], pi1m_path: Path, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    corpus = hash_ranked_pi1m(pi1m_path, int(config["pi1m_limit"]))
    all_smiles = corpus + list(keys)
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 7),
        n_features=int(config["pi1m_hash_features"]),
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float64,
    )
    hashed = vectorizer.transform(all_smiles).tocsr()
    n_components = min(int(config["pi1m_svd_components"]), hashed.shape[0] - 1, hashed.shape[1] - 1)
    if n_components < 8:
        raise RuntimeError("Too few rows/features for PI1M SVD")
    svd = TruncatedSVD(n_components=n_components, random_state=int(config["seed"]))
    features = svd.fit_transform(hashed)[-len(keys) :].astype(np.float64, copy=False)
    return features, {
        "pi1m_path": str(pi1m_path),
        "pi1m_sha256": sha256_file(pi1m_path),
        "pi1m_hash_ranked_unique_rows_used": int(len(corpus)),
        "hash_features": int(config["pi1m_hash_features"]),
        "svd_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "labels_used": False,
    }


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    result = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        result[validation] = fold
    if np.any(result < 0):
        raise RuntimeError("Failed to assign all grouped folds")
    return result


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    delta_by_group = {}
    for group in unique:
        rows = groups == group
        # R2 is not additive; this grouped squared-error delta is a conservative
        # stability proxy for the residual direction.
        delta_by_group[group] = (y[rows] - parent[rows]) ** 2 - (y[rows] - candidate[rows]) ** 2
    rng = np.random.default_rng(20260807)
    draws = np.empty(1000, dtype=np.float64)
    for i in range(len(draws)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        draws[i] = float(np.mean(np.concatenate([delta_by_group[group] for group in selected])))
    return float(np.quantile(draws, 0.025))


def target_model(target: str, config: dict[str, Any], seed: int):
    if target == "ei":
        return Ridge(alpha=float(config["ridge_alpha"]))
    return ExtraTreesRegressor(
        n_estimators=int(config["extra_trees_estimators"]),
        min_samples_leaf=int(config["extra_trees_min_leaf"]),
        max_features=0.8,
        random_state=seed,
        n_jobs=4,
    )


def build_current_only_parent(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, int], np.ndarray, dict[str, Any], dict[str, Any]]:
    archive = train.iloc[0:0].copy()
    raw_labels, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, descriptor_names = reference.descriptor_matrix(molecules)
    physical, physical_names = reference.physical_matrix(molecules, keys)
    dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=int(config["morgan_bits"])),
        reference.morgan_count_matrix(molecules, radius=3, bits=int(config["morgan_bits"])),
        reference.text_matrix(keys, int(config["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=int(config["morgan_bits"]))
    detail, oof, model_report = reference.fit_targets(
        pooled,
        test,
        keys,
        dense_base,
        cross_values,
        cross_available,
        sparse_parts,
        fingerprints,
        config,
    )
    final_detail, override_report = reference.apply_official_overrides(detail, test, raw_labels)
    feature_report = {
        "rdkit_descriptors": int(len(descriptor_names)),
        "physical_features": int(len(physical_names)),
        "morgan_radii": [2, 3],
        "morgan_bits": int(config["morgan_bits"]),
        "text_features": int(config["text_features"]),
    }
    return raw_labels, pooled, final_detail, oof, keys, key_to_index, dense_base, model_report, {"official_overrides": override_report, "features": feature_report}


def run_experiment(data_dir: Path, run_dir: Path, output_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse existing run directory: {run_dir}")
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite output: {output_path}")
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train, test, inputs = c282.load_current_only_inputs(data_dir)
    pi1m_path = data_dir / "PI1M.csv"
    if sha256_file(pi1m_path) != "c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8":
        raise RuntimeError("Official PI1M hash mismatch")
    protocol = {
        "schema_version": "ppp.round2.c285.current-only-pi1m-svd-weak-residual.protocol.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "hypothesis": "A from-scratch PI1M character-SVD residual can improve weak no-archive targets ei/nc/eps over the C282 current-only parent without touching stronger targets.",
        "changed_factor": "target-limited residual heads over label-free PI1M SVD features",
        "active_targets": list(ACTIVE_TARGETS),
        "official_only": True,
        "archive_labels_used": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "config": config,
    }
    write_json(run_dir / "protocol.json", protocol)
    raw_labels, pooled, final_detail, oof, keys, key_to_index, _, parent_report, aux_report = build_current_only_parent(train, test, config)
    pi1m_features, pi1m_report = pi1m_char_svd(keys, pi1m_path, config)
    scaler = StandardScaler()
    X = scaler.fit_transform(pi1m_features)
    parent_submission = final_detail[["id", "target_type", "target", "override", "model_prediction"]].copy()
    candidate_submission = parent_submission.copy()
    target_reports: dict[str, Any] = {}
    oof_records: list[pd.DataFrame] = []
    component_records: list[pd.DataFrame] = []
    for target in TARGETS:
        target_oof = oof[oof["target_type"] == target].reset_index(drop=True)
        y = target_oof["target"].to_numpy(float)
        parent_pred = target_oof["prediction"].to_numpy(float)
        canonical = target_oof["canonical"].astype(str).to_numpy()
        indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
        if target not in ACTIVE_TARGETS:
            target_reports[target] = {
                "active": False,
                "parent_oof_r2": float(r2_score(y, parent_pred)),
                "selected": False,
            }
            oof_records.append(pd.DataFrame({"target_type": target, "target": y, "parent": parent_pred, "candidate": parent_pred, "selected": False}))
            continue
        folds = grouped_folds(canonical)
        residual_oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_rows: list[dict[str, Any]] = []
        for fold in range(5):
            validation = np.flatnonzero(folds == fold)
            training = np.flatnonzero(folds != fold)
            model = target_model(target, config, seed=int(config["seed"]) + 17 * fold + TARGETS.index(target))
            model.fit(X[indices[training]], y[training] - parent_pred[training])
            residual_oof[validation] = model.predict(X[indices[validation]])
            cand_fold = parent_pred[validation] + float(config["residual_weight"]) * residual_oof[validation]
            parent_r2 = float(r2_score(y[validation], parent_pred[validation]))
            cand_r2 = float(r2_score(y[validation], cand_fold))
            fold_rows.append({"fold": int(fold), "rows": int(len(validation)), "parent_r2": parent_r2, "candidate_r2": cand_r2, "delta_r2": cand_r2 - parent_r2})
        candidate_oof = parent_pred + float(config["residual_weight"]) * residual_oof
        parent_r2 = float(r2_score(y, parent_pred))
        candidate_r2 = float(r2_score(y, candidate_oof))
        delta = candidate_r2 - parent_r2
        positive_folds = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        lower = bootstrap_lower(y, parent_pred, candidate_oof, canonical)
        selected = bool(
            delta >= float(config["gate_min_oof_delta"])
            and positive_folds >= int(config["gate_min_positive_folds"])
            and lower > 0.0
        )
        target_reports[target] = {
            "active": True,
            "parent_oof_r2": parent_r2,
            "candidate_oof_r2": candidate_r2,
            "delta_oof_r2": delta,
            "positive_folds": positive_folds,
            "grouped_error_delta_bootstrap_lower": lower,
            "selected": selected,
            "folds": fold_rows,
        }
        oof_records.append(pd.DataFrame({"target_type": target, "target": y, "parent": parent_pred, "residual_oof": residual_oof, "candidate": candidate_oof, "selected": selected, "fold": folds, "canonical": canonical}))
        test_rows = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_indices = np.asarray([key_to_index[value] for value in test_rows["canonical"]], dtype=np.int64)
        full_model = target_model(target, config, seed=int(config["seed"]) + 101 + TARGETS.index(target))
        full_model.fit(X[indices], y - parent_pred)
        test_residual = full_model.predict(X[test_indices])
        ids = test_rows["id"].astype(int).to_numpy()
        pred_rows = candidate_submission["id"].isin(ids)
        base_values = candidate_submission.loc[pred_rows, "target"].to_numpy(float)
        override_values = candidate_submission.loc[pred_rows, "override"].astype(str).to_numpy()
        adjusted = base_values + float(config["residual_weight"]) * test_residual
        final_values = np.where(override_values == "model", adjusted, base_values)
        if selected:
            candidate_submission.loc[pred_rows, "target"] = final_values
        component_records.append(pd.DataFrame({"id": ids, "target_type": target, "parent": base_values, "residual": test_residual, "candidate": final_values, "selected": selected, "override": override_values}))
    output = candidate_submission[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(output) != 4940 or not np.array_equal(output["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(output["target"].to_numpy(float)).all():
        raise RuntimeError("Candidate output contract failed")
    output.to_csv(output_path, index=False)
    selected_targets = [target for target in ACTIVE_TARGETS if target_reports[target]["selected"]]
    report = {
        "schema_version": "ppp.round2.c285.current-only-pi1m-svd-weak-residual.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "official_only": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "inputs": inputs | {"PI1M.csv": {"path": str(pi1m_path), "sha256": sha256_file(pi1m_path), "bytes": pi1m_path.stat().st_size}},
        "parent": {
            "method": "rebuilt C282 current-only reference",
            "validation": parent_report,
            **aux_report,
        },
        "pi1m_representation": pi1m_report,
        "target_reports": target_reports,
        "selected_targets": selected_targets,
        "submission": {"path": str(output_path), "sha256": sha256_file(output_path), "rows": int(len(output))},
        "decision": "candidate_complete_selected_targets_pending_local_eval_score" if selected_targets else "rejected_no_clean_oof_gate_target",
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", config)
    pd.concat(oof_records, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    pd.concat(component_records, ignore_index=True).to_csv(run_dir / "component_predictions.csv", index=False)
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"sklearn={__import__('sklearn').__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Selected targets: `{','.join(selected_targets) or 'none'}`. Official current-only plus label-free PI1M SVD; no archive labels, local_eval, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    manifest.append(f"{sha256_file(Path(__file__))}  SOURCE tools/round2_c285_current_only_pi1m_svd_weak_residual.py")
    manifest.append(f"{sha256_file(Path(reference.__file__))}  SOURCE tools/initial_reference_pipeline.py")
    manifest.append(f"{sha256_file(Path(c282.__file__))}  SOURCE tools/round2_c282_current_only_reference.py")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pi1m-limit", type=int, default=DEFAULT_CONFIG["pi1m_limit"])
    parser.add_argument("--pi1m-components", type=int, default=DEFAULT_CONFIG["pi1m_svd_components"])
    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)
    config["pi1m_limit"] = int(args.pi1m_limit)
    config["pi1m_svd_components"] = int(args.pi1m_components)
    report = run_experiment(
        data_dir=Path(args.data_dir).resolve(),
        run_dir=Path(args.run_dir).resolve(),
        output_path=Path(args.output).resolve(),
        config=config,
    )
    print(json.dumps({"experiment_id": report["experiment_id"], "selected_targets": report["selected_targets"], "decision": report["decision"], "submission": report["submission"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
