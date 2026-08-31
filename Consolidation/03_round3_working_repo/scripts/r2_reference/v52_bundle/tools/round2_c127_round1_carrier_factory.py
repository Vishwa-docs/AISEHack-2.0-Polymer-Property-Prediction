#!/usr/bin/env python3
"""C127: direct Round-1-inspired target-specific carrier factory.

This is a clean official-only experiment.  It rebuilds the C050 parent from
source, audits source parity against the canonical C050 artifacts, and then
fits direct structure-only Ridge and ExtraTrees arms on the portable Round 1
feature families.  A target is eligible for the compound only when its fresh
OOF result passes the fixed component gates.
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
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_eea_cross_target_oof_residual_stack as plumbing


TARGETS = tuple(reference.TARGETS)
SEED = 2026
N_FOLDS = 5
RIDGE_ALPHA = 30.0
TREE_ESTIMATORS = 160
TREE_LEAF = 2
DIRECT_BLOCKS = (
    "maccs_bit",
    "morgan_count_r1",
    "morgan_count_r2",
    "morgan_count_r3",
    "morgan_count_r4",
    "morgan_count_r5",
    "morgan_bit_r2",
    "atom_pair_count",
    "topological_torsion_count",
    "char_text",
    "periodic_morgan_count_r2",
    "periodic_morgan_count_r3",
    "capped_morgan_count_r2",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    folds = np.full(len(groups), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=N_FOLDS)
    for fold, (_, validation) in enumerate(splitter.split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def source_parity(root: Path, parent: dict[str, Any], canonical_run: Path) -> dict[str, Any]:
    """Audit the rebuilt parent without using canonical values as model inputs."""
    canonical_oof = pd.read_csv(canonical_run / "oof_predictions.csv")
    canonical_predictions = pd.read_csv(canonical_run / "predictions.csv")
    oof_deltas: list[float] = []
    for target in TARGETS:
        info = parent["target_info"][target]
        expected = canonical_oof[canonical_oof["target_type"] == target].set_index("canonical")["candidate_prediction"]
        replay = pd.Series(info["parent"], index=pd.Index(info["canonical"], name="canonical"))
        joined = pd.concat([replay.rename("replay"), expected.rename("canonical")], axis=1, join="inner").dropna()
        if len(joined) != len(replay):
            raise RuntimeError(f"C050 OOF identity mismatch for {target}")
        oof_deltas.extend(np.abs(joined["replay"].to_numpy(float) - joined["canonical"].to_numpy(float)).tolist())
    replay_test = parent["test_parent_detail"]["target"].to_numpy(float)
    canonical_test = canonical_predictions["target"].to_numpy(float)
    if len(replay_test) != len(canonical_test):
        raise RuntimeError("C050 test row count mismatch")
    test_max = float(np.max(np.abs(replay_test - canonical_test)))
    oof_max = float(np.max(oof_deltas)) if oof_deltas else float("inf")
    return {
        "oof_rows": int(len(oof_deltas)),
        "test_rows": int(len(replay_test)),
        "oof_max_abs": oof_max,
        "test_max_abs": test_max,
        "tolerance": 1.0e-12,
        "pass": bool(oof_max <= 1.0e-12 and test_max <= 1.0e-12),
    }


def build_round1_features(root: Path, smiles: list[str]) -> tuple[np.ndarray, sparse.csr_matrix, dict[str, Any]]:
    round1_dir = root / "Polymer Prediction Challenge" / "tools"
    sys.path.insert(0, str(round1_dir))
    import polymer_official_train_eval_loop as round1

    built = round1.build_features(
        smiles,
        n_bits=2048,
        text_features=32768,
        motif_hash_features=0,
        rich_features=True,
        periodic_features=True,
        periodic_dense_features=True,
        capped_dense_features=True,
        motif_features=True,
        physics_features=True,
        mordred_features=False,
        oligomer_features=True,
        oligomer_repeats=2,
        oligomer_slope_features=False,
        oligomer_ffox_features=False,
        oligomer_3d_features=False,
        rdkit_3d_features=False,
        backbone_sidechain_features=True,
        conjugation_features=True,
        mobility_features=True,
        huckel_features=False,
        electronic_tail_features=True,
        topological_autocorr_features=False,
        infinite_chain_features=True,
        bicerano_features=False,
        map4_features=True,
        map4_hash_features=16384,
        map4_max_distance=10,
        map4_env_radius=1,
        region_sparse_features=False,
        endpoint_path_sparse_features=True,
        endpoint_path_hash_features=16384,
        endpoint_path_max_bonds=8,
        rooted_smiles_features=True,
        rooted_smiles_max_roots=8,
        rooted_smiles_text_features=16384,
        random_smiles_features=False,
        kekule_smiles_features=True,
        kekule_smiles_text_features=16384,
        exact_sparse_features=False,
        wl_sparse_features=False,
    )
    dense = np.asarray(built["dense"], dtype=np.float64)
    blocks = [built["blocks"][name] for name in DIRECT_BLOCKS if name in built["blocks"]]
    if not blocks:
        raise RuntimeError("C127 produced no sparse feature blocks")
    sparse_features = sparse.hstack(blocks, format="csr").astype(np.float64)
    report = {
        "dense_shape": [int(value) for value in dense.shape],
        "sparse_shape": [int(value) for value in sparse_features.shape],
        "sparse_nnz": int(sparse_features.nnz),
        "selected_blocks": [name for name in DIRECT_BLOCKS if name in built["blocks"]],
        "feature_reports": built["feature_reports"],
        "round1_source": str(round1_dir / "polymer_official_train_eval_loop.py"),
    }
    return dense, sparse_features, report


def dense_pair(dense: np.ndarray, train_rows: np.ndarray, prediction_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(dense, dtype=np.float64).copy()
    clean[~np.isfinite(clean) | (np.abs(clean) > 1.0e12)] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(clean[train_rows]))
    prediction = scaler.transform(imputer.transform(clean[prediction_rows]))
    return train, prediction


def panel_delta(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, selected: np.ndarray, minimum: int = 10) -> float | None:
    if int(np.sum(selected)) < minimum or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in selected_groups])
        if float(np.var(y[rows])) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def fit_target(
    info: dict[str, Any],
    dense: np.ndarray,
    sparse_features: sparse.csr_matrix,
    test_indices: np.ndarray,
    test_parent: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    indices = np.asarray(info["indices"], dtype=np.int64)
    groups = np.asarray(info["groups"], dtype=object)
    folds = grouped_folds(groups)
    direct_oof = np.full((len(y), 2), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_dense, validation_dense = dense_pair(dense, indices[training], indices[validation])
        ridge = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
        ridge.fit(sparse.hstack([sparse_features[indices[training]], sparse.csr_matrix(train_dense)], format="csr"), y[training])
        direct_oof[validation, 0] = ridge.predict(sparse.hstack([sparse_features[indices[validation]], sparse.csr_matrix(validation_dense)], format="csr"))
        tree = ExtraTreesRegressor(
            n_estimators=TREE_ESTIMATORS,
            min_samples_leaf=TREE_LEAF,
            max_features=0.65,
            random_state=SEED,
            n_jobs=2,
        )
        tree.fit(train_dense, y[training])
        direct_oof[validation, 1] = tree.predict(validation_dense)
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "parent_r2": float(r2_score(y[validation], parent[validation])),
            "ridge_r2": float(r2_score(y[validation], direct_oof[validation, 0])),
            "tree_r2": float(r2_score(y[validation], direct_oof[validation, 1])),
        })
    arms = np.column_stack([parent, direct_oof])
    weights, intercept, blend_name, blend_r2 = reference.blend_from_oof(y, arms)
    candidate = arms @ weights + intercept
    full_dense, test_dense = dense_pair(dense, indices, test_indices)
    full_ridge = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", max_iter=5000, tol=1.0e-4)
    full_ridge.fit(sparse.hstack([sparse_features[indices], sparse.csr_matrix(full_dense)], format="csr"), y)
    test_ridge = full_ridge.predict(sparse.hstack([sparse_features[test_indices], sparse.csr_matrix(test_dense)], format="csr"))
    full_tree = ExtraTreesRegressor(
        n_estimators=TREE_ESTIMATORS,
        min_samples_leaf=TREE_LEAF,
        max_features=0.65,
        random_state=SEED,
        n_jobs=2,
    )
    full_tree.fit(full_dense, y)
    if test_parent is None:
        test_parent = np.zeros(len(test_indices), dtype=np.float64)
    else:
        test_parent = np.asarray(test_parent, dtype=np.float64)
        if len(test_parent) != len(test_indices):
            raise RuntimeError("test_parent length must match test_indices")
    test_arms = np.column_stack([test_parent, test_ridge, full_tree.predict(test_dense)])
    return {
        "candidate": candidate,
        "test_direct": test_arms @ weights + intercept,
        "folds": fold_rows,
        "weights": weights,
        "intercept": float(intercept),
        "blend_name": blend_name,
        "blend_r2": float(blend_r2),
        "direct_oof": direct_oof,
    }


def evaluate_target(info: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    candidate = np.asarray(result["candidate"], dtype=float)
    groups = np.asarray(info["groups"], dtype=object)
    scaffolds = np.asarray(info["scaffolds"], dtype=object)
    folds = grouped_folds(groups)
    nearest = np.full(len(y), np.nan, dtype=float)
    # Similarity is computed only against the outer training portion.
    for fold in range(N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        train_fps = [info["fingerprints"][int(info["indices"][row])] for row in training]
        for row in validation:
            nearest[row] = max(reference.DataStructs.BulkTanimotoSimilarity(info["fingerprints"][int(info["indices"][row])], train_fps))
    panel_specs: dict[str, np.ndarray] = {
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
    panel_values: list[float] = []
    for name, selected in panel_specs.items():
        delta = panel_delta(y, parent, candidate, selected)
        panels[name] = {"rows": int(np.sum(selected)), "delta_r2": delta, "status": "evaluable" if delta is not None else "inapplicable"}
        if delta is not None:
            panel_values.append(delta)
    fold_rows = []
    for fold in range(N_FOLDS):
        selected = folds == fold
        fold_rows.append({
            "fold": fold,
            "rows": int(np.sum(selected)),
            "parent_r2": float(r2_score(y[selected], parent[selected])),
            "candidate_r2": float(r2_score(y[selected], candidate[selected])),
            "delta_r2": float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected])),
        })
    delta = float(r2_score(y, candidate) - r2_score(y, parent))
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    lower = bootstrap_lower(y, parent, candidate, groups)
    minimum_panel = min(panel_values) if panel_values else 0.0
    passed = bool(delta >= 0.01 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0)
    return {
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_panel_delta": minimum_panel,
        "panels": panels,
        "folds": fold_rows,
        "pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", default="experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7")
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
    canonical_run = (root / args.canonical_run).resolve()
    parity = source_parity(root, parent, canonical_run)
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    dense, sparse_features, feature_report = build_round1_features(root, parent["keys"])
    target_reports: dict[str, Any] = {}
    result_by_target: dict[str, dict[str, Any]] = {}
    oof_parts: list[pd.DataFrame] = []
    direct_test_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = dict(parent["target_info"][target])
        info["fingerprints"] = parent["fingerprints"]
        test_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_detail = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        if not np.array_equal(test_rows["id"].to_numpy(int), test_detail["id"].to_numpy(int)):
            raise RuntimeError(f"C127 test ID alignment failed for {target}")
        result = fit_target(
            info,
            dense,
            sparse_features,
            np.asarray([parent["key_to_index"][value] for value in test_rows["canonical"]], dtype=np.int64),
            test_detail["target"].to_numpy(float),
        )
        report = evaluate_target(info, result)
        report.update({"blend_name": result["blend_name"], "blend_weights": [float(value) for value in result["weights"]], "blend_intercept": result["intercept"], "feature_rows": int(len(info["y"]))})
        target_reports[target] = report
        result_by_target[target] = result
        oof_parts.append(pd.DataFrame({
            "canonical": info["canonical"],
            "target_type": target,
            "target": info["y"],
            "parent": info["parent"],
            "candidate": result["candidate"],
            "group": info["groups"],
            "scaffold": info["scaffolds"],
            "fold": grouped_folds(np.asarray(info["groups"], dtype=object)),
        }))
        direct_test_parts.append(pd.DataFrame({"id": test_rows["id"].astype(int), "target_type": target, "direct_candidate": result["test_direct"]}))
    banked = [target for target in TARGETS if target_reports[target]["pass"]]
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_oof = []
    for part in oof_parts:
        target = str(part["target_type"].iloc[0])
        part = part.copy()
        part["assembled"] = part["candidate"] if target in banked else part["parent"]
        assembled_oof.append(part)
    oof = pd.concat(assembled_oof, ignore_index=True)
    assembled_mean = float(np.mean([r2_score(part["target"], part["assembled"]) for part in assembled_oof]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    direct_test = pd.concat(direct_test_parts, ignore_index=True)
    predictions = parent_test.merge(direct_test, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(predictions["target_type"].isin(banked), predictions["direct_candidate"], predictions["target"])
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C127 complete output contract failed")
    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003 and len(banked) > 0)
    report = {
        "schema_version": "ppp.round2.c127.round1-carrier-factory.run.v1",
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
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(predictions)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_pass,
        "decision": "candidate_pass_pending_clean_reproduction" if full_pass else "rejected_component_or_full_gate",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "parent_builder": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"),
            "reference": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py"),
            "round1_feature_source": sha256_file(root / "Polymer Prediction Challenge/tools/polymer_official_train_eval_loop.py"),
        },
    }
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    direct_test.to_csv(run_dir / "component_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {
        "schema_version": "ppp.round2.c127.round1-carrier-factory.v1",
        "seed": SEED,
        "folds": "grouped no-stereo for direct arms; exact C050 source replay as fallback",
        "direct_blocks": list(DIRECT_BLOCKS),
        "ridge_alpha": RIDGE_ALPHA,
        "extra_trees": {"n_estimators": TREE_ESTIMATORS, "min_samples_leaf": TREE_LEAF, "max_features": 0.65},
        "banking": "target-wise component gate before compound assembly",
        "local_eval_read": False,
    })
    (run_dir / "environment.txt").write_text("\n".join([
        f"python={platform.python_version()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
        f"rdkit={reference.Chem.rdBase.rdkitVersion}",
        f"platform={platform.platform()}",
    ]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{parent_mean:.12f}`; assembled `{assembled_mean:.12f}`; gain `{assembled_mean - parent_mean:+.12f}`. No local_eval read.\n",
        encoding="utf-8",
    )
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "banked_targets": banked, "mean_parent_r2": parent_mean, "mean_candidate_r2": assembled_mean, "mean_gain": assembled_mean - parent_mean, "decision": report["decision"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
