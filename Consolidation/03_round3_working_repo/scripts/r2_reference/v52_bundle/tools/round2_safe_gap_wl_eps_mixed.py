#!/usr/bin/env python3
"""Strictly nested official-only guarded gap plus WL-EPS component experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = ("ei", "eea", "eps")
GAP_TARGETS = {"ei", "eea"}
GAP_ALPHA = 10.0
WL_STEPS = 3
SIMILARITY_BARRIER = 0.70


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def no_stereo(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("official SMILES failed RDKit parsing")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "INVALID"
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    except Exception:
        value = ""
    return value or "ACYCLIC"


def folds_for(groups: np.ndarray, n_splits: int) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    if len(np.unique(groups)) < n_splits:
        raise RuntimeError(f"need {n_splits} groups, found {len(np.unique(groups))}")
    folds = np.full(len(groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups)):
        folds[validation] = fold
    return folds


def bootstrap_r2_lower(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(500):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in chosen])
        if len(rows) > 1 and np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], baseline[rows])))
    return float(np.quantile(values, 0.025)) if values else float("-inf")


def nearest_similarity(fingerprints: list[object], validation: np.ndarray, training: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[index] for index in training]
    if not train_fps:
        return np.full(len(validation), np.nan, dtype=np.float64)
    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps)) for index in validation],
        dtype=np.float64,
    )


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def graph_tokens(molecule: Chem.Mol, steps: int = WL_STEPS) -> list[str]:
    labels = []
    for atom in molecule.GetAtoms():
        labels.append(
            "a:" + ":".join([
                str(atom.GetAtomicNum()),
                str(int(atom.GetIsAromatic())),
                str(atom.GetTotalDegree()),
                str(atom.GetFormalCharge()),
                str(int(atom.IsInRing())),
            ])
        )
    tokens: list[str] = []
    for iteration in range(steps + 1):
        tokens.extend(f"wl{iteration}:{label}" for label in labels)
        if iteration == steps:
            break
        next_labels = []
        for atom in molecule.GetAtoms():
            neighbors = []
            for neighbor in atom.GetNeighbors():
                bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                neighbors.append(f"{bond.GetBondTypeAsDouble()}:{labels[neighbor.GetIdx()]}")
            payload = labels[atom.GetIdx()] + "|" + "|".join(sorted(neighbors))
            next_labels.append("h:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20])
        labels = next_labels
    return tokens


def fit_wl_model(tokens: list[list[str]], y: np.ndarray, train_rows: np.ndarray, prediction_rows: np.ndarray):
    if len(train_rows) < 8:
        return None, np.full(len(prediction_rows), np.nan, dtype=np.float64), 0
    vectorizer = CountVectorizer(analyzer=lambda value: value, lowercase=False, token_pattern=None, dtype=np.float64)
    train_matrix = vectorizer.fit_transform([tokens[index] for index in train_rows])
    prediction_matrix = vectorizer.transform([tokens[index] for index in prediction_rows])
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=10.0, solver="lsqr", max_iter=5000))
    model.fit(train_matrix, y[train_rows])
    return model, np.asarray(model.predict(prediction_matrix), dtype=np.float64), int(len(vectorizer.vocabulary_))


def make_gap_table(pooled: pd.DataFrame) -> pd.DataFrame:
    pivot = pooled.pivot(index="canonical", columns="target_type", values="target").reset_index()
    for target in ("ei", "eea", "egc"):
        if target not in pivot:
            pivot[target] = np.nan
    pivot["group"] = [no_stereo(value) for value in pivot["canonical"]]
    pivot["scaffold"] = [scaffold(value) for value in pivot["canonical"]]
    pivot["gap"] = pivot["ei"] - pivot["eea"]
    return pivot


def fit_gap_model(gap_table: pd.DataFrame, forbidden_groups: set[str], forbidden_scaffolds: set[str] | None = None):
    forbidden_scaffolds = forbidden_scaffolds or set()
    usable = (
        gap_table["ei"].notna()
        & gap_table["eea"].notna()
        & gap_table["egc"].notna()
        & ~gap_table["group"].isin(forbidden_groups)
        & ~gap_table["scaffold"].isin(forbidden_scaffolds)
    )
    if int(np.sum(usable)) < 8:
        return None, int(np.sum(usable))
    model = make_pipeline(StandardScaler(), Ridge(alpha=GAP_ALPHA))
    model.fit(gap_table.loc[usable, ["egc"]].to_numpy(float), gap_table.loc[usable, "gap"].to_numpy(float))
    return model, int(np.sum(usable))


def nested_parent(
    target: str,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    target_dense: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    global_indices: np.ndarray,
    y_global: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    inner_folds = folds_for(groups[outer_train], 4)
    inner_arms = np.full((len(outer_train), 4), np.nan, dtype=np.float64)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_arms[inner_folds == fold] = reference.predict_base_models(
            target_dense, sparse_parts, fingerprints, y_global,
            global_indices[local_train], global_indices[local_validation], config, target,
        )
    weights, intercept, blend_name, inner_r2 = reference.blend_from_oof(y[outer_train], inner_arms)
    outer_arms = reference.predict_base_models(
        target_dense, sparse_parts, fingerprints, y_global,
        global_indices[outer_train], global_indices[outer_validation], config, target,
    )
    parent = reference.clip_prediction(y[outer_train], outer_arms @ weights + intercept)
    return parent, {
        "blend_name": blend_name,
        "blend_weights": [float(value) for value in weights],
        "blend_intercept": float(intercept),
        "inner_parent_r2": float(inner_r2),
    }


def make_route(
    target: str,
    canonical: np.ndarray,
    parent: np.ndarray,
    y_train: np.ndarray,
    nearest: np.ndarray,
    gap_model,
    wl_prediction: np.ndarray,
    maps: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.asarray(parent, dtype=np.float64).copy()
    eligible_similarity = np.isfinite(nearest) & (nearest < SIMILARITY_BARRIER)
    partner = np.full(len(canonical), np.nan, dtype=np.float64)
    egc = np.asarray([maps["egc"].get(value, np.nan) for value in canonical], dtype=np.float64)
    predicted_gap = np.full(len(canonical), np.nan, dtype=np.float64)
    if target in GAP_TARGETS:
        partner_target = "eea" if target == "ei" else "ei"
        partner = np.asarray([maps[partner_target].get(value, np.nan) for value in canonical], dtype=np.float64)
        route = eligible_similarity & np.isfinite(partner) & np.isfinite(egc) & (gap_model is not None)
        if gap_model is not None and np.any(route):
            predicted_gap[route] = gap_model.predict(egc[route, None])
            raw = partner[route] + predicted_gap[route] if target == "ei" else partner[route] - predicted_gap[route]
            candidate[route] = reference.clip_prediction(y_train, raw)
    else:
        route = eligible_similarity & np.isfinite(wl_prediction)
        if np.any(route):
            candidate[route] = reference.clip_prediction(y_train, wl_prediction[route])
    return candidate, route, partner, predicted_gap


def build_panels(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    nearest: np.ndarray,
    scaffolds: np.ndarray,
    measurements: np.ndarray,
    routed: np.ndarray,
    missing_support: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, object], float | None, bool]:
    panels: dict[str, object] = {}
    values: list[float] = []

    def add(name: str, selected: np.ndarray, control: bool = False) -> None:
        delta = panel_delta(y, baseline, candidate, selected)
        rows = int(np.sum(selected))
        if rows < 5:
            status = "inapplicable_zero_support"
        elif delta is None:
            status = "incomplete_constant_support"
        elif control and float(np.max(np.abs(candidate[selected] - baseline[selected]))) > 1.0e-12:
            status = "failed_parent_only_control"
        else:
            status = "evaluable"
        panels[name] = {"rows": rows, "delta_r2": delta, "status": status}
        if delta is not None:
            values.append(delta)

    add("similarity_lt_0.30", nearest < 0.30)
    add("similarity_0.30_0.50", (nearest >= 0.30) & (nearest < 0.50))
    add("similarity_0.50_0.70", (nearest >= 0.50) & (nearest < 0.70))
    add("similarity_ge_0.70", nearest >= 0.70, control=True)
    add("exact_archive_measurements_ge_2", measurements >= 2)
    add("sparse_singleton_measurements_eq_1", measurements == 1)
    add("route_eligible", routed)
    add("missing_support_parent_only", missing_support, control=True)
    for scaffold_name in sorted(set(scaffolds)):
        selected = scaffolds == scaffold_name
        if int(np.sum(selected)) >= 10:
            add(f"scaffold_slice_{scaffold_name}", selected)
    incomplete = any(
        isinstance(value, dict) and value["status"] in {"incomplete_constant_support", "failed_parent_only_control"}
        for value in panels.values()
    )
    return panels, (min(values) if values else None), incomplete


def evaluate_target(
    target: str,
    pooled: pd.DataFrame,
    keys: list[str],
    key_to_index: dict[str, int],
    dense_base: np.ndarray,
    cross_values: np.ndarray,
    cross_available: np.ndarray,
    sparse_parts: list[object],
    fingerprints: list[object],
    tokens: list[list[str]],
    gap_table: pd.DataFrame,
    maps: dict[str, dict[str, float]],
    config: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    canonical = frame["canonical"].to_numpy(object)
    groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([scaffold(value) for value in canonical], dtype=object)
    measurements = frame["measurements"].to_numpy(int)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[global_indices] = y
    target_dense = reference.target_dense_features(dense_base, cross_values, cross_available, target)
    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan, dtype=np.float64)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    routed = np.zeros(len(y), dtype=bool)
    partner = np.full(len(y), np.nan, dtype=np.float64)
    predicted_gap = np.full(len(y), np.nan, dtype=np.float64)
    nearest = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        parent, parent_meta = nested_parent(
            target, y, groups, training, validation, target_dense, sparse_parts,
            fingerprints, global_indices, y_global, config,
        )
        nearest_validation = nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
        gap_model = None
        gap_count = None
        wl_prediction = np.full(len(validation), np.nan, dtype=np.float64)
        wl_vocab = None
        if target in GAP_TARGETS:
            gap_model, gap_count = fit_gap_model(gap_table, set(groups[validation]))
        else:
            _, wl_prediction, wl_vocab = fit_wl_model(tokens, y, training, validation)
        routed_candidate, route, route_partner, route_gap = make_route(
            target, canonical[validation], parent, y[training], nearest_validation,
            gap_model, wl_prediction, maps,
        )
        baseline[validation] = parent
        candidate[validation] = routed_candidate
        routed[validation] = route
        partner[validation] = route_partner
        predicted_gap[validation] = route_gap
        nearest[validation] = nearest_validation
        parent_score = float(r2_score(y[validation], parent))
        candidate_score = float(r2_score(y[validation], routed_candidate))
        fold_rows.append({
            "target": target,
            "fold": fold,
            "rows": int(len(validation)),
            "baseline_r2": parent_score,
            "candidate_r2": candidate_score,
            "delta_r2": candidate_score - parent_score,
            "routed_rows": int(np.sum(route)),
            "gap_training_groups": gap_count,
            "wl_vocabulary": wl_vocab,
            "outer_fold_validation_groups": sorted(set(groups[validation])),
            "parent_blend": parent_meta,
        })

    scaffold_holdout: dict[str, object] = {}
    for scaffold_name in sorted(set(scaffolds)):
        validation = np.flatnonzero(scaffolds == scaffold_name)
        if len(validation) < 10:
            continue
        training = np.flatnonzero(scaffolds != scaffold_name)
        parent_holdout, _ = nested_parent(
            target, y, groups, training, validation, target_dense, sparse_parts,
            fingerprints, global_indices, y_global, config,
        )
        nearest_holdout = nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
        gap_model = None
        gap_count = None
        wl_prediction = np.full(len(validation), np.nan, dtype=np.float64)
        wl_vocab = None
        if target in GAP_TARGETS:
            gap_model, gap_count = fit_gap_model(gap_table, set(groups[validation]), {str(scaffold_name)})
        else:
            _, wl_prediction, wl_vocab = fit_wl_model(tokens, y, training, validation)
        holdout_candidate, route, _, _ = make_route(
            target, canonical[validation], parent_holdout, y[training], nearest_holdout,
            gap_model, wl_prediction, maps,
        )
        base_score = float(r2_score(y[validation], parent_holdout))
        cand_score = float(r2_score(y[validation], holdout_candidate))
        scaffold_holdout[str(scaffold_name)] = {
            "rows": int(len(validation)),
            "baseline_r2": base_score,
            "candidate_r2": cand_score,
            "delta_r2": cand_score - base_score,
            "routed_rows": int(np.sum(route)),
            "gap_training_groups": gap_count,
            "wl_vocabulary": wl_vocab,
        }

    panels, min_panel, panel_incomplete = build_panels(
        y, baseline, candidate, nearest, scaffolds, measurements, routed, ~routed, groups,
    )
    baseline_score = float(r2_score(y, baseline))
    candidate_score = float(r2_score(y, candidate))
    delta = candidate_score - baseline_score
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    scaffold_min = min((float(value["delta_r2"]) for value in scaffold_holdout.values()), default=None)
    positive_folds = int(sum(float(row["delta_r2"]) > 0.0 for row in fold_rows))
    route = {
        "barrier": SIMILARITY_BARRIER,
        "routed_rows": int(np.sum(routed)),
        "parent_only_rows": int(np.sum(~routed)),
        "route_changed_rows": int(np.sum(np.abs(candidate - baseline) > 1.0e-12)),
        "missing_support_rows": int(np.sum(~routed)) if target == "eps" else int(np.sum(~(np.isfinite(partner) & np.isfinite(np.asarray([maps["egc"].get(value, np.nan) for value in canonical]))))),
        "missing_support_max_change": float(np.max(np.abs(candidate[~routed] - baseline[~routed]))) if np.any(~routed) else 0.0,
    }
    report: dict[str, object] = {
        "rows": int(len(y)),
        "canonical_groups": int(len(np.unique(groups))),
        "baseline_r2_nested_parent": baseline_score,
        "candidate_r2_guarded_route": candidate_score,
        "delta_r2": delta,
        "positive_outer_folds": positive_folds,
        "group_r2_bootstrap_lower": float(bootstrap),
        "outer_folds": fold_rows,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": scaffold_min,
        "min_panel_delta": min_panel,
        "panel_incomplete": panel_incomplete,
        "panels": panels,
        "route_definition": route,
        "support_counts": {
            "exact_archive_measurement_rows": int(np.sum(measurements >= 2)),
            "singleton_rows": int(np.sum(measurements == 1)),
            "paired_gap_groups": int(np.sum(gap_table["ei"].notna() & gap_table["eea"].notna() & gap_table["egc"].notna())),
        },
    }
    rows = pd.DataFrame({
        "canonical": canonical,
        "target_type": target,
        "target": y,
        "baseline": baseline,
        "candidate": candidate,
        "route": routed,
        "partner": partner,
        "predicted_gap": predicted_gap,
        "nearest_similarity": nearest,
        "scaffold": scaffolds,
        "no_stereo_group": groups,
        "measurements": measurements,
        "outer_fold": main_folds,
    })
    return report, rows, fold_rows, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError(f"a protocol-only run directory is required: {run_dir}")
    start_time = datetime.now().astimezone()
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
        reference.morgan_count_matrix(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.text_matrix(keys, int(reference.DEFAULT_CONFIG["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"]))
    tokens = [graph_tokens(molecule) for molecule in molecules]
    gap_table = make_gap_table(pooled)
    maps = {
        target: dict(zip(frame["canonical"], frame["target"].astype(float), strict=True))
        for target in ("ei", "eea", "egc")
        for frame in [pooled[pooled["target_type"] == target].reset_index(drop=True)]
    }
    config = dict(reference.DEFAULT_CONFIG)
    config.update({
        "seed": 2026,
        "folds": 5,
        "gap_alpha": GAP_ALPHA,
        "wl_steps": WL_STEPS,
        "similarity_barrier": SIMILARITY_BARRIER,
        "wl_representation": "fold-local hashed Weisfeiler-Lehman subtree-count tokens",
    })
    reports: dict[str, object] = {}
    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for target in TARGETS:
        report, rows, target_folds, _ = evaluate_target(
            target, pooled, keys, key_to_index, dense_base, cross_values, cross_available,
            sparse_parts, fingerprints, tokens, gap_table, maps, config,
        )
        reports[target] = report
        oof_parts.append(rows)
        fold_rows.extend(target_folds)

    preliminary_pass: dict[str, bool] = {}
    for target, report in reports.items():
        scaffold_min = report["scaffold_holdout_min_delta"]
        preliminary_pass[target] = bool(
            report["delta_r2"] >= 0.01
            and report["positive_outer_folds"] >= 4
            and report["group_r2_bootstrap_lower"] > 0.0
            and (report["min_panel_delta"] is None or report["min_panel_delta"] >= -0.003)
            and (scaffold_min is None or scaffold_min >= -0.003)
            and not report["panel_incomplete"]
            and report["route_definition"]["missing_support_max_change"] <= 1.0e-12
        )
    # The two reconstructed targets must not pass if their paired target loses more than 0.003.
    if reports["ei"]["delta_r2"] < -0.003:
        preliminary_pass["eea"] = False
    if reports["eea"]["delta_r2"] < -0.003:
        preliminary_pass["ei"] = False
    for target, report in reports.items():
        report["pass"] = preliminary_pass[target]
        report["decision"] = "component_pass" if preliminary_pass[target] else "rejected_component_gate"

    passing_targets = [target for target in TARGETS if preliminary_pass[target]]
    all_planned_pass = len(passing_targets) == len(TARGETS)
    source_hashes = {
        "script": sha256_file(Path(__file__).resolve()),
        "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py"),
    }
    config_hash = reference.canonical_json_hash(config)
    metrics = {
        "schema_version": "ppp.round2.safe-gap-wl-eps-mixed-run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "R2-C041-20260803-2330-ei-eea-gap-identity",
        "baseline_reference": "C001 official clean incumbent, freshly regenerated as nested parent per target",
        "lineage_parent": "R2-C041-20260803-2330-ei-eea-gap-identity",
        "official_inputs": inputs,
        "targets": reports,
        "passing_targets": passing_targets,
        "all_planned_targets_pass": all_planned_pass,
        "pass": bool(all_planned_pass),
        "decision": "mixed_components_pass" if all_planned_pass else ("partial_component_pass" if passing_targets else "rejected_component_gate"),
        "source_hashes": source_hashes,
        "config_sha256": config_hash,
        "elapsed_seconds": float(time.time() - started),
    }
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "config.json", config)
    (run_dir / "environment.txt").write_text(
        "\n".join([
            f"python={platform.python_version()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"rdkit={Chem.rdBase.rdkitVersion}",
            f"platform={platform.platform()}",
        ]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "decision.md").write_text(
        f"# {run_dir.name}\n\nDecision: **{metrics['decision']}**. Passing targets: `{', '.join(passing_targets) or 'none'}`. No candidate or local_eval diagnostic was created by this component run.\n",
        encoding="utf-8",
    )
    finish_time = datetime.now().astimezone()
    (run_dir / "run.log").write_text(
        f"experiment_id={run_dir.name}\n"
        f"started_at={start_time.isoformat()}\n"
        f"finished_at={finish_time.isoformat()}\n"
        f"decision={metrics['decision']}\n"
        f"passing_targets={','.join(passing_targets)}\n"
        f"elapsed_seconds={metrics['elapsed_seconds']:.3f}\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([
        f"{source_hashes['script']}  SOURCE tools/round2_safe_gap_wl_eps_mixed.py",
        f"{source_hashes['reference_module']}  SOURCE tools/initial_reference_pipeline.py",
    ])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": metrics["decision"], "passing_targets": passing_targets, "all_planned_targets_pass": all_planned_pass, "elapsed_seconds": metrics["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
