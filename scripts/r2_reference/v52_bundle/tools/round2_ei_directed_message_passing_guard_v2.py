#!/usr/bin/env python3
"""Strictly nested official-only Ei directed message-passing component."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
TARGET = "ei"
SIMILARITY_BARRIER = 0.70
HIDDEN = 64
STEPS = 3


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
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fps)) for index in validation], dtype=float)


def panel_delta(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, selected: np.ndarray) -> float | None:
    if int(np.sum(selected)) < 5 or float(np.var(y[selected])) <= 1.0e-15:
        return None
    return float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], baseline[selected]))


def atom_feature(atom: Chem.Atom) -> np.ndarray:
    value = np.zeros(80, dtype=np.float64)
    atomic_number = min(int(atom.GetAtomicNum()), 63)
    value[atomic_number] = 1.0
    value[64] = float(atom.GetIsAromatic())
    value[65 + min(atom.GetTotalDegree(), 5)] = 1.0
    value[71 + min(max(atom.GetFormalCharge(), -2) + 2, 0)] = 1.0
    value[76] = float(atom.IsInRing())
    value[77] = min(float(atom.GetTotalNumHs()), 4.0) / 4.0
    value[78] = float(atom.GetIsotope() > 0)
    value[79] = float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    return value


def bond_feature(bond: Chem.Bond, dimension: int) -> np.ndarray:
    value = np.zeros(dimension, dtype=np.float64)
    bond_type = float(bond.GetBondTypeAsDouble())
    order_index = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3}.get(bond_type, 4)
    value[order_index] = 1.0
    value[5] = float(bond.GetIsAromatic())
    value[6] = float(bond.IsInRing())
    value[7] = float(bond.GetIsConjugated())
    value[8] = float(bond.GetStereo() != Chem.BondStereo.STEREONONE)
    return value


def directed_graph_embedding(molecule: Chem.Mol) -> np.ndarray:
    atom = np.vstack([atom_feature(item) for item in molecule.GetAtoms()]).astype(np.float64)
    edges: list[tuple[int, int, np.ndarray]] = []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_feature(bond, atom.shape[1])
        edges.extend([(left, right, feature), (right, left, feature)])
    if not edges:
        pools = [atom.sum(axis=0), atom.mean(axis=0), atom.max(axis=0)]
        return np.concatenate(pools * 4)
    edge_src = np.asarray([item[0] for item in edges], dtype=np.int64)
    edge_dst = np.asarray([item[1] for item in edges], dtype=np.int64)
    bond_matrix = np.vstack([item[2] for item in edges])
    reverse = {(int(src), int(dst)): index for index, (src, dst) in enumerate(zip(edge_src, edge_dst, strict=True))}
    message = atom[edge_src] + 0.1 * bond_matrix
    summaries: list[np.ndarray] = []
    for _ in range(STEPS):
        incoming = np.zeros_like(atom)
        counts = np.zeros(len(atom), dtype=np.float64)
        for index, destination in enumerate(edge_dst):
            incoming[destination] += message[index]
            counts[destination] += 1.0
        node = atom + incoming / np.maximum(counts[:, None], 1.0)
        summaries.extend([node.sum(axis=0), node.mean(axis=0), node.max(axis=0)])
        new_message = np.empty_like(message)
        for index, (source, destination) in enumerate(zip(edge_src, edge_dst, strict=True)):
            reverse_message = message[reverse[(int(destination), int(source))]]
            context = incoming[source] - reverse_message
            new_message[index] = np.tanh(0.65 * message[index] + 0.25 * context + 0.10 * atom[source] + 0.05 * bond_matrix[index])
        message = new_message
    summaries.extend([message.sum(axis=0), message.mean(axis=0), message.max(axis=0)])
    return np.concatenate(summaries).astype(np.float64)


def fit_graph_model(
    graph_features: np.ndarray,
    y: np.ndarray,
    train_local: np.ndarray,
    train_global: np.ndarray,
    prediction_global: np.ndarray,
):
    if len(train_local) != len(train_global):
        raise RuntimeError("local/global training index lengths differ")
    if np.any(train_global < 0) or np.any(train_global >= len(graph_features)):
        raise RuntimeError("global training graph index out of bounds")
    if np.any(prediction_global < 0) or np.any(prediction_global >= len(graph_features)):
        raise RuntimeError("global prediction graph index out of bounds")
    if not np.isfinite(y[train_local]).all():
        raise RuntimeError("non-finite local training labels")
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(HIDDEN,), activation="relu", solver="adam", alpha=0.0001,
            learning_rate_init=0.005, max_iter=400, batch_size=32, random_state=2026,
            early_stopping=False, shuffle=True,
        ),
    )
    model.fit(graph_features[train_global], y[train_local])
    return model, np.asarray(model.predict(graph_features[prediction_global]), dtype=np.float64)


def nested_parent(
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
    inner_arms = np.full((len(outer_train), 4), np.nan, dtype=float)
    for fold in range(4):
        local_train = outer_train[np.flatnonzero(inner_folds != fold)]
        local_validation = outer_train[np.flatnonzero(inner_folds == fold)]
        inner_arms[inner_folds == fold] = reference.predict_base_models(
            target_dense, sparse_parts, fingerprints, y_global,
            global_indices[local_train], global_indices[local_validation], config, TARGET,
        )
    weights, intercept, blend_name, inner_r2 = reference.blend_from_oof(y[outer_train], inner_arms)
    outer_arms = reference.predict_base_models(
        target_dense, sparse_parts, fingerprints, y_global,
        global_indices[outer_train], global_indices[outer_validation], config, TARGET,
    )
    parent = reference.clip_prediction(y[outer_train], outer_arms @ weights + intercept)
    return parent, {"blend_name": blend_name, "blend_weights": [float(v) for v in weights], "blend_intercept": float(intercept), "inner_parent_r2": float(inner_r2)}


def build_panels(y, baseline, candidate, nearest, scaffolds, measurements, routed, groups):
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
    add("parent_only_control", ~routed, control=True)
    for name in sorted(set(scaffolds)):
        selected = scaffolds == name
        if int(np.sum(selected)) >= 10:
            add(f"scaffold_slice_{name}", selected)
    incomplete = any(isinstance(value, dict) and value["status"] in {"incomplete_constant_support", "failed_parent_only_control"} for value in panels.values())
    return panels, (min(values) if values else None), incomplete


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
    dense_base = np.hstack([descriptor, physical]).astype(np.float64)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.morgan_count_matrix(molecules, 3, int(reference.DEFAULT_CONFIG["morgan_bits"])),
        reference.text_matrix(keys, int(reference.DEFAULT_CONFIG["text_features"])),
    ]
    fingerprints = reference.morgan_bits(molecules, 2, int(reference.DEFAULT_CONFIG["morgan_bits"]))
    graph_features = np.vstack([directed_graph_embedding(molecule) for molecule in molecules])
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    y = frame["target"].to_numpy(float)
    canonical = frame["canonical"].to_numpy(object)
    groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    scaffolds = np.asarray([scaffold(value) for value in canonical], dtype=object)
    measurements = frame["measurements"].to_numpy(int)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[global_indices] = y
    target_dense = reference.target_dense_features(dense_base, cross_values, cross_available, TARGET)
    config = dict(reference.DEFAULT_CONFIG)
    config.update({"seed": 2026, "folds": 5, "similarity_barrier": SIMILARITY_BARRIER, "graph_steps": STEPS, "graph_hidden": HIDDEN})
    main_folds = folds_for(groups, 5)
    baseline = np.full(len(y), np.nan, dtype=float)
    candidate = np.full(len(y), np.nan, dtype=float)
    routed = np.zeros(len(y), dtype=bool)
    nearest = np.full(len(y), np.nan, dtype=float)
    graph_prediction = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = np.flatnonzero(main_folds == fold)
        training = np.flatnonzero(main_folds != fold)
        parent, parent_meta = nested_parent(y, groups, training, validation, target_dense, sparse_parts, fingerprints, global_indices, y_global, config)
        nearest_validation = nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
        _, graph_pred = fit_graph_model(graph_features, y, training, global_indices[training], global_indices[validation])
        route = (nearest_validation < SIMILARITY_BARRIER) & np.isfinite(graph_pred)
        routed_candidate = parent.copy()
        routed_candidate[route] = reference.clip_prediction(y[training], graph_pred[route])
        baseline[validation] = parent
        candidate[validation] = routed_candidate
        routed[validation] = route
        nearest[validation] = nearest_validation
        graph_prediction[validation] = graph_pred
        parent_score = float(r2_score(y[validation], parent))
        candidate_score = float(r2_score(y[validation], routed_candidate))
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "baseline_r2": parent_score,
            "candidate_r2": candidate_score,
            "delta_r2": candidate_score - parent_score,
            "routed_rows": int(np.sum(route)),
            "outer_fold_validation_groups": sorted(set(groups[validation])),
            "parent_blend": parent_meta,
        })

    scaffold_holdout: dict[str, object] = {}
    for scaffold_name in sorted(set(scaffolds)):
        validation = np.flatnonzero(scaffolds == scaffold_name)
        if len(validation) < 10:
            continue
        training = np.flatnonzero(scaffolds != scaffold_name)
        parent_holdout, _ = nested_parent(y, groups, training, validation, target_dense, sparse_parts, fingerprints, global_indices, y_global, config)
        nearest_holdout = nearest_similarity(fingerprints, global_indices[validation], global_indices[training])
        _, graph_pred = fit_graph_model(graph_features, y, training, global_indices[training], global_indices[validation])
        route = (nearest_holdout < SIMILARITY_BARRIER) & np.isfinite(graph_pred)
        holdout_candidate = parent_holdout.copy()
        holdout_candidate[route] = reference.clip_prediction(y[training], graph_pred[route])
        base_score = float(r2_score(y[validation], parent_holdout))
        cand_score = float(r2_score(y[validation], holdout_candidate))
        scaffold_holdout[str(scaffold_name)] = {"rows": int(len(validation)), "baseline_r2": base_score, "candidate_r2": cand_score, "delta_r2": cand_score - base_score, "routed_rows": int(np.sum(route))}

    panels, min_panel, panel_incomplete = build_panels(y, baseline, candidate, nearest, scaffolds, measurements, routed, groups)
    baseline_score = float(r2_score(y, baseline))
    candidate_score = float(r2_score(y, candidate))
    delta = candidate_score - baseline_score
    bootstrap = bootstrap_r2_lower(y, baseline, candidate, groups)
    scaffold_min = min((float(value["delta_r2"]) for value in scaffold_holdout.values()), default=None)
    positive_folds = int(sum(float(row["delta_r2"]) > 0.0 for row in fold_rows))
    report = {
        "rows": int(len(y)),
        "canonical_groups": int(len(np.unique(groups))),
        "baseline_r2_nested_parent": baseline_score,
        "candidate_r2_guarded_message_passing": candidate_score,
        "delta_r2": delta,
        "positive_outer_folds": positive_folds,
        "group_r2_bootstrap_lower": float(bootstrap),
        "outer_folds": fold_rows,
        "scaffold_holdout": scaffold_holdout,
        "scaffold_holdout_min_delta": scaffold_min,
        "min_panel_delta": min_panel,
        "panel_incomplete": panel_incomplete,
        "panels": panels,
        "route_definition": {"barrier": SIMILARITY_BARRIER, "routed_rows": int(np.sum(routed)), "parent_only_rows": int(np.sum(~routed)), "route_changed_rows": int(np.sum(np.abs(candidate - baseline) > 1.0e-12)), "parent_only_max_change": float(np.max(np.abs(candidate[~routed] - baseline[~routed]))) if np.any(~routed) else 0.0},
        "support_counts": {"exact_archive_measurement_rows": int(np.sum(measurements >= 2)), "singleton_rows": int(np.sum(measurements == 1))},
    }
    passed = bool(delta >= 0.01 and positive_folds >= 4 and bootstrap > 0.0 and (min_panel is None or min_panel >= -0.003) and (scaffold_min is None or scaffold_min >= -0.003) and not panel_incomplete and report["route_definition"]["parent_only_max_change"] <= 1.0e-12)
    report["pass"] = passed
    report["decision"] = "component_pass" if passed else "rejected_component_gate"
    source_hashes = {"script": sha256_file(Path(__file__).resolve()), "reference_module": sha256_file(root / "tools" / "initial_reference_pipeline.py")}
    config_hash = reference.canonical_json_hash(config)
    oof = pd.DataFrame({"canonical": canonical, "target_type": TARGET, "target": y, "baseline": baseline, "candidate": candidate, "route": routed, "graph_prediction": graph_prediction, "nearest_similarity": nearest, "scaffold": scaffolds, "no_stereo_group": groups, "measurements": measurements, "outer_fold": main_folds})
    oof.to_csv(run_dir / "oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)
    write_json(run_dir / "config.json", config)
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"rdkit={Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
    metrics = {"schema_version": "ppp.round2.ei-directed-message-passing-guard-run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C042-20260803-2045-safe-gap-wl-eps-mixed", "baseline_reference": "C001 official clean incumbent, freshly regenerated as nested parent", "official_inputs": inputs, "target": TARGET, "metrics": report, "source_hashes": source_hashes, "config_sha256": config_hash, "pass": passed, "decision": report["decision"], "elapsed_seconds": float(time.time() - started)}
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. No candidate or local_eval diagnostic was created by this component run.\n", encoding="utf-8")
    finish_time = datetime.now().astimezone()
    (run_dir / "run.log").write_text(f"experiment_id={run_dir.name}\nstarted_at={start_time.isoformat()}\nfinished_at={finish_time.isoformat()}\ndecision={report['decision']}\nelapsed_seconds={metrics['elapsed_seconds']:.3f}\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    manifest.extend([f"{source_hashes['script']}  SOURCE tools/round2_ei_directed_message_passing_guard.py", f"{source_hashes['reference_module']}  SOURCE tools/initial_reference_pipeline.py"])
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "delta_r2": delta, "passing": passed, "elapsed_seconds": metrics["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
