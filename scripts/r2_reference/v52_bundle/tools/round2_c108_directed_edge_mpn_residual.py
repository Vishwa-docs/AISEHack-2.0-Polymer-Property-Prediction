#!/usr/bin/env python3
"""C108: clean directed edge-conditioned graph residual.

This is a fixed, official-only experiment.  It rebuilds the C050 parent twice
from source, uses the exact parent target-local folds, and trains one small
from-scratch directed edge model per active target and replica.  It deliberately
does not read saved predictions, checkpoints, caches, or local_eval files.
"""

from __future__ import annotations

import argparse
import gc
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
import torch
from rdkit import Chem, RDLogger
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c098_target_routed_qspr_full as c098
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")

TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("eps", "nc", "ei", "tg")
REPLICA_SEEDS = (2026, 2027, 2028)
RESIDUAL_WEIGHT = 0.20
EPOCHS = 12
HIDDEN = 48
NODE_DIM = 8
EDGE_DIM = 17


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def edge_features(begin: Chem.Atom, end: Chem.Atom, bond: Chem.Bond | None) -> list[float]:
    values = [0.0] * 4
    aromatic = conjugated = ring = 0.0
    stereo = [0.0] * 8
    if bond is not None:
        kind = str(bond.GetBondType()).upper()
        values[{"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}.get(kind, 0)] = 1.0
        aromatic = float(bond.GetIsAromatic())
        conjugated = float(bond.GetIsConjugated())
        ring = float(bond.IsInRing())
        stereo[min(max(int(bond.GetStereo()), 0), len(stereo) - 1)] = 1.0
    values.extend([aromatic, conjugated, ring])
    values.extend(stereo)
    values.extend([float(begin.GetAtomicNum() == 0), float(end.GetAtomicNum() == 0)])
    if len(values) != EDGE_DIM:
        raise RuntimeError(f"edge feature width mismatch: {len(values)} != {EDGE_DIM}")
    return values


def graph_tensors(molecules: list[Chem.Mol]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_rows: list[list[float]] = []
    edge_rows: list[list[float]] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    graph_rows: list[int] = []
    root_rows: list[int] = []
    offset = 0
    for graph, molecule in enumerate(molecules):
        atoms = list(molecule.GetAtoms())
        if not atoms:
            raise RuntimeError("empty molecule encountered")
        root_rows.append(offset)
        for atom in atoms:
            atomic = atom.GetAtomicNum()
            node_rows.append([
                min(float(atomic), 20.0) / 20.0,
                min(float(atom.GetDegree()), 6.0) / 6.0,
                max(-2.0, min(2.0, float(atom.GetFormalCharge()))) / 2.0,
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                min(float(int(atom.GetHybridization())), 6.0) / 6.0,
                min(float(atom.GetTotalNumHs()), 4.0) / 4.0,
                min(float(atom.GetDegree()), 6.0) / 6.0,
            ])
            graph_rows.append(graph)
        for bond in molecule.GetBonds():
            left_atom = molecule.GetAtomWithIdx(int(bond.GetBeginAtomIdx()))
            right_atom = molecule.GetAtomWithIdx(int(bond.GetEndAtomIdx()))
            left = offset + int(bond.GetBeginAtomIdx())
            right = offset + int(bond.GetEndAtomIdx())
            edge_src.extend([left, right])
            edge_dst.extend([right, left])
            edge_rows.extend([edge_features(left_atom, right_atom, bond), edge_features(right_atom, left_atom, bond)])
        endpoints = [atom for atom in atoms if atom.GetAtomicNum() == 0]
        if len(endpoints) == 2:
            left = offset + int(endpoints[0].GetIdx())
            right = offset + int(endpoints[1].GetIdx())
            edge_src.extend([left, right])
            edge_dst.extend([right, left])
            edge_rows.extend([edge_features(endpoints[0], endpoints[1], None), edge_features(endpoints[1], endpoints[0], None)])
        offset += len(atoms)
    return (
        torch.tensor(node_rows, dtype=torch.float32),
        torch.tensor(edge_src, dtype=torch.long),
        torch.tensor(edge_dst, dtype=torch.long),
        torch.tensor(edge_rows, dtype=torch.float32),
        torch.tensor(graph_rows, dtype=torch.long),
        torch.tensor(root_rows, dtype=torch.long),
    )


class DirectedEdgeResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_input = torch.nn.Linear(NODE_DIM, HIDDEN)
        self.edge_input = torch.nn.Linear(EDGE_DIM, HIDDEN)
        self.message_layers = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(3)])
        self.self_layers = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(3)])
        self.head = torch.nn.Linear(HIDDEN, 1)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edges: torch.Tensor,
        graph_rows: torch.Tensor,
        root_rows: torch.Tensor,
        graph_count: int,
    ) -> torch.Tensor:
        hidden = torch.relu(self.node_input(nodes))
        edge_hidden = self.edge_input(edges)
        for message_layer, self_layer in zip(self.message_layers, self.self_layers, strict=True):
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, edge_dst, hidden[edge_src] + edge_hidden)
            hidden = torch.relu(self_layer(hidden) + message_layer(aggregate))
        pooled = torch.zeros((graph_count, HIDDEN), dtype=hidden.dtype)
        pooled.index_add_(0, graph_rows, hidden)
        counts = torch.bincount(graph_rows, minlength=graph_count).clamp_min(1).to(hidden.dtype).unsqueeze(1)
        pooled = pooled / counts + hidden[root_rows]
        return self.head(pooled).squeeze(1)


def fit_model(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    graph_indices: np.ndarray,
    residual: np.ndarray,
    rows: np.ndarray,
    seed: int,
) -> tuple[DirectedEdgeResidual, float]:
    nodes, edge_src, edge_dst, edges, graph_rows, root_rows = tensors
    torch.manual_seed(seed)
    model = DirectedEdgeResidual()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1.0e-4)
    scale = max(float(np.std(residual[rows])), 1.0e-6)
    graph_index = torch.tensor(graph_indices[rows], dtype=torch.long)
    expected = torch.tensor(residual[rows] / scale, dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        output = model(nodes, edge_src, edge_dst, edges, graph_rows, root_rows, len(root_rows))
        loss = torch.mean((output[graph_index] - expected) ** 2)
        loss.backward()
        optimizer.step()
    return model, scale


def predict_model(
    model: DirectedEdgeResidual,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> np.ndarray:
    nodes, edge_src, edge_dst, edges, graph_rows, root_rows = tensors
    model.eval()
    with torch.no_grad():
        return model(nodes, edge_src, edge_dst, edges, graph_rows, root_rows, len(root_rows)).numpy()


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray, seed: int) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15:
            values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    if not values:
        return float("nan")
    return float(np.quantile(values, 0.025))


def panel_report(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    scaffolds: np.ndarray,
    similarity: np.ndarray,
    available: np.ndarray,
) -> tuple[dict[str, Any], float]:
    panels: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected))
        item: dict[str, Any] = {"rows": count, "delta_r2": 0.0, "status": "inapplicable"}
        if count >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"delta_r2": delta, "status": "evaluable"})
            deltas.append(delta)
        panels[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    add("similarity_lt_0.30", similarity < 0.30)
    add("similarity_0.30_0.50", (similarity >= 0.30) & (similarity < 0.50))
    add("similarity_0.50_0.70", (similarity >= 0.50) & (similarity < 0.70))
    add("similarity_ge_0.70", similarity >= 0.70)
    add("cross_property_available", available)
    add("cross_property_missing", ~available)
    for scaffold in sorted(set(str(value) for value in scaffolds)):
        add(f"scaffold_{scaffold}", np.asarray([str(value) == scaffold for value in scaffolds]), minimum=10)
    return panels, float(min(deltas)) if deltas else 0.0


def target_report(bundle: dict[str, Any], target: str, candidate: np.ndarray, seed: int) -> dict[str, Any]:
    info = bundle["target_info"][target]
    y = np.asarray(info["y"], dtype=float)
    parent = np.asarray(info["parent"], dtype=float)
    folds = np.asarray(info["folds"], dtype=int)
    if set(folds.tolist()) != set(range(5)):
        raise RuntimeError(f"unexpected C050 fold map for {target}: {sorted(set(folds.tolist()))}")
    similarity = np.full(len(y), np.nan, dtype=float)
    for fold in range(5):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        validation_graph = np.asarray(info["indices"], dtype=np.int64)[validation]
        training_graph = np.asarray(info["indices"], dtype=np.int64)[training]
        similarity[validation] = fixed_features.nearest_similarity(bundle["fingerprints"], validation_graph, training_graph)
    cross = bundle["cross_available"][np.asarray(info["indices"], dtype=np.int64)]
    target_index = TARGETS.index(target)
    available = np.any(np.delete(cross, target_index, axis=1), axis=1)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        rows = np.flatnonzero(folds == fold)
        parent_r2 = float(r2_score(y[rows], parent[rows]))
        candidate_r2 = float(r2_score(y[rows], candidate[rows]))
        fold_rows.append({"fold": fold, "rows": int(len(rows)), "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2})
    delta = float(r2_score(y, candidate) - r2_score(y, parent))
    panels, minimum_panel = panel_report(y, parent, candidate, info["groups"], info["scaffolds"], similarity, available)
    lower = bootstrap_lower(y, parent, candidate, info["groups"], seed + 97)
    positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
    return {
        "parent_r2": float(r2_score(y, parent)),
        "candidate_r2": float(r2_score(y, candidate)),
        "delta_r2": delta,
        "positive_folds": positive,
        "group_bootstrap_lower": lower,
        "minimum_panel_delta": minimum_panel,
        "folds": fold_rows,
        "panels": panels,
        "feature_count": NODE_DIM + EDGE_DIM,
        "pass": bool(delta >= 0.01 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0),
    }


def parent_signature(bundle: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    oof = {target: np.asarray(bundle["target_info"][target]["parent"], dtype=float).copy() for target in TARGETS}
    test = bundle["test_detail"].sort_values("id")["model_prediction"].to_numpy(float).copy()
    return oof, test


def compare_parent_signatures(left: tuple[dict[str, np.ndarray], np.ndarray], right: tuple[dict[str, np.ndarray], np.ndarray]) -> tuple[float, float]:
    left_oof, left_test = left
    right_oof, right_test = right
    max_oof = 0.0
    for target in TARGETS:
        if left_oof[target].shape != right_oof[target].shape:
            raise RuntimeError(f"parent OOF shape mismatch for {target}")
        max_oof = max(max_oof, float(np.max(np.abs(left_oof[target] - right_oof[target]))))
    if left_test.shape != right_test.shape:
        raise RuntimeError("parent test shape mismatch")
    max_test = float(np.max(np.abs(left_test - right_test)))
    if max(max_oof, max_test) > 1.0e-12:
        raise RuntimeError(f"independent C050 parent replay mismatch: oof={max_oof} test={max_test}")
    return max_oof, max_test


def target_graph_context(bundle: dict[str, Any], target: str) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray, np.ndarray]:
    info = bundle["target_info"][target]
    test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    local_keys = sorted(set(info["canonical"]) | set(test_frame["canonical"]))
    local_molecules = [bundle["molecules"][bundle["key_to_index"][key]] for key in local_keys]
    local_key_to_index = {key: index for index, key in enumerate(local_keys)}
    train_graph_indices = np.asarray([local_key_to_index[value] for value in info["canonical"]], dtype=np.int64)
    test_graph_indices = np.asarray([local_key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
    return graph_tensors(local_molecules), train_graph_indices, test_graph_indices


def run_replica(
    bundle: dict[str, Any],
    contexts: dict[str, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, Any]:
    candidate_oof = {target: np.asarray(bundle["target_info"][target]["parent"], dtype=float).copy() for target in TARGETS}
    parent_test = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    test_model = parent_test.copy()
    for target in ACTIVE_TARGETS:
        info = bundle["target_info"][target]
        tensors, graph_indices, test_graph_indices = contexts[target]
        residual = np.asarray(info["y"], dtype=float) - np.asarray(info["parent"], dtype=float)
        folds = np.asarray(info["folds"], dtype=int)
        for fold in range(5):
            train_rows = np.flatnonzero(folds != fold)
            validation_rows = np.flatnonzero(folds == fold)
            model, scale = fit_model(tensors, graph_indices, residual, train_rows, seed + 11 * TARGETS.index(target) + fold)
            output = predict_model(model, tensors)
            candidate_oof[target][validation_rows] = np.asarray(info["parent"], dtype=float)[validation_rows] + RESIDUAL_WEIGHT * output[graph_indices[validation_rows]] * scale
        train_rows = np.arange(len(residual), dtype=np.int64)
        model, scale = fit_model(tensors, graph_indices, residual, train_rows, seed + 11 * TARGETS.index(target) + 91)
        output = predict_model(model, tensors)
        test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        parent_values = test_model.loc[test_model["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float)
        candidate_values = parent_values + RESIDUAL_WEIGHT * output[test_graph_indices] * scale
        replacement = pd.Series(candidate_values, index=test_frame["id"].to_numpy())
        mask = test_model["target_type"].to_numpy(object) == target
        test_model.loc[mask, "model_prediction"] = test_model.loc[mask, "id"].map(replacement).to_numpy(float)
    detail, override_report = reference.apply_official_overrides(test_model, bundle["test"], bundle["raw_labels"])
    submission = detail[["id", "target"]].copy()
    if len(submission) != len(bundle["test"]) or not submission["id"].equals(bundle["test"]["id"]):
        raise RuntimeError("C108 output ID/order mismatch")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C108 output contains duplicate IDs or non-finite predictions")
    reports = {target: target_report(bundle, target, candidate_oof[target], seed) if target in ACTIVE_TARGETS else {
        "parent_r2": float(r2_score(bundle["target_info"][target]["y"], bundle["target_info"][target]["parent"])),
        "candidate_r2": float(r2_score(bundle["target_info"][target]["y"], candidate_oof[target])),
        "delta_r2": 0.0,
        "positive_folds": 0,
        "group_bootstrap_lower": 0.0,
        "minimum_panel_delta": 0.0,
        "folds": [],
        "panels": {"unchanged_parent": {"rows": int(len(candidate_oof[target])), "delta_r2": 0.0, "status": "unchanged"}},
        "feature_count": 0,
        "pass": True,
    } for target in TARGETS}
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(reports[target]["delta_r2"] for target in TARGETS))
    complete_pass = bool(mean_candidate >= mean_parent + 0.002 and max_loss >= -0.003 and all(reports[target]["pass"] for target in ACTIVE_TARGETS))
    return {
        "seed": seed,
        "oof": candidate_oof,
        "submission": submission,
        "reports": reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": max_loss,
        "complete_candidate_gate_pass": complete_pass,
        "override_report": override_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    data_dir = (root / args.data_dir).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only C108 directory required")
    started = time.time()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    first_parent = c098.parent_bundle(root, data_dir)
    first_signature = parent_signature(first_parent)
    del first_parent
    gc.collect()
    second_parent = c098.parent_bundle(root, data_dir)
    second_signature = parent_signature(second_parent)
    parent_replay_oof, parent_replay_test = compare_parent_signatures(first_signature, second_signature)
    del second_parent, second_signature
    gc.collect()
    bundle = c098.parent_bundle(root, data_dir)
    final_signature = parent_signature(bundle)
    final_replay_oof, final_replay_test = compare_parent_signatures(first_signature, final_signature)
    parent_replay_oof = max(parent_replay_oof, final_replay_oof)
    parent_replay_test = max(parent_replay_test, final_replay_test)
    del first_signature, final_signature
    gc.collect()
    contexts = {target: target_graph_context(bundle, target) for target in ACTIVE_TARGETS}
    replicas = [run_replica(bundle, contexts, seed) for seed in REPLICA_SEEDS]
    replica_passes = int(sum(bool(replica["complete_candidate_gate_pass"] and all(replica["reports"][target]["pass"] for target in ACTIVE_TARGETS)) for replica in replicas))

    aggregate_oof = {target: np.mean([replica["oof"][target] for replica in replicas], axis=0) for target in TARGETS}
    aggregate_reports = {target: target_report(bundle, target, aggregate_oof[target], 2026) if target in ACTIVE_TARGETS else {
        "parent_r2": float(r2_score(bundle["target_info"][target]["y"], bundle["target_info"][target]["parent"])),
        "candidate_r2": float(r2_score(bundle["target_info"][target]["y"], aggregate_oof[target])),
        "delta_r2": 0.0,
        "positive_folds": 0,
        "group_bootstrap_lower": 0.0,
        "minimum_panel_delta": 0.0,
        "folds": [],
        "panels": {"unchanged_parent": {"rows": int(len(aggregate_oof[target])), "delta_r2": 0.0, "status": "unchanged"}},
        "feature_count": 0,
        "pass": True,
    } for target in TARGETS}
    aggregate_values = np.mean([replica["submission"].sort_values("id")["target"].to_numpy(float) for replica in replicas], axis=0)
    aggregate_raw = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    aggregate_raw["model_prediction"] = aggregate_values
    aggregate_detail, aggregate_override_report = reference.apply_official_overrides(aggregate_raw, bundle["test"], bundle["raw_labels"])
    aggregate_submission = aggregate_detail[["id", "target"]].copy()
    if len(aggregate_submission) != 4940 or not aggregate_submission["id"].equals(bundle["test"]["id"]):
        raise RuntimeError("C108 aggregate output does not contain the exact 4,940 ordered test IDs")
    mean_parent = float(np.mean([aggregate_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([aggregate_reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(aggregate_reports[target]["delta_r2"] for target in TARGETS))
    aggregate_gate = bool(mean_candidate >= mean_parent + 0.002 and max_loss >= -0.003 and all(aggregate_reports[target]["pass"] for target in ACTIVE_TARGETS))
    decision = "candidate_pending_notebook_parity" if aggregate_gate and replica_passes >= 2 else "rejected_full_candidate_gate"

    aggregate_submission.to_csv(run_dir / "predictions.csv", index=False)
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = bundle["target_info"][target]
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": info["y"], "parent": info["parent"], "candidate": aggregate_oof[target], "group": info["groups"], "scaffold": info["scaffolds"], "outer_fold": info["folds"]}))
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    for index, replica in enumerate(replicas, start=1):
        replica["submission"].to_csv(run_dir / f"replica_{index}_predictions.csv", index=False)
        replica_parts: list[pd.DataFrame] = []
        for target in TARGETS:
            info = bundle["target_info"][target]
            replica_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": info["y"], "parent": info["parent"], "candidate": replica["oof"][target], "group": info["groups"], "scaffold": info["scaffolds"], "outer_fold": info["folds"]}))
        pd.concat(replica_parts, ignore_index=True).to_csv(run_dir / f"replica_{index}_oof.csv", index=False)

    source_paths = {
        "script": root / "tools" / "round2_c108_directed_edge_mpn_residual.py",
        "parent_builder": root / "tools" / "round2_c098_target_routed_qspr_full.py",
        "reference": root / "tools" / "initial_reference_pipeline.py",
        "similarity": root / "tools" / "round2_c063_egb_endpoint_conjugation_residual.py",
        "grouping": root / "tools" / "round2_eea_cross_target_oof_residual_stack.py",
    }
    report = {
        "schema_version": "ppp.round2.c108.directed-edge-mpn-residual.run.v1",
        "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": "C050 rebuilt twice from official inputs in memory; no saved parent predictions loaded",
        "official_inputs": bundle["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "submission": False,
        "prior_prediction_input": False,
        "pretrained_weights": False,
        "active_targets": list(ACTIVE_TARGETS),
        "replica_seeds": list(REPLICA_SEEDS),
        "replica_pass_count": replica_passes,
        "replicas": [{"seed": replica["seed"], "mean_parent_r2": replica["mean_parent_r2"], "mean_candidate_r2": replica["mean_candidate_r2"], "mean_gain": replica["mean_gain"], "maximum_target_loss": replica["maximum_target_loss"], "complete_candidate_gate_pass": replica["complete_candidate_gate_pass"], "target_reports": replica["reports"]} for replica in replicas],
        "target_reports": aggregate_reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(aggregate_submission)),
        "complete_output_order_pass": True,
        "complete_candidate_gate_pass": aggregate_gate,
        "replica_rule_pass": replica_passes >= 2,
        "notebook_parity_pass": False,
        "parent_replay_required": True,
        "parent_replay_oof_max_abs": parent_replay_oof,
        "parent_replay_test_max_abs": parent_replay_test,
        "architecture": {"node_features": NODE_DIM, "directed_edge_features": EDGE_DIM, "hidden": HIDDEN, "message_layers": 3, "epochs": EPOCHS, "learning_rate": 0.005, "weight_decay": 0.0001, "residual_weight": RESIDUAL_WEIGHT, "root_plus_mean_pooling": True, "graph_scope": "target-local train/test canonical union", "torch_version": torch.__version__},
        "official_override_report": aggregate_override_report,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "elapsed_seconds": float(time.time() - started),
        "decision": decision,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": REPLICA_SEEDS[0], "replica_seeds": list(REPLICA_SEEDS), "targets": list(TARGETS), "active_targets": list(ACTIVE_TARGETS), "residual_weight": RESIDUAL_WEIGHT, "epochs": EPOCHS, "hidden": HIDDEN, "message_layers": 3, "edge_features": ["bond_type_one_hot", "aromatic", "conjugated", "ring", "stereo_class", "dummy_endpoint_flags"], "graph_scope": "target-local train/test canonical union", "outer_fold_source": "exact target-local C050 parent folds from independent source rebuild", "external_label_file_read": False, "local_eval_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\ntorch={torch.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{decision}**. Aggregate parent mean `{mean_parent:.12f}`; candidate mean `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. Replica gate count `{replica_passes}/3`. Independent parent replay OOF max `{parent_replay_oof:.3e}`; test max `{parent_replay_test:.3e}`. Official-only; no local_eval, external_label file, Kaggle, upload, or submission action.\n", encoding="utf-8")
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, path in source_paths.items():
        manifest.append(f"{sha256_file(path)}  source/{name}/{path.name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": decision, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "replica_pass_count": replica_passes, "parent_replay_oof_max_abs": parent_replay_oof, "parent_replay_test_max_abs": parent_replay_test, "target_deltas": {target: aggregate_reports[target]["delta_r2"] for target in TARGETS}}, sort_keys=True))


if __name__ == "__main__":
    main()
