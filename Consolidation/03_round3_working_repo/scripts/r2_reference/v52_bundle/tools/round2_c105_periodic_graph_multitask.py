#!/usr/bin/env python3
"""C105: small from-scratch periodic graph multitask residual diagnostic."""

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
import torch
from rdkit import RDLogger
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c098_target_routed_qspr_full as c098
import round2_eea_cross_target_oof_residual_stack as plumbing


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
SEED = 2026
RESIDUAL_WEIGHT = 0.20
EPOCHS = 8
HIDDEN = 32


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def graph_tensors(molecules: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_rows: list[list[float]] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    graph_rows: list[int] = []
    graph_node_counts: list[int] = []
    offset = 0
    for graph, molecule in enumerate(molecules):
        atoms = list(molecule.GetAtoms())
        graph_node_counts.append(len(atoms))
        for atom in atoms:
            atomic = atom.GetAtomicNum()
            node_rows.append([
                min(float(atomic), 20.0) / 20.0,
                min(float(atom.GetDegree()), 6.0) / 6.0,
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                max(-2.0, min(2.0, float(atom.GetFormalCharge()))) / 2.0,
                min(float(atom.GetTotalNumHs()), 4.0) / 4.0,
                float(atomic == 0),
                float(atomic not in (0, 1, 6)),
            ])
            graph_rows.append(graph)
        for bond in molecule.GetBonds():
            left = offset + int(bond.GetBeginAtomIdx())
            right = offset + int(bond.GetEndAtomIdx())
            edge_src.extend([left, right])
            edge_dst.extend([right, left])
        endpoints = [offset + int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() == 0]
        if len(endpoints) == 2:
            edge_src.extend([endpoints[0], endpoints[1]])
            edge_dst.extend([endpoints[1], endpoints[0]])
        offset += len(atoms)
    return (
        torch.tensor(node_rows, dtype=torch.float32),
        torch.tensor(edge_src, dtype=torch.long),
        torch.tensor(edge_dst, dtype=torch.long),
        torch.tensor(graph_rows, dtype=torch.long),
    )


class PeriodicGraphModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = torch.nn.Linear(8, HIDDEN)
        self.message = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(3)])
        self.self_layers = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(3)])
        self.head = torch.nn.Linear(HIDDEN, len(TARGETS))

    def forward(self, nodes: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor, graph_rows: torch.Tensor, graph_count: int) -> torch.Tensor:
        hidden = torch.relu(self.input(nodes))
        for message_layer, self_layer in zip(self.message, self.self_layers, strict=True):
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, edge_dst, hidden[edge_src])
            hidden = torch.relu(self_layer(hidden) + message_layer(aggregate))
        pooled = torch.zeros((graph_count, HIDDEN), dtype=hidden.dtype)
        pooled.index_add_(0, graph_rows, hidden)
        counts = torch.bincount(graph_rows, minlength=graph_count).clamp_min(1).to(hidden.dtype).unsqueeze(1)
        return self.head(pooled / counts)


def fit_graph(
    nodes: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    graph_rows: torch.Tensor,
    graph_count: int,
    train_rows: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    target_graph_rows: dict[str, np.ndarray],
    active_targets: tuple[str, ...] = TARGETS,
    epochs: int = EPOCHS,
) -> tuple[PeriodicGraphModel, dict[str, float]]:
    torch.manual_seed(SEED)
    model = PeriodicGraphModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0e-4)
    scales: dict[str, float] = {}
    for target in TARGETS:
        if target in active_targets:
            values = residuals[target][train_rows[target]]
            scales[target] = max(float(np.std(values)), 1.0e-6)
        else:
            scales[target] = 1.0
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(nodes, edge_src, edge_dst, graph_rows, graph_count)
        losses: list[torch.Tensor] = []
        for index, target in enumerate(active_targets):
            rows = train_rows[target]
            graph_index = torch.tensor(target_graph_rows[target][rows], dtype=torch.long)
            expected = torch.tensor(residuals[target][rows] / scales[target], dtype=torch.float32)
            losses.append(torch.mean((prediction[graph_index, index] - expected) ** 2))
        torch.stack(losses).mean().backward()
        optimizer.step()
    return model, scales


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def panel_report(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, scaffolds: np.ndarray, similarity: np.ndarray) -> tuple[dict[str, Any], float]:
    report: dict[str, Any] = {}
    deltas: list[float] = []

    def add(name: str, selected: np.ndarray, minimum: int = 5) -> None:
        count = int(np.sum(selected))
        item: dict[str, Any] = {"rows": count, "delta_r2": 0.0, "status": "inapplicable"}
        if count >= minimum and np.var(y[selected]) > 1.0e-15:
            delta = float(r2_score(y[selected], candidate[selected]) - r2_score(y[selected], parent[selected]))
            item.update({"delta_r2": delta, "status": "evaluable"})
            deltas.append(delta)
        report[name] = item

    add("all_rows", np.ones(len(y), dtype=bool))
    add("similarity_lt_0.30", similarity < 0.30)
    add("similarity_ge_0.70", similarity >= 0.70)
    for scaffold in sorted(set(scaffolds)):
        add(f"scaffold_{scaffold}", scaffolds == scaffold, minimum=10)
    return report, float(min(deltas)) if deltas else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-specific", action="store_true", help="fit an independent graph encoder/head per active target")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    torch.set_num_threads(2)
    bundle = c098.parent_bundle(root, (root / args.data_dir).resolve())
    keys = bundle["keys"]
    key_to_graph = {key: index for index, key in enumerate(keys)}
    nodes, edge_src, edge_dst, graph_rows = graph_tensors(bundle["molecules"])
    graph_count = len(keys)
    target_info = bundle["target_info"]
    residuals = {target: target_info[target]["y"] - target_info[target]["parent"] for target in TARGETS}
    target_graph_rows = {target: np.asarray([key_to_graph[value] for value in target_info[target]["canonical"]], dtype=np.int64) for target in TARGETS}
    all_groups = sorted({group for target in TARGETS for group in target_info[target]["groups"]})
    group_array = np.asarray(all_groups, dtype=object)
    group_folds = np.full(len(all_groups), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(all_groups)), groups=group_array)):
        group_folds[validation] = fold
    group_to_fold = {group: int(fold) for group, fold in zip(all_groups, group_folds, strict=True)}
    folds_by_target = {target: np.asarray([group_to_fold[group] for group in target_info[target]["groups"]], dtype=np.int64) for target in TARGETS}
    active_targets = ("eps", "nc", "ei", "tg") if args.target_specific else TARGETS
    candidate_oof = {target: target_info[target]["parent"].copy() for target in TARGETS}
    fold_reports: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    for fold in range(5):
        validation_rows = {target: np.flatnonzero(folds_by_target[target] == fold) for target in TARGETS}
        if args.target_specific:
            for target in active_targets:
                train_rows = {target: np.flatnonzero(folds_by_target[target] != fold)}
                trained, scales = fit_graph(nodes, edge_src, edge_dst, graph_rows, graph_count, train_rows, residuals, target_graph_rows, active_targets=(target,))
                trained.eval()
                with torch.no_grad():
                    outputs = trained(nodes, edge_src, edge_dst, graph_rows, graph_count).numpy()
                rows = validation_rows[target]
                correction = RESIDUAL_WEIGHT * outputs[target_graph_rows[target][rows], TARGETS.index(target)] * scales[target]
                candidate_oof[target][rows] = target_info[target]["parent"][rows] + correction
        else:
            train_rows = {target: np.flatnonzero(folds_by_target[target] != fold) for target in TARGETS}
            trained, scales = fit_graph(nodes, edge_src, edge_dst, graph_rows, graph_count, train_rows, residuals, target_graph_rows)
            trained.eval()
            with torch.no_grad():
                outputs = trained(nodes, edge_src, edge_dst, graph_rows, graph_count).numpy()
            for index, target in enumerate(TARGETS):
                rows = validation_rows[target]
                correction = RESIDUAL_WEIGHT * outputs[target_graph_rows[target][rows], index] * scales[target]
                candidate_oof[target][rows] = target_info[target]["parent"][rows] + correction
    raw_test = bundle["test_detail"][["id", "target_type", "model_prediction"]].copy()
    train_rows_all = {target: np.arange(len(target_info[target]["y"])) for target in TARGETS}
    if args.target_specific:
        fitted_models = {}
        for target in active_targets:
            fitted_models[target] = fit_graph(nodes, edge_src, edge_dst, graph_rows, graph_count, {target: train_rows_all[target]}, residuals, target_graph_rows, active_targets=(target,))
    else:
        shared_model, shared_scales = fit_graph(nodes, edge_src, edge_dst, graph_rows, graph_count, train_rows_all, residuals, target_graph_rows)
        shared_model.eval()
        with torch.no_grad():
            shared_outputs = shared_model(nodes, edge_src, edge_dst, graph_rows, graph_count).numpy()
    for index, target in enumerate(TARGETS):
        test_frame = bundle["test"].loc[bundle["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_graph_rows = np.asarray([key_to_graph[value] for value in test_frame["canonical"]], dtype=np.int64)
        parent_test = raw_test.loc[raw_test["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(float)
        if target in active_targets:
            if args.target_specific:
                trained, scales = fitted_models[target]
                trained.eval()
                with torch.no_grad():
                    outputs = trained(nodes, edge_src, edge_dst, graph_rows, graph_count).numpy()
                candidate_test = parent_test + RESIDUAL_WEIGHT * outputs[test_graph_rows, TARGETS.index(target)] * scales[target]
            else:
                candidate_test = parent_test + RESIDUAL_WEIGHT * shared_outputs[test_graph_rows, index] * shared_scales[target]
        else:
            candidate_test = parent_test
        mask = raw_test["target_type"].to_numpy(object) == target
        raw_test.loc[mask, "model_prediction"] = candidate_test
    target_reports: dict[str, Any] = {}
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = target_info[target]
        y = info["y"]
        parent = info["parent"]
        candidate = candidate_oof[target]
        groups = info["groups"]
        scaffolds = info["scaffolds"]
        global_validation = np.asarray([key_to_graph[value] for value in info["canonical"]], dtype=np.int64)
        similarity = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            validation = np.flatnonzero(folds_by_target[target] == fold)
            training = np.flatnonzero(folds_by_target[target] != fold)
            similarity[validation] = fixed_features.nearest_similarity(bundle["fingerprints"], global_validation[validation], global_validation[training])
        fold_rows = []
        for fold in range(5):
            validation = np.flatnonzero(folds_by_target[target] == fold)
            fold_rows.append({"fold": fold, "rows": int(len(validation)), "parent_r2": float(r2_score(y[validation], parent[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation]))})
        delta = float(r2_score(y, candidate) - r2_score(y, parent))
        panel, minimum_panel = panel_report(y, parent, candidate, scaffolds, similarity)
        lower = bootstrap_lower(y, parent, candidate, groups)
        positive = int(sum(row["delta_r2"] > 0.0 for row in fold_rows))
        target_reports[target] = {"parent_r2": float(r2_score(y, parent)), "candidate_r2": float(r2_score(y, candidate)), "delta_r2": delta, "positive_folds": positive, "group_bootstrap_lower": lower, "minimum_panel_delta": minimum_panel, "folds": fold_rows, "panels": panel, "feature_count": 8, "pass": bool(target not in active_targets or (delta >= 0.01 and positive >= 4 and lower > 0.0 and minimum_panel >= 0.0)), "unchanged_parent": target not in active_targets}
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": y, "parent": parent, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": folds_by_target[target]}))
    detail, override_report = reference.apply_official_overrides(raw_test, bundle["test"], bundle["raw_labels"])
    submission = detail[["id", "target"]].copy()
    if len(submission) != len(bundle["test"]) or not submission["id"].equals(bundle["test"]["id"]):
        raise RuntimeError("C105 output ID/order mismatch")
    if submission["id"].duplicated().any() or not np.isfinite(submission["target"].to_numpy(float)).all():
        raise RuntimeError("C105 output invalid")
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in TARGETS]))
    max_loss = float(min(target_reports[target]["delta_r2"] for target in TARGETS))
    complete_pass = bool(mean_candidate >= mean_parent + 0.002 and max_loss >= -0.003 and all(target_reports[target]["pass"] for target in active_targets))
    submission.to_csv(run_dir / "predictions.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
    source_paths = {"script": root / "tools" / "round2_c105_periodic_graph_multitask.py", "c098_parent": root / "tools" / "round2_c098_target_routed_qspr_full.py", "reference": root / "tools" / "initial_reference_pipeline.py", "fixed_features": root / "tools" / "round2_c063_egb_endpoint_conjugation_residual.py", "plumbing": root / "tools" / "round2_eea_cross_target_oof_residual_stack.py"}
    report = {"schema_version": "ppp.round2.c106.target-specific-periodic-graph.run.v1" if args.target_specific else "ppp.round2.c105.periodic-graph-multitask.run.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "C050 source rebuild via C098 parent bundle; graph trained from scratch", "official_inputs": bundle["inputs"], "official_only": True, "external_label_file_read": False, "local_eval_read": False, "kaggle_compute": False, "prior_prediction_input": False, "pretrained_weights": False, "active_targets": list(active_targets), "target_reports": target_reports, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "maximum_target_loss": max_loss, "complete_output_rows": int(len(submission)), "complete_output_order_pass": True, "complete_candidate_gate_pass": complete_pass, "parent_replay_required": True, "parent_replay_oof_max_abs": None, "parent_replay_test_max_abs": None, "architecture": {"node_features": 8, "hidden": HIDDEN, "message_layers": 3, "epochs": EPOCHS, "periodic_endpoint_edges": True, "target_specific": args.target_specific, "torch_version": torch.__version__}, "official_override_report": override_report, "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()}, "elapsed_seconds": float(time.time() - started), "decision": "candidate_pending_fresh_replay" if complete_pass else "rejected_full_candidate_gate"}
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "targets": list(TARGETS), "residual_weight": RESIDUAL_WEIGHT, "epochs": EPOCHS, "hidden": HIDDEN, "message_layers": 3, "periodic_endpoint_edges": True, "outer": "shared canonical_no_stereo GroupKFold(5)", "external_label_file_read": False, "local_eval_read": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\ntorch={torch.__version__}\nrdkit={reference.Chem.rdBase.rdkitVersion}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Mean parent R2 `{mean_parent:.12f}`; candidate R2 `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for path in source_paths.values():
        manifest.append(f"{sha256_file(path)}  ../../../tools/{path.name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "target_deltas": {target: target_reports[target]["delta_r2"] for target in TARGETS}}, sort_keys=True))


if __name__ == "__main__":
    main()
