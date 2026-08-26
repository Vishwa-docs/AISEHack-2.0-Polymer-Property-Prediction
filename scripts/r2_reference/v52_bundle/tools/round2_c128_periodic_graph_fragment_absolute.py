#!/usr/bin/env python3
"""C128: absolute periodic graph + fragment encoder for weak targets.

This is intentionally different from C105/C106: it is target-local, absolute
regression (not a shared residual trunk), adds fragment/global descriptors, and
uses a directed edge-conditioned periodic graph.  It never reads C127 outputs.
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
import torch
from rdkit import RDLogger
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c063_egb_endpoint_conjugation_residual as fixed_features
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as c127


RDLogger.DisableLog("rdApp.*")
TARGETS = tuple(reference.TARGETS)
ACTIVE_TARGETS = ("eps", "nc", "ei", "eea")
SEED = 2026
N_FOLDS = 5
HIDDEN = 96
LAYERS = 3
EPOCHS = 20


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


def graph_tensors(molecules: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_rows: list[list[float]] = []
    edge_rows: list[list[float]] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    graph_rows: list[int] = []
    offset = 0
    element_bins = (6, 7, 8, 9, 16, 17)
    for graph, molecule in enumerate(molecules):
        atoms = list(molecule.GetAtoms())
        for atom in atoms:
            atomic = atom.GetAtomicNum()
            node_rows.append([
                min(float(atomic), 54.0) / 54.0,
                min(float(atom.GetDegree()), 6.0) / 6.0,
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                max(-2.0, min(2.0, float(atom.GetFormalCharge()))) / 2.0,
                min(float(atom.GetTotalNumHs()), 4.0) / 4.0,
                float(atomic == 0),
                float(atomic not in (0, 1, 6)),
                *[float(atomic == value) for value in element_bins],
            ])
            graph_rows.append(graph)

        def add_edge(left: int, right: int, bond_type: float, ring: float, conjugated: float, periodic: float) -> None:
            left_atom = molecule.GetAtomWithIdx(left - offset)
            right_atom = molecule.GetAtomWithIdx(right - offset)
            begin_dummy = float(left_atom.GetAtomicNum() == 0)
            end_dummy = float(right_atom.GetAtomicNum() == 0)
            edge_rows.append([
                float(bond_type == 1.0), float(bond_type == 2.0), float(bond_type == 3.0),
                float(bond_type == 1.5), ring, conjugated,
                float(begin_dummy or end_dummy), begin_dummy, end_dummy, periodic,
            ])
            edge_src.append(left)
            edge_dst.append(right)

        for bond in molecule.GetBonds():
            left = offset + int(bond.GetBeginAtomIdx())
            right = offset + int(bond.GetEndAtomIdx())
            bond_type = float(bond.GetBondTypeAsDouble())
            ring = float(bond.IsInRing())
            conjugated = float(bond.GetIsAromatic() or bond_type > 1.0)
            add_edge(left, right, bond_type, ring, conjugated, 0.0)
            add_edge(right, left, bond_type, ring, conjugated, 0.0)

        endpoints = [int(atom.GetIdx()) for atom in atoms if atom.GetAtomicNum() == 0]
        neighbors: list[int] = []
        for endpoint in endpoints:
            neighbors.extend(int(neighbor.GetIdx()) for neighbor in molecule.GetAtomWithIdx(endpoint).GetNeighbors())
        if len(endpoints) == 2:
            left, right = offset + endpoints[0], offset + endpoints[1]
            add_edge(left, right, 1.0, 0.0, 0.0, 1.0)
            add_edge(right, left, 1.0, 0.0, 0.0, 1.0)
        if len(neighbors) >= 2:
            left, right = offset + neighbors[0], offset + neighbors[-1]
            add_edge(left, right, 1.0, 0.0, 1.0, 1.0)
            add_edge(right, left, 1.0, 0.0, 1.0, 1.0)
        offset += len(atoms)
    return (
        torch.tensor(node_rows, dtype=torch.float32),
        torch.tensor(edge_src, dtype=torch.long),
        torch.tensor(edge_dst, dtype=torch.long),
        torch.tensor(edge_rows, dtype=torch.float32),
        torch.tensor(graph_rows, dtype=torch.long),
    )


class PeriodicFragmentEncoder(torch.nn.Module):
    def __init__(self, global_dim: int) -> None:
        super().__init__()
        self.node_input = torch.nn.Linear(14, HIDDEN)
        self.edge_input = torch.nn.Linear(10, HIDDEN)
        self.global_input = torch.nn.Linear(global_dim, HIDDEN)
        self.message_layers = torch.nn.ModuleList([torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(LAYERS)])
        self.update_layers = torch.nn.ModuleList([torch.nn.Linear(2 * HIDDEN, HIDDEN) for _ in range(LAYERS)])
        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(HIDDEN) for _ in range(LAYERS)])
        self.pool = torch.nn.Linear(2 * HIDDEN, HIDDEN)
        self.head = torch.nn.Sequential(torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.SiLU(), torch.nn.Dropout(0.10), torch.nn.Linear(HIDDEN, 1))

    def forward(self, nodes: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor, edges: torch.Tensor, graph_rows: torch.Tensor, globals_: torch.Tensor, graph_count: int) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.node_input(nodes))
        edge_hidden = self.edge_input(edges)
        for message_layer, update_layer, norm in zip(self.message_layers, self.update_layers, self.norm_layers, strict=True):
            messages = message_layer(hidden[edge_src] + edge_hidden)
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, edge_dst, messages)
            hidden = norm(hidden + torch.nn.functional.silu(update_layer(torch.cat([hidden, aggregate], dim=1))))
        mean_pool = torch.zeros((graph_count, HIDDEN), dtype=hidden.dtype, device=hidden.device)
        mean_pool.index_add_(0, graph_rows, hidden)
        counts = torch.bincount(graph_rows, minlength=graph_count).clamp_min(1).to(hidden.dtype).unsqueeze(1)
        mean_pool = mean_pool / counts
        max_pool = torch.full((graph_count, HIDDEN), -torch.inf, dtype=hidden.dtype, device=hidden.device)
        expanded_rows = graph_rows.unsqueeze(1).expand(-1, HIDDEN)
        max_pool.scatter_reduce_(0, expanded_rows, hidden, reduce="amax", include_self=True)
        graph_hidden = torch.nn.functional.silu(self.pool(torch.cat([mean_pool, max_pool], dim=1)))
        combined = torch.nn.functional.silu(graph_hidden + self.global_input(globals_))
        return self.head(combined).squeeze(1)


def scaled_global(raw: np.ndarray, training_graphs: np.ndarray) -> np.ndarray:
    clean = np.asarray(raw, dtype=np.float32).copy()
    clean[~np.isfinite(clean) | (np.abs(clean) > 1.0e12)] = np.nan
    median = np.nanmedian(clean[training_graphs], axis=0)
    median[~np.isfinite(median)] = 0.0
    clean = np.where(np.isfinite(clean), clean, median[None, :])
    scale = np.nanstd(clean[training_graphs], axis=0)
    scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
    return ((clean - np.nanmean(clean[training_graphs], axis=0)[None, :]) / scale[None, :]).astype(np.float32)


def fit_model(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    globals_scaled: np.ndarray,
    graph_indices: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    seed: int,
) -> tuple[PeriodicFragmentEncoder, float, float, torch.device]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes, edge_src, edge_dst, edges, graph_rows = [value.to(device) for value in tensors]
    global_tensor = torch.from_numpy(globals_scaled).to(device)
    model = PeriodicFragmentEncoder(globals_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=2.0e-4)
    mean = float(np.mean(y[train_rows]))
    scale = max(float(np.std(y[train_rows])), 1.0e-6)
    expected = torch.from_numpy(((y[train_rows] - mean) / scale).astype(np.float32)).to(device)
    graph_index = torch.from_numpy(graph_indices[train_rows].astype(np.int64)).to(device)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        output = model(nodes, edge_src, edge_dst, edges, graph_rows, global_tensor, globals_scaled.shape[0])
        loss = torch.nn.functional.smooth_l1_loss(output[graph_index], expected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model, mean, scale, device


def predict_model(model: PeriodicFragmentEncoder, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], globals_scaled: np.ndarray, graph_count: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        nodes, edge_src, edge_dst, edges, graph_rows = [value.to(device) for value in tensors]
        output = model(nodes, edge_src, edge_dst, edges, graph_rows, torch.from_numpy(globals_scaled).to(device), graph_count)
    return output.detach().cpu().numpy()


def target_run(parent: dict[str, Any], target: str, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], globals_raw: np.ndarray, checkpoint_path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    info = dict(parent["target_info"][target])
    info["fingerprints"] = parent["fingerprints"]
    y = np.asarray(info["y"], dtype=float)
    parent_oof = np.asarray(info["parent"], dtype=float)
    groups = np.asarray(info["groups"], dtype=object)
    folds = c127.grouped_folds(groups)
    graph_indices = np.asarray(info["indices"], dtype=np.int64)
    direct_oof = np.full(len(y), np.nan, dtype=float)
    for fold in range(N_FOLDS):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        training_graphs = graph_indices[training]
        scaled = scaled_global(globals_raw, training_graphs)
        model, mean, scale, device = fit_model(tensors, scaled, graph_indices, y, training, SEED + 17 * (TARGETS.index(target) + 1) + fold)
        outputs = predict_model(model, tensors, scaled, len(parent["keys"]), device)
        direct_oof[validation] = mean + scale * outputs[graph_indices[validation]]
    arms = np.column_stack([parent_oof, direct_oof])
    weights, intercept, blend_name, _ = reference.blend_from_oof(y, arms)
    candidate = arms @ weights + intercept
    test_frame = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
    test_indices = np.asarray([parent["key_to_index"][value] for value in test_frame["canonical"]], dtype=np.int64)
    full_scaled = scaled_global(globals_raw, graph_indices)
    model, mean, scale, device = fit_model(tensors, full_scaled, graph_indices, y, np.arange(len(y), dtype=np.int64), SEED + 1000 + TARGETS.index(target))
    outputs = predict_model(model, tensors, full_scaled, len(parent["keys"]), device)
    direct_test = mean + scale * outputs[test_indices]
    test_parent = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id")["target"].to_numpy(float)
    test_candidate = np.column_stack([test_parent, direct_test]) @ weights + intercept
    report = c127.evaluate_target(info, {"candidate": candidate})
    report.update({"blend_name": blend_name, "blend_weights": [float(value) for value in weights], "blend_intercept": float(intercept), "direct_oof_r2": float(r2_score(y, direct_oof)), "feature_count": int(globals_raw.shape[1])})
    checkpoint(checkpoint_path, f"target_{target}_complete", target=target, delta_r2=report["delta_r2"], candidate_r2=report["candidate_r2"], pass_gate=report["pass"])
    return report, candidate, test_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")
    started = time.time()
    checkpoint_path = run_dir / "progress.jsonl"
    checkpoint(checkpoint_path, "started", experiment_id=run_dir.name, device="cuda" if torch.cuda.is_available() else "cpu")
    data_dir = (root / args.data_dir).resolve()
    parent = parent_builder.build_parent(root, data_dir)
    parity = c127.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")
    checkpoint(checkpoint_path, "parent_parity", **parity)
    nodes, edge_src, edge_dst, edges, graph_rows = graph_tensors(parent["molecules"])
    tensors = (nodes, edge_src, edge_dst, edges, graph_rows)
    grammar = parent_builder.grammar_features(parent["molecules"])
    endpoint, _ = fixed_features.fixed_features(parent["molecules"], list(range(len(parent["molecules"]))))
    globals_raw = np.hstack([grammar, endpoint]).astype(np.float32)
    checkpoint(checkpoint_path, "graph_constructed", graph_count=len(parent["keys"]), node_count=int(nodes.shape[0]), edge_count=int(edges.shape[0]), global_feature_count=int(globals_raw.shape[1]))

    target_reports: dict[str, Any] = {}
    candidate_oof: dict[str, np.ndarray] = {}
    candidate_test: dict[str, np.ndarray] = {}
    for target in ACTIVE_TARGETS:
        report, oof, test_prediction = target_run(parent, target, tensors, globals_raw, checkpoint_path)
        target_reports[target] = report
        candidate_oof[target] = oof
        candidate_test[target] = test_prediction
    for target in TARGETS:
        if target not in target_reports:
            info = parent["target_info"][target]
            y = np.asarray(info["y"], dtype=float)
            parent_values = np.asarray(info["parent"], dtype=float)
            target_reports[target] = {"parent_r2": float(r2_score(y, parent_values)), "candidate_r2": float(r2_score(y, parent_values)), "delta_r2": 0.0, "positive_folds": 0, "group_bootstrap_lower": 0.0, "minimum_panel_delta": 0.0, "panels": {"unchanged_parent": {"rows": int(len(y)), "delta_r2": 0.0, "status": "unchanged"}}, "folds": [], "pass": True, "unchanged_parent": True}
            candidate_oof[target] = parent_values
            target_rows = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
            candidate_test[target] = parent["test_parent_detail"].loc[parent["test_parent_detail"]["target_type"] == target].sort_values("id")["target"].to_numpy(float)
    banked = [target for target in ACTIVE_TARGETS if target_reports[target]["pass"]]
    oof_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        info = parent["target_info"][target]
        oof_parts.append(pd.DataFrame({"canonical": info["canonical"], "target_type": target, "target": info["y"], "parent": info["parent"], "candidate": candidate_oof[target], "assembled": candidate_oof[target] if target in banked else info["parent"], "group": info["groups"], "scaffold": info["scaffolds"], "outer_fold": c127.grouped_folds(np.asarray(info["groups"], dtype=object))}))
    oof = pd.concat(oof_parts, ignore_index=True)
    mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    mean_candidate = float(np.mean([r2_score(part["target"], part["assembled"]) for part in oof_parts]))
    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(mean_candidate >= mean_parent + 0.002 and max_loss >= -0.003 and len(banked) > 0)
    parent_detail = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    test_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        frame = parent["test"].loc[parent["test"]["target_type"] == target].sort_values("id").reset_index(drop=True)
        values = candidate_test[target] if target in banked else parent_detail.loc[parent_detail["target_type"] == target].sort_values("id")["target"].to_numpy(float)
        test_parts.append(pd.DataFrame({"id": frame["id"].astype(int), "target_type": target, "model_prediction": values}))
    raw = pd.concat(test_parts, ignore_index=True).sort_values("id")
    raw_labels, _ = reference.build_label_pool(parent["train"], parent["archive"])
    detail, override_report = reference.apply_official_overrides(raw, parent["test"], raw_labels)
    predictions = detail[["id", "target"]].copy()
    if len(predictions) != 4940 or not predictions["id"].equals(parent["test"]["id"]) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C128 complete output contract failed")
    source_paths = {
        "runner": Path(__file__),
        "parent_builder": root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py",
        "reference": root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py",
        "fragment_features": root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py",
        "endpoint_features": root / "Polymer Prediction Challenge Round 2/tools/round2_c063_egb_endpoint_conjugation_residual.py",
    }
    report = {
        "schema_version": "ppp.round2.c128.periodic-graph-fragment-absolute.run.v1",
        "experiment_id": run_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "parent": "C050 source rebuild; no C127 artifacts used",
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "pretrained_weights": False,
        "parent_replay_parity": parity,
        "architecture": {"node_features": 14, "edge_features": 10, "global_features": int(globals_raw.shape[1]), "hidden": HIDDEN, "message_layers": LAYERS, "epochs": EPOCHS, "periodic_edges": True, "fragment_global_side_channel": True, "absolute_target_heads": True, "device": "cuda" if torch.cuda.is_available() else "cpu"},
        "active_targets": list(ACTIVE_TARGETS),
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
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c128.periodic-graph-fragment-absolute.v1", "active_targets": list(ACTIVE_TARGETS), "seed": SEED, "hidden": HIDDEN, "layers": LAYERS, "epochs": EPOCHS, "no_c127_artifact_input": True, "local_eval_read": False})
    (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"torch={torch.__version__}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"cuda={torch.cuda.is_available()}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. Banked targets: `{','.join(banked) or 'none'}`. Mean parent `{mean_parent:.12f}`; assembled `{mean_candidate:.12f}`; gain `{mean_candidate - mean_parent:+.12f}`. No local_eval read.\n", encoding="utf-8")
    checkpoint(checkpoint_path, "metrics_written", decision=report["decision"], mean_candidate_r2=mean_candidate, banked_targets=banked)
    manifest: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file():
            manifest.append(f"{sha256_file(path)}  {path.name}")
    for name, path in source_paths.items():
        manifest.append(f"{sha256_file(path)}  SOURCE {name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "banked_targets": banked, "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
