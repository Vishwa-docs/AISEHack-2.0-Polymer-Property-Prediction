#!/usr/bin/env python3
"""Official-only scratch character-CNN residual screen for EPS/Nc."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import initial_reference_pipeline as reference


RDLogger.DisableLog("rdApp.*")
TARGETS = reference.TARGETS
CHANGED = ("eps", "nc")
SEED = 2026
MAX_LEN = 320
EMBED_DIM = 32
CHANNELS = 64
MLP_DIM = 64
DROPOUT = 0.10
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1.0e-4
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 32
RESIDUAL_WEIGHT = 0.20


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def no_stereo(value: object) -> str:
    molecule = Chem.MolFromSmiles(str(value).replace("[*]", "*"))
    if molecule is None:
        return str(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or np.isclose(np.var(y), 0.0):
        return float("nan")
    return float(r2_score(y, prediction))


def bootstrap_lower(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        if not np.isclose(np.var(y[rows]), 0.0):
            values.append(r2(y[rows], candidate[rows]) - r2(y[rows], parent[rows]))
    return float(np.quantile(values, 0.025)) if values else float("nan")


def nearest_similarity(fingerprints: list[Any], query: np.ndarray, train: np.ndarray) -> np.ndarray:
    train_fps = [fingerprints[int(index)] for index in train]
    result = np.zeros(len(query), dtype=np.float64)
    for row, index in enumerate(query):
        result[row] = max(DataStructs.BulkTanimotoSimilarity(fingerprints[int(index)], train_fps))
    return result


def encode_smiles(values: list[str], vocabulary: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.zeros((len(values), MAX_LEN), dtype=np.int64)
    lengths = np.zeros(len(values), dtype=np.int64)
    for row, value in enumerate(values):
        tokens = [vocabulary.get(char, 1) for char in str(value)[:MAX_LEN]]
        encoded[row, : len(tokens)] = tokens
        lengths[row] = len(tokens)
    return encoded, lengths


class CharCNN(torch.nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, EMBED_DIM, padding_idx=0)
        self.convolutions = torch.nn.ModuleList([
            torch.nn.Conv1d(EMBED_DIM, CHANNELS, kernel_size=kernel, padding=kernel // 2)
            for kernel in (3, 5, 7)
        ])
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(CHANNELS * 3, MLP_DIM),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(MLP_DIM, 1),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        pooled: list[torch.Tensor] = []
        for convolution in self.convolutions:
            activation = torch.relu(convolution(embedded))
            activation = activation.masked_fill(~mask[:, None, :], -1.0e4)
            pooled.append(torch.amax(activation, dim=2))
        return self.head(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


def predict_model(model: CharCNN, tokens: np.ndarray, lengths: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(tokens).to(device),
            torch.from_numpy(lengths).to(device),
        )
    return output.detach().cpu().numpy().astype(np.float64)


def train_cnn(
    train_tokens: np.ndarray,
    train_lengths: np.ndarray,
    train_values: np.ndarray,
    validation_tokens: np.ndarray | None,
    validation_lengths: np.ndarray | None,
    validation_values: np.ndarray | None,
    vocabulary_size: int,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[CharCNN, float, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharCNN(vocabulary_size).to(device)
    scale = max(float(np.std(train_values)), 1.0e-8)
    normalized = torch.from_numpy((train_values / scale).astype(np.float32)).to(device)
    tokens = torch.from_numpy(train_tokens).to(device)
    lengths = torch.from_numpy(train_lengths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = torch.nn.SmoothL1Loss()
    epochs = fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_values), device=device)
        for start in range(0, len(train_values), BATCH_SIZE):
            selected = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tokens[selected], lengths[selected])
            loss = loss_function(prediction, normalized[selected])
            loss.backward()
            optimizer.step()
        if validation_tokens is None or validation_lengths is None or validation_values is None:
            continue
        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                torch.from_numpy(validation_tokens).to(device),
                torch.from_numpy(validation_lengths).to(device),
            )
            validation_loss = float(loss_function(validation_prediction, torch.from_numpy((validation_values / scale).astype(np.float32)).to(device)).item())
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_epoch == 0:
        best_epoch = epochs
        best_loss = float("nan")
    return model, scale, best_epoch


def panel_report(y: np.ndarray, candidate: np.ndarray, parent: np.ndarray, groups: np.ndarray, similarity: np.ndarray, scaffolds: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    deltas: list[float] = []
    for name, selected in {
        "similarity_lt_0.30": similarity < 0.30,
        "similarity_0.30_0.50": (similarity >= 0.30) & (similarity < 0.50),
        "similarity_0.50_0.70": (similarity >= 0.50) & (similarity < 0.70),
        "similarity_ge_0.70": similarity >= 0.70,
    }.items():
        rows = int(np.sum(selected))
        group_count = int(np.unique(groups[selected]).size)
        eligible = rows >= 20 and group_count >= 5
        delta = r2(y[selected], candidate[selected]) - r2(y[selected], parent[selected]) if eligible else None
        report[name] = {"rows": rows, "groups": group_count, "eligible": bool(eligible), "delta_r2": delta}
        if delta is not None:
            deltas.append(delta)
    scaffold_values: list[float] = []
    for scaffold in sorted(set(scaffolds)):
        selected = scaffolds == scaffold
        if int(np.sum(selected)) >= 10 and int(np.unique(groups[selected]).size) >= 3 and not np.isclose(np.var(y[selected]), 0.0):
            scaffold_values.append(r2(y[selected], candidate[selected]) - r2(y[selected], parent[selected]))
    report["scaffold_groups_ge_3"] = {"evaluated": len(scaffold_values), "minimum_delta_r2": min(scaffold_values) if scaffold_values else None}
    if scaffold_values:
        deltas.append(min(scaffold_values))
    report["minimum_panel_delta"] = min(deltas) if deltas else 0.0
    return report


def masked_cross_arrays(pooled: pd.DataFrame, keys: list[str], excluded: set[str]) -> tuple[np.ndarray, np.ndarray]:
    if not excluded:
        return reference.cross_property_arrays(pooled, keys)
    groups = pooled["canonical"].map(no_stereo)
    return reference.cross_property_arrays(pooled.loc[~groups.isin(excluded)], keys)


def parent_oof_for_split(
    target: str,
    pooled: pd.DataFrame,
    keys: list[str],
    key_to_index: dict[str, int],
    base_dense: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    rows: np.ndarray,
    splitter: Any,
    groups: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> np.ndarray:
    frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
    y = frame["target"].to_numpy(dtype=float)
    canonical = frame["canonical"].astype(str).to_numpy(dtype=object)
    group_values = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[global_indices] = y
    output = np.full(len(rows), np.nan, dtype=np.float64)
    for local_train, local_validation in splitter.split(rows, groups=groups if isinstance(splitter, GroupKFold) else None):
        train_rows = rows[np.asarray(local_train, dtype=np.int64)]
        valid_rows = rows[np.asarray(local_validation, dtype=np.int64)]
        excluded = set(group_values[valid_rows].tolist())
        values, available = masked_cross_arrays(pooled, keys, excluded)
        dense = reference.target_dense_features(base_dense, values, available, target)
        arms = reference.predict_base_models(
            dense,
            sparse_parts,
            fingerprints,
            y_global,
            global_indices[train_rows],
            global_indices[valid_rows],
            reference.DEFAULT_CONFIG,
            target,
        )
        output[np.asarray(local_validation, dtype=np.int64)] = arms @ weights + intercept
    return output


def evaluate_target(
    target: str,
    pooled: pd.DataFrame,
    test: pd.DataFrame,
    keys: list[str],
    key_to_index: dict[str, int],
    base_dense: np.ndarray,
    sparse_parts: list[Any],
    fingerprints: list[Any],
    smiles_tokens: np.ndarray,
    smiles_lengths: np.ndarray,
    vocabulary_size: int,
    weights: np.ndarray,
    intercept: float,
    parent_test: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
    y = frame["target"].to_numpy(dtype=float)
    canonical = frame["canonical"].astype(str).to_numpy(dtype=object)
    groups = np.asarray([no_stereo(value) for value in canonical], dtype=object)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    y_global = np.full(len(keys), np.nan, dtype=np.float64)
    y_global[global_indices] = y
    folds = GroupKFold(n_splits=5)
    parent_oof = np.full(len(y), np.nan, dtype=np.float64)
    candidate_oof = np.full(len(y), np.nan, dtype=np.float64)
    similarity_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for fold, (outer_train, validation) in enumerate(folds.split(np.arange(len(y)), groups=groups)):
        outer_train = np.asarray(outer_train, dtype=np.int64)
        validation = np.asarray(validation, dtype=np.int64)
        excluded_outer = set(groups[validation].tolist())
        outer_values, outer_available = masked_cross_arrays(pooled, keys, excluded_outer)
        outer_dense = reference.target_dense_features(base_dense, outer_values, outer_available, target)
        inner_splitter = GroupKFold(n_splits=4)
        inner_parent = np.full(len(outer_train), np.nan, dtype=np.float64)
        inner_iterator = list(inner_splitter.split(outer_train, groups=groups[outer_train]))
        for inner_local_train, inner_local_validation in inner_iterator:
            inner_train = outer_train[np.asarray(inner_local_train, dtype=np.int64)]
            inner_validation = outer_train[np.asarray(inner_local_validation, dtype=np.int64)]
            excluded_inner = excluded_outer | set(groups[inner_validation].tolist())
            inner_values, inner_available = masked_cross_arrays(pooled, keys, excluded_inner)
            inner_dense = reference.target_dense_features(base_dense, inner_values, inner_available, target)
            arms = reference.predict_base_models(
                inner_dense,
                sparse_parts,
                fingerprints,
                y_global,
                global_indices[inner_train],
                global_indices[inner_validation],
                reference.DEFAULT_CONFIG,
                target,
            )
            inner_parent[np.asarray(inner_local_validation, dtype=np.int64)] = arms @ weights + intercept
        if not np.isfinite(inner_parent).all():
            raise RuntimeError(f"inner parent incomplete for {target} fold {fold}")
        best_epochs: list[int] = []
        for inner_number, (inner_local_train, inner_local_validation) in enumerate(inner_iterator):
            train_rows = outer_train[np.asarray(inner_local_train, dtype=np.int64)]
            validation_rows = outer_train[np.asarray(inner_local_validation, dtype=np.int64)]
            residual_train = y[train_rows] - inner_parent[np.asarray(inner_local_train, dtype=np.int64)]
            residual_validation = y[validation_rows] - inner_parent[np.asarray(inner_local_validation, dtype=np.int64)]
            _, _, best_epoch = train_cnn(
                smiles_tokens[global_indices[train_rows]],
                smiles_lengths[global_indices[train_rows]],
                residual_train,
                smiles_tokens[global_indices[validation_rows]],
                smiles_lengths[global_indices[validation_rows]],
                residual_validation,
                vocabulary_size,
                SEED + fold * 100 + inner_number,
            )
            best_epochs.append(best_epoch)
        selected_epochs = int(np.clip(round(float(np.median(best_epochs))), 8, MAX_EPOCHS))
        residual_outer = y[outer_train] - inner_parent
        model, scale, _ = train_cnn(
            smiles_tokens[global_indices[outer_train]],
            smiles_lengths[global_indices[outer_train]],
            residual_outer,
            None,
            None,
            None,
            vocabulary_size,
            SEED + 1000 + fold,
            fixed_epochs=selected_epochs,
        )
        outer_residual = scale * predict_model(model, smiles_tokens[global_indices[validation]], smiles_lengths[global_indices[validation]], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        outer_arms = reference.predict_base_models(
            outer_dense,
            sparse_parts,
            fingerprints,
            y_global,
            global_indices[outer_train],
            global_indices[validation],
            reference.DEFAULT_CONFIG,
            target,
        )
        outer_parent = outer_arms @ weights + intercept
        outer_candidate = outer_parent + RESIDUAL_WEIGHT * outer_residual
        sim = nearest_similarity(fingerprints, global_indices[validation], global_indices[outer_train])
        parent_oof[validation] = outer_parent
        candidate_oof[validation] = outer_candidate
        similarity_oof[validation] = sim
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "parent_r2": r2(y[validation], outer_parent),
            "candidate_r2": r2(y[validation], outer_candidate),
            "delta_r2": r2(y[validation], outer_candidate) - r2(y[validation], outer_parent),
            "selected_epochs": selected_epochs,
            "inner_best_epochs": best_epochs,
            "outer_group_count": int(np.unique(groups[validation]).size),
        })
    scaffolds = np.asarray([
        MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=False) or "ACYCLIC"
        for value in canonical
    ], dtype=object)
    panels = panel_report(y, candidate_oof, parent_oof, groups, similarity_oof, scaffolds)
    report = {
        "rows": int(len(y)),
        "group_count": int(np.unique(groups).size),
        "parent_r2": r2(y, parent_oof),
        "candidate_r2": r2(y, candidate_oof),
        "delta_r2": r2(y, candidate_oof) - r2(y, parent_oof),
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "group_bootstrap_lower": bootstrap_lower(y, candidate_oof, parent_oof, groups),
        "folds": fold_rows,
        "panels": panels,
    }
    test_frame = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
    target_global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    full_splitter = GroupKFold(n_splits=5)
    full_parent_oof = parent_oof_for_split(
        target,
        pooled,
        keys,
        key_to_index,
        base_dense,
        sparse_parts,
        fingerprints,
        np.arange(len(y), dtype=np.int64),
        full_splitter,
        groups,
        weights,
        intercept,
    )
    full_inner = GroupKFold(n_splits=4)
    best_epochs: list[int] = []
    for inner_number, (inner_train, inner_validation) in enumerate(full_inner.split(np.arange(len(y)), groups=groups)):
        residual_train = y[inner_train] - full_parent_oof[inner_train]
        residual_validation = y[inner_validation] - full_parent_oof[inner_validation]
        _, _, best_epoch = train_cnn(
            smiles_tokens[target_global_indices[inner_train]],
            smiles_lengths[target_global_indices[inner_train]],
            residual_train,
            smiles_tokens[target_global_indices[inner_validation]],
            smiles_lengths[target_global_indices[inner_validation]],
            residual_validation,
            vocabulary_size,
            SEED + 2000 + inner_number,
        )
        best_epochs.append(best_epoch)
    selected_epochs = int(np.clip(round(float(np.median(best_epochs))), 8, MAX_EPOCHS))
    final_model, final_scale, _ = train_cnn(
        smiles_tokens[target_global_indices],
        smiles_lengths[target_global_indices],
        y - full_parent_oof,
        None,
        None,
        None,
        vocabulary_size,
        SEED + 3000,
        fixed_epochs=selected_epochs,
    )
    test_indices = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
    test_residual = final_scale * predict_model(final_model, smiles_tokens[test_indices], smiles_lengths[test_indices], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    test_candidate = parent_test + RESIDUAL_WEIGHT * test_residual
    test_output = pd.DataFrame({
        "id": test_frame["id"].astype(int),
        "target_type": target,
        "parent_prediction": parent_test.astype(float),
        "residual_prediction": test_residual.astype(float),
        "candidate_prediction": test_candidate.astype(float),
        "selected_epochs": selected_epochs,
    })
    report["test_rows"] = int(len(test_output))
    report["test_selected_epochs"] = selected_epochs
    report["test_residual_scale"] = float(final_scale)
    return report, test_output, pd.DataFrame({
        "canonical": canonical,
        "target_type": target,
        "target": y,
        "parent": parent_oof,
        "candidate": candidate_oof,
        "similarity": similarity_oof,
        "group": groups,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} - {"protocol.json"}:
        raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time()
    data_dir = (root / args.data_dir).resolve()
    train, test, archive, inputs = reference.load_inputs(data_dir)
    _, pooled = reference.build_label_pool(train, archive)
    keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
    key_to_index = {key: index for index, key in enumerate(keys)}
    molecules = reference.build_molecules(keys)
    descriptor, _ = reference.descriptor_matrix(molecules)
    physical, _ = reference.physical_matrix(molecules, keys)
    base_dense = np.hstack([descriptor, physical]).astype(np.float64, copy=False)
    cross_values, cross_available = reference.cross_property_arrays(pooled, keys)
    sparse_parts = [
        reference.morgan_count_matrix(molecules, radius=2, bits=4096),
        reference.morgan_count_matrix(molecules, radius=3, bits=4096),
        reference.text_matrix(keys, 65536),
    ]
    fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
    chars = sorted({char for value in keys for char in str(value)})
    vocabulary = {char: index + 2 for index, char in enumerate(chars)}
    smiles_tokens, smiles_lengths = encode_smiles(keys, vocabulary)
    detail, parent_oof_frame, parent_report = reference.fit_targets(
        pooled,
        test,
        keys,
        base_dense,
        cross_values,
        cross_available,
        sparse_parts,
        fingerprints,
        reference.DEFAULT_CONFIG,
    )
    reports: dict[str, Any] = {}
    test_outputs: list[pd.DataFrame] = []
    oof_outputs: list[pd.DataFrame] = []
    for target in CHANGED:
        parent_test_frame = detail[detail["target_type"] == target].sort_values("id").reset_index(drop=True)
        parent_test = parent_test_frame["model_prediction"].to_numpy(dtype=float)
        weights = np.asarray([parent_report["target_reports"][target]["blend_weights"][name] for name in ("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local")], dtype=float)
        intercept = float(parent_report["target_reports"][target]["blend_intercept"])
        report, test_output, oof_output = evaluate_target(
            target,
            pooled,
            test,
            keys,
            key_to_index,
            base_dense,
            sparse_parts,
            fingerprints,
            smiles_tokens,
            smiles_lengths,
            len(vocabulary) + 2,
            weights,
            intercept,
            parent_test,
        )
        report["parent_weights"] = {name: float(value) for name, value in zip(("sparse_ridge", "dense_ridge", "extra_trees", "tanimoto_local"), weights, strict=True)}
        report["parent_intercept"] = intercept
        reports[target] = report
        test_outputs.append(test_output)
        oof_outputs.append(oof_output)
    component = pd.concat(test_outputs, ignore_index=True).sort_values("id").reset_index(drop=True)
    if len(component) != 306 or component["id"].duplicated().any() or not np.isfinite(component["candidate_prediction"].to_numpy(dtype=float)).all():
        raise RuntimeError("component output contract failed")
    oof = pd.concat(oof_outputs, ignore_index=True)
    component.to_csv(run_dir / "eps_nc_component_predictions.csv", index=False)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {
        target: {
            "gain_pass": bool(reports[target]["delta_r2"] >= 0.010),
            "fold_pass": bool(reports[target]["positive_folds"] >= 4),
            "bootstrap_pass": bool(reports[target]["group_bootstrap_lower"] > 0.0),
            "panel_pass": bool(reports[target]["panels"]["minimum_panel_delta"] >= 0.0),
            "rows_finite": True,
        }
        for target in CHANGED
    }
    passed = bool(all(all(values.values()) for values in gates.values()))
    mean_parent = float(np.mean([reports[target]["parent_r2"] for target in CHANGED]))
    mean_candidate = float(np.mean([reports[target]["candidate_r2"] for target in CHANGED]))
    report = {
        "schema_version": "ppp.round2.c057.scratch-char-cnn-residual.v1",
        "experiment_id": run_dir.name,
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; parent arms regenerated from official inputs",
        "created_at": datetime.now().astimezone().isoformat(),
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "changed_targets": list(CHANGED),
        "targets": reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "gates": gates,
        "decision": "pass_component_gate" if passed else "rejected_component_gate",
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(oof))},
        "architecture": {"embedding_dim": EMBED_DIM, "kernels": [3, 5, 7], "channels": CHANNELS, "mlp_dim": MLP_DIM, "dropout": DROPOUT, "masked_pooling": True},
        "training": {"optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE, "residual_weight": RESIDUAL_WEIGHT, "seed": SEED},
        "source_sha256": sha256_file(root / "tools" / "round2_c057_scratch_char_cnn_residual.py"),
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "architecture": report["architecture"], "training": report["training"], "changed_targets": list(CHANGED), "outer_folds": "GroupKFold(n_splits=5, groups=canonical_no_stereo)", "inner_folds": "GroupKFold(n_splits=4) for CNN early stopping and parent residuals", "external_label_file_read": False, "pretrained_weights": False, "prior_prediction_input": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\ntorch={torch.__version__}\nnumpy={np.__version__}\npandas={pd.__version__}\ncuda={torch.cuda.is_available()}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# C057 decision\n\nParent mean: {mean_parent:.12f}\nCandidate mean: {mean_candidate:.12f}\nGain: {mean_candidate - mean_parent:+.12f}\n\nDecision: {'PASS COMPONENT GATE' if passed else 'REJECT COMPONENT GATE'}\n\nScratch CNN weights were initialized randomly in-process and were not persisted. No external_label file, local_eval value, pretrained asset, cross-property label, or prior prediction artifact was read.\n", encoding="utf-8")
    manifest = [run_dir / name for name in ("protocol.json", "config.json", "metrics.json", "eps_nc_component_predictions.csv", "oof_predictions.csv", "environment.txt", "command.txt", "decision.md")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "targets": {target: reports[target]["delta_r2"] for target in CHANGED}}, sort_keys=True))


if __name__ == "__main__":
    main()
