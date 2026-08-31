#!/usr/bin/env python3
"""Corrected exact-v7-parent scratch character-CNN residual screen."""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import GroupKFold, KFold

import initial_reference_pipeline as reference
import round2_c057_scratch_char_cnn_residual as cnn


TARGETS = reference.TARGETS
CHANGED = ("eps", "nc")
SEED = 2026
RESIDUAL_WEIGHT = cnn.RESIDUAL_WEIGHT


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return cnn.r2(y, prediction)


def train_mode(
    target: str,
    frame: pd.DataFrame,
    parent_oof: np.ndarray,
    keys: list[str],
    key_to_index: dict[str, int],
    tokens: np.ndarray,
    lengths: np.ndarray,
    vocabulary_size: int,
    fingerprints: list[Any],
    splitter: Any,
    seed_offset: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = frame["target"].to_numpy(dtype=float)
    canonical = frame["canonical"].astype(str).to_numpy(dtype=object)
    groups = np.asarray([cnn.no_stereo(value) for value in canonical], dtype=object)
    global_indices = np.asarray([key_to_index[value] for value in canonical], dtype=np.int64)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    similarity = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (outer_train, validation) in enumerate(splitter.split(np.arange(len(y)), groups=groups if isinstance(splitter, GroupKFold) else None)):
        outer_train = np.asarray(outer_train, dtype=np.int64)
        validation = np.asarray(validation, dtype=np.int64)
        inner_splitter = GroupKFold(n_splits=4)
        inner_iterator = list(inner_splitter.split(outer_train, groups=groups[outer_train]))
        best_epochs: list[int] = []
        residual = y - parent_oof
        for inner_number, (inner_local_train, inner_local_validation) in enumerate(inner_iterator):
            train_rows = outer_train[np.asarray(inner_local_train, dtype=np.int64)]
            validation_rows = outer_train[np.asarray(inner_local_validation, dtype=np.int64)]
            _, _, best_epoch = cnn.train_cnn(
                tokens[global_indices[train_rows]],
                lengths[global_indices[train_rows]],
                residual[train_rows],
                tokens[global_indices[validation_rows]],
                lengths[global_indices[validation_rows]],
                residual[validation_rows],
                vocabulary_size,
                SEED + seed_offset + fold * 100 + inner_number,
            )
            best_epochs.append(best_epoch)
        selected_epochs = int(np.clip(round(float(np.median(best_epochs))), 8, cnn.MAX_EPOCHS))
        model, scale, _ = cnn.train_cnn(
            tokens[global_indices[outer_train]],
            lengths[global_indices[outer_train]],
            residual[outer_train],
            None,
            None,
            None,
            vocabulary_size,
            SEED + seed_offset + 1000 + fold,
            fixed_epochs=selected_epochs,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicted_residual = scale * cnn.predict_model(model, tokens[global_indices[validation]], lengths[global_indices[validation]], device)
        candidate[validation] = parent_oof[validation] + RESIDUAL_WEIGHT * predicted_residual
        similarity[validation] = cnn.nearest_similarity(fingerprints, global_indices[validation], global_indices[outer_train])
        fold_rows.append({
            "fold": fold,
            "rows": int(len(validation)),
            "parent_r2": r2(y[validation], parent_oof[validation]),
            "candidate_r2": r2(y[validation], candidate[validation]),
            "delta_r2": r2(y[validation], candidate[validation]) - r2(y[validation], parent_oof[validation]),
            "selected_epochs": selected_epochs,
            "inner_best_epochs": best_epochs,
            "group_count": int(np.unique(groups[validation]).size),
        })
    scaffolds = np.asarray([
        cnn.MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=False) or "ACYCLIC"
        for value in canonical
    ], dtype=object)
    panel = cnn.panel_report(y, candidate, parent_oof, groups, similarity, scaffolds)
    report = {
        "rows": int(len(y)),
        "group_count": int(np.unique(groups).size),
        "parent_r2": r2(y, parent_oof),
        "candidate_r2": r2(y, candidate),
        "delta_r2": r2(y, candidate) - r2(y, parent_oof),
        "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)),
        "group_bootstrap_lower": cnn.bootstrap_lower(y, candidate, parent_oof, groups),
        "folds": fold_rows,
        "panels": panel,
    }
    return report, candidate, similarity, groups, np.asarray(scaffolds, dtype=object)


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
    vocabulary = {char: index + 2 for index, char in enumerate(sorted({char for value in keys for char in str(value)}))}
    tokens, lengths = cnn.encode_smiles(keys, vocabulary)
    detail, parent_oof_frame, parent_report = reference.fit_targets(
        pooled, test, keys, base_dense, cross_values, cross_available, sparse_parts, fingerprints, reference.DEFAULT_CONFIG
    )
    reports: dict[str, Any] = {}
    test_outputs: list[pd.DataFrame] = []
    oof_outputs: list[pd.DataFrame] = []
    for target in CHANGED:
        frame = pooled[pooled["target_type"] == target].reset_index(drop=True)
        parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == target].reset_index(drop=True)
        if not np.array_equal(frame["canonical"].astype(str).to_numpy(), parent_rows["canonical"].astype(str).to_numpy()):
            raise RuntimeError(f"parent row alignment failed for {target}")
        parent_oof = parent_rows["prediction"].to_numpy(dtype=float)
        kfold_report, kfold_candidate, kfold_similarity, groups, scaffolds = train_mode(
            target, frame, parent_oof, keys, key_to_index, tokens, lengths, len(vocabulary) + 2, fingerprints,
            KFold(n_splits=5, shuffle=True, random_state=SEED), 0,
        )
        strict_report, strict_candidate, strict_similarity, strict_groups, _ = train_mode(
            target, frame, parent_oof, keys, key_to_index, tokens, lengths, len(vocabulary) + 2, fingerprints,
            GroupKFold(n_splits=5), 10000,
        )
        target_train = frame["target"].to_numpy(dtype=float)
        residual = target_train - parent_oof
        full_splitter = GroupKFold(n_splits=4)
        full_best: list[int] = []
        global_indices = np.asarray([key_to_index[value] for value in frame["canonical"]], dtype=np.int64)
        full_groups = np.asarray([cnn.no_stereo(value) for value in frame["canonical"].astype(str)], dtype=object)
        for inner_number, (inner_train, inner_validation) in enumerate(full_splitter.split(np.arange(len(frame)), groups=full_groups)):
            _, _, best_epoch = cnn.train_cnn(
                tokens[global_indices[inner_train]], lengths[global_indices[inner_train]], residual[inner_train],
                tokens[global_indices[inner_validation]], lengths[global_indices[inner_validation]], residual[inner_validation],
                len(vocabulary) + 2, SEED + 20000 + inner_number,
            )
            full_best.append(best_epoch)
        selected_epochs = int(np.clip(round(float(np.median(full_best))), 8, cnn.MAX_EPOCHS))
        final_model, final_scale, _ = cnn.train_cnn(
            tokens[global_indices], lengths[global_indices], residual, None, None, None,
            len(vocabulary) + 2, SEED + 30000, fixed_epochs=selected_epochs,
        )
        test_frame = test[test["target_type"] == target].sort_values("id").reset_index(drop=True)
        test_indices = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
        # `fit_targets` returned the exact parent test arm above; it was
        # regenerated in this process and never loaded from a prior file.
        parent_test = detail[detail["target_type"] == target].sort_values("id")["model_prediction"].to_numpy(dtype=float)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_residual = final_scale * cnn.predict_model(final_model, tokens[test_indices], lengths[test_indices], device)
        candidate_test = parent_test + RESIDUAL_WEIGHT * test_residual
        test_outputs.append(pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": target, "parent_prediction": parent_test, "residual_prediction": test_residual, "candidate_prediction": candidate_test, "selected_epochs": selected_epochs}))
        oof_outputs.append(pd.DataFrame({"canonical": frame["canonical"].astype(str), "target_type": target, "target": target_train, "parent": parent_oof, "candidate_kfold": kfold_candidate, "similarity_kfold": kfold_similarity, "candidate_strict": strict_candidate, "similarity_strict": strict_similarity, "group": strict_groups}))
        reports[target] = {"exact_v7_kfold": kfold_report, "strict_group": strict_report, "parent_weights": parent_report["target_reports"][target]["blend_weights"], "parent_intercept": parent_report["target_reports"][target]["blend_intercept"], "test_rows": int(len(test_frame)), "test_selected_epochs": selected_epochs, "test_residual_scale": float(final_scale)}
    component = pd.concat(test_outputs, ignore_index=True).sort_values("id").reset_index(drop=True)
    if len(component) != 306 or component["id"].duplicated().any() or not np.isfinite(component["candidate_prediction"].to_numpy(dtype=float)).all():
        raise RuntimeError("component output contract failed")
    oof = pd.concat(oof_outputs, ignore_index=True)
    component.to_csv(run_dir / "eps_nc_component_predictions.csv", index=False)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    gates = {
        target: {
            "gain_pass": bool(reports[target]["exact_v7_kfold"]["delta_r2"] >= 0.010),
            "fold_pass": bool(reports[target]["exact_v7_kfold"]["positive_folds"] >= 4),
            "bootstrap_pass": bool(reports[target]["exact_v7_kfold"]["group_bootstrap_lower"] > 0.0),
            "panel_pass": bool(reports[target]["exact_v7_kfold"]["panels"]["minimum_panel_delta"] >= 0.0),
            "strict_no_regression": bool(reports[target]["strict_group"]["delta_r2"] >= -0.003),
        }
        for target in CHANGED
    }
    passed = bool(all(all(values.values()) for values in gates.values()))
    mean_parent = float(np.mean([reports[target]["exact_v7_kfold"]["parent_r2"] for target in CHANGED]))
    mean_candidate = float(np.mean([reports[target]["exact_v7_kfold"]["candidate_r2"] for target in CHANGED]))
    report = {
        "schema_version": "ppp.round2.c058.exact-v7-char-cnn.v1",
        "experiment_id": run_dir.name,
        "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 OOF parent regenerated in-process",
        "created_at": datetime.now().astimezone().isoformat(),
        "official_inputs": inputs,
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "prior_prediction_input": False,
        "changed_targets": list(CHANGED),
        "targets": reports,
        "mean_parent_r2": mean_parent,
        "mean_candidate_r2": mean_candidate,
        "mean_gain": mean_candidate - mean_parent,
        "gates": gates,
        "decision": "pass_component_gate" if passed else "rejected_component_gate",
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(oof))},
        "source_sha256": sha256_file(root / "tools" / "round2_c058_exact_v7_char_cnn.py"),
        "dependency_sha256": {"tools/round2_c057_scratch_char_cnn_residual.py": sha256_file(root / "tools" / "round2_c057_scratch_char_cnn_residual.py"), "tools/initial_reference_pipeline.py": sha256_file(root / "tools" / "initial_reference_pipeline.py")},
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "changed_targets": list(CHANGED), "parent": "exact regenerated v7 OOF weights and predictions", "cnn_dependency": "scratch character CNN from C057 source, fresh in-memory weights", "residual_weight": RESIDUAL_WEIGHT, "primary_outer": "KFold(n_splits=5, shuffle=true, random_state=2026) exact v7 OOF comparison", "secondary_outer": "GroupKFold(n_splits=5) strict diagnostic", "inner": "GroupKFold(n_splits=4) early stopping", "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\ntorch={torch.__version__}\nnumpy={np.__version__}\npandas={pd.__version__}\ncuda={torch.cuda.is_available()}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# C058 decision\n\nExact-v7 parent mean: {mean_parent:.12f}\nExact-v7 candidate mean: {mean_candidate:.12f}\nGain: {mean_candidate - mean_parent:+.12f}\n\nDecision: {'PASS COMPONENT GATE' if passed else 'REJECT COMPONENT GATE'}\n\nThis correction uses the exact regenerated v7 OOF parent for residual training and evaluation. No external_label/local_eval file, prior prediction, pretrained weight, or cross-property target was read.\n", encoding="utf-8")
    manifest = [run_dir / name for name in ("protocol.json", "config.json", "metrics.json", "eps_nc_component_predictions.csv", "oof_predictions.csv", "environment.txt", "command.txt", "decision.md")]
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "mean_parent_r2": mean_parent, "mean_candidate_r2": mean_candidate, "mean_gain": mean_candidate - mean_parent, "targets": {target: reports[target]["exact_v7_kfold"]["delta_r2"] for target in CHANGED}}, sort_keys=True))


if __name__ == "__main__":
    main()
