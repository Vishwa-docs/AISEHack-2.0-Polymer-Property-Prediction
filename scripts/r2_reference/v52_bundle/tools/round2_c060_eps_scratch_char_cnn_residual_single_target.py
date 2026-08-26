#!/usr/bin/env python3
"""Single-target EPS scratch character-CNN residual screen against exact v7."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold

import initial_reference_pipeline as reference
import round2_c057_scratch_char_cnn_residual as cnn
import round2_c058_exact_v7_char_cnn as exact_cnn


TARGET = "eps"
SEED = 2026
RESIDUAL_WEIGHT = cnn.RESIDUAL_WEIGHT


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
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
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True)
    parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET].reset_index(drop=True)
    if not np.array_equal(frame["canonical"].astype(str).to_numpy(), parent_rows["canonical"].astype(str).to_numpy()):
        raise RuntimeError("exact v7 EPS parent row alignment failed")
    parent_oof = parent_rows["prediction"].to_numpy(dtype=float)
    primary, candidate, similarity, groups, scaffolds = exact_cnn.train_mode(
        TARGET, frame, parent_oof, keys, key_to_index, tokens, lengths, len(vocabulary) + 2, fingerprints,
        KFold(n_splits=5, shuffle=True, random_state=SEED), 0,
    )
    strict, strict_candidate, _, strict_groups, _ = exact_cnn.train_mode(
        TARGET, frame, parent_oof, keys, key_to_index, tokens, lengths, len(vocabulary) + 2, fingerprints,
        GroupKFold(n_splits=5), 10000,
    )
    y = frame["target"].to_numpy(dtype=float)
    if not np.isfinite(candidate).all():
        raise RuntimeError("non-finite EPS OOF prediction")
    test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True)
    global_indices = np.asarray([key_to_index[value] for value in frame["canonical"]], dtype=np.int64)
    frame_groups = np.asarray([cnn.no_stereo(value) for value in frame["canonical"].astype(str)], dtype=object)
    residual = y - parent_oof
    inner = GroupKFold(n_splits=4)
    best_epochs: list[int] = []
    for inner_number, (inner_train, inner_validation) in enumerate(inner.split(np.arange(len(frame)), groups=frame_groups)):
        _, _, best_epoch = cnn.train_cnn(
            tokens[global_indices[inner_train]], lengths[global_indices[inner_train]], residual[inner_train],
            tokens[global_indices[inner_validation]], lengths[global_indices[inner_validation]], residual[inner_validation],
            len(vocabulary) + 2, SEED + 20000 + inner_number,
        )
        best_epochs.append(best_epoch)
    selected_epochs = int(np.clip(round(float(np.median(best_epochs))), 8, cnn.MAX_EPOCHS))
    final_model, final_scale, _ = cnn.train_cnn(
        tokens[global_indices], lengths[global_indices], residual, None, None, None,
        len(vocabulary) + 2, SEED + 30000, fixed_epochs=selected_epochs,
    )
    test_indices = np.asarray([key_to_index[value] for value in test_frame["canonical"]], dtype=np.int64)
    parent_test = detail[detail["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(dtype=float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_residual = final_scale * cnn.predict_model(final_model, tokens[test_indices], lengths[test_indices], device)
    candidate_test = parent_test + RESIDUAL_WEIGHT * test_residual
    component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": parent_test, "residual_prediction": test_residual, "candidate_prediction": candidate_test, "selected_epochs": selected_epochs})
    if len(component) != 153 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all():
        raise RuntimeError("EPS component output contract failed")
    component.to_csv(run_dir / "eps_component_predictions.csv", index=False)
    pd.DataFrame({"canonical": frame["canonical"].astype(str), "target": y, "parent": parent_oof, "candidate": candidate, "group": groups, "scaffold": scaffolds, "outer_fold": np.nan, "nearest_similarity": similarity}).to_csv(run_dir / "oof_predictions.csv", index=False)
    parent_r2 = float(primary["parent_r2"])
    candidate_r2 = float(primary["candidate_r2"])
    delta = float(primary["delta_r2"])
    bootstrap_lower = float(primary["group_bootstrap_lower"])
    minimum_panel = float(primary["panels"]["minimum_panel_delta"])
    gates = {"gain_pass": delta >= 0.01, "fold_pass": primary["positive_folds"] >= 4, "bootstrap_pass": bootstrap_lower > 0.0, "panel_pass": minimum_panel >= 0.0, "strict_no_regression": strict["delta_r2"] >= -0.003, "component_rows_pass": len(component) == 153}
    passed = bool(all(gates.values()))
    source_names = ("round2_c060_eps_scratch_char_cnn_residual_single_target.py", "round2_c058_exact_v7_char_cnn.py", "round2_c057_scratch_char_cnn_residual.py", "initial_reference_pipeline.py")
    report = {
        "schema_version": "ppp.round2.c060.eps-scratch-char-cnn-residual.v1", "experiment_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 EPS regenerated in-process",
        "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False,
        "target": TARGET, "residual_weight": RESIDUAL_WEIGHT, "primary": primary, "strict_group": strict,
        "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": primary["positive_folds"], "group_bootstrap_lower": bootstrap_lower, "minimum_panel_delta": minimum_panel,
        "gates": gates, "decision": "pass_component_gate" if passed else "rejected_component_gate",
        "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))},
        "source_hashes": {name: sha256_file(root / "tools" / name) for name in source_names}, "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "metrics.json", report)
    write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "cnn_source": "C057 fixed architecture", "residual_weight": RESIDUAL_WEIGHT, "primary_outer": "KFold(5, shuffle=true, random_state=2026)", "strict_outer": "GroupKFold(5)", "inner": "GroupKFold(4)", "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False})
    (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\ntorch={torch.__version__}\nnumpy={np.__version__}\npandas={pd.__version__}\ncuda={torch.cuda.is_available()}\nplatform={platform.platform()}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. EPS parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file(): manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items(): manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": primary["positive_folds"], "group_bootstrap_lower": bootstrap_lower, "minimum_panel_delta": minimum_panel}, sort_keys=True))


if __name__ == "__main__":
    main()
