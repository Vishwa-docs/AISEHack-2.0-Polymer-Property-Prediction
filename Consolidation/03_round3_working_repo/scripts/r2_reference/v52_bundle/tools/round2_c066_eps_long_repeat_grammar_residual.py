#!/usr/bin/env python3
"""Fixed official-only EPS long-repeat-unit grammar residual diagnostic."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import initial_reference_pipeline as reference
import round2_eea_cross_target_oof_residual_stack as plumbing


SEED = 2026
TARGET = "eps"
WEIGHT = 0.20
ALPHA = 10.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def grammar(values: list[str]) -> tuple[np.ndarray, list[str]]:
    names = ["length", "branches", "brackets", "wildcards", "ring_digits", "double_bonds", "triple_bonds", "aromatic_bonds", "equals_colons", "carbon_count", "hetero_count", "aromatic_count", "branch_ratio", "hetero_ratio", "aromatic_ratio", "length_per_atom"]
    out = np.zeros((len(values), len(names)), dtype=float)
    for row, value in enumerate(values):
        text = str(value); atoms = [ch for ch in text if ch.isalpha() and ch.isupper()]
        carbon = sum(ch == "C" for ch in atoms); hetero = sum(ch in "NOSPFIBrCl" for ch in atoms); aromatic = sum(ch in "cnosp" for ch in text)
        atom_count = max(len(atoms), 1)
        out[row] = [len(text), text.count("("), text.count("["), text.count("*"), sum(ch.isdigit() for ch in text), text.count("="), text.count("#"), text.count(":"), text.count("=") + text.count(":"), carbon, hetero, aromatic, text.count("(") / max(len(text), 1), hetero / atom_count, aromatic / max(len(text), 1), len(text) / atom_count]
    return out, names


def bootstrap_lower(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups); members = {group: np.flatnonzero(groups == group) for group in unique}; rng = np.random.default_rng(SEED); values = []
    for _ in range(2000):
        selected = rng.choice(unique, size=len(unique), replace=True); rows = np.concatenate([members[group] for group in selected])
        if np.var(y[rows]) > 1.0e-15: values.append(float(r2_score(y[rows], candidate[rows]) - r2_score(y[rows], parent[rows])))
    return float(np.quantile(values, 0.025))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--data-dir", default="ppp-round-2"); parser.add_argument("--run-dir", required=True); args = parser.parse_args(); root = Path(args.root).resolve(); run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != {"protocol.json"}: raise RuntimeError("pre-created empty run directory with protocol.json is required")
    started = time.time(); train, test, archive, inputs = reference.load_inputs((root / args.data_dir).resolve()); _, pooled = reference.build_label_pool(train, archive); keys = sorted(set(pooled["canonical"]) | set(test["canonical"])); key_to_index = {key: index for index, key in enumerate(keys)}; molecules = reference.build_molecules(keys); descriptor, _ = reference.descriptor_matrix(molecules); physical, _ = reference.physical_matrix(molecules, keys); dense_base = np.hstack([descriptor, physical]).astype(np.float64, copy=False); cross_values, cross_available = reference.cross_property_arrays(pooled, keys); sparse_parts = [reference.morgan_count_matrix(molecules, 2, 4096), reference.morgan_count_matrix(molecules, 3, 4096), reference.text_matrix(keys, 65536)]; fingerprints = reference.morgan_bits(molecules, 2, 4096); detail, parent_oof_frame, _ = reference.fit_targets(pooled, test, keys, dense_base, cross_values, cross_available, sparse_parts, fingerprints, reference.DEFAULT_CONFIG)
    frame = pooled[pooled["target_type"] == TARGET].reset_index(drop=True); parent_rows = parent_oof_frame[parent_oof_frame["target_type"] == TARGET].reset_index(drop=True); y = frame["target"].to_numpy(float); parent = parent_rows["prediction"].to_numpy(float); strings = frame["smiles"].astype(str).tolist(); test_frame = test[test["target_type"] == TARGET].sort_values("id").reset_index(drop=True); test_strings = test_frame["smiles"].astype(str).tolist(); train_grammar, feature_names = grammar(strings); test_grammar, _ = grammar(test_strings); threshold = float(np.quantile(np.asarray([len(value) for value in strings], dtype=float), 0.75)); long_mask = np.asarray([len(value) >= threshold for value in strings], dtype=bool); test_long = np.asarray([len(value) >= threshold for value in test_strings], dtype=bool); expected_long = int(np.sum(long_mask)); groups = np.asarray([plumbing.no_stereo(value) for value in frame["canonical"]], dtype=object); folds = plumbing.folds_for(groups, 5); residual = y - parent; candidate = parent.copy(); fold_rows = []; long_deltas = []
    for fold in range(5):
        validation = np.flatnonzero(folds == fold); training = np.flatnonzero((folds != fold) & long_mask); valid_long = validation[long_mask[validation]]; fitted = None
        if len(training) >= 8: fitted = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ALPHA)); fitted.fit(train_grammar[training], residual[training])
        if fitted is not None and len(valid_long): candidate[valid_long] = parent[valid_long] + WEIGHT * fitted.predict(train_grammar[valid_long])
        fold_delta = float(r2_score(y[validation], candidate[validation]) - r2_score(y[validation], parent[validation])); long_delta = float(r2_score(y[valid_long], candidate[valid_long]) - r2_score(y[valid_long], parent[valid_long])) if len(valid_long) >= 5 and np.var(y[valid_long]) > 1.0e-15 else 0.0; long_deltas.append(long_delta); fold_rows.append({"fold": fold, "rows": int(len(validation)), "long_rows": int(len(valid_long)), "parent_r2": float(r2_score(y[validation], parent[validation])), "candidate_r2": float(r2_score(y[validation], candidate[validation])), "delta_r2": fold_delta, "long_delta_r2": long_delta})
    parent_r2 = float(r2_score(y, parent)); candidate_r2 = float(r2_score(y, candidate)); delta = candidate_r2 - parent_r2; lower = bootstrap_lower(y, parent, candidate, groups); test_parent = detail[detail["target_type"] == TARGET].sort_values("id")["model_prediction"].to_numpy(float); full_train = np.flatnonzero(long_mask); full = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ALPHA)); full.fit(train_grammar[full_train], residual[full_train]); test_candidate = test_parent.copy(); test_candidate[test_long] = test_parent[test_long] + WEIGHT * full.predict(test_grammar[test_long]); component = pd.DataFrame({"id": test_frame["id"].astype(int), "target_type": TARGET, "parent_prediction": test_parent, "candidate_prediction": test_candidate, "long_slice": test_long})
    if len(component) != 153 or component["id"].duplicated().any() or not np.array_equal(component["id"].to_numpy(), test_frame["id"].to_numpy()) or not np.isfinite(component["candidate_prediction"].to_numpy()).all(): raise RuntimeError("EPS long component output contract failed")
    component.to_csv(run_dir / "eps_long_component_predictions.csv", index=False); pd.DataFrame({"canonical": frame["canonical"].astype(str), "target": y, "parent": parent, "candidate": candidate, "group": groups, "long_slice": long_mask, "outer_fold": folds}).to_csv(run_dir / "oof_predictions.csv", index=False); gates = {"gain_pass": delta >= 0.01, "fold_pass": sum(row["delta_r2"] > 0 for row in fold_rows) >= 4, "bootstrap_pass": lower > 0.0, "panel_pass": min(long_deltas) >= 0.0, "support_pass": expected_long == 62, "component_rows_pass": len(component) == 153}; passed = bool(all(gates.values())); source_names = ("round2_c066_eps_long_repeat_grammar_residual.py", "initial_reference_pipeline.py", "round2_eea_cross_target_oof_residual_stack.py")
    report = {"schema_version": "ppp.round2.c066.eps-long-repeat-grammar-residual.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "parent": "R2-C050-20260803-2130-mixed-c001-gap-components-v7; exact v7 EPS regenerated in-process", "official_inputs": inputs, "official_only": True, "external_label_file_read": False, "local_eval_read": False, "pretrained_weights": False, "prior_prediction_input": False, "target": TARGET, "feature_names": feature_names, "slice_threshold": threshold, "oof_long_rows": expected_long, "test_long_rows": int(np.sum(test_long)), "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "folds": fold_rows, "long_slice_min_delta": float(min(long_deltas)), "group_bootstrap_lower": lower, "gates": gates, "decision": "pass_component_gate" if passed else "rejected_component_gate", "rows": {"train": int(len(train)), "archive": int(len(archive)), "test": int(len(test)), "component_test": int(len(component)), "oof": int(len(y))}, "source_hashes": {name: sha256_file(root / "tools" / name) for name in source_names}, "elapsed_seconds": time.time() - started}
    write_json(run_dir / "metrics.json", report); write_json(run_dir / "config.json", {"seed": SEED, "target": TARGET, "slice": "official EPS SMILES length >= 75th percentile", "expected_oof_long_rows": 62, "feature_family": "fixed repeat-unit grammar counts", "residual_weight": WEIGHT, "ridge_alpha": ALPHA, "outer": "canonical_no_stereo GroupKFold(5)", "bootstrap_resamples": 2000, "no_hyperparameter_sweep": True, "external_label_file_read": False, "local_eval_read": False}); (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\nplatform={platform.platform()}\n", encoding="utf-8"); (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8"); (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{report['decision']}**. EPS parent R2 `{parent_r2:.12f}`, candidate R2 `{candidate_r2:.12f}`, delta `{delta:+.12f}`. Long OOF rows `{expected_long}`. Official-only; no local_eval, external_label file, or Kaggle action.\n", encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.name != "artifact_manifest.sha256" and path.is_file(): manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    for name, digest in report["source_hashes"].items(): manifest.append(f"{digest}  SOURCE tools/{name}")
    (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8"); print(json.dumps({"experiment_id": run_dir.name, "decision": report["decision"], "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": delta, "positive_folds": report["positive_folds"], "group_bootstrap_lower": lower, "long_slice_min_delta": report["long_slice_min_delta"], "oof_long_rows": expected_long}, sort_keys=True))


if __name__ == "__main__":
    main()
