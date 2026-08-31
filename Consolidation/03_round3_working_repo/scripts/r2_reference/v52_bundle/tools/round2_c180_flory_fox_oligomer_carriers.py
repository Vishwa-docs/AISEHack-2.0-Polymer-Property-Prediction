#!/usr/bin/env python3
"""C180: clean Flory--Fox-style oligomer asymptote carrier experiment.

This is deliberately a bounded, official-data-only experiment.  It replays
the exact C050 parent, derives 1-mer/2-mer/3-mer descriptors from each
official SMILES, extrapolates descriptor values against 1/n (the
Flory--Fox-style asymptotic coordinate), and evaluates target-specific direct
Ridge/ExtraTrees carriers with the same fresh grouped OOF gates as C127.

The test external_labels file is never opened.  The resulting complete candidate is a
clean research artifact and is not a submission until independently reviewed
and reproduced.
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

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier


TARGETS = tuple(reference.TARGETS)
SEED = 2026
N_BITS = 2048
TEXT_FEATURES = 32768
FFOX_MAX_REPEATS = 3
FFOX_TRANSFORM = "both"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_features(root: Path, smiles: list[str]) -> tuple[np.ndarray, sparse.csr_matrix, dict[str, Any]]:
    """Build the C127 portable block set plus clean Flory--Fox features."""
    round1_dir = root / "Polymer Prediction Challenge" / "tools"
    sys.path.insert(0, str(round1_dir))
    import polymer_official_train_eval_loop as round1

    built = round1.build_features(
        smiles,
        n_bits=N_BITS,
        text_features=TEXT_FEATURES,
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
        oligomer_ffox_features=True,
        oligomer_ffox_max_repeats=FFOX_MAX_REPEATS,
        oligomer_ffox_transform=FFOX_TRANSFORM,
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
    blocks = [built["blocks"][name] for name in carrier.DIRECT_BLOCKS if name in built["blocks"]]
    if not blocks:
        raise RuntimeError("C180 produced no sparse feature blocks")
    sparse_features = sparse.hstack(blocks, format="csr").astype(np.float64)
    return dense, sparse_features, {
        "dense_shape": [int(value) for value in dense.shape],
        "sparse_shape": [int(value) for value in sparse_features.shape],
        "sparse_nnz": int(sparse_features.nnz),
        "selected_blocks": [name for name in carrier.DIRECT_BLOCKS if name in built["blocks"]],
        "feature_reports": built["feature_reports"],
        "round1_source": str(round1_dir / "polymer_official_train_eval_loop.py"),
        "flory_fox": {
            "max_repeats": FFOX_MAX_REPEATS,
            "transform": FFOX_TRANSFORM,
            "fit_coordinate": "1/n",
            "normalization": "descriptor value divided by heavy-atom count before extrapolation",
            "source": "official train/test SMILES only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--canonical-run", default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7")
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
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    dense, sparse_features, feature_report = build_features(root, parent["keys"])
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
            raise RuntimeError(f"C180 test ID alignment failed for {target}")
        test_indices = np.asarray([
            parent["key_to_index"][value]
            for value in test_rows["canonical"]
        ], dtype=np.int64)
        result = carrier.fit_target(info, dense, sparse_features, test_indices, test_detail["target"].to_numpy(float))
        report = carrier.evaluate_target(info, result)
        report.update({
            "blend_name": result["blend_name"],
            "blend_weights": [float(value) for value in result["weights"]],
            "blend_intercept": result["intercept"],
            "feature_rows": int(len(info["y"])),
        })
        target_reports[target] = report
        result_by_target[target] = result
        folds = carrier.grouped_folds(np.asarray(info["groups"], dtype=object))
        oof_parts.append(pd.DataFrame({
            "canonical": info["canonical"],
            "target_type": target,
            "target": info["y"],
            "parent": info["parent"],
            "candidate": result["candidate"],
            "group": info["groups"],
            "scaffold": info["scaffolds"],
            "fold": folds,
        }))
        direct_test_parts.append(pd.DataFrame({
            "id": test_rows["id"].astype(int),
            "target_type": target,
            "direct_candidate": result["test_direct"],
        }))

    banked = [target for target in TARGETS if target_reports[target]["pass"]]
    parent_mean = float(np.mean([target_reports[target]["parent_r2"] for target in TARGETS]))
    assembled_oof: list[pd.DataFrame] = []
    for part in oof_parts:
        target = str(part["target_type"].iloc[0])
        part = part.copy()
        part["assembled"] = part["candidate"] if target in banked else part["parent"]
        assembled_oof.append(part)
    oof = pd.concat(assembled_oof, ignore_index=True)
    assembled_mean = float(np.mean([carrier.r2_score(part["target"], part["assembled"]) for part in assembled_oof]))
    parent_test = parent["test_parent_detail"][["id", "target_type", "target"]].copy()
    direct_test = pd.concat(direct_test_parts, ignore_index=True)
    predictions = parent_test.merge(direct_test, on=["id", "target_type"], how="left", validate="one_to_one")
    predictions["target"] = np.where(predictions["target_type"].isin(banked), predictions["direct_candidate"], predictions["target"])
    predictions = predictions[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.array_equal(predictions["id"].to_numpy(), np.arange(1, 4941)) or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C180 complete output contract failed")
    max_loss = float(min(target_reports[target]["delta_r2"] if target in banked else 0.0 for target in TARGETS))
    full_pass = bool(assembled_mean - parent_mean >= 0.002 and max_loss >= -0.003 and len(banked) > 0)
    report = {
        "schema_version": "ppp.round2.c180.flory-fox-oligomer-carriers.v1",
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
            "carrier": sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"),
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
        "schema_version": "ppp.round2.c180.flory-fox-oligomer-carriers.v1",
        "seed": SEED,
        "source": "C127 direct carrier protocol with clean Flory-Fox-style asymptotic descriptors",
        "flory_fox_max_repeats": FFOX_MAX_REPEATS,
        "flory_fox_transform": FFOX_TRANSFORM,
        "folds": "grouped no-stereo; exact C050 source replay as fallback",
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
    print(json.dumps({
        "experiment_id": run_dir.name,
        "banked_targets": banked,
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": assembled_mean,
        "mean_gain": assembled_mean - parent_mean,
        "decision": report["decision"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
