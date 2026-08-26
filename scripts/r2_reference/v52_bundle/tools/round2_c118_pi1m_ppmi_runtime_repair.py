#!/usr/bin/env python3
"""C118 versioned runtime repair for the C117 fragment-PPMI probe."""

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

import initial_reference_pipeline as reference
import round2_c112_c050_parent_parity_control as parent_control
import round2_c117_pi1m_fragment_ppmi_probe as c117


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run_dir = (root / args.run_dir).resolve()
    if {path.name for path in run_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("C118 requires a fresh protocol-only directory")
    started = time.time()
    progress = run_dir / "progress.jsonl"

    def checkpoint(name: str, **fields: Any) -> None:
        record = {"checkpoint": name, "at": datetime.now().astimezone().isoformat(), **fields}
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    try:
        data_dir = (root / args.data_dir).resolve()
        pi1m_path = data_dir / "PI1M.csv"
        pi_hash = sha256_file(pi1m_path)
        if pi_hash != c117.PI1M_SHA256:
            raise RuntimeError(f"PI1M hash mismatch: {pi_hash}")
        raw_pi1m = pd.read_csv(pi1m_path, usecols=["SMILES"])
        selected, selected_hashes, sample_report = c117.select_pi1m(raw_pi1m)
        checkpoint("sample_selected", rows=len(selected), sample_report=sample_report)
        generator = c117.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=c117.BUCKETS)
        smoke, invalid_smoke, _ = c117.token_matrix(selected[:5000], generator)
        if invalid_smoke / 5000.0 > 0.005 or smoke.nnz == 0:
            raise RuntimeError(f"fragment smoke failed: invalid={invalid_smoke}, nnz={smoke.nnz}")
        parent_predictions, parent_oof, context = parent_control.rebuild_parent(root, data_dir, run_dir)
        checkpoint("parent_replay", oof_rows=len(parent_oof), test_rows=len(parent_predictions))
        canonical_dir = root / "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7"
        canonical_predictions = pd.read_csv(canonical_dir / "predictions.csv")
        canonical_oof = pd.read_csv(canonical_dir / "oof_predictions.csv")
        test_delta = float(np.max(np.abs(parent_predictions["target"].to_numpy(float) - canonical_predictions["target"].to_numpy(float))))
        left = parent_oof.sort_values(["target_type", "canonical", "target"], kind="mergesort").reset_index(drop=True)
        right = canonical_oof.sort_values(["target_type", "canonical", "target"], kind="mergesort").reset_index(drop=True)
        oof_delta = float(np.max(np.abs(left["parent_prediction"].to_numpy(float) - right["candidate_prediction"].to_numpy(float))))
        checkpoint("parent_parity", oof_max_abs=oof_delta, test_max_abs=test_delta)
        if oof_delta > 1e-12 or test_delta > 1e-12:
            raise RuntimeError(f"parent parity failed: oof={oof_delta}, test={test_delta}")
        train, test, archive, inputs = context["train"], context["test"], context["archive"], context["inputs"]
        _, pooled = reference.build_label_pool(train, archive)
        keys = sorted(set(pooled["canonical"]) | set(test["canonical"]))
        molecules = reference.build_molecules(keys)
        official_union = pd.concat([train["smiles"], test["smiles"], archive["smiles"]], ignore_index=True).astype(str).tolist()
        control_corpus = [official_union[index % len(official_union)] for index in range(c117.SAMPLE_SIZE)]
        pi_matrix, invalid_pi, pi_canonicals_list = c117.token_matrix(selected, generator)
        control_matrix, invalid_control, _ = c117.token_matrix(control_corpus, generator)
        pi_embedding = c117.ppm_embedding(pi_matrix)
        control_embedding = c117.ppm_embedding(control_matrix)
        official_matrix, invalid_official, _ = c117.token_matrix(keys, generator)
        pi_features = c117.molecule_features(official_matrix, pi_embedding)
        control_features = c117.molecule_features(official_matrix, control_embedding)
        checkpoint("ppmi_fit", pi_nnz=int(pi_matrix.nnz), control_nnz=int(control_matrix.nnz), feature_shape=list(pi_features.shape))
        fingerprints = reference.morgan_bits(molecules, radius=2, bits=4096)
        pi_canonicals = {value for value in pi_canonicals_list if value is not None}
        target_reports: dict[str, Any] = {}
        oof_parts: list[pd.DataFrame] = []
        for target in c117.ACTIVE:
            report, oof = c117.evaluate_target(target, parent_oof, keys, pi_features, control_features, fingerprints, pi_canonicals)
            target_reports[target] = report
            oof_parts.append(oof)
            checkpoint("head_fit", target=target, delta_r2=report["candidate_delta_r2"], control_delta_r2=report["control_delta_r2"])
        for target in c117.TARGETS:
            if target in c117.ACTIVE:
                continue
            rows = parent_oof[parent_oof["target_type"] == target].copy()
            rows["candidate_prediction"] = rows["parent_prediction"]
            rows["control_prediction"] = rows["parent_prediction"]
            oof_parts.append(rows[["canonical", "target_type", "target", "parent_prediction", "candidate_prediction", "control_prediction"]])
            base = float(reference.r2_score(rows["target"], rows["parent_prediction"]))
            target_reports[target] = {"parent_r2": base, "candidate_r2": base, "control_r2": base, "candidate_delta_r2": 0.0, "control_delta_r2": 0.0, "pi1m_minus_control": 0.0, "pass": True, "unchanged_parent": True}
        mean_parent = float(np.mean([target_reports[target]["parent_r2"] for target in c117.TARGETS]))
        mean_candidate = float(np.mean([target_reports[target]["candidate_r2"] for target in c117.TARGETS]))
        clean_pass = bool(mean_candidate - mean_parent >= 0.002 and min(target_reports[target]["candidate_delta_r2"] for target in c117.TARGETS) >= -0.003 and all(target_reports[target]["pass"] for target in c117.ACTIVE))
        audit = {
            "schema_version": "ppp.round2.c118.pi1m-ppmi-runtime-repair.run.v1",
            "experiment_id": run_dir.name,
            "created_at": datetime.now().astimezone().isoformat(),
            "official_only": True,
            "local_eval_read": False,
            "external_label_file_read": False,
            "kaggle_compute": False,
            "kaggle_upload": False,
            "official_inputs": inputs,
            "pi1m_sha256": pi_hash,
            "pi1m_rows_available": int(len(raw_pi1m)),
            "pi1m_rows_used": int(len(selected)),
            "sample_report": sample_report,
            "parse_invalid_selected": int(invalid_pi),
            "parse_invalid_control": int(invalid_control),
            "parse_invalid_official_keys": int(invalid_official),
            "parent_replay_oof_max_abs": oof_delta,
            "parent_replay_test_max_abs": test_delta,
            "targets": target_reports,
            "mean_parent_r2": mean_parent,
            "mean_candidate_r2": mean_candidate,
            "mean_gain": mean_candidate - mean_parent,
            "decision": "clean_gate_pass_pending_test_fit" if clean_pass else "rejected_component_or_full_gate",
            "elapsed_seconds": float(time.time() - started),
        }
        pd.DataFrame([{"target": target, "parent_r2": target_reports[target]["parent_r2"], "candidate_r2": target_reports[target]["candidate_r2"], "control_r2": target_reports[target]["control_r2"], "candidate_delta_r2": target_reports[target]["candidate_delta_r2"], "control_delta_r2": target_reports[target]["control_delta_r2"]} for target in c117.TARGETS]).to_csv(run_dir / "metrics.csv", index=False)
        pd.concat(oof_parts, ignore_index=True).to_csv(run_dir / "oof_predictions.csv", index=False)
        write_json(run_dir / "metrics.json", audit)
        write_json(run_dir / "config.json", {"schema_version": "ppp.round2.c118.pi1m-ppmi-runtime-repair.v1", "sample_size": c117.SAMPLE_SIZE, "buckets": c117.BUCKETS, "rank": c117.RANK, "blend": c117.BLEND, "ridge_alpha": c117.ALPHA, "pi1m_sha256": pi_hash, "official_inputs": inputs})
        (run_dir / "sample_manifest.json").write_text(json.dumps({"sha256": selected_hashes, "rows": len(selected), "raw_smiles_not_tracked": True}, indent=2) + "\n", encoding="utf-8")
        (run_dir / "environment.txt").write_text("\n".join([f"python={platform.python_version()}", f"numpy={np.__version__}", f"pandas={pd.__version__}", f"scikit_learn={__import__('sklearn').__version__}", f"rdkit={reference.Chem.rdBase.rdkitVersion}", f"platform={platform.platform()}"]) + "\n", encoding="utf-8")
        (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: **{audit['decision']}**. No full-data test fit or local_eval action was opened before the clean gates.\n", encoding="utf-8")
        checkpoint("metrics_written", decision=audit["decision"], mean_gain=audit["mean_gain"])
    except Exception as exc:
        error = {"schema_version": "ppp.round2.c118.error.v1", "experiment_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(), "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": float(time.time() - started), "scientific_score": None, "local_eval_read": False}
        write_json(run_dir / "error.json", error)
        (run_dir / "environment.txt").write_text(f"python={platform.python_version()}\nplatform={platform.platform()}\n", encoding="utf-8")
        (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        (run_dir / "decision.md").write_text(f"# {run_dir.name}\n\nDecision: failed_pre_metrics. No scientific score, full-data fit, or local_eval action. Error: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        checkpoint("terminal_error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest = []
        for path in sorted(run_dir.iterdir()):
            if path.name != "artifact_manifest.sha256" and path.is_file():
                manifest.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
        source_paths = [Path(__file__), root / "tools/round2_c117_pi1m_fragment_ppmi_probe.py", root / "tools/round2_c112_c050_parent_parity_control.py", root / "tools/initial_reference_pipeline.py"]
        for path in source_paths:
            if path.exists():
                manifest.append(f"{sha256_file(path)}  SOURCE {path.relative_to(root)}")
        (run_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
