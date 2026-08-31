#!/usr/bin/env python3
"""C211: deterministic clean component compound audit v7.

This audit-only child runs after C210. It preserves C209's C208 Tg and C207 Egc
priorities, adds C210 as the first Nc priority, and preserves the existing
banked Ei/Eea/EPS priorities and strict eligibility checks.

It does not fit a new model family, does not select same-OOF maxima, does not
read local_eval/public feedback, and does not authorize notebook/upload/submission
actions. Missing or failed C207/C208/C210 metrics are skipped with explicit
reasons.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import initial_reference_pipeline as reference
import round2_c097_graph_grammar_hgb_full as parent_builder
import round2_c127_round1_carrier_factory as carrier
import round2_c200_clean_component_compound_audit_v3 as c200
import round2_c203_clean_component_compound_audit_v4 as c203


TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")

# Fixed priority, not a same-OOF max selector.
COMPONENT_PRIORITY: dict[str, list[str]] = {
    "tg": [
        "R2-C208-20260805-0352-tg-robust-group-measurement-v1",
    ],
    "egc": [
        "R2-C207-20260805-0344-egc-c180-transfer-guard-v1",
    ],
    "egb": [
        "R2-C201-20260805-0305-safe-egb-cross-property-stage2-v1",
    ],
    "ei": [
        "R2-C205-20260805-0332-ei-dense-oligomer-confirmation-v1",
        "R2-C199-20260805-0254-ei-c196-transfer-guard-v1",
        "R2-C196-20260805-0225-ei-ffox-shrinkage-confirmation-v1",
        "R2-C194-20260805-0152-safe-ei-cross-property-stage2-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ],
    "eea": [
        "R2-C204-20260805-0323-safe-eea-gap-identity-stage2-v1",
        "R2-C189-20260804-ffox-eea-confirmation-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ],
    "nc": [
        "R2-C210-20260805-0415-nc-optical-dispersion-gap-v1",
        "R2-C202-20260805-0315-nc-support-uncertainty-refractivity-v1",
        "R2-C197-20260805-0237-nc-c195-consensus-gated-v1",
        "R2-C195-20260805-0215-nc-nearmiss-residual-diversity-v1",
        "R2-C191-20260805-0027-nested-predicted-eps-to-nc-v1",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ],
    "eps": [
        "R2-C190-20260805-0023-ionic-eps-reproduction-v3",
        "R2-C188-20260804-fragment-path-kernel-v3",
        "R2-C192-20260805-0035-pi1m-support-conditioned-residual-v1",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="Polymer Prediction Challenge Round 2/ppp-round-2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--canonical-run",
        default="Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-v7",
    )
    args = parser.parse_args()
    started = time.time()
    root = Path(args.root).resolve()
    output_dir = Path(args.run_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    if not output_dir.is_dir() or {path.name for path in output_dir.iterdir()} != {"protocol.json"}:
        raise RuntimeError("pre-created protocol-only run directory required")

    parent = parent_builder.build_parent(root, (root / args.data_dir).resolve())
    parity = carrier.source_parity(root, parent, (root / args.canonical_run).resolve())
    if not parity["pass"]:
        raise RuntimeError(f"C050 parent parity failed: {parity}")

    selected: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    test_prediction = parent["test_parent_detail"][["id", "target_type", "target"]].copy()

    for target in TARGETS:
        info = parent["target_info"][target]
        y = np.asarray(info["y"], dtype=float)
        base = np.asarray(info["parent"], dtype=float)
        candidate = base.copy()
        source = "C050_parent"
        source_run: str | None = None
        if target in COMPONENT_PRIORITY:
            for run_id in COMPONENT_PRIORITY[target]:
                directory = c200.run_dir(root, run_id)
                metrics = c200.load_json(directory / "metrics.json")
                if metrics is None:
                    skipped.append({"target": target, "run_id": run_id, "reason": "missing_metrics"})
                    continue
                ok, reason = c203.metric_passes(metrics, target)
                if not ok:
                    skipped.append({"target": target, "run_id": run_id, "reason": reason})
                    continue
                candidate = c200.aligned_candidate(parent, target, directory)
                values = c200.full_prediction_values(parent, target, directory)
                mask = test_prediction["target_type"].astype(str).eq(target)
                test_prediction.loc[mask, "target"] = test_prediction.loc[mask, "id"].astype(int).map(values).to_numpy(float)
                source = f"component:{run_id}"
                source_run = run_id
                break
        selected[target] = {
            "source": source,
            "run_id": source_run,
            "parent_r2": float(r2_score(y, base)),
            "candidate_r2": float(r2_score(y, candidate)),
            "delta_r2": float(r2_score(y, candidate) - r2_score(y, base)),
            "rows": int(len(y)),
        }
        oof_parts.append(
            pd.DataFrame(
                {
                    "target_type": target,
                    "target": y,
                    "parent": base,
                    "candidate": candidate,
                    "selected_source": source,
                }
            )
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    parent_mean = float(np.mean([selected[target]["parent_r2"] for target in TARGETS]))
    candidate_mean = float(np.mean([selected[target]["candidate_r2"] for target in TARGETS]))
    test_prediction = test_prediction[["id", "target"]].sort_values("id").reset_index(drop=True)
    if len(test_prediction) != 4940 or not np.array_equal(test_prediction["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("C209 ID coverage/order contract failed")
    if not np.isfinite(test_prediction["target"].to_numpy(float)).all():
        raise RuntimeError("C209 produced non-finite predictions")
    max_loss = float(min(item["delta_r2"] for item in selected.values()))
    full_candidate_gate_pass = bool(candidate_mean - parent_mean >= 0.002 and max_loss >= -0.003)
    goal_0_95_met = bool(full_candidate_gate_pass and candidate_mean >= 0.95)
    report = {
        "schema_version": "ppp.round2.c211.clean-component-compound-audit.v7",
        "experiment_id": output_dir.name,
        "created_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "official_inputs": parent["inputs"],
        "official_only": True,
        "external_label_file_read": False,
        "local_eval_read": False,
        "pretrained_weights": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "audit_only_not_final_notebook": True,
        "component_priority": COMPONENT_PRIORITY,
        "parent_replay_parity": parity,
        "selected_components": selected,
        "skipped_components": skipped,
        "banked_targets": [target for target, item in selected.items() if item["run_id"] is not None],
        "mean_parent_r2": parent_mean,
        "mean_candidate_r2": candidate_mean,
        "mean_gain": candidate_mean - parent_mean,
        "gap_to_0_93": 0.93 - candidate_mean,
        "gap_to_0_95": 0.95 - candidate_mean,
        "maximum_target_loss": max_loss,
        "complete_output_rows": int(len(test_prediction)),
        "complete_output_order_pass": True,
        "full_candidate_gate_pass": full_candidate_gate_pass,
        "goal_0_95_met": goal_0_95_met,
        "decision": "compound_audit_goal_met_pending_notebook" if goal_0_95_met else "compound_audit_goal_unmet_continue_loop",
        "elapsed_seconds": float(time.time() - started),
        "source_hashes": {
            "runner": c200.sha256_file(Path(__file__)),
            "c203_helper": c200.sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c203_clean_component_compound_audit_v4.py"),
            "c200_helper": c200.sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c200_clean_component_compound_audit_v3.py"),
            "parent_builder": c200.sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c097_graph_grammar_hgb_full.py"),
            "carrier": c200.sha256_file(root / "Polymer Prediction Challenge Round 2/tools/round2_c127_round1_carrier_factory.py"),
            "reference": c200.sha256_file(root / "Polymer Prediction Challenge Round 2/tools/initial_reference_pipeline.py"),
        },
    }
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_prediction.to_csv(output_dir / "predictions.csv", index=False)
    c200.write_json(output_dir / "metrics.json", report)
    c200.write_json(
        output_dir / "config.json",
        {
            "schema_version": report["schema_version"],
            "component_priority": COMPONENT_PRIORITY,
            "selection_rule": "first completed clean-passing target component in frozen priority order; no local_eval/public feedback; no same-OOF max search",
            "local_eval_read": False,
        },
    )
    (output_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"sklearn={__import__('sklearn').__version__}",
                f"rdkit={reference.Chem.rdBase.rdkitVersion}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "decision.md").write_text(
        f"# {output_dir.name}\n\nDecision: `{report['decision']}`. "
        f"Mean parent `{parent_mean:.12f}`; compound `{candidate_mean:.12f}`; "
        f"gain `{candidate_mean - parent_mean:+.12f}`. Gap to 0.95: `{0.95 - candidate_mean:+.12f}`. "
        "Audit-only; no local_eval read; no Kaggle action.\n",
        encoding="utf-8",
    )
    manifest = [
        f"{c200.sha256_file(path)}  {path.name}"
        for path in sorted(output_dir.iterdir())
        if path.name != "artifact_manifest.sha256" and path.is_file()
    ]
    for name, digest in report["source_hashes"].items():
        manifest.append(f"{digest}  SOURCE {name}")
    (output_dir / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": output_dir.name,
                "banked_targets": report["banked_targets"],
                "mean_parent_r2": parent_mean,
                "mean_candidate_r2": candidate_mean,
                "mean_gain": candidate_mean - parent_mean,
                "goal_0_95_met": report["goal_0_95_met"],
                "decision": report["decision"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
