"""C280: target-wise post-freeze transfer audit for the C279 arm.

Construction is official-only: each hybrid replaces exactly one target slice
of the frozen Sandman incumbent with the completed C279 prediction.  The
isolated post-freeze scorer is the only later consumer of the local diagnostic
external_label panels.  These files are research-only and cannot enter the submission
namespace or clean ledger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/LOCAL_DIAGNOSTIC_ONLY/R2-C280-20260805-c279-target-transfer-ablation"
DATA = ROOT / "ppp-round-2"
INCUMBENT = ROOT / "final_submissions/Sandman_Version_1_5th_Aug.csv"
COMPONENT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C279-polymer-genome-hierarchical-portfolio-v1/predictions.csv"
TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(DATA / "test.csv")
    incumbent = pd.read_csv(INCUMBENT)
    component = pd.read_csv(COMPONENT)
    expected_ids = test["id"].astype(int).to_numpy()
    if len(test) != 4940 or list(incumbent.columns) != ["id", "target"] or list(component.columns) != ["id", "target"]:
        raise RuntimeError("C280 input schema failure")
    for frame in (incumbent, component):
        if len(frame) != len(test) or not np.array_equal(frame["id"].astype(int).to_numpy(), expected_ids) or not np.isfinite(frame["target"].to_numpy(float)).all():
            raise RuntimeError("C280 prediction contract failure")
    protocol = {"schema_version": "ppp.round2.c280.c279-target-transfer-ablation.v1", "experiment_id": RUN.name, "status": "research_only_post_freeze_ablation", "hypothesis": "A C279 hierarchical arm may transfer for individual targets even when its complete seven-target artifact does not.", "construction": "Replace exactly one target_type slice in frozen incumbent with the completed official-only C279 prediction.", "targets": list(TARGETS), "local_eval_read_during_construction": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False, "promotion": False, "source_hashes": {"test": sha256(DATA / "test.csv"), "incumbent": sha256(INCUMBENT), "component": sha256(COMPONENT)}}
    (RUN / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    out_dir = RUN / "hybrids"
    out_dir.mkdir(exist_ok=True)
    records = []
    for target in TARGETS:
        mask = test["target_type"].astype(str).eq(target).to_numpy()
        hybrid = incumbent.copy()
        hybrid.loc[mask, "target"] = component.loc[mask, "target"].to_numpy(float)
        out = out_dir / f"hybrid_replace_{target}.csv"
        hybrid.to_csv(out, index=False)
        records.append({"target_replaced": target, "path": str(out.relative_to(ROOT)), "sha256": sha256(out), "rows": int(len(hybrid)), "changed_rows": int(mask.sum())})
    (RUN / "ablation_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": RUN.name, "hybrids": len(records), "rows_each": len(test), "local_eval_read_during_construction": False}, sort_keys=True))


if __name__ == "__main__":
    main()
