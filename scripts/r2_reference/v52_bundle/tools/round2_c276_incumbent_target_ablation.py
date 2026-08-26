"""C276: preregistered one-target artifact ablation.

This creates seven local hybrid candidates by replacing exactly one target in
the frozen Sandman incumbent with the clean C257 component candidate. It does
not read or use an external_label file; scoring is performed later by the isolated
post-freeze diagnostic tool. The hybrids are research artifacts, not a new
submission or a model-generated incumbent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/LOCAL_DIAGNOSTIC_ONLY/R2-C276-20260805-one-target-artifact-ablation"
DATA = ROOT / "ppp-round-2"
INCUMBENT = ROOT / "final_submissions/Sandman_Version_1_5th_Aug.csv"
COMPONENT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C257-20260805-0920-clean-component-compound-audit-v28/predictions.csv"
TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(DATA / "test.csv")
    incumbent = pd.read_csv(INCUMBENT)
    component = pd.read_csv(COMPONENT)
    assert list(test.columns) == ["id", "smiles", "target_type"]
    assert list(incumbent.columns) == ["id", "target"]
    assert list(component.columns) == ["id", "target"]
    expected_ids = test["id"].astype(int).to_numpy()
    for frame in (incumbent, component):
        assert len(frame) == len(test)
        assert frame["id"].astype(int).to_numpy().tolist() == expected_ids.tolist()
        assert np.isfinite(frame["target"].to_numpy(float)).all()
    RUN.joinpath("protocol.json").write_text(json.dumps({
        "schema_version": "ppp.round2.c276.target-ablation.v1",
        "experiment_id": RUN.name,
        "status": "research_only_post_freeze_ablation",
        "hypothesis": "The public incumbent and the clean C257 component have different target strengths; one-target hybrids identify which replacements are worth later public testing.",
        "construction": "Replace exactly one target column by target_type using frozen official-only candidate CSVs; retain the incumbent for the other six targets.",
        "targets": list(TARGETS),
        "local_eval_read_during_construction": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "promotion": False,
        "source_hashes": {"test": sha256(DATA / "test.csv"), "incumbent": sha256(INCUMBENT), "component": sha256(COMPONENT)},
    }, indent=2) + "\n", encoding="utf-8")
    out_dir = RUN / "hybrids"
    out_dir.mkdir(exist_ok=True)
    records = []
    for target in TARGETS:
        hybrid = incumbent.copy()
        mask = test["target_type"].astype(str).eq(target).to_numpy()
        hybrid.loc[mask, "target"] = component.loc[mask, "target"].to_numpy(float)
        out = out_dir / f"hybrid_replace_{target}.csv"
        hybrid.to_csv(out, index=False)
        records.append({"target_replaced": target, "path": str(out.relative_to(ROOT)), "sha256": sha256(out), "rows": int(len(hybrid)), "changed_rows": int(mask.sum())})
    RUN.joinpath("ablation_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": RUN.name, "hybrids": len(records), "rows_each": len(test), "local_eval_read_during_construction": False}, sort_keys=True))


if __name__ == "__main__":
    main()
