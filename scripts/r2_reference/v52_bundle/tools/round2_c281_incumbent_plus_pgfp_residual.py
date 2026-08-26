"""C281: research-only orthogonal PGFP residual on the frozen incumbent.

This is deliberately not a submission-eligible candidate because its clean OOF
parent is C050 while its test carrier is the frozen Sandman artifact.  It is a
bounded transfer diagnostic: determine whether C279's hierarchical residual
contains useful signal when added to the stronger incumbent carrier.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from round2_c279_polymer_genome_hierarchical_portfolio import build_features, digest, fit_predict, select_inner
import initial_reference_pipeline as reference


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ppp-round-2"
RUN = ROOT / "experiments/LOCAL_DIAGNOSTIC_ONLY/R2-C281-20260805-incumbent-plus-pgfp-residual"
PARENT_DIR = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-notebook-runtime-v8"
INCUMBENT = ROOT / "final_submissions/Sandman_Version_1_5th_Aug.csv"
TARGETS = tuple(reference.TARGETS)


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(PARENT_DIR / "oof_predictions.csv")
    parent_test = pd.read_csv(PARENT_DIR / "notebook_predictions.csv")
    incumbent = pd.read_csv(INCUMBENT)
    test = pd.read_csv(DATA_DIR / "test.csv")
    test["canonical"] = test["smiles"].map(reference.canonicalize)
    parent_test = parent_test.merge(test[["id", "target_type", "canonical"]], on="id", how="left", validate="one_to_one")
    keys = sorted(set(oof["canonical"].astype(str)) | set(test["canonical"].astype(str)))
    progress = RUN / "progress.jsonl"
    progress.write_text(json.dumps({"stage": "started", "experiment_id": RUN.name}) + "\n", encoding="utf-8")
    X, feature_report = build_features(keys, progress)
    key_index = {key: index for index, key in enumerate(keys)}
    target_reports = {}
    test_parts = []
    for target in TARGETS:
        frame = oof[oof["target_type"].astype(str).eq(target)].reset_index(drop=True)
        y = frame["target"].to_numpy(float)
        base = frame["candidate_prediction"].to_numpy(float)
        groups = frame["group"].astype(str).to_numpy(object)
        indices = np.asarray([key_index[value] for value in frame["canonical"].astype(str)], dtype=int)
        alpha, weight, inner = select_inner(X[indices], y, base, groups)
        residual_test_parent = parent_test[parent_test["target_type"].astype(str).eq(target)].copy()
        test_indices = np.asarray([key_index[value] for value in residual_test_parent["canonical"].astype(str)], dtype=int)
        residual = fit_predict(X[indices], y - base, X[test_indices], alpha)
        mask = test["target_type"].astype(str).eq(target).to_numpy()
        incumbent_values = incumbent.loc[mask, "target"].to_numpy(float)
        candidate = incumbent_values + weight * residual
        test_parts.append(pd.DataFrame({"id": test.loc[mask, "id"].astype(int).to_numpy(), "target": candidate}))
        target_reports[target] = {"alpha": alpha, "weight": weight, "inner": inner, "changed_rows": int(mask.sum())}
    predictions = test[["id"]].merge(pd.concat(test_parts, ignore_index=True), on="id", how="left", validate="one_to_one").sort_values("id").reset_index(drop=True)
    if len(predictions) != 4940 or not np.isfinite(predictions["target"].to_numpy(float)).all():
        raise RuntimeError("C281 output contract failed")
    predictions.to_csv(RUN / "predictions.csv", index=False)
    report = {"schema_version": "ppp.round2.c281.incumbent-plus-pgfp-residual.v1", "experiment_id": RUN.name, "status": "research_only_transfer_diagnostic", "clean_official_features": True, "test_carrier": "frozen Sandman incumbent", "clean_oof_parent": "C050", "submission_eligible": False, "local_eval_read_during_construction": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False, "feature_report": feature_report, "targets": target_reports, "complete_output_rows": int(len(predictions)), "source_hashes": {"runner": digest(Path(__file__)), "c279_runner": digest(ROOT / "tools" / "round2_c279_polymer_genome_hierarchical_portfolio.py"), "incumbent": digest(INCUMBENT), "parent_oof": digest(PARENT_DIR / "oof_predictions.csv")}}
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (RUN / "protocol.json").write_text(json.dumps({"schema_version": "ppp.round2.c281.incumbent-plus-pgfp-residual.v1", "experiment_id": RUN.name, "hypothesis": "C279 hierarchical residuals may be orthogonal corrections to the stronger Sandman test carrier.", "classification": "LOCAL_DIAGNOSTIC_ONLY", "submission_eligible": False, "local_eval_read_during_construction": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False}, indent=2) + "\n", encoding="utf-8")
    (RUN / "decision.md").write_text("# C281\n\nResearch-only transfer diagnostic. It is not submission-eligible because its clean OOF parent and test carrier are not the same pipeline.\n", encoding="utf-8")
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{digest(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": RUN.name, "rows": len(predictions), "submission_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
