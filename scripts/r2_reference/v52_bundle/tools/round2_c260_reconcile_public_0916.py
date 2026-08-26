"""C260: audit the reported public 0.916 artifact without scoring or tuning it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C260-20260805-reconcile-public-0916-v1"
PRED = ROOT / "submissions/R2-BEST-KNOWN-CLEAN-COMPOSITE-20260805.csv"
TEST = ROOT / "ppp-round-2/test.csv"
NOTEBOOK_CANDIDATES = [
    ROOT / "R2-BEST-DEFENSIBLE-COMPOSITE-SUBMISSION.ipynb",
    ROOT / "submissions/R2-BEST-DEFENSIBLE-COMPOSITE-SUBMISSION.ipynb",
    ROOT / "notebooks/R2-BEST-DEFENSIBLE-COMPOSITE-SUBMISSION.ipynb",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PRED)
    test = pd.read_csv(TEST, usecols=["id", "smiles"])
    notebook = next((path for path in NOTEBOOK_CANDIDATES if path.exists()), None)
    source = notebook.read_text(encoding="utf-8", errors="replace") if notebook else ""
    forbidden_hits = [token for token in ("test_external_labels", "local_eval", "LOCAL_DIAGNOSTIC_ONLY") if token.lower() in source.lower()]
    ids = frame["id"].to_numpy()
    targets = frame["target"].to_numpy(dtype=float)
    expected_ids = test["id"].to_numpy()
    report = {
        "schema_version": "ppp.round2.c260.public-reconciliation-audit.v1",
        "experiment_id": RUN.name,
        "status": "audited_not_scored",
        "prediction_path": str(PRED.relative_to(ROOT)),
        "prediction_sha256": digest(PRED),
        "prediction_matches_recorded_hash": digest(PRED) == "cdb3601c3f9c86b2e08a11eaf92df1c2ccee6d7a7c59a5c5bc0daf91ebbc768c",
        "prediction_rows": int(len(frame)),
        "prediction_columns": list(frame.columns),
        "prediction_ids_unique": bool(frame["id"].is_unique),
        "prediction_ids_finite": bool(pd.Series(ids).notna().all()),
        "prediction_targets_finite": bool(pd.Series(targets).notna().all()),
        "prediction_ids_match_current_test": bool(len(ids) == len(expected_ids) and (ids == expected_ids).all()),
        "current_test_rows": int(len(test)),
        "current_test_sha256": digest(TEST),
        "notebook_path": str(notebook.relative_to(ROOT)) if notebook else None,
        "notebook_source_scan": {"found": notebook is not None, "forbidden_hits": forbidden_hits, "source_complete_unknown": notebook is None},
        "user_reported_public_score": 0.916,
        "public_score_verified_here": False,
        "local_r2_computed": False,
        "local_eval_read": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "model_selection": False,
        "decision": "artifact_contract_verified_but_score_unresolved",
    }
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (RUN / "decision.md").write_text(
        "# C260 decision\n\n"
        "The artifact contract is audited locally, but the public leaderboard score cannot be recomputed from the supplied files. "
        "This audit does not convert the reported 0.916 into clean OOF evidence and does not select or modify a model.\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{digest(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
