#!/usr/bin/env python3
"""C927 no-archive C282 repeat-view residual wrapper.

This reuses the C278 repeat-view residual implementation, but replaces its
archive-derived C050 parent with the current-only C282 parent artifacts that
were materialized by C340.  It reads only official current train/test-derived
artifacts, performs no local_eval scoring/selection, and writes a branch-local
without_archive CSV for separate post-freeze local_eval scoring.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import round2_c278_repeat_view_nested_portfolio as c278


ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C340-20260808-noarchive-c282-polymer-genome-wrapper-v1"
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C927-20260808-noarchive-c282-repeat-view-wrapper"
OUTPUT = ROOT / "experiments/final_submission_runs/without_archive/R2-C927-NOARCHIVE-C282-REPEAT-VIEW-WRAPPER-20260808.csv"
MANIFEST = OUTPUT.with_suffix(".manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or OUTPUT.exists() or MANIFEST.exists():
        raise RuntimeError("Refusing overwrite/reuse for C927")
    parent_oof = PARENT_DIR / "parent_c282_oof_for_c279.csv"
    parent_test = PARENT_DIR / "parent_c282_test_for_c279.csv"
    parent_metrics = PARENT_DIR / "metrics.json"
    for path in (parent_oof, parent_test, parent_metrics):
        if not path.is_file():
            raise RuntimeError(f"Missing parent artifact: {path}")
    metrics = json.loads(parent_metrics.read_text())
    if metrics.get("archive_file_read") is not False or metrics.get("archive_labels_used") is not False:
        raise RuntimeError("C340 parent is not current-only by metrics")
    if metrics.get("local_eval_read") is not False:
        raise RuntimeError("C340 parent is not local_eval-clean by metrics")

    # Override C278 module constants before invoking its main implementation.
    c278.RUN = RUN
    c278.PARENT_DIR = PARENT_DIR
    c278.PARENT_OOF = parent_oof
    c278.PARENT_TEST = parent_test
    c278.main()

    predictions = RUN / "predictions.csv"
    if not predictions.is_file():
        raise RuntimeError("C278 wrapper did not produce predictions.csv")
    frame = pd.read_csv(predictions)
    if list(frame.columns) != ["id", "target"] or len(frame) != 4940:
        raise RuntimeError("Unexpected C927 predictions schema")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    record = {
        "schema_version": "ppp.round2.c927.noarchive-c282-repeat-view-wrapper.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "without_archive",
        "method": "C278 repeat-view residual wrapper over current-only C282 parent artifacts",
        "official_current_train_used": True,
        "archive_labels_used": False,
        "archive_file_read": False,
        "pi1m_used": False,
        "local_eval_read_by_builder": False,
        "kaggle_compute": False,
        "kaggle_upload": False,
        "kaggle_submission": False,
        "parent": {
            "run_dir": str(PARENT_DIR),
            "oof": {"path": str(parent_oof), "sha256": sha256_file(parent_oof), "bytes": parent_oof.stat().st_size},
            "test": {"path": str(parent_test), "sha256": sha256_file(parent_test), "bytes": parent_test.stat().st_size},
            "metrics": {"path": str(parent_metrics), "sha256": sha256_file(parent_metrics), "bytes": parent_metrics.stat().st_size},
        },
        "run_dir": str(RUN),
        "run_metrics": {"path": str(RUN / "metrics.json"), "sha256": sha256_file(RUN / "metrics.json"), "bytes": (RUN / "metrics.json").stat().st_size},
        "output": {"path": str(OUTPUT), "sha256": sha256_file(OUTPUT), "rows": int(len(frame)), "bytes": OUTPUT.stat().st_size},
    }
    MANIFEST.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": record["output"], "run_dir": str(RUN)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
