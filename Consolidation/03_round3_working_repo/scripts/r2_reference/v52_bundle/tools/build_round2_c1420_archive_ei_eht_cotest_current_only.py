#!/usr/bin/env python3
"""C1420 archive EI EHT + co-test residual source.

This is a thin archive-branch wrapper around the C380 current-only EI
EHT/co-test residual builder.  The residual source still trains only from
official current train OOF artifacts and current test structures.  The only
archive-branch input is the frozen archive base CSV receiving the EI residual.

No local_eval, external_label, Kaggle, pretrained, or external target inputs are read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import build_round2_c380_noarchive_ei_eht_cotest_current_only as c380


def archive_guard_path(path: Path, *, role: str, allow_output: bool = False) -> None:
    low = str(path).lower()
    for token in ("local_eval", "external_label", "test_external_labels"):
        if token in low:
            raise RuntimeError(f"Refusing forbidden {role} path: {path}")
    if role in {"base candidate", "base"}:
        if "/with_archive/" not in low:
            raise RuntimeError(f"Archive EI overlay base must be branch-local with_archive: {path}")
    elif allow_output:
        if "/with_archive/" not in low:
            raise RuntimeError(f"{role} path must stay in with_archive namespace: {path}")
    else:
        if "/archive/" in low or low.endswith("/archive"):
            raise RuntimeError(f"Refusing raw archive path for current-only residual source: {path}")
    if "Polymer Prediction Challenge Round 2" not in str(path.resolve()):
        raise RuntimeError(f"{role} path is outside Round 2 boundary: {path}")


def patch_manifest(argv: list[str]) -> None:
    manifest_path: Path | None = None
    output_path: Path | None = None
    for index, item in enumerate(argv):
        if item == "--manifest" and index + 1 < len(argv):
            manifest_path = Path(argv[index + 1]).resolve()
        if item == "--output" and index + 1 < len(argv):
            output_path = Path(argv[index + 1]).resolve()
    if manifest_path is None or not manifest_path.is_file():
        return
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["schema_version"] = "ppp.round2.c1420.archive-ei-eht-cotest-current-only.v1"
    record["branch"] = "with_archive"
    record["archive_branch_base_allowed"] = True
    record["archive_file_read_by_builder"] = False
    record["archive_labels_used_by_residual_source"] = False
    record["method"] = (
        "EI-only EHT orbital plus co-test residual source trained from official "
        "current-only C282 OOF and applied over an archive-branch base CSV"
    )
    if output_path is not None:
        record["output"]["path"] = str(output_path)
        record["output"]["sha256"] = c380.sha256_file(output_path)
        record["output"]["bytes"] = output_path.stat().st_size
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    c380.guard_path = archive_guard_path
    c380.main()
    patch_manifest(sys.argv)


if __name__ == "__main__":
    main()
