#!/usr/bin/env python3
"""Build the self-contained official-data Round 2 submission notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


MARKER = "# NOTEBOOK_ENTRYPOINT"
FORBIDDEN_TEXT = ("local_eval", "nonofficial", "test_external_labels", "../polymer prediction challenge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="tools/initial_reference_pipeline.py")
    parser.add_argument("--output", default="notebooks/initial_reference_official_only.ipynb")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    source = source_path.read_text(encoding="utf-8")
    if source.count(MARKER) != 1:
        raise RuntimeError(f"Expected exactly one {MARKER!r} marker")
    reusable = source.split(MARKER, maxsplit=1)[0].rstrip()
    cell_source = reusable + """


notebook_report = run_pipeline(
    data_dir=None,
    output_path=Path.cwd() / "submission.csv",
    run_dir=Path.cwd() / "notebook_runtime",
)
print(json.dumps({
    "submission": notebook_report["submission"],
    "mean_oof_r2": notebook_report["validation"]["mean_selected_oof_r2"],
    "official_overrides": notebook_report["official_overrides"]["total_overrides"],
}, indent=2))
"""
    lowered = cell_source.lower()
    hits = [value for value in FORBIDDEN_TEXT if value in lowered]
    if hits:
        raise RuntimeError(f"Submission notebook source contains forbidden local-evaluation references: {hits}")

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# PPP Round 2 — official-data initial reference\n\n"
                "This notebook performs the complete load-to-`submission.csv` workflow in one run. "
                "It reads only the official competition files, derives all molecular features in memory, "
                "fits every model from random initialization, and writes all 4,940 test IDs."
            ),
            nbformat.v4.new_code_cell(cell_source),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "ppp_round2": {
                "lane": "CLEAN_OFFICIAL_ONLY",
                "method": "initial_reference_v1",
                "expected_rows": 4940,
            },
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
