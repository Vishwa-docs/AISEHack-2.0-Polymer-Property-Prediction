#!/usr/bin/env python3
"""Generate a self-contained clean-only notebook from pinned local sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("initial_reference_pipeline", "initial_reference_pipeline.py"),
    ("round2_eea_cross_target_oof_residual_stack", "round2_eea_cross_target_oof_residual_stack.py"),
    ("round2_ei_scaffold_abstaining_gap_identity_v4", "round2_ei_scaffold_abstaining_gap_identity_v4.py"),
    ("round2_eea_scaffold_abstaining_gap_identity_v7", "round2_eea_scaffold_abstaining_gap_identity_v7.py"),
    ("round2_mixed_candidate_v3", "round2_mixed_candidate_v3.py"),
]
RUNTIME = "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-notebook-runtime-v4"


def main() -> None:
    sources = {
        name: (ROOT / "tools" / filename).read_text(encoding="utf-8")
        for name, filename in FILES
    }
    module_paths = {name: f"tools/{filename}" for name, filename in FILES}
    module_names = [name for name, _ in FILES]
    code_lines = [
        "from pathlib import Path",
        "import json, sys, types",
        f"ROOT = Path({str(ROOT)!r})",
        f"RUNTIME = ROOT / {RUNTIME!r}",
        f"SOURCES = {sources!r}",
        f"MODULE_PATHS = {module_paths!r}",
        f"for module_name in {module_names!r}:",
        "    module = types.ModuleType(module_name)",
        "    module.__file__ = str(ROOT / MODULE_PATHS[module_name])",
        "    sys.modules[module_name] = module",
        "    exec(compile(SOURCES[module_name], module.__file__, 'exec'), module.__dict__)",
        "RUNTIME.mkdir(parents=True, exist_ok=True)",
        "output = RUNTIME / 'predictions.csv'",
        "result = sys.modules['round2_mixed_candidate_v3'].run_candidate('ppp-round-2', RUNTIME, output)",
        "print(json.dumps({'experiment_id': result['experiment_id'], 'rows': result['submission']['rows'], 'mean_candidate_r2': result['mean_candidate_r2']}, sort_keys=True))",
    ]
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Polymer Round 2 mixed candidate v3\n",
                    "This notebook embeds and executes the exact clean official-only source for all seven target pipelines.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["\n".join(code_lines) + "\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "notebooks" / "R2-C050-mixed-candidate-v3.ipynb"
    output.write_text(json.dumps(notebook, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
