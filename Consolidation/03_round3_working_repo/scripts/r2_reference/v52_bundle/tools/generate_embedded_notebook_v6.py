#!/usr/bin/env python3
"""Generate a self-contained clean-only notebook from pinned local sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("initial_reference_pipeline", "initial_reference_pipeline.py"),
    ("round2_eea_cross_target_oof_residual_stack", "round2_eea_cross_target_oof_residual_stack.py"),
    ("round2_ei_scaffold_abstaining_gap_identity_v4_portable", "round2_ei_scaffold_abstaining_gap_identity_v4_portable.py"),
    ("round2_eea_scaffold_abstaining_gap_identity_v7_portable", "round2_eea_scaffold_abstaining_gap_identity_v7_portable.py"),
    ("round2_mixed_candidate_v6", "round2_mixed_candidate_v6.py"),
]
RUNTIME = "experiments/CLEAN_OFFICIAL_ONLY/R2-C050-20260803-2130-mixed-c001-gap-components-notebook-runtime-v7"


def main() -> None:
    sources = {
        name: (ROOT / "tools" / filename).read_text(encoding="utf-8")
        for name, filename in FILES
    }
    module_paths = {name: f"tools/{filename}" for name, filename in FILES}
    module_names = [name for name, _ in FILES]
    code_lines = [
        "from pathlib import Path",
        "import hashlib, json, sys, tempfile, types",
        "def discover_data_dir():",
        "    bases = [Path.cwd(), Path('/kaggle/input'), Path('/kaggle/working')]",
        "    candidates = []",
        "    for base in bases:",
        "        candidates.extend([base / 'ppp-round-2', base])",
        "        if base.exists():",
        "            candidates.extend(base.glob('*/ppp-round-2'))",
        "            candidates.extend(base.glob('*'))",
        "    for candidate in candidates:",
        "        if (candidate / 'train.csv').is_file() and (candidate / 'test.csv').is_file() and (candidate / 'archive' / 'train.csv').is_file():",
        "            return candidate.resolve()",
        "    raise FileNotFoundError('official Round 2 train/test/archive files were not found')",
        "DATA_DIR = discover_data_dir()",
        "ROOT = DATA_DIR.parent",
        f"SOURCES = {sources!r}",
        f"MODULE_PATHS = {module_paths!r}",
        f"for module_name in {module_names!r}:",
        "    module = types.ModuleType(module_name)",
        "    sys.modules[module_name] = module",
        "    exec(compile(SOURCES[module_name], MODULE_PATHS[module_name], 'exec'), module.__dict__)",
        "source_hashes = {name + '_module': hashlib.sha256(SOURCES[name].encode('utf-8')).hexdigest() for name in SOURCES}",
        "source_hashes = {'candidate_script': source_hashes['round2_mixed_candidate_v6_module'], 'reference_module': source_hashes['initial_reference_pipeline_module'], 'ei_route_module': source_hashes['round2_ei_scaffold_abstaining_gap_identity_v4_portable_module'], 'eea_route_module': source_hashes['round2_eea_scaffold_abstaining_gap_identity_v7_portable_module'], 'metric_plumbing': source_hashes['round2_eea_cross_target_oof_residual_stack_module']}",
        "with tempfile.TemporaryDirectory(prefix='ppp-round2-v6-runtime-') as runtime_dir:",
        "    runtime = Path(runtime_dir)",
        "    (runtime / 'protocol.json').write_text(json.dumps({'schema_version': 'ppp.round2.notebook-runtime.v3', 'parent': 'R2-C050-20260803-2130-mixed-c001-gap-components-v6', 'execution_mode': 'saved_embedded_notebook'}, sort_keys=True), encoding='utf-8')",
        "    output = Path.cwd() / 'submission.csv'",
        "    result = sys.modules['round2_mixed_candidate_v6'].run_candidate(DATA_DIR, runtime, output, root_override=ROOT, source_hashes_override=source_hashes)",
        "print(json.dumps({'experiment_id': result['experiment_id'], 'rows': result['submission']['rows'], 'mean_candidate_r2': result['mean_candidate_r2']}, sort_keys=True))",
    ]
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Polymer Round 2 mixed candidate v6\n",
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
    output = ROOT / "notebooks" / "R2-C050-mixed-candidate-v6.ipynb"
    output.write_text(json.dumps(notebook, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
