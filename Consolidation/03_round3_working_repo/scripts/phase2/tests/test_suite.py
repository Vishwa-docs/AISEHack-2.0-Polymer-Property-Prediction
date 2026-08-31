"""Verify the Phase_2 suite: all 150 experiment files are real, importable, and
produce valid outputs in smoke mode.  No oracle, no old-file references.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"


def list_experiments() -> list[Path]:
    return sorted(EXP_DIR.glob("exp*.py"))


@pytest.fixture(scope="module")
def experiments() -> list[Path]:
    files = list_experiments()
    assert len(files) == 150, f"expected 150 experiment files, found {len(files)}"
    return files


def test_experiment_count():
    assert len(list_experiments()) == 150


def test_all_importable(experiments):
    for path in experiments:
        name = f"exp_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None, f"{path.name} cannot be loaded"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "run_experiment"), f"{path.name} lacks run_experiment()"
        assert module.EXP_ID, f"{path.name} lacks EXP_ID"
        assert module.TARGETS == ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


def test_smoke_run(experiments, tmp_path):
    """Run every experiment in smoke mode; it must write metrics + predictions."""
    for path in experiments:
        out_dir = tmp_path / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, str(path), "--smoke", "--output", str(out_dir)],
            capture_output=True, text=True, cwd=ROOT, timeout=600,
        )
        assert proc.returncode == 0, f"{path.name} failed:\n{proc.stderr[-3000:]}"
        metrics_path = out_dir / "metrics.json"
        preds_path = out_dir / "predictions.csv"
        assert metrics_path.is_file(), f"{path.name} did not write metrics.json"
        assert preds_path.is_file(), f"{path.name} did not write predictions.csv"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "mean_r2" in metrics, f"{path.name} metrics.json lacks mean_r2"
        assert "per_target" in metrics, f"{path.name} metrics.json lacks per_target"


def test_no_oracle_references(experiments):
    """Clean-source scan: no experiment may mention oracle/Oracle/ORACLE_ASSISTED."""
    for path in experiments:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "oracle" not in text.lower(), f"{path.name} references oracle"


def test_no_old_file_references(experiments):
    """No experiment may read old CSVs / hashes / experiment records."""
    banned = ["v52_bundle", "v53_base", "real_v57", "recipe_records", "final_submission_runs",
              "latest_submission", "logs/experiments", "NO_NOTEBOOK_SANDMAN"]
    for path in experiments:
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in banned:
            assert token not in text, f"{path.name} references banned token {token}"
