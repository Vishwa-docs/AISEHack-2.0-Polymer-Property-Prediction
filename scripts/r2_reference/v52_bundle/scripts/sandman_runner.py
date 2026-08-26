
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_bundle() -> Path:
    here = Path.cwd().resolve()
    candidates = []
    for parent in (here, *here.parents):
        candidates.append(parent / "ppp-round-2")
    candidates.extend([
        Path("/kaggle/input/ppp-round-2"),
        Path("/kaggle/input/aisehack-2-0"),
        Path("/kaggle/input/competitions/aisehack-2-0"),
        Path("/kaggle/input/polymer-property-prediction-round-2/ppp-round-2"),
        Path("/kaggle/input/aisehack-2-0-polymer-property-prediction-round-2/ppp-round-2"),
    ])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(p for p in kaggle_input.glob("*") if p.is_dir())
        candidates.extend(p for p in kaggle_input.glob("*/*") if p.is_dir())
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "train.csv").is_file() and (candidate / "test.csv").is_file():
            return candidate
    raise FileNotFoundError("Official train/test files were not found")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_script(script: Path, cwd: Path) -> None:
    completed = subprocess.run([sys.executable, str(script)], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Embedded generation script failed: {script.name}")


_EXPECTED_TEST_IDS = None

def expected_test_ids() -> np.ndarray:
    global _EXPECTED_TEST_IDS
    if _EXPECTED_TEST_IDS is None:
        _EXPECTED_TEST_IDS = pd.read_csv(locate_bundle() / "test.csv", usecols=["id"])["id"].to_numpy(int)
    return _EXPECTED_TEST_IDS

def validate_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "target"]:
        raise RuntimeError(f"Unexpected CSV columns for {path}")
    expected_ids = expected_test_ids()
    if len(frame) != len(expected_ids):
        raise RuntimeError(f"Unexpected row count for {path}: {len(frame)}")
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"].to_numpy(int), expected_ids):
        raise RuntimeError(f"Unexpected test IDs for {path}")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite target values for {path}")
    return frame


def make_generated_baseline_method(bundle_root: Path):
    def ensure_generated_baseline(self):
        if self.archive_baseline_done:
            return
        output = self.work_root / "R2-BEST-DEFENSIBLE-COMPOSITE-SUBMISSION.csv"
        if not output.is_file():
            run_script(bundle_root / "scripts" / "baseline_defensible.py", self.work_root)
        validate_csv(output)
        for rel in [
            "submissions/R2-BEST-KNOWN-CLEAN-COMPOSITE-20260805.csv",
            "submissions/R2-BEST-DEFENSIBLE-COMPOSITE-SUBMISSION.csv",
            "experiments/CLEAN_OFFICIAL_ONLY/R2-BEST-DEFENSIBLE-COMPOSITE-LOCAL-ONLY.csv",
            "experiments/final_submission_runs/with_archive/R2-BEST-KNOWN-0916-baseline.csv",
            "final_submission/with_archive/R2-BEST-COMPOUND-with_archive-V2.csv",
        ]:
            self.copy_generated(output, rel)
            self.require_csv(rel)
        self.archive_baseline_done = True
    return ensure_generated_baseline


def make_final_compound_leaf(original_generate_leaf, bundle_root: Path):
    def generate_leaf(self, rel: str) -> bool:
        name = Path(rel).name
        if "R2-FINAL-BEST-COMPOUND-SUBMISSION" in name:
            output = self.work_root / "submission.csv"
            if not output.is_file():
                run_script(bundle_root / "scripts" / "final_compound.py", self.work_root)
            validate_csv(output)
            self.copy_generated(output, rel)
            self.require_csv(rel)
            return True
        return original_generate_leaf(self, rel)
    return generate_leaf


def build_target_blend(rebuilder, data_dir: Path, base_rel: str, selected: dict, output_path: Path, internal_frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    internal_frames = internal_frames or {}
    test = pd.read_csv(data_dir / "test.csv")
    target_type = test["target_type"].astype(str).str.lower().to_numpy(object)
    base_path = rebuilder.materialize(base_rel)
    base = validate_csv(base_path)
    values = base["target"].to_numpy(float).copy()

    for target in TARGETS:
        item = selected[target]
        weight = float(item["weight"])
        if abs(weight) <= 0.0:
            continue
        source_key = str(item["path"])
        if source_key in internal_frames:
            source = internal_frames[source_key]
        else:
            source = validate_csv(rebuilder.materialize(source_key))
        mask = target_type == target
        source_values = source["target"].to_numpy(float)
        base_values = base["target"].to_numpy(float)
        values[mask] = base_values[mask] + weight * (source_values[mask] - base_values[mask])

    out = pd.DataFrame({"id": test["id"].to_numpy(int), "target": values})
    out.to_csv(output_path, index=False)
    validate_csv(output_path)
    return out


def main() -> None:
    variant_key = os.environ["SANDMAN_VARIANT_KEY"]
    bundle_root = Path(os.environ["SANDMAN_BUNDLE_ROOT"]).resolve()
    configs = json.loads((bundle_root / "recipes" / "variant_configs.json").read_text(encoding="utf-8"))
    config = configs[variant_key]
    output_path = Path(os.environ["SANDMAN_OUTPUT_CSV"]).resolve()
    data_dir = locate_bundle()

    data_link = bundle_root / "ppp-round-2"
    if not data_link.exists():
        try:
            data_link.symlink_to(data_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(data_dir, data_link)

    recipes = json.loads((bundle_root / "recipes" / "recipe_records.json").read_text(encoding="utf-8"))
    reconstruct = load_module(bundle_root / "tools" / "reconstruct_v50_v53_standalone_noarchive_20260809.py", "sandman_reconstruct")
    reconstruct.ROOT = bundle_root
    reconstruct.load_manifest_records = lambda: recipes
    reconstruct.Rebuilder.ensure_archive_baseline = make_generated_baseline_method(bundle_root)
    reconstruct.Rebuilder.generate_leaf = make_final_compound_leaf(reconstruct.Rebuilder.generate_leaf, bundle_root)

    work_dir = Path.cwd() / ".sandman_generation_workspace" / variant_key
    rebuilder = reconstruct.Rebuilder(work_dir, verbose=True)
    rebuilder.setup()

    internal_frames: dict[str, pd.DataFrame] = {}
    if variant_key == "noarchive_rank1":
        internal_one = work_dir / "internal_noarchive_blend1.csv"
        internal_frames["__internal_noarchive_blend1__"] = build_target_blend(
            rebuilder,
            data_dir,
            config["internal_noarchive_blend1"]["base"],
            config["internal_noarchive_blend1"]["selected"],
            internal_one,
        )

    result = build_target_blend(rebuilder, data_dir, config["base"], config["selected"], output_path, internal_frames)
    digest = sha256_file(output_path)
    print(json.dumps({"output": str(output_path), "rows": int(len(result)), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
