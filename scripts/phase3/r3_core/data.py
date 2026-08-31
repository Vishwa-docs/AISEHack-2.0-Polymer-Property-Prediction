"""Official data loading, canonical keys, grouped folds, target statistics.

Reads ONLY Dataset/train.csv and Dataset/test.csv (and Dataset/PI1M.csv /
Dataset/smile_r3.csv only through explicit opt-in helpers).  No oracle, no old
artifacts, no hashes of historical files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")
TARGET_ORDER = {name: i for i, name in enumerate(TARGETS)}
SEED = 2026

# Frozen verified facts (EXPERIMENT_LOOP.md Stage 0) — re-verified at load time.
TRAIN_SHA = "609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2"
TEST_SHA = "d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d"


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_dataset() -> Path:
    """Find the official Dataset/ directory containing train.csv + test.csv."""
    here = Path.cwd().resolve()
    candidates = []
    for parent in (here, *here.parents):
        candidates.append(parent / "Dataset")
        candidates.append(parent / "ppp-round-3")
        candidates.append(parent / "data")
    candidates.extend([
        Path("/kaggle/input/ppp-round-3"),
        Path("/kaggle/input/aisehack-2-0"),
        Path("/home/vishwa/Desktop/r3_runtime"),
    ])
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "train.csv").is_file() and (candidate / "test.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate official Dataset/ (train.csv + test.csv)")


def load_official_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and sanity-check official train/test. Returns (train, test)."""
    if data_dir is None:
        data_dir = locate_dataset()
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    # Verify the frozen data identity (byte-level) before trusting anything.
    actual_train = sha256_file(train_path)
    actual_test = sha256_file(test_path)
    if actual_train != TRAIN_SHA:
        raise RuntimeError(f"train.csv hash mismatch: {actual_train} != {TRAIN_SHA}")
    if actual_test != TEST_SHA:
        raise RuntimeError(f"test.csv hash mismatch: {actual_test} != {TEST_SHA}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["target_type"] = train["target_type"].astype(str).str.lower()
    test["target_type"] = test["target_type"].astype(str).str.lower()
    if list(train.columns) != ["smiles", "target", "target_type"]:
        raise RuntimeError("unexpected train columns")
    if list(test.columns) != ["id", "smiles", "target_type"]:
        raise RuntimeError("unexpected test columns")
    if len(train) != 7409 or len(test) != 4940:
        raise RuntimeError("unexpected row counts")
    if not np.array_equal(test["id"].to_numpy(int), np.arange(1, 4941)):
        raise RuntimeError("test ids are not 1..4940")
    return train, test


def canonicalize(smiles: str) -> str:
    """Canonical no-stereo structure key (grouping key for folds)."""
    mol = Chem.MolFromSmiles(str(smiles).replace("[*]", "*"))
    if mol is None:
        # Keep the raw string as its own key so no row is dropped.
        return f"UNPARSED|{smiles}"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def add_structure_keys(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add canonical structure keys and a shared structure index."""
    train = train.copy()
    test = test.copy()
    train["canonical"] = train["smiles"].map(canonicalize)
    test["canonical"] = test["smiles"].map(canonicalize)
    all_keys = sorted(set(train["canonical"]) | set(test["canonical"]))
    key_index = {key: i for i, key in enumerate(all_keys)}
    train["structure_index"] = train["canonical"].map(key_index).astype(int)
    test["structure_index"] = test["canonical"].map(key_index).astype(int)
    return train, test


def per_target_rows(train: pd.DataFrame) -> dict[str, int]:
    return {t: int((train["target_type"] == t).sum()) for t in TARGETS}


def grouped_folds(
    train: pd.DataFrame,
    target: str,
    n_splits: int = 5,
    seed: int = SEED,
    stratify: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Structure-grouped folds: a canonical structure never straddles folds.

    Rows are grouped by structure_index; groups are split by target label
    quantiles when stratify=True, else shuffled.  Returns fold train/val index
    pairs into the target-masked row positions.
    """
    mask = (train["target_type"] == target).to_numpy()
    positions = np.where(mask)[0]
    groups = train.loc[mask, "structure_index"].to_numpy(int)
    labels = train.loc[mask, "target"].to_numpy(float)

    unique_groups, group_inv = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)

    if stratify and len(labels) >= n_splits * 2 and n_groups >= n_splits:
        # Aggregate per-group label stats, then order groups by median label so
        # bin assignment is deterministic and label-stratified.
        group_labels = np.full(n_groups, np.nan)
        for g in range(n_groups):
            vals = labels[group_inv == g]
            group_labels[g] = np.median(vals) if len(vals) else np.nan
        order = np.argsort(group_labels, kind="stable")
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_groups)

    group_fold = np.empty(n_groups, dtype=int)
    for i, g in enumerate(order):
        group_fold[g] = i % n_splits

    folds = []
    for fold in range(n_splits):
        val = group_fold[group_inv] == fold
        folds.append((positions[~val], positions[val]))
    return folds


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    test_ids: np.ndarray


def overlap_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    """The frozen 457 train/test shared-structure audit (informational only)."""
    train_keys = set(train["canonical"])
    test_keys = set(test["canonical"])
    shared = train_keys & test_keys
    return {
        "shared_structures": len(shared),
        "shared_test_rows": int(test["canonical"].isin(shared).sum()),
    }
