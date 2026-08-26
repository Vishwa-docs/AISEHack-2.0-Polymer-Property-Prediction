"""Shared harness for the Fable experiment ladder (F-series).

Design goals, in order:
1. Deployment-matched validation. Every OOF estimate must be computed under the
   same information availability the real test rows will see:
   - partner LABELS (another property measured on the same canonical structure,
     present in official train.csv or archive/train.csv) are ALWAYS legitimate
     features - they
     exist at inference time. They are never masked.
   - partner PREDICTIONS (model-imputed values for missing partner labels) must
     come from models fitted with the evaluated row's structure fully excluded
     (structure-grouped cross-fitting). This is the exact fix for the C132/C139
     circular-fallback failure.
2. Shift-matched scoring. Weak-target test rows sit far from train (median NN
   Tanimoto ~0.55-0.57, train-vs-test domain AUC ~ 1.0). Random-grouped CV is
   therefore optimistic by 0.02-0.08 R2 (measured: ei -0.08, eps -0.05,
   nc -0.02, egb -0.02). Every experiment reports, per target:
     a. grouped 5-fold OOF R2 (comparable to all historical numbers),
     b. cluster-holdout R2 (Butina clusters, coarse),
     c. SHIFT-MATCHED R2: OOF residuals reweighted so the held-out rows'
        NN-similarity-to-training-fold histogram matches the actual
        test-to-train histogram. This is the primary decision metric.
3. No oracle anywhere. This module never reads any file outside the official
   ppp-round-2 inputs.

All F-series scripts import from here so folds, seeds, canonicalization and
metrics are identical across experiments and across targets.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROUND2_DIR = os.path.dirname(os.path.dirname(HERE))  # .../Polymer Prediction Challenge Round 2
DATA_DIR = os.path.join(ROUND2_DIR, "ppp-round-2")

TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ELECTRONIC = ["egc", "egb", "ei", "eea", "nc", "eps"]
SEED = 20260805

# archive/train.csv IS permitted (official Round 2 bundle, per AGENTS.md section 4)
# and is the single most valuable input: its 2,449 archive-only rows supply exact
# same-property labels for 1,645 tg and 804 egc current test rows.
# archive/test.csv is unlabeled and archive/sample_submission.csv is a 10-row
# illustration; neither is loaded here.
OFFICIAL_SHA256 = {
    "train.csv": "609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2",
    "test.csv": "d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d",
    "PI1M.csv": "c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8",
    os.path.join("archive", "train.csv"): "b12cadb31c747b26e3616474ce3a2839a22e4b2603e392b6b528e714b6864f68",
}

FORBIDDEN_PATH_TOKENS = ("oracle", "test_answers", "ORACLE_ASSISTED")


def assert_permitted_path(path: str) -> str:
    """Guard every read. Raises if the path touches a banned source."""
    low = str(path).lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok.lower() in low:
            raise RuntimeError(f"forbidden input path (contains {tok!r}): {path}")
    return path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_official_inputs(include_archive: bool = True) -> None:
    for rel, want in OFFICIAL_SHA256.items():
        if not include_archive and rel.startswith("archive"):
            continue
        path = assert_permitted_path(os.path.join(DATA_DIR, rel))
        got = sha256(path)
        if got != want:
            raise RuntimeError(f"official input hash mismatch for {rel}: {got}")


def canon_nostereo(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


@dataclass
class Round2Data:
    train: pd.DataFrame          # official train.csv + can column
    test: pd.DataFrame           # official test.csv + can column
    archive: pd.DataFrame        # official archive/train.csv + can column
    all_labels: pd.DataFrame     # train + archive long labels, deduplicated
    wide: pd.DataFrame           # canonical structure x target label matrix (mean-aggregated)
    label_sets: dict = field(default_factory=dict)  # target -> set(can) with a label


def load_data(verify: bool = True, include_archive: bool | None = None) -> Round2Data:
    """Load official inputs in the selected archive-enabled or current-only mode."""
    if include_archive is None:
        include_archive = os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1"
    if verify:
        verify_official_inputs(include_archive=include_archive)
    train = pd.read_csv(assert_permitted_path(os.path.join(DATA_DIR, "train.csv")))
    test = pd.read_csv(assert_permitted_path(os.path.join(DATA_DIR, "test.csv")))
    if include_archive:
        archive = pd.read_csv(assert_permitted_path(os.path.join(DATA_DIR, "archive", "train.csv")))
    else:
        archive = pd.DataFrame(columns=train.columns)
    for df in (train, test, archive):
        df["can"] = df["smiles"].map(canon_nostereo)
        if df["can"].isna().any():
            raise RuntimeError("canonicalization failure in official data")
    all_labels = (
        pd.concat(
            [train[["can", "target_type", "target"]], archive[["can", "target_type", "target"]]],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    wide = all_labels.groupby(["can", "target_type"])["target"].mean().unstack()
    label_sets = {t: set(wide.index[wide[t].notna()]) if t in wide.columns else set()
                  for t in TARGETS}
    return Round2Data(train, test, archive, all_labels, wide, label_sets)


def exact_lookup_table(data: Round2Data, target: str) -> pd.DataFrame:
    """Conflict-aware same-property lookup from official train + archive.

    Returns a frame indexed by canonical structure with columns
    ``value`` (mean), ``spread`` (max - min) and ``n``. A caller must ABSTAIN
    from a hard override wherever ``spread`` exceeds its declared tolerance:
    tg has 6 conflicting groups with spread up to 24 K, egc has none.
    """
    labs = data.all_labels[data.all_labels.target_type == target]
    g = labs.groupby("can")["target"].agg(["mean", "min", "max", "count"])
    out = pd.DataFrame({"value": g["mean"], "spread": g["max"] - g["min"], "n": g["count"]})
    return out


# ---------------------------------------------------------------------------
# fingerprints / similarity
# ---------------------------------------------------------------------------

def morgan_fp(can: str, radius: int = 2, nbits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(can), radius, nbits)


def morgan_fp_list(cans, radius: int = 2, nbits: int = 2048):
    return [morgan_fp(c, radius, nbits) for c in cans]


def nn_similarity(query_fps, ref_fps) -> np.ndarray:
    """Max Tanimoto of each query fp against the reference set (excluding identical index handling is caller's job)."""
    out = np.empty(len(query_fps))
    for i, f in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(f, ref_fps)
        out[i] = max(sims) if sims else 0.0
    return out


def butina_clusters(cans, cutoff: float = 0.6) -> np.ndarray:
    """Cluster structures by Tanimoto distance with the Butina algorithm."""
    from rdkit.ML.Cluster import Butina

    fps = morgan_fp_list(cans)
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - np.asarray(sims))
    clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)
    labels = np.empty(n, dtype=int)
    for ci, members in enumerate(clusters):
        for m in members:
            labels[m] = ci
    return labels


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------

def grouped_folds(cans, n_splits: int = 5, seed: int = SEED) -> np.ndarray:
    """Structure-grouped folds: identical canonical structures always share a fold."""
    rng = np.random.RandomState(seed)
    uniq = pd.unique(pd.Series(list(cans)))
    perm = rng.permutation(len(uniq))
    fold_of_group = {u: perm[i] % n_splits for i, u in enumerate(uniq)}
    return np.array([fold_of_group[c] for c in cans])


def cluster_folds(cans, n_splits: int = 5, cutoff: float = 0.6, seed: int = SEED) -> np.ndarray:
    """Butina-cluster-grouped folds: whole similarity clusters held out together."""
    uniq = list(dict.fromkeys(cans))
    labels = butina_clusters(uniq, cutoff=cutoff)
    rng = np.random.RandomState(seed)
    cluster_ids = np.unique(labels)
    perm = rng.permutation(len(cluster_ids))
    fold_of_cluster = {c: perm[i] % n_splits for i, c in enumerate(cluster_ids)}
    fold_of_can = {u: fold_of_cluster[labels[i]] for i, u in enumerate(uniq)}
    return np.array([fold_of_can[c] for c in cans])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def r2_score_manual(y, pred) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def shift_matched_r2(y, pred, oof_nn_sim, test_nn_sim, bins=None) -> float:
    """R2 with held-out rows importance-weighted so that their NN-similarity
    histogram matches the test rows' NN-similarity histogram.

    oof_nn_sim: for each OOF row, max Tanimoto to its TRAINING folds only.
    test_nn_sim: for each real test row, max Tanimoto to full train.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if bins is None:
        bins = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0001])
    oof_bin = np.digitize(oof_nn_sim, bins) - 1
    test_bin = np.digitize(test_nn_sim, bins) - 1
    w = np.zeros(len(y))
    for b in range(len(bins) - 1):
        n_oof = (oof_bin == b).sum()
        n_test = (test_bin == b).sum()
        if n_oof > 0:
            w[oof_bin == b] = n_test / n_oof
    if w.sum() == 0:
        return float("nan")
    w = w / w.sum() * len(y)
    mu = float((w * y).sum() / w.sum())
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - mu) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def grouped_bootstrap_lower(y, pred_a, pred_b, groups, n_boot: int = 2000,
                            alpha: float = 0.025, seed: int = SEED) -> float:
    """2.5% lower bound of R2(pred_b) - R2(pred_a) under group bootstrap."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[g] for g in take])
        deltas[b] = r2_score_manual(y[idx], np.asarray(pred_b)[idx]) - r2_score_manual(
            y[idx], np.asarray(pred_a)[idx]
        )
    return float(np.quantile(deltas, alpha))


# ---------------------------------------------------------------------------
# evaluation report
# ---------------------------------------------------------------------------

def evaluate_target(
    name: str,
    target: str,
    y: np.ndarray,
    oof_pred: np.ndarray,
    baseline_oof: np.ndarray | None,
    cans: list,
    folds: np.ndarray,
    data: Round2Data,
    extra: dict | None = None,
) -> dict:
    """Standard per-target evaluation block used by every F-series experiment."""
    train_cans = list(dict.fromkeys(cans))
    fps_all = {c: morgan_fp(c) for c in set(cans) | set(data.test[data.test.target_type == target]["can"])}

    # OOF NN-sim: each row vs the cans in OTHER folds only
    oof_sim = np.empty(len(cans))
    fold_arr = np.asarray(folds)
    fps_by_fold = {f: [fps_all[c] for c, ff in zip(cans, fold_arr) if ff != f] for f in np.unique(fold_arr)}
    for i, (c, f) in enumerate(zip(cans, fold_arr)):
        sims = DataStructs.BulkTanimotoSimilarity(fps_all[c], fps_by_fold[f])
        oof_sim[i] = max(sims) if sims else 0.0

    test_cans = data.test[data.test.target_type == target]["can"].tolist()
    train_fp_list = [fps_all[c] for c in train_cans]
    test_sim = nn_similarity([fps_all[c] for c in test_cans], train_fp_list)

    rep = {
        "experiment": name,
        "target": target,
        "n": int(len(y)),
        "oof_r2": r2_score_manual(y, oof_pred),
        "shift_matched_r2": shift_matched_r2(y, oof_pred, oof_sim, test_sim),
    }
    if baseline_oof is not None:
        rep["baseline_oof_r2"] = r2_score_manual(y, baseline_oof)
        rep["delta_oof"] = rep["oof_r2"] - rep["baseline_oof_r2"]
        rep["shift_matched_baseline_r2"] = shift_matched_r2(y, baseline_oof, oof_sim, test_sim)
        rep["delta_shift_matched"] = rep["shift_matched_r2"] - rep["shift_matched_baseline_r2"]
        rep["grouped_bootstrap_lower_2p5"] = grouped_bootstrap_lower(y, baseline_oof, oof_pred, np.asarray(cans))
    if extra:
        rep.update(extra)
    return rep


def save_report(report: dict | list, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise RuntimeError(f"refusing to overwrite existing artifact: {out_path}")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"wrote {out_path}")
