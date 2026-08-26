"""C157: from-scratch PI1M randomized-SMILES contrastive representation.

This is an official-data-only representation experiment. PI1M, official train
SMILES, and official test SMILES are used only as unlabeled covariates. Target
labels enter only the fold-local weak-target residual heads. The local_eval is not
read by this program.
"""
from pathlib import Path
import hashlib
import json
import runpy
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
OUT = ROOT / "experiments/LOCAL_DIAGNOSTIC_ONLY"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260804
HASH_DIM = 1024
LATENT = 64
SSL_ROWS = 50_000
SSL_EPOCHS = 1
BATCH = 256
TAU = 0.20
SSL_LR = 0.08
SSL_WD = 2e-4
RESIDUAL_BLEND = 0.25
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
started = time.time()
SSL_CHECKPOINT = OUT / "R2-C157-pi1m-contrastive-weak-targets-LOCAL_DIAGNOSTIC_ONLY-ssl-checkpoint.npz"


def stable_hash(s):
    return int.from_bytes(hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest(), "little")


def r2(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def randomized_smiles(s, rng):
    """Deterministically randomize atom order; fall back for invalid SMILES."""
    s = str(s)
    mol = Chem.MolFromSmiles(s)
    if mol is None or mol.GetNumAtoms() < 2:
        return s
    order = np.arange(mol.GetNumAtoms())
    rng.shuffle(order)
    try:
        return Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, order.tolist()),
            canonical=False,
            isomericSmiles=True,
        )
    except Exception:
        return s


def hash_vector(s):
    """Character n-gram hashing with no learned or imported vocabulary."""
    out = np.zeros(HASH_DIM, dtype=np.float32)
    s = str(s)
    for n in (1, 2, 3, 4, 5):
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            h = 2166136261
            for ch in s[i:i + n]:
                h = ((h ^ ord(ch)) * 16777619) & 0xffffffff
            out[h % HASH_DIM] += 1.0
    norm = np.linalg.norm(out)
    if norm > 0:
        out /= norm
    return out


def batch_matrix(strings):
    return np.vstack([hash_vector(s) for s in strings]).astype(np.float32)


def normalize_rows(z):
    n = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(n, 1e-8)


def train_contrastive(strings):
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.035, size=(HASH_DIM, LATENT)).astype(np.float32)
    order = np.arange(len(strings))
    history = []
    start_epoch = 0
    if SSL_CHECKPOINT.exists():
        saved = np.load(SSL_CHECKPOINT)
        if saved["W"].shape == W.shape and int(saved["corpus_size"]) == len(strings):
            W = saved["W"].astype(np.float32)
            start_epoch = int(saved["next_epoch"])
            print("C157 resuming SSL checkpoint", {"next_epoch": start_epoch, "corpus_size": len(strings)}, flush=True)
    for epoch in range(start_epoch, SSL_EPOCHS):
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH]
            if len(ids) < 8:
                continue
            rr = np.random.default_rng(SEED + epoch * 1000003 + start)
            v1 = [strings[i] for i in ids]
            v2 = [randomized_smiles(strings[i], rr) for i in ids]
            x1 = batch_matrix(v1)
            x2 = batch_matrix(v2)
            z1 = normalize_rows(x1 @ W)
            z2 = normalize_rows(x2 @ W)
            logits = np.clip((z1 @ z2.T) / TAU, -30, 30)
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-8)
            labels = np.arange(len(ids))
            loss = -np.log(np.maximum(prob[labels, labels], 1e-12)).mean()
            losses.append(float(loss))
            ds = prob
            ds[labels, labels] -= 1.0
            ds /= len(ids) * TAU
            dz1 = ds @ z2
            dz2 = ds.T @ z1
            dz1 -= z1 * np.sum(z1 * dz1, axis=1, keepdims=True)
            dz2 -= z2 * np.sum(z2 * dz2, axis=1, keepdims=True)
            gW = x1.T @ dz1 + x2.T @ dz2
            gW = np.clip(gW, -5.0, 5.0)
            W -= SSL_LR * (gW.astype(np.float32) / len(ids) + SSL_WD * W)
        history.append({
            "epoch": epoch + 1,
            "batches": int(len(losses)),
            "mean_info_nce": float(np.mean(losses)),
        })
        np.savez_compressed(SSL_CHECKPOINT, W=W, next_epoch=epoch + 1, corpus_size=len(strings))
        print("C157 SSL", history[-1], flush=True)
    return W, history


def embed(strings, W):
    vals = []
    for s in strings:
        rr = np.random.default_rng(SEED + stable_hash(s) % 1000003)
        vals.append(0.5 * (
            hash_vector(s) @ W
            + hash_vector(randomized_smiles(s, rr)) @ W
        ))
    return normalize_rows(np.asarray(vals, dtype=np.float32)).astype(np.float64)


print("C157 loading official PI1M and competition covariate SMILES", flush=True)
pi = pd.read_csv(ROOT / "ppp-round-2/PI1M.csv", usecols=["SMILES"])
pi_strings = pd.unique(pi["SMILES"].dropna().astype(str)).tolist()
pi_strings.sort(key=stable_hash)
train = pd.read_csv(ROOT / "ppp-round-2/train.csv")
test = pd.read_csv(ROOT / "ppp-round-2/test.csv")
official_strings = pd.unique(
    pd.concat([train["smiles"], test["smiles"]], ignore_index=True)
    .dropna()
    .astype(str)
).tolist()
corpus = list(dict.fromkeys(pi_strings[:SSL_ROWS] + official_strings))
print("C157 unlabeled corpus", {
    "pi1m_unique": len(pi_strings),
    "pi1m_used": min(SSL_ROWS, len(pi_strings)),
    "official_added": len(official_strings),
    "total": len(corpus),
}, flush=True)
W, ssl_history = train_contrastive(corpus)

# Replay the clean C144/C148 parent construction. These scripts use only
# official labels and structure features; neither reads local_eval/test_external_labels.
c144 = runpy.run_path(str(ROOT / "tools/round2_c144_log_ionic_reconstruction.py"))
c148 = runpy.run_path(str(ROOT / "tools/round2_c148_corrected_tree_ionic.py"))
F = c148["F"]
L = c148["L"]
OBS = c148["OBS"]
CUR = c148["CUR"].copy()
test = c148["test"]
path2 = c148["path2"]
groups = np.asarray(F["scaffolds"], dtype=object)
canon_strings = list(F["canon_list"])
E = embed(canon_strings, W)

# Corrected C143-style Ei carrier: counterpart identity is used only where
# both official partners are observed.
both_ei = OBS[:, ti["ei"]] & OBS[:, ti["egc"]] & OBS[:, ti["eea"]]
CUR[both_ei, ti["ei"]] = (
    0.5 * CUR[both_ei, ti["ei"]]
    + 0.5 * (L[both_ei, ti["egc"]] + L[both_ei, ti["eea"]])
)

# C144 clean OOF carrier for EPS/Nc, then C148 ExtraTrees EPS-only OOF arm.
for target in ("eps", "nc"):
    rows = np.where(OBS[:, ti[target]])[0]
    CUR[rows, ti[target]] = c144["oof"][target]
pair_rows = c148["pair_rows"]
ionic = c148["ionic"]
X148 = c148["X"]
eps_rows = np.where(OBS[:, ti["eps"]])[0]
eps_pos = {r: i for i, r in enumerate(eps_rows)}
eps_oof = CUR[eps_rows, ti["eps"]].copy()
for tr, va in KFold(5, shuffle=True, random_state=SEED + 148).split(pair_rows):
    m = ExtraTreesRegressor(
        n_estimators=700,
        max_features=.5,
        min_samples_leaf=2,
        n_jobs=10,
        random_state=SEED,
    )
    m.fit(X148[pair_rows[tr]], ionic[tr])
    for r, ip in zip(pair_rows[va], m.predict(X148[pair_rows[va]])):
        eps_oof[eps_pos[r]] = L[r, ti["nc"]] ** 2 + ip
CUR[eps_rows, ti["eps"]] = eps_oof


def fit_ridge(x, y):
    return make_pipeline(StandardScaler(), Ridge(alpha=50.0)).fit(x, y)


def fit_et(x, y):
    return ExtraTreesRegressor(
        n_estimators=600,
        max_features=.7,
        min_samples_leaf=2,
        n_jobs=10,
        random_state=SEED,
    ).fit(x, y)


metrics = {}
selected = {}
oof_predictions = CUR.copy()
for target in ("ei", "nc", "eps"):
    j = ti[target]
    rows = np.where(OBS[:, j])[0]
    y = L[rows, j]
    parent = CUR[rows, j]
    ridge_oof = parent.copy()
    et_oof = parent.copy()
    folds = []
    for fold, (tr, va) in enumerate(
        GroupKFold(5).split(rows, groups=groups[rows]), 1
    ):
        residual = y[tr] - parent[tr]
        mr = fit_ridge(E[rows[tr]], residual)
        me = fit_et(E[rows[tr]], residual)
        ridge_oof[va] = parent[va] + RESIDUAL_BLEND * mr.predict(E[rows[va]])
        et_oof[va] = parent[va] + RESIDUAL_BLEND * me.predict(E[rows[va]])
        folds.append({
            "fold": fold,
            "fit_rows": int(len(tr)),
            "validation_rows": int(len(va)),
        })
    candidates = {"ridge": ridge_oof, "extra_trees": et_oof, "none": parent}
    scores = {name: r2(y, pred) for name, pred in candidates.items()}
    best = max(scores, key=scores.get)
    metrics[target] = {
        "parent_r2": scores["none"],
        "ridge_r2": scores["ridge"],
        "extra_trees_r2": scores["extra_trees"],
        "selected": best,
        "delta_selected": scores[best] - scores["none"],
        "folds": folds,
    }
    if best != "none":
        oof_predictions[rows, j] = candidates[best]
    selected[target] = best
    print("C157 OOF", target, json.dumps(metrics[target], sort_keys=True), flush=True)

parent_scores = []
candidate_scores = []
for target in TARGETS:
    rows = np.where(OBS[:, ti[target]])[0]
    parent_scores.append(r2(L[rows, ti[target]], CUR[rows, ti[target]]))
    candidate_scores.append(r2(L[rows, ti[target]], oof_predictions[rows, ti[target]]))
mean_parent = float(np.mean(parent_scores))
mean_candidate = float(np.mean(candidate_scores))
print("C157 aggregate OOF", json.dumps({
    "parent_mean": mean_parent,
    "candidate_mean": mean_candidate,
    "delta": mean_candidate - mean_parent,
}, sort_keys=True), flush=True)

# Enforce the lifecycle boundary: no full-data fitting or candidate generation
# after a clean-gate miss.  The first pilot predates this correction and its
# isolated CSV is retained only as a quarantined research artifact.
name = "R2-C157-pi1m-contrastive-weak-targets-LOCAL_DIAGNOSTIC_ONLY"
bankable_target = any(metrics[t]["delta_selected"] >= 0.01 for t in ("ei", "nc", "eps"))
complete_clean_gate = bankable_target and (mean_candidate - mean_parent >= 0.002)
if not complete_clean_gate:
    report = {
        "experiment": name,
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "state": "clean_gate_miss_no_full_data_fit",
        "metrics": metrics,
        "parent_mean_oof": mean_parent,
        "candidate_mean_oof": mean_candidate,
        "selected": selected,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / f"{name}-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("C157 clean gate miss; no full-data candidate generated", flush=True)
    raise SystemExit(0)

# Full-data heads are trained from official labels and OOF-parent residuals,
# then applied to the frozen C148 full-data candidate.
candidate = pd.read_csv(path2)
for target in ("ei", "nc", "eps"):
    method = selected[target]
    if method == "none":
        continue
    j = ti[target]
    rows = np.where(OBS[:, j])[0]
    residual = L[rows, j] - CUR[rows, j]
    model = fit_ridge(E[rows], residual) if method == "ridge" else fit_et(E[rows], residual)
    for i, row in test.iterrows():
        if row.target_type == target:
            fi = int(row.fi)
            candidate.loc[i, "target"] += RESIDUAL_BLEND * float(model.predict(E[fi:fi + 1])[0])

path = OUT / f"{name}.csv"
candidate.to_csv(path, index=False)
report = {
    "experiment": name,
    "official_only_fitting": True,
    "local_eval_read": False,
    "pretrained_weights": False,
    "mechanism": "from-scratch randomized-SMILES Siamese linear encoder on hash-ranked PI1M plus official unlabeled SMILES; target-specific Ridge/ExtraTrees residual heads with GroupKFold-by-scaffold and fixed 0.25 blend",
    "ssl": {
        "pi1m_unique": len(pi_strings),
        "pi1m_used": min(SSL_ROWS, len(pi_strings)),
        "official_added": len(official_strings),
        "corpus_total": len(corpus),
        "hash_dim": HASH_DIM,
        "latent": LATENT,
        "epochs": SSL_EPOCHS,
        "history": ssl_history,
    },
    "metrics": metrics,
    "parent_mean_oof": mean_parent,
    "candidate_mean_oof": mean_candidate,
    "candidate_path": str(path),
    "selected": selected,
    "elapsed_seconds": time.time() - started,
}
(OUT / f"{name}-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("C157 candidate", path, len(candidate), flush=True)
