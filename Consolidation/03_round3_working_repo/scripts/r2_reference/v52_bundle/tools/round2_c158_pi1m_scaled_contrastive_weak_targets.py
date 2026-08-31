"""C158: bounded scaled nonlinear PI1M contrastive representation.

This is a clean-gate-only experiment. It scans the complete official PI1M
corpus, trains a random-initialized nonlinear Siamese encoder on an exact
hash-ranked subset, and evaluates only fold-local residual heads. It must stop
before any full-data candidate or local_eval action unless the clean gates pass.
"""
from pathlib import Path
import hashlib
import json
import runpy
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.sparse import csr_matrix
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
HASH_DIM = 8192
HIDDEN = 512
LATENT = 128
PROJECTION = 64
SSL_ROWS = 250_000
SSL_EPOCHS = 5
BATCH = 256
TAU = 0.20
SSL_LR = 0.025
SSL_WD = 2e-4
RESIDUAL_BLEND = 0.25
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
CHECKPOINT = OUT / "R2-C158-pi1m-scaled-contrastive-LOCAL_DIAGNOSTIC_ONLY-checkpoint.npz"
started = time.time()


def stable_hash(s):
    return int.from_bytes(hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest(), "little")


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def randomized_smiles(s, rng):
    s = str(s)
    mol = Chem.MolFromSmiles(s)
    if mol is None or mol.GetNumAtoms() < 2:
        return s
    order = np.arange(mol.GetNumAtoms())
    rng.shuffle(order)
    try:
        return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order.tolist()), canonical=False, isomericSmiles=True)
    except Exception:
        return s


def hashed_items(s):
    counts = {}
    s = str(s)
    for n in (1, 2, 3, 4, 5):
        for i in range(max(0, len(s) - n + 1)):
            h = 2166136261
            for ch in s[i:i + n]:
                h = ((h ^ ord(ch)) * 16777619) & 0xffffffff
            j = h % HASH_DIM
            counts[j] = counts.get(j, 0.0) + 1.0
    norm = np.sqrt(sum(v * v for v in counts.values()))
    if norm > 0:
        return [(j, v / norm) for j, v in counts.items()]
    return []


def sparse_batch(strings):
    rr = []; cc = []; dd = []
    for i, s in enumerate(strings):
        for j, v in hashed_items(s):
            rr.append(i); cc.append(j); dd.append(v)
    return csr_matrix((np.asarray(dd, dtype=np.float32), (rr, cc)), shape=(len(strings), HASH_DIM), dtype=np.float32)


def relu(x):
    return np.maximum(x, 0.0)


def normal_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def init_weights(rng):
    return (
        (rng.normal(0, np.sqrt(2 / HASH_DIM), (HASH_DIM, HIDDEN))).astype(np.float32),
        (rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, LATENT))).astype(np.float32),
        (rng.normal(0, np.sqrt(2 / LATENT), (LATENT, PROJECTION))).astype(np.float32),
        np.zeros(HIDDEN, dtype=np.float32),
        np.zeros(LATENT, dtype=np.float32),
    )


def forward(x, weights):
    W1, W2, W3, b1, b2 = weights
    h = relu(np.asarray(x @ W1) + b1)
    z = relu(h @ W2 + b2)
    p = normal_rows(z @ W3)
    return h, z, p


def save_checkpoint(weights, next_epoch, corpus_size, history):
    np.savez_compressed(CHECKPOINT, W1=weights[0], W2=weights[1], W3=weights[2], b1=weights[3], b2=weights[4], next_epoch=next_epoch, corpus_size=corpus_size, history=json.dumps(history))


def train_ssl(strings):
    rng = np.random.default_rng(SEED)
    weights = init_weights(rng)
    start_epoch = 0
    history = []
    if CHECKPOINT.exists():
        saved = np.load(CHECKPOINT, allow_pickle=False)
        if saved["W1"].shape == weights[0].shape and int(saved["corpus_size"]) == len(strings):
            weights = (saved["W1"], saved["W2"], saved["W3"], saved["b1"], saved["b2"])
            start_epoch = int(saved["next_epoch"])
            history = json.loads(str(saved["history"]))
            print("C158 resuming checkpoint", {"next_epoch": start_epoch, "corpus_size": len(strings)}, flush=True)
    order = np.arange(len(strings))
    for epoch in range(start_epoch, SSL_EPOCHS):
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH]
            if len(ids) < 8:
                continue
            rr = np.random.default_rng(SEED + epoch * 1000003 + start)
            x1 = sparse_batch([strings[i] for i in ids])
            x2 = sparse_batch([randomized_smiles(strings[i], rr) for i in ids])
            h1, z1, p1 = forward(x1, weights)
            h2, z2, p2 = forward(x2, weights)
            logits = np.clip((p1 @ p2.T) / TAU, -30, 30)
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits); prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-8)
            labels = np.arange(len(ids))
            losses.append(float(-np.log(np.maximum(prob[labels, labels], 1e-12)).mean()))
            ds = prob
            ds[labels, labels] -= 1.0
            ds /= len(ids) * TAU
            dp1 = ds @ p2; dp2 = ds.T @ p1
            dp1 -= p1 * np.sum(p1 * dp1, axis=1, keepdims=True)
            dp2 -= p2 * np.sum(p2 * dp2, axis=1, keepdims=True)
            W1, W2, W3, b1, b2 = weights
            gW3 = z1.T @ dp1 + z2.T @ dp2
            dz1 = (dp1 @ W3.T) * (z1 > 0)
            dz2 = (dp2 @ W3.T) * (z2 > 0)
            gW2 = h1.T @ dz1 + h2.T @ dz2
            dh1 = (dz1 @ W2.T) * (h1 > 0)
            dh2 = (dz2 @ W2.T) * (h2 > 0)
            gW1 = x1.T @ dh1 + x2.T @ dh2
            gb1 = dh1.sum(0) + dh2.sum(0)
            gb2 = dz1.sum(0) + dz2.sum(0)
            grads = [np.asarray(gW1), gW2, gW3, gb1, gb2]
            scale = max(1.0, max(float(np.linalg.norm(g)) for g in grads) / 10.0)
            weights = tuple((w - SSL_LR * (g.astype(np.float32) / scale / len(ids) + SSL_WD * w)).astype(np.float32) for w, g in zip(weights, grads))
        rec = {"epoch": epoch + 1, "batches": len(losses), "mean_info_nce": float(np.mean(losses))}
        history.append(rec)
        save_checkpoint(weights, epoch + 1, len(strings), history)
        print("C158 SSL", rec, flush=True)
    return weights, history


def embed(strings, weights):
    vals = []
    for s in strings:
        rr = np.random.default_rng(SEED + stable_hash(s) % 1000003)
        x1 = sparse_batch([s]); x2 = sparse_batch([randomized_smiles(s, rr)])
        z1 = forward(x1, weights)[1][0]
        z2 = forward(x2, weights)[1][0]
        vals.append(0.5 * (z1 + z2))
    return normal_rows(np.asarray(vals, dtype=np.float64))


print("C158 loading full official PI1M scan", flush=True)
pi = pd.read_csv(ROOT / "ppp-round-2/PI1M.csv", usecols=["SMILES"])
pi_strings = pd.unique(pi["SMILES"].dropna().astype(str)).tolist()
pi_strings.sort(key=stable_hash)
train = pd.read_csv(ROOT / "ppp-round-2/train.csv")
test = pd.read_csv(ROOT / "ppp-round-2/test.csv")
official_strings = pd.unique(pd.concat([train["smiles"], test["smiles"]], ignore_index=True).dropna().astype(str)).tolist()
corpus = list(dict.fromkeys(pi_strings[:SSL_ROWS] + official_strings))
print("C158 unlabeled corpus", {"pi1m_unique": len(pi_strings), "pi1m_used": min(SSL_ROWS, len(pi_strings)), "official_added": len(official_strings), "total": len(corpus)}, flush=True)
weights, ssl_history = train_ssl(corpus)

# Reconstruct the corrected C148/C144 OOF parent using official labels only.
c144 = runpy.run_path(str(ROOT / "tools/round2_c144_log_ionic_reconstruction.py"))
c148 = runpy.run_path(str(ROOT / "tools/round2_c148_corrected_tree_ionic.py"))
F = c148["F"]; L = c148["L"]; OBS = c148["OBS"]; CUR = c148["CUR"].copy(); test = c148["test"]
groups = np.asarray(F["scaffolds"], dtype=object); canon_strings = list(F["canon_list"]); E = embed(canon_strings, weights)
both_ei = OBS[:, ti["ei"]] & OBS[:, ti["egc"]] & OBS[:, ti["eea"]]
CUR[both_ei, ti["ei"]] = 0.5 * CUR[both_ei, ti["ei"]] + 0.5 * (L[both_ei, ti["egc"]] + L[both_ei, ti["eea"]])
for target in ("eps", "nc"):
    rows = np.where(OBS[:, ti[target]])[0]
    CUR[rows, ti[target]] = c144["oof"][target]
pair_rows = c148["pair_rows"]; ionic = c148["ionic"]; X148 = c148["X"]; path2 = c148["path2"]
eps_rows = np.where(OBS[:, ti["eps"]])[0]; eps_pos = {r: i for i, r in enumerate(eps_rows)}
eps_oof = CUR[eps_rows, ti["eps"]].copy()
for tr, va in KFold(5, shuffle=True, random_state=SEED + 148).split(pair_rows):
    m = ExtraTreesRegressor(n_estimators=700, max_features=.5, min_samples_leaf=2, n_jobs=10, random_state=SEED)
    m.fit(X148[pair_rows[tr]], ionic[tr])
    for r, ip in zip(pair_rows[va], m.predict(X148[pair_rows[va]])):
        eps_oof[eps_pos[r]] = L[r, ti["nc"]] ** 2 + ip
CUR[eps_rows, ti["eps"]] = eps_oof


def fit_ridge(x, y):
    return make_pipeline(StandardScaler(), Ridge(alpha=50.0)).fit(x, y)


def fit_et(x, y):
    return ExtraTreesRegressor(n_estimators=600, max_features=.7, min_samples_leaf=2, n_jobs=10, random_state=SEED).fit(x, y)


metrics = {}; selected = {}; oof_predictions = CUR.copy()
for target in ("ei", "nc", "eps"):
    j = ti[target]; rows = np.where(OBS[:, j])[0]; y = L[rows, j]; parent = CUR[rows, j]
    ridge_oof = parent.copy(); et_oof = parent.copy(); fold_meta = []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(rows, groups=groups[rows]), 1):
        residual = y[tr] - parent[tr]
        mr = fit_ridge(E[rows[tr]], residual); me = fit_et(E[rows[tr]], residual)
        ridge_oof[va] = parent[va] + RESIDUAL_BLEND * mr.predict(E[rows[va]])
        et_oof[va] = parent[va] + RESIDUAL_BLEND * me.predict(E[rows[va]])
        fold_meta.append({"fold": fold, "fit_rows": int(len(tr)), "validation_rows": int(len(va))})
    scores = {"none": r2(y, parent), "ridge": r2(y, ridge_oof), "extra_trees": r2(y, et_oof)}
    best = max(scores, key=scores.get)
    metrics[target] = {"parent_r2": scores["none"], "ridge_r2": scores["ridge"], "extra_trees_r2": scores["extra_trees"], "selected": best, "delta_selected": scores[best] - scores["none"], "folds": fold_meta}
    selected[target] = best
    if best != "none":
        oof_predictions[rows, j] = {"ridge": ridge_oof, "extra_trees": et_oof}[best]
    print("C158 OOF", target, json.dumps(metrics[target], sort_keys=True), flush=True)

parent_scores = []; candidate_scores = []
for target in TARGETS:
    rows = np.where(OBS[:, ti[target]])[0]
    parent_scores.append(r2(L[rows, ti[target]], CUR[rows, ti[target]]))
    candidate_scores.append(r2(L[rows, ti[target]], oof_predictions[rows, ti[target]]))
mean_parent = float(np.mean(parent_scores)); mean_candidate = float(np.mean(candidate_scores))
print("C158 aggregate OOF", json.dumps({"parent_mean": mean_parent, "candidate_mean": mean_candidate, "delta": mean_candidate - mean_parent}, sort_keys=True), flush=True)

name = "R2-C158-pi1m-scaled-contrastive-weak-targets-LOCAL_DIAGNOSTIC_ONLY"
bankable_target = any(metrics[t]["delta_selected"] >= .01 for t in ("ei", "nc", "eps"))
report = {"experiment": name, "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "state": "oof_complete_no_full_data_candidate", "mechanism": "from-scratch nonlinear 8192->512->128 encoder with 64-dimensional projection; five-epoch InfoNCE on fixed 250k hash-ranked PI1M rows plus official unlabeled covariates; Ridge/ExtraTrees residual heads under GroupKFold-by-scaffold", "ssl": {"pi1m_unique": len(pi_strings), "pi1m_used": min(SSL_ROWS, len(pi_strings)), "official_added": len(official_strings), "corpus_total": len(corpus), "hash_dim": HASH_DIM, "hidden": HIDDEN, "latent": LATENT, "projection": PROJECTION, "epochs": SSL_EPOCHS, "history": ssl_history}, "metrics": metrics, "parent_mean_oof": mean_parent, "candidate_mean_oof": mean_candidate, "selected": selected, "bankable_target_preliminary": bankable_target, "elapsed_seconds": time.time() - started}
(OUT / f"{name}-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("C158 stopped before full-data candidate; clean OOF artifact", OUT / f"{name}-oof.json", flush=True)

