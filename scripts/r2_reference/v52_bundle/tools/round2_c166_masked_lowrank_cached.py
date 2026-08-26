"""C166: cached-feature masked low-rank multi-task residual audit.

The earlier C091 protocol never reached metrics because it recomputed expensive
fingerprint panels. This version uses the already materialized official
structure-only representation, preserving the scientific factors: shared
canonical-group folds, target-masked residual heads, and a fixed rank-3
coefficient factorization.
"""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
CLEAN_OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
RANK = 3
ALPHA = 30.0
WEIGHT = 0.5
started = time.time()


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    P = pickle.loads((SCR / "physics.pkl").read_bytes())
    G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    import importlib.util
    spec = importlib.util.spec_from_file_location("c162_helpers", ROOT / "tools/round2_c162_ionic_coordinate_ensemble.py")
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
    X = mod.features(F, P, G).astype(np.float64)
    idx, cmap = F["idx"], F["canon_map"]
    ns = len(F["canon_list"])
    train = pd.read_csv(BASE / "train.csv"); archive = pd.read_csv(BASE / "archive/train.csv")
    for frame in (train, archive): frame["canon"] = frame["smiles"].map(cmap)
    labels = np.full((ns, 7), np.nan)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items(): labels[idx[canon], j] = value
    obs = np.isfinite(labels)
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    both = obs[:, ti["ei"]] & obs[:, ti["eea"]] & obs[:, ti["egc"]]
    parent[both, ti["ei"]] = .5 * parent[both, ti["ei"]] + .5 * (labels[both, ti["eea"]] + labels[both, ti["egc"]])
    # Every canonical group receives one outer fold; all target cells for a
    # group are held out together, so no cross-property label from that group
    # can enter the shared coefficient fit.
    groups = np.asarray(F["canon_list"], dtype=object)
    gkf = GroupKFold(n_splits=5)
    group_fold = np.full(ns, -1, dtype=int)
    group_index = np.arange(ns)
    for fold, (_, va) in enumerate(gkf.split(group_index, groups=groups)):
        group_fold[va] = fold
    candidate = parent.copy()
    fold_meta = []
    target_fold = {target: [] for target in TARGETS}
    for fold in range(5):
        validation_group = group_fold == fold
        training_group = ~validation_group
        coefficients = np.zeros((7, X.shape[1]), dtype=np.float64)
        intercepts = np.zeros(7, dtype=np.float64)
        fitted = np.zeros(7, dtype=bool)
        scaler = StandardScaler().fit(X[training_group])
        Xtr_all = scaler.transform(X[training_group])
        Xva_all = scaler.transform(X[validation_group])
        train_global = np.where(training_group)[0]
        valid_global = np.where(validation_group)[0]
        train_pos = {r: k for k, r in enumerate(train_global)}
        valid_pos = {r: k for k, r in enumerate(valid_global)}
        for code, target in enumerate(TARGETS):
            rows = np.where(obs[:, code] & training_group)[0]
            if len(rows) < 8: continue
            y = labels[rows, code] - parent[rows, code]
            pos = np.asarray([train_pos[int(r)] for r in rows])
            mean = float(np.mean(y)); sd = float(np.std(y)); sd = sd if sd > 1e-8 else 1.0
            ridge = Ridge(alpha=ALPHA).fit(Xtr_all[pos], (y - mean) / sd)
            coefficients[code] = ridge.coef_ * sd
            intercepts[code] = float(ridge.intercept_ * sd + mean)
            fitted[code] = True
        rank = min(RANK, coefficients.shape[0], coefficients.shape[1])
        left, singular, right = np.linalg.svd(coefficients, full_matrices=False)
        lowrank = (left[:, :rank] * singular[:rank]) @ right[:rank]
        for code, target in enumerate(TARGETS):
            rows = np.where(obs[:, code] & validation_group)[0]
            if len(rows) == 0: continue
            pos = np.asarray([valid_pos[int(r)] for r in rows])
            pred_res = intercepts[code] + Xva_all[pos] @ lowrank[code]
            pred = parent[rows, code] + WEIGHT * pred_res
            candidate[rows, code] = pred
            target_fold[target].append({"fold": fold, "rows": int(len(rows)), "parent_r2": r2(labels[rows, code], parent[rows, code]), "candidate_r2": r2(labels[rows, code], pred), "delta_r2": r2(labels[rows, code], pred) - r2(labels[rows, code], parent[rows, code])})
        fold_meta.append({"fold": fold, "train_groups": int(np.sum(training_group)), "validation_groups": int(np.sum(validation_group)), "fitted_heads": [TARGETS[i] for i in range(7) if fitted[i]], "rank": rank})

    metrics = {}
    for j, target in enumerate(TARGETS):
        rows = np.where(obs[:, j])[0]
        pr = r2(labels[rows, j], parent[rows, j]); cr = r2(labels[rows, j], candidate[rows, j])
        metrics[target] = {"n": int(len(rows)), "parent_r2": pr, "candidate_r2": cr, "delta_r2": cr - pr, "folds": target_fold[target]}
    parent_mean = float(np.mean([x["parent_r2"] for x in metrics.values()])); candidate_mean = float(np.mean([x["candidate_r2"] for x in metrics.values()])); deltas = np.array([x["delta_r2"] for x in metrics.values()])
    gate = {"mean_gain_at_least_0.002": bool(candidate_mean - parent_mean >= .002), "no_target_loss_worse_than_0.003": bool(np.min(deltas) >= -.003), "weak_target_gain_at_least_0.010": bool(max(metrics["ei"]["delta_r2"], metrics["nc"]["delta_r2"], metrics["eps"]["delta_r2"]) >= .010), "passed": bool(candidate_mean - parent_mean >= .002 and np.min(deltas) >= -.003 and max(metrics["ei"]["delta_r2"], metrics["nc"]["delta_r2"], metrics["eps"]["delta_r2"]) >= .010)}
    report = {"schema_version": "ppp.round2.clean-oof.v1", "experiment": "R2-C166-masked-lowrank-cached", "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "same_group_label_lookup": False, "mechanism": "fixed rank-3 shared coefficient factorization over target-masked residual Ridge heads, cached official structure-only representation, canonical-group outer folds", "rank": RANK, "ridge_alpha": ALPHA, "residual_weight": WEIGHT, "feature_count": int(X.shape[1]), "folds": fold_meta, "metrics": metrics, "parent_mean_r2": parent_mean, "candidate_mean_r2": candidate_mean, "gate": gate, "elapsed_seconds": time.time() - started}
    CLEAN_OUT.mkdir(parents=True, exist_ok=True); (CLEAN_OUT / "R2-C166-masked-lowrank-cached-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2), flush=True)
    if not gate["passed"]: print("C166 STOP: clean gate failed; no full-data fit and no score verification.", flush=True)


if __name__ == "__main__": main()
