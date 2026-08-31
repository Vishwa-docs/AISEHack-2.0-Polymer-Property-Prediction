#!/usr/bin/env python3
"""P5A-002 - Tg OOF residual kriging: Tanimoto kNN correction of tg predictions, alpha fitted on train OOF only. Clean lane: reads ONLY official train.csv / test.csv
(plus smile_r3.csv where stated). Fixed seeds. Grouped folds."""
import argparse, json, os
import numpy as np
from scipy import sparse as sp
import exp_core as core
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

CONFIG = {
    "exp_id": "P5A-002",
    "focus": "Tg OOF residual kriging: Tanimoto kNN correction of tg predictions, alpha fitted on train OOF only",
    "n_estimators": 300,
    "char_max_features": 4096,
    "morgan_nbits": 1024,
    "clip": True,
}

def kriging_hook(train, test, res, pa, fo, ft, config):
    t = "tg"
    idx = res[t]["idx"]
    y = res[t]["y"]
    mor_tr = core.morgan(train.smiles.values[idx])
    mor_te = core.morgan(test.smiles.values)
    resid = y - res[t]["oof"]
    nn = NearestNeighbors(n_neighbors=min(11, len(idx)), metric="jaccard", n_jobs=-1)
    nn.fit(mor_tr)
    dist_tr, ind_tr = nn.kneighbors(mor_tr)
    sim_tr = 1.0 - dist_tr
    corr_oof = np.zeros(len(idx))
    for i in range(len(idx)):
        w = sim_tr[i][1:]
        corr_oof[i] = np.sum(w * resid[ind_tr[i][1:]]) / max(np.sum(w), 1e-9)
    best = (0.0, r2_score(y, fo[t]))
    for a in np.linspace(0.0, 0.9, 19):
        r2v = r2_score(y, fo[t] + a * corr_oof)
        if r2v > best[1]:
            best = (a, r2v)
    alpha = best[0]
    dist_te, ind_te = nn.kneighbors(mor_te)
    sim_te = 1.0 - dist_te
    corr_te = np.zeros(len(test))
    for i in range(len(test)):
        w = sim_te[i]
        corr_te[i] = np.sum(w * resid[ind_te[i]]) / max(np.sum(w), 1e-9)
    fo[t] = fo[t] + alpha * corr_oof
    ft[t] = ft[t] + alpha * corr_te
    print("kriging tg: alpha=%.3f oof_r2=%.4f" % (alpha, r2_score(y, fo[t])))
    return fo, ft

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    cfg = dict(CONFIG)
    cfg["smoke"] = a.smoke
    cfg["output_dir"] = a.output_dir
    if a.smoke:
        cfg["n_estimators"] = 20
        cfg["char_max_features"] = 512
        cfg["morgan_nbits"] = 256
    train, test = core.load(a.data_dir)
    train = core.smoke_sample(train, cfg)
    Xtr, Xte = core.build_features(train.smiles.values, test.smiles.values, cfg)

    sub, metrics = core.run_pipeline(train, test, Xtr, Xte, cfg, post_hook=kriging_hook)
    with open(os.path.join(a.output_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=1)
    print("DONE P5A-002 mean_oof_r2=%.4f" % metrics["mean_oof_r2"])

if __name__ == "__main__":
    main()
