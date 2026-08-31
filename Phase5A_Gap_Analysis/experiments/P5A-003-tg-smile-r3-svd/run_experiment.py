#!/usr/bin/env python3
"""P5A-003 - Tg representation at scale: char hashing + TruncatedSVD(48) fitted on smile_r3 sample, appended as features for all targets. Clean lane: reads ONLY official train.csv / test.csv
(plus smile_r3.csv where stated). Fixed seeds. Grouped folds."""
import argparse, json, os
import numpy as np
from scipy import sparse as sp
import exp_core as core
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD

CONFIG = {
    "exp_id": "P5A-003",
    "focus": "Tg representation at scale: char hashing + TruncatedSVD(48) fitted on smile_r3 sample, appended as features for all targets",
    "svd_read_cap": 1000000,
    "svd_n_sample": 400000,
    "svd_dims": 48,
    "n_estimators": 300,
    "char_max_features": 4096,
    "morgan_nbits": 1024,
    "clip": True,
}

def svd_block(data_dir, train_smiles, test_smiles, cfg):
    import pandas as pd
    cap = 100000 if cfg.get("smoke") else cfg.get("svd_read_cap", 1000000)
    n = 30000 if cfg.get("smoke") else cfg.get("svd_n_sample", 400000)
    df = pd.read_csv(os.path.join(data_dir, "smile_r3.csv"), nrows=cap)
    sample = df["smiles"].sample(n=min(n, len(df)), random_state=core.SEED).tolist()
    hv = HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=2 ** 18,
                           alternate_sign=False)
    H = hv.fit_transform(sample)
    svd = TruncatedSVD(n_components=cfg.get("svd_dims", 48), random_state=core.SEED)
    svd.fit(H)
    tr = svd.transform(hv.transform(train_smiles)).astype(np.float32)
    te = svd.transform(hv.transform(test_smiles)).astype(np.float32)
    print("smile_r3 SVD block: %s / %s" % (tr.shape, te.shape))
    return tr, te

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
    tr, te = svd_block(a.data_dir, train.smiles.values, test.smiles.values, cfg)
    Xtr = sp.hstack([Xtr, sp.csr_matrix(tr)]).tocsr().astype(np.float32)
    Xte = sp.hstack([Xte, sp.csr_matrix(te)]).tocsr().astype(np.float32)

    sub, metrics = core.run_pipeline(train, test, Xtr, Xte, cfg)
    with open(os.path.join(a.output_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=1)
    print("DONE P5A-003 mean_oof_r2=%.4f" % metrics["mean_oof_r2"])

if __name__ == "__main__":
    main()
