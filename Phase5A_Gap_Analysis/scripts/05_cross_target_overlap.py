#!/usr/bin/env python3
"""Phase5A Gap Analysis - 05: cross-target polymer overlap (partner availability) and Tg difficulty.
Pure data analysis on official train/test - no oracle needed."""
import os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]

train = pd.read_csv(os.path.join(ROOT, "Dataset", "train.csv"))
test  = pd.read_csv(os.path.join(ROOT, "Dataset", "test.csv"))

def overlap_matrix(df, outname, title):
    sets = {t: set(df[df.target_type == t]["smiles"]) for t in TARGETS}
    mat = pd.DataFrame(index=TARGETS, columns=TARGETS, dtype=float)
    for a in TARGETS:
        for b in TARGETS:
            mat.loc[a, b] = len(sets[a] & sets[b])
    mat.to_csv(os.path.join(OUT, outname))
    print(title)
    print(mat.astype(int).to_string())
    print()
    return sets

s_train = overlap_matrix(train, "05_overlap_train.csv", "TRAIN: # SMILES shared between target pairs (same polymer, two labels)")
s_test  = overlap_matrix(test,  "05_overlap_test.csv",  "TEST:  # SMILES appearing under both target types")

# partner availability: for each test row of target t, does the same SMILES appear in TRAIN under another target?
part = []
for t in TARGETS:
    tt = test[test.target_type == t]
    other = set(train[train.target_type != t]["smiles"])
    n_partner = tt["smiles"].isin(other).sum()
    part.append(dict(target=t, test_rows=len(tt), partner_in_train= n_partner,
                     frac= n_partner / len(tt)))
pdf = pd.DataFrame(part)
pdf.to_csv(os.path.join(OUT, "05_partner_availability.csv"), index=False)
print("TEST rows whose SMILES has a DIFFERENT target's label in TRAIN (cross-property partner availability):")
print(pdf.round(4).to_string(index=False))
print()

# same-target train coverage for test rows (base retrieval)
cov = []
for t in TARGETS:
    tt = test[test.target_type == t]
    same = set(train[train.target_type == t]["smiles"])
    cov.append(dict(target=t, test_rows=len(tt), seen_same_target= tt["smiles"].isin(same).sum(),
                    frac_seen=tt["smiles"].isin(same).mean()))
cdf = pd.DataFrame(cov)
cdf.to_csv(os.path.join(OUT, "05_train_coverage.csv"), index=False)
print("TEST rows whose SMILES was seen in TRAIN under the SAME target (retrieval coverage):")
print(cdf.round(4).to_string(index=False))
print()

# Tg vs DFT polymer set separation
dft = set(train[train.target_type != "tg"]["smiles"])
tgt = set(train[train.target_type == "tg"]["smiles"])
print(f"TRAIN: Tg polymers {len(tgt)} | DFT-target polymers {len(dft)} | overlap {len(tgt & dft)}")
dft_t = set(test[test.target_type != "tg"]["smiles"])
tgt_t = set(test[test.target_type == "tg"]["smiles"])
print(f"TEST:  Tg polymers {len(tgt_t)} | DFT-target polymers {len(dft_t)} | overlap {len(tgt_t & dft_t)}")

# how many test rows share smiles with ANY train row (any target)
tr_all = set(train["smiles"])
print(f"\nTEST rows whose SMILES appears in train at all: {test['smiles'].isin(tr_all).sum()} / {len(test)} "
      f"({test['smiles'].isin(tr_all).mean()*100:.1f}%)")
tt = test["smiles"].isin(tr_all)
print(f"  of which Tg rows: {test[tt & (test.target_type=='tg')].shape[0]}")
print(f"  of which DFT rows: {test[tt & (test.target_type!='tg')].shape[0]}")
summary = f"""TRAIN multi-target polymers: {len(train.groupby('smiles')['target_type'].nunique())} unique smiles, {(train.groupby('smiles')['target_type'].nunique()>1).sum()} with >=2 targets
TEST multi-target polymers: {len(test.groupby('smiles')['target_type'].nunique())} unique smiles, {(test.groupby('smiles')['target_type'].nunique()>1).sum()} with >=2 targets
"""
with open(os.path.join(OUT, "05_summary.txt"), "w") as f:
    f.write(summary)
print(summary)
