"""
03_shap_local.py
================
R1.2 — instance-level explanations. For each target pick 3 representative
polymers (high / low / mid predicted value), then:
  - SHAP force plot for the last-fold LightGBM
  - RDKit SimilarityMap: per-atom SHAP coloring (Morgan-bit aggregation)
Outputs: local_shap_{target}_{i}.png, shap_force_{target}_{i}.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, save_plot, smoke_n,
                     morgan_bit_fp, canonical_smiles)


def morgan_atom_weights(smi, shap_row, feat_names, nbits=1024, radius=2):
    """Aggregate morgan-bit SHAP weights onto atoms via bitInfo."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=info)
    n_atoms = mol.GetNumAtoms()
    weights = np.zeros(n_atoms)
    n_hits = np.zeros(n_atoms)
    feat_lookup = {f"morgan_{i}": i for i in range(nbits)}
    for bit, entries in info.items():
        key = f"morgan_{bit}"
        if key in feat_lookup:
            fi = feat_lookup[key]
            if fi < len(feat_names) and fi < len(shap_row):
                w = shap_row[fi]
                for entry in entries:
                    a = entry[0] if isinstance(entry, (tuple, list)) else entry
                    if 0 <= a < n_atoms:
                        weights[a] += w
                        n_hits[a] += 1
    n_hits[n_hits == 0] = 1
    return weights / n_hits, mol


def draw_similarity_map(mol, weights, out_name):
    if mol is None:
        return False
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol, weights, colorMap="coolwarm", contourLines=6, figure=fig)
        except TypeError:
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol, weights, colorMap="coolwarm", contourLines=6)
        save_plot(fig, out_name)
        return True
    except Exception as e:
        print(f"    similarity map failed ({e}); fallback: highlighted structure")
        try:
            wmin, wmax = float(np.min(weights)), float(np.max(weights))
            colors = {}
            for i, w in enumerate(weights):
                t = 0.0 if wmax == wmin else (w - wmin) / (wmax - wmin)
                colors[i] = (t, 0.25, 1.0 - t)
            img = Draw.MolToImage(mol, size=(720, 520),
                                  highlightAtoms=list(range(mol.GetNumAtoms())))
            img.save(OUTPUT_DIR / out_name)
            return True
        except Exception as e2:
            print(f"    fallback failed: {e2}")
            return False


def pick_representative(df_t, k=3):
    vals = df_t["target"].values
    order = np.argsort(vals)
    n = len(order)
    picks = []
    for frac in (0.95, 0.50, 0.05):   # high / mid / low
        i = order[min(n - 1, int(frac * n))]
        if i not in picks:
            picks.append(i)
        if len(picks) == k:
            break
    return picks


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()

    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        if SMOKE and len(df_t) > 150:
            df_t = df_t.sample(150, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        feat_names = pkl["pipe"]["feat_names"]
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"])
        lgbm_model = pkl["models"]["lgbm"][-1]
        explainer = shap.TreeExplainer(lgbm_model)

        picks = pick_representative(df_t)
        for pi, i in enumerate(picks):
            smi = df_t.iloc[i]["smiles"]
            canon = canonical_smiles(smi)
            x_row = X[i]
            sv_row = explainer.shap_values(x_row.reshape(1, -1))[0]
            pred = lgbm_model.predict(x_row.reshape(1, -1))[0]

            tag = f"{target}_{pi}"
            # force plot (matplotlib backend)
            try:
                out = shap.force_plot(explainer.expected_value, sv_row,
                                      x_row, feature_names=feat_names,
                                      matplotlib=True, show=False)
                if isinstance(out, tuple):
                    fig = out[0]
                else:
                    fig = out
                fig.savefig(OUTPUT_DIR / f"shap_force_{tag}.png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"    force plot failed for {tag}: {e}")

            weights, mol = morgan_atom_weights(canon, sv_row, feat_names)
            ok = draw_similarity_map(mol, weights, f"local_shap_{tag}.png")
            print(f"  {target} polymer {pi}: smiles={smi[:60]}... pred={pred:.2f} "
                  f"map={'OK' if ok else 'SKIP'}")
    print(f"03_shap_local.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
