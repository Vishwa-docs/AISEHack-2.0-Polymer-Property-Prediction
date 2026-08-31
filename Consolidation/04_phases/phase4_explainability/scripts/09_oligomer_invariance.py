"""
09_oligomer_invariance.py
=========================
R2.4 — polymer chain extension invariance: monomer vs dimer predictions.
Dimer = repeat unit doubled between the * attachment points (sanitized via
RDKit; skipped when construction fails). |delta| < 3σ pass rate reported.
Outputs: oligomer_invariance.csv, oligomer_invariance_plot.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_proxy, featurize, predict_ensemble, save_plot,
                     smoke_n, style_ax)


def build_dimer(smi):
    """Double the repeat unit between first and last '*' attachment points."""
    try:
        s = str(smi).replace("[*]", "*")
        first, last = s.find("*"), s.rfind("*")
        if first == -1 or first == last:
            return None
        inner = s[first + 1:last]
        if not inner:
            return None
        dimer = "*" + inner + inner + "*"
        mol = Chem.MolFromSmiles(dimer)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    N_MAX = smoke_n(50, 10)
    rows = []

    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        df_t = df_t[df_t["smiles"].astype(str).str.contains(r"\*", regex=True)]
        df_t = df_t.sample(min(N_MAX, len(df_t)), random_state=SEED) if len(df_t) else df_t
        pkl = load_proxy(target)
        tstd = float(train[train["target_type"] == target]["target"].std())

        for _, row in df_t.iterrows():
            dimer = build_dimer(row["smiles"])
            if dimer is None or dimer == row["smiles"]:
                continue
            X_m, _ = featurize([row["smiles"]], pipe=pkl["pipe"])
            X_d, _ = featurize([dimer], pipe=pkl["pipe"])
            pm = predict_ensemble(X_m, pkl)[0]
            pd_ = predict_ensemble(X_d, pkl)[0]
            rows.append({"target": target, "monomer_smiles": row["smiles"],
                         "dimer_smiles": dimer,
                         "pred_monomer": pm, "pred_dimer": pd_,
                         "delta": pd_ - pm,
                         "delta_sigma": (pd_ - pm) / tstd if tstd else float("nan")})
        print(f"  {target}: {sum(1 for r in rows if r['target'] == target)} valid dimers")

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUTPUT_DIR / "oligomer_invariance.csv", index=False)
        pass_rate = float((np.abs(df["delta_sigma"]) < 3.0).mean())
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["pred_monomer"], df["pred_dimer"], s=55, alpha=0.7,
                   color="steelblue")
        lims = [min(df["pred_monomer"].min(), df["pred_dimer"].min()) * 0.95 - 5,
                max(df["pred_monomer"].max(), df["pred_dimer"].max()) * 1.05 + 5]
        ax.plot(lims, lims, "k--", alpha=0.5, label="No change")
        style_ax(ax, "Oligomer Invariance — Monomer vs Dimer Predictions",
                 "Predicted property (monomer)", "Predicted property (dimer)")
        ax.legend()
        save_plot(fig, "oligomer_invariance_plot.png")
        print(f"pass rate (|delta| < 3σ): {pass_rate:.3f}")
    else:
        (OUTPUT_DIR / "oligomer_invariance.csv").write_text(
            "target,monomer_smiles,dimer_smiles,pred_monomer,pred_dimer,delta,delta_sigma\n")
        print("no valid dimers constructed — wrote empty CSV")
    print(f"09_oligomer_invariance.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
