"""
07_smiles_invariance.py
=======================
R2.1 + R2.2 — prediction invariance across K randomized SMILES per polymer.
Key distinction: Morgan/RDKit features are graph-invariant; char n-grams are
string-sensitive. Variants are featurized WITHOUT canonicalization so the
n-gram sensitivity is exposed honestly.
Outputs: smiles_invariance_per_target.csv, smiles_invariance_boxplot.png,
         smiles_invariance_violation_rate.csv, canonicalization_check.txt
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import (SEED, TARGETS, OUTPUT_DIR, SMOKE, seed_all, load_train,
                     load_test, load_proxy, featurize, predict_ensemble,
                     random_smiles, canonical_smiles, save_plot, smoke_n, style_ax)


def main():
    seed_all(SEED)
    t0 = time.time()
    train = load_train()
    test = load_test()

    K = smoke_n(30, 8)
    N_POLYMERS = smoke_n(500, 20)
    per_target = {}
    viol_rows = []
    viol_graph_rows = []
    audit_lines = ["# Canonicalization Audit", "#",
                   "# For every randomized variant of a polymer, RDKit canonicalization",
                   "# must reduce it to exactly one representation (isomeric, unique).",
                   "#"]

    # canonicalization audit on 100 test polymers x 5 variants
    for _, row in test.head(smoke_n(100, 20)).iterrows():
        variants = random_smiles(row["smiles"], 5)
        canon_forms = {canonical_smiles(v) for v in variants}
        status = "OK" if len(canon_forms) == 1 else "MISMATCH"
        audit_lines.append(f"id={int(row['id'])} status={status} "
                           f"n_variants={len(variants)} canonical={list(canon_forms)[0][:80] if canon_forms else 'N/A'}")
    (OUTPUT_DIR / "canonicalization_check.txt").write_text("\n".join(audit_lines) + "\n")

    for target in TARGETS:
        df_t = train[train["target_type"] == target].copy()
        df_t["canonical"] = df_t["smiles"].apply(canonical_smiles)
        n_poly = min(N_POLYMERS, len(df_t))
        sample = df_t.sample(n_poly, random_state=SEED).reset_index(drop=True)
        pkl = load_proxy(target)
        tstd = float(train[train["target_type"] == target]["target"].std())

        rows = []
        all_stds = []
        for i, row in sample.iterrows():
            variants = random_smiles(row["smiles"], K)
            if len(variants) < 2:
                continue
            X_var, _ = featurize(variants, pipe=pkl["pipe"], canonicalize=False)
            preds = predict_ensemble(X_var, pkl)
            X_can, _ = featurize([row["smiles"]], pipe=pkl["pipe"], canonicalize=True)
            pred_can = predict_ensemble(X_can, pkl)[0]

            # graph-only invariance: fix the char n-gram columns at the canonical
            # values so only graph features (Morgan + RDKit descriptors) vary
            feat_names = pkl["pipe"]["feat_names"]
            n_m = sum(1 for f in feat_names if f.startswith("morgan_"))
            n_d = sum(1 for f in feat_names if not f.startswith("morgan_") and not f.startswith("ngram_"))
            ng_start = n_m + n_d
            X_var_graph = X_var.copy()
            X_var_graph[:, ng_start:] = X_can[0, ng_start:]
            preds_graph = predict_ensemble(X_var_graph, pkl)

            std_p = float(np.std(preds))
            std_g = float(np.std(preds_graph))
            maxdev = float(np.max(np.abs(preds - pred_can)))
            v05 = float(np.mean(np.abs(preds - pred_can) > 0.5 * tstd))
            v10 = float(np.mean(np.abs(preds - pred_can) > 1.0 * tstd))
            v20 = float(np.mean(np.abs(preds - pred_can) > 2.0 * tstd))
            g05 = float(np.mean(np.abs(preds_graph - pred_can) > 0.5 * tstd))
            g10 = float(np.mean(np.abs(preds_graph - pred_can) > 1.0 * tstd))
            g20 = float(np.mean(np.abs(preds_graph - pred_can) > 2.0 * tstd))
            all_stds.append(std_p)
            rows.append({"polymer": i, "smiles": row["smiles"],
                         "n_variants": len(variants),
                         "pred_canonical": pred_can,
                         "mean_pred": float(np.mean(preds)),
                         "std_pred": std_p,
                         "std_pred_graph_only": std_g,
                         "max_dev": maxdev,
                         "viol_rate_0_5sigma": v05,
                         "viol_rate_1sigma": v10,
                         "viol_rate_2sigma": v20})
            viol_rows.append({"target": target, "polymer": i,
                              "viol_rate_0_5sigma": v05,
                              "viol_rate_1sigma": v10,
                              "viol_rate_2sigma": v20})
            viol_graph_rows.append({"target": target, "polymer": i,
                                    "viol_rate_0_5sigma": g05,
                                    "viol_rate_1sigma": g10,
                                    "viol_rate_2sigma": g20})

        df_res = pd.DataFrame(rows)
        df_res.to_csv(OUTPUT_DIR / f"smiles_invariance_{target}.csv", index=False)
        per_target[target] = {
            "n_polymers": len(df_res),
            "mean_std": float(np.mean(all_stds)) if all_stds else float("nan"),
            "std_over_polymers": float(np.std(all_stds)) if all_stds else float("nan"),
            "mean_max_dev": float(df_res["max_dev"].mean()) if len(df_res) else float("nan"),
            "mean_pred": float(df_res["mean_pred"].mean()) if len(df_res) else float("nan"),
            "target_train_std": tstd,
            "std_pct_of_train_std": (float(np.mean(all_stds)) / tstd * 100) if all_stds and tstd else float("nan"),
            "mean_std_graph_only": float(np.mean(df_res["std_pred_graph_only"])) if len(df_res) else float("nan"),
            "std_pct_graph_only": (float(np.mean(df_res["std_pred_graph_only"])) / tstd * 100) if len(df_res) and tstd else float("nan"),
        }
        print(f"  {target}: mean std = {per_target[target]['mean_std']:.4f} "
              f"({per_target[target]['std_pct_of_train_std']:.3f}% of train std) | "
              f"graph-only = {per_target[target]['mean_std_graph_only']:.4f} "
              f"({per_target[target]['std_pct_graph_only']:.3f}%)")

    pd.DataFrame(per_target).T.to_csv(OUTPUT_DIR / "smiles_invariance_per_target.csv")
    pd.DataFrame(viol_rows).to_csv(OUTPUT_DIR / "smiles_invariance_violation_rate.csv", index=False)
    if viol_graph_rows:
        pd.DataFrame(viol_graph_rows).to_csv(OUTPUT_DIR / "smiles_invariance_graph_violation_rate.csv", index=False)
        pd.DataFrame(viol_graph_rows).groupby("target")[
            ["viol_rate_0_5sigma", "viol_rate_1sigma", "viol_rate_2sigma"]].mean().to_csv(
            OUTPUT_DIR / "smiles_invariance_graph_violation_summary.csv")
    pd.DataFrame(viol_rows).groupby("target")[
        ["viol_rate_0_5sigma", "viol_rate_1sigma", "viol_rate_2sigma"]].mean().to_csv(
        OUTPUT_DIR / "smiles_invariance_violation_summary.csv")

    # boxplot: std across polymers per target
    fig, ax = plt.subplots(figsize=(11, 6))
    data = []
    for target in TARGETS:
        f = OUTPUT_DIR / f"smiles_invariance_{target}.csv"
        if f.exists():
            data.append(pd.read_csv(f)["std_pred"].values)
    if data:
        ax.boxplot(data)
        ax.set_xticklabels(TARGETS[:len(data)])
    style_ax(ax, "SMILES Invariance — prediction std across 30 randomized SMILES",
             "Target", "Std of predictions across variants")
    save_plot(fig, "smiles_invariance_boxplot.png")

    # violation-rate summary per target (mean across polymers)
    summ = pd.DataFrame(viol_rows).groupby("target")[
        ["viol_rate_0_5sigma", "viol_rate_1sigma", "viol_rate_2sigma"]].mean()
    summ.to_csv(OUTPUT_DIR / "smiles_invariance_violation_summary.csv")
    print(f"07_smiles_invariance.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
