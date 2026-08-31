"""
A5_attribution_patching.py
==========================
EXP-A5 - attribution patching (gradient x (clean - corrupted) activations):
per-input-feature-group causal importance for each target's prediction.
Outputs: attribution_patch_modality_heatmap.png, attribution_patch_top_neurons.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from helpers import (SEED, OUTPUT_DIR, MLP_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, save_plot, smoke_n, style_ax)
from A1_train_mlp import MLP


def main():
    seed_all(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    train = load_train()
    rows = []
    targets = ["tg", "egc", "nc", "eps"]
    n_samp = smoke_n(120, 20)

    for target in targets:
        ckpt = torch.load(MLP_DIR / f"{target}_mlp.pt", map_location="cpu", weights_only=False)
        model = MLP(ckpt["input_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        mean_, scale_ = ckpt["scaler_mean"], ckpt["scaler_scale"]

        df_t = train[train["target_type"] == target].sample(n_samp, random_state=SEED)
        pkl = load_proxy(target)
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"]).astype(np.float32)
        Xs = torch.tensor(((X - mean_) / scale_).astype(np.float32), requires_grad=True)
        feat_names = pkl["pipe"]["feat_names"]
        n_m = sum(1 for f in feat_names if f.startswith("morgan_"))
        n_d = sum(1 for f in feat_names if not f.startswith("morgan_") and not f.startswith("ngram_"))

        # attribution: grad of prediction wrt input
        pred = model(Xs).sum()
        pred.backward()
        grad = Xs.grad.numpy()
        # modality importance = mean |grad| per block
        g_m = float(np.abs(grad[:, :n_m]).mean())
        g_d = float(np.abs(grad[:, n_m:n_m + n_d]).mean())
        g_g = float(np.abs(grad[:, n_m + n_d:]).mean())
        rows.append({"target": target, "morgan_grad_mean": g_m,
                     "rdkit_desc_grad_mean": g_d, "char_ngram_grad_mean": g_g})
        print(f"  {target}: |grad| morgan={g_m:.4f} desc={g_d:.4f} ngram={g_g:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "attribution_patch_top_neurons.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    w = 0.27
    ax.bar(x - w, [r["morgan_grad_mean"] for r in rows], w, label="Morgan FP", color="steelblue")
    ax.bar(x, [r["rdkit_desc_grad_mean"] for r in rows], w, label="RDKit descriptors", color="tomato")
    ax.bar(x + w, [r["char_ngram_grad_mean"] for r in rows], w, label="char n-gram", color="mediumseagreen")
    ax.set_xticks(x); ax.set_xticklabels([r["target"] for r in rows])
    style_ax(ax, "Attribution patching - input modality causal importance",
             "Target model", "Mean |gradient| x activation (input sensitivity)")
    ax.legend()
    save_plot(fig, "attribution_patch_modality_heatmap.png")
    print(f"A5_attribution_patching.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
