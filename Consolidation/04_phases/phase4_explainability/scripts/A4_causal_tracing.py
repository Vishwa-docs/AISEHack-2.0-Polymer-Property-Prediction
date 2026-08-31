"""
A4_causal_tracing.py
====================
EXP-A4 - at which layer does the model "commit" to its prediction?
Noise the input, restore each layer's activations from the clean run, and
measure prediction recovery per layer.
Outputs: causal_tracing_{target}.png (tg, egc, nc, eps), causal_tracing_summary.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from helpers import (SEED, OUTPUT_DIR, MLP_DIR, SMOKE, seed_all, load_train,
                     load_proxy, rebuild_features, save_plot, smoke_n, style_ax)
from A1_train_mlp import MLP

LAYERS = ["block1", "block2", "block3"]


def run_trace(model, Xt, layer):
    """Corrupt input, then restore each layer's activations one at a time."""
    with torch.no_grad():
        pred_clean = model(Xt).item()
    noise = torch.randn_like(Xt) * 1.0
    Xn = Xt + noise
    with torch.no_grad():
        pred_corrupt = model(Xn).item()

    clean_acts = {}
    def grab_clean(name):
        def f(m, inp, out):
            clean_acts[name] = out.detach()
        return f
    hooks = []
    for l in LAYERS:
        hooks.append(getattr(model, l).register_forward_hook(grab_clean(l)))
    with torch.no_grad():
        model(Xt)
    for h in hooks:
        h.remove()

    recoveries = []
    for l in LAYERS:
        act = clean_acts[l]
        def patch(m, inp, out):
            return act
        h = getattr(model, l).register_forward_hook(patch)
        with torch.no_grad():
            pred_restored = model(Xn).item()
        h.remove()
        denom = (pred_clean - pred_corrupt) if abs(pred_clean - pred_corrupt) > 1e-9 else 1e-9
        rec = (pred_restored - pred_corrupt) / denom
        recoveries.append(float(rec))
    return pred_clean, pred_corrupt, recoveries


def main():
    seed_all(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    train = load_train()
    summary = []
    for target in ["tg", "egc", "nc", "eps"]:
        ckpt = torch.load(MLP_DIR / f"{target}_mlp.pt", map_location="cpu", weights_only=False)
        model = MLP(ckpt["input_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        mean_, scale_ = ckpt["scaler_mean"], ckpt["scaler_scale"]

        df_t = train[train["target_type"] == target].sample(
            smoke_n(100, 15), random_state=SEED)
        pkl = load_proxy(target)
        X = rebuild_features(df_t["smiles"].tolist(), pkl["pipe"]).astype(np.float32)
        Xs = torch.tensor(((X - mean_) / scale_).astype(np.float32))
        recs = []
        for i in range(len(Xs)):
            _, _, rec = run_trace(model, Xs[i:i + 1], "block2")
            recs.append(rec)
        recs = np.array(recs)  # (n, 3)
        for j, l in enumerate(LAYERS):
            summary.append({"target": target, "layer": l,
                            "mean_recovery": float(recs[:, j].mean())})
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(LAYERS, recs.mean(axis=0), "o-", color="steelblue", markersize=9)
        ax.fill_between(LAYERS, recs.mean(axis=0) - recs.std(axis=0),
                        recs.mean(axis=0) + recs.std(axis=0), alpha=0.2, color="steelblue")
        style_ax(ax, f"Causal tracing - {target.upper()}",
                 "Restored layer", "Prediction recovery fraction")
        ax.axhline(0, color="gray", ls=":")
        save_plot(fig, f"causal_tracing_{target}.png")
        print(f"  {target}: " + " ".join(f"{l}={recs[:, j].mean():.2f}" for j, l in enumerate(LAYERS)))
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "causal_tracing_summary.csv", index=False)
    print(f"A4_causal_tracing.py DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
