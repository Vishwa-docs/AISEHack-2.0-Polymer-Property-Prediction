"""C165: fixed scaffold/family residual-bias correction for Ei and Nc."""
from pathlib import Path
import importlib.util
import json
import pickle
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
CLEAN_OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
started = time.time()


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def c162_helpers():
    path = ROOT / "tools/round2_c162_ionic_coordinate_ensemble.py"
    spec = importlib.util.spec_from_file_location("c162_helpers", path)
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); return mod


def make_families(smiles):
    # The family bins are deterministic SMILES motif screens. They are
    # intentionally label-free and avoid requiring an RDKit import in the
    # lightweight audit environment.
    patterns = [
        ("imide", r"C\(=O\)N.*C\(=O\)"),
        ("carbonate", r"O=C\(O[^)]*\)O"),
        ("sulfone", r"S\(=O\)\(=O\)"),
        ("siloxane", r"Si"),
        ("nitrile", r"C#N"),
        ("ester", r"C\(=O\)O"),
        ("ether", r"COC"),
        ("fluorinated", r"F"),
        ("aromatic_heterocycle", r"[nops]1"),
        ("aromatic", r"c1ccccc1"),
    ]
    out = []
    for value in smiles:
        text = str(value)
        hits = [name for name, pat in patterns if re.search(pat, text)]
        out.append("+".join(hits) if hits else "other")
    return np.asarray(out, dtype=object)


def c162_oof(mod, X, labels, parent, pair):
    eps_y = labels[pair, ti["eps"]]; nc_y = labels[pair, ti["nc"]]; z = np.log(eps_y - nc_y ** 2)
    eo = {k: parent[pair, ti["eps"]].copy() for k in ("ridge", "et", "hgb")}; no = {k: parent[pair, ti["nc"]].copy() for k in ("ridge", "et", "hgb")}
    from sklearn.model_selection import KFold
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + 162).split(pair), 1):
        for kind in ("ridge", "et", "hgb"):
            m = mod.model(kind, SEED + fold)
            if kind == "ridge":
                mu = X[pair[tr]].mean(0); sd = X[pair[tr]].std(0); sd[sd < 1e-12] = 1.0; m.fit((X[pair[tr]] - mu) / sd, z[tr]); pred = m.predict((X[pair[va]] - mu) / sd)
            else:
                m.fit(X[pair[tr]], z[tr]); pred = m.predict(X[pair[va]])
            ip = np.exp(np.clip(pred, -8, 4)); eo[kind][va] = nc_y[va] ** 2 + ip; no[kind][va] = np.sqrt(np.maximum(eps_y[va] - ip, .05 ** 2))
    return .5 * parent[pair, ti["eps"]] + .5 * np.mean(np.column_stack([eo[k] for k in ("ridge", "et", "hgb")]), axis=1), .5 * parent[pair, ti["nc"]] + .5 * np.mean(np.column_stack([no[k] for k in ("ridge", "et", "hgb")]), axis=1)


def main():
    mod = c162_helpers(); F = pickle.loads((SCR / "features.pkl").read_bytes()); P = pickle.loads((SCR / "physics.pkl").read_bytes()); G = pickle.loads((SCR / "pgfp.pkl").read_bytes())
    idx, cmap = F["idx"], F["canon_map"]; ns = len(F["canon_list"])
    train = pd.read_csv(BASE / "train.csv"); archive = pd.read_csv(BASE / "archive/train.csv")
    for frame in (train, archive): frame["canon"] = frame["smiles"].map(cmap)
    labels = np.full((ns, 7), np.nan)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items(): labels[idx[canon], j] = value
    obs = np.isfinite(labels); X = mod.features(F, P, G); base = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy(); P1 = np.load(SCR / "out_clean_corrected/P1.npy")
    both = obs[:, ti["ei"]] & obs[:, ti["eea"]] & obs[:, ti["egc"]]; base[both, ti["ei"]] = .5 * base[both, ti["ei"]] + .5 * (labels[both, ti["eea"]] + labels[both, ti["egc"]])
    pair = np.where(obs[:, ti["eps"]] & obs[:, ti["nc"]])[0]; c162_eps, c162_nc = c162_oof(mod, X, labels, base, pair)
    candidate = base.copy(); candidate[pair, ti["eps"]] = c162_eps; candidate[pair, ti["nc"]] = c162_nc
    families = make_families(F["canon_list"])
    family_records = {}
    for target in ("ei", "nc"):
        j = ti[target]; rows = np.where(obs[:, j])[0]; y = labels[rows, j]; parent = candidate[rows, j].copy(); residual = y - parent; out = parent.copy(); groups = np.asarray([str(x) for x in rows], dtype=object)
        fold_rows = []
        for fold, (tr, va) in enumerate(GroupKFold(5).split(rows, y, groups=groups), 1):
            medians = {}
            for fam in np.unique(families[rows[tr]]):
                rr = tr[families[rows[tr]] == fam]
                if len(rr) >= 5: medians[fam] = float(np.median(residual[rr]))
            corrections = np.array([.25 * medians.get(families[r], 0.0) for r in rows[va]])
            out[va] = parent[va] + corrections
            fold_rows.append({"fold": fold, "fit_rows": int(len(tr)), "supported_families": int(len(medians)), "validation_rows": int(len(va)), "routed_rows": int(np.sum(corrections != 0))})
        candidate[rows, j] = out; family_records[target] = fold_rows
    metrics = {}
    for j, target in enumerate(TARGETS):
        rows = np.where(obs[:, j])[0]; pr = r2(labels[rows, j], base[rows, j]); cr = r2(labels[rows, j], candidate[rows, j]); metrics[target] = {"n": int(len(rows)), "parent_r2": pr, "candidate_r2": cr, "delta_r2": cr - pr}
    mean_parent = float(np.mean([x["parent_r2"] for x in metrics.values()])); mean_candidate = float(np.mean([x["candidate_r2"] for x in metrics.values()])); deltas = np.array([x["delta_r2"] for x in metrics.values()])
    gate = {"mean_gain_at_least_0.002": bool(mean_candidate - mean_parent >= .002), "no_target_loss_worse_than_0.003": bool(np.min(deltas) >= -.003), "eps_at_least_c162_minus_0.003": bool(metrics["eps"]["candidate_r2"] >= .8927568748456813 - .003), "ei_or_nc_gain_at_least_0.010": bool(max(metrics["ei"]["delta_r2"], metrics["nc"]["delta_r2"]) >= .010), "passed": bool(mean_candidate - mean_parent >= .002 and np.min(deltas) >= -.003 and metrics["eps"]["candidate_r2"] >= .8927568748456813 - .003 and max(metrics["ei"]["delta_r2"], metrics["nc"]["delta_r2"]) >= .010)}
    report = {"schema_version": "ppp.round2.clean-oof.v1", "experiment": "R2-C165-family-bias-correction", "official_only_fitting": True, "local_eval_read": False, "pretrained_weights": False, "mechanism": "C162 EPS carrier plus fixed 0.25 scaffold-family median residual correction for Ei/Nc with n>=5 support", "families": sorted(set(families.tolist())), "folds": family_records, "metrics": metrics, "parent_mean_r2": mean_parent, "candidate_mean_r2": mean_candidate, "gate": gate, "elapsed_seconds": time.time() - started}
    CLEAN_OUT.mkdir(parents=True, exist_ok=True); (CLEAN_OUT / "R2-C165-family-bias-correction-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2), flush=True)
    if not gate["passed"]: print("C165 STOP: clean gate failed; no full-data fit and no score verification.", flush=True)


if __name__ == "__main__": main()
