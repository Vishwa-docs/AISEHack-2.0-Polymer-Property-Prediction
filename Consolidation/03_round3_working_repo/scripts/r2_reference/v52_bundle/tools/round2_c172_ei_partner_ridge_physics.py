"""C172: fold-local physics-augmented Ei partner Ridge.

This is a clean follow-up to C171.  It keeps the C171 availability routing
and fixed shrinkages, but augments only the missing-partner models with the
official, unlabeled physics feature block.  Ei is excluded from all partner
features and every partner model is refit inside the outer Ei fold.
"""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path(
    "/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/"
    "b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"
)
OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
TI = {name: i for i, name in enumerate(TARGETS)}
SEED = 20260804


def load():
    f = pickle.loads((SCR / "features.pkl").read_bytes())
    idx, cmap = f["idx"], f["canon_map"]
    n = len(f["canon_list"])
    train = pd.read_csv(BASE / "train.csv")
    archive = pd.read_csv(BASE / "archive/train.csv")
    labels = np.full((n, 7), np.nan)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            z = frame[frame.target_type.eq(target)].copy()
            z["canon"] = z["smiles"].map(cmap)
            for canon, value in z.groupby("canon")["target"].mean().items():
                if canon in idx:
                    labels[idx[canon], j] = float(value)
    p1 = np.load(SCR / "out_clean_corrected/P1.npy")
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy")
    physics = pickle.loads((SCR / "physics.pkl").read_bytes())["M"]
    return f, labels, p1, parent, physics


def finite_block(x, clip):
    x = np.asarray(x, float).copy()
    x[~np.isfinite(x)] = np.nan
    med = np.nanmedian(x, axis=0)
    med[~np.isfinite(med)] = 0.0
    bad = ~np.isfinite(x)
    x[bad] = np.take(med, np.where(bad)[1])
    return np.clip(x, -clip, clip)


def main():
    started = time.time()
    f, labels, p1, pfinal, physics = load()
    observed = np.isfinite(labels)
    structure = finite_block(
        np.hstack([f["blocks"]["desc"], f["blocks"]["extra"], f["blocks"]["ipc"]]),
        1e6,
    )
    physics = finite_block(physics, 1e5)
    bank = np.where(observed, labels, p1)
    ei_rows = np.where(observed[:, TI["ei"]])[0]
    y = labels[ei_rows, TI["ei"]]
    parent = pfinal[ei_rows, TI["ei"]].copy()
    both = observed[ei_rows, TI["eea"]] & observed[ei_rows, TI["egc"]]
    parent[both] = 0.5 * parent[both] + 0.5 * (
        labels[ei_rows[both], TI["eea"]] + labels[ei_rows[both], TI["egc"]]
    )
    missing_egc = observed[ei_rows, TI["eea"]] & ~observed[ei_rows, TI["egc"]]
    missing_eea = ~observed[ei_rows, TI["eea"]] & observed[ei_rows, TI["egc"]]
    # These were fixed from C171's nested training-only coefficients before
    # this experiment: they are not reselected on C172 validation rows.
    weights = {"missing_egc": 0.40, "missing_eea": 0.65}
    out = np.zeros(len(ei_rows))
    folds = []
    splitter = KFold(5, shuffle=True, random_state=SEED)
    for train_pos, valid_pos in splitter.split(ei_rows):
        valid_global = ei_rows[valid_pos]
        pred = parent[valid_pos].copy()
        model_preds = {}
        for target in ("egc", "eea"):
            j = TI[target]
            other = [k for k in range(7) if k not in (j, TI["ei"])]
            x = np.hstack([structure, physics, bank[:, other], observed[:, other].astype(float)])
            fit_rows = np.where(observed[:, j])[0]
            alpha = 100.0 if target == "egc" else 10.0
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[fit_rows], labels[fit_rows, j])
            model_preds[target] = model.predict(x[valid_global])
        vm1 = observed[valid_global, TI["eea"]] & ~observed[valid_global, TI["egc"]]
        vm2 = ~observed[valid_global, TI["eea"]] & observed[valid_global, TI["egc"]]
        raw1 = labels[valid_global, TI["eea"]] + model_preds["egc"]
        raw2 = labels[valid_global, TI["egc"]] + model_preds["eea"]
        pred[vm1] += weights["missing_egc"] * (raw1[vm1] - parent[valid_pos][vm1])
        pred[vm2] += weights["missing_eea"] * (raw2[vm2] - parent[valid_pos][vm2])
        out[valid_pos] = pred
        folds.append({
            "train_rows": int(len(train_pos)),
            "valid_rows": int(len(valid_pos)),
            "missing_egc_valid": int(vm1.sum()),
            "missing_eea_valid": int(vm2.sum()),
        })

    ei_parent = float(r2_score(y, parent))
    ei_candidate = float(r2_score(y, out))
    c143 = {
        "tg": 0.9153573687299403,
        "egc": 0.9249057744047514,
        "egb": 0.9541078896040173,
        "eea": 0.9295377162544358,
        "nc": 0.9205304844029711,
        "eps": 0.877040481326824,
    }
    parent_mean = float(np.mean(list(c143.values()) + [ei_parent]))
    candidate_mean = float(np.mean(list(c143.values()) + [ei_candidate]))
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C172-ei-partner-ridge-physics",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "same_row_ei_label_as_feature": False,
        "mechanism": "fold-local physics-augmented partner Ridge with fixed C171 availability shrinkage",
        "partner_models": {
            "egc": {"alpha": 100.0, "physics_features": int(physics.shape[1])},
            "eea": {"alpha": 10.0, "physics_features": int(physics.shape[1])},
        },
        "availability": {
            "ei_rows": int(len(ei_rows)),
            "both_partners": int(both.sum()),
            "missing_egc": int(missing_egc.sum()),
            "missing_eea": int(missing_eea.sum()),
            "missing_both": int((~observed[ei_rows, TI["eea"]] & ~observed[ei_rows, TI["egc"]]).sum()),
        },
        "fixed_weights_from_c171": weights,
        "folds": folds,
        "metrics": {
            "ei": {
                "n": int(len(ei_rows)),
                "parent_r2": ei_parent,
                "candidate_r2": ei_candidate,
                "delta_r2": ei_candidate - ei_parent,
            },
            "composite": {
                "parent_mean_r2": parent_mean,
                "candidate_mean_r2": candidate_mean,
                "delta_mean_r2": candidate_mean - parent_mean,
            },
        },
        "gate": {
            "ei_gain_at_least_0.010": bool(ei_candidate - ei_parent >= 0.010),
            "positive_fold_local_route": bool(ei_candidate > ei_parent),
            "passed": bool(ei_candidate - ei_parent >= 0.010),
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2-C172-ei-partner-ridge-physics-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
