"""C173: strict audit of the C171 Ei partner-Ridge route.

Every outer validation canonical group is excluded from both partner-model
fits, even when that validation group carries a partner label.  The route and
rounded shrinkages are fixed from C171.  This is a clean audit only: no full
test fit and no local_eval action occur here.
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


def finite_block(x, clip):
    x = np.asarray(x, float).copy()
    x[~np.isfinite(x)] = np.nan
    med = np.nanmedian(x, axis=0)
    med[~np.isfinite(med)] = 0.0
    bad = ~np.isfinite(x)
    x[bad] = np.take(med, np.where(bad)[1])
    return np.clip(x, -clip, clip)


def r2(y, p):
    return float(r2_score(np.asarray(y, float), np.asarray(p, float)))


def main():
    started = time.time()
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
    observed = np.isfinite(labels)
    p1 = np.load(SCR / "out_clean_corrected/P1.npy")
    pfinal = np.load(SCR / "out_clean_corrected/PFINAL.npy")
    blocks = f["blocks"]
    structure = finite_block(
        np.hstack([blocks["desc"], blocks["extra"], blocks["ipc"]]), 1e6
    )
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
    weights = {"missing_egc": 0.41, "missing_eea": 0.64}
    out = np.zeros(len(ei_rows))
    fold_records = []
    splitter = KFold(5, shuffle=True, random_state=SEED)

    for fold, (train_pos, valid_pos) in enumerate(splitter.split(ei_rows), 1):
        valid_global = ei_rows[valid_pos]
        valid_set = set(valid_global.tolist())
        pred = parent[valid_pos].copy()
        raw = {}
        for target in ("egc", "eea"):
            j = TI[target]
            other = [k for k in range(7) if k not in (j, TI["ei"])]
            x = np.hstack([structure, bank[:, other], observed[:, other].astype(float)])
            fit_rows = np.array(
                [row for row in np.where(observed[:, j])[0] if row not in valid_set],
                dtype=int,
            )
            alpha = 100.0 if target == "egc" else 10.0
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[fit_rows], labels[fit_rows, j])
            pred_partner = model.predict(x[valid_global])
            if target == "egc":
                raw["missing_egc"] = labels[valid_global, TI["eea"]] + pred_partner
            else:
                raw["missing_eea"] = labels[valid_global, TI["egc"]] + pred_partner

        vm1 = missing_egc[valid_pos]
        vm2 = missing_eea[valid_pos]
        pred[vm1] += weights["missing_egc"] * (raw["missing_egc"][vm1] - parent[valid_pos][vm1])
        pred[vm2] += weights["missing_eea"] * (raw["missing_eea"][vm2] - parent[valid_pos][vm2])
        out[valid_pos] = pred
        fold_records.append({
            "fold": fold,
            "fit_egc": int(np.sum(observed[:, TI["egc"]]) - np.sum(np.isin(np.where(observed[:, TI["egc"]])[0], valid_global))),
            "fit_eea": int(np.sum(observed[:, TI["eea"]]) - np.sum(np.isin(np.where(observed[:, TI["eea"]])[0], valid_global))),
            "valid_rows": int(len(valid_pos)),
            "parent_r2": r2(y[valid_pos], parent[valid_pos]),
            "candidate_r2": r2(y[valid_pos], pred),
            "delta_r2": r2(y[valid_pos], pred) - r2(y[valid_pos], parent[valid_pos]),
            "missing_egc_rows": int(vm1.sum()),
            "missing_eea_rows": int(vm2.sum()),
        })

    ei_parent = r2(y, parent)
    ei_candidate = r2(y, out)
    panel_rows = {
        "both_partners": both,
        "missing_egc": missing_egc,
        "missing_eea": missing_eea,
        "missing_both": ~observed[ei_rows, TI["eea"]] & ~observed[ei_rows, TI["egc"]],
    }
    panels = {}
    for name, mask in panel_rows.items():
        if int(mask.sum()) >= 2:
            panels[name] = {
                "n": int(mask.sum()),
                "parent_r2": r2(y[mask], parent[mask]),
                "candidate_r2": r2(y[mask], out[mask]),
                "delta_r2": r2(y[mask], out[mask]) - r2(y[mask], parent[mask]),
            }

    # A grouped bootstrap over scaffold labels is a conservative uncertainty
    # diagnostic, not a selection mechanism.
    scaffolds = np.asarray(f["scaffolds"], dtype=object)[ei_rows]
    unique_scaffolds = np.unique(scaffolds)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(1000):
        sampled = rng.choice(unique_scaffolds, size=len(unique_scaffolds), replace=True)
        take = np.concatenate([np.where(scaffolds == group)[0] for group in sampled])
        deltas.append(r2(y[take], out[take]) - r2(y[take], parent[take]))
    deltas = np.asarray(deltas, float)
    bootstrap = {
        "groups": int(len(unique_scaffolds)),
        "replicates": int(len(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_lower_2p5": float(np.quantile(deltas, 0.025)),
        "delta_upper_97p5": float(np.quantile(deltas, 0.975)),
    }

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
        "experiment": "R2-C173-ei-strict-partner-ridge-panels",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "same_row_ei_label_as_feature": False,
        "mechanism": "C171 fixed partner Ridge with all outer-validation canonical groups excluded from partner fits; rounded availability shrinkage; scaffold bootstrap and availability panels",
        "fixed_weights_from_c171": weights,
        "availability": {
            "ei_rows": int(len(ei_rows)),
            "both_partners": int(both.sum()),
            "missing_egc": int(missing_egc.sum()),
            "missing_eea": int(missing_eea.sum()),
            "missing_both": int((~observed[ei_rows, TI["eea"]] & ~observed[ei_rows, TI["egc"]]).sum()),
        },
        "folds": fold_records,
        "panels": panels,
        "grouped_bootstrap": bootstrap,
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
                "c162_eps_only_mean_r2": candidate_mean + (0.8927568748456813 - 0.877040481326824) / 7.0,
            },
        },
        "gate": {
            "ei_gain_at_least_0.010": bool(ei_candidate - ei_parent >= 0.010),
            "all_folds_positive": bool(all(x["delta_r2"] > 0 for x in fold_records)),
            "bootstrap_lower_positive": bool(bootstrap["delta_lower_2p5"] > 0),
            "all_panels_nonnegative": bool(all(x["delta_r2"] >= 0 for x in panels.values())),
            "passed": bool(
                ei_candidate - ei_parent >= 0.010
                and all(x["delta_r2"] > 0 for x in fold_records)
                and bootstrap["delta_lower_2p5"] > 0
                and all(x["delta_r2"] >= 0 for x in panels.values())
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2-C173-ei-strict-partner-ridge-panels-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
