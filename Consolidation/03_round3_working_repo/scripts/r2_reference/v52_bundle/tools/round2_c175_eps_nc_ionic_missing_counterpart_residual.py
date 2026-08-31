"""C175: missing-counterpart ionic reconstruction for EPS/Nc.

Only paired official EPS/Nc rows provide the ionic coordinate target.  For
rows where one counterpart is absent, three fixed C162 ionic models are fit on
paired rows, with validation scaffold groups removed, and combined with the
structure-only P1 counterpart prediction.  The final reconstruction is fixed
at a half-parent/half-reconstruction blend.  This is a clean diagnostic; no
full-data or local_eval action is present.
"""
from pathlib import Path
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


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


def r2(y, p):
    return float(r2_score(np.asarray(y, float), np.asarray(p, float)))


def main():
    started = time.time()
    sys.path.insert(0, str(ROOT / "tools"))
    import round2_c162_ionic_coordinate_ensemble as c162

    f = pickle.loads((SCR / "features.pkl").read_bytes())
    physics = pickle.loads((SCR / "physics.pkl").read_bytes())
    pgfp = pickle.loads((SCR / "pgfp.pkl").read_bytes())
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
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    both_ei = observed[:, TI["ei"]] & observed[:, TI["eea"]] & observed[:, TI["egc"]]
    parent[both_ei, TI["ei"]] = 0.5 * parent[both_ei, TI["ei"]] + 0.5 * (
        labels[both_ei, TI["eea"]] + labels[both_ei, TI["egc"]]
    )
    x = c162.features(f, physics, pgfp)
    scaffolds = np.asarray(f["scaffolds"], dtype=object)
    pair = np.where(observed[:, TI["eps"]] & observed[:, TI["nc"]])[0]
    ionic = labels[pair, TI["eps"]] - labels[pair, TI["nc"]] ** 2
    if np.any(ionic <= 0):
        raise RuntimeError("non-positive ionic coordinate")
    log_ionic = np.log(ionic)
    candidate = parent.copy()
    target_reports = {}
    for target, counter in (("eps", "nc"), ("nc", "eps")):
        j, cj = TI[target], TI[counter]
        missing = np.where(observed[:, j] & ~observed[:, cj])[0]
        oof = parent[missing, j].copy()
        fold_records = []
        splitter = KFold(5, shuffle=True, random_state=SEED + j)
        for fold, (_, valid_pos) in enumerate(splitter.split(missing), 1):
            valid_rows = missing[valid_pos]
            val_scaffolds = set(scaffolds[valid_rows].tolist())
            fit_pair = pair[~np.isin(scaffolds[pair], list(val_scaffolds))]
            preds = []
            for kind in ("ridge", "et", "hgb"):
                model = c162.model(kind, SEED + fold)
                if kind == "ridge":
                    mu = x[fit_pair].mean(0)
                    sd = x[fit_pair].std(0)
                    sd[sd < 1e-12] = 1.0
                    model.fit((x[fit_pair] - mu) / sd, log_ionic[np.isin(pair, fit_pair)])
                    pred_log = model.predict((x[valid_rows] - mu) / sd)
                else:
                    model.fit(x[fit_pair], log_ionic[np.isin(pair, fit_pair)])
                    pred_log = model.predict(x[valid_rows])
                preds.append(np.exp(np.clip(pred_log, -8, 4)))
            ionic_hat = np.mean(np.column_stack(preds), axis=1)
            counterpart = p1[valid_rows, cj]
            if target == "eps":
                raw = counterpart ** 2 + ionic_hat
            else:
                raw = np.sqrt(np.maximum(counterpart - ionic_hat, 0.05 ** 2))
            # Fixed half-parent blend, matching the C162 algebraic deployment.
            oof[valid_pos] = 0.5 * oof[valid_pos] + 0.5 * raw
            fold_records.append({
                "fold": fold,
                "fit_pair_rows": int(len(fit_pair)),
                "validation_rows": int(len(valid_rows)),
                "validation_scaffold_groups": int(len(val_scaffolds)),
            })
        candidate[missing, j] = oof
        delta = r2(labels[missing, j], oof) - r2(labels[missing, j], parent[missing, j])
        target_reports[target] = {
            "missing_counterpart_rows": int(len(missing)),
            "parent_r2": r2(labels[missing, j], parent[missing, j]),
            "candidate_r2": r2(labels[missing, j], oof),
            "delta_r2": delta,
            "folds": fold_records,
        }

    # Scaffold bootstrap is reported separately for each missing-counterpart
    # panel; it is not used to tune the blend.
    bootstrap = {}
    rng = np.random.default_rng(SEED)
    for target, counter in (("eps", "nc"), ("nc", "eps")):
        j, cj = TI[target], TI[counter]
        rows = np.where(observed[:, j] & ~observed[:, cj])[0]
        groups = np.unique(scaffolds[rows])
        vals = []
        for _ in range(1000):
            sample = rng.choice(groups, size=len(groups), replace=True)
            take = np.concatenate([np.where(scaffolds[rows] == g)[0] for g in sample])
            vals.append(
                r2(labels[rows[take], j], candidate[rows[take], j])
                - r2(labels[rows[take], j], parent[rows[take], j])
            )
        vals = np.asarray(vals)
        bootstrap[target] = {
            "groups": int(len(groups)),
            "replicates": int(len(vals)),
            "delta_median": float(np.median(vals)),
            "delta_lower_2p5": float(np.quantile(vals, 0.025)),
            "delta_upper_97p5": float(np.quantile(vals, 0.975)),
        }

    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C175-eps-nc-ionic-missing-counterpart-residual",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "same_row_hidden_counterpart_as_feature": False,
        "mechanism": "scaffold-excluded paired ionic-coordinate models plus structure-only counterpart fallback on missing EPS/Nc rows",
        "paired_rows": int(len(pair)),
        "target_panels": target_reports,
        "grouped_bootstrap": bootstrap,
        "metrics": {
            target: target_reports[target] for target in ("eps", "nc")
        },
        "gate": {
            "eps_gain_at_least_0.010": bool(target_reports["eps"]["delta_r2"] >= 0.010),
            "nc_gain_at_least_0.010": bool(target_reports["nc"]["delta_r2"] >= 0.010),
            "eps_bootstrap_lower_positive": bool(bootstrap["eps"]["delta_lower_2p5"] > 0),
            "nc_bootstrap_lower_positive": bool(bootstrap["nc"]["delta_lower_2p5"] > 0),
            "passed": bool(
                target_reports["eps"]["delta_r2"] >= 0.010
                or target_reports["nc"]["delta_r2"] >= 0.010
            ) and bool(
                bootstrap["eps"]["delta_lower_2p5"] > 0
                or bootstrap["nc"]["delta_lower_2p5"] > 0
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2-C175-eps-nc-ionic-missing-counterpart-residual-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
