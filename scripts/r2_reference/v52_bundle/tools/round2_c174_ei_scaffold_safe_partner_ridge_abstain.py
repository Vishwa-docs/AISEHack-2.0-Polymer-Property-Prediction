"""C174: scaffold-safe abstention child for the C173 Ei correction."""
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
    scaffolds = np.asarray(f["scaffolds"], dtype=object)[ei_rows]
    out = np.zeros(len(ei_rows))
    fold_records = []
    splitter = KFold(5, shuffle=True, random_state=SEED)

    for fold, (train_pos, valid_pos) in enumerate(splitter.split(ei_rows), 1):
        train_global = ei_rows[train_pos]
        valid_global = ei_rows[valid_pos]
        valid_set = set(valid_global.tolist())
        parent_train = parent[train_pos]
        parent_valid = parent[valid_pos]
        pred_train = parent_train.copy()
        pred_valid = parent_valid.copy()
        raw_train, raw_valid = {}, {}

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
            all_train_pred = model.predict(x[train_global])
            all_valid_pred = model.predict(x[valid_global])
            if target == "egc":
                raw_train["missing_egc"] = labels[train_global, TI["eea"]] + all_train_pred
                raw_valid["missing_egc"] = labels[valid_global, TI["eea"]] + all_valid_pred
            else:
                raw_train["missing_eea"] = labels[train_global, TI["egc"]] + all_train_pred
                raw_valid["missing_eea"] = labels[valid_global, TI["egc"]] + all_valid_pred

        # Build one correction for each mode, then accept only scaffold
        # families whose training-fold squared-error improvement is positive.
        route_counts = {"missing_egc": 0, "missing_eea": 0}
        accepted_scaffolds = {"missing_egc": 0, "missing_eea": 0}
        for name, mode in (("missing_egc", missing_egc), ("missing_eea", missing_eea)):
            train_mode = mode[train_pos]
            valid_mode = mode[valid_pos]
            train_delta = raw_train[name] - parent_train
            valid_delta = raw_valid[name] - parent_valid
            for scaffold in np.unique(scaffolds[train_pos][train_mode]):
                tm = train_mode & (scaffolds[train_pos] == scaffold)
                if int(tm.sum()) < 2:
                    continue
                # Positive means the correction lowers squared error on the
                # training fold for this scaffold family.
                improvement = np.mean(
                    (y[train_pos][tm] - parent_train[tm]) ** 2
                    - (y[train_pos][tm] - (parent_train[tm] + train_delta[tm])) ** 2
                )
                if improvement <= 0:
                    continue
                accepted_scaffolds[name] += 1
                vm = valid_mode & (scaffolds[valid_pos] == scaffold)
                pred_valid[vm] += weights[name] * valid_delta[vm]
                route_counts[name] += int(vm.sum())
                # Training rows are only used for acceptance accounting; the
                # reported prediction is always the untouched outer validation.

        out[valid_pos] = pred_valid
        fold_records.append({
            "fold": fold,
            "valid_rows": int(len(valid_pos)),
            "candidate_r2": r2(y[valid_pos], pred_valid),
            "parent_r2": r2(y[valid_pos], parent_valid),
            "delta_r2": r2(y[valid_pos], pred_valid) - r2(y[valid_pos], parent_valid),
            "accepted_scaffolds": accepted_scaffolds,
            "routed_validation_rows": route_counts,
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

    scaffolds_unique = np.unique(scaffolds)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(1000):
        sample = rng.choice(scaffolds_unique, size=len(scaffolds_unique), replace=True)
        take = np.concatenate([np.where(scaffolds == s)[0] for s in sample])
        boot.append(r2(y[take], out[take]) - r2(y[take], parent[take]))
    boot = np.asarray(boot)
    bootstrap = {
        "groups": int(len(scaffolds_unique)),
        "replicates": int(len(boot)),
        "delta_median": float(np.median(boot)),
        "delta_lower_2p5": float(np.quantile(boot, 0.025)),
        "delta_upper_97p5": float(np.quantile(boot, 0.975)),
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
        "experiment": "R2-C174-ei-scaffold-safe-partner-ridge-abstain",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "same_row_ei_label_as_feature": False,
        "mechanism": "C173 partner Ridge with training-scaffold error-based abstention; unseen/unreliable scaffold families fallback to parent",
        "fixed_weights_from_c173": weights,
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
    path = OUT / "R2-C174-ei-scaffold-safe-partner-ridge-abstain-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
