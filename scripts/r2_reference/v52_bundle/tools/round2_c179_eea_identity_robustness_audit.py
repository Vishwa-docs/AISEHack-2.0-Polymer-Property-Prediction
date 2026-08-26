"""C179: full clean robustness audit of the observed-Ei Eea identity arm.

The fixed route is Eea = Ei - Egc when Ei is officially observed, blended
50/50 with the corrected structure-only parent.  This script audits the
positive C160 point result with repeated folds, scaffold-group bootstrap, and
availability panels.  It stops before any full-data candidate or post-freeze
diagnostic.
"""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
TI = {name: i for i, name in enumerate(TARGETS)}
SEED = 20260804
started = time.time()


def r2(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def labels_from_official(cmap, idx, n):
    train = pd.read_csv(BASE / "train.csv")
    archive = pd.read_csv(BASE / "archive/train.csv")
    test = pd.read_csv(BASE / "test.csv")
    for frame in (train, archive, test):
        frame["canon"] = frame["smiles"].map(cmap)
        frame["fi"] = frame["canon"].map(idx).astype(int)
    labels = np.full((n, len(TARGETS)), np.nan, dtype=float)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            values = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in values.items():
                labels[idx[canon], j] = float(value)
    return train, archive, test, labels


def metric(y, parent, candidate, rows):
    rows = np.asarray(rows, dtype=int)
    p = r2(y[rows], parent[rows])
    c = r2(y[rows], candidate[rows])
    return {"n": int(len(rows)), "parent_r2": p, "candidate_r2": c, "delta_r2": c - p}


def bootstrap(y, parent, candidate, rows, groups, reps=4000):
    rows = np.asarray(rows, dtype=int)
    rg = np.asarray(groups[rows], dtype=object)
    unique = np.unique(rg)
    by_group = {g: rows[rg == g] for g in unique}
    rng = np.random.default_rng(SEED + 179)
    deltas = np.empty(reps)
    for i in range(reps):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([by_group[g] for g in chosen])
        deltas[i] = r2(y[sample], candidate[sample]) - r2(y[sample], parent[sample])
    return {"groups": int(len(unique)), "replicates": reps, "delta_median": float(np.quantile(deltas, .5)), "delta_lower_2p5": float(np.quantile(deltas, .025)), "delta_upper_97p5": float(np.quantile(deltas, .975)), "positive_fraction": float(np.mean(deltas > 0))}


def main():
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    idx, cmap = F["idx"], F["canon_map"]
    n = len(F["canon_list"])
    train, archive, test, labels = labels_from_official(cmap, idx, n)
    observed = np.isfinite(labels)
    p1 = np.load(SCR / "out_clean_corrected/P1.npy")
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    both_ei = observed[:, TI["ei"]] & observed[:, TI["eea"]] & observed[:, TI["egc"]]
    parent[both_ei, TI["ei"]] = .5 * parent[both_ei, TI["ei"]] + .5 * (labels[both_ei, TI["eea"]] + labels[both_ei, TI["egc"]])

    bank = np.where(observed, labels, p1)
    candidate = parent.copy()
    route = observed[:, TI["ei"]]
    candidate[route, TI["eea"]] = .5 * parent[route, TI["eea"]] + .5 * (bank[route, TI["ei"]] - bank[route, TI["egc"]])
    rows = np.flatnonzero(observed[:, TI["eea"]])
    y = labels[:, TI["eea"]]
    groups = np.asarray(F["scaffolds"], dtype=object)
    fold_records = []
    fold_values = []
    for repeat in range(5):
        rng = np.random.default_rng(SEED + 179 * (repeat + 1))
        shuffled = rows[rng.permutation(len(rows))]
        for fold, valid in enumerate(np.array_split(shuffled, 5), start=1):
            item = metric(y, parent[:, TI["eea"]], candidate[:, TI["eea"]], valid)
            fold_values.append(item["delta_r2"])
            fold_records.append({"repeat": repeat + 1, "fold": fold, **item})

    panels = {}
    for name, mask in {
        "ei_partner_available": route,
        "ei_partner_missing": ~route,
    }.items():
        selected = rows[mask[rows]]
        panels[name] = metric(y, parent[:, TI["eea"]], candidate[:, TI["eea"]], selected)

    test_fi = test["fi"].to_numpy(dtype=int)
    test_eea = test.target_type.eq("eea").to_numpy()
    test_support = {"eea_rows": int(np.sum(test_eea)), "ei_partner_available": int(np.sum(test_eea & observed[test_fi, TI["ei"]])), "ei_partner_missing": int(np.sum(test_eea & ~observed[test_fi, TI["ei"]]))}
    overall = metric(y, parent[:, TI["eea"]], candidate[:, TI["eea"]], rows)
    group_bootstrap = bootstrap(y, parent[:, TI["eea"]], candidate[:, TI["eea"]], rows, groups)
    positive_folds = int(np.sum(np.asarray(fold_values) > 0))
    gate = {
        "eea_gain_at_least_0.010": bool(overall["delta_r2"] >= .010),
        "at_least_4_of_5_each_repeat_fold_positive": bool(positive_folds >= 20),
        "scaffold_bootstrap_lower_positive": bool(group_bootstrap["delta_lower_2p5"] > 0),
        "all_availability_panels_nonnegative": bool(all(item["delta_r2"] >= 0 for item in panels.values())),
        "passed": bool(overall["delta_r2"] >= .010 and positive_folds >= 20 and group_bootstrap["delta_lower_2p5"] > 0 and all(item["delta_r2"] >= 0 for item in panels.values())),
    }
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C179-eea-identity-robustness-audit",
        "official_only_fitting": True,
        "test_structures_used_only_as_unlabeled_covariates": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "mechanism": "fixed observed-Ei Eea=Ei-Egc identity with 0.5 parent blend; structure-only P1 fallback for unavailable Egc",
        "metrics": {"eea": overall},
        "folds": fold_records,
        "positive_fold_count": positive_folds,
        "grouped_bootstrap": group_bootstrap,
        "availability_panels": panels,
        "test_support_audit": test_support,
        "gate": gate,
        "decision": "bank_clean_component" if gate["passed"] else "reject_robustness_gate_no_candidate",
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2-C179-eea-identity-robustness-audit-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
