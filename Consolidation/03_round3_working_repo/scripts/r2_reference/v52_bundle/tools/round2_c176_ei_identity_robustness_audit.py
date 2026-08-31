"""C176: clean robustness audit for the strongest observed-partner Ei arms.

This is an official-data-only audit.  It deliberately stops after clean OOF,
availability panels, repeated validation partitions, and scaffold-group
bootstrap.  It does not materialize a full-data candidate.  The two fixed
arms replay the already preregistered C160 identity route and C161
availability-gated Ei route; the purpose is to test whether the apparent Ei
gain is robust enough to bank.
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


def metric(y, base, candidate, rows):
    rows = np.asarray(rows, dtype=int)
    parent = r2(y[rows], base[rows])
    trial = r2(y[rows], candidate[rows])
    return {"n": int(len(rows)), "parent_r2": parent, "candidate_r2": trial, "delta_r2": trial - parent}


def labels_from_official(cmap, idx, n_structures):
    train = pd.read_csv(BASE / "train.csv")
    archive = pd.read_csv(BASE / "archive/train.csv")
    test = pd.read_csv(BASE / "test.csv")
    for frame in (train, archive, test):
        frame["canon"] = frame["smiles"].map(cmap)
        frame["fi"] = frame["canon"].map(idx).astype(int)
    labels = np.full((n_structures, len(TARGETS)), np.nan, dtype=float)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            values = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in values.items():
                labels[idx[canon], j] = float(value)
    return train, archive, test, labels


def make_parent(labels, observed, pfinal):
    parent = np.asarray(pfinal, dtype=float).copy()
    both = observed[:, TI["ei"]] & observed[:, TI["eea"]] & observed[:, TI["egc"]]
    parent[both, TI["ei"]] = 0.5 * parent[both, TI["ei"]] + 0.5 * (
        labels[both, TI["eea"]] + labels[both, TI["egc"]]
    )
    return parent, both


def make_arm(parent, labels, observed, p1, availability_gate):
    bank = np.where(observed, labels, p1)
    candidate = parent.copy()
    eea_mask = observed[:, TI["ei"]]
    egb_mask = observed[:, TI["egc"]]
    other = observed.copy()
    other[:, TI["ei"]] = False
    n_other = other.sum(axis=1)
    ei_mask = observed[:, TI["eea"]]
    if availability_gate:
        ei_mask &= n_other >= 2
    candidate[eea_mask, TI["eea"]] = 0.50 * parent[eea_mask, TI["eea"]] + 0.50 * (
        bank[eea_mask, TI["ei"]] - bank[eea_mask, TI["egc"]]
    )
    candidate[egb_mask, TI["egb"]] = 0.75 * parent[egb_mask, TI["egb"]] + 0.25 * (
        1.1178 * bank[egb_mask, TI["egc"]] - 0.9221
    )
    candidate[ei_mask, TI["ei"]] = 0.75 * parent[ei_mask, TI["ei"]] + 0.25 * (
        bank[ei_mask, TI["eea"]] + bank[ei_mask, TI["egc"]]
    )
    return candidate, {"eea": eea_mask, "egb": egb_mask, "ei": ei_mask, "n_other": n_other}


def repeated_fold_deltas(y, base, candidate, rows, repeats=5):
    rows = np.asarray(rows, dtype=int)
    values = []
    records = []
    for repeat in range(repeats):
        rng = np.random.default_rng(SEED + 1009 * repeat)
        shuffled = rows[rng.permutation(len(rows))]
        folds = np.array_split(shuffled, 5)
        fold_values = []
        for fold, valid in enumerate(folds, start=1):
            result = metric(y, base, candidate, valid)
            fold_values.append(result["delta_r2"])
            records.append({"repeat": repeat + 1, "fold": fold, **result})
        values.extend(fold_values)
    return records, values


def grouped_bootstrap(y, base, candidate, rows, groups, replicates=4000):
    rows = np.asarray(rows, dtype=int)
    labels = np.asarray(groups[rows], dtype=object)
    unique = np.unique(labels)
    rows_by_group = {group: rows[labels == group] for group in unique}
    rng = np.random.default_rng(SEED + 7717)
    deltas = np.empty(replicates, dtype=float)
    for i in range(replicates):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([rows_by_group[group] for group in chosen])
        deltas[i] = metric(y, base, candidate, sample)["delta_r2"]
    return {
        "groups": int(len(unique)),
        "replicates": int(replicates),
        "delta_median": float(np.quantile(deltas, 0.50)),
        "delta_lower_2p5": float(np.quantile(deltas, 0.025)),
        "delta_upper_97p5": float(np.quantile(deltas, 0.975)),
        "positive_fraction": float(np.mean(deltas > 0.0)),
    }


def main():
    features = pickle.loads((SCR / "features.pkl").read_bytes())
    idx, cmap = features["idx"], features["canon_map"]
    n_structures = len(features["canon_list"])
    train, archive, test, labels = labels_from_official(cmap, idx, n_structures)
    observed = np.isfinite(labels)
    p1 = np.load(SCR / "out_clean_corrected/P1.npy")
    pfinal = np.load(SCR / "out_clean_corrected/PFINAL.npy")
    if p1.shape != labels.shape or pfinal.shape != labels.shape:
        raise RuntimeError(f"official-only shape mismatch: p1={p1.shape}, pfinal={pfinal.shape}, labels={labels.shape}")
    parent, both_ei = make_parent(labels, observed, pfinal)
    c160, route160 = make_arm(parent, labels, observed, p1, availability_gate=False)
    c161, route161 = make_arm(parent, labels, observed, p1, availability_gate=True)
    groups = np.asarray(features["scaffolds"], dtype=object)
    if len(groups) != n_structures:
        raise RuntimeError("scaffold-group length mismatch")

    metrics = {"parent": {}, "c160": {}, "c161": {}}
    for target, j in TI.items():
        rows = np.flatnonzero(observed[:, j])
        metrics["parent"][target] = {"n": int(len(rows)), "r2": r2(labels[rows, j], parent[rows, j])}
        metrics["c160"][target] = metric(labels[:, j], parent[:, j], c160[:, j], rows)
        metrics["c161"][target] = metric(labels[:, j], parent[:, j], c161[:, j], rows)

    ei_rows = np.flatnonzero(observed[:, TI["ei"]])
    y_ei = labels[:, TI["ei"]]
    folds160, values160 = repeated_fold_deltas(y_ei, parent[:, TI["ei"]], c160[:, TI["ei"]], ei_rows)
    folds161, values161 = repeated_fold_deltas(y_ei, parent[:, TI["ei"]], c161[:, TI["ei"]], ei_rows)
    bootstrap160 = grouped_bootstrap(y_ei, parent[:, TI["ei"]], c160[:, TI["ei"]], ei_rows, groups)
    bootstrap161 = grouped_bootstrap(y_ei, parent[:, TI["ei"]], c161[:, TI["ei"]], ei_rows, groups)

    panels = {}
    panel_masks = {
        "both_eea_egc": observed[:, TI["eea"]] & observed[:, TI["egc"]],
        "eea_only": observed[:, TI["eea"]] & ~observed[:, TI["egc"]],
        "egc_only": ~observed[:, TI["eea"]] & observed[:, TI["egc"]],
        "neither": ~observed[:, TI["eea"]] & ~observed[:, TI["egc"]],
    }
    for name, mask in panel_masks.items():
        rows = ei_rows[mask[ei_rows]]
        panels[name] = {
            "n": int(len(rows)),
            "c160": metric(y_ei, parent[:, TI["ei"]], c160[:, TI["ei"]], rows),
            "c161": metric(y_ei, parent[:, TI["ei"]], c161[:, TI["ei"]], rows),
        }

    test_fi = test["fi"].to_numpy(dtype=int)
    test_ei = test.target_type.eq("ei").to_numpy()
    test_other = observed[test_fi].copy()
    test_other[:, TI["ei"]] = False
    test_n_other = test_other.sum(axis=1)
    test_eea = observed[test_fi, TI["eea"]]
    test_egc = observed[test_fi, TI["egc"]]
    test_support = {
        "ei_rows": int(np.sum(test_ei)),
        "both_eea_egc": int(np.sum(test_ei & test_eea & test_egc)),
        "eea_only": int(np.sum(test_ei & test_eea & ~test_egc)),
        "egc_only": int(np.sum(test_ei & ~test_eea & test_egc)),
        "neither": int(np.sum(test_ei & ~test_eea & ~test_egc)),
        "c160_ei_route": int(np.sum(test_ei & test_eea)),
        "c161_ei_route": int(np.sum(test_ei & test_eea & (test_n_other >= 2))),
    }

    c161_ei_delta = metrics["c161"]["ei"]["delta_r2"]
    c161_positive = int(np.sum(np.asarray(values161) > 0.0))
    c161_gate = {
        "ei_gain_at_least_0.010": bool(c161_ei_delta >= 0.010),
        "at_least_4_of_5_each_repeat_fold_positive": bool(c161_positive >= 20),
        "scaffold_bootstrap_lower_positive": bool(bootstrap161["delta_lower_2p5"] > 0.0),
        "all_ei_availability_panels_nonnegative": bool(all(panel["c161"]["delta_r2"] >= 0.0 for panel in panels.values())),
        "passed": bool(
            c161_ei_delta >= 0.010
            and c161_positive >= 20
            and bootstrap161["delta_lower_2p5"] > 0.0
            and all(panel["c161"]["delta_r2"] >= 0.0 for panel in panels.values())
        ),
    }
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C176-ei-identity-robustness-audit",
        "official_only_fitting": True,
        "test_structures_used_only_as_unlabeled_covariates": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "mechanism": "fixed C160 observed-partner physical identities replayed beside C161 Ei availability gate; no full-data candidate",
        "parent": "corrected PFINAL plus C143 direct Ei identity where both Eea and Egc are officially observed",
        "metrics": metrics,
        "ei_robustness": {
            "c160": {"repeat_fold_records": folds160, "positive_fold_count": int(np.sum(np.asarray(values160) > 0.0)), "bootstrap": bootstrap160},
            "c161": {"repeat_fold_records": folds161, "positive_fold_count": c161_positive, "bootstrap": bootstrap161},
        },
        "ei_availability_panels": panels,
        "test_support_audit": test_support,
        "gate": c161_gate,
        "decision": "bank_clean_component" if c161_gate["passed"] else "reject_robustness_gate_no_candidate",
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2-C176-ei-identity-robustness-audit-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
