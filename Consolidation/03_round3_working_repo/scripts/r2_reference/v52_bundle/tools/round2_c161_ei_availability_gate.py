"""C161: availability-gated Ei arm on top of the C160 physical composite.

The gate is fixed before any post-freeze scoring: the Ei identity arm is used
only when the canonical polymer has at least two other official properties
available. Eea and Egb keep the C160 fixed observed-partner routes.
"""
from pathlib import Path
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
CLEAN_OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
LOCAL_EVAL_OUT = ROOT / "experiments/LOCAL_DIAGNOSTIC_ONLY"
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
SEED = 20260804
started = time.time()


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    idx, cmap = F["idx"], F["canon_map"]
    ns = len(F["canon_list"])
    train = pd.read_csv(BASE / "train.csv")
    archive = pd.read_csv(BASE / "archive/train.csv")
    test = pd.read_csv(BASE / "test.csv")
    for frame in (train, archive, test):
        frame["canon"] = frame["smiles"].map(cmap)
        frame["fi"] = frame["canon"].map(idx).astype(int)
    labels = np.full((ns, 7), np.nan)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items():
                labels[idx[canon], j] = value
    obs = np.isfinite(labels)
    p1 = np.load(SCR / "out_clean_corrected/P1.npy")
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    both_ei = obs[:, ti["ei"]] & obs[:, ti["eea"]] & obs[:, ti["egc"]]
    parent[both_ei, ti["ei"]] = 0.5 * parent[both_ei, ti["ei"]] + 0.5 * (labels[both_ei, ti["eea"]] + labels[both_ei, ti["egc"]])
    bank = np.where(obs, labels, p1)

    candidate = parent.copy()
    eea_mask = obs[:, ti["ei"]]
    egb_mask = obs[:, ti["egc"]]
    other_for_ei = obs.copy()
    other_for_ei[:, ti["ei"]] = False
    n_other = other_for_ei.sum(1)
    ei_mask = obs[:, ti["eea"]] & (n_other >= 2)
    candidate[eea_mask, ti["eea"]] = 0.50 * parent[eea_mask, ti["eea"]] + 0.50 * (bank[eea_mask, ti["ei"]] - bank[eea_mask, ti["egc"]])
    candidate[egb_mask, ti["egb"]] = 0.75 * parent[egb_mask, ti["egb"]] + 0.25 * (1.1178 * bank[egb_mask, ti["egc"]] - 0.9221)
    candidate[ei_mask, ti["ei"]] = 0.75 * parent[ei_mask, ti["ei"]] + 0.25 * (bank[ei_mask, ti["eea"]] + bank[ei_mask, ti["egc"]])

    metrics = {}
    for j, target in enumerate(TARGETS):
        rows = np.where(obs[:, j])[0]
        parent_r2 = r2(labels[rows, j], parent[rows, j])
        candidate_r2 = r2(labels[rows, j], candidate[rows, j])
        metrics[target] = {"n": int(len(rows)), "parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2}
    parent_mean = float(np.mean([x["parent_r2"] for x in metrics.values()]))
    candidate_mean = float(np.mean([x["candidate_r2"] for x in metrics.values()]))
    deltas = np.array([x["delta_r2"] for x in metrics.values()])
    gate = {
        "mean_gain_at_least_0.002": bool(candidate_mean - parent_mean >= 0.002),
        "no_target_loss_worse_than_0.003": bool(np.min(deltas) >= -0.003),
        "eea_gain_at_least_0.010": bool(metrics["eea"]["delta_r2"] >= 0.010),
        "passed": bool(candidate_mean - parent_mean >= 0.002 and np.min(deltas) >= -0.003 and metrics["eea"]["delta_r2"] >= 0.010),
    }
    test_fi = test["fi"].to_numpy()
    test_other = obs[test_fi].copy()
    test_other[:, ti["ei"]] = False
    test_n_other = test_other.sum(1)
    test_route = test[test["target_type"].eq("ei")]["fi"].to_numpy()
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C161-ei-availability-gate",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "mechanism": "C160 observed-partner Eea/Egb routes plus Ei=Eea+Egc only when at least two other official properties are observed",
        "n_other_threshold": 2,
        "availability": {
            "clean_ei_route_rows": int(np.sum(ei_mask)),
            "clean_ei_partner_rows": int(np.sum(obs[:, ti["eea"]])),
            "test_ei_rows": int(len(test_route)),
            "test_ei_route_rows": int(np.sum(test_n_other[test["target_type"].to_numpy() == "ei"] >= 2)),
            "test_ei_n_other_counts": {str(k): int(v) for k, v in zip(*np.unique(test_n_other[test["target_type"].to_numpy() == "ei"], return_counts=True))},
        },
        "metrics": metrics,
        "parent_mean_r2": parent_mean,
        "candidate_mean_r2": candidate_mean,
        "gate": gate,
        "elapsed_seconds": time.time() - started,
    }
    CLEAN_OUT.mkdir(parents=True, exist_ok=True)
    (CLEAN_OUT / "R2-C161-ei-availability-gate-clean-oof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not gate["passed"]:
        print("C161 STOP: clean gate failed; no full-data fit and no local_eval diagnostic.", flush=True)
        return

    source = LOCAL_EVAL_OUT / "R2-C148-et-eps-only-LOCAL_DIAGNOSTIC_ONLY.csv"
    full = pd.read_csv(source)
    for i, row in test.iterrows():
        fi = int(row.fi)
        target = row.target_type
        if target == "eea" and obs[fi, ti["ei"]]:
            egc = labels[fi, ti["egc"]] if obs[fi, ti["egc"]] else p1[fi, ti["egc"]]
            full.loc[i, "target"] = 0.50 * full.loc[i, "target"] + 0.50 * (labels[fi, ti["ei"]] - egc)
        elif target == "egb" and obs[fi, ti["egc"]]:
            full.loc[i, "target"] = 0.75 * full.loc[i, "target"] + 0.25 * (1.1178 * labels[fi, ti["egc"]] - 0.9221)
        elif target == "ei" and obs[fi, ti["eea"]] and test_n_other[i] >= 2:
            egc = labels[fi, ti["egc"]] if obs[fi, ti["egc"]] else p1[fi, ti["egc"]]
            full.loc[i, "target"] = 0.75 * full.loc[i, "target"] + 0.25 * (labels[fi, ti["eea"]] + egc)
    out = LOCAL_EVAL_OUT / "R2-C161-ei-availability-gate-LOCAL_DIAGNOSTIC_ONLY.csv"
    full.to_csv(out, index=False)
    post = {"schema_version": "ppp.round2.postfreeze-candidate.v1", "experiment": report["experiment"], "classification": "LOCAL_DIAGNOSTIC_ONLY", "local_eval_read": False, "clean_gate": gate, "candidate_path": str(out), "candidate_sha256": sha256(out), "rows": int(len(full))}
    (LOCAL_EVAL_OUT / "R2-C161-ei-availability-gate-LOCAL_DIAGNOSTIC_ONLY.json").write_text(json.dumps(post, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(post, indent=2), flush=True)


if __name__ == "__main__":
    main()
