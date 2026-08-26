"""C160: fixed observed-partner physical identities against corrected C141.

This experiment is deliberately clean-only until its preregistered OOF gates
pass.  Official labels for *other* target types on the same canonical polymer
are allowed covariates.  When a partner is unavailable, the fallback is the
structure-only corrected Stage-1 prediction; no predicted partner model may
consume the current row's target label.
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
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
ti = {t: i for i, t in enumerate(TARGETS)}
started = time.time()


def r2(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_labels(cmap, idx, ns):
    train = pd.read_csv(BASE / "train.csv")
    test = pd.read_csv(BASE / "test.csv")
    archive = pd.read_csv(BASE / "archive/train.csv")
    for frame in (train, test, archive):
        frame["canon"] = frame["smiles"].map(cmap)
        frame["fi"] = frame["canon"].map(idx).astype(int)
    labels = np.full((ns, len(TARGETS)), np.nan)
    for j, target in enumerate(TARGETS):
        for frame in (archive, train):
            vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items():
                labels[idx[canon], j] = value
    return train, test, archive, labels


def main():
    F = pickle.loads((SCR / "features.pkl").read_bytes())
    idx, cmap = F["idx"], F["canon_map"]
    ns = len(F["canon_list"])
    train, test, archive, labels = load_labels(cmap, idx, ns)
    obs = np.isfinite(labels)

    # Corrected Stage-1/Stage-2 structure-only OOF predictions.  These arrays
    # contain no local_eval values and are the only fallback for missing partners.
    p1 = np.load(SCR / "out_clean_corrected/P1.npy")
    cur = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    if p1.shape != cur.shape or cur.shape != labels.shape:
        raise RuntimeError(f"shape mismatch: p1={p1.shape}, cur={cur.shape}, labels={labels.shape}")

    # Fixed parent correction already established in C143: use the direct Ei =
    # Eea + Egc identity only where both independent partner labels exist.
    both_ei = obs[:, ti["ei"]] & obs[:, ti["eea"]] & obs[:, ti["egc"]]
    ei_identity = labels[:, ti["eea"]] + labels[:, ti["egc"]]
    parent = cur.copy()
    parent[both_ei, ti["ei"]] = 0.5 * parent[both_ei, ti["ei"]] + 0.5 * ei_identity[both_ei]

    # A partner vector is direct official data when observed and a fixed
    # structure-only estimate otherwise.  It never uses the current target.
    bank = np.where(obs, labels, p1)

    # Pre-registered fixed arms.  These coefficients were fixed from the
    # earlier clean physical-identity audit before this C160 score is read.
    # Each formula is blended only on an independently observed counterpart.
    candidate = parent.copy()
    eea_mask = obs[:, ti["ei"]]
    egb_mask = obs[:, ti["egc"]]
    ei_mask = obs[:, ti["eea"]]
    candidate[eea_mask, ti["eea"]] = (
        0.50 * parent[eea_mask, ti["eea"]]
        + 0.50 * (bank[eea_mask, ti["ei"]] - bank[eea_mask, ti["egc"]])
    )
    candidate[egb_mask, ti["egb"]] = (
        0.75 * parent[egb_mask, ti["egb"]]
        + 0.25 * (1.1178 * bank[egb_mask, ti["egc"]] - 0.9221)
    )
    candidate[ei_mask, ti["ei"]] = (
        0.75 * parent[ei_mask, ti["ei"]]
        + 0.25 * (bank[ei_mask, ti["eea"]] + bank[ei_mask, ti["egc"]])
    )

    metrics = {}
    for j, target in enumerate(TARGETS):
        rows = np.where(obs[:, j])[0]
        metrics[target] = {
            "n": int(len(rows)),
            "parent_r2": r2(labels[rows, j], parent[rows, j]),
            "candidate_r2": r2(labels[rows, j], candidate[rows, j]),
            "delta_r2": r2(labels[rows, j], candidate[rows, j]) - r2(labels[rows, j], parent[rows, j]),
        }

    parent_mean = float(np.mean([v["parent_r2"] for v in metrics.values()]))
    candidate_mean = float(np.mean([v["candidate_r2"] for v in metrics.values()]))
    deltas = np.array([v["delta_r2"] for v in metrics.values()])
    # Same conservative bank gate used by the loop: material aggregate gain,
    # no target loss worse than 0.003, and the weak-target component must move.
    gate = {
        "mean_gain_at_least_0.002": bool(candidate_mean - parent_mean >= 0.002),
        "no_target_loss_worse_than_0.003": bool(np.min(deltas) >= -0.003),
        "eea_gain_at_least_0.010": bool(metrics["eea"]["delta_r2"] >= 0.010),
        "passed": bool(candidate_mean - parent_mean >= 0.002 and np.min(deltas) >= -0.003 and metrics["eea"]["delta_r2"] >= 0.010),
    }
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C160-observed-physical-identities",
        "official_only_fitting": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "seed": SEED,
        "mechanism": "fixed observed-partner Ei=Eea+Egc, Eea=Ei-Egc, Egb=1.1178*Egc-0.9221; structure-only fallback; target-specific fixed shrinkage",
        "availability": {
            "ei_identity_both_partner_rows": int(np.sum(both_ei)),
            "eea_rows_with_ei_partner": int(np.sum(eea_mask)),
            "egb_rows_with_egc_partner": int(np.sum(egb_mask)),
            "ei_rows_with_eea_partner": int(np.sum(ei_mask)),
        },
        "metrics": metrics,
        "parent_mean_r2": parent_mean,
        "candidate_mean_r2": candidate_mean,
        "gate": gate,
        "elapsed_seconds": time.time() - started,
    }
    CLEAN_OUT.mkdir(parents=True, exist_ok=True)
    report_path = CLEAN_OUT / "R2-C160-observed-physical-identities-clean-oof.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if not gate["passed"]:
        print("C160 STOP: clean gate failed; no full-data fit and no local_eval diagnostic.", flush=True)
        return

    # Only after the clean gate passes may this script materialize a full-data
    # candidate for the separately frozen post-freeze diagnostic lane.
    source = LOCAL_EVAL_OUT / "R2-C148-et-eps-only-LOCAL_DIAGNOSTIC_ONLY.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    full = pd.read_csv(source)
    test_fi = test["fi"].to_numpy()
    for i, fi in enumerate(test_fi):
        target = test.iloc[i]["target_type"]
        if target == "eea" and obs[fi, ti["ei"]]:
            raw = labels[fi, ti["ei"]] - (labels[fi, ti["egc"]] if obs[fi, ti["egc"]] else p1[fi, ti["egc"]])
            full.loc[i, "target"] = 0.50 * full.loc[i, "target"] + 0.50 * raw
        elif target == "egb" and obs[fi, ti["egc"]]:
            raw = 1.1178 * labels[fi, ti["egc"]] - 0.9221
            full.loc[i, "target"] = 0.75 * full.loc[i, "target"] + 0.25 * raw
        elif target == "ei" and obs[fi, ti["eea"]]:
            egc = labels[fi, ti["egc"]] if obs[fi, ti["egc"]] else p1[fi, ti["egc"]]
            raw = labels[fi, ti["eea"]] + egc
            full.loc[i, "target"] = 0.75 * full.loc[i, "target"] + 0.25 * raw
    out = LOCAL_EVAL_OUT / "R2-C160-observed-physical-identities-LOCAL_DIAGNOSTIC_ONLY.csv"
    full.to_csv(out, index=False)
    post = {
        "schema_version": "ppp.round2.postfreeze-candidate.v1",
        "experiment": report["experiment"],
        "classification": "LOCAL_DIAGNOSTIC_ONLY",
        "local_eval_read": False,
        "clean_gate": gate,
        "source_candidate": str(source),
        "candidate_path": str(out),
        "candidate_sha256": sha256(out),
        "rows": int(len(full)),
        "finite_id_target": bool(full["id"].nunique() == len(full) and np.isfinite(full["target"]).all()),
    }
    (LOCAL_EVAL_OUT / "R2-C160-observed-physical-identities-LOCAL_DIAGNOSTIC_ONLY.json").write_text(json.dumps(post, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(post, indent=2), flush=True)


if __name__ == "__main__":
    main()
