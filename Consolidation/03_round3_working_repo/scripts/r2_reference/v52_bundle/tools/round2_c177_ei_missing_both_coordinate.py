"""C177: structure-only coordinate reconstruction for unsupported Ei rows.

The partner family helps only when Eea or Egc is available.  This experiment
therefore freezes the current corrected parent on every supported row and
changes only the rows with neither partner.  It learns two physical
coordinates from official labels in each outer group fold:

    chi = (Ei + Eea) / 2
    Ei  = chi + Egc / 2

Both the chi and Egc regressors are structure-only Ridge models with fold-local
imputation/scaling.  No test external_label, local_eval artifact, predicted partner, or
previous partner-route prediction is used.
"""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE = ROOT / "ppp-round-2"
SCR = Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad")
OUT = ROOT / "experiments/CLEAN_OFFICIAL_ONLY"
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
TI = {name: i for i, name in enumerate(TARGETS)}
SEED = 20260804
RIDGE_ALPHA = 30.0
BLEND = 0.50
started = time.time()


def r2(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def no_stereo(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def read_labels(cmap, idx, n_structures):
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


def feature_matrix(features, physics):
    # Compact deterministic structure-only blocks.  Fingerprint bits are not
    # used here: the coordinate hypothesis is deliberately a smooth physical
    # extrapolator for the unsupported stratum.
    blocks = features["blocks"]
    matrix = np.hstack([blocks["desc"], blocks["extra"], physics["M"]]).astype(np.float64)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def model():
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def grouped_bootstrap(y, parent, candidate, rows, scaffolds, replicates=4000):
    rows = np.asarray(rows, dtype=int)
    row_groups = np.asarray(scaffolds[rows], dtype=object)
    unique = np.unique(row_groups)
    rows_by_group = {group: rows[row_groups == group] for group in unique}
    rng = np.random.default_rng(SEED + 177)
    deltas = np.empty(replicates, dtype=float)
    for i in range(replicates):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([rows_by_group[group] for group in chosen])
        deltas[i] = r2(y[sample], candidate[sample]) - r2(y[sample], parent[sample])
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
    physics = pickle.loads((SCR / "physics.pkl").read_bytes())
    idx, cmap = features["idx"], features["canon_map"]
    n_structures = len(features["canon_list"])
    train, archive, test, labels = read_labels(cmap, idx, n_structures)
    observed = np.isfinite(labels)
    pfinal = np.load(SCR / "out_clean_corrected/PFINAL.npy")
    if pfinal.shape != labels.shape:
        raise RuntimeError(f"shape mismatch: pfinal={pfinal.shape}, labels={labels.shape}")

    # Corrected C143 parent: direct identity only where both independent
    # official Eea and Egc labels exist.  C177 never changes those rows.
    parent_all = pfinal.copy()
    both = observed[:, TI["ei"]] & observed[:, TI["eea"]] & observed[:, TI["egc"]]
    parent_all[both, TI["ei"]] = 0.5 * parent_all[both, TI["ei"]] + 0.5 * (
        labels[both, TI["eea"]] + labels[both, TI["egc"]]
    )

    X = feature_matrix(features, physics)
    canon_groups = np.asarray([no_stereo(value) for value in features["canon_list"]], dtype=object)
    scaffolds = np.asarray(features["scaffolds"], dtype=object)
    ei_rows = np.flatnonzero(observed[:, TI["ei"]])
    chi_rows = np.flatnonzero(observed[:, TI["ei"]] & observed[:, TI["eea"]])
    egc_rows = np.flatnonzero(observed[:, TI["egc"]])
    missing_both = (~observed[:, TI["eea"]]) & (~observed[:, TI["egc"]])
    missing_both_ei = missing_both[ei_rows]
    if int(np.sum(missing_both_ei)) < 10:
        raise RuntimeError("unsupported Ei stratum unexpectedly small")

    y_ei = labels[:, TI["ei"]]
    candidate = parent_all[:, TI["ei"]].copy()
    fold_records = []
    folds = GroupKFold(n_splits=5).split(ei_rows, y_ei[ei_rows], groups=canon_groups[ei_rows])
    for fold, (train_pos, valid_pos) in enumerate(folds, start=1):
        valid = ei_rows[valid_pos]
        valid_groups = set(canon_groups[valid])
        fit_chi = chi_rows[~np.isin(canon_groups[chi_rows], list(valid_groups))]
        fit_egc = egc_rows[~np.isin(canon_groups[egc_rows], list(valid_groups))]
        chi_model = model()
        egc_model = model()
        chi_model.fit(X[fit_chi], 0.5 * (labels[fit_chi, TI["ei"]] + labels[fit_chi, TI["eea"]]))
        egc_model.fit(X[fit_egc], labels[fit_egc, TI["egc"]])
        route = missing_both[valid]
        routed = valid[route]
        if len(routed):
            chi_hat = chi_model.predict(X[routed])
            egc_hat = egc_model.predict(X[routed])
            coordinate_ei = chi_hat + 0.5 * egc_hat
            candidate[routed] = (1.0 - BLEND) * parent_all[routed, TI["ei"]] + BLEND * coordinate_ei
        fold_delta = r2(y_ei[valid], candidate[valid]) - r2(y_ei[valid], parent_all[valid, TI["ei"]])
        routed_delta = (
            r2(y_ei[routed], candidate[routed]) - r2(y_ei[routed], parent_all[routed, TI["ei"]])
            if len(routed) >= 3 else 0.0
        )
        fold_records.append({
            "fold": fold,
            "valid_rows": int(len(valid)),
            "fit_chi_rows": int(len(fit_chi)),
            "fit_egc_rows": int(len(fit_egc)),
            "missing_both_valid_rows": int(len(routed)),
            "delta_r2_all_ei": float(fold_delta),
            "delta_r2_missing_both": float(routed_delta),
        })

    all_metric = {
        "n": int(len(ei_rows)),
        "parent_r2": r2(y_ei[ei_rows], parent_all[ei_rows, TI["ei"]]),
        "candidate_r2": r2(y_ei[ei_rows], candidate[ei_rows]),
    }
    all_metric["delta_r2"] = all_metric["candidate_r2"] - all_metric["parent_r2"]
    mb_rows = ei_rows[missing_both_ei]
    missing_metric = {
        "n": int(len(mb_rows)),
        "parent_r2": r2(y_ei[mb_rows], parent_all[mb_rows, TI["ei"]]),
        "candidate_r2": r2(y_ei[mb_rows], candidate[mb_rows]),
    }
    missing_metric["delta_r2"] = missing_metric["candidate_r2"] - missing_metric["parent_r2"]
    bootstrap = grouped_bootstrap(y_ei, parent_all[:, TI["ei"]], candidate, ei_rows, scaffolds)

    # Test-side support is an unlabeled covariate audit only.  It is not used
    # to fit, tune, or score the route.
    test_fi = test["fi"].to_numpy(dtype=int)
    test_ei = test.target_type.eq("ei").to_numpy()
    test_missing_both = test_ei & ~observed[test_fi, TI["eea"]] & ~observed[test_fi, TI["egc"]]
    test_support = {
        "ei_rows": int(np.sum(test_ei)),
        "missing_both_rows": int(np.sum(test_missing_both)),
        "routed_fraction": float(np.mean(test_missing_both[test_ei])) if np.sum(test_ei) else 0.0,
    }

    positive_folds = int(np.sum([row["delta_r2_all_ei"] > 0.0 for row in fold_records]))
    gate = {
        "ei_gain_at_least_0.010": bool(all_metric["delta_r2"] >= 0.010),
        "at_least_4_of_5_outer_folds_positive": bool(positive_folds >= 4),
        "missing_both_panel_positive": bool(missing_metric["delta_r2"] > 0.0),
        "scaffold_bootstrap_lower_positive": bool(bootstrap["delta_lower_2p5"] > 0.0),
        "passed": bool(
            all_metric["delta_r2"] >= 0.010
            and positive_folds >= 4
            and missing_metric["delta_r2"] > 0.0
            and bootstrap["delta_lower_2p5"] > 0.0
        ),
    }
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C177-ei-missing-both-coordinate",
        "official_only_fitting": True,
        "test_structures_used_only_as_unlabeled_covariates": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "mechanism": "structure-only chi=(Ei+Eea)/2 plus structure-only Egc coordinate, reconstructed Ei=chi+Egc/2 only on rows missing both partners",
        "architecture": {
            "chi_model": "median-impute, standardize, Ridge",
            "egc_model": "median-impute, standardize, Ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "coordinate_blend_with_parent": BLEND,
            "features": "official repeat-unit descriptors, extra descriptors, and deterministic physics block; no fingerprints or learned PI1M embeddings",
            "outer_split": "5-fold no-stereo canonical GroupKFold; validation groups excluded from both coordinate training pools",
        },
        "rows": {
            "ei": int(len(ei_rows)),
            "chi_fit_pool": int(len(chi_rows)),
            "egc_fit_pool": int(len(egc_rows)),
            "missing_both_ei": int(len(mb_rows)),
        },
        "metrics": {"ei": all_metric, "missing_both_ei": missing_metric},
        "folds": fold_records,
        "positive_folds": positive_folds,
        "grouped_bootstrap": bootstrap,
        "test_support_audit": test_support,
        "gate": gate,
        "decision": "bank_clean_component" if gate["passed"] else "reject_no_candidate",
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2-C177-ei-missing-both-coordinate-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
