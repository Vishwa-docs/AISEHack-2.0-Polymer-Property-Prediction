"""C178: official-only 3D electrostatic/dielectric tensor specialist.

This is a deliberately new EPS/Nc representation family.  A single
deterministic RDKit ETKDGv3 conformer is generated from each capped repeat
unit.  The model uses charge moments, approximate polarizability anisotropy,
shape tensors, and fixed dielectric functional-group counts.  EPS and Nc are
fit in their bounded Clausius-Mossotti/Lorentz-Lorenz coordinates and blended
with the frozen corrected structure-only parent at a fixed 0.25 model weight.

The script is clean-only until all gates pass.  It never reads an local_eval,
test-external_label file, stored prediction CSV, or external target table.
"""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
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
MODEL_WEIGHT = 0.25
started = time.time()


def r2(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def no_stereo(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else str(smiles)


def capped_molecule(smiles):
    # The official polymers use '*' as repeat endpoints.  Carbon caps give
    # ETKDG/UFF a finite monomer without importing a polymer conformer.
    return Chem.MolFromSmiles(str(smiles).replace("*", "C"))


def safe_charge(atom):
    try:
        return float(atom.GetProp("_GasteigerCharge"))
    except Exception:
        return 0.0


def tensor_features(smiles, index):
    mol = capped_molecule(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(SEED + index)
        params.useRandomCoords = False
        code = AllChem.EmbedMolecule(mol, params)
        if code != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        AllChem.ComputeGasteigerCharges(mol)
        conf = mol.GetConformer()
        atoms = list(mol.GetAtoms())
        heavy = [atom for atom in atoms if atom.GetAtomicNum() > 1]
        if len(heavy) < 2:
            return None
        coords = np.asarray([[conf.GetAtomPosition(atom.GetIdx()).x, conf.GetAtomPosition(atom.GetIdx()).y, conf.GetAtomPosition(atom.GetIdx()).z] for atom in atoms], dtype=float)
        hcoords = np.asarray([coords[atom.GetIdx()] for atom in heavy], dtype=float)
        center = hcoords.mean(axis=0)
        rel = hcoords - center
        cov = (rel.T @ rel) / max(1, len(heavy))
        moments = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))
        rg = float(np.sqrt(np.trace(cov)))
        box = np.ptp(hcoords, axis=0)

        charges = np.asarray([safe_charge(atom) for atom in atoms], dtype=float)
        qheavy = np.asarray([safe_charge(atom) for atom in heavy], dtype=float)
        qsum = float(np.sum(qheavy))
        dip = np.sum(qheavy[:, None] * rel, axis=0)
        pos = qheavy > 0.02
        neg = qheavy < -0.02
        pos_center = np.average(hcoords[pos], axis=0, weights=qheavy[pos]) if np.any(pos) else center
        neg_weights = -qheavy[neg]
        neg_center = np.average(hcoords[neg], axis=0, weights=neg_weights) if np.any(neg) else center
        separation = float(np.linalg.norm(pos_center - neg_center))
        dip_norm = float(np.linalg.norm(dip))
        if dip_norm > 1e-10:
            axis = dip / dip_norm
        else:
            axis = np.asarray([1.0, 0.0, 0.0])
        parallel = float(axis @ cov @ axis)
        perpendicular = float(max(0.0, np.trace(cov) - parallel))

        # Fixed atom polarizability proxies are used only to form a geometric
        # tensor; they are not learned from any target table.
        alpha = {1: 0.67, 5: 3.05, 6: 1.76, 7: 1.10, 8: 0.80, 9: 0.56, 14: 5.38, 15: 3.63, 16: 2.90, 17: 2.18, 35: 3.05, 53: 4.01}
        polar = np.asarray([alpha.get(atom.GetAtomicNum(), 1.5) for atom in heavy], dtype=float)
        ptensor = (rel.T * polar) @ rel / max(1, len(heavy))
        peigs = np.sort(np.maximum(np.linalg.eigvalsh(ptensor), 0.0))
        ptr = float(np.sum(peigs))
        paniso = float((peigs[-1] - peigs[0]) / max(ptr, 1e-9))

        polar_bonds = 0
        dipole_sum = 0.0
        aromatic_bonds = 0
        ring_bonds = 0
        for bond in mol.GetBonds():
            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()
            # Pauling-scale proxy for the common elements; unknowns use a
            # neutral fallback and therefore do not create a target lookup.
            en = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
            delta = abs(en.get(a.GetAtomicNum(), 1.8) - en.get(b.GetAtomicNum(), 1.8))
            order = float(bond.GetBondTypeAsDouble())
            dipole_sum += delta * order
            polar_bonds += int(delta > 0.35)
            aromatic_bonds += int(bond.GetIsAromatic())
            ring_bonds += int(bond.IsInRing())

        smarts = [
            "S(=O)(=O)", "C(=O)", "C(=O)N", "COC", "CSC", "C#N", "[F,Cl,Br,I]", "a[O,N,S]", "a:a:a:a:a:a",
        ]
        smarts_counts = []
        for pattern in smarts:
            query = Chem.MolFromSmarts(pattern)
            smarts_counts.append(float(len(mol.GetSubstructMatches(query))) if query is not None else 0.0)

        values = [
            float(len(heavy)), float(mol.GetNumAtoms()), qsum, float(np.mean(qheavy)), float(np.std(qheavy)),
            float(np.max(qheavy)), float(np.min(qheavy)), dip_norm, separation, dipole_sum,
            float(polar_bonds), float(aromatic_bonds), float(ring_bonds),
            float(moments[0]), float(moments[1]), float(moments[2]), rg,
            float(box[0]), float(box[1]), float(box[2]), float(np.prod(np.maximum(box, 1e-6))),
            parallel, perpendicular, ptr, float(peigs[0]), float(peigs[1]), float(peigs[2]), paniso,
            float(Descriptors.MolWt(mol)), float(Descriptors.TPSA(mol)), float(Lipinski.NumRotatableBonds(mol)),
            *smarts_counts,
        ]
        values = np.asarray(values, dtype=float)
        return values if np.all(np.isfinite(values)) else None
    except Exception:
        return None


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
            vals = frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
            for canon, value in vals.items():
                labels[idx[canon], j] = float(value)
    return train, archive, test, labels


def coordinate(target, value):
    if target == "eps":
        return (value - 1.0) / (value + 2.0)
    return (value * value - 1.0) / (value * value + 2.0)


def inverse_coordinate(target, value):
    value = np.clip(value, -0.95, 0.95)
    if target == "eps":
        return (1.0 + 2.0 * value) / np.maximum(1.0 - value, 1e-6)
    return np.sqrt(np.maximum((1.0 + 2.0 * value) / np.maximum(1.0 - value, 1e-6), 0.05))


def models(seed):
    return [
        make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=50.0)),
        make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000)),
        make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), ExtraTreesRegressor(n_estimators=400, max_depth=5, min_samples_leaf=6, max_features=0.6, n_jobs=8, random_state=seed)),
    ]


def bootstrap(y, parent, candidate, rows, groups, reps=4000):
    rows = np.asarray(rows, dtype=int)
    rg = np.asarray(groups[rows], dtype=object)
    unique = np.unique(rg)
    by_group = {g: rows[rg == g] for g in unique}
    rng = np.random.default_rng(SEED + 178)
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
    parent = np.load(SCR / "out_clean_corrected/PFINAL.npy").copy()
    both_ei = observed[:, TI["ei"]] & observed[:, TI["eea"]] & observed[:, TI["egc"]]
    parent[both_ei, TI["ei"]] = .5 * parent[both_ei, TI["ei"]] + .5 * (labels[both_ei, TI["eea"]] + labels[both_ei, TI["egc"]])

    test_fi = test["fi"].to_numpy(dtype=int)
    active_rows = set(np.flatnonzero(observed[:, TI["eps"]]).tolist()) | set(np.flatnonzero(observed[:, TI["nc"]]).tolist()) | set(test_fi.tolist())
    feature_map = {}
    supported = np.zeros(n, dtype=bool)
    for count, fi in enumerate(sorted(active_rows), start=1):
        value = tensor_features(F["canon_list"][fi], fi)
        if value is not None:
            feature_map[fi] = value
            supported[fi] = True
        if count % 500 == 0:
            print(f"3d_features={count}/{len(active_rows)}", flush=True)
    width = len(next(iter(feature_map.values()))) if feature_map else 0
    X = np.full((n, width), np.nan, dtype=float)
    for fi, value in feature_map.items():
        X[fi] = value
    groups_all = np.asarray([no_stereo(value) for value in F["canon_list"]], dtype=object)
    scaffolds = np.asarray(F["scaffolds"], dtype=object)

    target_reports = {}
    candidates = parent.copy()
    fold_reports = {}
    for target in ("eps", "nc"):
        j = TI[target]
        rows = np.flatnonzero(observed[:, j])
        valid_rows = rows[supported[rows]]
        y = labels[:, j]
        parent_coord = coordinate(target, parent[valid_rows, j])
        y_coord = coordinate(target, y[valid_rows])
        pred = parent[valid_rows, j].copy()
        folds = GroupKFold(n_splits=5).split(valid_rows, y[valid_rows], groups=groups_all[valid_rows])
        records = []
        for fold, (tr, va) in enumerate(folds, start=1):
            fit_rows = valid_rows[tr]
            hold_rows = valid_rows[va]
            preds = []
            for model in models(SEED + 100 * j + fold):
                model.fit(X[fit_rows], y_coord[tr])
                preds.append(model.predict(X[hold_rows]))
            model_coord = np.mean(np.column_stack(preds), axis=1)
            pred[va] = inverse_coordinate(target, (1.0 - MODEL_WEIGHT) * parent_coord[va] + MODEL_WEIGHT * model_coord)
            records.append({"fold": fold, "fit_rows": int(len(fit_rows)), "valid_rows": int(len(hold_rows)), "candidate_supported_rows": int(len(hold_rows))})
        candidates[valid_rows, j] = pred
        all_rows = rows
        target_reports[target] = {
            "n": int(len(rows)),
            "conformer_supported_rows": int(len(valid_rows)),
            "conformer_fallback_rows": int(len(rows) - len(valid_rows)),
            "parent_r2": r2(y[all_rows], parent[all_rows, j]),
            "candidate_r2": r2(y[all_rows], candidates[all_rows, j]),
            "delta_r2": r2(y[all_rows], candidates[all_rows, j]) - r2(y[all_rows], parent[all_rows, j]),
            "supported_parent_r2": r2(y[valid_rows], parent[valid_rows, j]),
            "supported_candidate_r2": r2(y[valid_rows], candidates[valid_rows, j]),
            "supported_delta_r2": r2(y[valid_rows], candidates[valid_rows, j]) - r2(y[valid_rows], parent[valid_rows, j]),
            "bootstrap": bootstrap(y, parent[:, j], candidates[:, j], all_rows, scaffolds),
        }
        fold_reports[target] = records

    panel_reports = {}
    for target in ("eps", "nc"):
        j = TI[target]
        rows = np.flatnonzero(observed[:, j])
        counterpart = TI["nc"] if target == "eps" else TI["eps"]
        masks = {
            "conformer_supported": supported[rows],
            "conformer_fallback": ~supported[rows],
            "counterpart_available": observed[rows, counterpart],
            "counterpart_missing": ~observed[rows, counterpart],
        }
        panel_reports[target] = {}
        for name, mask in masks.items():
            selected = rows[mask]
            panel_reports[target][name] = {"n": int(len(selected)), "delta_r2": float(r2(labels[selected, j], candidates[selected, j]) - r2(labels[selected, j], parent[selected, j])) if len(selected) >= 3 else 0.0}

    test_support = {}
    for target in ("eps", "nc"):
        mask = test.target_type.eq(target).to_numpy()
        counterpart = TI["nc"] if target == "eps" else TI["eps"]
        test_support[target] = {"rows": int(np.sum(mask)), "conformer_supported": int(np.sum(mask & supported[test_fi])), "conformer_fallback": int(np.sum(mask & ~supported[test_fi])), "counterpart_available": int(np.sum(mask & observed[test_fi, counterpart]))}

    eps_pass = target_reports["eps"]["delta_r2"] >= .010 and target_reports["eps"]["bootstrap"]["delta_lower_2p5"] > 0 and sum(1 for row in fold_reports["eps"] if row["candidate_supported_rows"] > 0) >= 4
    nc_pass = target_reports["nc"]["delta_r2"] >= .010 and target_reports["nc"]["bootstrap"]["delta_lower_2p5"] > 0 and sum(1 for row in fold_reports["nc"] if row["candidate_supported_rows"] > 0) >= 4
    panel_pass = all(item["delta_r2"] >= 0 for report in panel_reports.values() for item in report.values())
    gate = {"eps_component_pass": bool(eps_pass), "nc_component_pass": bool(nc_pass), "all_declared_panels_nonnegative": bool(panel_pass), "combined_gain_at_least_0.014": bool(target_reports["eps"]["delta_r2"] + target_reports["nc"]["delta_r2"] >= .014), "passed": bool((eps_pass or nc_pass) and panel_pass)}
    report = {
        "schema_version": "ppp.round2.clean-oof.v1",
        "experiment": "R2-C178-3d-electrostatic-dielectric-tensor",
        "official_only_fitting": True,
        "test_structures_used_only_as_unlabeled_covariates": True,
        "local_eval_read": False,
        "pretrained_weights": False,
        "mechanism": "single ETKDGv3/UFF conformer; charge moments, polarizability tensor, shape tensor, and fixed dielectric SMARTS counts; EPS/Nc bounded-coordinate ensemble",
        "architecture": {"conformer": "RDKit ETKDGv3 fixed seed per structure, UFF maxIters=200", "models": "Ridge(alpha=50), Huber(epsilon=1.35), ExtraTrees(400,max_depth=5,min_samples_leaf=6,max_features=0.6)", "model_weight": MODEL_WEIGHT, "outer_split": "5-fold no-stereo canonical GroupKFold", "preprocessing": "fold-local imputation and scaling", "feature_count": int(width)},
        "feature_support": {"active_structures": int(len(active_rows)), "conformer_supported": int(np.sum(supported[list(active_rows)])), "conformer_fallback": int(len(active_rows) - np.sum(supported[list(active_rows)]))},
        "metrics": target_reports,
        "folds": fold_reports,
        "panels": panel_reports,
        "test_support_audit": test_support,
        "gate": gate,
        "decision": "bank_clean_component" if gate["passed"] else "reject_no_candidate",
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "R2-C178-3d-electrostatic-dielectric-tensor-clean-oof.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not gate["passed"]:
        print("C178 STOP: clean gate failed; no full-data candidate.", flush=True)


if __name__ == "__main__":
    main()
