"""C271: EHT atom-response screen for EPS, Nc, and Ei."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdEHTTools
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUN = ROOT / "experiments/CLEAN_OFFICIAL_ONLY/R2-C271-20260805-eht-response-weak-target-screen-v1"
sys.path.insert(0, str(TOOLS))
import round2_c097_graph_grammar_hgb_full as parent_builder  # noqa: E402
import round2_c127_round1_carrier_factory as carrier  # noqa: E402
import round2_c258_ei_eht_orbital_residual as eht  # noqa: E402


TARGETS = ("eps", "nc", "ei")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(smiles: str, variant: str) -> int:
    return int(hashlib.sha256(f"C271|{variant}|{smiles}".encode()).hexdigest()[:8], 16) % 2_000_000_000 + 1


def atom_response_features(molecule: Chem.Mol | None, seed: int) -> tuple[list[float], bool]:
    embedded = eht.embed_for_eht(molecule, seed)
    if embedded is None:
        return [np.nan] * 66, False
    try:
        ok, result = rdEHTTools.RunMol(embedded)
    except Exception:
        return [np.nan] * 66, False
    if not ok:
        return [np.nan] * 66, False
    energies = np.asarray(result.GetOrbitalEnergies(), dtype=float)
    charges = np.asarray(result.GetAtomicCharges(), dtype=float)
    matrix = np.asarray(result.GetReducedChargeMatrix(), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != embedded.GetNumAtoms() or not np.isfinite(matrix).all() or not np.isfinite(energies).all() or not np.isfinite(charges).all():
        return [np.nan] * 66, False
    electrons = eht.valence_electron_count(embedded)
    homo_index = max(0, min(len(energies) - 2, int(electrons // 2) - 1))
    lumo_index = homo_index + 1
    atom_numbers = np.asarray([atom.GetAtomicNum() for atom in embedded.GetAtoms()], dtype=float)
    hetero = atom_numbers != 6
    aromatic = np.asarray([atom.GetIsAromatic() for atom in embedded.GetAtoms()], dtype=bool)
    conf = embedded.GetConformer()
    coords = np.asarray([list(conf.GetAtomPosition(i)) for i in range(embedded.GetNumAtoms())], dtype=float)
    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords - center, axis=1)
    out: list[float] = []
    for orbital_index in (max(0, homo_index - 1), homo_index, lumo_index, min(len(energies) - 1, lumo_index + 1)):
        weights = np.square(matrix[:, orbital_index])
        total = float(np.sum(weights))
        weights = weights / total if total > 1e-12 else np.full(len(weights), 1.0 / len(weights))
        entropy = float(-np.sum(weights * np.log(weights + 1e-12)))
        out.extend([
            float(np.sum(weights * hetero)),
            float(np.sum(weights * aromatic)),
            float(np.sum(weights * atom_numbers) / max(1.0, np.sum(weights))),
            float(np.sum(weights * radius)),
            float(np.sum(weights * weights)),
            entropy,
            float(np.sum(weights * charges)),
            float(np.sqrt(np.sum(weights * np.square(charges)))),
            float(np.max(weights)),
            float(np.quantile(weights, 0.90)),
        ])
    transition = []
    occupied = range(max(0, homo_index - 2), homo_index + 1)
    virtual = range(lumo_index, min(len(energies), lumo_index + 3))
    for occupied_index in occupied:
        for virtual_index in virtual:
            overlap = np.abs(matrix[:, occupied_index]) * np.abs(matrix[:, virtual_index])
            gap = max(1e-5, abs(float(energies[virtual_index] - energies[occupied_index])))
            transition.append(float(np.sum(overlap) / gap))
            transition.append(float(np.sum(overlap * radius) / gap))
    out.extend([
        float(np.mean(transition)),
        float(np.max(transition)),
        float(np.std(transition)),
        float(np.min(charges)),
        float(np.max(charges)),
        float(np.std(charges)),
        float(np.sum(np.abs(charges))),
        float(energies[lumo_index] - energies[homo_index]),
    ])
    return out, True


def stable_response(smiles: str) -> tuple[np.ndarray, dict[str, bool]]:
    hcap, h_ok = atom_response_features(eht.remove_dummy_caps(smiles), stable_seed(smiles, "hcap"))
    ring, r_ok = atom_response_features(eht.ring_close_dummy_caps(smiles), stable_seed(smiles, "ring"))
    delta = [hcap[i] - ring[i] if h_ok and r_ok else np.nan for i in range(len(hcap))]
    return np.asarray(hcap + ring + delta, dtype=float), {"hcap_supported": h_ok, "ring_supported": r_ok}


def main() -> None:
    started = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    progress = RUN / "progress.jsonl"
    progress.write_text(json.dumps({"stage": "started", "experiment_id": RUN.name}) + "\n", encoding="utf-8")
    root = ROOT.parent
    parent = parent_builder.build_parent(root, ROOT / "ppp-round-2")
    all_info = {target: dict(parent["target_info"][target]) for target in TARGETS}
    for info in all_info.values():
        info["fingerprints"] = parent["fingerprints"]
    train_global = np.unique(np.concatenate([np.asarray(info["indices"], dtype=int) for info in all_info.values()]))
    test_global = []
    for target in TARGETS:
        rows = parent["test"].loc[parent["test"]["target_type"].astype(str).eq(target)].sort_values("id")
        test_global.extend(parent["key_to_index"][value] for value in rows["canonical"])
    feature_global = np.unique(np.concatenate([train_global, np.asarray(test_global, dtype=int)]))
    feature_keys = [parent["keys"][int(index)] for index in feature_global]
    local_index = {int(index): i for i, index in enumerate(feature_global)}
    checkpoint = lambda stage, **payload: progress.open("a", encoding="utf-8").write(json.dumps({"stage": stage, **payload}) + "\n")
    checkpoint("parent_ready", feature_rows=int(len(feature_keys)), targets=list(TARGETS))
    matrix = []
    support = []
    for i, smiles in enumerate(feature_keys):
        row, report = stable_response(str(smiles))
        matrix.append(row)
        support.append(report)
        if (i + 1) % 25 == 0:
            checkpoint("response_features", processed=i + 1)
    matrix = np.asarray(matrix, dtype=float)
    target_reports = {}
    oof_parts = []
    for target in TARGETS:
        info = all_info[target]
        indices = np.asarray(info["indices"], dtype=int)
        X = matrix[[local_index[int(index)] for index in indices]]
        y = np.asarray(info["y"], dtype=float)
        base = np.asarray(info["parent"], dtype=float)
        groups = np.asarray(info["groups"], dtype=object)
        folds = carrier.grouped_folds(groups)
        residual = np.full(len(y), np.nan)
        fold_rows = []
        for fold in range(carrier.N_FOLDS):
            valid = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=60.0, solver="lsqr", max_iter=5000, tol=1e-4))
            model.fit(X[train], y[train] - base[train])
            residual[valid] = model.predict(X[valid])
            pred = base[valid] + 0.35 * residual[valid]
            fold_rows.append({"fold": fold, "rows": int(len(valid)), "parent_r2": float(r2_score(y[valid], base[valid])), "candidate_r2": float(r2_score(y[valid], pred)), "delta_r2": float(r2_score(y[valid], pred) - r2_score(y[valid], base[valid]))})
        candidate = base + 0.35 * residual
        parent_r2 = float(r2_score(y, base))
        candidate_r2 = float(r2_score(y, candidate))
        report = {"parent_r2": parent_r2, "candidate_r2": candidate_r2, "delta_r2": candidate_r2 - parent_r2, "positive_folds": int(sum(row["delta_r2"] > 0 for row in fold_rows)), "rows": int(len(y)), "folds": fold_rows}
        target_reports[target] = report
        oof_parts.append(pd.DataFrame({"target_type": target, "target": y, "parent": base, "candidate": candidate, "fold": folds}))
        print(json.dumps({"target": target, **report}), flush=True)
    screen_pass = any(r["delta_r2"] >= 0.010 and r["positive_folds"] >= 4 for r in target_reports.values())
    report = {"schema_version": "ppp.round2.clean-oof.v1", "experiment_id": RUN.name, "status": "feasibility_pass" if screen_pass else "rejected_screen", "official_only": True, "targets": target_reports, "screen_pass": screen_pass, "feature_count": int(matrix.shape[1]), "hcap_supported": int(sum(x["hcap_supported"] for x in support)), "ring_supported": int(sum(x["ring_supported"] for x in support)), "local_eval_read": False, "kaggle_compute": False, "kaggle_upload": False, "kaggle_submission": False, "promotion_eligible": False, "stable_seed": True, "elapsed_seconds": time.time() - started, "source_hashes": {"runner": digest(Path(__file__)), "eht": digest(TOOLS / "round2_c258_ei_eht_orbital_residual.py")}}
    (RUN / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.concat(oof_parts, ignore_index=True).to_csv(RUN / "oof_predictions.csv", index=False)
    (RUN / "decision.md").write_text(f"# C271 decision\n\nStatus: **{report['status']}**. This is a C050 feasibility screen only; any positive target requires selected-parent comparison and full gates before promotion.\n", encoding="utf-8")
    manifest = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.sha256":
            manifest.append(f"{digest(path)}  {path.name}")
    (RUN / "artifact_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    checkpoint("finished", status=report["status"], screen_pass=screen_pass)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
