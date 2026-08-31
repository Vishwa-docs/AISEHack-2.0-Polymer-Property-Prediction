"""Validation panels: scaffold/family/similarity-cluster folds + Tanimoto sim.

All computed from official SMILES in-process; no external artifacts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def _canonical_ring_system(mol) -> str:
    """Scaffold key: canonical SMILES of the ring system (Bemis-Murcko style)."""
    ri = mol.GetRingInfo()
    if not ri.AtomRings():
        return "ACYCLIC"
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    try:
        mol = Chem.RWMol(mol)
        for atom in sorted(mol.GetAtoms(), reverse=True):
            if atom.GetIdx() not in ring_atoms:
                mol.RemoveAtom(atom.GetIdx())
        return Chem.MolToSmiles(mol.GetMol(), canonical=True, isomericSmiles=False)
    except Exception:
        return "SCAFFOLD_FAIL"


def scaffold_folds(train: pd.DataFrame, target: str, n_splits: int = 5, seed: int = 2026):
    """Group rows by scaffold key, then split groups into n_splits folds."""
    mask = (train["target_type"] == target).to_numpy()
    positions = np.where(mask)[0]
    mols = []
    for smi in train.loc[mask, "smiles"]:
        mol = Chem.MolFromSmiles(str(smi).replace("[*]", "*"))
        mols.append(mol if mol is not None else Chem.MolFromSmiles("C"))
    scaffolds = np.array([_canonical_ring_system(m) for m in mols])
    unique, inv = np.unique(scaffolds, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    group_fold = np.empty(len(unique), dtype=int)
    for i, g in enumerate(order):
        group_fold[g] = i % n_splits
    folds = []
    for fold in range(n_splits):
        val = group_fold[inv] == fold
        folds.append((positions[~val], positions[val]))
    return folds


def tanimoto_similarity(train_smiles: list, test_smiles: list, radius: int = 2, bits: int = 1024) -> np.ndarray:
    """Nearest-train Tanimoto similarity for each test structure (fps in-memory)."""
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=bits)

    def fps(smiles_list):
        out = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(str(smi).replace("[*]", "*"))
            if mol is None:
                mol = Chem.MolFromSmiles("C")
            out.append(generator.GetFingerprint(mol))
        return out

    train_fps = fps(train_smiles)
    is_self = (test_smiles is train_smiles) or (len(test_smiles) == len(train_smiles) and test_smiles == train_smiles)
    test_fps = train_fps if is_self else fps(test_smiles)

    sims = np.zeros(len(test_fps), dtype=float)
    for i, tfp in enumerate(test_fps):
        scores = list(DataStructs.BulkTanimotoSimilarity(tfp, train_fps))
        if is_self and i < len(scores):
            scores[i] = -1.0
        sims[i] = float(max(scores)) if scores else 0.0
    return sims


def kmeans_folds(train: pd.DataFrame, target: str, n_splits: int = 5, seed: int = 2026,
                 n_clusters: int | None = None):
    """Structural-similarity folds: Morgan fingerprints -> MiniBatchKMeans
    clusters; clusters are kept whole inside folds (fold-design comparison)."""
    from sklearn.cluster import MiniBatchKMeans

    mask = (train["target_type"] == target).to_numpy()
    positions = np.where(mask)[0]
    mols = []
    for smi in train.loc[mask, "smiles"]:
        mol = Chem.MolFromSmiles(str(smi).replace("[*]", "*"))
        mols.append(mol if mol is not None else Chem.MolFromSmiles("C"))
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    X = np.array([np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(), dtype=np.uint8) - 48 for m in mols])
    k = n_clusters or max(n_splits, int(np.ceil(len(mols) / 30)))
    k = min(k, max(len(mols) // 2, 1))
    km = MiniBatchKMeans(n_clusters=k, batch_size=512, random_state=seed, n_init=3)
    labels = km.fit_predict(X)
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    cluster_fold = np.empty(k, dtype=int)
    for i, c in enumerate(order):
        cluster_fold[c] = i % n_splits
    folds = []
    for fold in range(n_splits):
        val = cluster_fold[labels] == fold
        folds.append((positions[~val], positions[val]))
    return folds


def tanimoto_to_set(smiles_list: list, other_smiles: list, radius: int = 2, bits: int = 1024) -> np.ndarray:
    """For each molecule in smiles_list, max Tanimoto similarity to the other set."""
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=bits)

    def fps(items):
        out = []
        for smi in items:
            mol = Chem.MolFromSmiles(str(smi).replace("[*]", "*"))
            if mol is None:
                mol = Chem.MolFromSmiles("C")
            out.append(generator.GetFingerprint(mol))
        return out

    mine = fps(smiles_list)
    others = fps(other_smiles)
    sims = np.zeros(len(mine), dtype=float)
    for i, fp in enumerate(mine):
        scores = list(DataStructs.BulkTanimotoSimilarity(fp, others))
        sims[i] = float(max(scores)) if scores else 0.0
    return sims
