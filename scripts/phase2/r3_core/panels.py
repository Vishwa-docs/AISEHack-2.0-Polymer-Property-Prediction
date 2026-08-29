"""Validation panels: scaffold/family/similarity-cluster folds + Tanimoto sim.

All computed from official SMILES in-process; no external artifacts.
"""
from __future__ import annotations

import numpy as np
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
    test_fps = fps(test_smiles)
    sims = np.zeros(len(test_fps), dtype=float)
    for i, tfp in enumerate(test_fps):
        sims[i] = float(DataStructs.BulkTanimotoSimilarity(tfp, train_fps).max())
    return sims
