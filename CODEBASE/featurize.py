#!/usr/bin/env python3
"""
featurize.py — shared, deterministic SMILES featurizer used by BOTH
build_weights.py and inference.py so training and inference agree exactly.

Produces a fixed-length vector: Morgan count fingerprint (radius 2, 2048) +
a handful of stable RDKit 2D descriptors. Robust to '*' polymer wildcards
and to parse failures (returns a zero vector).
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

_FP_SIZE = 2048
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=_FP_SIZE)

# stable, cheap descriptors (order fixed — part of the contract)
_DESC = [
    ('MolWt', Descriptors.MolWt),
    ('HeavyAtomCount', Descriptors.HeavyAtomCount),
    ('NumHDonors', Descriptors.NumHDonors),
    ('NumHAcceptors', Descriptors.NumHAcceptors),
    ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('RingCount', Descriptors.RingCount),
    ('NumAromaticRings', Descriptors.NumAromaticRings),
    ('FractionCSP3', Descriptors.FractionCSP3),
    ('TPSA', Descriptors.TPSA),
    ('MolLogP', Descriptors.MolLogP),
]
N_DESC = len(_DESC)
N_FEATURES = _FP_SIZE + N_DESC
FEATURE_NAMES = [f'morgan_{i}' for i in range(_FP_SIZE)] + [n for n, _ in _DESC]


def canonical(smiles):
    """RDKit isomeric-canonical SMILES — the join key used throughout V57."""
    try:
        mol = Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _mol(smiles):
    try:
        return Chem.MolFromSmiles(str(smiles).replace('[*]', '*'))
    except Exception:
        return None


def featurize_one(smiles):
    mol = _mol(smiles)
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    if mol is None:
        return vec
    try:
        vec[:_FP_SIZE] = _MORGAN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    except Exception:
        pass
    for j, (_, fn) in enumerate(_DESC):
        try:
            v = fn(mol)
            vec[_FP_SIZE + j] = np.float32(v) if np.isfinite(v) else 0.0
        except Exception:
            vec[_FP_SIZE + j] = 0.0
    return vec


def featurize_many(smiles_list):
    return np.vstack([featurize_one(s) for s in smiles_list])
