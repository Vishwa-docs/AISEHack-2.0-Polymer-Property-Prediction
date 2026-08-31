"""Feature builders — all fitted/derived from scratch inside the run.

Only official SMILES are consumed.  Every block returns a dense/sparse numpy
array with a matching row count, so callers can hstack blocks freely.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.impute import SimpleImputer

RDLogger.DisableLog("rdApp.*")

RNG_SEED = 2026


def parse_mols(smiles_list) -> list:
    mols = []
    for text in smiles_list:
        text = str(text).replace("[*]", "*")
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            mol = Chem.RWMol()
            mol.AddAtom(Chem.Atom(6))
            mol = mol.GetMol()
        mols.append(mol)
    return mols


# ---------------------------------------------------------------------------
# Handcrafted blocks
# ---------------------------------------------------------------------------

def descriptor_block(mols: list) -> np.ndarray:
    items = [(name, fn) for name, fn in Descriptors._descList if name != "Ipc"]
    out = np.full((len(mols), len(items)), np.nan, dtype=np.float64)
    for i, molecule in enumerate(mols):
        for j, (_, fn) in enumerate(items):
            try:
                value = float(fn(molecule))
                out[i, j] = value if math.isfinite(value) else np.nan
            except Exception:
                pass
    return out


def _conjugation_stats(mol) -> tuple[int, int]:
    eligible = {
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 0 and (
            atom.GetIsAromatic()
            or atom.GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)
        )
    }
    graph = {node: [] for node in eligible}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in eligible and b in eligible:
            graph[a].append(b)
            graph[b].append(a)

    def farthest(start: int):
        distances = {start: 0}
        queue = [start]
        for node in queue:
            for nxt in graph[node]:
                if nxt not in distances:
                    distances[nxt] = distances[node] + 1
                    queue.append(nxt)
        return max(distances, key=distances.get), distances

    best = 0
    unseen = set(graph)
    while unseen:
        start = next(iter(unseen))
        first, distances = farthest(start)
        component = set(distances)
        unseen -= component
        _, diameter = farthest(first)
        best = max(best, max(diameter.values(), default=0) + 1)
    return len(eligible), best


def topology_block(mols: list, smiles_list: list) -> np.ndarray:
    rows = []
    for molecule, text in zip(mols, smiles_list, strict=True):
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        try:
            Chem.ComputeGasteigerCharges(molecule)
            charge_values = []
            for atom in atoms:
                raw = atom.GetProp("_GasteigerCharge")
                value = float(raw)
                if math.isfinite(value):
                    charge_values.append(value)
        except Exception:
            charge_values = []
        charges = np.asarray(charge_values if charge_values else [0.0], dtype=float)
        stars = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 0]
        star_neighbors = [
            nb.GetAtomicNum() for index in stars for nb in molecule.GetAtomWithIdx(index).GetNeighbors()
        ]
        try:
            backbone = list(Chem.GetShortestPath(molecule, stars[0], stars[1])) if len(stars) == 2 else []
        except Exception:
            backbone = []
        conjugated_atoms, longest_path = _conjugation_stats(molecule)
        weights = [bond.GetBondTypeAsDouble() * (1.5 if bond.GetIsAromatic() else 1.0) for bond in bonds]
        eig = np.linalg.eigvalsh(np.asarray(Chem.GetAdjacencyMatrix(molecule), dtype=float)) if molecule.GetNumAtoms() else np.zeros(1)
        heavy = molecule.GetNumHeavyAtoms()
        rows.append([
            len(text), molecule.GetNumAtoms(), heavy, sum(a.GetAtomicNum() == 0 for a in atoms),
            molecule.GetRingInfo().NumRings(), sum(a.GetIsAromatic() for a in atoms),
            sum(a.GetAtomicNum() not in (0, 1, 6) for a in atoms), sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in atoms),
            rdMolDescriptors.CalcNumRotatableBonds(molecule), sum(b.GetBondTypeAsDouble() == 2 for b in bonds),
            sum(b.GetBondTypeAsDouble() == 3 for b in bonds), text.count("("), Chem.GetFormalCharge(molecule),
            Descriptors.MolWt(molecule), Crippen.MolLogP(molecule), Crippen.MolMR(molecule),
            rdMolDescriptors.CalcTPSA(molecule), rdMolDescriptors.CalcLabuteASA(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule), rdMolDescriptors.CalcNumHBA(molecule), rdMolDescriptors.CalcNumHBD(molecule),
            charges.min(), charges.max(), np.abs(charges).mean(), charges.std(), conjugated_atoms, longest_path,
            len(backbone), max(0, molecule.GetNumAtoms() - len(backbone)), len(stars), sum(star_neighbors),
            sum(weights), sum(b.GetIsAromatic() for b in bonds),
            sum(a.GetAtomicNum() not in (0, 1, 6) for a in atoms) / max(1, heavy),
            heavy / max(1, len(text)), molecule.GetRingInfo().NumRings() / max(1, heavy),
            sum(a.GetAtomicNum() == 8 for a in atoms), sum(a.GetAtomicNum() == 7 for a in atoms), sum(a.GetAtomicNum() == 14 for a in atoms),
            sum(a.GetAtomicNum() == 16 for a in atoms), sum(a.GetAtomicNum() == 15 for a in atoms), sum(a.GetAtomicNum() == 9 for a in atoms),
            sum(a.GetAtomicNum() == 17 for a in atoms), sum(a.GetAtomicNum() == 35 for a in atoms), sum(a.GetAtomicNum() == 53 for a in atoms),
            eig[-1] - eig[0], eig[-1], eig[0],
        ])
    return np.asarray(rows, dtype=np.float64)


def _atom_token(atom: Chem.Atom) -> str:
    return "X" if atom.GetAtomicNum() == 0 else f"{atom.GetSymbol()}{atom.GetDegree()}"


def polymer_genome_block(mols: list, max_features: int = 384) -> np.ndarray:
    """Polymer-genome style S|P|T atom/bond/triple counts (log1p)."""
    counters = []
    vocabulary: Counter = Counter()
    for molecule in mols:
        atom_types = {atom.GetIdx(): _atom_token(atom) for atom in molecule.GetAtoms()}
        counts: Counter = Counter()
        for atom in molecule.GetAtoms():
            counts["S|" + atom_types[atom.GetIdx()]] += 1
        for bond in molecule.GetBonds():
            pair = sorted((atom_types[bond.GetBeginAtomIdx()], atom_types[bond.GetEndAtomIdx()]))
            counts["P|" + "-".join(pair)] += 1
        for atom in molecule.GetAtoms():
            neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
            for left in range(len(neighbors)):
                for right in range(left + 1, len(neighbors)):
                    ends = sorted((atom_types[neighbors[left]], atom_types[neighbors[right]]))
                    counts["T|" + ends[0] + "-" + atom_types[atom.GetIdx()] + "-" + ends[1]] += 1
        counters.append(counts)
        vocabulary.update(counts)
    vocab = [key for key, _ in vocabulary.most_common(max_features)]
    position = {key: i for i, key in enumerate(vocab)}
    matrix = np.zeros((len(mols), len(vocab)), dtype=np.float32)
    for i, counts in enumerate(counters):
        for key, value in counts.items():
            if key in position:
                matrix[i, position[key]] = math.log1p(value)
    return matrix


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def bit_matrix(mols: list, radius: int, bits: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    result = np.zeros((len(mols), bits), dtype=np.float32)
    for i, molecule in enumerate(mols):
        vector = np.zeros(bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), vector)
        result[i] = vector
    return result


def count_matrix(mols: list, radius: int, bits: int) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    rows, cols, values = [], [], []
    for row, molecule in enumerate(mols):
        for column, value in generator.GetCountFingerprint(molecule).GetNonzeroElements().items():
            rows.append(row)
            cols.append(int(column))
            values.append(math.log1p(float(value)))
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(mols), bits), dtype=np.float32)


def char_ngrams(smiles_list: list, ngram_range=(2, 5), n_features: int = 1024) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=ngram_range, n_features=n_features,
        alternate_sign=False, norm="l2", lowercase=False,
    )
    return vectorizer.transform(list(smiles_list)).astype(np.float32)


def char_tfidf(smiles_list: list, ngram_range=(2, 5), max_features: int = 5000) -> sparse.csr_matrix:
    vectorizer = CountVectorizer(
        analyzer="char", ngram_range=ngram_range, max_features=max_features,
        lowercase=False,
    )
    return vectorizer.fit_transform(list(smiles_list)).astype(np.float32)


# ---------------------------------------------------------------------------
# Assemblies
# ---------------------------------------------------------------------------

def handcrafted_matrix(smiles_list: list) -> np.ndarray:
    """Descriptors + topology + polymer-genome, median-imputed, finite-safe."""
    mols = parse_mols(smiles_list)
    desc = descriptor_block(mols)
    topo = topology_block(mols, list(smiles_list))
    pg = polymer_genome_block(mols)
    block = np.hstack([desc, topo, pg])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    out = imputer.fit_transform(block)
    out[~np.isfinite(out)] = 0.0
    return out


def full_feature_stack(
    smiles_list: list,
    *,
    use_svd: bool = True,
    svd_dim: int = 64,
    char_n_features: int = 1024,
) -> np.ndarray:
    """The R2-style full stack: handcrafted + Morgan r2/r3 + SVD of counts+char."""
    texts = list(smiles_list)
    mols = parse_mols(texts)
    hand = handcrafted_matrix(texts)
    morgan_r2 = bit_matrix(mols, 2, 512)
    morgan_r3 = bit_matrix(mols, 3, 512)
    counts = count_matrix(mols, 2, 1024)
    char = char_ngrams(texts, ngram_range=(2, 5), n_features=char_n_features)
    if use_svd:
        combined = sparse.hstack([counts, char], format="csr")
        svd = TruncatedSVD(n_components=svd_dim, random_state=RNG_SEED)
        svd_block = svd.fit_transform(combined).astype(np.float32)
        stack = np.hstack([hand.astype(np.float32), svd_block, morgan_r2, morgan_r3]).astype(np.float32)
    else:
        stack = np.hstack([hand.astype(np.float32), morgan_r2, morgan_r3]).astype(np.float32)
    stack[~np.isfinite(stack)] = 0.0
    return stack
