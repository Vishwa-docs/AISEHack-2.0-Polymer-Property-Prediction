"""SSL / representation learning on OFFICIAL unlabeled data only (PI1M, smile_r3).

Every representation is fitted from scratch, in-process, with fixed seeds:
  - tfidf_svd      : char TF-IDF + TruncatedSVD over a corpus sample
  - ppmi_svd       : PPMI-weighted token co-occurrence + SVD ("word2vec analogue",
                     deterministic, no gensim dependency)
  - atom_ppmi      : same PPMI-SVD machinery but over CHEMICALLY-AWARE atom tokens
                     (multi-char atoms like Cl/Br/Si and [*] are single tokens)
  - morgan_idf     : Morgan substructure-frequency IDF re-weighted fingerprints
  - mlm            : tiny char-level masked-LM transformer (r3_core.nn)
  - atom_mlm       : masked-LM transformer over atom-level tokens (preferred:
                     char-level tokenization fragments functional groups)

No external data, no pretrained weights, no oracle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

RDLogger.DisableLog("rdApp.*")

SEED = 2026


def corpus_path(name: str, data_dir) -> object:
    import pathlib

    data_dir = pathlib.Path(data_dir)
    mapping = {"pi1m": "PI1M.csv", "smile_r3": "smile_r3.csv"}
    return data_dir / mapping[name]


def load_corpus(name: str, data_dir, n: int | None = None, seed: int = SEED,
                exclude: list[str] | None = None) -> list[str]:
    """Read a sample of unlabeled SMILES from PI1M.csv / smile_r3.csv."""
    path = corpus_path(name, data_dir)
    col = pd.read_csv(path, usecols=[0], nrows=None if n is None else min(int(n * 1.35), 6_100_000))
    c = col.columns[0]
    smiles = col[c].astype(str)
    if n is not None and len(smiles) > n:
        smiles = smiles.sample(n=n, random_state=seed)
    out = list(dict.fromkeys(smiles.tolist()))
    if exclude:
        excl = set(exclude)
        out = [s for s in out if s not in excl]
    return out[:n] if n is not None else out


def tfidf_svd_features(corpus: list[str], target_texts: list[str], *, dim: int = 128,
                       ngram_range=(2, 5), max_features: int = 200_000, seed: int = SEED):
    vec = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, max_features=max_features,
                          lowercase=False, sublinear_tf=True)
    Xc = vec.fit_transform(corpus)
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    svd.fit(Xc)
    Xt = vec.transform(target_texts)
    return svd.transform(Xt).astype(np.float32), float(svd.explained_variance_ratio_.sum())


def _ppmi_matrix(corpus: list[str], token_fn, *, window: int = 4, vocab_size: int = 60,
                 max_chars: int = 500_000):
    """Build PPMI matrix over tokens (characters or atom tokens). Returns (P, stoi, V)."""
    from collections import Counter

    counts = Counter()
    for s in corpus[:200_000]:
        counts.update(token_fn(str(s)))
    vocab = [t for t, _ in counts.most_common(vocab_size)]
    if not vocab:
        vocab = ["C", "N", "O"]
    stoi = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    C = np.zeros((V, V), dtype=np.float64)
    for s in corpus[:max_chars]:
        ids = [stoi.get(t, -1) for t in token_fn(str(s))]
        ids = [i for i in ids if i >= 0]
        for i, a in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    C[a, ids[j]] += 1.0
    total = C.sum() + 1e-9
    row = C.sum(axis=1, keepdims=True) + 1e-9
    col = C.sum(axis=0, keepdims=True) + 1e-9
    pmi = np.log((C / total) / ((row @ col) / (total ** 2)) + 1e-9)
    P = np.maximum(pmi, 0.0)
    return P, stoi, V


def _ppmi_embed(P, stoi, V, target_texts, token_fn, *, dim: int):
    """Project PPMI matrix via SVD and embed token sequences (mean of token vectors)."""
    u, s, vt = np.linalg.svd(P, full_matrices=False)
    k = min(dim, V)                      # FIX: vocab may be smaller than requested dim
    if k == 0:
        k = 1
    W = (u[:, :k] * np.sqrt(s[:k])[None, :])          # (V, k)
    D = (np.sqrt(s[:k])[:, None] * vt[:k, :])          # (k, V)

    def embed(texts):
        out = np.zeros((len(texts), k), dtype=np.float32)
        for i, t in enumerate(texts):
            ids = [stoi.get(tok, -1) for tok in token_fn(str(t))]
            ids = [j for j in ids if j >= 0]
            if ids:
                out[i] = W[ids].mean(axis=0) + 0.3 * D[:, ids].mean(axis=1)
        return out

    return embed(target_texts)


def _char_tokens(s: str):
    return list(s)


def ppmi_svd_features(corpus: list[str], target_texts: list[str], *, dim: int = 128,
                      window: int = 4, vocab_size: int = 60, max_chars: int = 500_000,
                      seed: int = SEED):
    """Word2vec-analogue: PPMI over token co-occurrence within a sliding window, then SVD."""
    P, stoi, V = _ppmi_matrix(corpus, _char_tokens, window=window, vocab_size=vocab_size,
                              max_chars=max_chars)
    return _ppmi_embed(P, stoi, V, target_texts, _char_tokens, dim=dim)


def atom_tokens(s: str) -> list[str]:
    """Chemically-aware tokenization: multi-char atoms, [*] endpoints, bonds and
    ring-closure digits are each single tokens (ask sec 3.2 -- do not fragment Cl/Br/Si)."""
    from .nn import ATOM_TOKEN_RE
    return ATOM_TOKEN_RE.findall(str(s))


def atom_ppmi_svd_features(corpus: list[str], target_texts: list[str], *, dim: int = 128,
                           window: int = 4, vocab_size: int = 60, max_chars: int = 500_000,
                           seed: int = SEED):
    """PPMI-SVD over atom-level tokens (chemically aware, permutation-robust analogue)."""
    P, stoi, V = _ppmi_matrix(corpus, atom_tokens, window=window, vocab_size=vocab_size,
                              max_chars=max_chars)
    return _ppmi_embed(P, stoi, V, target_texts, atom_tokens, dim=dim)


def morgan_idf_features(corpus: list[str], target_texts: list[str], *, radius: int = 2,
                        bits: int = 2048, max_mols: int = 300_000, seed: int = SEED):
    """Substructure document frequency -> IDF weights -> re-weighted Morgan counts."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    df = np.zeros(bits, dtype=np.float64)
    n = 0
    for s in corpus[:max_mols]:
        mol = Chem.MolFromSmiles(str(s).replace("[*]", "*"))
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros(bits, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        df += (arr > 0)
        n += 1
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0

    def embed(texts):
        out = np.zeros((len(texts), bits), dtype=np.float32)
        for i, s in enumerate(texts):
            mol = Chem.MolFromSmiles(str(s).replace("[*]", "*"))
            if mol is None:
                continue
            fp = gen.GetCountFingerprint(mol)
            arr = np.zeros(bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            out[i] = arr * idf.astype(np.float32)
        return out

    return embed(target_texts)


def mlm_features(corpus: list[str], target_texts: list[str], *, dim: int = 64, layers: int = 2,
                 heads: int = 4, epochs: int = 2, batch: int = 256, seed: int = SEED,
                 max_len: int = 256, atom: bool = False):
    from . import nn as nnmod

    model, tok = nnmod.pretrain_mlm(
        corpus, dim=dim, layers=layers, heads=heads, epochs=epochs, batch=batch,
        seed=seed, max_len=max_len, log_every=500, atom=atom,
    )
    return nnmod.mlm_embed(model, tok, target_texts)


def atom_mlm_features(corpus: list[str], target_texts: list[str], *, dim: int = 64,
                      layers: int = 2, heads: int = 4, epochs: int = 2, batch: int = 256,
                      seed: int = SEED, max_len: int = 256):
    """Masked-language model over atom-level tokens (ask sec 3.2 tokenization)."""
    return mlm_features(corpus, target_texts, dim=dim, layers=layers, heads=heads,
                        epochs=epochs, batch=batch, seed=seed, max_len=max_len, atom=True)


def ssl_features(cfg: dict, target_texts: list[str], data_dir, *, seed: int = SEED):
    """Dispatch on cfg: {"method": "svd"|"ppmi"|"atom_ppmi"|"w2v"|"morgan_idf"|"mlm"|"atom_mlm",
    "corpus": "smile_r3"|"pi1m"|"combined", "n": int, "dim": int, ...}."""
    method = cfg.get("method", "svd")
    corpora = cfg.get("corpus", "smile_r3")
    names = ["pi1m", "smile_r3"] if corpora == "combined" else [corpora]
    exclude = list({t.replace("[*]", "*") for t in target_texts})
    corpus: list[str] = []
    for nm in names:
        corpus.extend(load_corpus(nm, data_dir, n=cfg.get("n"), seed=seed, exclude=exclude))
    corpus = list(dict.fromkeys(corpus))
    print(f"  [ssl] corpus={corpora} n={len(corpus)} method={method}", flush=True)
    if method == "svd":
        return tfidf_svd_features(corpus, target_texts, dim=cfg.get("dim", 128),
                                  ngram_range=tuple(cfg.get("ngram_range", (2, 5))),
                                  seed=seed)[0]
    if method in ("ppmi", "w2v"):
        return ppmi_svd_features(corpus, target_texts, dim=cfg.get("dim", 128),
                                 window=cfg.get("window", 4), seed=seed)
    if method == "atom_ppmi":
        return atom_ppmi_svd_features(corpus, target_texts, dim=cfg.get("dim", 128),
                                      window=cfg.get("window", 4), seed=seed)
    if method == "morgan_idf":
        return morgan_idf_features(corpus, target_texts, radius=cfg.get("radius", 2),
                                   bits=cfg.get("bits", 2048), seed=seed)
    if method in ("mlm", "atom_mlm"):
        return mlm_features(corpus, target_texts, dim=cfg.get("dim", 64),
                            layers=cfg.get("layers", 2), heads=cfg.get("heads", 4),
                            epochs=cfg.get("epochs", 2), batch=cfg.get("batch", 256),
                            seed=seed, max_len=cfg.get("max_len", 256),
                            atom=(method == "atom_mlm"))
    raise ValueError(f"unknown ssl method: {method}")

