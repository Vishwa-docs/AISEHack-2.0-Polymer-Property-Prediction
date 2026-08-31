# Multi-Metric Train-Test Similarity Report (eda_similarity_analysis.py)

**Execution Date:** 2026-08-30 22:11:40  
**Evaluated Unique Test Polymers:** 4,497

## 1. Similarity Summary Across Fingerprint Spaces

| Metric Space | Mean Max Similarity | Median Max Similarity | Min Similarity (Most OOD) | % with Sim < 0.3 |
|---|---|---|---|---|
| **Morgan Bit-Vector (r=2, 2048)** | 0.829 | 0.898 | 0.160 | 0.8% |
| **MACCS Substructure Keys (166)** | 0.944 | 1.000 | 0.500 | 0.0% |

## 2. Strategic Takeaway

- Morgan fingerprints show a substantial tail of low-similarity test molecules (< 0.35) that are vulnerable to tree split starvation.
- Multi-scale representations combining MACCS keys, graph features, and continuous SMILES embeddings (TF-IDF + SVD / Autoencoder) mitigate out-of-distribution blind spots.
