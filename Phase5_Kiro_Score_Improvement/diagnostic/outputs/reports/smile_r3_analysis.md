# 5.97M Molecular SMILES Characterization Report (eda_smile_r3.py)

**Execution Date:** 2026-08-30 22:11:09  
**Total Rows in smile_r3.csv:** 5,973,369  
**Sample Analyzed:** 100,000 rows (10,000 for deep chemical descriptors)  
**RDKit Validity Rate:** 100.00%

## 1. Physicochemical Properties (Sample Statistics)

- **Mean SMILES Length:** 55.09 ± 5.05 chars (Range: 38 - 102)
- **Mean Molecular Weight:** 416.18 Da
- **Mean Wildman-Crippen LogP:** 3.41
- **Mean Rotatable Bonds:** 6.54
- **Mean Ring Count:** 2.76

## 2. Atom Composition (Top Elements)

| Element | Occurrence Count | Proportion |
|---|---|---|
| **C** | 194,310 | 69.40% |
| **O** | 42,510 | 15.18% |
| **N** | 23,888 | 8.53% |
| **F** | 9,129 | 3.26% |
| **S** | 5,935 | 2.12% |
| **Cl** | 2,518 | 0.90% |
| **Br** | 1,593 | 0.57% |
| **I** | 107 | 0.04% |

## 3. Representation Learning Strategy

1. **Massive Character N-Gram Corpus:** With ~6M diverse organic SMILES, character n-grams (2-7 grams) with TF-IDF + TruncatedSVD (128-256 components) capture universal sub-monomer grammar without overfitting.
2. **From-Scratch Embedding Viability:** Fits smoothly in memory on Mac / GPU using incremental batching and sparse linear algebra.
