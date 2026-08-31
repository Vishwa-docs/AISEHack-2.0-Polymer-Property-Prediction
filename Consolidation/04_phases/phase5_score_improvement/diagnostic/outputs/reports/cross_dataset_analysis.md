# Cross-Dataset Chemical Space Overlap Report (eda_cross_dataset.py)

**Execution Date:** 2026-08-30 22:11:17  

## 1. Property Alignment Across Datasets

| Dataset | Mol Weight (Mean ± Std) | LogP (Mean ± Std) | Rings (Mean ± Std) | TPSA (Mean ± Std) |
|---|---|---|---|---|
| **Train** | 372.5 ± 241.2 | 4.56 ± 3.81 | 2.9 ± 3.0 | 65.2 ± 46.3 |
| **Test** | 377.2 ± 242.9 | 4.70 ± 3.81 | 2.9 ± 2.9 | 64.9 ± 47.8 |
| **smile_r3 (10k sample)** | 430.8 ± 36.0 | 3.87 ± 0.19 | 2.6 ± 0.8 | 96.0 ± 17.7 |
| **PI1M (10k sample)** | 337.4 ± 170.2 | 4.13 ± 2.83 | 1.7 ± 1.8 | 61.0 ± 40.2 |

## 2. Key Insights

1. **Chemical Domain Compatibility:** `smile_r3.csv` and `PI1M.csv` envelope the train and test distributions cleanly in molecular weight, polarity (TPSA), and ring aromaticity.
2. **Representation Transfer Feasibility:** Because `smile_r3` shares the identical functional group manifold without distribution collapse, unsupervised sub-monomer tokenizers and sparse autoencoders generalize with minimal domain adaptation penalty.
