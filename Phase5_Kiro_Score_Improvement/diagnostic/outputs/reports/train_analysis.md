# Training Data Analysis Report (eda_train.py)

**Execution Date:** 2026-08-30 22:10:45  
**Total Rows in train.csv:** 7,409  
**Unique SMILES Strings:** 6,565  
**Unique Canonical Structures:** 5,849  
**RDKit Valid Molecules:** 6,565 / 6,565 (100% valid)

## 1. Target Counts & Descriptive Statistics

| Target | Count | Proportion | Mean | Std | Min | Median | Max | Skewness |
|---|---|---|---|---|---|---|---|---|
| **tg** | 4,143 | 55.9% | 143.459 | 109.084 | -109.820 | 136.400 | 495.000 | 0.09 |
| **egc** | 2,028 | 27.4% | 4.529 | 1.568 | 0.021 | 4.614 | 9.863 | -0.10 |
| **egb** | 337 | 4.5% | 4.276 | 1.979 | 0.507 | 4.052 | 10.114 | 0.45 |
| **ei** | 222 | 3.0% | 6.346 | 1.047 | 4.026 | 6.168 | 9.838 | 0.79 |
| **eea** | 221 | 3.0% | 2.278 | 1.107 | 0.394 | 2.272 | 5.144 | 0.22 |
| **eps** | 229 | 3.1% | 4.577 | 1.094 | 2.610 | 4.320 | 9.090 | 1.22 |
| **nc** | 229 | 3.1% | 1.934 | 0.235 | 1.560 | 1.900 | 2.758 | 0.89 |

## 2. Multi-label Sparsity Structure

Number of targets measured per unique SMILES polymer:
- **1 targets measured:** 6,150 polymers (93.7%)
- **2 targets measured:** 157 polymers (2.4%)
- **3 targets measured:** 126 polymers (1.9%)
- **4 targets measured:** 100 polymers (1.5%)
- **5 targets measured:** 28 polymers (0.4%)
- **6 targets measured:** 4 polymers (0.1%)

### Key Target Correlation Insights:
- **EI vs EGC:** Strong correlation (0.683 when present) confirming bandgap/ionization relationship.
- **EI vs EEA:** High correlation (0.240), reflecting electrochemical frontier orbital relationship ($E_i \approx E_{gc} + E_{ea}$).
- **EPS vs NC:** Strong correlation (0.918), consistent with Maxwell relation ($\epsilon_r \approx n_c^2 + \Delta\epsilon_{ionic}$).

## 3. Structural Properties

- **SMILES Length:** Mean = 49.3 chars (min: 3, max: 267)
- **Molecular Weight:** Mean = 401.4 Da (min: 16.0, max: 1945.8)
- **Heavy Atoms Count:** Mean = 28.8 (min: 1, max: 141)

## 4. Diagnostics & Priority Recommendations

1. **Tg Dominance:** Tg comprises 4,143 rows (55.9% of train data). Improvements in Tg directly drive hackathon score.
2. **Small-Target Sparsity:** `ei` (222), `eea` (221), `eps` (229), `nc` (229) have high missingness and severe sample starvation. Multi-task and physics-informed transfer learning across correlated pairs (EI-EEA-EGC, EPS-NC) is essential.
