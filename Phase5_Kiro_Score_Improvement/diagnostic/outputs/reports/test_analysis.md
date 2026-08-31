# Test Data & Similarity Analysis Report (eda_test.py)

**Execution Date:** 2026-08-30 22:11:00  
**Total Test Rows:** 4,940  
**Unique SMILES Strings:** 4,497  
**Unique Canonical Structures:** 4,133  
**Direct Train-Test Structure Overlap:** 1,063 unique structures (1,631 test rows = 33.02%)

## 1. Train-Test Tanimoto Similarity Breakdown

| Similarity Bin | Test Rows | Percentage | Category Characterization |
|---|---|---|---|
| **< 0.3 (Very Low / OOD)** | 25 | 0.5% | High private-LB risk (OOD) |
| **0.3 - 0.5 (Low / Novel)** | 270 | 5.5% | Requires robust generalization |
| **0.5 - 0.7 (Medium)** | 797 | 16.1% | Requires robust generalization |
| **0.7 - 1.0 (High / Scaffold Match)** | 3,848 | 77.9% | Requires robust generalization |

## 2. Key Findings for Modeling & Validation

1. **The 457 Overlap SMILES:** 
   - There are exactly 1063 unique structures appearing in both train and test.
   - **Crucial Rule:** Folds must be grouped by canonical SMILES so that no identical structure is ever in both train and val folds during CV.
2. **Distribution Shift / Low-Similarity Bins:**
   - **25 rows (0.51%)** have max Tanimoto similarity < 0.3 to any training molecule.
   - Standard tree models memorizing local Morgan bits collapse on these rows.
   - Self-supervised representation learning from `smile_r3.csv` (5.97M molecules) and latent property regularization directly target this OOD segment.
