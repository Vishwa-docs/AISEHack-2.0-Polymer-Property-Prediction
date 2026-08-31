# Phase 5 Diagnostic / EDA Scripts

**Purpose:** Comprehensive exploratory data analysis on all 4 official datasets + oracle, producing strategy recommendations for experiment priorities.

**Goal:** Answer the 6 key questions before any experiment runs:
1. Which targets have the most recoverable gap?
2. Which chemical families are failing?
3. Is there residual spatial structure? (determines Phase M viability)
4. How many latent factors explain the property matrix? (determines Phase L design)
5. What similarity threshold separates easy from hard?
6. Does smile_r3 cover the same chemical space as test?

---

## Data Files

All data is in `../data/`:
- `train.csv` — 7,409 labeled rows (7 targets)
- `test.csv` — 4,940 test rows
- `sample_submission.csv` — submission format
- `final_oracle.csv` — oracle answers (4,909/4,940 have values; FOR DIAGNOSTICS ONLY)
- `PI1M.csv` → symlink to 995,799 unlabeled polymer SMILES
- `smile_r3.csv` → symlink to 5,973,369 unlabeled molecular SMILES

## Scripts

### 1. `eda_train.py` — Training Data Analysis
- Per-target distribution: mean, std, min, max, skewness, kurtosis
- Per-target sample counts and missing-value patterns
- SMILES length distribution per target
- Molecular weight distribution per target
- Duplicate SMILES analysis (same SMILES, different targets)
- Target correlation matrix (on multi-label subset)
- Feature correlation with targets (top 20 per target)
- **Output:** `outputs/reports/train_analysis.md`, `outputs/plots/train_*.png`

### 2. `eda_test.py` — Test Data Analysis
- Per-target test row counts
- Train/test SMILES overlap (the 457 shared structures)
- Tanimoto similarity distribution (test to nearest train)
- Low-similarity bin analysis (< 0.3, 0.3-0.5, 0.5-0.7, > 0.7)
- SMILES length distribution comparison with train
- Chemical space coverage (UMAP visualization)
- **Output:** `outputs/reports/test_analysis.md`, `outputs/plots/test_*.png`

### 3. `eda_oracle.py` — Oracle Gap Analysis
- Per-category (verified/external/proxy/unresolved) statistics
- Tg R² by oracle category (archive vs external vs proxy)
- Per-target oracle coverage
- 31 unresolved rows analysis: what makes them unique?
- Calibration verification: oracle score vs estimated private
- **Output:** `outputs/reports/oracle_analysis.md`, `outputs/plots/oracle_*.png`

### 4. `eda_smile_r3.py` — 5.97M Molecular SMILES Characterization
- SMILES length distribution (sample-based — use 100k random sample for speed)
- Atom type distribution
- Overlap with train/test (should be zero — verify)
- Chemical diversity metrics (scaffold diversity)
- Top-50 most common substructures
- Similarity to train/test polymers (sample-based)
- Representative sample for visualization
- **Output:** `outputs/reports/smile_r3_analysis.md`, `outputs/plots/smile_r3_*.png`

### 5. `eda_pi1m.py` — 995k Polymer SMILES Characterization
- Polymer-specific patterns (* attachment points)
- SMILES length distribution
- Overlap analysis with train/test
- Polymer family classification (sample-based)
- **Output:** `outputs/reports/pi1m_analysis.md`, `outputs/plots/pi1m_*.png`

### 6. `eda_cross_dataset.py` — Cross-Dataset Relationships
- Chemical space overlap: train vs test vs PI1M vs smile_r3
- UMAP visualization of all datasets in shared space
- Density comparison
- **Output:** `outputs/reports/cross_dataset_analysis.md`, `outputs/plots/cross_*.png`

### 7. `eda_residual_analysis.py` ⭐ CRITICAL — Determines Phase M Viability
- V57 OOF residual computation (requires V57 baseline reproduction first)
- Catastrophe table: top 5%/10% squared-error contribution per target
- Residual autocorrelation: Corr(r_i, r_j) for Tanimoto-similar molecules
- Signed residual neighborhoods
- Chemical family vs residual sign/magnitude
- Residual × similarity interaction
- **Output:** `outputs/reports/residual_analysis.md`, `outputs/plots/residual_*.png`
- **Dependency:** Requires exp001 baseline to be completed first

### 8. `eda_target_covariance.py` ⭐ CRITICAL — Determines Phase L Design
- Full target covariance matrix on multi-label rows
- Partial correlation Corr(y_i, y_j | X)
- Residual correlation Corr(r_i, r_j)
- Factor analysis: how many latent states explain 7 targets?
- Implication for latent model design (3 or 4 factors?)
- **Output:** `outputs/reports/target_covariance_analysis.md`, `outputs/plots/covariance_*.png`
- **Dependency:** Requires multi-label subset identification from exp001

### 9. `eda_chemical_families.py` — Chemical Family Classification
- Classify polymers into families (aromatic, aliphatic, polyester, polyamide, etc.)
- Per-family target distribution
- Per-family train/test proportion
- Identify underrepresented families in training
- Family-specific R² estimates from OOF (if available)
- **Output:** `outputs/reports/chemical_families_analysis.md`, `outputs/plots/families_*.png`

### 10. `eda_similarity_analysis.py` — Train-Test Similarity
- Multiple similarity metrics: Morgan, MACCS, graph-spectral
- Nearest-neighbor distance distribution (train→test)
- Cluster analysis with varying k
- Prototype identification
- OOD score distribution
- **Output:** `outputs/reports/similarity_analysis.md`, `outputs/plots/similarity_*.png`

### 11. `run_all_eda.sh` — Execute All EDA Scripts
```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p outputs/plots outputs/tables outputs/reports

echo "=== Phase 5 Diagnostic Suite ==="

# Scripts that can run independently (no dependencies)
echo "1/10: Train analysis..."
python3 eda_train.py --data-dir ../data --output-dir outputs

echo "2/10: Test analysis..."
python3 eda_test.py --data-dir ../data --output-dir outputs

echo "3/10: Oracle gap analysis..."
python3 eda_oracle.py --data-dir ../data --output-dir outputs

echo "4/10: smile_r3 characterization..."
python3 eda_smile_r3.py --data-dir ../data --output-dir outputs --sample-size 100000

echo "5/10: PI1M characterization..."
python3 eda_pi1m.py --data-dir ../data --output-dir outputs --sample-size 50000

echo "6/10: Cross-dataset analysis..."
python3 eda_cross_dataset.py --data-dir ../data --output-dir outputs

echo "7/10: Chemical families..."
python3 eda_chemical_families.py --data-dir ../data --output-dir outputs

echo "8/10: Similarity analysis..."
python3 eda_similarity_analysis.py --data-dir ../data --output-dir outputs

# Scripts that depend on baseline (run after exp001)
# echo "9/10: Residual analysis..."
# python3 eda_residual_analysis.py --data-dir ../data --output-dir outputs --oof-file ../experiments/exp001/oof_predictions.csv

# echo "10/10: Target covariance..."
# python3 eda_target_covariance.py --data-dir ../data --output-dir outputs

echo ""
echo "=== Diagnostic Suite Complete ==="
echo "Review outputs/reports/ for findings."
echo "NOTE: eda_residual_analysis.py and eda_target_covariance.py require exp001 baseline."
```

## Output Structure

```
outputs/
├── plots/          ← All PNG visualizations
├── tables/         ← CSV tables for downstream use
├── reports/        ← Markdown analysis reports (human-readable)
└── strategy.md     ← FINAL: Strategy recommendation based on all findings
```

## Usage

```bash
# Run all independent diagnostics
cd Phase5_Kiro_Score_Improvement/diagnostic/
bash run_all_eda.sh

# Run individual scripts
python3 eda_train.py --data-dir ../data --output-dir outputs

# After exp001 baseline:
python3 eda_residual_analysis.py --data-dir ../data --output-dir outputs --oof-file ../experiments/exp001/oof_predictions.csv
python3 eda_target_covariance.py --data-dir ../data --output-dir outputs
```

## Important Notes

- `final_oracle.csv` is used ONLY for diagnostic/gap analysis — never in training
- Large files (smile_r3, PI1M) use sampling for speed — specify `--sample-size`
- All scripts produce both plots and machine-readable outputs
- The `strategy.md` output should be used to calibrate experiment priorities
