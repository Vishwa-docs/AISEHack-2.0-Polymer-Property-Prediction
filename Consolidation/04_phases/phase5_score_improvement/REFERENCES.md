# Phase 5: Literature & Technical References

**Purpose:** Citations and technical details for all methods used in Phase 5 experiments  
**Created:** 2026-08-30

---

## Self-Supervised Learning & Transformers

### polyBERT (Atom-Level Tokenization)

**Citation:**  
Kuenneth, C., et al. (2023). "polyBERT: a chemical language model to enable fully machine-learned molecular dynamics simulations." *Nature Communications*, 14, 3099.

**Key Insights:**
- **Atom-level tokenization** (not character n-grams): Treats `[*]`, `C(`, `=O`, `c1ccccc1` as single tokens
- Vocabulary size: ~2000-5000 tokens (vs 50-100 for char-level)
- Whole-token masking during MLM (mask entire `[O]` not char `O`)
- DeBERTa architecture (6 layers, 256 hidden, 8 heads)
- Trained on 2M SMILES with masked language modeling
- Fine-tuned on 29 polymer properties, achieved R²=0.80 average

**Relevance to Phase 5:**
- **Experiments B028, B036, B037** implement atom-level tokenization
- This is the chemically correct tokenization (vs char n-grams in R2)
- Expected to improve representation quality significantly

**Implementation Notes:**
- Use regex tokenizer: `r'Br|Cl|\[[^\]]+\]|[A-Z][a-z]?|.'`
- Build vocabulary from smile_r3.csv (5.97M samples)
- Transformer training: 6-12 hours on GPU for 5M SMILES

---

### ChemBERTa (Baseline SSL)

**Citation:**  
Chithrananda, S., Grand, G., & Ramsundar, B. (2020). "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." *arXiv:2010.09885*.

**Key Insights:**
- BERT-style masked language modeling on SMILES
- Trained on 77M molecules from PubChem and ZINC
- Character-level tokenization (simpler than atom-level)
- Transfer learning: pretrain → fine-tune
- Improves performance on small datasets (n<1000)

**Relevance to Phase 5:**
- **Experiments B033-B035** use char-level MLM as baseline
- Compare to atom-level (B036-B037) to validate polyBERT claim
- May help small targets (ei, eps) via transfer from large targets

---

### MolBERT

**Citation:**  
Wang, S., et al. (2019). "SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction." *ACM BCB*.

**Key Insights:**
- First BERT application to SMILES
- Masked LM + next-sentence prediction
- Significant improvements on MoleculeNet benchmarks

**Relevance to Phase 5:**
- Proof-of-concept that SSL works for molecules
- Justifies Phase B investment

---

## Multi-Task Learning & Physics Constraints

### Multi-Task for Sparse Targets

**Citation:**  
Kuenneth, C., et al. (2021). "Biasing chemical language models toward property prediction." *Patterns*, 2(4), 100238.

**Key Insights:**
- **Joint prediction of multiple properties improves sparse targets**
- Shared encoder learns from all available labels
- Masked loss function: only compute loss for available labels
- Physics-informed auxiliary losses improve consistency
- On polymer properties: multi-task beats single-task by 10-30% on small targets

**Relevance to Phase 5:**
- **Phase D (Exp 071-095):** Multi-task for ei/eea/egc and eps/nc
- Key for weak targets: ei (n=222), eps (n=229)
- Transfer learning from large targets (egc n=2028) to small

**Implementation:**
```python
# Masked loss (only available labels)
loss = ((y_pred - y_true) ** 2 * mask).sum() / mask.sum()

# Physics auxiliary loss
aux_loss = lambda * (ei_pred - eea_pred - egc_known) ** 2
total_loss = task_loss + aux_loss
```

---

### Physics-Informed Neural Networks

**Citation:**  
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

**Relevance to Phase 5:**
- **Exp D071-D095:** Soft physics constraints as auxiliary losses
- Known relationships:
  - ei - eea = egc (ionization - affinity = gap)
  - eps - nc² ≈ ionic (dielectric - optical = ionic)
  - egb ≈ a·egc + b (bulk vs chain gap affinity)

---

## Graph Neural Networks

### D-MPNN (Directed Message Passing)

**Citation:**  
Yang, K., et al. (2019). "Analyzing Learned Molecular Representations for Property Prediction." *Journal of Chemical Information and Modeling*, 59(8), 3370-3388.

**Key Insights:**
- **Edge-directed message passing** (messages on bonds, not nodes)
- Encodes bond features: type, stereo, conjugation, ring membership
- Outperforms Morgan fingerprints on MoleculeNet
- Scales to large molecules (drugs, polymers)

**Relevance to Phase 5:**
- **Phase C (Exp 046-070):** GNN baseline for polymers
- Expected to capture long-range interactions better than fingerprints
- **Exp 048:** MPNN implementation

---

### Graph Isomorphism Network (GIN)

**Citation:**  
Xu, K., et al. (2019). "How Powerful are Graph Neural Networks?" *ICLR*.

**Key Insights:**
- **Sum aggregation is theoretically optimal** (vs mean/max)
- As powerful as Weisfeiler-Lehman graph isomorphism test
- Provably more expressive than GCN/GraphSAGE

**Relevance to Phase 5:**
- **Exp 050:** GIN for polymer property prediction
- Theoretically strongest architecture

---

### Graph Attention Networks (GAT)

**Citation:**  
Brody, S., Alon, U., & Yahav, E. (2021). "How Attentive are Graph Attention Networks?" *ICLR*.

**Key Insights:**
- GATv2 fixes attention computation (v1 was static)
- Learnable attention weights per edge
- Good for heterogeneous graphs (many atom/bond types)

**Relevance to Phase 5:**
- **Exp 047:** GATv2 for polymers
- Attention may help identify critical functional groups for Tg

---

## Gaussian Processes for Small Data

### GP Regression with Tanimoto Kernel

**Citation:**  
Ralaivola, L., et al. (2005). "Graph kernels for chemical informatics." *Neural Networks*, 18(8), 1093-1110.

**Key Insights:**
- **Tanimoto kernel for binary fingerprints:**
  - K(x, y) = (x·y) / (||x||² + ||y||² - x·y)
  - Equivalent to Jaccard similarity
- **Gaussian Process Regression is theoretically optimal for small n**
- Provides calibrated uncertainty estimates
- No hyperparameter tuning needed (kernel learning)

**Relevance to Phase 5:**
- **Exp 077, 126-127:** GPR for ei (n=222), eps (n=229)
- Optimal method for small datasets
- Expected +0.015-0.035 improvement on ei

**Implementation:**
```python
from sklearn.gaussian_process import GaussianProcessRegressor

def tanimoto_kernel(X, Y):
    dot = X @ Y.T
    norm_x = (X ** 2).sum(axis=1, keepdims=True)
    norm_y = (Y ** 2).sum(axis=1, keepdims=True)
    return dot / (norm_x + norm_y.T - dot)

gpr = GaussianProcessRegressor(kernel=tanimoto_kernel, alpha=0.01)
gpr.fit(X_train, y_train)
```

---

### Kernel Ridge Regression

**Citation:**  
Rupp, M., et al. (2012). "Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning." *Physical Review Letters*, 108, 058301.

**Key Insights:**
- Linear ridge regression in kernel space
- Faster than GPR (O(n²) vs O(n³))
- Similar performance for small n
- Coulomb matrix as molecular representation

**Relevance to Phase 5:**
- **Exp 127:** KRR with Tanimoto kernel
- Faster alternative to GPR
- May enable ensembling multiple kernel methods

---

## Polymer-Specific Methods

### Bicerano Group Contribution Method

**Citation:**  
Bicerano, J. (2002). *Prediction of Polymer Properties* (3rd ed.). Marcel Dekker.

**Key Insights:**
- **Group contribution method for Tg:**
  - Count ~50 functional groups: amide, ester, ether, aromatic, etc.
  - Each group has empirical Tg contribution
  - Tg ≈ Σ(n_i · ΔTg_i) / M_w
- Based on 1000+ experimental measurements
- Accuracy: MAE ≈ 20-30 K on known polymers

**Relevance to Phase 5:**
- **Exp 096:** Bicerano features for Tg
- Add group counts as features (not use model directly)
- Expected +0.008-0.018 on Tg
- Captures chemical intuition about structure-Tg relationships

**Groups to count:**
- Backbone: C-C, C-O, C=O, aromatic, amide, ester, ether, urethane
- Side chains: CH3, OH, COOH, NH2, halogens
- Rigidity: rings, double bonds, conjugation

---

### Polymer Genome Database

**Citation:**  
Kim, C., et al. (2018). "Polymer Genome: A Data-Powered Polymer Informatics Platform for Property Predictions." *International Journal of Molecular Sciences*, 19(9), 2809.

**Key Insights:**
- Hierarchical feature representation for polymers
- Separates backbone, side chains, end groups
- Accounts for tacticity, branching, chain length
- 1000+ polymers with experimental properties

**Relevance to Phase 5:**
- Inspiration for **Exp 097:** Backbone/side-chain decomposition
- PI1M dataset (995k polymers) is from PolyInfo (related source)

---

### Free Volume Theory for Tg

**Citation:**  
Fox, T. G., & Flory, P. J. (1950). "Second-Order Transition Temperatures and Related Properties of Polystyrene." *Journal of Applied Physics*, 21, 581.

**Key Insights:**
- **Tg correlates with fractional free volume:**
  - Below Tg: frozen, rigid, low free volume
  - Above Tg: mobile, flexible, high free volume
- Molecular packing efficiency predicts Tg
- VdW volume, rigid groups, bulky side chains reduce free volume → increase Tg

**Relevance to Phase 5:**
- **Exp 099:** Free volume proxy features
- VdW volume, packing efficiency, branching index
- Physical basis for Tg prediction

---

## Ensemble & Calibration Methods

### Non-Negative Least Squares (NNLS) Stacking

**Citation:**  
Breiman, L. (1996). "Stacked Regressions." *Machine Learning*, 24(1), 49-64.

**Key Insights:**
- Meta-learning: combine predictions from diverse models
- NNLS ensures non-negative weights (physical interpretation)
- Out-of-fold predictions prevent overfitting

**Relevance to Phase 5:**
- **V57 (baseline):** 5-model NNLS stack
- **Phase G (Exp 156-175):** Larger ensembles
- Simple, robust, no hyperparameter tuning

**Implementation:**
```python
from scipy.optimize import nnls

# Train on OOF predictions
weights, _ = nnls(oof_predictions, y_true)
weights /= weights.sum()  # Normalize

# Predict
y_pred = test_predictions @ weights
```

---

### Isotonic Regression Calibration

**Citation:**  
Zadrozny, B., & Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." *KDD*.

**Key Insights:**
- Non-parametric monotonic calibration
- Learns piecewise-constant calibration curve from OOF
- Guaranteed non-negative R² impact

**Relevance to Phase 5:**
- **Exp 167:** Isotonic calibration per target
- Safe post-processing (can't hurt)

---

## Test-Time Augmentation

### Randomized SMILES Augmentation

**Citation:**  
Bjerrum, E. J. (2017). "SMILES Enumeration as Data Augmentation for Neural Network Modeling of Molecules." *arXiv:1703.07076*.

**Key Insights:**
- **Any SMILES has multiple valid representations:**
  - Start numbering from different atoms
  - Randomize ring numbering, branch ordering
- Generate K variants → predict on each → average
- Variance reduction for sequence models (RNN, Transformer)
- **Fingerprints are invariant** (no benefit for GBM)

**Relevance to Phase 5:**
- **Exp 116-118:** TTA for Tg with K=5, 20
- **Phase H (Exp 176-182):** TTA for Transformer models
- Expected +0.003-0.010 variance reduction

**Implementation:**
```python
from rdkit import Chem

def randomize_smiles(smiles, n=10):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    return [Chem.MolToSmiles(mol, doRandom=True) for _ in range(n)]

# At test time
variants = randomize_smiles(test_smiles, n=20)
preds = [model.predict(v) for v in variants]
final_pred = np.median(preds)  # Median more robust than mean
```

---

## Validation Strategies

### Scaffold-Based Splitting

**Citation:**  
Bemis, G. W., & Murcko, M. A. (1996). "The Properties of Known Drugs. 1. Molecular Frameworks." *Journal of Medicinal Chemistry*, 39(15), 2887-2893.

**Key Insights:**
- **Murcko scaffold:** Core structure without side chains
- Molecules with same scaffold are similar
- Scaffold-stratified CV ensures diverse folds
- More realistic test of generalization than random splits

**Relevance to Phase 5:**
- **Exp 005, 119, 186:** Scaffold CV for validation
- Expect OOF R² drop (more honest)
- Should improve private LB prediction

**Implementation:**
```python
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

scaffolds = [get_scaffold(s) for s in train['smiles']]
# Use scaffolds as groups in GroupKFold
```

---

### Adversarial Validation

**Citation:**  
Commonly used in Kaggle competitions; no single citation.

**Key Insights:**
- Train classifier: train=0 vs test=1
- High AUC → distribution shift
- Features with high importance → where shift occurs
- Can reweight training data to match test distribution

**Relevance to Phase 5:**
- **Exp 194:** Adversarial validation diagnostic
- Understand pub-priv gap (0.026 in R2)
- May guide feature selection

---

## Feature Engineering

### Morgan Fingerprints

**Citation:**  
Rogers, D., & Hahn, M. (2010). "Extended-Connectivity Fingerprints." *Journal of Chemical Information and Modeling*, 50(5), 742-754.

**Key Insights:**
- Circular fingerprints: hash atom neighborhoods
- Radius 2 = 2 bonds away
- 2048 bits (typical)
- Captures local substructure

**Relevance to Phase 5:**
- **Baseline (V57):** Morgan 2048, radius 2
- Used in all experiments as baseline feature

---

### RDKit Descriptors

**Citation:**  
RDKit: Open-Source Cheminformatics Software. http://www.rdkit.org

**Relevant Descriptors:**
- Molecular weight, logP, TPSA
- Rotatable bonds, aromatic rings
- H-bond donors/acceptors
- Formal charge, radical electrons

**Relevance to Phase 5:**
- **Baseline (V57):** 200 RDKit descriptors
- **Exp 103:** Mordred descriptors (1600 total)

---

### Mordred Descriptors

**Citation:**  
Moriwaki, H., et al. (2018). "Mordred: a molecular descriptor calculator." *Journal of Cheminformatics*, 10, 4.

**Key Insights:**
- 1826 molecular descriptors
- Includes 2D and 3D features
- Topological, constitutional, geometric, electronic
- Redundancy: many correlated features

**Relevance to Phase 5:**
- **Exp 103:** Full Mordred descriptors for Tg
- Prune by variance + correlation
- Expected +0.008-0.020

---

## Technical Implementation Notes

### Data Hashes (Verification)

```
train.csv:      SHA-256 = 609b0f48...
test.csv:       SHA-256 = d8a0da26...
PI1M.csv:       SHA-256 = c5e1017b...
smile_r3.csv:   SHA-256 = c64f96ee...
```

Always verify before training to prevent data corruption errors.

---

### Python Environment

**GPU Laptop:**
```
Python 3.12.3
rdkit 2026.3.4
torch 2.11.0+cu128
torch-geometric 2.8.0
xgboost 3.3.0
lightgbm 4.7.0
scikit-learn 1.9.0
transformers (latest)
```

**Mac:**
```
Python 3.x (system)
Basic packages: numpy, pandas, scikit-learn, rdkit
No GPU required for CPU experiments
```

---

### Hardware Requirements

**Experiments by Hardware:**

| Type | Experiments | Hardware | Est. Time |
|------|-------------|----------|-----------|
| Feature Engineering | A, E1, F | Mac CPU | 5-60 min |
| GBM Training | A, E2, G | Mac CPU | 10-90 min |
| SVD/TF-IDF | B1 | Mac CPU | 20-120 min |
| word2vec | B2 | GPU | 30-90 min |
| Transformers | B3 | GPU | 2-16 hours |
| GNN | C | GPU | 1-8 hours |
| Multi-Task MLP | D | GPU | 30-180 min |
| GPR/KRR | F | Mac CPU | 5-30 min |

**GPU memory:**
- Small Transformer (128 hidden): ~2-4 GB
- Medium Transformer (256 hidden): ~6-10 GB
- GNN: ~4-8 GB
- RTX 5090 24GB is sufficient for all experiments

---

## Additional Resources

### Online Resources

- **RDKit Documentation:** https://www.rdkit.org/docs/
- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io/
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers/
- **Kaggle Polymer Dataset:** https://www.kaggle.com/competitions/aisehack-2-0

### Polymer Databases

- **PolyInfo:** https://polymer.nims.go.jp/ (PI1M source)
- **Khazana:** https://khazana.gatech.edu/ (oracle source)
- **Polymer Genome:** http://polymergenome.org/

### Code References

- **Round 2 Codebase:** `~/Desktop/AISEHack-2.0/` on GPU laptop
- **V57 Submission:** `final_submissions/` in main repo
- **Baseline Model:** `Dataset/base_line_model.ipynb`

---

**Document Version:** 1.0  
**Last Updated:** 2026-08-30  
**Maintained By:** Phase 5 Team
