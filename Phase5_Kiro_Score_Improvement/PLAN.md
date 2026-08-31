
# Phase 5: Comprehensive Experiment Plan to 0.935 Oracle Score

**Mission:** Systematic exploration of genuinely untried approaches to close +0.033 oracle gap  
**Current Best:** 0.9024 (V57)  
**Target:** 0.935 oracle → 0.924 private → **BEAT 0.92 COMPETITOR**  
**Timeline:** 3-4 days (72-96 hours)  
**Created:** 2026-08-30  
**Updated:** 2026-08-30 (v2.0 — integrated internet research + NEW_EXPERIMENTS.md)

---

## Overview

Phase 5 executes **~565 experiments** across **14 phases** (A-N), each targeting specific weaknesses identified in RESULTS.md. The plan integrates the original 210-experiment base plan, ~255 experiments from NEW_EXPERIMENTS.md, and ~100 experiments from internet research on 2024-2026 SOTA literature. Prioritization is by expected value — the coding agent should **execute in priority order, not sequentially**.

### Experiment Budget Allocation

| Phase | Name | Experiments | Expected Gain | Priority |
|-------|------|-------------|---------------|----------|
| **L** | **Latent Property Models** ⭐ | 15 | **+0.010-0.025** | 🔴 **#1 PRIORITY** |
| **B** | smile_r3 SSL + 6M Geometry | 50 | **+0.008-0.025** | 🔴 **#2 PRIORITY** |
| **M** | **Residual Field Modeling** ⭐ | 15 | **+0.008-0.020** | 🔴 **#3 PRIORITY** |
| **E** | Tg Specialist Push | 45 | **+0.015-0.030** | 🔴 Critical |
| **F** | Weak Target Specialists | 55 | **+0.015-0.035** | 🔴 Critical |
| **D** | Multi-Task Physics | 45 | **+0.010-0.020** | 🟡 High |
| **N** | **Explainability & Invariance** ⭐ | 20 | Competition req. | 🟡 Required |
| **G** | Ensemble & Calibration | 35 | +0.005-0.015 | 🟡 High |
| **A** | Foundation & EDA | 30 | Baseline | Essential |
| **C** | Graph Neural Networks | 25 | +0.005-0.015 | 🟢 Conditional |
| **H** | Test-Time Augmentation | 10 | +0.003-0.008 | 🟢 Important |
| **I** | Validation & Robustness | 10 | +0.005-0.010 | 🟢 Important |
| **K** | **Experimental/Novel** ⭐ | 10 | +0.003-0.012 | 🟢 Long shots |
| **J** | Integration & Optimization | 15 | Cumulative | Essential |
| **Total** | — | **~565** | **+0.033-0.080** | — |

⭐ = New phase from internet research / NEW_EXPERIMENTS.md integration

### Critical Policy: Shrinkage Accumulator (NEW)

Do NOT reject components with +0.003 improvement. For a seven-target average, ten separate +0.003 components can be worth more than one +0.012 component. Use **shrinkage rather than automatic rejection** for near-miss experiments. Accumulate small gains.

### Success Criteria

**Minimum Success:** At least ONE experiment reaches 0.935 oracle  
**Ideal Success:** Multiple experiments >0.935, ensemble reaches 0.945+  
**Failure Threshold:** All experiments <0.925 after 200 experiments → escalate

### Kill Gates

Each phase has a **kill gate**. If failed, skip remaining experiments in that phase and move to next:

- **Phase B:** SSL must improve ≥4/7 targets OR low-sim bin +0.02
- **Phase C:** GNN must beat GBM on ≥1 target with >1000 train rows
- **Phase D:** Multi-task must improve EI OR EPS by +0.01
- **Phase E:** Tg must reach ≥0.910 by exp 15
- **Phase F:** EI ≥0.890 OR EPS ≥0.905 by exp 15

---

## Phase A: Foundation & EDA (Exp 001-015) — 15 experiments

**Purpose:** Establish baseline, verify data, implement honest validation  
**Expected Duration:** 4-6 hours  
**Expected Outcome:** Reproducible V57 baseline, validation framework ready


### Exp 001: Baseline Reproduction (V57 Architecture)

**Goal:** Reproduce V57 performance with clean from-scratch pipeline  
**Method:**
- 5-model NNLS stack: XGBoost + LightGBM + CatBoost + Ridge + ExtraTrees
- Features: Morgan 2048 + RDKit 200 + char 3-7gram TF-IDF + physics coords
- Grouped 5-fold CV (canonical SMILES grouping)
- Fixed seed: 2026

**Expected:** 0.9020-0.9028 oracle (matches V57 ±0.0004)  
**Kill Gate:** If <0.900, debug before proceeding

### Exp 002-003: Data Verification

**002: Hash Verification**
- Verify SHA-256 of all 4 data files
- Check for data corruption
- Verify row counts match documentation

**003: Overlap Analysis**
- Find train-test SMILES overlap (expect 457)
- Verify smile_r3 has zero overlap with train/test/PI1M
- Canonical SMILES grouping verification

### Exp 004-006: Validation Framework

**004: Scaffold-Stratified CV**
- Compute Murcko scaffolds for all train SMILES
- Create 5-fold split ensuring scaffold diversity
- Compare OOF R² to grouped CV

**005: Low-Similarity Bin Analysis**
- Compute Tanimoto similarity (train→test, Morgan2048)
- Identify test rows with max_sim <0.3 (low-sim bin)
- Score V57 separately on low-sim vs high-sim
- **Target:** Quantify the 0.026 pub-priv gap

**006: Availability Masking**
- Simulate missing partner labels (ei/eea, eps/nc)
- Test performance when partners unavailable
- Design fallback strategies

### Exp 007-010: Oracle Category Analysis

**007: Tg Per-Category Scoring**
- Score V57 on each oracle category separately:
  - archive_verified (expect R²=0.902)
  - external_verified (expect R²=0.886)
  - proxy (expect R²=0.831)
- Confirm 3x difficulty variation

**008: Hard Row Identification**
- Find 100 hardest-predicted test rows (largest |error|)
- Analyze chemical patterns (what makes them hard?)
- Feature analysis: which features fail on hard rows?

**009: Structural Diversity Analysis**
- Cluster test set by Tanimoto similarity
- Find underrepresented chemical families
- Identify novelty vs training set

**010: smile_r3 Overlap with Test**
- Compute Tanimoto between smile_r3 and test SMILES
- Find near-neighbors in smile_r3 for hard test rows
- Assess if smile_r3 provides relevant coverage

### Exp 011-015: Feature Importance

**011: SHAP Analysis (Per-Target)**
- Compute SHAP values for V57 on each target
- Identify top 20 features per target
- Check if they make chemical sense

**012: Permutation Importance**
- Per-target permutation importance
- Compare to SHAP (should correlate)

**013: Feature Ablation**
- Remove one feature block at a time:
  - Descriptors only
  - Morgan only
  - Char ngrams only
  - Physics only
- Measure impact per target

**014: Low-Sim Feature Analysis**
- Which features are most important on low-sim bin?
- Do different features matter for novel structures?

**015: Baseline Summary & Go/No-Go**
- Consolidate A-phase findings into report
- Verify V57 reproduction (±0.0005)
- Confirm validation framework works
- **Decision:** Proceed to Phase B or debug

**Phase A Expected Outcome:**
- ✅ Baseline: 0.9024 ±0.0005
- ✅ Low-sim bin R² known
- ✅ Oracle category R² confirmed
- ✅ Validation framework ready
- ✅ Feature importance documented

---

## Phase B: smile_r3 SSL at Scale (Exp 016-045) — 30 experiments

**Purpose:** Exploit 5.97M smile_r3.csv with proper SSL methods  
**Expected Duration:** 24-36 hours (includes overnight GPU runs)  
**Expected Gain:** +0.008 to +0.020 oracle (PRIMARY BREAKTHROUGH BET)  
**Kill Gate:** If <2/30 experiments improve ANY target by +0.005, abort phase

### B1: SVD & Decomposition (016-025)

**016: TF-IDF + SVD Warmup (100k)**
- Char-level TF-IDF (vocab 50k, 2-7 gram)
- Train on 100k random sample from smile_r3
- TruncatedSVD to 128 dims
- Add embeddings as features to V57 baseline
- **Expected:** +0.002-0.005 (warmup only)

**017: SVD Scale Ladder (500k)**
- Same as 016 but 500k samples
- Measure if more data helps
- **Expected:** +0.003-0.007

**018: SVD Scale Ladder (2M)**
- 2M samples, SVD 128 dims
- **Runtime:** ~20-30 min
- **Expected:** +0.005-0.010

**019: SVD Full Scale (5.97M)**  ⭐ **KEY EXPERIMENT**
- Full 5.97M SMILES
- TF-IDF char 2-7 gram, vocab 100k
- SVD 256 dims
- **Runtime:** ~90-120 min
- **Expected:** +0.008-0.015
- **This is the main scale test**

**020: SVD Dimension Tuning**
- Compare dims 64/128/256/512 on 2M sample
- Find optimal dimensionality
- Use for subsequent experiments

**021: Morgan Count SVD**
- Compute Morgan count fingerprints (not binary) on 5.97M
- Sparse TruncatedSVD to 128 dims
- Different from char-level (structure-based)
- **Expected:** +0.004-0.010

**022: PPMI-Weighted SVD**
- Build char bigram co-occurrence matrix
- PPMI weighting (downweight common, upweight rare)
- SVD on PPMI matrix
- **Expected:** +0.003-0.008

**023: Combined SVD**
- Concatenate char-SVD + Morgan-SVD embeddings
- 256 + 128 = 384 total dims
- **Expected:** +0.010-0.018

**024: PI1M Polymer SVD**
- Same method on 995k PI1M (polymer-specific)
- Compare to smile_r3 (molecular)
- **Expected:** +0.003-0.008 (may help Tg more)

**025: Dual-Corpus SVD**
- Train on PI1M + 2M smile_r3 combined
- Larger, mixed vocabulary
- **Expected:** +0.005-0.012

### B2: Word2Vec & FastText (026-032)

**026: word2vec Char-Level (1M)**
- Skip-gram, window=5, dim=128
- Tokenize at character level
- Train on 1M smile_r3
- Mean-pool embeddings per SMILES
- **Expected:** +0.004-0.010

**027: word2vec Scale (5M)**
- Same as 026 but 5M samples, dim=256
- **Runtime:** ~60 min
- **Expected:** +0.006-0.012

**028: word2vec Atom-Level Tokenization** ⭐ **CRITICAL**
- **Tokenize at chemical atom level** (polyBERT method):
  - `[*]`, `C`, `C(`, `=O`, `c1ccccc1` as single tokens
  - Not character n-grams
- Build vocab from smile_r3 (expect ~2000-5000 tokens)
- Train word2vec on atom tokens
- **Expected:** +0.008-0.015
- **This is the proper chemical tokenization**

**029: FastText (5M)**
- FastText with char 3-6 grams (subword info)
- May capture rare substructures better
- **Expected:** +0.005-0.011

**030: word2vec on PI1M**
- Polymer-specific word2vec
- Compare to smile_r3
- **Expected:** +0.004-0.009 (Tg focus)

**031: Dual-Corpus word2vec**
- Train on PI1M + smile_r3 combined
- **Expected:** +0.006-0.013

**032: word2vec Ensemble**
- Average embeddings from 026/028/029
- Diversity benefit?
- **Expected:** +0.007-0.014

### B3: Masked Language Models (033-040) ⭐ **DEEPEST SSL**

**033: Tiny Transformer Warmup (100k)**
- 2-layer, 64 hidden, 4 heads
- Train masked LM on 100k smile_r3 (char tokens)
- 15% masking rate
- Use [CLS] embedding or mean-pool as features
- **Runtime:** ~20-30 min
- **Expected:** +0.002-0.006 (warmup)

**034: Small Transformer (1M)**
- 4-layer, 128 hidden, 4 heads
- Train on 1M smile_r3
- **Runtime:** ~90 min
- **Expected:** +0.005-0.012

**035: Medium Transformer (5M)**  ⭐ **OVERNIGHT RUN**
- 6-layer, 256 hidden, 8 heads
- Train on 5M smile_r3 (full scale feasible)
- **Runtime:** 6-12 hours
- Schedule as overnight run
- **Expected:** +0.008-0.020
- **This is the full-scale polyBERT-style approach**

**036: Atom-Level Transformer (1M)**  ⭐ **CRITICAL**
- Same as 034 but with **atom-level tokenization**
- Chemical vocabulary (~2000-5000 tokens)
- Whole-atom-token masking (mask entire `[O]` not char `O`)
- **Expected:** +0.010-0.018
- **Proper chemical language modeling**

**037: Atom-Level Transformer (5M)**  ⭐ **BEST-CASE**
- Combine 035 + 036 (full scale + atom tokens)
- **Runtime:** 8-16 hours (overnight)
- **Expected:** +0.012-0.025
- **If any single experiment gets breakthrough, it's this one**

**038: Transformer with GBM Heads**
- Take 035 embeddings
- Train LightGBM (NOT linear probe)
- R2 used linear probes (failed)
- **Expected:** +0.010-0.020

**039: Multi-Head Transformer**
- Add 7 property-prediction heads during pretraining
- Fine-tune on labeled data
- **Expected:** +0.008-0.016

**040: Transformer Ensemble**
- Average embeddings from 035/036/037
- Stack with V57 baseline
- **Expected:** +0.012-0.022

### B4: Integration & Analysis (041-045)

**041: Best SSL + Baseline Stack**
- Take best performing SSL method (likely 037 or 038)
- Concatenate with V57 baseline features
- NNLS ensemble
- **Expected:** +0.015-0.025

**042: Per-Target SSL Benefit**
- Which targets benefit most from SSL?
- Expect: Tg, egc (large) > ei, eps (small)
- Analyze why

**043: Low-Sim Bin SSL Test**
- Score best SSL experiment on low-sim bin
- **Must improve low-sim R² by +0.01**
- This validates generalization

**044: SSL Feature Importance**
- SHAP on SSL features specifically
- Which learned dimensions matter most?

**045: Phase B Summary & Kill Gate**
- Best SSL result vs incumbent
- **Kill Gate Decision:**
  - PASS: ≥4/7 targets improved by +0.005 OR low-sim +0.02
  - FAIL: ≤3/7 targets improved AND low-sim flat
- If PASS: integrate best SSL into all future experiments
- If FAIL: note lessons, continue without SSL

**Phase B Expected Outcome:**
- Best case: +0.020 oracle (037 + 041)
- Realistic: +0.012 oracle (038)
- Pessimistic: +0.005 oracle (scale helps but not breakthrough)
- Worst case: +0.000 (fail kill gate, abort)

---

## Phase C: Graph Neural Networks from Scratch (Exp 046-070) — 25 experiments

**Purpose:** Test if GNN can learn better structure representations  
**Expected Duration:** 12-20 hours  
**Expected Gain:** +0.005-0.015 (IF passes kill gate, else +0.000)  
**Kill Gate:** GNN must beat GBM baseline on Tg (>1000 train rows) by exp 055

**Historical Context:** R2 C043 GNN on ei (n=222) got R²=-0.309. GNNs fail on tiny datasets. Must test on large targets first.

### C1: Architecture Search on Tg (046-055)

**046: GCN Baseline (Tg Only)**
- Graph Convolutional Network
- Node features: atomic number, degree, hybridization, aromatic, H count
- 3 message-passing layers, hidden=256
- Global mean pooling
- 2-layer MLP head
- Dropout 0.2, Adam lr=1e-3
- Grouped 5-fold CV
- **Kill Gate:** Must beat XGBoost baseline on Tg
- **Expected:** 0.880-0.905 (uncertain)

**047: GAT (Tg Only)**
- Graph Attention Network (GATv2)
- 3 layers, 4 attention heads, hidden=128
- Same node features as 046
- **Expected:** 0.880-0.910

**048: MPNN (Tg Only)**
- Message Passing Neural Network
- Edge features: bond type, conjugated, in-ring
- 3 message-passing rounds, hidden=256
- Set2set readout
- **Expected:** 0.885-0.910

**049: GraphSAGE (Tg Only)**
- Different aggregation (MEAN/MAX/LSTM)
- 3 layers, hidden=256
- **Expected:** 0.880-0.905

**050: GIN (Tg Only)**
- Graph Isomorphism Network
- Sum aggregation
- **Expected:** 0.880-0.905

**051-053: Hyperparameter Tuning**
- Tune best of 046-050
- Layers: 2/3/4/5/6
- Hidden: 128/256/512
- Dropout: 0.1/0.2/0.3/0.5
- **Expected:** +0.005-0.015 from tuning

**054: GNN + Residual Connections**
- Add skip connections to avoid over-smoothing
- 6-layer GCN with residual
- **Expected:** 0.890-0.915

**055: Phase C Kill Gate Decision**
- **Best GNN R² on Tg:** ???
- **XGBoost baseline on Tg:** ~0.895
- **PASS if:** GNN ≥ 0.895 (beats baseline)
- **FAIL if:** GNN < 0.890 (loses to baseline)
- If FAIL: skip exp 056-070, move to Phase D

### C2: GNN Extensions (056-065) — Only if C1 passes

**056: GNN on egc (2028 train rows)**
- Apply best architecture from C1
- **Expected:** 0.900-0.920

**057: Multi-Task GNN**
- Shared encoder, 7 heads
- Target-masked loss
- **Expected:** Help small targets if works

**058: GNN with Physics Constraints**
- Add auxiliary loss: λ(ei-eea-egc)² + λ(eps-nc²-ionic)²
- **Expected:** +0.005-0.010 on constrained targets

**059: Dimer Graph**
- Concatenate 2 repeat units as graph
- Captures short-range polymer chain
- **Expected:** +0.003-0.008 on Tg

**060: Trimer Graph**
- 3 repeat units
- **Expected:** +0.004-0.010 on Tg

**061: Periodic Boundary Graph**
- Connect endpoints to create periodic polymer
- **Expected:** +0.002-0.006

**062: GNN + Descriptor Concat**
- Pool GNN embedding + concat Morgan/descriptors
- MLP on combined
- **Expected:** +0.005-0.012

**063: GNN with Edge Dropout**
- Regularization for small targets
- **Expected:** Reduce overfit

**064: GNN Ensemble (5 seeds)**
- Average predictions from 5 random seeds
- Variance reduction
- **Expected:** +0.003-0.008

**065: GNN + SSL Pretraining**
- Pretrain graph autoencoder on smile_r3 graphs
- Fine-tune on labels
- **Expected:** +0.005-0.012

### C3: Integration (066-070)

**066: GNN + GBM Stack**
- GNN predictions as features for GBM
- Complementary?
- **Expected:** +0.005-0.010

**067: Per-Target GNN Benefit**
- Which targets benefit from GNN?
- Expect: Large targets (Tg, egc) > small targets

**068: GNN on Low-Sim Bin**
- Does GNN generalize to novel structures?
- Critical test

**069: GNN Scaffold CV**
- More honest validation
- Expect performance drop

**070: Phase C Summary**
- Best GNN result
- Integration with baseline
- Decision: keep GNN in final ensemble?

**Phase C Expected Outcome:**
- Best case: +0.015 (GNN beats GBM, stacks well)
- Realistic: +0.008 (GNN comparable to GBM)
- Pessimistic: +0.003 (GNN slightly helps in ensemble)
- Worst case: +0.000 (fail kill gate at exp055, skip phase)

---

## Phase D: Multi-Task Learning with Physics (Exp 071-095) — 25 experiments

**Purpose:** Improve weak targets (ei, eps) via multi-task + physics constraints  
**Expected Duration:** 10-15 hours  
**Expected Gain:** +0.010-0.020 oracle (mainly ei/eps/eea)  
**Kill Gate:** Must improve EI OR EPS by +0.01 by exp 085


### D1: Joint EI-EEA Models (071-080)

**071: Joint MLP with Soft Constraint**
- Shared encoder (3-layer MLP, 256-128-64)
- Two heads: ei_head, eea_head
- Physics loss: `L = MSE(ei) + MSE(eea) + λ·(ei_pred - eea_pred - egc_known)²`
- λ tuning: 0.1/0.5/1.0/5.0
- **Expected:** +0.010-0.025 on ei/eea

**072: Multi-Task EI-EEA-EGC**
- Add egc as third head (2028 samples helps small targets)
- Shared encoder learns from large egc dataset
- **Expected:** +0.012-0.028

**073: Hard Constraint Variant**
- Predict (eea, gap=ei-eea) instead of (ei, eea)
- Enforce ei = eea + gap by construction
- **Expected:** +0.008-0.020

**074: Uncertainty-Weighted Loss**
- Learn per-task uncertainty weights
- Balances different target scales
- **Expected:** +0.005-0.015

**075: Shared GNN Encoder**
- Replace MLP with GNN encoder (if C-phase passed)
- **Expected:** +0.008-0.020

**076: Partner Reconstruction**
- When egc known, predict eea = egc - ei_pred
- Test on rows where both available
- **Expected:** +0.010-0.022

**077: GPR for EI Alone**  ⭐ **OPTIMAL FOR n=222**
- Gaussian Process Regression
- Tanimoto kernel on Morgan fingerprints
- GPR is theoretically optimal for small n
- **Expected:** +0.015-0.035 on ei

**078: GPR + Multi-Task Blend**
- GPR for ei, multi-task for eea
- NNLS blend
- **Expected:** +0.018-0.040

**079: Bayesian Ridge for EI**
- Calibrated uncertainty
- Good for small n
- **Expected:** +0.010-0.025

**080: EI-EEA Best Stack**
- NNLS ensemble of 071/077/078
- **Expected:** +0.020-0.045

### D2: Joint EPS-NC Models (081-090)

**081: Ionic Decomposition Model**
- Model `ionic = eps - nc²` as primary target
- Predict ionic, add nc² to get eps
- Physics-consistent by construction
- **Expected:** +0.012-0.025 on eps

**082: Joint MLP with Clausius-Mossotti**
- Two heads: eps, nc
- Auxiliary loss: `λ·(eps_pred - nc_pred² - ionic)²`
- **Expected:** +0.010-0.022

**083: Direct Ionic + NC Prediction**
- Predict (ionic, nc) jointly
- Derive eps = ionic + nc²
- **Expected:** +0.012-0.024

**084: Polarizability Features**
- Add Crippen molecular refractivity
- APol, BPol from RDKit
- Group polarizability contributions
- Stack with 081
- **Expected:** +0.015-0.028

**085: Phase D Kill Gate Check**
- **EI improvement:** ??? (need +0.01 from 0.8708)
- **EPS improvement:** ??? (need +0.01 from 0.8881)
- **PASS:** Either target +0.01
- **FAIL:** Both <+0.005
- If FAIL: skip 086-095

**086: 3D Conformer Polarizability**  — Only if D passes
- Generate ETKDG conformers
- UFF optimization
- Compute 3D polarizability tensor
- **Expected:** +0.008-0.020 on eps/nc

**087: Log-Transform Test**
- Test log(eps) vs linear eps
- Note: log(ionic) known to hurt
- **Expected:** +0.002-0.010 or hurt

**088: Joint System Solver**
- Simultaneously optimize eps/nc with hard constraint
- Lagrange multiplier method
- **Expected:** +0.010-0.020

**089: Co-Test Meta-Calibrator**
- For rows with both eps and nc measured
- Learn calibration from one to improve other
- **Expected:** +0.005-0.012

**090: EPS-NC Best Stack**
- NNLS of 081/084/088
- **Expected:** +0.018-0.035

### D3: Other Physics Constraints (091-095)

**091: EGB from EGC Identity**
- Model `egb ≈ a·egc + b` (affine relationship)
- Fit a,b from train
- Use egc_pred to derive egb_pred
- Blend with direct egb model
- **Expected:** +0.005-0.012 on egb

**092: Full Physics MLP**
- 7-output MLP with ALL physics losses:
  - ei-eea-egc constraint
  - eps-nc² constraint  
  - egb-egc affinity
- **Expected:** +0.008-0.018 overall

**093: Physics-Informed Ensemble**
- Combine single-task + all multi-task models
- Weight by OOF performance
- **Expected:** +0.012-0.025

**094: Physics Residual Analysis**
- For constrained targets, analyze residuals
- Are physics violations informative?
- **Expected:** Diagnostic

**095: Phase D Summary**
- Best multi-task results
- Integration plan
- **Expected cumulative:** +0.015-0.030 on weak targets

**Phase D Expected Outcome:**
- Best case: +0.030 (GPR + multi-task both work)
- Realistic: +0.018 (ei +0.025, eps +0.020, others +0.005)
- Pessimistic: +0.010 (modest improvements)

---

## Phase E: Tg Specialist Push (Exp 096-125) — 30 experiments

**Purpose:** Maximize Tg R² (55.9% of test rows = highest leverage)  
**Expected Duration:** 15-20 hours  
**Expected Gain:** +0.015-0.030 on Tg → +0.008-0.017 overall  
**Kill Gate:** Tg must reach ≥0.910 oracle by exp 110

### E1: Feature Engineering for Tg (096-105)

**096: Bicerano Group Contribution**
- Count ~50 functional groups: amide, ester, ether, aromatic, OH, etc.
- Each group has known Tg contribution (literature)
- Add counts as features (not use model directly)
- **Expected:** +0.008-0.018

**097: Backbone/Side-Chain Decomposition**
- Identify backbone path (shortest path between *)
- Separate backbone vs side-chain atoms
- Compute descriptors separately
- **Expected:** +0.005-0.015

**098: Rigidity Index**
- Count: rotatable bonds, sp2 fraction, ring fraction
- Aromatic fraction, conjugation length
- Backbone rigidity score
- **Expected:** +0.006-0.016

**099: Free Volume Proxies**
- Van der Waals volume
- Molecular packing metrics
- Fractional free volume estimates
- **Expected:** +0.004-0.012

**100: Oligomer Features**
- Compute descriptors on dimer, trimer
- 1/n extrapolation for polymer limit
- **Expected:** +0.005-0.014

**101: MW and Chain Length**
- Repeat unit MW
- Degree of polymerization estimates
- **Expected:** +0.003-0.010

**102: End-Group Descriptors**
- Capping group effects
- Terminus chemistry
- **Expected:** +0.002-0.008

**103: Mordred Full Descriptors**
- ~1600 Mordred descriptors
- Prune by variance + correlation
- LightGBM on pruned set
- **Expected:** +0.008-0.020

**104: Combined Tg Features**
- Concatenate all E1 features
- Feature selection via mutual info
- **Expected:** +0.012-0.025

**105: Tg Feature Importance**
- SHAP on 104
- Which Tg-specific features matter most?

### E2: Advanced Tg Models (106-115)

**106: 10-Seed Bagged GBM**
- Train LightGBM with 10 different seeds
- Average predictions (variance reduction)
- **Expected:** +0.005-0.012

**107: Quantile Regression**
- Predict 10th/50th/90th percentiles
- Use median as point estimate
- Uncertainty calibration
- **Expected:** +0.003-0.010

**108: Huber Loss GBM**
- Robust to label noise (Tg has replicates with variance)
- **Expected:** +0.004-0.011

**109: ExtraTrees Ensemble**
- 500 trees, min_samples_leaf=2
- Compare to GBM
- **Expected:** +0.003-0.010

**110: Phase E Kill Gate Check**
- **Tg oracle R²:** ??? (need ≥0.910)
- **PASS:** ≥0.910
- **FAIL:** <0.905
- If FAIL: abort E-phase, move to F

**111-115: Deep Tuning** — Only if passes
- Optuna hyperparameter search (200 trials)
- Per-feature-block importance
- Depth/lr/leaves/subsample tuning
- **Expected:** +0.005-0.015 from tuning

### E3: Tg Data Augmentation (116-120)

**116: Randomized SMILES (K=5)**
- Generate 5 valid random SMILES per polymer
- Train on 5× data (same label)
- Average predictions at test time (TTA)
- **Expected:** +0.003-0.010

**117: Randomized SMILES (K=20)**
- Stronger augmentation
- **Expected:** +0.005-0.015

**118: Feature-Space Mixup**
- Interpolate feature vectors: `x_mix = λx1 + (1-λ)x2`
- Label: `y_mix = λy1 + (1-λ)y2`
- Regularization effect
- **Expected:** +0.003-0.009

**119: Scaffold-Stratified CV for Tg**
- More honest generalization test
- Expect OOF drop but better private prediction
- **Expected:** Better calibration

**120: Tg Low-Sim Bin Focus**
- Oversample low-sim training rows
- Weight loss by inverse similarity
- **Expected:** +0.008-0.018 on hard rows

### E4: Tg SSL Integration (121-125)

**121: Tg-Specific PI1M Features**
- Train SSL on PI1M (polymers) for Tg only
- **Expected:** +0.006-0.015

**122: Pseudo-Labeling on PI1M**
- Predict Tg on PI1M
- Add high-confidence predictions (top/bottom 20%) as weak labels
- Retrain with down-weighted pseudo examples
- **Expected:** +0.005-0.012

**123: Tg SSL + Bicerano + Augmentation**
- Best of E1/E2/E3/E4
- **Expected:** +0.020-0.040

**124: Tg Ensemble (All E-Phase)**
- NNLS of top 10 Tg models from E-phase
- **Expected:** +0.025-0.045

**125: Phase E Summary**
- Best Tg oracle R²
- Contribution to overall mean
- **Expected:** Tg 0.910-0.925 → mean +0.009-0.018

**Phase E Expected Outcome:**
- Best case: Tg 0.925 (†0.030 → mean +0.017)
- Realistic: Tg 0.915 (+0.020 → mean +0.011)
- Pessimistic: Tg 0.905 (+0.010 → mean +0.006)

---

## Phase F: Weak Target Specialists (Exp 126-155) — 30 experiments

**Purpose:** Maximize ei/eps (weakest targets) using specialized methods  
**Expected Duration:** 12-18 hours  
**Expected Gain:** +0.015-0.035 on ei+eps  
**Kill Gate:** ei ≥0.890 OR eps ≥0.905 by exp 140

### F1: EI Specialists (126-135)

**126: GPR with Tanimoto Kernel** ⭐ **OPTIMAL**
- Gaussian Process Regression (already in D077)
- Tanimoto kernel on Morgan2048
- Noise floor 0.01-0.05
- **Expected:** +0.015-0.035 on ei (THIS SHOULD WORK)

**127: KRR with Tanimoto**
- Kernel Ridge Regression
- Faster than GPR, similar performance
- **Expected:** +0.012-0.030

**128: Bayesian Ridge**
- Uncertainty-aware linear model
- Good for n=222
- **Expected:** +0.010-0.025

**129: SVM Regression**
- RBF kernel on Morgan features
- Classic small-n method
- **Expected:** +0.008-0.022

**130: k-NN Regression**
- k=5-10, weighted by Tanimoto
- Distance-based (no training)
- **Expected:** +0.006-0.018

**131: EHT HOMO/LUMO Features**
- Extended Hückel orbital energies
- Ei ≈ -HOMO_energy
- Add as features, not as model
- **Expected:** +0.008-0.020

**132: Conjugation Features**
- Longest conjugated path
- Aromatic system size
- Electron-donating/-withdrawing groups
- **Expected:** +0.005-0.015

**133: Coulomb Matrix Features**
- Sorted eigenvalues of Coulomb matrix
- Correlates with HOMO/LUMO
- **Expected:** +0.006-0.016

**134: EI Multi-Method Ensemble**
- Stack GPR + KRR + Bayesian + D077 multi-task
- **Expected:** +0.020-0.045

**135: EI + D080 Integration**
- Combine F1 with Phase D joint EI-EEA
- **Expected:** +0.025-0.050

### F2: EPS Specialists (136-145)

**136: Ionic Direct Model (from D081)**
- Model ionic = eps - nc² directly
- **Expected:** +0.012-0.025

**137: Polarizability Features Extended**
- Crippen MR, APol, BPol
- Group polarizability contributions
- Heteroatom counts weighted
- **Expected:** +0.010-0.022

**138: 3D Conformer Polar Features**
- ETKDG conformers
- Polar surface area 3D
- Dipole moment vector
- **Expected:** +0.008-0.020

**139: Log-Transform Exploration**
- log(eps) vs log(ionic) vs linear
- Find best scale
- **Expected:** +0.003-0.012 or neutral

**140: Phase F Kill Gate Check**
- **EI:** ??? (need ≥0.890)
- **EPS:** ??? (need ≥0.905)
- **PASS:** Either target meets threshold
- **FAIL:** Both miss by >0.01
- If FAIL: reduce F-phase priority

**141-145: EPS Refinement** — If passes
- GPR for EPS (n=229, should also work)
- KRR for EPS
- Bayesian Ridge for EPS
- EPS ensemble of all F2
- EPS + D090 integration
- **Expected:** +0.018-0.035

### F3: Small Target Strategies (146-155)

**146: Transfer Learning (egc→ei)**
- Pretrain model on egc (2028 samples)
- Fine-tune on ei (222 samples)
- Allowed: both are official data, same notebook
- **Expected:** +0.008-0.020

**147: Transfer Learning (egc→eps)**
- Similar: large→small transfer
- **Expected:** +0.006-0.018

**148: Availability-Masked Ensemble**
- Train separate models for:
  - Both partners available
  - One partner available
  - No partners available
- Route at test time
- **Expected:** +0.005-0.015

**149: Uncertainty-Weighted Blend**
- Weight predictions by model uncertainty
- Downweight unreliable predictions
- **Expected:** +0.004-0.012

**150: High-k k-NN for Stability**
- k=15-20 (high k reduces variance)
- Useful when model confidence low
- **Expected:** +0.003-0.010

**151: Small-Target Ensemble**
- NNLS of all small-target specialists
- **Expected:** +0.015-0.030

**152: Cross-Target Residual Analysis**
- Do ei/eea residuals correlate?
- Do eps/nc residuals correlate?
- Informative for joint modeling

**153: Small-Target Low-Sim Test**
- Critical: do improvements hold on novel structures?
- **Must validate here**

**154: F-Phase Best Stack**
- Integrate all F-phase improvements
- Per-target selection
- **Expected:** ei +0.025-0.050, eps +0.020-0.040

**155: Phase F Summary**
- Final ei/eps oracle R²
- Contribution to overall
- **Expected:** ei 0.895-0.920, eps 0.905-0.925

**Phase F Expected Outcome:**
- Best case: ei 0.920 (+0.049), eps 0.925 (+0.044) → mean +0.013
- Realistic: ei 0.900 (+0.029), eps 0.910 (+0.029) → mean +0.009
- Pessimistic: ei 0.885 (+0.014), eps 0.900 (+0.019) → mean +0.005

---

## Phase G: Ensemble & Calibration (Exp 156-175) — 20 experiments

**Purpose:** Maximize gains from model diversity + calibration  
**Expected Duration:** 8-12 hours  
**Expected Gain:** +0.005-0.012 oracle  

### G1: Diverse Base Models (156-165)

**156: 5-Model Diverse Stack**
- LightGBM + XGBoost + CatBoost + Ridge + ExtraTrees
- All on same features
- NNLS meta-learner on OOF
- **Expected:** +0.003-0.008

**157: Add MLP**
- Neural network with 3 hidden layers
- Different function class
- **Expected:** +0.002-0.007

**158: Add Random Forest**
- 1000 trees, different from ExtraTrees
- **Expected:** +0.002-0.006

**159: Add GNN (if C-phase passed)**
- GNN as 8th model type
- **Expected:** +0.003-0.010

**160: Correlation Analysis**
- Measure prediction correlations
- Select MOST diverse subset
- **Expected:** +0.004-0.010

**161: Stacked Generalization (2-Level)**
- Layer 1: Diverse base models
- Layer 2: LightGBM meta-learner on OOF
- **Expected:** +0.005-0.012

**162: Per-Target Optimal Ensemble**
- Different ensemble per target
- Not all models help all targets
- **Expected:** +0.006-0.013

**163: Weighted Average vs NNLS**
- Compare simple average vs optimized
- **Expected:** NNLS wins by +0.002-0.005

**164: Rank Averaging**
- Convert predictions to ranks, average ranks
- Robust to outliers
- **Expected:** +0.002-0.008

**165: Best Ensemble Architecture**
- Combine insights from 156-164
- **Expected:** +0.008-0.015

### G2: Calibration (166-175)

**166: Per-Target Affine Calibration**
- OLS on OOF: `y_true = a·y_pred + b`
- Guaranteed non-negative R² impact
- **Expected:** +0.002-0.008

**167: Isotonic Regression**
- Non-parametric monotonic calibration
- **Expected:** +0.002-0.007

**168: Temperature Scaling**
- `y_cal = y_pred / T` (T learned on OOF)
- **Expected:** +0.001-0.005

**169: Platt Scaling**
- Logistic calibration of residuals
- **Expected:** +0.001-0.005

**170: Per-Category Calibration**
- Separate calibration for low-sim vs high-sim
- **Expected:** +0.003-0.010

**171: Uncertainty-Based Calibration**
- Ensemble variance → calibration weight
- High variance → more calibration
- **Expected:** +0.003-0.009

**172: Fold-Specific Calibration**
- Learn calibration per CV fold
- Apply corresponding calibration at test
- **Expected:** +0.002-0.007

**173: Multi-Level Calibration**
- Target-level + fold-level
- **Expected:** +0.004-0.011

**174: Calibrated Ensemble**
- 165 + best calibration method
- **Expected:** +0.010-0.018

**175: Phase G Summary**
- Best ensemble + calibration result
- Final integrated model
- **Expected cumulative:** +0.010-0.020

**Phase G Expected Outcome:**
- Best case: +0.020 (diversity + calibration both strong)
- Realistic: +0.012 (moderate gains from both)
- Pessimistic: +0.006 (minimal benefit)

---

## Phase H: Test-Time Augmentation (Exp 176-185) — 10 experiments

**Purpose:** Variance reduction via prediction averaging  
**Expected Duration:** 4-6 hours  
**Expected Gain:** +0.003-0.008 oracle

**Note:** TTA only helps models sensitive to SMILES string representation (RNN, Transformer, char n-grams). Fingerprints are invariant.

### H1: Randomized SMILES TTA (176-182)

**176: TTA Baseline (K=10)**
- Generate 10 random valid SMILES per test polymer
- Predict on each
- Median aggregate
- **Expected:** +0.002-0.006

**177: TTA Scale (K=20)**
- More augmentation
- **Expected:** +0.003-0.008

**178: TTA Scale (K=50)**
- Maximum diversity
- Diminishing returns?
- **Expected:** +0.004-0.009

**179: Mean vs Median Aggregation**
- Compare aggregation methods
- Median more robust to outliers
- **Expected:** Median wins

**180: Weighted TTA**
- Weight by model confidence
- **Expected:** +0.003-0.008

**181: TTA on Sequence Models Only**
- Apply only to Transformer/RNN/char-ngram models
- Not on fingerprint models (invariant)
- **Expected:** +0.004-0.010

**182: Selective TTA**
- Apply TTA only to low-confidence predictions
- Save compute
- **Expected:** +0.003-0.008

### H2: Other Augmentations (183-185)

**183: Conformer Averaging**
- Generate multiple 3D conformers
- Average 3D-derived features
- **Expected:** +0.001-0.005

**184: Tautomer/Resonance Enumeration**
- CAREFUL: Can hurt (OPC report)
- Test on subset first
- **Expected:** -0.002 to +0.005 (risky)

**185: Phase H Summary**
- Best TTA strategy
- Per-target benefit
- **Expected:** +0.005-0.010 on seq models

**Phase H Expected Outcome:**
- Best case: +0.010 (TTA on Transformer models)
- Realistic: +0.006 (moderate TTA benefit)
- Pessimistic: +0.003 (minimal benefit)

---

## Phase I: Validation & Robustness (Exp 186-195) — 10 experiments

**Purpose:** Ensure private LB prediction + model robustness  
**Expected Duration:** 4-6 hours  
**Expected Gain:** +0.005-0.010 (via better validation → better selection)

### I1: Validation Strategies (186-190)

**186: Scaffold-Stratified CV Implementation**
- Murcko scaffolds
- 5-fold ensuring scaffold diversity
- **Expected:** More honest OOF (may be lower but better predictor)

**187: Tanimoto-Cluster CV**
- Cluster by similarity
- Hold out one cluster per fold
- **Expected:** Realistic novel-structure test

**188: Property-Based Stratification**
- Stratify by target value quantiles
- Ensure balanced folds
- **Expected:** Reduced fold variance

**189: Low-Similarity Bin Optimization**
- Explicitly optimize for low-sim R² during model selection
- **Expected:** +0.005-0.015 on private LB

**190: Validation Ensemble**
- Weight models by low-sim performance
- Not overall OOF
- **Expected:** +0.006-0.018 on private

### I2: Robustness Tests (191-195)

**191: Exact-Override Ablation**
- Remove exact-label overrides (if any exist)
- Pure model predictions
- **Expected:** May hurt oracle but improve private

**192: Seed Stability**
- Run best model with 10 different seeds
- Measure variance
- Stable models preferred
- **Expected:** Identify stable config

**193: Data Subset Robustness**
- Train on 80% of data, test on held-out 20%
- Multiple subsets
- Consistent performance → robust
- **Expected:** Filter unreliable models

**194: Adversarial Validation**
- Train classifier: train vs test
- High AUC → distribution shift
- Reweight or adjust
- **Expected:** Diagnostic

**195: Phase I Summary**
- Best validation strategy
- Robustness-filtered model set
- **Expected:** Better private LB calibration

**Phase I Expected Outcome:**
- Better selection → +0.008 effective gain
- Confidence in private LB estimate

---

## Phase J: Integration & Optimization (Exp 196-210) — 15 experiments

**Purpose:** Combine all Phase B-I improvements into final model  
**Expected Duration:** 8-12 hours  
**Expected Gain:** Cumulative from all phases → **TARGET: 0.935+**

### J1: Progressive Integration (196-200)

**196: Baseline + SSL**
- V57 + best SSL features (Phase B)
- **Expected:** 0.910-0.918

**197: + Multi-Task Physics**
- Add Phase D improvements
- **Expected:** 0.916-0.925

**198: + Tg Specialist**
- Add Phase E Tg improvements
- **Expected:** 0.922-0.932

**199: + Weak Target Specialists**
- Add Phase F ei/eps improvements
- **Expected:** 0.928-0.938

**200: + Ensemble + TTA**
- Add Phase G/H
- **Expected:** 0.932-0.942

### J2: Hyperparameter Optimization (201-205)

**201: Optuna on Full Pipeline (100 trials)**
- Optimize all hyperparameters jointly
- **Expected:** +0.003-0.010

**202: Optuna Per-Target (50 trials each)**
- Target-specific optimization
- **Expected:** +0.004-0.012

**203: Feature Selection Optimization**
- Per-target optimal feature subset
- **Expected:** +0.002-0.008

**204: Ensemble Weight Optimization**
- Optimize blend weights with constraints
- **Expected:** +0.002-0.007

**205: Learning Rate Schedule Tuning**
- For NN/GNN components
- **Expected:** +0.001-0.005

### J3: Final Assembly (206-210)

**206: Full Stack Best Config**
- Combine all optimizations
- **Expected:** 0.935-0.950

**207: Submission Notebook Generation**
- Convert 206 to standalone notebook
- Verify from-scratch regeneration
- **Expected:** Byte parity

**208: Byte-Parity Verification**
- Run notebook → compare CSV hash
- Must match 206 exactly
- **Expected:** Pass

**209: Clean Scan & Compliance Check**
- `grep -rn "oracle\|Oracle\|/Users/\|Desktop" notebook.ipynb`
- Must return ZERO results
- **Expected:** Pass

**210: Phase J Summary & Final Decision**
- **Best oracle score:** ???
- **Estimated private:** ??? (oracle - 0.011)
- **Decision:** 
  - If ≥0.935: **SUCCESS** → prepare for submission
  - If 0.925-0.934: **CLOSE** → user decides on submission
  - If <0.925: **INSUFFICIENT** → analyze failures, recommend next steps

**Phase J Expected Outcome:**
- Best case: 0.945 oracle (0.934 private) → **DOMINANT WIN**
- Realistic: 0.935 oracle (0.924 private) → **BEATS 0.92**
- Pessimistic: 0.925 oracle (0.914 private) → Close but uncertain

---

## Phase K: Experimental/Novel (Exp 211-220) — 10 experiments ⭐ NEW

**Purpose:** Unusual but potentially valuable approaches from literature  
**Expected Duration:** 6-8 hours  
**Expected Gain:** +0.003-0.012 (long shots with high upside)  
**Source:** NEW_EXPERIMENTS.md exps 246-255 + internet research

**211: Nearest-Neighbor Residual Covariance**
- Model Cov(e_i, e_j) as function of chemical similarity
- Expected: +0.003-0.008

**212: Gaussian Markov Random Field over Chemical Clusters**
- Smooth predictions within cluster while preserving supervised signal
- Expected: +0.003-0.010

**213: Local Graph-Harmonic Interpolation**
- Use only local neighborhood around test molecule (not full 6M graph)
- Avoids degeneracy problem of naïve Laplacian SSL with huge unlabeled sets
- Expected: +0.003-0.008

**214: Prototype Regression**
- Replace training samples with chemically coherent prototype centroids
- Learn prototype-to-target mappings
- Expected: +0.003-0.010

**215: Prototype Residual Model**
- Predict from global model, correct relative to nearest chemical prototype
- Expected: +0.003-0.010

**216: Conformalized Ensemble Diagnostics**
- Conformal-style residual intervals to identify unreliable regions
- Don't use intervals directly for predictions
- Expected: +0.000-0.005 (mainly diagnostic)

**217: Error-Directed Representation Search**
- Take worst 10% OOF errors, search for descriptors that discriminate them
- Error-first feature discovery, not global feature selection
- Expected: +0.005-0.012

**218: Adversarial Feature Perturbation**
- Perturb chemically meaningful descriptor groups during training
- Require prediction stability
- Expected: +0.003-0.008

**219: Representation Dropout Ensemble**
- Train models where each removes an entire feature family (no topology, no polarity, etc.)
- Creates more useful diversity than seeds
- Expected: +0.003-0.010

**220: Cross-Representation Nearest-Neighbor Agreement**
- Compare neighbors under Morgan, graph spectrum, physics, SMILES, 6M embedding
- Disagreement = learned uncertainty feature
- Expected: +0.003-0.008

---

## Phase L: Latent Property Models (Exp 221-235) — 15 experiments ⭐ NEW — HIGHEST PRIORITY

**Purpose:** Attack the problem as a partially-observed property matrix, not 7 independent regressions  
**Expected Duration:** 18-24 hours  
**Expected Gain:** +0.010-0.025 oracle  
**Kill Gate:** Must beat current grouped OOF by ~0.005 on small targets by exp 228

**Rationale:** The data is a partially-observed matrix Y_{m,t} where m=molecule, t=target. Same structures appear across targets with ~60% partner-label availability. Current system treats these as independent regressions with manual identities. Literature (AISTATS 2024, MI Research 2025) strongly supports structured multi-task with task-relation graphs.

### L1: Diagnostic (221-223)

**221: Target Covariance Analysis**
- Full Corr(y_i, y_j | X) and Corr(r_i, r_j) on multi-label subset
- Are residuals correlated after controlling for features?
- Expected: Diagnostic — determines whether L-phase is viable

**222: Latent Factor Analysis**
- PCA / factor analysis on multi-label subset
- How many latent states explain 7 targets? (hypothesis: 3-4)
- Expected: Diagnostic

**223: Task Graph Construction**
- Build empirical task graph from: (1) shared polymers, (2) correlations, (3) residual correlations, (4) physical identities
- Expected: Egc↔Ei↔Eea cluster, Nc↔Eps cluster, Tg mostly separate

### L2: Property Matrix Models (224-230)

**224: Property Matrix Factorization**
- Y_{m,t} ≈ μ_t + U_m^T V_t with U_m = f(X_m)
- Chemistry-conditioned latent molecular states + target embeddings
- Missing-label masking during training
- **Expected:** +0.008-0.020

**225: Target-Graph Message Passing**
- Learn task graph, then message-pass over tasks for each molecule
- Different from old naïve low-rank: uses nonlinear target-specific heads
- **Expected:** +0.005-0.015

**226: Chemistry State Model** ⭐ TOP PRIORITY
- SMILES → chemistry encoder → {thermal, electronic, optical, topology} states
- Each state → target-specific decoder
- Physics constraints: Ei≈Egc+Eea, eps≈nc²+ionic
- Cross-target attention between state embeddings
- **Expected:** +0.010-0.025

**227: Cross-Target Attention**
- Transformer attention between target embeddings
- Allow property interactions within a molecule's prediction
- **Expected:** +0.005-0.012

**228: Kill Gate Check**
- If L-phase hasn't improved any small target by +0.005: STOP
- If positive: continue to L3

**229: Missing-Label Pseudo-Labeling**
- Use confident model predictions as pseudo-labels
- Retrain with fuller property matrix
- Uncertainty-based label selection
- **Expected:** +0.004-0.010

**230: Physics-Constrained Matrix Model**
- Model 226 + hard constraint enforcement:
  - Ei ≈ Egc + Eea (soft penalty λ₁)
  - eps ≈ nc² + ionic (soft penalty λ₂)
  - Egb ≈ aEgc + b (soft penalty λ₃)
- **Expected:** +0.010-0.022

### L3: Transfer & Integration (231-235)

**231: Representation Transfer: Large→Small**
- Train rich z on Tg (n=4143) + Egc (n=2028)
- Use z as features for Ei/Eps/Eea/Nc/Egb (n~220-337)
- Key: transfer representation, NOT predictions (avoids circularity)
- **Expected:** +0.008-0.018

**232: Bayesian Matrix Completion**
- GP-based matrix completion with Tanimoto kernel on molecule dimension
- **Expected:** +0.006-0.015

**233: Multi-Fidelity Target Learning**
- Treat Egc (n=2028) as low-fidelity proxy for Ei (n=222)
- Learn delta model
- **Expected:** +0.005-0.012

**234: Property Matrix + 6M Embeddings**
- Model 224 but with 6M-learned embeddings as molecular representation
- Combines Phase B and Phase L
- **Expected:** +0.010-0.025

**235: Phase L Summary**
- Best latent model oracle R²
- Contribution to overall mean
- Which targets improved most?
- **Expected:** +0.010-0.020 on weak targets, +0.005-0.010 overall

**Phase L Expected Outcome:**
- Best case: +0.020 overall (structured latent captures missing information)
- Realistic: +0.012 (improves Ei/Eps significantly)
- Pessimistic: +0.005 (marginal gains, latent adds some regularization)

---

## Phase M: Residual Field Modeling (Exp 236-250) — 15 experiments ⭐ NEW

**Purpose:** Model WHERE the current best model is wrong, then correct it locally  
**Expected Duration:** 10-14 hours  
**Expected Gain:** +0.008-0.020 oracle  
**Kill Gate:** Residual autocorrelation test (exp 236) must show significant spatial structure

**Rationale:** Current system doesn't model the structure of its own errors. Residual errors likely have spatial structure in chemical space (entire chemical families may be systematically over/underpredicted). Physics-based Residual Kriging (Phy-RK) literature supports this approach.

### M1: Residual Diagnostics (236-240)

**236: Residual Autocorrelation Test** ⭐ DO THIS FIRST
- Compute V57 OOF residuals r_i = y_i - ŷ_i
- For each pair (i,j) with Tanimoto > 0.7: compute Corr(r_i, r_j)
- If near zero → skip M-phase (residuals are unstructured)
- If positive → goldmine! Local correction is justified
- **Expected:** Diagnostic (determines viability of entire phase)

**237: Catastrophe Table**
- Per target: rank OOF by squared error contribution
- Top 1% → % of total SSE, top 5% → %, top 10% → %
- If "5% of molecules = 42% of SSE" → specialist needed
- **Expected:** Diagnostic

**238: Signed Residual Neighborhoods**
- For each molecule: find 5-10 nearest neighbors, check their residual signs
- Consistent signs (all negative) → systematic local bias
- **Expected:** Diagnostic

**239: Chemical Family Residual Patterns**
- Do polyamides, polyesters, aromatics show systematic bias?
- Per-family mean residual + std
- **Expected:** Diagnostic

**240: Residual Field Visualization**
- 2D UMAP colored by signed residual per target
- Identify spatial clusters of error
- **Expected:** Diagnostic

### M2: Residual Correction (241-247)

**241: Kriging of Model Error**
- ŷ_corrected = ŷ_global + α(x)·r̂_local
- α depends smoothly on local density
- r̂_local from nearby training residuals
- **Expected:** +0.005-0.015

**242: Error-Directed Feature Discovery**
- Take worst 10% OOF errors per target
- Find descriptors that discriminate bad predictions from good
- Add these as features to next model iteration
- **Expected:** +0.005-0.012

**243: Per-Family Affine Correction**
- y_corrected = a_family · y_pred + b_family
- Learned per chemical family on OOF
- **Expected:** +0.003-0.010

**244: Residual GP Smoothing**
- Gaussian process on residuals with Tanimoto kernel
- Smooth residual field, then correct
- **Expected:** +0.005-0.012

**245: GNN as Residual Encoder**
- Small GNN → 32-dim residual representation → Ridge/ET correction
- GNN isn't learning the property — only what's missing from descriptors
- **Expected:** +0.005-0.015

**246: Multi-Representation Residual**
- Compare residual structure under Morgan vs graph-spectral vs SMILES
- Use whichever captures most spatial structure
- **Expected:** +0.003-0.008

**247: Conditional Residual by Target**
- Different residual correction models per target
- Tg residuals may need different features than Ei residuals
- **Expected:** +0.005-0.012

### M3: Integration (248-250)

**248: Cascaded Residual Correction**
- Correct → recompute residual → correct again (2-3 rounds)
- Stop when residual autocorrelation drops
- **Expected:** +0.005-0.012

**249: Residual + Uncertainty Interaction**
- Do high-variance predictions have larger systematic bias?
- Use variance × residual as routing signal
- **Expected:** +0.003-0.008

**250: Phase M Summary**
- Best residual correction strategy
- Per-target impact
- **Expected cumulative:** +0.008-0.015

**Phase M Expected Outcome:**
- Best case: +0.020 (residuals are highly structured → large correction)
- Realistic: +0.010 (moderate spatial structure → useful correction)
- Pessimistic: +0.003 (residuals mostly unstructured → minimal benefit)

---

## Phase N: Explainability & Invariance (Exp 251-270) — 20 experiments ⭐ NEW — COMPETITION REQUIREMENT

**Purpose:** Fulfill Round 3's model explainability and polymer-invariance robustness requirements  
**Expected Duration:** 8-12 hours  
**Expected Gain:** Competition judging requirement (not directly R²)  
**Note:** Round 3 **additionally requires** explainability and invariance as judged themes. This is NOT optional.

### N1: Feature Importance & Explainability (251-260)

**251: TreeSHAP Per-Target**
- SHAP feature importance for best GBM model, all 7 targets
- Summary plots + force plots for top features
- **Deliverable:** 7 SHAP summary plots

**252: SHAP Interaction Effects**
- SHAP interaction values for top 20 feature pairs per target
- Identify nonlinear feature interactions the model exploits
- **Deliverable:** Interaction heatmaps

**253: SHAP for Extreme Errors**
- Waterfall plots for 10 worst predictions per target
- WHY does the model fail on these molecules?
- **Deliverable:** Error explanation narratives

**254: Per-Family SHAP Profiles**
- Compare SHAP patterns across polyamide vs polyester vs aromatic
- Do different families use different features?
- **Deliverable:** Family-comparison SHAP plots

**255: Partial Dependence Plots**
- PDP for top 10 features per target
- **Deliverable:** 70 PDP plots (10 × 7 targets)

**256: Accumulated Local Effects (ALE)**
- ALE plots (better than PDP for correlated features)
- **Deliverable:** ALE plots for top features

**257: LIME Per-Molecule Explanations**
- Local linear explanations for 50 most interesting test predictions
- **Deliverable:** LIME explanation table

**258: Feature Importance Stability**
- Bootstrap SHAP: how stable are importance rankings across resamples?
- **Deliverable:** Stability report

**259: Physics Consistency Checks**
- Do predictions satisfy Ei≈Egc+Eea, eps≈nc²+ionic?
- Report violation counts and magnitudes
- **Deliverable:** Physics consistency report

**260: Physics Violation as Feature**
- Count of physics violations → uncertainty → ensemble weight
- Expected: +0.002-0.006 R² improvement

### N2: Invariance & Robustness (261-268)

**261: SMILES Enumeration Invariance Test**
- Generate 20 valid SMILES per molecule, compare prediction variance
- **Deliverable:** Invariance score distribution

**262: SMILES Invariance Score as Feature**
- Prediction variance across SMILES variants → uncertainty feature
- Expected: +0.003-0.008

**263: Canonical SMILES Consistency**
- Verify RDKit canonical gives identical predictions for same molecule
- **Deliverable:** Consistency verification report

**264: Invariance-Regularized Training**
- L = L_property + λ · L_invariance (same molecule, different SMILES)
- Expected: +0.003-0.010

**265: Tautomer/Resonance Invariance**
- Test prediction stability across tautomeric forms
- Expected: +0.001-0.005 (risky, can hurt)

**266: Atom Ordering Invariance (Graph)**
- Verify graph-based models are permutation invariant
- **Deliverable:** Invariance verification

**267: TTA as Invariance**
- Predict on 10 random SMILES, median aggregate
- Specifically for demonstrating invariance, not just performance
- Expected: +0.002-0.006

**268: Polymer Repeat-Unit Rearrangement**
- Iterative rearrangement of repeating units
- Test prediction stability
- **Deliverable:** Rearrangement invariance report

### N3: Reports (269-270)

**269: Explainability Full Report**
- Combine 251-260 into comprehensive explainability narrative
- Per-target chemical insight: what drives each property?
- **Deliverable:** Markdown report suitable for competition judges

**270: Invariance Full Report**
- Combine 261-268 into robustness demonstration
- Show model is invariant to SMILES representation choices
- **Deliverable:** Markdown report with quantitative evidence

**Phase N Expected Outcome:**
- Explainability report: required for competition judging
- Invariance report: required for competition judging
- Bonus R² from invariance features: +0.002-0.008

---

## Extended Phase Additions (from NEW_EXPERIMENTS.md Integration)

### Phase A Extensions (Exp A016-A030)

From NEW_EXPERIMENTS.md exps 121-135 (Dataset Analysis):

**A016: Target-Support Topology** — Which structures appear in multiple target pools? Connectivity map.
**A017: Family Entropy** — Entropy of target-type distribution per chemical family.
**A018: Within-Family Residual Correlation** — Do within-family errors correlate?
**A019: Backbone/Side-Chain Grouping** — Classify structures by backbone type.
**A020: Functional-Group Co-Occurrence Matrix** — Count co-occurrence of functional groups.
**A021: Property-Space/Chemical-Space Alignment** — Do similar molecules have similar properties?
**A022: Laplacian Spectral Descriptors** — Graph Laplacian eigenvalues as features.
**A023: Conjugated-Component Distributions** — Distribution of conjugated system sizes.
**A024: Donor/Acceptor Distance Features** — Graph distance between D/A groups.
**A025: Ratio Descriptors** — MW/TPSA, nRot/nArom, etc.
**A026: Error-Atlas-Driven Feature Discovery** — Spatial map of errors → feature search.
**A027: Representation-Specific Distance→Error Curves** — Per-representation error analysis.
**A028: HOMO Proxy from Graph Spectral** — Graph spectral as cheap HOMO-like latent.
**A029: LUMO Proxy** — Same architecture targeting electron-accepting behavior.
**A030: Donor-Strength Index / Acceptor-Strength Index** — Structural electron scores.

### Phase B Extensions (Exp B046-B065)

From NEW_EXPERIMENTS.md 6M experiments + research R031-R050:

**B046: 6M Density Field** — Chemical-space density from 6M corpus.
**B047: Density Ratio Feature** — ρ_6M(x) / ρ_labeled(x).
**B048: 6M Prototype Discovery (5k)** — Cluster 6M → 5k prototypes.
**B049: 6M Prototype Discovery (20k)** — Finer granularity.
**B050: Prototype Membership Entropy** — Entropy of soft assignment as feature.
**B051: Property-Conditioned Manifold** — Which 6M directions correlate with each target?
**B052: 6M OOD Score as Feature** — Distance to nearest 6M cluster.
**B053: 6M Neighborhood Consensus** — Do 6M neighbors agree on predicted property?
**B054: 6M-Based Training Reweighting** — Reweight by density in 6M space.
**B055: 6M Fragment Vocabulary** — Frequent subgraph patterns as features.
**B056: 6M Masked-Span MLM** — Mask contiguous SMILES spans.
**B057: 6M Graph Masked-Node** — Mask node features in molecular graph.
**B058: 6M Graph Contrastive** — Contrastive with Tanimoto similarity.
**B059: Fragment-Based Contrastive (FraSICL)** — Fragment pairs for semantic-invariant views.
**B060: 6M Descriptor Reconstruction** — Predict RDKit descriptors from embeddings.
**B061: 6M Consensus SSL** — Combine char-LM + graph-contrastive + prototype.
**B062: Atom-Token MLM (500k sample)** — Quick validation of tokenization.
**B063: Atom-Token MLM (Full 5.97M)** — Full scale.
**B064: Char-Level Language Model on 6M** — Hidden states as embeddings.
**B065: 6M Embedding + GBM Head** — Test all SSL embeddings with strong nonlinear head.

### Phase D Extensions (Exp D091-D115)

From NEW_EXPERIMENTS.md exps 136-155, 186-200:

**D091: Electronic-Property Latent Factor Model** — [Egc,Egb,Ei,Eea]^T = Λz + ε
**D092: Conditional Latent Electronic Model** — z_electronic with target-specific residual paths.
**D093: Identity-Constrained with Inequality** — Soft Ei=Egc+Eea constraint with uncertainty.
**D094: Egb Conditional on Egc + Structural Delta** — Egb = aEgc + b + g(X).
**D095: Within-Family Residual Correlation Exploitation** — Use family error patterns.
**D096: Donor × Acceptor Separation** — Graph distance D↔A as feature.
**D097: Conjugation × Donor/Acceptor Interactions** — D × L_π and A × L_π.
**D098: Conjugation Saturation Feature** — 1 - e^(-L_π/τ) with learned τ.
**D099: Heteroatom Substitution Position** — Direct/one-bond/remote encoding.
**D100: Resonance-Path Count** — Count paths connecting D/A through conjugated bonds.
**D101: Aromatic Fusion Score** — Isolated/fused/hetero-fused aromatic.
**D102: Multiple Ionic Estimators** — Ridge + ET + LightGBM + Huber → ensemble for ionic.
**D103: Ionic Uncertainty** — Uncertainty on eps - nc² as routing variable.
**D104: Ionic Heteroscedastic Model** — Predict E[ionic|X] and Var(ionic|X).
**D105: Polarizability-Density Interaction** — α/V as optical density proxy.
**D106: Polarizability × Aromaticity** — Explicit nonlinear interaction.
**D107: Polarizability × Heteroatom Type** — Separate O/N/halogen/S effects.
**D108: Polarizability Per Repeat Mass** — α/M normalized forms.
**D109: Nc Model → Eps Correction** — Nc as first-stage, Eps learns residual after nc².
**D110: Eps Model → Nc Correction** — Reverse direction.
**D111: Joint [nc, ionic] Covariance Model** — Model p(n, ε_ionic | X).
**D112: Ei Low-Dim Electronic Latent** — Only 10-20 electronic features → heavily regularized.
**D113: Eps/Nc Joint Latent Model** — Predict (nc, ε_ionic) jointly then ε = nc² + ε_ionic.
**D114: Property-Specific ARD Kernels** — Automatic relevance determination per target.
**D115: Cross-Target Residual Analysis** — ei/eea, eps/nc residual correlations.

### Phase E Extensions (Exp E126-E140)

From NEW_EXPERIMENTS.md exps 226-235 + research R086-R100:

**E126: Bicerano Group Contribution Features** — 50+ functional group counts with Tg contributions.
**E127: Torsional Entropy Proxy** — Number/diversity of low-cost torsional DOF.
**E128: Rotational-State Diversity** — Backbone/side-chain/aryl/heteroatom-adjacent rotors.
**E129: Steric Rotor Suppression** — Local steric penalty for each rotatable bond.
**E130: Backbone Rigidity Score** — Constraint by rings and unsaturation.
**E131: Side-Chain Packing Score** — Length + branching + aromaticity + heteroatom density.
**E132: Symmetry Score** — Approximate molecular symmetry measures.
**E133: Conformational Diversity** — Multiple conformers → RMSD/Rg distribution stats.
**E134: Tg High-Tail Specialist** — Explicit upper-regime model + global blend.
**E135: Tg Pairwise-Difference Model** — Tg_i - Tg_j = f(X_i - X_j), reconstruct absolute.
**E136: Tg Physics-ML Blend** — NeurIPS 5th-place: ML + physics-based expression.
**E137: Tg Distribution Shift Correction** — NeurIPS 1st-place: post-process for train/test shift.
**E138: Tg Family-Hierarchical Model** — Per-family models + ensemble.
**E139: Tg + 6M SSL Embedding Features** — Combine group contribution with 6M embeddings.
**E140: Tg Mega-Ensemble** — NNLS of top 10 Tg models.

### Phase G Extensions (Exp G176-G195)

From NEW_EXPERIMENTS.md exps 236-245 + research R071-R085:

**G176: Error-Correlation Constrained NNLS** — Max pairwise residual correlation threshold.
**G177: Minimum-Description Ensemble** — Smallest subset within 0.001 of best.
**G178: Stability-Weighted Ensemble** — Downweight high fold-variance components.
**G179: Regime-Specific Model Weights** — f(density, OOD, uncertainty) low-capacity meta.
**G180: Leave-One-Family-Out Selection** — Family-held-out OOF for ensemble choice.
**G181: Test-Geometry Weighted Ensemble** — Weights by test similarity distribution.
**G182: Physics-Aware Ensemble Scoring** — Score = R² - λ·PhysicsViolation.
**G183: Conformal Prediction Intervals** — Conformalized molecular regression.
**G184: Distribution Shift Calibration** — Post-process for train/test shift.
**G185: Per-Category Calibration** — Separate for low-sim vs high-sim regions.
**G186: Fold-Consensus Weight Stability** — Keep only stable cross-fold weights.
**G187: Bootstrap Weight Stability** — Bootstrap blend fitting, discard unstable.
**G188: Shrinkage Accumulator** — Accumulate +0.003 components instead of rejecting.
**G189: Near-Miss Accumulator** — Revisit Round 2 positive-but-below-gate components.
**G190: Late Fusion Multimodal Ensemble** — Separate SMILES + graph + physics ensembles, late-fuse.
**G191-G195: Ensemble integration of G176-G190 winners.**

---

## Updated Execution Strategy

### Priority-Based Execution Order (NOT sequential)

**Phase 1: Diagnostics (Day 1, first 6 hours)**
1. Run Phase A foundation (exp 001-015)
2. Run diagnostic/ EDA scripts
3. Run L-phase diagnostics (exp 221-223)
4. Run M-phase residual diagnostic (exp 236-240)
5. Review all diagnostics → adjust phase priorities

**Phase 2: High-Priority Experiments (Day 1-2)**
6. Phase L latent models (exp 224-235) — HIGHEST PRIORITY
7. Phase B SSL at scale (exp 035-065) — GPU-intensive, run overnight
8. Phase M residual correction (exp 241-250) — if diagnostics positive

**Phase 3: Target-Specific (Day 2-3)**
9. Phase E Tg specialist (exp 100-140)
10. Phase F weak targets (exp 126-155 + extensions)
11. Phase D multi-task physics (exp 076-115)

**Phase 4: Ensemble & Polish (Day 3-4)**
12. Phase G ensemble + calibration (exp 156-195)
13. Phase N explainability + invariance (exp 251-270)
14. Phase J integration + final assembly (exp 196-210)

**Phase 5: Long Shots (if time permits)**
15. Phase C GNN (conditional on kill gate)
16. Phase K experimental/novel
17. Phase H TTA
18. Phase I validation

### Parallel Execution Strategy

**GPU Laptop (1 job at a time):**
- Phase B SSL (long transformer/GNN training)
- Phase C GNN experiments
- Phase L latent models (if neural)

**Mac (unlimited parallel CPU jobs):**
- Phase A EDA
- Phase D physics features
- Phase E/F small-target models (GPR, Ridge, KRR)
- Phase G ensemble weight optimization
- Phase M residual analysis
- Phase N SHAP/LIME computations

## Execution Strategy

### Parallel vs Sequential

**Sequential (Safe):**
- Run experiments 001→210 in order
- Each informs the next
- Kill gates work properly
- **Timeline:** 3-4 days continuous

**Parallel (Faster but Riskier):**
- Run independent phases in parallel
- E.g., B + D in parallel (both use baseline)
- **Timeline:** 1.5-2 days
- **Risk:** May waste compute on wrong directions

**Recommended:** Hybrid
- A sequential (foundation)
- B/D parallel (both critical, independent)
- C conditional (depends on B results)
- E/F parallel (both target-specific)
- G/H/I sequential (depend on E/F)
- J sequential (integration)

### Resource Allocation

**GPU Laptop:**
- Phases B (SSL), C (GNN): GPU-intensive
- Run overnight for long experiments (B035, B037, C-phase)

**Mac:**
- All scoring, analysis, plotting
- Short CPU experiments (SVD, Ridge, KRR)

### Time Budget

| Phase | Wall Time | Parallel? |
|-------|-----------|-----------|
| A | 6h | No |
| B | 30h | Partially (CPU vs GPU) |
| C | 18h | No (conditional) |
| D | 12h | Yes (with B) |
| E | 18h | Yes (with F) |
| F | 15h | Yes (with E) |
| G | 10h | No |
| H | 5h | Yes (with G) |
| I | 5h | No |
| J | 10h | No |
| **Total** | **129h** | → **~72h with parallelization** |

### Checkpointing

After each phase:
1. Commit best result to git
2. Update `logs/phase5_summary.tsv`
3. Write phase summary report
4. Decision: continue, adjust, or escalate

---

## Risk Mitigation

### If SSL (Phase B) Fails

**Symptoms:** All SSL experiments ≤ baseline  
**Action:**
- Analyze: tokenization wrong? GBM vs linear probe?
- Try quick fixes (B041-B045)
- If still fails: abort B, rely on D/E/F

**Backup Plan:** Phases D/E/F alone can reach 0.925-0.930

### If Multi-Task (Phase D) Fails

**Symptoms:** Joint models hurt all targets  
**Action:**
- Fall back to single-task per target
- Keep GPR for ei (should work standalone)
- Continue with E/F

### If Tg Specialist (Phase E) Fails

**Symptoms:** Tg stuck at 0.900-0.905  
**Action:**
- This is highest-leverage target → CRITICAL
- Deep dive: why not improving?
- Try more radical approaches (deep GNN, heavy TTA)
- May need to accept lower final score

### If All Phases Underdeliver

**Symptoms:** Best after J-phase <0.925  
**Action:**
- Analyze which approaches showed ANY promise
- Double down on best-performing direction
- Consider ensemble of top-10 experiments
- May need to submit conservative V57 + aggressive best

---

## Success Metrics Summary

| Milestone | Target Oracle | Status |
|-----------|--------------|--------|
| Phase A complete | 0.9024 | Baseline reproduced |
| Phase B best | 0.9100 | SSL shows promise |
| Phase D best | 0.9150 | Multi-task helps ei/eps |
| Phase E best | 0.9200 | Tg improves significantly |
| Phase F best | 0.9250 | Weak targets fixed |
| Phase J integrated | **0.9350+** | **SUCCESS** |

### Per-Target Milestones

| Target | Current | Phase B | Phase D/F | Phase E | Phase J | Gap Closed |
|--------|---------|---------|-----------|---------|---------|------------|
| tg | 0.8945 | 0.900 | 0.905 | **0.920** | 0.920 | **75%** |
| ei | 0.8708 | 0.875 | **0.900** | 0.900 | 0.905 | **100%** |
| eps | 0.8881 | 0.895 | **0.915** | 0.915 | 0.915 | **100%** |
| egc | 0.9091 | **0.920** | 0.920 | 0.920 | 0.925 | **100%** |
| nc | 0.9088 | 0.915 | **0.920** | 0.920 | 0.920 | **100%** |
| eea | 0.9150 | 0.920 | **0.930** | 0.930 | 0.930 | **100%** |
| egb | 0.9305 | 0.935 | 0.935 | 0.935 | **0.938** | **100%** |
| **MEAN** | **0.9024** | **0.9086** | **0.9179** | **0.9214** | **0.9347** | **✓** |

---

## References & Literature

**SSL & Transformers:**
- polyBERT (Kuenneth 2023): Atom tokenization, DeBERTa, R²=0.80 on 29 properties
- ChemBERTa (Chithrananda 2020): SMILES masked LM
- MolBERT (Wang 2019): BERT for molecules

**Multi-Task:**
- Kuenneth 2021 (Patterns): Multi-task beats single-task on sparse targets
- Soft physics constraints as auxiliary losses

**GNN:**
- D-MPNN (Yang 2019): Message-passing with edge features
- GATv2 (Brody 2021): Improved attention
- GIN (Xu 2019): Graph isomorphism network

**Gaussian Processes:**
- Rasmussen & Williams (2006): GP for regression
- Tanimoto kernel for molecules (Ralaivola 2005)

**Polymer Property Prediction:**
- Polymer Genome (Kim 2018): Handcrafted features
- PolyMetriX (Huan 2015): Hierarchical features
- Bicerano (2002): Group contribution for Tg

---

## Updated Success Metrics

| Milestone | Target Oracle | Key Phase |
|-----------|--------------|----------|
| Diagnostic complete | N/A | A + diagnostic/ |
| Phase L best | 0.9150 | Latent model shows promise |
| Phase B best | 0.9100 | SSL at scale works |
| Phase M best | 0.9120 | Residual correction works |
| Phase E best | 0.9200 | Tg improves significantly |
| Phase F best | 0.9250 | Weak targets fixed |
| Phase J integrated | **0.9350+** | **SUCCESS** |

### Per-Target Updated Milestones

| Target | Current | After L | After B | After E | After F | After J | Gap Closed |
|--------|---------|---------|---------|---------|---------|---------|------------|
| tg | 0.8945 | 0.900 | 0.905 | **0.920** | 0.920 | 0.920 | **75%** |
| ei | 0.8708 | **0.890** | 0.895 | 0.895 | **0.905** | 0.905 | **100%** |
| eps | 0.8881 | **0.905** | 0.910 | 0.910 | **0.915** | 0.915 | **100%** |
| egc | 0.9091 | 0.915 | **0.920** | 0.920 | 0.920 | 0.925 | **100%** |
| nc | 0.9088 | **0.915** | 0.918 | 0.918 | **0.920** | 0.920 | **100%** |
| eea | 0.9150 | **0.925** | 0.925 | 0.925 | **0.930** | 0.930 | **100%** |
| egb | 0.9305 | 0.933 | 0.935 | 0.935 | 0.935 | **0.938** | **100%** |
| **MEAN** | **0.9024** | **0.9119** | **0.9154** | **0.9190** | **0.9207** | **0.9347** | **✓** |

---

**Plan Version:** 2.0  
**Created:** 2026-08-30  
**Updated:** 2026-08-30 (internet research + NEW_EXPERIMENTS.md integration)  
**Total Experiments:** ~565 (210 base + 100 research + 255 NEW_EXPERIMENTS)  
**Expected Timeline:** 72-96 hours with priority-based parallel execution  
**Target:** 0.935 oracle → 0.924 private → **BEAT 0.92 COMPETITOR**  
**Status:** READY TO EXECUTE  
**Key Change:** Priority order is now L > B > M > E > F > D > N > G > C > K > H > I  
**Key Policy:** Shrinkage accumulator — don't reject +0.003 components
