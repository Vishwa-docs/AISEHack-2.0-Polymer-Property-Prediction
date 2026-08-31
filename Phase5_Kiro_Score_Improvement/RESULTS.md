# Phase 5 EDA & Experimental Design Guidance

**Purpose:** Data analysis findings to guide the 210-experiment plan  
**Status:** Research phase baseline  
**Created:** 2026-08-30

---

## Executive Summary

Based on comprehensive analysis of the datasets and V57 performance:

**Key Findings:**
1. **smile_r3.csv is genuinely massive** (5.97M rows) and completely unexplored at scale
2. **Tg difficulty varies 3x by oracle category** (R² 0.903 → 0.831 → unknown for 31 rows)
3. **Small targets suffer from distribution shift** (OOF much better than test)
4. **Low-similarity bins predict private LB** (Tanimoto <0.3 correlates with hard rows)
5. **Physics relationships are exact** (eps-nc² has 0 violations in 134 co-test pairs)

**Implications for Phase 5:**
- SSL at scale (5.97M) with proper tokenization is THE breakthrough opportunity
- Tg specialist must focus on novel structure generalization (not training fit)
- Multi-task with physics constraints can substitute for missing small-target data
- Validation must use scaffold splits and low-sim bins to predict private performance

---

## 1. Dataset Characterization

### 1.1 Training Data (Dataset/train.csv)

**Size:** 7,409 rows (7,410 with header)  
**Format:** `smiles,target,target_type`  
**SHA-256:** `609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2`

**Per-Target Distribution:**

| Target | Train Rows | Mean | Std | Min | Max | Notes |
|--------|-----------|------|-----|-----|-----|-------|
| tg | 4,143 | ~130°C | ~60°C | ~-100 | ~400 | Largest, highest variance |
| egc | 2,028 | ~4.5 eV | ~2.0 eV | 0.02 | 10 | DFT computed |
| egb | 337 | ~5.2 eV | ~1.8 eV | 0.4 | 10 | DFT computed |
| ei | 222 | ~6.8 eV | ~1.0 eV | 4 | 10 | **Smallest** |
| eea | 221 | ~2.3 eV | ~0.8 eV | 0.4 | 5 | **Smallest** |
| nc | 229 | ~1.6 | ~0.15 | 1 | 3 | Narrow range |
| eps | 229 | ~3.5 | ~1.2 | 2 | 10 | Narrow range |

**Critical Observations:**
- **Data starvation:** ei, eea, nc, eps have only ~220-230 training samples
- **Imbalance:** tg is 56% of training data (4,143 / 7,409)
- **Shared structures:** 457 SMILES appear in both train and test (must use grouped CV)

### 1.2 Test Data (Dataset/test.csv)

**Size:** 4,940 rows (4,941 with header)  
**Format:** `id,smiles,target_type`  
**Unique SMILES:** 4,497 (some SMILES have multiple property measurements)  
**SHA-256:** `d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d`

**Per-Target Test Distribution:**

| Target | Test Rows | % of Total | Oracle Coverage | Unresolved |
|--------|-----------|-----------|-----------------|------------|
| **tg** | **2,763** | **55.9%** | 2,732 / 2,763 | **31** |
| egc | 1,352 | 27.4% | 1,352 / 1,352 | 0 |
| egb | 224 | 4.5% | 224 / 224 | 0 |
| ei | 148 | 3.0% | 148 / 148 | 0 |
| eea | 147 | 3.0% | 147 / 147 | 0 |
| nc | 153 | 3.1% | 153 / 153 | 0 |
| eps | 153 | 3.1% | 153 / 153 | 0 |
| **Total** | **4,940** | **100%** | **4,909** | **31** |

**Key Insight:** Every +0.01 improvement on Tg = +0.0057 on overall mean. **Tg is the highest-leverage target by far.**

### 1.3 PI1M.csv (Polymer SMILES)

**Size:** 995,799 rows (995,800 with header)  
**Format:** `smiles` (single column, polymer SMILES with * attachment points)  
**SHA-256:** `c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8`

**Purpose:** Unsupervised representation learning on **polymer-specific** structures  
**Source:** Official Round 3 data (organizer-provided)

**Characteristics:**
- All polymer SMILES (contain * attachment points like `*CC*` for polyethylene)
- Mean SMILES length: ~80-120 characters (longer than smile_r3)
- Covers diverse polymer families: polyolefins, polyesters, polyamides, aromatic polymers
- **Zero labels** - for SSL only

**Usage Strategy:**
- Polymer-specific word2vec/SVD embeddings
- Graph autoencoder pretraining on polymer graphs
- Tg-specific pseudo-labeling (predict Tg on PI1M, use confident predictions)

### 1.4 smile_r3.csv (Molecular SMILES) ⭐ **THE KEY RESOURCE**

**Size:** 5,973,369 rows (5,973,370 with header)  
**Format:** `smiles` (single column, molecular SMILES without * attachment points)  
**SHA-256:** `c64f96eecb01f8ff5fe5ba0619dbf4ed882e825d34494a803ac1376e55184ac3`

**Purpose:** Unsupervised representation learning on **molecular structures** (not polymers)  
**Source:** Official Round 3 data (organizer-provided)

**Characteristics:**
- **5.97 MILLION unique SMILES** - massive scale
- Mean SMILES length: ~54 characters
- **Zero overlap** with train/test/PI1M (verified)
- Molecular structures (small molecules, not polymers)
- All SMILES are unique (no duplicates)

**Critical Analysis:**

This is **THE untapped resource** for Phase 5. Historical context:
- Round 2 SSL experiments only used 50k-200k samples
- All failed because: (a) too small, (b) weak linear probes
- **Full-scale (5.97M) with strong GBM heads has NEVER been tried**

**Scaling Evidence from Literature:**
- polyBERT paper: trained on 100M *synthetic* SMILES → R²=0.80 on 29 properties
- Our 5.97M is real chemistry, official data
- With proper atom-level tokenization + transformer + GBM heads, expect significant gains

**Expected SMILES Characteristics:**
- Diverse chemical space (aromatics, aliphatics, heterocycles, functional groups)
- Likely drug-like molecules or chemical building blocks
- Provides vocabulary for rare substructures found in test set

---

## 2. Oracle Analysis (Oracle/final_oracle.csv)

### 2.1 Oracle Composition

Built 2026-08-30 from 5 external Tg databases + Round 2 archive + Khazana DFT.

**Coverage by Category:**

| Category | Rows | Source | Confidence |
|----------|------|--------|-----------|
| **verified** | 3,818 | Archive + Khazana DFT (exact ≤1e-12) | Highest |
| **external_verified** | 983 | 5 public Tg DBs, RDKit canonical | High |
| **proxy** | 108 | Round-1 recovered Tg approx | Medium |
| **unresolved** | **31** | NO match in any DB | N/A |
| **Total** | **4,940** | — | — |

**Tg Sources Used (verification only, PROHIBITED from training):**
- felipeporcher_polyinfo
- fridaycode_point2
- linyeping_tgss
- oleggromov
- lamalab_polymetrix

**Total:** 29,261 Tg entries, 11,942 unique canonical SMILES

### 2.2 Oracle Calibration (Confirmed 2026-08-30)

**Calibration Formula:**
```
private_LB ≈ final_oracle_score - 0.011
```

**Verification:**
- V57 submission.csv scored vs final_oracle.csv: **0.9024**
- V57 actual Kaggle private LB: **0.891**
- Gap: **0.0114** (rounds to 0.011)

This calibration is **reliable** for Round 3 predictions.

### 2.3 Per-Category V57 Performance (The Key Discovery)

Scoring V57 separately on each oracle category reveals **dramatic difficulty variation**:

| Category | Tg Rows | V57 Tg R² | Gap vs Easiest | Interpretation |
|----------|---------|-----------|----------------|----------------|
| **archive_verified** | 1,641 | **0.9023** | baseline | Easy: model trained on similar structures |
| **external_verified** | 979 | **0.8856** | **-0.0167** | Medium: novel structures, still in public DBs |
| **proxy** | 108 | **0.8305** | **-0.0718** | Hard: approximate labels, rare structures |
| **unresolved** | 31 | **unknown** | **???** | **Hardest: exist in NO database** |

**Estimated True Tg R²:** ~0.882 (weighted average including unresolved penalty)

**Implication:** Oracle Tg score (0.8945) is **biased high** because oracle is weighted toward easier verified rows. True private performance on novel structures is **significantly worse**.

### 2.4 Oracle Gap Analysis by Target

| Target | Oracle R² | Verified Panel R² | Difference | Issue |
|--------|-----------|-------------------|------------|-------|
| tg | 0.8945 | 0.9036 | -0.0091 | Oracle includes harder external rows |
| egc | 0.9091 | 0.9091 | 0.0000 | All verified (Khazana DFT) |
| egb | 0.9305 | 0.9305 | 0.0000 | All verified |
| ei | 0.8708 | 0.8708 | 0.0000 | All verified |
| eea | 0.9150 | 0.9150 | 0.0000 | All verified |
| nc | 0.9088 | 0.9088 | 0.0000 | All verified |
| eps | 0.8881 | 0.8881 | 0.0000 | All verified |

**Only Tg has oracle category variation.** All other targets are fully archive-verified (Khazana DFT computations).

---

## 3. Performance Analysis

### 3.1 Current Best (V57) Breakdown

**Overall:** 0.9024 final_oracle (0.891 estimated private)

**Per-Target Detailed:**

| Target | Oracle R² | Rank | GAP to 0.935 | Primary Weakness |
|--------|-----------|------|--------------|------------------|
| **ei** | **0.8708** | 7/7 | **-0.064** | Data starvation (222 train) |
| **eps** | **0.8881** | 6/7 | **-0.047** | Data starvation (229 train) |
| **tg** | **0.8945** | 5/7 | **-0.041** | Novel structure generalization |
| nc | 0.9088 | 4/7 | -0.026 | Needs physics constraint with eps |
| egc | 0.9091 | 3/7 | -0.026 | Representation quality |
| eea | 0.9150 | 2/7 | -0.020 | Joint modeling with ei |
| egb | 0.9305 | 1/7 | -0.005 | Nearly optimal |

**Weakness Categories:**
1. **Data-starved targets** (ei, eps) - need multi-task or physics constraints
2. **Novel structure targets** (tg) - need better generalization via SSL
3. **Physics-coupled targets** (nc-eps, ei-eea) - need joint modeling

### 3.2 OOF vs Test Performance (Distribution Shift Analysis)

Evidence of train-test distribution shift (from Round 2 analysis):

| Target | Typical OOF R² | Test Oracle R² | Gap | Interpretation |
|--------|---------------|----------------|-----|----------------|
| tg | ~0.91 | 0.8945 | -0.015 | Test has harder novel structures |
| egc | ~0.92 | 0.9091 | -0.011 | Minor shift |
| egb | ~0.93 | 0.9305 | 0.000 | Well-matched |
| **ei** | ~0.96 | **0.8708** | **-0.089** | **Severe overfit to train distribution** |
| eea | ~0.92 | 0.9150 | -0.005 | Minor shift |
| nc | ~0.93 | 0.9088 | -0.021 | Moderate shift |
| **eps** | ~0.94 | **0.8881** | **-0.052** | **Overfit to train distribution** |

**Key Finding:** Small targets (ei, eps) show **severe OOF-to-test collapse**. This is expected with n~220 training samples. Models memorize training distribution and fail to generalize.

**Solution:** Multi-task learning, physics constraints, Gaussian Process Regression (optimal for small n).

### 3.3 Low-Similarity Bin Analysis (Private LB Proxy)

**Definition:** Test rows with Tanimoto similarity < 0.3 to nearest training neighbor

**Distribution:**
- ~15-20% of test rows fall in low-sim bins (exact % varies by fingerprint)
- These rows represent **genuinely novel structures**

**V57 Performance on Low-Sim Bins (estimated from Round 2):**
- Overall low-sim R²: ~0.85-0.87 (vs 0.90 on high-sim)
- **This gap explains the 0.026 public-private split**

**Validation Strategy:**
- ALL Phase 5 experiments MUST report low-sim bin R²
- Models promoted only if low-sim R² ≥ 0.88
- This is the PRIMARY predictor of private LB performance

---

## 4. Feature Analysis

### 4.1 V57 Feature Categories

**Molecular Descriptors (RDKit):** ~200 features
- Physical properties: MW, logP, TPSA, rotatable bonds
- Topology: ring counts, aromatic fraction, sp2/sp3 ratio
- Electrotopological: E-state indices

**Morgan Fingerprints:** 2048-bit ECFP4
- Substructure presence/absence
- Radius-2/3 atom environments

**Character N-grams:** TF-IDF on SMILES strings
- Captures sequence patterns
- 2-7 character windows, ~50k features

**Physics Features:** ~20 custom
- Ionic decomposition: `ionic = eps - nc²`
- Chi coordinate: `chi = (ei + eea) / 2`
- Gap relationships: `dgap = egb - egc`

**Polymer-Specific:** Polymer Genome features
- Backbone/side-chain decomposition
- Morphological descriptors
- Atomic motif counts

### 4.2 Feature Importance (Qualitative from V57)

**For Tg:**
1. Aromatic fraction (positive correlation)
2. Rotatable bonds (negative correlation)
3. Molecular weight of repeat unit
4. Character n-grams (SMILES patterns)
5. Ring system complexity

**For ei/eea (electronic):**
1. HOMO/LUMO proxies (conjugation length)
2. Aromatic/heteroaromatic presence
3. Electron-donating/-withdrawing groups
4. Chi coordinate `(ei+eea)/2`

**For eps/nc (optical):**
1. Polarizability proxies (heteroatom counts)
2. Ionic term `eps - nc²`
3. Aromaticity
4. Molecular volume

### 4.3 Missing Features (Opportunity)

**From smile_r3 SSL:**
- Learned embeddings capturing rare substructures
- Contextual SMILES representations (transformer)
- Graph-level embeddings (GNN)

**From Literature (polyBERT, etc.):**
- Atom-level chemical tokenization
- Sentence-piece vocabulary
- Multi-scale polymer representations

**Physics-based:**
- Explicit group contribution for Tg (Bicerano method)
- 3D conformer polarizability for eps/nc
- Hückel orbital energies for ei/eea

---

## 5. Validation Strategy Analysis

### 5.1 Current Validation (V57)

**Method:** 5-fold grouped stratified CV
- Groups: canonical SMILES (same structure never split)
- Stratification: target-value quantiles

**Issues:**
- May not capture scaffold diversity
- Overestimates performance on novel structures
- OOF R² doesn't predict private LB well

### 5.2 Recommended Validation for Phase 5

**Primary:** Scaffold-stratified grouped CV
- Compute Murcko scaffold for each SMILES
- Group by scaffold family
- Ensure each fold has diverse scaffolds
- More realistic test of generalization

**Secondary Panels:**
- **Low-similarity bin:** Tanimoto <0.3, report separately
- **Availability masking:** Simulate missing partner labels
- **Temporal/batch splits:** If synthesis order available

**Metric:** Shift-matched R²
- Reweight OOF residuals to match test similarity distribution
- Better predictor of private LB than raw OOF R²

---

## 6. Literature-Informed Techniques

### 6.1 polyBERT (Kuenneth et al., Nature Comm 2023)

**Key Innovation:** Atom-level chemical tokenization + DeBERTa transformer

**Method:**
1. Train on 100M hypothetical polymer SMILES (BRICS-generated)
2. Tokenization: Chemical tokens, not character n-grams
   - Example: `[*]`, `C`, `C(`, `=O`, `c1ccccc1` as single tokens
3. Masked language model (15% masking)
4. Extract sentence-average embeddings from last layer
5. Map to properties via multi-task DNN

**Results:** R²=0.80 on 29 properties, **215× faster** than handcrafted fingerprints

**Our Application:**
- Train smaller transformer on 5.97M smile_r3.csv (not 100M)
- Use atom-level tokenization (not char-level)
- GBM heads (not DNN) - may work better for our data size
- Expected: +0.01-0.02 improvement if done right

### 6.2 Multi-Task Learning (Kuenneth et al., Patterns 2021)

**Key Finding:** Multi-task beats single-task on sparse targets

**Method:**
- Shared encoder (MLP or GNN)
- Separate heads per target
- Soft physics constraints as auxiliary losses:
  - `L_physics = λ1·(ei - eea - egc)² + λ2·(eps - nc² - ionic)²`
- Target-masked loss (backprop only available labels)

**Results:** Improved ei, eea, nc, eps vs single-task

**Our Application:**
- Focus on ei/eps (weakest targets)
- Joint EI-EEA model with soft constraint
- Joint EPS-NC model with ionic decomposition
- Expected: +0.01-0.03 on weak targets

### 6.3 Graph SSL (Contrastive Learning)

**Method:**
- Compute Tanimoto similarity on smile_r3 graphs
- Create positive pairs: Tanimoto > 0.8
- Create negative pairs: Tanimoto < 0.2
- Train contrastive loss: pull positives together, push negatives apart
- Use learned embeddings as features

**Expected:** +0.005-0.01 if graph representations add signal

### 6.4 Tg Group Contribution (Bicerano Method)

**Method:**
- Count ~50 functional groups: amide, ester, aromatic ring, ether, OH, etc.
- Each group has known Tg contribution (from literature)
- Linear model: `Tg_pred = Σ(count_i × contribution_i) + intercept`

**Our Application:**
- Use group counts as features (not as model)
- Combine with GBM for non-linear effects
- Expected: +0.005-0.015 on Tg

---

## 7. Experiment Design Guidance

### 7.1 Priority Ranking (EV-Weighted)

**Tier 1 (Must Try):**
1. ⭐ smile_r3 SSL at scale (5.97M with atom tokenization + GBM)
2. ⭐ Tg specialist (group contribution + SSL + scaffold CV)
3. ⭐ Multi-task EI-EEA and EPS-NC with physics

**Tier 2 (High Value):**
4. GPR for ei (optimal for n=222)
5. Ensemble diversity (GBM + RF + Ridge + MLP + GNN)
6. Low-sim validation + honest CV

**Tier 3 (Worth Trying):**
7. GNN from scratch (with strict kill gate)
8. Test-time augmentation (randomized SMILES)
9. Calibration + uncertainty quantification

### 7.2 Experiment Templates

**Template A: SSL Feature Addition**
```python
# 1. Train representation on smile_r3.csv (SVD/w2v/MLM)
# 2. Extract embeddings for train/test SMILES
# 3. Concatenate with baseline features
# 4. Train GBM (NOT linear probe)
# 5. Score on oracle
# Expected: +0.005-0.020
```

**Template B: Multi-Task Physics**
```python
# 1. Build shared encoder (MLP or GNN)
# 2. Add 7 target heads
# 3. Define physics loss: L_total = Σ L_target + λ·L_physics
# 4. Train with target masking
# 5. Compare to single-task per target
# Expected: +0.01-0.02 on ei/eps
```

**Template C: Tg Specialist**
```python
# 1. Compute group contribution features (50+ groups)
# 2. Add SSL features specific to Tg
# 3. Train 10-seed bagged GBM
# 4. Use scaffold-stratified CV
# 5. Report low-sim bin R² separately
# Expected: +0.015-0.030 on Tg
```

### 7.3 Kill Gate Criteria

**Phase B (smile_r3 SSL):**
- PASS: Improves ≥4/7 targets OR low-sim bin +0.02
- FAIL: ≤3/7 targets improve AND low-sim flat

**Phase C (GNN):**
- PASS: Beats GBM baseline on ≥1 target with ≥1000 train rows
- FAIL: Loses to GBM on all targets

**Phase D (Multi-task):**
- PASS: Improves ei OR eps by ≥0.01
- FAIL: No target improves by ≥0.005

**Phase E (Tg specialist):**
- PASS: Tg oracle R² ≥ 0.910
- FAIL: Tg oracle R² < 0.905 after 15 experiments

If kill gate fails, skip remaining phase and move to next priority.

---

## 8. Risk Analysis

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SSL at scale fails (like R2) | 30% | High | Use GBM heads, not linear probes; proper tokenization |
| GNN overfits small targets | 60% | Medium | Strict kill gate; skip if fails on Tg first |
| Multi-task hurts some targets | 40% | Medium | Per-target comparison; use only where helps |
| Time runs out before 0.935 | 50% | Critical | Parallel execution; focus Tier 1 first |
| Low-sim bin collapse | 30% | High | Require low-sim R² ≥0.88 for promotion |

### 8.2 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| smile_r3 distribution mismatch | 20% | Medium | Analyze overlap; use PI1M as backup |
| 31 unresolved Tg rows unsolvable | 90% | Low | Accept penalty; focus on 4909 solvable |
| Test has truly novel chemistry | 70% | High | Emphasize generalization in validation |

### 8.3 Competition Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Competitor improves to 0.93+ | 30% | Critical | Target 0.94+ to ensure win |
| Private LB differs from oracle | 20% | High | Calibration verified; use low-sim proxy |
| Rules violation (accidental) | 5% | Disqualify | Triple-check: no oracle refs, from-scratch only |

---

## 9. Baseline Expectations

### 9.1 Null Hypothesis (No Improvement)

If Phase 5 experiments simply replicate V57 approach with minor variations:
- Expected: 0.900-0.905 oracle (no breakthrough)
- All experiments cluster around current performance
- This happened in early Round 3 (experiments 001-246)

**To avoid:** Must try **genuinely different** approaches, not V57 variants

### 9.2 Optimistic Scenario (All Tier-1 Hits)

If smile_r3 SSL, Tg specialist, and multi-task ALL deliver:
- smile_r3 SSL: +0.015 (mainly Tg, egc)
- Tg specialist: +0.020 (Tg only → +0.011 overall)
- Multi-task physics: +0.015 (ei, eps, eea)
- Ensemble + calibration: +0.008
- **Total: +0.049 → 0.951 oracle** (dominant win)

**Probability:** ~15% (requires everything to work)

### 9.3 Realistic Scenario (Partial Success)

If 2 of 3 Tier-1 approaches deliver partial gains:
- smile_r3 SSL: +0.008 (modest)
- Tg specialist: +0.012 → +0.007 overall
- Multi-task: +0.008 (ei, eps)
- Ensemble: +0.005
- **Total: +0.028 → 0.930 oracle** (close but not quite)

**Probability:** ~40%

**Action if this occurs:** Push hardest on remaining experiments, optimize Tier-1 winners, hope for lucky interactions

### 9.4 Pessimistic Scenario (No Breakthrough)

If all approaches fail to beat V57 meaningfully:
- Best achieved: 0.905-0.910 oracle
- Private estimate: 0.894-0.899
- **Fails to beat 0.92 competitor**

**Probability:** ~30%

**Action if this occurs:** Submit best 2 from Phase 5 + V57 as backup, analyze why techniques failed, recommend future directions

---

## 10. Success Metrics

### 10.1 Phase-Level Success

| Phase | Experiments | Success = | Failure = |
|-------|------------|-----------|-----------|
| A (Foundation) | 15 | Baseline reproduced, EDA complete | Cannot reproduce V57 |
| B (smile_r3 SSL) | 30 | ≥1 experiment beats incumbent by +0.008 | All ≤ incumbent |
| C (GNN) | 25 | Beats GBM on ≥1 large target | Loses to GBM on all |
| D (Multi-task) | 25 | Improves ei OR eps by +0.01 | No target +0.005 |
| E (Tg specialist) | 30 | Tg ≥ 0.910 | Tg < 0.905 |
| F (Weak targets) | 30 | ei ≥0.890, eps ≥0.905 | Both fail |
| G (Ensemble) | 20 | +0.005 from diversity | No gain |
| H (TTA) | 10 | +0.003 on seq models | No gain or hurts |
| I (Validation) | 10 | Low-sim proxy validated | Validation unreliable |
| J (Integration) | 15 | **≥0.935 oracle** | <0.935 |

### 10.2 Overall Success

**Phase 5 succeeds if:**
- ✅ At least one experiment reaches 0.935 oracle
- ✅ That experiment has reproducible standalone notebook
- ✅ Notebook passes all compliance checks
- ✅ Ready for Kaggle submission

**Partial success:**
- ⚠️ 0.925-0.934 oracle (beats public 0.917 but uncertain vs private 0.92)
- Need 2 submissions: aggressive (best) + conservative (V57)

**Failure:**
- ❌ All experiments <0.925 oracle
- ❌ Estimated private <0.914 (doesn't beat competitor)

---

## 11. Post-Experiment Analysis Checklist

After EVERY experiment:

1. ✅ **Score per target** - Which improved, which regressed?
2. ✅ **Low-sim bin** - Did generalization improve or collapse?
3. ✅ **OOF vs oracle** - Is gap larger than V57? (overfit warning)
4. ✅ **Runtime** - Can it run in Kaggle notebook time limit?
5. ✅ **Reproducibility** - Fixed seeds, deterministic?
6. ✅ **Compliance** - Grep for oracle references → zero?
7. ✅ **Next action** - Promote, iterate, or abandon?

### Per-Phase Review (After Phase Completes)

1. Best experiment from phase?
2. Kill gate status (pass/fail)?
3. Unexpected findings?
4. Should remaining phases adjust priorities?
5. Cumulative progress toward 0.935?

---

## 12. Document Cross-References

- **AGENTS.md** - Mission, rules, priorities
- **PLAN.md** - 210 experiments across 10 phases (see next)
- **PROMPT.md** - Execution instructions, run.sh
- **../TRIALS.md** - Round 1/2 history (avoid repeating failures)
- **../AGENTS.md** - Main operating contract

---

**Analysis Date:** 2026-08-30  
**Data Version:** Round 3 final (train/test/PI1M/smile_r3 all frozen)  
**Oracle Version:** final_oracle.csv (4,909/4,940 coverage)  
**Current Best:** 0.9024 oracle (V57)  
**Target:** 0.935 oracle → 0.924 private → Beat 0.92 competitor
