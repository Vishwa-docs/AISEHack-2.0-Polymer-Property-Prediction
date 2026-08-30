# Round 3 Priority Action Plan
### To beat private LB 0.92 — target oracle >= 0.935

Current state (2026-08-30):
- Best V57 oracle: **0.9035** → estimated private **0.891**
- Best R3 experiment oracle: **0.9028** (below V57)
- Gap to close: **+0.0315 oracle points = ~+0.031 private points**

---

## Calibration Formula

```
private_LB ≈ oracle_verified − 0.013
```

To beat 0.92 private: oracle_verified ≥ 0.933 (conservative: 0.935)

---

## Priority 1 🔴 — Better Tg Model
**Expected gain: +0.010–0.020 oracle (= +0.010–0.020 private)**

### Why it's #1
- Tg has 2,763 test rows = **55.9% of the entire test set**
- Current Tg verified R² = 0.9023 (measured on easy archive-matched rows only)
- True Tg R² on all rows likely ~0.87–0.89 (40.6% of rows are hard/novel)
- Every +0.01 on Tg = +0.01/7 = +0.0014 on the overall mean

### What to try
1. **smile_r3.csv SVD/word2vec for Tg** — train an unsupervised SMILES representation
   on 5.97M unlabeled SMILES from scratch; use as Tg features. This specifically helps
   novel structures (the 1,122 hard rows) because it learns general polymer SMILES patterns
   without needing labels.

2. **Cross-target Tg regression** — Tg correlates with rigidity (chain stiffness), which
   correlates with bandgap (egc). Use egc/egb as auxiliary multitask targets for Tg
   in a joint model. Even a weak multitask signal from egc can regularize the Tg model.

3. **Better Tg base model** — current: XGBoost/LGBM on Morgan+char features. Try:
   - Graph neural network on the polymer graph (WL subtree kernel features, higher order)
   - Attention-based ensemble over repeat unit and backbone descriptors
   - Scaffold-diversity-aware data augmentation (random SMILES, stereo variants)

4. **Structure-stratified Tg CV** — current CV likely overestimates generalization by
   grouping train-test SMILES overlap into training folds. Ensure the 457 overlap SMILES
   are ALWAYS in the training fold (never in validation) → more honest Tg R².

**Kill gate**: if Tg proxy R² doesn't reach ≥ 0.910 (currently 0.895), reject.

---

## Priority 2 🔴 — Reduce Public→Private Gap (Chain Variance)
**Expected gain: +0.008–0.015 private (may not show in oracle)**

### Why it's #2
- Current public→private gap: **0.026** (normal: 0.010–0.018)
- The 339-node deep chain has high variance on novel structures
- This gap does NOT show in the oracle (oracle also tracks easy rows)
- This is why oracle = 0.904 but private = 0.891 despite the 0.013 correction

### What to try
1. **Shallow 3–6 model stack** — replace the deep chain with a clean OOF meta-learner.
   Train base models (XGBoost, LGBM, Ridge, ExtraTrees) with structure-grouped CV,
   then stack OOF predictions with a linear meta-learner. No 300+ step chain.

2. **Tanimoto similarity binning** — during validation, measure R² separately for:
   - High-similarity test rows (Tanimoto ≥ 0.7 to training)
   - Low-similarity test rows (Tanimoto < 0.3 to training)
   Report both; only promote if low-similarity R² improves.
   This directly measures generalization to novel structures.

3. **Remove exact overrides** — the exact-label override mechanism is the primary
   driver of the public→private gap. If we remove it (predict all rows with the model),
   public score will drop but private score may improve.
   Test: run current model without overrides → measure oracle proxy delta.

**Kill gate**: Tanimoto low-similarity bin R² must be ≥ 0.88 across all targets.

---

## Priority 3 🟡 — EI/EEA Improvement
**Expected gain: +0.004–0.008 oracle**

### Why it's #3
- ei R² = 0.871 is the weakest target (only 222 train / 148 test rows)
- eea R² = 0.915 is good but could improve
- Every +0.01 on ei = +0.0014 on overall mean

### What to try
1. **Multi-task EI+EEA joint model** — ionization energy and electron affinity are
   physically coupled: `eea ≈ egc − ei` (approximately). Use this as a hard constraint
   or auxiliary loss.

2. **Better electronic features for EI** — EHT orbital features already tried (C1398).
   Try: Coulomb matrix descriptors, extended connectivity fingerprints at higher radii,
   charge partial features.

3. **Gaussian process regression for EI** — with only 222 training points, a GPR with
   Tanimoto kernel can outperform gradient boosting by providing calibrated uncertainty
   and using the full covariance structure.

**Kill gate**: ei proxy R² ≥ 0.890 (from 0.871).

---

## Priority 4 🟡 — smile_r3.csv Representation Learning
**Expected gain: +0.003–0.008 oracle (mainly via Tg + novel structures)**

### What it is
- 5,973,369 unlabeled molecular SMILES (organizer-provided, rules-compliant)
- Mean SMILES length: 54 characters
- Zero overlap with train/test/PI1M

### What to try
1. **SMILES char-level SVD / word2vec** (scratch-trained inside notebook):
   - Tokenize SMILES characters, train word2vec skip-gram on 5.97M sequences
   - Use 100-dim embeddings as additional features for all 7 targets
   - Must be trained from random init, all inside the notebook run

2. **Morgan fingerprint vocabulary extension** — use the 5.97M SMILES to build a
   larger substructure vocabulary (vs just train/test), then apply TFIDF-style weighting.
   More complete vocabulary → better feature for novel test SMILES.

3. **Scaffold-based nearest-neighbor features** — for each test SMILES, find its
   nearest neighbors in the 5.97M corpus by Tanimoto; use their structural properties
   as additional features.

**Kill gate**: must show improvement on low-similarity Tanimoto bin.

---

## Priority 5 🟢 — Better Cross-Validation
**Expected gain: indirect (prevents selection of overfitting models)**

### What to implement
1. **Mandatory low-similarity bin reporting** — for every experiment, report R² on
   test rows with Tanimoto < 0.3 to training. This is the "private split" proxy.

2. **Structure-grouped CV** — ensure no canonical SMILES straddles train/val across
   all five folds (currently enforced, but verify).

3. **Target-stratified + scaffold-stratified folds** — current: target-stratified only.
   Add scaffold grouping (Murcko scaffold or structural family) to prevent within-family
   overfitting.

4. **Update promotion gate** — add low-similarity R² requirement: must be within 0.02
   of high-similarity R² to promote. This penalizes models that overfit easy rows.

---

## Experiment Schedule (Recommended Order)

| # | Experiment | Oracle target | Expected private | Priority |
|---|-----------|--------------|-----------------|---------|
| 1 | smile_r3 SVD/w2v + current base | ≥ 0.906 | ~0.893 | Warmup |
| 2 | Shallow 3-model stack + Tanimoto CV | ≥ 0.908 | ~0.895 | Gap reduction |
| 3 | Multi-task Tg+egc joint model | ≥ 0.912 | ~0.899 | Tg |
| 4 | GPR for EI + constrained EEA | ≥ 0.913 | ~0.900 | EI |
| 5 | Full stack: shallow + smile_r3 + MTL | ≥ 0.920 | ~0.907 | Combined |
| 6 | Calibrated ensemble (diversity) | ≥ 0.928 | ~0.915 | Ensemble |
| 7 | Final tuning round | ≥ 0.935 | ~0.922 | **BEAT 0.92** |

---

## What NOT to Try (Already Proven Low-Yield — from TRIALS.md)

These approaches were tried in Round 2 and did not improve significantly:

| Approach | Outcome | Why failed |
|----------|---------|------------|
| Deeper chain (>200 nodes) | 0.838 on fresh run | Amplified leaf model variance |
| More char arm features | Marginal +0.001 | Already saturated |
| Heavy blend cascades | No improvement | Same base predictions |
| EHT co-test overlay | +0.001 | Marginal |
| Physics projection overlays | +0.001 | Marginal |
| TF-IDF feature expansion (>1000) | No improvement | Already saturated |
| PI1M SVD (C284/C285) | +0.001 | Small absolute effect |
