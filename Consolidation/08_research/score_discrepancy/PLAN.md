# Phase 3 Experiment Plan — Polymer Property Prediction
## Goal: final_oracle mean R² ≥ 0.935 → estimated private LB ≥ 0.924

**Date written:** 2026-08-30  
**Current best:** V57 final_oracle = **0.9024** (private LB 0.891)  
**Gap to close:** +0.033 oracle points across 7 targets  
**Deadline:** 30 August 2026  
**All experiments run on GPU laptop at:** `~/Desktop/r3_runtime/Phase_3/`  
**Nothing touches the Mac until experiments are done and results are ready to analyse.**

---

## Context (read first)

This plan is written for a coding agent who will scaffold experiments.  
Before touching anything:

1. Read `../AGENTS.md` (operating contract — governs everything)
2. Read `../EXPERIMENT_LOOP.md` (gates, promotion rules, loop discipline)
3. Read `../CONTEXT.md` (full history, what works, what fails)
4. Read `AGENTS.md` in this folder (per-target scores, oracle system, SSH details)
5. Read `priority_action_plan.md` in this folder (ranked priorities)
6. Read `oracle_vs_private.md` in this folder (why the gap exists)

**Key facts you must not rediscover:**

| Item | Value |
|------|-------|
| Oracle calibration | `private_LB ≈ final_oracle − 0.011` (confirmed) |
| Current best oracle | 0.9024 (V57) |
| Target oracle | **0.935** |
| Tg rows in test | 2,763 / 4,940 (55.9%) — highest-leverage target |
| Weakest targets | ei (0.871), eps (0.888), tg (0.8945) |
| Phase 2 best OOF | 0.852 (exp061_eE12 on OOF; oracle-scored separately) |
| Phase 2 oracle ceiling | ~0.9028 — all 150 experiments are variants of same base |
| smile_r3.csv at real scale | **GENUINELY UNTRIED** — only tiny probes (50k-200k) failed |
| PI1M at real scale | Same — untried at 1M+ with strong GBM heads |
| Deep chain (339 nodes) | Dead end — 0.838 standalone, DO NOT repeat |
| 7-arm V53 base | Dead end — 0.838, arms amplify divergence |

**Phase 2 diagnosis:** All 150 experiments tested variations of the same V57 base  
(different feature subsets, hyperparams, minor architecture tweaks). Peak OOF ≈ 0.852  
(vs target ~0.935 oracle). The Phase 2 framework is good infrastructure — reuse it.  
The problem was insufficient architectural diversity. Phase 3 must test fundamentally  
different approaches.

---

## Infrastructure — GPU Laptop Setup

### Location
All Phase 3 work lives at: `~/Desktop/r3_runtime/Phase_3/`

### Data paths on the laptop (use these in all scripts)
```
TRAIN    = ~/Desktop/r3_runtime/data/train.csv        (7,409 rows)
TEST     = ~/Desktop/r3_runtime/data/test.csv         (4,940 rows)
PI1M     = ~/Desktop/r3_runtime/data/PI1M.csv         (995,799 rows)
SMILE_R3 = ~/Desktop/r3_runtime/data/smile_r3.csv     (5,973,369 rows)
```
These are symlinks or copies of `Dataset/` files. Verify SHA-256 hashes before use  
(hashes in `../EXPERIMENT_LOOP.md` Stage 0).

### Python environment
```
~/Desktop/AISEHack-2.0/.venv-polymer/bin/python
```
Libraries available: rdkit 2026.3.4, torch 2.11.0+cu128, torch-geometric 2.8.0.post1,  
xgboost 3.3.0, lightgbm 4.7.0, catboost, sklearn 1.9.0, mordredcommunity 2.0.7,  
numpy, pandas, scipy, shap.

### Directory structure to create
```
~/Desktop/r3_runtime/Phase_3/
  run.sh                      ← master runner (see §run.sh spec below)
  r3_core/                    ← shared utilities (copy from Phase_2/r3_core, extend)
  experiments/
    exp001_p3A01.py
    exp002_p3A02.py
    ...
    exp250_p3Z50.py
  outputs_and_logs/
    output/                   ← one subdir per experiment
    logs/
      summary.tsv             ← appended by run.sh after each experiment
      <expname>.log           ← full stdout per experiment
```

### run.sh specification

Model exactly after `Phase_2/run.sh`. Requirements:
- Usage: `./run.sh [START [END]] [--smoke]`
- Default: runs exp001 through exp250 in order
- Each experiment: `python -u expXXX_pY.py --output outputs_and_logs/output/<name> --data-dir ~/Desktop/r3_runtime/data` 
- Tees stdout+stderr to `outputs_and_logs/logs/<name>.log`
- After each experiment appends one TSV line to `outputs_and_logs/logs/summary.tsv`:
  `<idx>\t<name>\t<status>\t<mean_r2>\t<wall_sec>`
- On failure: records `failed` in summary, continues to next experiment
- Prints a final banner with pass/fail count and elapsed time
- Uses `PYTHON` env var, defaults to the `.venv-polymer` path

Every experiment script must:
1. Accept `--output DIR`, `--data-dir DIR`, `--smoke` flags
2. Read ONLY from `--data-dir` (never hardcoded paths, never `Oracle/`)
3. Write `metrics.json` and `predictions.csv` (4,940 rows `id,target`) to `--output`
4. Print `mean OOF R2 = X.XXXX` to stdout (run.sh extracts this)
5. Exit 0 on success, nonzero on failure

---

## Scoring After Experiments Complete

Once results are copied back to Mac (into `experiments/Phase3-results/`), score against oracle:

```python
import pandas as pd, numpy as np
from sklearn.metrics import r2_score
TARGETS = ("tg","egc","egb","ei","eea","nc","eps")
cand   = pd.read_csv("PATH/predictions.csv")
oracle = pd.read_csv("Oracle/final_oracle.csv")
merged = oracle[["id","target_type","target"]].merge(
    cand.rename(columns={"target":"pred"}), on="id", how="left"
)
scores = []
for t in TARGETS:
    rows = merged[(merged["target_type"]==t) & merged["target"].notna()]
    r2 = r2_score(rows["target"].to_numpy(float), rows["pred"].to_numpy(float))
    print(f"  {t}: {r2:.4f}  [{len(rows)} rows]")
    scores.append(r2)
mean = np.mean(scores)
print(f"  MEAN: {mean:.4f}  |  est. private: {mean-0.011:.4f}")
```

Oracle must NEVER be referenced from any experiment script. Scan before scoring:
`grep -rn "oracle\|Oracle\|ORACLE" experiments/expXXX*.py` → must return empty.

---

## Gap Analysis — What We Need Per Target

| Target | Current oracle | Need | Delta | Rows | Leverage |
|--------|---------------|------|-------|------|---------|
| tg     | 0.8945 | 0.920 | +0.026 | 2,763 | Highest (55.9%) |
| ei     | 0.8708 | 0.910 | +0.039 | 148  | Low count but weak |
| eps    | 0.8881 | 0.915 | +0.027 | 153  | Physics available |
| eea    | 0.9150 | 0.930 | +0.015 | 147  | Coupled to ei |
| egc    | 0.9091 | 0.925 | +0.016 | 1,352 | Good representation |
| nc     | 0.9088 | 0.918 | +0.009 | 153  | Lorentz-Lorenz |
| egb    | 0.9305 | 0.938 | +0.008 | 224  | Nearly there |

**Tg is 56% of all test rows.** Every +0.010 on Tg = +0.0057 on the mean. It is  
the single most leveraged target. ei and eps need physics-informed approaches.

---

## Experiment Phases

### Phase A — Strong Baseline (exp001–010)
*Purpose: establish a clean, reproducible baseline that beats V57 without the deep chain.*  
*Expected oracle: 0.904–0.908*

| # | Name | Description |
|---|------|-------------|
| 001 | p3A01-clean-stack-v1 | 5-model NNLS stack: XGB + LGBM + CatBoost + Ridge + ExtraTrees. Features: Morgan 2048 + RDKit 200 descriptors + char 3-gram TF-IDF 300 + topology block. Grouped 5-fold CV. No deep chain. |
| 002 | p3A02-clean-stack-v2 | Same as A01 but add SVD-64 on Morgan counts (not binary). Explore if continuous count SVD adds signal. |
| 003 | p3A03-physics-coords | Add physics features: ionic = eps − nc², chi = (ei+eea)/2, egb ≈ a·egc proxy, as extra columns in feature matrix. Pair with A01 stack. |
| 004 | p3A04-tanimoto-krr | Add Tanimoto KRR arm for each target (separate from GBM stack). Blend KRR + A01 predictions by target-local NNLS on OOF. |
| 005 | p3A05-partner-features | Add cross-property partner labels as test-time features (~60% coverage). For test rows where partner target is known (same SMILES in train), inject as feature. Measure per-target uplift. |
| 006 | p3A06-low-sim-audit | Reproduce A01 but report OOF R² on low-sim bin (Tanimoto < 0.3 to training). This is the private-split proxy. Must be ≥ 0.87 to proceed. |
| 007 | p3A07-seed-bag-3 | Run A01 with 3 random seeds, average predictions. Variance reduction baseline. |
| 008 | p3A08-seed-bag-7 | Run A01 with 7 random seeds. Measure if more seeds help after 3. |
| 009 | p3A09-no-overrides | Run A01 without exact-label overrides (pure model predictions for all rows). Measure oracle delta. This measures the public-gap contribution of exact overrides. |
| 010 | p3A10-catboost-only | CatBoost-only, 2000 trees, treating SMILES as categorical text internally + Morgan features. CatBoost's native text features may encode SMILES patterns without manual tokenization. |

---

### Phase B — Tg Specialist Push (exp011–030)
*Purpose: improve Tg R² from 0.895 to 0.910+. Highest leverage target.*  
*Kill gate: Tg oracle R² ≥ 0.910 before phase ends*

| # | Name | Description |
|---|------|-------------|
| 011 | p3B01-tg-group-contrib | Bicerano-style group contribution features for Tg: count occurrence of ~50 functional group fragments (amide, ester, aromatic ring, ether, etc.) as additional features for Tg model only. Hypothesis: explicit group counts capture Tg rigidity better than implicit Morgan bits. |
| 012 | p3B02-tg-backbone-sidechain | Separate features for backbone vs side chain of the repeat unit. For each SMILES: identify the backbone path (shortest path between attachment points [*]) and the side chains. Compute descriptors on each part separately. Combine as extra feature block for Tg. |
| 013 | p3B03-tg-dimer | Extend each polymer SMILES to a dimer by concatenating two repeat units (SMILES(*)-SMILES(*) linkage). Compute Morgan/descriptor features on the dimer representation. Hypothesis: dimer captures short-range chain interactions affecting Tg. |
| 014 | p3B04-tg-trimer | Same as B03 but trimer. Compare OOF vs dimer. Pick better. |
| 015 | p3B05-tg-randsmiles-aug | Randomized-SMILES augmentation for Tg training: for each train polymer, generate 5 valid random SMILES (RDKit Chem.MolToSmiles(mol, doRandom=True)), compute features for each, train on all augmented copies with the same label. At inference, average predictions over 5 random SMILES per test polymer (TTA). |
| 016 | p3B06-tg-tta-k10 | Same as B05 but K=10 random SMILES for stronger TTA averaging. |
| 017 | p3B07-tg-scaffold-cv | Replace target-stratified folds with scaffold-stratified folds (Murcko scaffold clustering) for Tg model selection. More honest OOF estimate for novel structures. |
| 018 | p3B08-tg-knn-features | k-nearest-neighbor features for Tg: for each train/test polymer, find k=5 nearest training neighbors by Tanimoto (Morgan 2048), add their Tg values (mean, std, min, max of neighbor labels) as features. Soft label propagation without leaking test. |
| 019 | p3B09-tg-multitask-egc | Multi-task Tg + egc joint model (shared backbone MLP, separate heads). Egc correlates with chain rigidity which drives Tg. Auxiliary task should regularize the Tg model. Use target-masked loss (only backprop the available targets per row). |
| 020 | p3B10-tg-catboost-text | CatBoost with SMILES as raw text feature (char tokenization, CatBoost's built-in text processing). No manual feature engineering. CatBoost learns an n-gram model internally. |
| 021 | p3B11-tg-flory-fox | Flory-Fox equation proxy: for copolymers detected in SMILES (two [*] attachment points with different chain environments), compute estimated Tg from Fox equation (1/Tg = w1/Tg1 + w2/Tg2). Use as additional feature. |
| 022 | p3B12-tg-conformer-3d | Generate 3D conformers for each repeat unit (RDKit ETKDG), compute 3D descriptors (PMI ratio, asphericity, eccentricity, TPSA 3D). Test if 3D shape captures Tg chain stiffness better. |
| 023 | p3B13-tg-mordred | Full Mordred descriptor set (~1600 descriptors, then prune by variance and correlation). Mordred has polymer-relevant descriptors not in base RDKit. Use with LightGBM. |
| 024 | p3B14-tg-wl-kernel | Weisfeiler-Lehman graph kernel features (sklearn-compatible via grakel or manual WL hash). Polynomial kernel on WL subtree counts. Hypothesis: WL captures higher-order structural patterns than Morgan. |
| 025 | p3B15-tg-ridge-char | High-dimensional char n-gram: character 1-4 grams, TF-IDF, top 2000 features, Ridge regression. Fast, interpretable. Tests whether SMILES string patterns alone can match GBM with descriptors. |
| 026 | p3B16-tg-ensemble-B | Ensemble of best B-phase Tg models (B01+B05+B09+B14) by NNLS on OOF. Combine group-contrib + random-SMILES TTA + multitask + WL kernel predictions. |
| 027 | p3B17-tg-lowsim-audit | Run best B-phase model, measure Tg R² on low-sim bin (Tanimoto < 0.3). This is the novel-structure test. Must reach ≥ 0.88 to proceed. |
| 028 | p3B18-tg-mixup | Feature-space mixup augmentation for Tg: interpolate pairs of training polymers' feature vectors with λ ∈ (0,1), average their Tg labels. Train GBM on augmented dataset (2× size). |
| 029 | p3B19-tg-quantile | Quantile regression for Tg uncertainty: train LightGBM with quantile loss (q=0.1, 0.5, 0.9). Use median prediction as point estimate. Calibration check: what fraction of test Tg falls within the 80% PI? |
| 030 | p3B20-tg-final-compound | Compound the best Tg improvements from B-phase into a single Tg prediction. Use NNLS on all B-phase Tg OOF outputs. This becomes the new Tg arm for global assembly. |

---

### Phase C — smile_r3 Representation Learning (exp031–060)
*Purpose: exploit the 5.97M unlabeled molecular SMILES. All representations trained from  
scratch inside the script. No external weights.*  
*Expected oracle lift: +0.003–0.015 (mainly via Tg and novel structures)*  
*Kill gate: low-sim bin improvement required vs Phase A baseline*

| # | Name | Description |
|---|------|-------------|
| 031 | p3C01-r3-svd-100k | Train char TF-IDF (50k vocab) + TruncatedSVD (128 dims) on a 100k random sample of smile_r3.csv. Apply to train/test SMILES. Add as features to A01 baseline. Warmup to verify pipeline. |
| 032 | p3C02-r3-svd-500k | Same as C01 but 500k SMILES. Measure if more unlabeled data helps. |
| 033 | p3C03-r3-svd-2m | Same as C01 but 2M SMILES. Compute budget: ~20 min for TF-IDF + SVD. |
| 034 | p3C04-r3-svd-full | Full 5.97M SMILES. TF-IDF on chars, SVD 256 dims. Expected ~90 min. This is the key scale test. |
| 035 | p3C05-r3-svd-dims | Compare SVD dims: 64 vs 128 vs 256 vs 512 on 2M subset. Find sweet spot. |
| 036 | p3C06-r3-morgan-vocab | Build Morgan substructure frequency vocabulary from 5.97M SMILES. Compute substructure IDF weights. Apply re-weighted Morgan features to train/test. Better vocabulary = better feature weighting for rare substructures. |
| 037 | p3C07-r3-w2v-100k | Word2vec skip-gram trained on 100k smile_r3 SMILES (char tokenization, window=4, dim=64). Use embeddings as mean-pooled features for train/test. |
| 038 | p3C08-r3-w2v-1m | Word2vec on 1M SMILES, dim=128. |
| 039 | p3C09-r3-w2v-5m | Word2vec on 5M SMILES, dim=256. Longest run in phase (~60 min). |
| 040 | p3C10-r3-w2v-atom | Word2vec where tokens are RDKit atom SMILES tokens (not chars). Vocabulary of ~1000 atom-level tokens. May capture chemistry better than raw chars. |
| 041 | p3C11-r3-ppmi-svd | PPMI-weighted co-occurrence matrix on char bigrams, then SVD. PPMI downweights common co-occurrences (like parentheses in SMILES), upweights informative ones. |
| 042 | p3C12-r3-mlm-tiny-100k | Tiny masked language model (2-layer transformer, 64 dim, 4 heads) pretrained on 100k smile_r3 SMILES with 15% masking. Use [CLS] token embedding as feature. Smoke test of MLM. |
| 043 | p3C13-r3-mlm-tiny-1m | Same MLM architecture on 1M SMILES. ~40 min training. |
| 044 | p3C14-r3-mlm-small-1m | Larger MLM (4 layers, 128 dim, 4 heads) on 1M SMILES. ~90 min. |
| 045 | p3C15-r3-mlm-small-5m | MLM (4 layers, 128 dim) on 5M SMILES. ~6 hours. Schedule as overnight run. |
| 046 | p3C16-r3-mlm-medium-5m | MLM (6 layers, 256 dim, 8 heads) on 5M SMILES. Best quality. ~12 hours. Schedule as full overnight. |
| 047 | p3C17-r3-pi1m-svd | SVD on all 995,799 PI1M polymer SMILES (not the 5.97M molecular ones). Polymer-specific vocabulary may complement the molecular one. |
| 048 | p3C18-r3-pi1m-w2v | Word2vec on PI1M. Compare to C09 (smile_r3 w2v). Do polymer SMILES have different useful patterns? |
| 049 | p3C19-r3-combined-svd | Combine PI1M + smile_r3 (2M sample) for joint TF-IDF/SVD. Larger, mixed vocabulary. |
| 050 | p3C20-r3-knn-r3 | k-nearest-neighbor lookup in smile_r3 corpus by Morgan Tanimoto (k=10). For each train/test polymer, find its nearest neighbors in the unlabeled 5.97M. Use neighbor-substructure statistics (mean neighbor MW, mean aromatic fraction) as additional features. Hypothesis: nearby unlabeled structures provide structural context. |
| 051 | p3C21-r3-svd-tg-only | Apply best SVD features (C04/C09) to Tg model only (not all targets). Measure Tg-specific lift. Compare to baseline. |
| 052 | p3C22-r3-denoising-ae | Denoising autoencoder on smile_r3: corrupt 20% of Morgan bits, train autoencoder to reconstruct original. Use bottleneck layer (128 dim) as feature. Different from vanilla SVD: learns nonlinear combinations. |
| 053 | p3C23-r3-contrastive | Contrastive representation: pairs of SMILES from smile_r3 with Tanimoto > 0.8 as positives, Tanimoto < 0.2 as negatives. SimCLR-style training on 500k pairs. 64-dim embeddings. Apply to train/test. |
| 054 | p3C24-r3-pseudo-label-egc | Pseudo-labeling for egc: train egc model, predict on PI1M/smile_r3, select high-confidence predictions (top/bottom 20% by magnitude), add as pseudo-labeled training examples. Retrain egc model. Kill if CV drops. |
| 055 | p3C25-r3-pseudo-label-tg | Pseudo-labeling for Tg: predict Tg on PI1M (polymer structures, more relevant), add predictions as soft labels. Retrain Tg model with down-weighted pseudo examples (weight 0.3). |
| 056 | p3C26-r3-pseudo-label-ei | Pseudo-labeling for ei (only 222 train rows): predict ei on PI1M, add top 500 high-confidence predictions. Test if this helps the data-starved ei target. |
| 057 | p3C27-r3-graph-pretrain | Graph autoencoder pretrained on smile_r3 (50k sample): encode molecular graph (atom features + bonds) to 64-dim vector, decode adjacency. Use encoder as feature extractor. Unlike MLM, this is structure-aware. |
| 058 | p3C28-r3-svd-full-combine | Best experiment: combine full-scale SVD (C04) + word2vec (C09) + PI1M SVD (C17) + denoising AE (C22). Stack with A01 base model. This is the scale-test mega-run. |
| 059 | p3C29-r3-audit-lowsim | Score best C-phase model on low-sim bin. Must improve vs A06. If not, SSL features are not helping on novel structures (exactly the hard problem). |
| 060 | p3C30-r3-compound | Compound best C-phase SSL features into the full 7-target model. NNLS blend with Phase A baseline. Final C-phase candidate. |

---

### Phase D — Physics-Informed Models (exp061–085)
*Purpose: improve ei, eps, eea, nc using physics constraints. These targets have  
known relationships; exploiting them can substitute for more training data.*

| # | Name | Description |
|---|------|-------------|
| 061 | p3D01-ei-eea-joint | Joint ei+eea model exploiting `ei ≈ eea + egc` (HOMO-LUMO gap identity). Train a two-output MLP with soft constraint loss: `L = MSE(ei) + MSE(eea) + λ·(ei_pred − eea_pred − egc_true)²`. Compare vs separate models. |
| 062 | p3D02-ei-gpr | Gaussian process regression for ei (only 222 train rows). Tanimoto kernel (Jaccard on Morgan bits). GPR is optimal for small-n regression with kernel structure. Expect improvement over GBM for ei. |
| 063 | p3D03-ei-gpr-combo | GPR predictions for ei blended with GBM. NNLS weight selection on OOF. |
| 064 | p3D04-eps-nc-joint | Joint eps + nc model with Clausius-Mossotti soft constraint: `eps ≈ (2nc²+1)/(nc²−1)·V_mol/3ε₀` simplified as `eps − nc² = ionic`. Model ionic term, add nc² as feature for eps. Pair constraint as auxiliary loss. |
| 065 | p3D05-eps-ionic-direct | Directly model `ionic = eps − nc²` as the primary regression target for eps. Use nc predictions (or known nc) to recover eps = ionic_pred + nc²_pred. Cleaner than modelling eps directly. |
| 066 | p3D06-eps-log-transform | Test if log(eps) is easier to predict than raw eps. Undo transform for submission. (Note: log(ionic) is known to hurt; log(eps) is different.) |
| 067 | p3D07-egb-egc-identity | For egb: use `egb ≈ a·egc + b` relationship (linear transform). Fit a,b from train rows that have both. Use egc predictions to derive egb predictions. Blend with direct egb model. |
| 068 | p3D08-eea-egc-route | `eea ≈ egc − ei` identity route. If egc and ei are predicted, derive eea from this identity. Blend identity-derived eea with direct eea model. |
| 069 | p3D09-full-physics-routing | Combine all physics routes: eps = ionic + nc², eea = egc − ei proxy, egb = f(egc) proxy. Use partner-feature injection at test time (known partners from train overlap rows). Extends the R2 approach systematically. |
| 070 | p3D10-multi-physics-mlp | MLP with all 7 targets as outputs. Custom loss: `Σ MSE(t) + λ₁·(ei−eea−egc)² + λ₂·(eps−nc²−ionic)² + λ₃·(egb−a·egc−b)²`. All physics constraints as soft regularisers. Shared encoder (256 dim, 3 layers), 7 separate heads. |
| 071 | p3D11-ei-coulomb | Coulomb matrix features for ei: construct Coulomb matrix C[i,j] = 0.5·Zi²·⁴ (diagonal), Zi·Zj/rij (off-diagonal), then sorted eigenvalues as feature vector. Correlates with HOMO/LUMO. |
| 072 | p3D12-ei-humo-proxy | Extended Hückel theory (EHT) HOMO/LUMO proxy via RDKit: compute HOMO energy proxy from conjugation path length, ionization energy from Ei = −HOMO. This is a physics-informed feature, not a pretrained model. |
| 073 | p3D13-eps-polarizability | Compute molecular polarizability proxy from RDKit (Crippen method) as feature for eps/nc. Add to existing features. Polarizability directly enters Clausius-Mossotti equation. |
| 074 | p3D14-nc-lorentz-residual | Model nc using Lorentz-Lorenz residual: predict (nc²−1)/(nc²+2) directly, then back-transform. Compare to direct nc prediction. Tests if the LL coordinate is easier to predict. |
| 075 | p3D15-physics-compound | Compound best physics-route results: NNLS blend of D01+D02+D04+D10+D13 per target. This is the physics-informed arm for global assembly. |
| 076 | p3D16-ei-svr | SVM regression (SVR) for ei with RBF kernel on Morgan features. Classic ML for small-n data. |
| 077 | p3D17-ei-knn | k-NN regression for ei. k=5, weighted by Tanimoto. Useful with only 222 training rows. |
| 078 | p3D18-ei-bayesridge | Bayesian Ridge regression for ei. Calibrated uncertainty, good for small n. |
| 079 | p3D19-eea-physics-audit | Check if using `eea = egc_pred − ei_pred` as the eea prediction (pure physics identity) beats the direct eea model. If yes, switch eea to physics route. |
| 080 | p3D20-eps-nc-system | Full system: joint eps/nc model → both predicted together → consistency regularisation → final outputs. This is the cleanest physics-informed approach for the two optical targets. |
| 081 | p3D21-egc-deeper | Deeper GBM for egc: 3000 trees, lower lr (0.01), more leaves. egc has 2028 train rows — more model capacity may help. |
| 082 | p3D22-egb-deeper | Same for egb: deeper GBM, 337 train rows. |
| 083 | p3D23-ei-feature-ablation | Systematically remove one feature block at a time from the ei model. Identify which blocks actually help (with only 148 test rows, the current model may be memorising noise). |
| 084 | p3D24-tg-physics-rigidity | Physics-informed Tg: compute a molecular rigidity index from the backbone (rotatable bonds, ring fraction, sp2 fraction, backbone conjugation length). Use as a Tg-specific feature that captures the known physics of Tg. |
| 085 | p3D25-physics-final-compound | Full physics-informed compound: combine D-phase arms with B-phase Tg arm and A-phase base. NNLS per-target on OOF. Best candidate before entering SSL phase. |

---

### Phase E — Graph Neural Networks from Scratch (exp086–105)
*Purpose: test GNNs trained from scratch. R2 experiments failed on small targets (C043  
ei: −0.309) but a properly regularised GNN on Tg (4143 train rows) is worth one shot.*  
*Kill gate: GNN must beat GBM baseline for the same target. If not, skip remaining GNN variants.*

| # | Name | Description |
|---|------|-------------|
| 086 | p3E01-gcn-tg | Graph Convolutional Network (GCN) for Tg only. Atom features: atomic num, degree, hybridization, aromaticity, H count. 3 message-passing layers, hidden 256, global mean pool, 2-layer MLP head. Dropout 0.2. Adam 1e-3. 5-fold grouped CV. Kill gate: must beat XGB baseline OOF R² for Tg. |
| 087 | p3E02-gat-tg | Graph Attention Network (GATv2) for Tg. Same atom features, 3 layers, 4 attention heads, hidden 128. |
| 088 | p3E03-mpnn-tg | Message Passing NN (MPNN) for Tg. Edge features: bond type, conjugation, ring membership. 3 passes, hidden 256, set2set readout. |
| 089 | p3E04-gnn-tg-augmented | Best GNN (from E01-E03) with randomized-SMILES augmentation. For each training polymer, generate 5 random valid SMILES → 5 isomorphic graphs. Train on augmented dataset with same label. TTA at inference. |
| 090 | p3E05-gnn-egc | Best GNN architecture applied to egc (2028 train rows). |
| 091 | p3E06-gnn-multitask | Multi-task GNN: shared graph encoder, 7 target heads. Target-masked loss. Expected to help small targets (ei, eps, nc, eea). |
| 092 | p3E07-gnn-blend | Blend best GNN outputs with GBM baseline (NNLS per target). GNN + GBM should be complementary — GNN captures graph topology, GBM captures descriptor interactions. |
| 093 | p3E08-gnn-physics-constrained | GNN with physics loss terms: `λ·(ei−eea−egc)²` etc., same as D10 but with GNN encoder instead of MLP. |
| 094 | p3E09-gnn-ssl-finetune | Pretrain GNN graph autoencoder on 100k smile_r3 graphs (reconstruct graph structure). Finetune on labeled data. This is the proper GNN+SSL combination. |
| 095 | p3E10-gnn-dimer | GNN trained on dimer graphs (repeat unit connected to itself). Same as B03 but as graph, not SMILES string. |
| 096 | p3E11-gcn-tg-deep | Deeper GCN (6 layers) with residual connections (add skip connections to avoid over-smoothing). Regularized with edge dropout. |
| 097 | p3E12-gnn-polymer-graph | Model the polymer chain explicitly: concatenate 3 repeat units as a chain graph, add inter-unit bonds. This represents polymer topology more faithfully than just one repeat unit. |
| 098 | p3E13-gnn-tg-scaffold-cv | Run best GNN with scaffold-stratified CV (Murcko scaffold). More honest estimate than simple grouped CV. |
| 099 | p3E14-gnn-egb-eea-ei | GNN applied to the three small electronic targets together (egb 337, eea 221, ei 222 rows). All use the same encoder with separate heads. |
| 100 | p3E15-gnn-compound | Compound all GNN outputs (per target) with Phase A+B+D baseline. Final GNN contribution arm. |
| 101 | p3E16-gnn-fps-concat | GNN + fingerprint concatenation: pool GNN graph embedding, concatenate with Morgan fingerprint vector, feed to final MLP. Combines learned and handcrafted features. |
| 102 | p3E17-sage-tg | GraphSAGE for Tg. Different aggregation (MEAN/MAX/LSTM) than GCN/GAT. |
| 103 | p3E18-gnn-tg-noise-aug | Add Gaussian noise to node feature vectors during training (dropout-style). Regularises the GNN on the 4143 Tg rows. |
| 104 | p3E19-gnn-tg-ensemble | Ensemble of 5 GNN seeds (E01+E02+E03+E11+E17) for Tg only. Average predictions. |
| 105 | p3E20-gnn-final-compound | Final compound: GNN ensemble (Tg) + GNN multitask (small targets) + Phase A+B+D baseline. Scored against oracle. |

---

### Phase F — SMILES Transformer from Scratch (exp106–125)
*Purpose: train a small Transformer encoder on SMILES strings, using smile_r3 for pretraining  
and labeled data for fine-tuning. This is the deepest SSL approach.*  
*Kill gate: must improve on Tg or ei vs Phase C baseline. If not, skip F05+.*

| # | Name | Description |
|---|------|-------------|
| 106 | p3F01-smiles-transformer-smoke | Tiny BERT-like encoder (2 layers, 64 dim, 4 heads) pretrained on 100k smile_r3 with masked token prediction (15% mask rate). Char-level tokenization. Finetune on each target separately with a regression head. This is a smoke test of the pipeline. |
| 107 | p3F02-smiles-transformer-1m | Same architecture, 1M pretraining SMILES. |
| 108 | p3F03-smiles-transformer-5m | Same architecture, 5M SMILES. Overnight run ~8 hours. |
| 109 | p3F04-smiles-bert-small | BERT-small (4 layers, 256 dim, 4 heads) pretrained on 5M smile_r3. Then finetune separately for each of 7 targets. This is the primary large-run SSL experiment. ~12-16 hours total (pretrain 8h + finetune 4h). |
| 110 | p3F05-smiles-bert-multitask | Pretrain as F04, but finetune with all 7 targets simultaneously (multi-task fine-tuning, availability-masked loss). |
| 111 | p3F06-smiles-bert-blend | Blend F04/F05 transformer predictions with Phase A+B+D GBM baseline by NNLS per target on OOF. |
| 112 | p3F07-smiles-transformer-polymer | Pretrain only on PI1M (polymer-specific SMILES, 995k rows). Then finetune. Compare polymer-specific vs generic molecular pretraining. |
| 113 | p3F08-smiles-selfies | Tokenize SMILES using SELFIES encoding (guaranteed valid). Same transformer architecture as F03. SELFIES tokens have different structure — may capture chemistry differently. |
| 114 | p3F09-smiles-roberta-style | Replace masked LM with SMILES permutation-invariance task: predict if two SMILES represent the same molecule (contrastive). Pretrain on smile_r3 pairs, finetune on labels. |
| 115 | p3F10-smiles-span-mask | Use span masking instead of token masking (mask contiguous substrings = functional groups). May teach the model about functional group semantics. |
| 116 | p3F11-smiles-tg-transfer | Specific transfer: pretrain on PI1M (polymers), intermediate training on egc (large labeled set, 2028 rows), then finetune on Tg (4143 rows). Multi-step transfer within allowed data. |
| 117 | p3F12-smiles-tta-20 | Apply TTA to best transformer model: 20 random SMILES permutations per test polymer, average predictions. |
| 118 | p3F13-smiles-curriculum | Curriculum learning for SMILES pretraining: sort smile_r3 by SMILES length, train short-to-long. Hypothesis: shorter SMILES (simpler molecules) are easier for the model to learn initially. |
| 119 | p3F14-smiles-contrastive-full | Full contrastive pretraining (SimCLR-style) on 2M smile_r3 pairs. Positive pairs: same molecule, different random SMILES. Negative pairs: random unrelated molecules. 128-dim embedding. |
| 120 | p3F15-smiles-transformer-compound | Compound best F-phase results with Phase A+B+D+E baseline. Per-target NNLS. |
| 121 | p3F16-smiles-rnn | BiLSTM/GRU on SMILES chars, hidden 256 per direction. Trained from scratch on labeled data only. Simpler sequence model, sometimes competitive with transformers for small data. |
| 122 | p3F17-smiles-1dcnn | 1D CNN on SMILES characters (char embedding 32 dim, 3 conv layers with kernel sizes 3,5,7, global max pool). Fast, good at local n-gram patterns. |
| 123 | p3F18-smiles-rnn-blend | Blend BiLSTM + 1DCNN + transformer predictions. Diverse sequence model ensemble. |
| 124 | p3F19-smiles-tg-final | Best transformer/sequence model for Tg specifically. Single target, full fine-tuning budget. |
| 125 | p3F20-smiles-final-compound | Compound F-phase sequence model predictions into global assembly. Measure oracle delta vs D+E baseline. |

---

### Phase G — Shallow Global Assembly & Invariance (exp126–150)
*Purpose: assemble all previous arms cleanly, enforce polymer-invariance (competition judged theme),  
and produce candidates for oracle scoring.*

| # | Name | Description |
|---|------|-------------|
| 126 | p3G01-global-assembly-v1 | Assemble best predictions from: A-phase base + B-phase Tg + C-phase SSL + D-phase physics + E-phase GNN. Per-target NNLS weights from OOF. No deep chain. Measure oracle. |
| 127 | p3G02-global-assembly-v2 | Same as G01 but add best F-phase transformer. Test if transformer adds signal on top of GBM+GNN+SSL. |
| 128 | p3G03-tta-global | Apply TTA (K=10 random SMILES) to the full global assembly pipeline. Measure invariance across permutations as diagnostic. |
| 129 | p3G04-tta-global-k20 | TTA K=20. Compare to K=10. Diminishing returns curve. |
| 130 | p3G05-invariance-audit | Measure SMILES representation invariance: for each test polymer, generate 20 valid SMILES, run predictions, measure std across predictions. Polymers with high prediction variance are likely OOD. Report variance histogram. This is the invariance robustness demonstration required by the competition's judged theme. |
| 131 | p3G06-no-override-clean | Run global assembly without any exact-label overrides. Pure model predictions. Measure oracle delta. This may hurt oracle (easy rows) but help private (novel rows). |
| 132 | p3G07-tanimoto-cv-final | Final experiment with Tanimoto-binned validation: measure R² separately for Tanimoto < 0.3, 0.3-0.7, >0.7 bins for all targets. Required for promotion decision. |
| 133 | p3G08-calibration-isotonic | Apply isotonic regression (fold-local) to calibrate per-target outputs. Sometimes improves R² by correcting systematic bias. |
| 134 | p3G09-calibration-linear | Apply linear scaling (fold-local bias+scale correction) per target. Compare to isotonic. |
| 135 | p3G10-stochastic-weight-avg | Train multiple GBM seeds, apply stochastic weight averaging across OOF predictions (weighted by inverse OOF error). |
| 136 | p3G11-polymer-invariance-aug | Train the full pipeline with randomized-SMILES augmentation applied globally (not just for Tg). For each train polymer, 3 random SMILES + canonical. Compare to canonical-only training. |
| 137 | p3G12-canonical-vs-random | Ablation: train with canonical SMILES only vs random SMILES only vs combined. Which input representation is better for each target? |
| 138 | p3G13-global-assembly-v3 | Best configuration identified so far. Add any missing component from B/C/D/E/F that scored positive in isolation. |
| 139 | p3G14-shap-analysis | Compute SHAP values for the final GBM arms. Output: top-10 features per target, SHAP summary plots. Store in outputs dir. Required for competition explainability theme. |
| 140 | p3G15-per-target-audit | Per-target audit of the final assembly: for each target, what is the relative contribution of each arm (A/B/C/D/E/F)? Which arm drives each target? Document for FINAL_REPORT.md. |
| 141 | p3G16-global-v3-tta | Apply TTA K=10 to global assembly v3. |
| 142 | p3G17-blend-search | Systematic NNLS weight search for global assembly: sweep regularization strengths. Test if non-NNLS blending (e.g. simplex optimize by target) gives additional gains. |
| 143 | p3G18-oracle-candidate-1 | First oracle-scored candidate: freeze predictions, compute SHA-256, score against final_oracle.csv. Document per-target R² and mean. |
| 144 | p3G19-oracle-candidate-2 | Second oracle candidate: apply TTA to G18. |
| 145 | p3G20-per-target-specialist | Per-target specialist models: for each of the 7 targets, select the single best-performing approach from all phases. Combine them. This sidesteps global architecture debates. |
| 146 | p3G21-low-sim-final-audit | Final low-similarity audit for oracle candidate: measure R² on Tanimoto < 0.3 bin. Required by promotion gate. |
| 147 | p3G22-global-v4 | If oracle candidate 1 or 2 < 0.928, combine additional approaches: add pseudo-labeling from C25 for Tg, add GNN ensemble from E15 for Tg. |
| 148 | p3G23-global-v5-max | Maximum everything: all phase arms + TTA + physics constraints + pseudo-labels + GNN. If still below 0.928, diagnose bottleneck. |
| 149 | p3G24-explainability-pkg | Package SHAP outputs, invariance metrics, and per-target explanations for FINAL_REPORT.md. Run SHAP on the final accepted model. |
| 150 | p3G25-final-submission-prep | Prepare final submission: freeze the best candidate CSV, verify it passes oracle scoring (> current best), verify clean source scan. Document all hashes. |

---

### Phase H — Target-Specific Deep Dives (exp151–175)
*Purpose: if by exp150 the oracle is still below 0.928, dig deeper on the weakest targets.*

| # | Name | Description |
|---|------|-------------|
| 151 | p3H01-tg-structural-families | Cluster train/test polymers by Murcko scaffold family. Fit a separate model per family for Tg (local model). Blend with global model by distance-weighted average. |
| 152 | p3H02-tg-similarity-weighted | Weight training samples by their Tanimoto similarity to each test polymer (higher weight for similar train rows). Refit GBM with sample weights. |
| 153 | p3H03-tg-outlier-removal | Identify Tg outliers in training (residual > 3σ from ensemble prediction). Remove them, retrain. Check if removing noisy Tg labels improves novel-structure generalization. |
| 154 | p3H04-tg-augment-pi1m | Use PI1M as Tg training augmentation via structural similarity: for each PI1M polymer that is structurally similar (Tanimoto > 0.6) to a labeled train polymer, assign the neighbor's Tg as a soft label. Train Tg model on augmented + real data. |
| 155 | p3H05-tg-ridge-high-dim | Very high-dimensional Ridge regression for Tg: Morgan 4096 (r=3) + atom pairs + topological torsion + char 5-gram TF-IDF 5000. Ridge handles collinearity. Sometimes beats GBM on high-dim sparse inputs. |
| 156 | p3H06-ei-more-features | For ei: add Coulomb matrix features (C01 above), EHT proxy (D12), all Mordred electronic descriptors, RDKit HOMO/LUMO gap estimate from conjugation. Stack all features. |
| 157 | p3H07-ei-gpr-large-k | GPR for ei with k=10 nearest training neighbors as inducing points. Sparse GPR. |
| 158 | p3H08-eps-free-volume | Fractional free volume (FFV) proxy from RDKit 3D conformers as feature for eps. FFV is physically linked to dielectric constant. |
| 159 | p3H09-eps-molar-refraction | Molar refraction (from RDKit Crippen_MolMR) as feature for eps/nc. Directly in the Clausius-Mossotti equation. |
| 160 | p3H10-eea-deep-model | Deeper CatBoost for eea: 2000 trees, lower lr, all feature blocks. eea has 221 train rows — test if more capacity helps. |
| 161 | p3H11-egc-molecular-orbital | Extended Hückel orbital features for egc: HOMO/LUMO level proxies from topology. Chain bandgap is LUMO−HOMO in Hückel. |
| 162 | p3H12-nc-eps-constraint-hard | Hard constraint: set eps_pred = nc_pred² + ionic_pred for all rows (where ionic is predicted). Compare hard constraint to soft constraint (D04). |
| 163 | p3H13-tg-polymer-weight | Per-polymer prediction error analysis: identify which chemical families have the largest Tg errors. Fit error-correction models for those families. |
| 164 | p3H14-all-targets-mordred | Full Mordred descriptor set for all 7 targets. Prune by variance + correlation before GBM training. ~1600 features. |
| 165 | p3H15-catboost-all-targets | CatBoost (only) for all 7 targets with carefully tuned hyperparameters (Optuna, 200 trials per target, 3-fold inner CV). |
| 166 | p3H16-lgbm-dart | LightGBM with DART boosting (dropout-additive regression trees) for all targets. DART reduces overfitting compared to gradient boosting. |
| 167 | p3H17-xgb-hist | XGBoost with histogram-based splits. Faster and sometimes better than exact splits for large feature sets. |
| 168 | p3H18-rf-extra-trees | ExtraTreesRegressor (fully random splits) for all targets. Extremely randomized — more diverse than standard RF, often good ensembled with GBM. |
| 169 | p3H19-blended-stack | Full stacking: base layer (XGB+LGBM+CatBoost+RF+ExtraTrees+GNN+Transformer), meta layer (Ridge per target). 7-level meta-learner. |
| 170 | p3H20-tg-domain-knowledge | Implement van Krevelen group contribution method for Tg: parse SMILES to identify ~100 known functional groups, look up their van Krevelen Tg contributions, sum them. This is a pure physics baseline for Tg that uses no ML. Measure its R² alone, then as feature. |
| 171 | p3H21-semi-supervised-label-spreading | Label spreading (sklearn) on the full train + unlabeled PI1M set. Build a Tanimoto-weighted graph. Propagate Tg labels to unlabeled polymers. Use spread labels as soft training examples. |
| 172 | p3H22-mean-teacher | Mean Teacher semi-supervised: train two models (student and teacher), teacher = EMA of student. Unlabeled PI1M data contributes via consistency loss between student and teacher predictions. Especially useful for Tg. |
| 173 | p3H23-self-training | Self-training: train GBM on labeled data, predict on PI1M, add top-confidence predictions as pseudo-labels, retrain. Iterate 3 times. Simple but sometimes effective. |
| 174 | p3H24-tg-mixup-advanced | Advanced Tg mixup: interpolate in graph embedding space (using GNN encoder), not just feature space. Manifold mixup in latent space may produce more chemically plausible augmented polymers. |
| 175 | p3H25-h-phase-compound | Compound best H-phase improvements into the global assembly. Re-score oracle. |

---

### Phase I — Hyperparameter Systematic Sweep (exp176–200)
*Purpose: systematic tuning of the best architecture found in Phases A–H. Not a primary  
signal source but can add +0.002–0.005.*

| # | Name | Description |
|---|------|-------------|
| 176 | p3I01-lgbm-optuna-tg | Optuna hyperparameter search for LGBM on Tg: 200 trials, 5-fold grouped CV. Parameters: num_leaves (31-511), lr (0.005-0.2), min_child_samples (5-100), feature_fraction (0.5-1.0), bagging_fraction (0.5-1.0). |
| 177 | p3I02-lgbm-optuna-ei | Same for ei. |
| 178 | p3I03-lgbm-optuna-eps | Same for eps. |
| 179 | p3I04-xgb-optuna-tg | Optuna for XGBoost on Tg: max_depth (4-12), eta (0.01-0.3), subsample (0.5-1.0), colsample (0.5-1.0), lambda (0-10). |
| 180 | p3I05-catboost-optuna-tg | Optuna for CatBoost on Tg: depth (4-10), lr (0.01-0.3), l2 (0-10), border_count (32-255). |
| 181 | p3I06-feature-select-tg | Feature selection for Tg: start with all blocks (Morgan + descriptors + char-ngram + SVD + topology + group-contrib + Mordred + SSL), use LGBM feature importance + permutation importance to prune. Find minimum feature set that preserves 99% of R². |
| 182 | p3I07-feature-select-ei | Same for ei. |
| 183 | p3I08-feature-select-eps | Same for eps. |
| 184 | p3I09-n-folds-sweep | Compare 3, 5, 7, 10 folds on Tg model. Does more folds reduce variance in OOF estimate? What is the optimal fold count? |
| 185 | p3I10-ensemble-size-sweep | How many seeds to bag for Tg: 1, 3, 5, 7, 10, 15 seeds. Diminishing returns curve. |
| 186 | p3I11-lgbm-gbm-cat-final | Final tuned LGBM + XGB + CatBoost ensemble with optimally tuned hyperparameters per target. |
| 187 | p3I12-stochastic-blend | Try Bayesian optimization of blend weights (instead of NNLS) for the global assembly. Minimize oracle-proxy score (if allowed post-freeze). |
| 188 | p3I13-feature-interactions | Add pairwise interaction features: product of top-5 SHAP-important features per target. Explicitly model known interactions (e.g. aromatic_fraction × rotatable_bonds for Tg). |
| 189 | p3I14-noise-regularization | Add Gaussian noise to features during training (σ = 0.01 × feature_std). Acts as Tikhonov regularization. Compare to no-noise baseline. |
| 190 | p3I15-target-transform | Test target transformations: Box-Cox on Tg, log(ei), standardize each target. Compare vs raw target prediction. |
| 191 | p3I16-sample-weights | Down-weight Tg training samples that are exact duplicates (train↔test overlap). Up-weight structurally novel train polymers (low Tanimoto to train center). |
| 192 | p3I17-lgbm-tg-deeper | LGBM for Tg with 5000 trees, very low lr (0.005), early stopping on grouped OOF. Slow but potentially better. |
| 193 | p3I18-catboost-tg-gpu | CatBoost for Tg using GPU training (`task_type=GPU`). Much faster, allows 10k+ trees. |
| 194 | p3I19-all-targets-optuna | Run Optuna jointly for all 7 targets simultaneously: shared hyperparams where possible, separate where not. Use mean R² as objective. |
| 195 | p3I20-final-tuned-compound | Final compound with all tuned hyperparameters applied. Oracle candidate. |
| 196 | p3I21-diversity-ensemble | Diversity-maximizing ensemble: compute pairwise prediction correlations, select subset of arms with minimum average correlation while maximizing mean R². |
| 197 | p3I22-blend-calibration | After NNLS blending, apply fold-local bias correction per target. |
| 198 | p3I23-final-audit | Final validation panel: grouped CV, scaffold CV, similarity CV, low-sim bin, Tanimoto distribution report, no oracle references. |
| 199 | p3I24-final-oracle-candidate | Freeze and score the best candidate to date against final_oracle.csv. |
| 200 | p3I25-decision-point | Review: if oracle ≥ 0.930, proceed to packaging. If not, identify the single biggest gap and continue with Phase J. |

---

### Phase J — Extended SSL at Maximum Scale (exp201–225)
*Activated only if oracle < 0.930 after Phase I. These are the most expensive runs.*

| # | Name | Description |
|---|------|-------------|
| 201 | p3J01-svd-full-rerun | Re-run full 5.97M SVD with 512 dims (upgraded from C04 256). More capacity. ~3 hours. |
| 202 | p3J02-w2v-full-pi1m-combined | Word2vec trained on PI1M + 5.97M combined (6.97M total). Polymer + molecular vocabulary. |
| 203 | p3J03-mlm-bert-base-5m | BERT-base scale (12 layers, 768 dim) trained on 5M smile_r3. ~24 hours. Only if budget allows. |
| 204 | p3J04-pi1m-pseudo-tg-large | Larger pseudo-labeling: predict Tg on ALL 995k PI1M polymers, select top 20k most confident, add to Tg training. Retrain. |
| 205 | p3J05-pi1m-pseudo-all-targets | Pseudo-labeling for all 7 targets on PI1M. Select top 2k per target. Retrain. |
| 206 | p3J06-r3-ssl-downstream-finetune | Best SSL representation from Phase C (full scale), fine-tune on all 7 targets simultaneously with multi-task loss. |
| 207 | p3J07-structural-curriculum | Curriculum training: sort train polymers by structural novelty (Tanimoto to training centroid), train on easy-first, hard-last. |
| 208 | p3J08-scaffold-split-correction | Fit a scaffold-level correction model: after base predictions, compute per-scaffold residual on OOF, fit a separate correction model on scaffold features. |
| 209 | p3J09-large-ensemble-60-models | 60-model ensemble: run all best configurations from all phases with 3 seeds each. Average. More diversity = more stable oracle score. |
| 210 | p3J10-final-ssl-compound | Compound best J-phase SSL results. Oracle score. Promotion decision. |

---

### Phase K — Final Push & Packaging (exp226–250)
*Final integration, submission preparation, and competition-required outputs.*

| # | Name | Description |
|---|------|-------------|
| 226 | p3K01-final-assembly | Full final assembly: best arms from all phases, per-target NNLS, TTA K=10. |
| 227 | p3K02-final-oracle | Score final assembly against Oracle/final_oracle.csv. Target: ≥ 0.935. |
| 228 | p3K03-submission-prep | Prepare 4,940-row submission.csv. Verify: no oracle references, reads only Dataset/, fixed seeds, one run. |
| 229 | p3K04-parity-check | Run the final script twice with same seeds. Verify CSV is byte-identical (determinism check). |
| 230 | p3K05-standalone-validation | Verify the standalone script is self-contained: run with a clean temp dir that has only Dataset/ files. No imports from scripts/, experiments/, Oracle/. |
| 231 | p3K06-shap-final | Final SHAP analysis on the accepted model. Output per-target top features + global importance. |
| 232 | p3K07-invariance-report | Final invariance report: mean and std of predictions across 20 SMILES permutations per test polymer. Distribution plots. Required for competition theme. |
| 233 | p3K08-final-report-data | Generate all data needed for FINAL_REPORT.md: per-target R² history, feature importance, physics constraint checks, invariance metrics. |
| 234 | p3K09-candidate-a | First final submission candidate frozen. SHA-256 recorded. |
| 235 | p3K10-candidate-b | Second final submission candidate (alternative configuration). SHA-256 recorded. |
| 236–250 | p3K11-p3K25-reserve | Reserve slots for any last-minute improvements identified during analysis. |

---

## Priority Order for the Coding Agent

Build and run in this order, using the kill gates to abort dead branches early:

1. **Phase A (exp001-010):** Clean baseline. Takes ~2 hours total. All 10 must run.
2. **Phase B (exp011-030):** Tg specialist. Run B01-B10 first; if kill gate fails (Tg < 0.905), skip B11-B20.
3. **Phase C (exp031-045):** SSL scale ladder (SVD then word2vec). Stop SVD scale at 2M if no improvement over 500k. Run MLM experiments separately (long overnight).
4. **Phase D (exp061-085):** Physics. Run D01-D10 first; if ei GPR not beating baseline, skip D11+.
5. **Phase E (exp086-095):** GNN smoke test first (E01-E03). Kill if GNN doesn't beat GBM on Tg.
6. **Phase F (exp106-108):** Transformer smoke (F01-F03). Kill if no Tg/ei improvement.
7. **Phase G (exp126-143):** Assembly and invariance. Run after at least A+B+D complete.
8. **Phase H, I:** Deep dives and tuning. Only if oracle still below 0.928.
9. **Phase J:** Maximum scale SSL. Only if oracle below 0.930 after Phase I.
10. **Phase K:** Final packaging. Always runs last.

---

## Kill Gates (do not proceed past these if gate fails)

| Gate | Condition | Action if failed |
|------|-----------|-----------------|
| B-phase | Tg OOF R² ≥ 0.905 in any B experiment | Skip B11-B20, proceed to C |
| C-phase | Low-sim bin R² ≥ 0.87 in any C experiment | Abort remaining SSL, SSL is not helping novel structures |
| D-phase | ei oracle R² ≥ 0.885 in any D experiment | Skip D16+, accept current ei |
| E-phase | GNN beats GBM for Tg OOF | Skip E05+, GNN is dead end |
| F-phase | Transformer beats Phase C baseline | Skip F04+, transformers are dead end |
| Global | Any oracle candidate ≥ 0.930 | Proceed to Phase K packaging immediately |
| Hard stop | oracle ≥ 0.935 | Stop experiments, begin packaging |

---

## Promotion Gate (for oracle candidates)

Promote to final_submissions only if ALL:
- `final_oracle mean R² > 0.9024` (beats incumbent V57)
- No individual target drops > 0.003 below its current V57 value
- Clean source scan: `grep -rn "oracle\|Oracle\|ORACLE\|sources/" script.py` → empty
- Low-similarity Tanimoto bin R² ≥ 0.88 on all targets

---

## What NOT to Build (proven dead ends)

| Approach | Evidence | Why dead |
|----------|---------|---------|
| Deep chain > 50 nodes | 0.838 standalone | Amplifies leaf model divergence |
| 7-arm V53 blending of old CSVs | 0.838 | Same — chain divergence |
| log(ionic) for eps | Known from R2 | Hurts 0.02 |
| Lorentz-Lorenz hard equality | Known from R2 | Worse than nc² |
| PI1M SSL tiny scale (50k-200k) | Phase 2 + R2 | All ≤ control |
| Generic GNN without regularisation | R2 C043 | ei −0.309 |
| Heavy char arm TF-IDF alone | R2 history | Saturated at +0.001 |
| Micro-blend weight sweeps as primary strategy | R2 246 exp | Never > +0.002 |
| EHT orbital overlay (C1398 style) | R2 | +0.001, marginal |
| AutoGluon / heavy AutoML | Rules risk + slow | Reproducibility concerns |
| Any pretrained weights (ChemBERTa, MolBERT etc.) | Competition rules | Disqualification |
| External data of any kind | Competition rules | Disqualification |

---

## Expected Oracle Score Trajectory

| After phase | Expected oracle | Est. private |
|-------------|----------------|-------------|
| Phase A | 0.904–0.908 | 0.893–0.897 |
| Phase A + B | 0.908–0.915 | 0.897–0.904 |
| Phase A + B + C | 0.912–0.920 | 0.901–0.909 |
| Phase A + B + C + D | 0.916–0.924 | 0.905–0.913 |
| Phase A + B + D + E + F | 0.920–0.928 | 0.909–0.917 |
| Full assembly (G) | 0.926–0.935 | 0.915–0.924 |
| Target | **≥ 0.935** | **≥ 0.924** |

These are optimistic estimates. Progress will be slower if SSL features don't help novel structures (the core risk). Tg is the swing factor — if Tg R² reaches 0.920, the mean target is achievable.

---

## References

- `../AGENTS.md` — operating contract (read first every session)
- `../EXPERIMENT_LOOP.md` — promotion gates, loop discipline, compute budget
- `../CONTEXT.md` — full history, what works, score calibration
- `AGENTS.md` (this folder) — per-target scores, oracle system, SSH/scoring details
- `priority_action_plan.md` — ranked priority actions with kill gates
- `oracle_vs_private.md` — root cause analysis of the pub/priv gap
- `NEW_EXPERIMENTS.md` — extended ML workflow reference (EDA, feature engineering, ensembling)
- `../Dataset/` — official data (train.csv, test.csv, PI1M.csv, smile_r3.csv)
- `../Oracle/final_oracle.csv` — post-freeze scoring ONLY, never in training scripts
