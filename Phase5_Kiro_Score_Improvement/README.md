# Phase 5: Score Improvement Campaign

**Goal:** Reach 0.935 oracle score (0.924 private) → Beat 0.92 competitor  
**Current Best:** 0.9024 oracle (V57)  
**Gap to Close:** +0.033 oracle improvement needed  
**Timeline:** 3-4 days (210 experiments)

---

## Quick Start

### 1. Read Documentation (REQUIRED)

Read in this order before starting ANY experiments:

1. **AGENTS.md** — Mission, rules, oracle authorization, execution workflow
2. **RESULTS.md** — Data analysis, oracle insights, literature review, experiment design guidance
3. **PLAN.md** — All 210 experiments across 10 phases with specifications
4. **PROMPT.md** — Agent execution instructions, templates, workflow

### 2. Run Your First Experiment

```bash
cd Phase5_Kiro_Score_Improvement

# Scaffold experiment directory
./scaffold_experiment.sh P5-001 baseline-v57-reproduction

# Write experiment script (see PROMPT.md templates)
# ... create experiments/P5-001-.../run_experiment.py ...

# Test locally (smoke test)
./run.sh --exp P5-001 --smoke

# Full run
./run.sh --exp P5-001

# Results automatically scored and logged
```

### 3. Check Results

```bash
# View summary
cat logs/phase5_summary.tsv

# View specific experiment
cat experiments/P5-001-.../oracle_scores.json
```

---

## Directory Structure

```
Phase5_Kiro_Score_Improvement/
├── AGENTS.md              # Agent instructions & rules
├── PLAN.md                # 210 experiment specifications
├── PROMPT.md              # Execution guide & templates
├── RESULTS.md             # Data analysis & literature
├── README.md              # This file
├── run.sh                 # Experiment runner (Mac/GPU)
├── experiments/           # One directory per experiment
│   ├── P5-001-20260830-1234-baseline/
│   │   ├── run_experiment.py
│   │   ├── predictions.csv
│   │   ├── metrics.json
│   │   ├── oracle_scores.json
│   │   └── prediction_hash.txt
│   └── ...
└── logs/
    ├── phase5_summary.tsv     # All results
    ├── phase_a_report.md      # Per-phase summaries
    └── kill_gate_decisions.md
```

---

## Experiment Workflow

**Standard workflow for each experiment:**

1. **Scaffold:** Create experiment directory
2. **Code:** Write self-contained `run_experiment.py`
3. **Test:** Smoke run (2-5 min, minimal features)
4. **Execute:** Full run (Mac CPU or GPU via SSH)
5. **Freeze:** Hash predictions.csv
6. **Score:** Against Oracle/final_oracle.csv (post-freeze only)
7. **Log:** Append to phase5_summary.tsv
8. **Analyze:** Compare to incumbent, check kill gates
9. **Decide:** Continue, skip, or escalate

**See PROMPT.md for complete details.**

---

## Phase Overview

| Phase | Experiments | Focus | Expected Gain | Priority |
|-------|-------------|-------|---------------|----------|
| **A** | 001-015 | Foundation & EDA | Baseline | Essential |
| **B** | 016-045 | smile_r3 SSL at Scale | +0.008-0.020 | 🔴 Critical |
| **C** | 046-070 | Graph Neural Networks | +0.005-0.015 | 🟡 Conditional |
| **D** | 071-095 | Multi-Task Physics | +0.010-0.020 | 🔴 Critical |
| **E** | 096-125 | Tg Specialist Push | +0.015-0.030 | 🔴 Critical |
| **F** | 126-155 | Weak Target Specialists | +0.015-0.035 | 🔴 Critical |
| **G** | 156-175 | Ensemble & Calibration | +0.005-0.012 | 🟢 Important |
| **H** | 176-185 | Test-Time Augmentation | +0.003-0.008 | 🟢 Important |
| **I** | 186-195 | Validation & Robustness | +0.005-0.010 | 🟢 Important |
| **J** | 196-210 | Integration & Optimization | Cumulative | Essential |

---

## Key Innovations

### 1. smile_r3.csv at Scale (Phase B) ⭐ **PRIMARY BET**

**What's new:** 5.97M SMILES (vs R2's max 200k)  
**What's different:**
- Atom-level tokenization (not char n-grams): `[*]`, `C(`, `=O` as tokens
- Full-scale Transformers (6-layer, overnight runs)
- Strong GBM heads (not linear probes)

**Expected:** +0.012-0.025 if successful

### 2. Multi-Task with Physics Constraints (Phase D)

**For ei/eea (weak targets):**
- Joint prediction with constraint: ei - eea = egc
- GPR with Tanimoto kernel (optimal for n=222)

**For eps/nc:**
- Model ionic = eps - nc² directly (physics-consistent)

**Expected:** +0.015-0.030 on weak targets

### 3. Tg Specialist (Phase E)

**Why:** 55.9% of test rows = highest leverage  
**Approaches:**
- Bicerano group contribution features
- Backbone/side-chain decomposition
- Rigidity index
- PI1M polymer-specific SSL

**Expected:** +0.015-0.030 on Tg → +0.008-0.017 overall

---

## Kill Gates (Critical Decision Points)

Each phase has a kill gate. If failed, skip remaining experiments:

- **Phase B (after exp 045):** ≥4/7 targets improved by +0.005 OR low-sim +0.02
- **Phase C (after exp 055):** GNN beats GBM on Tg
- **Phase D (after exp 085):** ei OR eps improved by +0.01
- **Phase E (after exp 110):** Tg ≥ 0.910
- **Phase F (after exp 140):** ei ≥ 0.890 OR eps ≥ 0.905

**Kill gates prevent wasting time on approaches that don't work.**

---

## Oracle Policy (CRITICAL)

**Authorization:** Oracle/final_oracle.csv may be used for **POST-FREEZE scoring ONLY**

**Workflow:**
1. Generate predictions.csv
2. **Freeze:** Compute SHA-256 hash, save to prediction_hash.txt
3. **Now oracle can be read:** Score against final_oracle.csv
4. Log results

**FORBIDDEN:**
- Oracle in training
- Oracle in feature engineering
- Oracle in model selection BEFORE freeze
- Oracle in calibration BEFORE freeze

**Violation = disqualification**

---

## Success Metrics

### Minimum Viable

- **0.935 oracle** (≈0.924 private) → Beats 0.92 competitor
- At least ONE submission ready by Sept 3
- Fully reproducible Kaggle notebook

### Stretch Goal

- **0.945 oracle** (≈0.934 private) → Dominant win
- Multiple strong submissions (hedge risk)

### Per-Target Milestones

| Target | Baseline | Phase B | Phase D/F | Phase E | Phase J | Required |
|--------|----------|---------|-----------|---------|---------|----------|
| tg | 0.8945 | 0.900 | 0.905 | **0.920** | 0.920 | 0.915+ |
| ei | 0.8708 | 0.875 | **0.900** | 0.900 | 0.905 | 0.890+ |
| eps | 0.8881 | 0.895 | **0.915** | 0.915 | 0.915 | 0.905+ |
| egc | 0.9091 | **0.920** | 0.920 | 0.920 | 0.925 | 0.920+ |
| nc | 0.9088 | 0.915 | **0.920** | 0.920 | 0.920 | 0.915+ |
| eea | 0.9150 | 0.920 | **0.930** | 0.930 | 0.930 | 0.925+ |
| egb | 0.9305 | 0.935 | 0.935 | 0.935 | **0.938** | 0.935+ |
| **MEAN** | **0.9024** | 0.9086 | 0.9179 | 0.9214 | **0.9347** | **≥0.935** |

---

## Tools & Scripts

### run.sh

**Main experiment runner**

```bash
# Mac run (CPU experiments)
./run.sh --exp P5-016

# GPU run (SSH to laptop)
./run.sh --exp P5-037 --gpu

# Quick test
./run.sh --exp P5-001 --smoke

# Score existing predictions
./run.sh --exp P5-016 --score-only
```

### Oracle Scoring

```bash
python3 ../Oracle/score_against_oracle.py \
  --predictions experiments/P5-016-.../predictions.csv \
  --oracle ../Oracle/final_oracle.csv \
  --output experiments/P5-016-.../oracle_scores.json
```

### Summary Analysis

```bash
# View all results
cat logs/phase5_summary.tsv | column -t -s$'\t'

# Best experiment
cat logs/phase5_summary.tsv | tail -n+2 | sort -t$'\t' -k3 -gr | head -n1

# Progress tracking
python3 -c "
import pandas as pd
df = pd.read_csv('logs/phase5_summary.tsv', sep='\t')
print(f'Experiments run: {len(df)}')
print(f'Best oracle: {df[\"oracle_r2\"].max():.4f}')
print(f'Gap to 0.935: {0.935 - df[\"oracle_r2\"].max():.4f}')
"
```

---

## Literature & References

**Self-Supervised Learning:**
- Kuenneth et al. (2023): "polyBERT: a chemical language model to enable fully machine-learned molecular dynamics simulations"
- Chithrananda et al. (2020): "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction"

**Multi-Task Learning:**
- Kuenneth et al. (2021, Patterns): "Biasing chemical language models toward property prediction"

**Graph Neural Networks:**
- Yang et al. (2019): "Analyzing Learned Molecular Representations for Property Prediction" (D-MPNN)
- Xu et al. (2019): "How Powerful are Graph Neural Networks?" (GIN)

**Gaussian Processes for Molecules:**
- Ralaivola et al. (2005): "Graph kernels for chemical informatics"
- Rasmussen & Williams (2006): "Gaussian Processes for Machine Learning"

**Polymer Property Prediction:**
- Kim et al. (2018): "Polymer Genome: A Data-Powered Polymer Informatics Platform"
- Bicerano (2002): "Prediction of Polymer Properties" (Group contribution methods)
- Huan et al. (2015): "PolymetriX: Data-driven polymer design"

---

## Troubleshooting

### Experiment fails to run

```bash
# Check script syntax
python3 -m py_compile experiments/P5-XXX-.../run_experiment.py

# Check data paths
ls -lh ../Dataset/train.csv

# Check GPU SSH
SSH_ASKPASS=/tmp/phase5_ssh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  ssh vishwa@100.116.22.29 "echo connected"
```

### Oracle scoring fails

```bash
# Check predictions format
head experiments/P5-XXX-.../predictions.csv
# Must have: id,tg,egc,egb,ei,eea,eps,nc
# 4940 rows (ids 1-4940)

# Check oracle exists
ls -lh ../Oracle/final_oracle.csv
```

### Results don't improve

**Check:**
1. Is validation honest? (grouped CV, scaffold splits)
2. Are features leaking test info?
3. Are kill gates working? (skip failed approaches)
4. Is the approach genuinely new? (consult RESULTS.md dead ends)

**Escalate to user if:**
- 150 experiments run, best <0.925
- All phases fail kill gates
- Unexpected data issues

---

## Competition Compliance Checklist

Before ANY submission:

- [ ] Standalone notebook regenerates CSV identically
- [ ] No oracle references in notebook (`grep -rn oracle`)
- [ ] No local paths in notebook (`grep -rn /Users/`)
- [ ] Reads only from `/kaggle/input/` (or equivalent Kaggle dir)
- [ ] All models trained from scratch (no external weights)
- [ ] All SSL trained inside notebook (no precomputed embeddings)
- [ ] Fixed seeds throughout
- [ ] No external data (only train/test/PI1M/smile_r3)
- [ ] Shared with hosts (view permissions)
- [ ] Reproduces in <9 hours (Kaggle timeout)

---

## Status Tracking

**Current Status:** READY TO START  
**Experiments Completed:** 0/210  
**Best Oracle Score:** 0.9024 (baseline)  
**Gap to Target:** +0.033  
**Days Remaining:** Until Sept 3, 2026

**Next Action:** Begin Phase A (Exp 001: Baseline Reproduction)

---

**Version:** 1.0  
**Created:** 2026-08-30  
**Owner:** Phase 5 Campaign  
**Contact:** See main repo AGENTS.md
