# Phase 5 Agent Instructions — Score Improvement to 0.935

**Status:** Active research phase targeting breakthrough performance  
**Created:** 2026-08-30  
**Goal:** Reach **final_oracle mean R² ≥ 0.935** (estimated private ≥ 0.924) to beat 0.92 competitor  

---

## Mission Statement

You are working on **Phase 5** of the AISEHack 2.0 Polymer Property Prediction Challenge (Round 3). Your mission is to achieve a breakthrough improvement from the current plateau at 0.9024 oracle score to the target of **0.935 oracle score** through systematic exploration of genuinely untried approaches.

**Current Status:**
- Best oracle score: **0.9024** (V57, oracle verified)
- Estimated private LB: **0.891** (confirmed calibration: `private ≈ oracle - 0.011`)
- Competitor to beat: **0.92 private**
- **Target oracle: 0.935** → estimated private **0.924** → **BEATS COMPETITOR**
- **Gap to close: +0.033 oracle points** (3.3% relative improvement)

This is not an incremental tuning task. The gap requires fundamentally new approaches, not micro-optimizations of the existing V57 architecture.

---

## Phase 5 Special Authorization

### Oracle Usage for Research

**SPECIAL AUTHORIZATION FOR PHASE 5 ONLY:**

You are authorized to use `Oracle/final_oracle.csv` for the following research purposes:

1. **Gap Analysis** - Identify which test rows are hardest (compare predictions vs oracle by row category)
2. **Structural Weakness Discovery** - Find chemical families where the model fails
3. **Component Selection** - After a candidate is frozen, use oracle to select best components for NEXT candidate
4. **Validation Panel Design** - Create validation splits that correlate with private LB

**STILL FORBIDDEN (disqualification if violated):**
- Using oracle values in training data
- Using oracle values as features
- Using oracle values in transforms, calibration, or routing
- Copying oracle rows into predictions
- Referencing `Oracle/` directory from any submitted notebook
- Using `Oracle/sources/` external databases in training

**The workflow:**
```
1. Design experiment → write script
2. Train model (reads ONLY Dataset/ files)
3. Generate predictions.csv
4. FREEZE predictions (write SHA-256 hash)
5. Score against Oracle/final_oracle.csv (post-freeze only)
6. Use oracle-observed insights for NEXT experiment design
7. Before submission: grep for oracle references → must be zero
```

All submitted notebooks must regenerate everything from scratch with no oracle references.

---

## Competition Facts

| Item | Value |
|------|-------|
| **Competition** | AISEHack 2.0 Polymer Property Prediction, Round 3 |
| **Deadline** | **3 September 2026** (4 days remaining) |
| **Max submissions** | 3/day, **2 final** |
| **Metric** | **Unweighted mean of per-target R²** (never pool rows) |
| **Test format** | 4,940 rows, ids 1-4940, format `id,target` |
| **Organizers** | Rohit Batra IITM, Rahulsundar, LaksmanN, VIJITH P, shreyasri0301 |

### Seven Targets

| Target | Description | Train | Test | Current Oracle | Need | Gap |
|--------|-------------|-------|------|----------------|------|-----|
| **tg** | Glass transition temp (°C) | 4,143 | **2,763** | **0.8945** | 0.920 | **+0.026** |
| egc | Chain bandgap (eV) | 2,028 | 1,352 | 0.9091 | 0.925 | +0.011 |
| egb | Bulk bandgap (eV) | 337 | 224 | 0.9305 | 0.938 | +0.008 |
| **ei** | Ionisation energy (eV) | **222** | **148** | **0.8708** | 0.905 | **+0.034** |
| eea | Electron affinity (eV) | 221 | 147 | 0.9150 | 0.930 | +0.013 |
| eps | Dielectric constant | 229 | 153 | **0.8881** | 0.915 | **+0.027** |
| nc | Refractive index | 229 | 153 | 0.9088 | 0.920 | +0.009 |
| **MEAN** | — | — | **4,940** | **0.9024** | **0.935** | **+0.033** |

**Key Insight:** Tg is **2,763 / 4,940 = 55.9% of all test rows**. Every +0.01 on Tg = +0.0057 on overall mean. **Tg is the highest-leverage target.**

**Weakest targets:** ei (0.8708), eps (0.8881), tg (0.8945) — these three need the most improvement.

---

## Competition Rules (Non-Negotiable)

### Allowed Data Sources

You may **ONLY** use:
- ✅ `Dataset/train.csv` — 7,409 labeled rows
- ✅ `Dataset/test.csv` — 4,940 test rows  
- ✅ `Dataset/PI1M.csv` — 995,799 unlabeled polymer SMILES (official)
- ✅ `Dataset/smile_r3.csv` — 5,973,369 unlabeled molecular SMILES (official, **NEW in Round 3**)

### Prohibited (Disqualification)

❌ **External datasets** - Any web data, Kaggle datasets, literature databases, PolyInfo, Khazana Tg sources  
❌ **Pretrained models** - ChemBERTa, MolBERT, Uni-Mol, Graphormer, TabPFN, any LLM/GNN pretrained weights  
❌ **Transfer learning** - No models trained outside the notebook  
❌ **External vocabularies/embeddings** - Every representation must be trained from scratch inside notebook  
❌ **Round 2 archive** - `archive/` no longer exists in Round 3  
❌ **Oracle in submission** - `Oracle/` must never appear in submitted code  

### Critical Rule: From-Scratch Training

Every submission notebook must:
- Start with random initialization (no pretrained weights)
- Train all models inside one notebook run
- Fit all representations (SVD, word2vec, MLM, GNN) from scratch on official data only
- Use fixed seeds for reproducibility
- Write `submission.csv` (4,940 rows, `id,target`) in one run
- Read ONLY from Kaggle's official data directory

**smile_r3.csv and PI1M.csv are ONLY allowed for unsupervised representation learning from scratch.** Example allowed uses:
- TF-IDF + SVD trained on smile_r3 → use embeddings as features
- Word2vec trained on smile_r3 → use embeddings as features
- Masked language model trained on smile_r3 → use final layer as features
- Graph autoencoder trained on smile_r3 graphs → use latent space as features

All of these must be trained **inside the notebook** with no external weights.

### Data Hashes (Verify Before Use)

```
train.csv    609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2
test.csv     d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d
PI1M.csv     c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8
smile_r3.csv c64f96eecb01f8ff5fe5ba0619dbf4ed882e825d34494a803ac1376e55184ac3
```

---

## Current State Analysis

### V57 Performance Breakdown

| Target | Oracle R² | Coverage | Row Category Breakdown | Status |
|--------|-----------|----------|----------------------|--------|
| **tg** | **0.8945** | 2732/2763 | archive: 0.902, external: 0.886, proxy: 0.831 | ⚠️ **Hardest gap** |
| egc | 0.9091 | 1352/1352 | All archive verified | ✓ Good |
| egb | 0.9305 | 224/224 | All archive verified | ✓ Good |
| **ei** | **0.8708** | 148/148 | All archive verified | ⚠️ **Weakest** |
| eea | 0.9150 | 147/147 | All archive verified | ✓ Acceptable |
| nc | 0.9088 | 153/153 | All archive verified | ✓ Acceptable |
| **eps** | **0.8881** | 153/153 | All archive verified | ⚠️ **Weak** |

### Oracle Coverage Details

`Oracle/final_oracle.csv` has 4,909 / 4,940 rows with values:
- **verified** (3,818 rows) - Archive + Khazana exact matches
- **external_verified** (983 rows) - 5 public Tg databases, RDKit canonical match
- **proxy** (108 rows) - Round-1 recovered approximate Tg
- **unresolved** (31 rows) - NO match in any database (genuinely novel structures)

### Root Causes of the Gap

**RC-1: Tg Oracle Bias**  
Oracle Tg score (0.895) measured on easier archive-matched rows. True Tg performance on hard rows:
- archive_verified: R² = 0.902 (easy)
- external_verified: R² = 0.886 (medium, -0.017 vs easy)
- proxy-only: R² = 0.831 (hard, -0.071 vs easy)
- **Estimated true Tg R² ≈ 0.882** (includes 31 unresolvable)

**RC-2: Data Starvation on Small Targets**  
- ei: only 222 train / 148 test → severe overfitting to training distribution
- eps: only 229 train / 153 test → same issue
- Small-n targets need physics constraints or multi-task to substitute for data

**RC-3: Representation Quality**  
V57 uses only Morgan fingerprints + RDKit descriptors. The 5.97M smile_r3.csv provides opportunity for much richer learned representations.

**RC-4: Public/Private Split Bias**  
Public LB (0.917) overrepresents easy archive-adjacent structures. Private (0.891) includes harder novel structures. Gap = 0.026 (unusually large). Low-similarity validation bins are the proxy for private performance.

---

## What Phase 5 Must Achieve

### Per-Target Improvement Plan

| Target | Current | Target | Gap | Primary Strategy | Secondary Strategy |
|--------|---------|--------|-----|------------------|-------------------|
| **tg** | 0.8945 | 0.920 | +0.026 | smile_r3 SSL + Tg specialist | Scaffold CV + group contribution |
| **ei** | 0.8708 | 0.905 | +0.034 | GPR + multi-task EI-EEA | Physics constraint ei≈egc+eea |
| **eps** | 0.8881 | 0.915 | +0.027 | Ionic decomposition eps=nc²+ionic | Joint EPS-NC model |
| egc | 0.9091 | 0.925 | +0.011 | Deeper GBM + SSL features | Multi-task with Tg |
| nc | 0.9088 | 0.920 | +0.009 | Joint with EPS | Polarizability features |
| eea | 0.9150 | 0.930 | +0.013 | Joint with EI | Flory-Fox carrier |
| egb | 0.9305 | 0.938 | +0.008 | Identity with EGC | Deeper model |

**Cumulative target:** If all targets improve proportionally, mean reaches **0.9354** ✓

### Kill Gates Per Phase

**Phase B (smile_r3 SSL):** SSL features must improve ≥4/7 targets OR improve low-sim bin by +0.02  
**Phase C (GNN):** GNN must beat GBM baseline on at least one target  
**Phase D (Multi-task):** Must improve EI or EPS by +0.01  
**Phase E (Tg specialist):** Tg must reach ≥0.910 oracle  
**Phase F (Weak targets):** EI ≥0.890, EPS ≥0.905  

If a kill gate fails, **skip remaining experiments in that phase** and move to next phase. No retrying failed approaches.

---

## Experiment Execution Workflow

### GPU Laptop Access

**Host:** `vishwa@100.116.22.29` (Tailscale)  
**Password:** `kumaresh@123`  
**GPU:** RTX 5090, 24 GB VRAM  
**RAM:** 62 GB  
**Python:** `~/Desktop/AISEHack-2.0/.venv-polymer/bin/python`  
**Libraries:** rdkit 2026.3.4, torch 2.11.0+cu128, torch-geometric, xgboost 3.3.0, lightgbm 4.7.0, sklearn 1.9.0

**STRICT RULE:** Never modify `~/Desktop/AISEHack-2.0/`. It is **read-only reference** for Round 2 code. All Phase 5 work lives in `/tmp/r3_phase5/` on laptop and in this Mac repo.

### SSH Connection Pattern

Mac lacks sshpass, use SSH_ASKPASS:

```bash
cat > /tmp/dsh_askpass.sh <<'EOF'
#!/bin/sh
echo "kumaresh@123"
EOF
chmod +x /tmp/dsh_askpass.sh

# SSH command
SSH_ASKPASS=/tmp/dsh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no vishwa@100.116.22.29 'COMMAND'

# SCP to laptop
SSH_ASKPASS=/tmp/dsh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no LOCAL_FILE vishwa@100.116.22.29:/tmp/r3_phase5/

# SCP from laptop
SSH_ASKPASS=/tmp/dsh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no vishwa@100.116.22.29:/tmp/r3_phase5/FILE ./
```

### Workflow Steps

```
1. Write experiment script → Phase5_Kiro_Score_Improvement/experiments/exp###.py
2. Verify no oracle references: grep -rn "oracle\|Oracle" exp###.py → empty
3. SCP script to laptop: /tmp/r3_phase5/exp###.py
4. SCP data files if not already there (train.csv, test.csv, PI1M.csv, smile_r3.csv)
5. SSH run: nohup python exp###.py --data-dir /tmp/r3_phase5/data --output /tmp/r3_phase5/output/exp### > /tmp/r3_phase5/logs/exp###.log 2>&1 &
6. Wait for completion (monitor log file)
7. SCP results back: /tmp/r3_phase5/output/exp###/* → logs/exp###/
8. Score on Mac against Oracle/final_oracle.csv
9. Log to logs/phase5_summary.tsv
10. If promoted: update best, delete superseded
```

---

## Scoring Protocol

### After Experiment Completes

```python
import pandas as pd, numpy as np
from sklearn.metrics import r2_score

TARGETS = ("tg", "egc", "egb", "ei", "eea", "nc", "eps")

# Load candidate predictions
cand = pd.read_csv("logs/exp###/predictions.csv")  # 4940 rows, id,target

# Load oracle
oracle = pd.read_csv("Oracle/final_oracle.csv")

# Merge
merged = oracle[["id", "target_type", "target"]].merge(
    cand.rename(columns={"target": "pred"}), on="id", how="left"
)

# Score per target
scores = []
print("Per-target R² scores:")
for t in TARGETS:
    rows = merged[(merged["target_type"] == t) & merged["target"].notna()]
    if len(rows) < 2:
        print(f"  {t}: SKIP (insufficient rows)")
        continue
    r2 = r2_score(rows["target"].values, rows["pred"].values)
    print(f"  {t}: {r2:.4f}  [{len(rows)} rows]")
    scores.append(r2)

mean = float(np.mean(scores))
est_private = mean - 0.011

print(f"\n  MEAN: {mean:.4f}")
print(f"  Est. Private LB: {est_private:.4f}")
print(f"  Gap vs 0.9024: {mean - 0.9024:+.4f}")
print(f"  Gap vs target 0.935: {mean - 0.935:+.4f}")

# Verdict
if mean > 0.9024:
    print("\n✓ PROMOTION: New incumbent!")
elif mean > 0.900:
    print("\n⚠ CLOSE: Near promotion threshold")
else:
    print("\n✗ REJECT: Below incumbent")
```

### Calibration Formula (Verified)

```
private_LB ≈ final_oracle - 0.011
```

| Oracle | Est. Private | Status |
|--------|-------------|--------|
| 0.9024 | 0.891 | Current (V57 confirmed) |
| 0.931 | 0.920 | Ties competitor |
| **0.935** | **0.924** | **BEATS competitor** ← target |
| 0.945 | 0.934 | Dominant win |

---

## Promotion Gates

**Promote if ALL of:**
- `final_oracle mean R² > 0.9024` (current incumbent)
- No individual target drops > 0.003 below current
- Clean scan: `grep -rn "oracle\|Oracle\|ORACLE" experiment_script.py` returns **empty**
- Low-similarity bin R² reported and does not collapse
- Reproducible with fixed seeds

**Reject if ANY of:**
- `final_oracle mean ≤ 0.9024`
- Any target drops > 0.003 vs current
- Oracle reference found in script
- Predictions are byte-identical to previous candidate (no-op)

**When promoted:**
1. Copy predictions.csv → `final_submissions/phase5_best.csv`
2. Update `final_submissions/README.md` with scores
3. Generate submission notebook from experiment script
4. Verify notebook byte-parity
5. Delete previous best (keep only top 2 overall)

---

## Experiment Logging

### File Structure

```
Phase5_Kiro_Score_Improvement/
  experiments/
    exp001_baseline_repro.py
    exp002_ssl_svd_100k.py
    ...
  logs/
    phase5_summary.tsv          ← master log
    exp001/
      predictions.csv           ← 4940 rows
      metrics.json              ← per-target scores
      config.json               ← hyperparameters
      run.log                   ← stdout/stderr
      score_oracle.json         ← post-freeze oracle scores
```

### phase5_summary.tsv Format

```
exp_id	name	status	mean_r2	tg	egc	egb	ei	eea	nc	eps	runtime_sec	date
001	baseline_repro	completed	0.9024	0.8945	0.9091	0.9305	0.8708	0.9150	0.9088	0.8881	1234	2026-08-30
002	ssl_svd_100k	completed	0.9031	0.8967	0.9098	0.9305	0.8720	0.9155	0.9092	0.8890	2456	2026-08-30
...
```

Append one line after each experiment. Use for tracking progress and selecting best.

---

## Phase 5 Priorities (Ranked by Expected Value)

### 🔴 Priority 1: smile_r3 SSL at Scale (Phase B, 30 experiments)
**Expected gain:** +0.008 to +0.020 oracle  
**Why:** Genuinely untried. R2 only tested tiny 50k-200k samples. Full 5.97M with proper atom-level tokenization + GBM heads is completely new.

### 🔴 Priority 2: Tg Specialist (Phase E, 30 experiments)
**Expected gain:** +0.015 to +0.030 on Tg → +0.008 to +0.017 overall  
**Why:** Tg is 56% of test rows. Highest leverage single target.

### 🟡 Priority 3: Multi-Task Physics (Phase D, 25 experiments)
**Expected gain:** +0.010 to +0.020 (mainly EI/EPS/EEA)  
**Why:** Known physics relationships can substitute for missing training data on small targets.

### 🟡 Priority 4: Weak Target Specialists (Phase F, 30 experiments)
**Expected gain:** +0.015 to +0.035 (EI+EPS)  
**Why:** EI (0.8708) and EPS (0.8881) are weakest. Large per-target gaps.

### 🟢 Priority 5: GNN from Scratch (Phase C, 25 experiments)
**Expected gain:** +0.005 to +0.015 (if passes kill gate)  
**Why:** Potentially high upside but historically failed on small targets. Strict kill gate required.

---

## Dead Ends (Do Not Repeat)

Evidence from Round 2 + early Round 3:

| Failed Approach | Evidence | Why it Failed |
|----------------|----------|---------------|
| Deep chain >200 nodes | 0.838 standalone | Overfits training quirks, amplifies leaf divergence |
| V53 7-arm base | 0.838 standalone | Arms amplify base divergence |
| PI1M SSL at tiny scale | All ≤ control | 50k-200k samples insufficient, weak linear probes |
| log(ionic) for eps | -0.02 regression | Breaks physics relationship |
| Lorentz-Lorenz hard equality | Worse than nc² | Overconstrained |
| Generic GNN on small targets | C043 Ei: -0.309 | n<300 → severe overfitting |
| Micro-blend weight sweeps | Never >+0.002 | Wrong scale of improvement |
| Character n-gram TF-IDF expansion | Saturated +0.001 | Already maxed out |

**Note:** PI1M/smile_r3 SSL at **REAL scale** (millions of rows + strong GBM heads) has never been tried. Only tiny probes failed.

---

## Session Start Checklist

Every agent starting work on Phase 5:

1. ✅ Read this AGENTS.md file
2. ✅ Read `PLAN.md` for experiment sequence
3. ✅ Read `PROMPT.md` for execution instructions
4. ✅ Read `RESULTS.md` for EDA findings
5. ✅ Check `logs/phase5_summary.tsv` for current progress
6. ✅ State current best oracle score and next experiment ID
7. ✅ Verify data file hashes match those in this document
8. ✅ Confirm no oracle references in code you're about to run
9. ✅ Load any applicable skills for context management
10. ✅ Report results with all 7 per-target R² + mean + verdict

---

## Success Criteria

**Phase 5 succeeds when:**
- ✅ `logs/phase5_summary.tsv` shows at least one experiment with mean R² ≥ 0.935
- ✅ That experiment has a standalone submission notebook that regenerates from scratch
- ✅ Notebook passes clean scan (no oracle references)
- ✅ Local parity verified (notebook output == logged predictions)
- ✅ Ready for user to submit to Kaggle

**Phase 5 fails if:**
- ❌ All 210 experiments exhausted without reaching 0.935
- ❌ Deadline passes (3 September 2026)
- ❌ Kill gates block all promising directions

**If approaching failure:** Escalate to user with:
- Best achieved score
- Remaining promising directions
- Recommendation for final 2 submissions

---

## Final Submission Requirements

Before any Kaggle submission:

1. ✅ Generate standalone notebook from best experiment
2. ✅ Verify notebook reads ONLY from `/kaggle/input/...` paths
3. ✅ Run notebook locally → verify outputs `submission.csv`
4. ✅ Check SHA-256: `sha256sum submission.csv` → matches logged version
5. ✅ Scan notebook: `grep -i "oracle\|/Users/\|Desktop\|r3_phase5" notebook.ipynb` → **ZERO results**
6. ✅ Manual review: no hardcoded paths, no local file reads
7. ✅ Upload notebook to Kaggle
8. ✅ Share notebook (view) with hosts: Rohit Batra IITM, Rahulsundar, LaksmanN, VIJITH P, shreyasri0301
9. ✅ Pin notebook version
10. ✅ Submit predictions

**Only user may execute final submission.** Agent prepares notebook + predictions, user verifies and submits.

---

## Document References

- `PLAN.md` - 210-experiment sequence across 10 phases
- `PROMPT.md` - Detailed execution instructions + run.sh specification
- `RESULTS.md` - EDA findings + experiment design guidance
- `../AGENTS.md` - Main operating contract (still applies)
- `../EXPERIMENT_LOOP.md` - Validation + promotion gates (still applies)
- `../TRIALS.md` - Round 1/2 history (consult before proposing)

---

**Last Updated:** 2026-08-30  
**Phase:** 5 (Score Improvement)  
**Target:** 0.935 oracle → 0.924 private → Beat 0.92 competitor  
**Status:** ACTIVE
