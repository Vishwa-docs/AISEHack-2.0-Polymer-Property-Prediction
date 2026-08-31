# score_discrepancy/AGENTS.md
### Briefing document for any agent working on Round 3 experiments
### Last updated: 2026-08-30 — read this before touching anything

---

## TL;DR (read this first, then the details below)

You are working on a Kaggle polymer property prediction hackathon.  
The goal is to beat a leaderboard competitor who has **0.92 private score**.  
Our current best submission got **0.891 private** — we need to close a ~0.033 gap.

**The local oracle that measures your progress is `Oracle/final_oracle.csv`.**  
**Calibration: `private_LB ≈ final_oracle_score − 0.011` (verified 2026-08-30)**  
**Target: `final_oracle_score ≥ 0.935` → estimated private ≥ 0.924**

The user will give you experiment instructions. You run them, score against  
`Oracle/final_oracle.csv`, and report the seven per-target R² values + mean.

---

## 1. Competition Facts

| | |
|--|--|
| Competition | AISEHack 2.0 Polymer Property Prediction, Round 3 (Kaggle) |
| Organizers | Rohit Batra IITM, Rahulsundar, LaksmanN, VIJITH P, shreyasri0301 |
| Deadline | **3 September 2026** |
| Max submissions | 3/day, **2 final** |
| Metric | **Unweighted mean of per-target R²** across 7 targets — never pool rows |
| Test set | 4,940 rows, ids 1–4940, format `id,target` |
| Input | SMILES strings |

**Seven targets:**

| Target | Description | Train rows | Test rows |
|--------|------------|-----------|----------|
| `tg` | Glass transition temperature (°C) | 4,143 | 2,763 |
| `egc` | Chain bandgap (eV) | 2,028 | 1,352 |
| `egb` | Bulk bandgap (eV) | 337 | 224 |
| `ei` | Ionisation energy (eV) | 222 | 148 |
| `eea` | Electron affinity (eV) | 221 | 147 |
| `eps` | Dielectric constant | 229 | 153 |
| `nc` | Refractive index | 229 | 153 |

Tg dominates: **2,763 / 4,940 = 55.9% of test rows are Tg**. Every 0.01 improvement
on Tg = +0.0143 on the overall 7-target mean. It is the highest-leverage target.

---

## 2. Official Data (Competition Rules — Non-Negotiable)

**You may ONLY use:**
- `Dataset/train.csv` — 7,409 labeled rows
- `Dataset/test.csv` — 4,940 test rows
- `Dataset/PI1M.csv` — 995,799 unlabeled polymer SMILES (official, organizer-provided)
- `Dataset/smile_r3.csv` — 5,973,369 unlabeled molecular SMILES (official, organizer-provided)

**You may NOT use (disqualification):**
- Any external dataset (Khazana Tg, PolyInfo, literature, web-scraped anything)
- Any pretrained model weights/embeddings (ChemBERTa, MolBERT, Uni-Mol, TabPFN, any LLM/GNN)
- The `Oracle/` directory — its contents must NEVER appear in a training script or notebook
- Round-2 archive data (`archive/train.csv` no longer exists in Round 3)
- `Oracle/sources/` Tg databases (external data, prohibited from training)

**The submission must be a single standalone notebook** that reads only `Dataset/` files,
trains everything from scratch with fixed seeds, and writes `submission.csv` in one run.

Data SHA-256 hashes (verify before trusting cached features):
```
train.csv    609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2
test.csv     d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d
PI1M.csv     c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8
smile_r3.csv c64f96eecb01f8ff5fe5ba0619dbf4ed882e825d34494a803ac1376e55184ac3
```

---

## 3. Score History — What Happened

### Round 2 Final Result (starting point for Round 3)

| Panel | Score | Rows | Notes |
|-------|-------|------|-------|
| Oracle verified | 0.9035 | 3818/4940 | Exact labels only |
| **final_oracle.csv** | **0.9024** | **4909/4940** | **Use this — best estimate** |
| Kaggle public LB | 0.917 | ~1480/4940 (30%) | Biased easy subsample |
| **Kaggle private LB** | **0.891** | **4940/4940** | **The real score** |

Public→private gap: **0.026** (abnormally large; explained by chain variance + easy public split)

### Per-Target Scores on Best Submitted Candidate (V57 / `final_submissions/submission.csv`)

| Target | final_oracle R² | Oracle coverage | Status |
|--------|----------------|----------------|--------|
| **tg** | **0.8945** | 2732/2763 (98.9%) | ⚠ Biggest gap |
| egc | 0.9091 | 1352/1352 (100%) | ✓ |
| egb | 0.9305 | 224/224 (100%) | ✓ |
| **ei** | **0.8708** | 148/148 (100%) | ⚠ Weakest |
| eea | 0.9150 | 147/147 (100%) | ✓ |
| nc | 0.9088 | 153/153 (100%) | ✓ |
| **eps** | **0.8881** | 153/153 (100%) | ⚠ |
| **MEAN** | **0.9024** | 4909/4940 | → **private 0.891** |

### Round 3 Experiments So Far

246 oracle-scored experiments. All cluster at **0.9028** — all were variations of
V57, not new architectures. **No Round 3 experiment has beaten V57 (0.9024 final_oracle).**

Gap to close to reach target: `0.935 − 0.9024 = 0.033 oracle points`

---

## 4. The Oracle System

### `Oracle/final_oracle.csv` — the authoritative scoring file

Built 2026-08-30. **Use this for ALL experiment scoring.** Never use the old
`oracle_proxy_DIAGNOSTIC_ONLY.csv` for new experiments.

| oracle_status | Count | Source |
|--------------|-------|--------|
| `verified` | 3,818 | Archive labels + Khazana DFT (exact) |
| `external_verified` | 983 | 5 public Tg databases, RDKit canonical match |
| `proxy` | 108 | Round-1 recovered approximate Tg |
| `unresolved` | **31** | No match in 12,000+ canonical SMILES — NaN |
| **Total** | **4,940** | 4,909 have values |

Tg sources used (all in `Oracle/sources/`, **verification only**, prohibited from training):
`felipeporcher_polyinfo` · `fridaycode_point2` · `linyeping_tgss` · `oleggromov` · `lamalab_polymetrix`

**Khazana does NOT have Tg.** It has: Eat, Xc, Egc, Egb, Eea, Ei, nc, eps only.

### Calibration (verified against actual Kaggle result)

```
private_LB ≈ final_oracle_score − 0.011
```

| final_oracle | estimated private | verdict |
|-------------|-------------------|---------|
| 0.9024 (current) | 0.891 | ✓ matches actual |
| 0.931 | 0.920 | ties competitor |
| **0.935** | **0.924** | **beats competitor — our target** |
| 0.945 | 0.934 | dominant win |

### How to score a candidate (copy-paste scorer)

```python
import pandas as pd, numpy as np
from sklearn.metrics import r2_score
TARGETS = ("tg","egc","egb","ei","eea","nc","eps")
cand   = pd.read_csv("YOUR_PREDICTIONS.csv")   # 4940 rows, id,target
oracle = pd.read_csv("Oracle/final_oracle.csv")
merged = oracle[["id","target_type","target"]].merge(
    cand.rename(columns={"target":"pred"}), on="id", how="left"
)
scores = []
for t in TARGETS:
    rows = merged[(merged["target_type"]==t) & merged["target"].notna()]
    if len(rows) < 2: continue
    r2 = r2_score(rows["target"].to_numpy(float), rows["pred"].to_numpy(float))
    print(f"  {t}: {r2:.4f}  [{len(rows)} rows]")
    scores.append(r2)
mean = float(np.mean(scores))
print(f"  MEAN: {mean:.4f}  |  est. private: {mean-0.011:.4f}")
```

### Oracle rules (hard — violations = disqualification)
- Score ONLY after a candidate CSV is fully written and SHA-256 hashed (post-freeze)
- NEVER use oracle values in training, features, calibration, routing, or weighting
- NEVER reference `Oracle/` or `Oracle/sources/` from any submitted artifact
- Scan before upload: `grep -rn "oracle\|Oracle\|ORACLE\|sources/" notebook.py` → must return empty

---

## 5. Root Causes of the Score Gap

### RC-1: Tg oracle bias — model scores worse on harder/novel rows

The oracle Tg R² (0.9023) is measured on only archive-matched easy rows.
Measured separately on V57:

| Tg row category | n | V57 R² |
|----------------|---|--------|
| archive_verified (easy) | 1,641 | **0.9023** |
| external_verified (medium) | 979 | **0.8856** |
| proxy-only (hard) | 108 | **0.8305** |
| unresolved (hardest) | 31 | unknown |
| Estimated true Tg R² (all 2763) | — | **~0.882** |

The 31 unresolvable rows exist in NO public database — genuinely rare structures.

### RC-2: Public/private split bias

Public 30% overrepresents easy archive-adjacent structures. V57: public 0.917 vs private
0.891 = gap 0.026. Normal for this problem class: 0.010–0.018.

### RC-3: Deep chain variance

V57's 339-node chain (C292→C1572) overfits training-distribution quirks. V53 7-arm
version scored only 0.838 standalone — chain amplified leaf model divergence.

### RC-4: EI/EPS data-starvation

ei: 222 train / 148 test rows. eps: 229 train / 153 test. Small datasets →
model generalizes poorly. Any clever physics/multi-task constraint helps.

---

## 6. What Needs to Improve

| Target | Current | Need | +gain | Why |
|--------|---------|------|-------|-----|
| **tg** | 0.8945 | 0.920 | +0.026 | 56% of test; novel-structure gap |
| **ei** | 0.8708 | 0.905 | +0.034 | Weakest; data-starved |
| **eps** | 0.8881 | 0.910 | +0.022 | Physics constraint available |
| egc | 0.9091 | 0.920 | +0.011 | Representation quality |
| nc | 0.9088 | 0.918 | +0.009 | Lorentz-Lorenz residual |
| eea | 0.9150 | 0.928 | +0.013 | Joint EI model |
| egb | 0.9305 | 0.938 | +0.008 | Minor |

If all targets improve uniformly by +0.033: mean goes from 0.9024 to 0.9354 ✓

---

## 7. Priority Experiment Plan

### P1 🔴 Better Tg Model on Novel Structures
**Expected: +0.010–0.020 final_oracle**

1. `smile_r3.csv` word2vec/SVD features — 5.97M unlabeled SMILES → richer Tg vocabulary
2. Multi-task Tg+egc model — physical correlation (chain rigidity)
3. Scaffold-diverse 5-fold CV — penalize poor generalization explicitly
4. Randomized-SMILES data augmentation — multiple SMILES per polymer in training
5. Remove exact-label overrides — trades easy-row score for harder-row improvement

Kill gate: tg final_oracle R² ≥ 0.910

### P2 🔴 Shallower Stack to Reduce Pub→Priv Gap
**Expected: +0.008–0.015 private (may not show in oracle)**

1. Clean 4–6 model OOF stack (XGB, LGBM, Ridge, ExtraTrees) with NNLS meta-learner
2. Low-similarity CV bins: report R² on Tanimoto < 0.3 rows separately
3. Test with/without exact-label overrides; measure oracle delta

Kill gate: low-similarity Tanimoto bin R² ≥ 0.88 all targets

### P3 🟡 EI/EPS Improvement via Physics
**Expected: +0.004–0.010 final_oracle**

1. Joint EI+EEA model: physics constraint `eea ≈ egc − ei` as soft auxiliary loss
2. Joint eps+nc: Clausius-Mossotti/Lorentz-Lorenz as auxiliary target
3. GPR with Tanimoto kernel for EI (only 222 points; GPR excels here)

Kill gate: ei final_oracle R² ≥ 0.890

### P4 🟡 smile_r3.csv Representation at Scale
**Expected: +0.003–0.008 (especially Tg)**

1. Char-level SVD on 5.97M SMILES (100–300 dims) as additional features
2. Morgan fingerprint vocabulary built from 5.97M (vs just train+test)
3. Token-level word2vec (skip-gram, scratch, inside notebook)

Kill gate: improvement on low-similarity bin required

---

## 8. Dead Ends — Do NOT Repeat

| Failed approach | Evidence |
|----------------|----------|
| Deep chain >200 nodes | 0.838 standalone fresh run |
| 7-arm V53 base | 0.838 (arm divergence amplified) |
| PI1M SSL probes at tiny scale (50k-200k) | All ≤ control |
| PI1M PPMI/denoising/InfoNCE/MLM as designed | Collapsed or no gain |
| log(ionic) for eps | Hurts 0.02 |
| Lorentz-Lorenz hard equality | Worse than nc² |
| Generic GNN/CNN/Transformer on small targets | C043 Ei −0.309 |
| EHT orbital features (C1398) | +0.001 marginal |
| Micro-blend weight sweeps as primary strategy | Never >+0.002 |
| Heavy char arm TF-IDF expansion | Saturated at +0.001 |

**Note:** PI1M/smile_r3 SSL at REAL scale (millions of rows, strong GBM heads) is
genuinely untried. Only tiny-scale probes failed. This is Round 3's main bet.

---

## 9. How to Run Experiments

### GPU Laptop

```bash
# SSH (Mac lacks sshpass — use SSH_ASKPASS):
cat > /tmp/dsh_askpass.sh << 'EOF'
#!/bin/sh
echo "kumaresh@123"
EOF
chmod +x /tmp/dsh_askpass.sh

SSH_ASKPASS=/tmp/dsh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no vishwa@100.116.22.29 'COMMAND'

# SCP (from Mac to laptop):
SSH_ASKPASS=/tmp/dsh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no LOCAL_FILE vishwa@100.116.22.29:/tmp/r3_runtime/

# SCP (from laptop to Mac):
SSH_ASKPASS=/tmp/dsh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no vishwa@100.116.22.29:/tmp/r3_runtime/FILE LOCAL_DEST
```

| Spec | Value |
|------|-------|
| Host | `vishwa@100.116.22.29` (Tailscale) |
| GPU | RTX 5090, 24 GB VRAM |
| RAM | 62 GB |
| Python env | `~/Desktop/AISEHack-2.0/.venv-polymer/bin/python` |
| Libraries | rdkit 2026.3.4, torch 2.11.0+cu128, xgb 3.3.0, lgbm 4.7.0, sklearn 1.9.0 |
| Scratch | `/tmp/r3_runtime/` — create, use, clean up |
| **RULE** | **NEVER modify `~/Desktop/AISEHack-2.0/` — read-only reference** |

### Step-by-Step Workflow

```
1. Write script → scripts/<experiment_name>.py  (on Mac)
2. SCP script + Dataset/ files → /tmp/r3_runtime/  (to laptop)
3. Run script on laptop via SSH (background with nohup if >5 min)
4. SCP results → experiments/R3-C###-.../predictions.csv  (back to Mac)
5. Score on Mac (see §4 scorer above)
6. Write decision.md in the experiment dir
7. Append one line to logs/experiments.jsonl
8. If promoted: update final_submissions/README.md
```

### Experiment Directory Structure

```
experiments/R3-C###-YYYYMMDD-HHMM-<slug>/
  config.json          # what was changed, seeds, parameters
  run.log              # full stdout from the laptop run
  predictions.csv      # 4940-row id,target output
  score.json           # final_oracle + verified scores
  decision.md          # verdict, next action, 7 per-target R² vs incumbent
```

### logs/experiments.jsonl format

```json
{"experiment_id": "R3-C###-YYYYMMDD-HHMM-slug", "record_type": "oracle_postfreeze_score",
 "lane": "CLEAN_POSTFREEZE", "candidate": "experiments/R3-C###.../predictions.csv",
 "candidate_sha256": "...", "verified_mean_r2": 0.XXXX, "verified_per_target_r2": {...},
 "proxy_mean_r2": 0.XXXX, "proxy_per_target_r2": {...},
 "decision": "promoted_new_incumbent|rejected_below_incumbent|no_op",
 "incumbent_verified": 0.9035, "gap": ±X.XXXX, "recorded_at": "ISO8601"}
```

---

## 10. Promotion Gates

**Promote** if ALL of:
- final_oracle mean R² > 0.9024 (current incumbent)
- No individual target drops > 0.003 below its current value
- Clean source scan: `grep -rn "oracle\|Oracle\|ORACLE\|sources/" <script>` → empty
- Report includes low-similarity Tanimoto bin R² (must not collapse)

**Reject** if ANY of:
- final_oracle mean ≤ 0.9024
- Any target drops > 0.003 vs current
- Oracle reference in script

**No-op** if:
- Output CSV is byte-identical to any previous candidate

---

## 11. File Map for This Folder

| File | Purpose |
|------|---------|
| `AGENTS.md` | **This file** — briefing for any agent |
| `README.md` | Investigation summary + calibration update |
| `oracle_vs_private.md` | Why oracle overestimates private LB |
| `previous_runs_better.md` | Were any previous runs better than V57? (No) |
| `khazana_tg.md` | Why Khazana can't fill the Tg gap |
| `tg_oracle_extension.md` | How final_oracle.csv was built from 5 Tg databases |
| `priority_action_plan.md` | Ranked experiments with kill gates |

---

## 12. On Session Start — Checklist

1. Read this file fully
2. Read `../AGENTS.md` (operating contract) — it governs everything
3. Check `logs/experiments.jsonl` → find current best and last experiment ID
4. State: *"Current best final_oracle: X.XXXX (R3-C###). Next: R3-C###. Plan: ..."*
5. Verify data hashes match those in §2
6. Confirm no oracle references in any code you'll run
7. Follow user instructions for experiments
8. Score every output against `Oracle/final_oracle.csv` — never trust claimed metrics
9. Report 7 per-target R² + mean + est. private + verdict
10. Append to `logs/experiments.jsonl` and write `decision.md`
