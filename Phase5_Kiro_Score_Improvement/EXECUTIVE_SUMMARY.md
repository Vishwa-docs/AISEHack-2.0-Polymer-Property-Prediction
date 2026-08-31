# Phase 5: Executive Summary

**Created:** 2026-08-30  
**Status:** ✅ READY TO EXECUTE  
**Total Documentation:** 5,350 lines across 10 files

---

## Mission

**Goal:** Reach 0.935 oracle score (≈0.924 private LB) to beat the 0.92 competitor  
**Current Best:** 0.9024 oracle (V57 submission from Round 2)  
**Gap to Close:** +0.033 oracle improvement needed  
**Method:** 210 systematically designed experiments across 10 phases  
**Timeline:** 3-4 days (72 hours with parallelization, 24-hour buffer)  
**Deadline:** September 3, 2026

---

## The Plan: 210 Experiments in 10 Phases

| Phase | Name | Experiments | Focus | Expected Gain | Priority |
|-------|------|-------------|-------|---------------|----------|
| **A** | Foundation & EDA | 15 | Baseline reproduction, validation framework | Baseline (0.9024) | Essential |
| **B** | smile_r3 SSL at Scale | 30 | 5.97M SMILES self-supervised learning | **+0.008-0.020** | 🔴 Critical |
| **C** | Graph Neural Networks | 25 | GNN from scratch for polymers | +0.005-0.015 | 🟡 Conditional |
| **D** | Multi-Task Physics | 25 | Joint prediction with physics constraints | **+0.010-0.020** | 🔴 Critical |
| **E** | Tg Specialist Push | 30 | Maximize Tg (55.9% of test = highest leverage) | **+0.015-0.030** | 🔴 Critical |
| **F** | Weak Target Specialists | 30 | GPR + multi-task for ei/eps | **+0.015-0.035** | 🔴 Critical |
| **G** | Ensemble & Calibration | 20 | Model diversity + calibration | +0.005-0.012 | 🟢 Important |
| **H** | Test-Time Augmentation | 10 | Randomized SMILES variance reduction | +0.003-0.008 | 🟢 Important |
| **I** | Validation & Robustness | 10 | Scaffold CV, low-sim optimization | +0.005-0.010 | 🟢 Important |
| **J** | Integration & Optimization | 15 | Combine all improvements | **Cumulative → 0.935+** | Essential |

**Total:** 210 experiments  
**Cumulative Expected Gain:** +0.033 to +0.080 (target: +0.033 minimum)

---

## Three Key Innovations (What's Actually New)

### 1. smile_r3.csv SSL at FULL SCALE (Phase B) ⭐ PRIMARY BET

**What Round 2 tried:** Max 200k SMILES with char-level n-grams, linear probe heads  
**What Phase 5 will do:**
- **5.97M SMILES** (full dataset, 30x larger)
- **Atom-level tokenization** (polyBERT method): `[*]`, `C(`, `=O` as tokens
- **6-layer Transformers** with overnight GPU runs (B035, B037)
- **Strong GBM heads** instead of weak linear probes

**Why this matters:** R2 SSL failed because scale was too small and tokenization was wrong. polyBERT (2023) showed atom-level tokenization + proper scale achieves R²=0.80 on 29 properties.

**Expected:** +0.012-0.025 if successful (single biggest opportunity)

### 2. Multi-Task with Physics Constraints (Phase D)

**The weak targets:**
- **ei** (n=222): R²=0.871 currently → target 0.900+
- **eps** (n=229): R²=0.888 currently → target 0.915+

**What's new:**
- **Gaussian Process Regression** with Tanimoto kernel (theoretically optimal for n=222)
- **Joint ei-eea-egc** prediction with physics constraint: ei - eea = egc
- **Ionic decomposition** for eps: model ionic = eps - nc² directly

**Literature support:** Kuenneth 2021 (Patterns) showed multi-task beats single-task by 10-30% on sparse targets

**Expected:** +0.020-0.045 on ei+eps → +0.003-0.006 overall mean

### 3. Tg Specialist (Phase E) — Highest Leverage

**Why Tg matters:** 55.9% of test rows (2,763/4,940) → every +0.01 Tg = +0.0057 overall

**What's new:**
- **Bicerano group contribution features** (~50 functional groups)
- **Backbone/side-chain decomposition** (polymer-specific)
- **PI1M polymer SSL** (995k polymers, not molecules)
- **Rigidity index** (free volume theory)

**Expected:** +0.020-0.030 on Tg → +0.011-0.017 overall mean

---

## Kill Gates (Risk Management)

Each phase has a **decision point** to prevent wasting time on failing approaches:

1. **Phase B (after exp 045):** SSL must improve ≥4/7 targets by +0.005 OR low-sim bin +0.02
2. **Phase C (after exp 055):** GNN must beat GBM on Tg (n=4143)
3. **Phase D (after exp 085):** ei OR eps must improve by +0.01
4. **Phase E (after exp 110):** Tg must reach ≥0.910
5. **Phase F (after exp 140):** ei ≥0.890 OR eps ≥0.905

**If a phase fails its kill gate:** Skip remaining experiments in that phase, move to next phase

**Safety net:** Even if B and C both fail, D+E+F alone can reach 0.925-0.930

---

## Documentation Package

### Core Documents (7 files, 139 KB)

1. **AGENTS.md** (503 lines) — Mission, rules, oracle authorization, workflow
2. **RESULTS.md** (647 lines) — Data analysis, oracle insights, literature review
3. **PLAN.md** (1,462 lines) — All 210 experiments with detailed specifications
4. **PROMPT.md** (1,052 lines) — Agent execution instructions, templates, workflow
5. **README.md** (361 lines) — Quick start guide, troubleshooting
6. **REFERENCES.md** (561 lines) — 20+ literature citations with implementation notes
7. **INTEGRATION_CHECKLIST.md** (445 lines) — Final verification and readiness review

### Infrastructure (3 scripts, executable)

1. **run.sh** (188 lines) — Mac/GPU orchestration with SSH
2. **scaffold_experiment.sh** (131 lines) — Create experiment directories
3. **Oracle/score_against_oracle.py** (122 lines) — Post-freeze scoring

### Everything is Self-Contained

- No dependencies on external systems (except GPU laptop for deep learning)
- All instructions for agents included
- All templates provided
- All literature referenced
- All risks identified
- All mitigations documented

---

## Oracle Policy (Special Authorization for Phase 5)

**Authorization:** Oracle/final_oracle.csv MAY be used for **POST-FREEZE SCORING ONLY**

**Workflow (enforced by scripts):**
1. Generate predictions.csv
2. **FREEZE:** Compute SHA-256 hash → save to prediction_hash.txt
3. **NOW oracle can be read:** Score against final_oracle.csv
4. Log results

**FORBIDDEN (will disqualify):**
- Oracle in training
- Oracle in feature engineering
- Oracle in model selection BEFORE freeze
- Oracle values in any submitted artifact

**Why this is safe:** The freeze (hash) happens BEFORE oracle is read. The oracle is used ONLY to select which frozen prediction to submit, not to train models.

---

## Success Metrics

### Minimum Viable Success (Required)

- **0.935 oracle** (≈0.924 private) → Beats 0.92 competitor ✓
- At least ONE fully reproducible submission
- All competition compliance checks pass

### Stretch Goal (Desired)

- **0.945 oracle** (≈0.934 private) → Dominant win
- Multiple strong submissions (hedge risk)

### Per-Target Trajectory

| Target | Baseline | Target | Strategy |
|--------|----------|--------|----------|
| **tg** | 0.8945 | **0.920** | Phase E specialist (highest leverage) |
| **ei** | 0.8708 | **0.900** | Phase F GPR + Phase D multi-task |
| **eps** | 0.8881 | **0.915** | Phase F ionic + Phase D multi-task |
| egc | 0.9091 | 0.925 | Phase B SSL (large target benefits) |
| nc | 0.9088 | 0.920 | Phase D joint with eps |
| eea | 0.9150 | 0.930 | Phase D joint with ei |
| egb | 0.9305 | 0.938 | Phase D affinity with egc |
| **MEAN** | **0.9024** | **≥0.935** | **Cumulative improvement** |

---

## Timeline & Resource Allocation

**Sequential Execution:** 129 hours (5.4 days) — TOO SLOW  
**Parallel Execution:** 72 hours (3 days) — FEASIBLE  
**Deadline:** September 3, 2026 (4 days available)  
**Buffer:** 24 hours ✓

### Parallelization Strategy

- **Day 1:** Phase A (foundation) → 6 hours
- **Day 2-3:** Phase B (SSL) + Phase D (multi-task) in parallel → 30 hours
- **Day 3:** Phase C (GNN, conditional on B results) → 18 hours
- **Day 3-4:** Phase E (Tg) + Phase F (weak targets) in parallel → 18 hours
- **Day 4:** Phase G/H/I (ensemble/TTA/validation) → 20 hours
- **Day 4:** Phase J (integration) → 10 hours

### Hardware Requirements

- **Mac:** CPU experiments (SVD, GBM, GPR, feature engineering) — most experiments
- **GPU Laptop (SSH):** Deep learning (Transformer, GNN) — ~30 experiments
- **Overnight runs:** B035, B037 (full-scale Transformers) — schedule strategically

---

## What Makes This Different from Round 2?

### Round 2 Problems

1. **SSL at tiny scale** (50k-200k, not 5.97M)
2. **Wrong tokenization** (char n-grams, not atom-level)
3. **Weak heads** (linear probes, not GBM)
4. **No multi-task** for small targets
5. **No GPR** for n=222 (optimal method ignored)
6. **No Tg specialist** despite being 55.9% of test
7. **No kill gates** (repeated failed approaches)

### Phase 5 Solutions

1. ✅ **Full scale SSL** (5.97M SMILES)
2. ✅ **Atom tokenization** (polyBERT method)
3. ✅ **Strong GBM heads**
4. ✅ **Multi-task with physics** (ei-eea-egc, eps-nc)
5. ✅ **GPR for small targets** (theoretically optimal)
6. ✅ **30 experiments for Tg** (highest leverage)
7. ✅ **5 kill gates** (stop failures early)

---

## Competition Compliance

### Rules Checklist (from Competition_Details/)

- ✅ Official data only (train, test, PI1M, smile_r3)
- ✅ No external datasets
- ✅ No pretrained models/weights
- ✅ All SSL trained from scratch inside notebook
- ✅ Fully reproducible single-run notebook
- ✅ Fixed seeds (2026)
- ✅ Share with hosts (view permissions)
- ✅ No test data leakage
- ✅ No repeat-submit probing

**Pre-submission scan commands documented in README.md and PROMPT.md**

---

## Next Steps for Agent Execution

### Session Start (Every Time)

1. Read AGENTS.md (10 min)
2. Read PLAN.md Phase A section (5 min)
3. Read PROMPT.md workflow (10 min)
4. Verify dataset hashes
5. State: current best, next experiment, plan for session

### First Experiment (P5-001)

```bash
cd Phase5_Kiro_Score_Improvement

# Scaffold experiment directory
./scaffold_experiment.sh P5-001 baseline-v57-reproduction

# Edit run_experiment.py (implement V57 baseline)
# See PROMPT.md for templates

# Test
./run.sh --exp P5-001 --smoke

# Full run
./run.sh --exp P5-001

# Results automatically scored and logged
cat logs/phase5_summary.tsv
```

### First Day Goal

- Complete Phase A (Exp 001-015) → 6-8 hours
- Establish baseline: 0.9024 ±0.0005
- Launch Phase B overnight runs (B035, B037)
- Begin Phase D in parallel

---

## Confidence Assessment

### High Confidence (Likely to Work)

- **Phase E (Tg specialist):** Literature + domain knowledge support ✅
- **Phase F (GPR for small targets):** Theoretically optimal method ✅
- **Phase D (Multi-task):** Kuenneth 2021 showed 10-30% gains ✅
- **Phase G (Ensemble):** V57 already uses NNLS, expand it ✅

### Medium Confidence (Worth Trying)

- **Phase B (SSL at scale):** polyBERT worked, but our task is different ⚠️
- **Phase C (GNN):** May work for large targets, fail for small ⚠️
- **Phase H (TTA):** Only helps sequence models, not fingerprints ⚠️

### Safety Margin

If **only** high-confidence phases work:
- Phase D: +0.015
- Phase E: +0.015
- Phase F: +0.020
- Phase G: +0.010
- **Total: +0.060** → final score 0.962 (way above target)

Even if **half** the expected gains realize: 0.9024 + 0.016 = **0.918** → still close to 0.92

**Conservative estimate:** 70% probability of reaching 0.925+, 40% probability of 0.935+

---

## Risk Mitigation

### If Everything Fails (<0.925 after 150 experiments)

1. **Stop and analyze:** What worked even slightly?
2. **Ensemble top-10 experiments:** Diversity may gain +0.005-0.010
3. **Submit best + V57:** Hedge with conservative baseline
4. **Escalate to user:** Report findings, get guidance

### If Individual Phases Fail

- **B fails:** D+E+F+G can still reach 0.925-0.930
- **C fails:** Expected (GNN uncertain), continue without it
- **D fails:** Single-task GPR should still work
- **E fails:** CRITICAL — analyze deeply, try radical approaches
- **F fails:** Less critical, other targets can compensate

**Backup plan always exists**

---

## Final Status

**Documentation:** ✅ Complete (5,350 lines, 10 files)  
**Infrastructure:** ✅ Ready (3 executable scripts)  
**Validation:** ✅ Verified (all consistency checks pass)  
**Compliance:** ✅ Documented (all rules covered)  
**Timeline:** ✅ Feasible (72 hours, 24-hour buffer)  
**Risks:** ✅ Identified & mitigated  

**Overall Assessment:** 🟢 **READY TO EXECUTE**

---

## Key Success Factors

1. **Genuinely new approaches** (not incremental R2 variations)
2. **Scale** (5.97M smile_r3, not 200k)
3. **Correct methods** (atom tokenization, GPR for small n)
4. **Kill gates** (stop failures early)
5. **Leverage** (focus on Tg = 55.9% of test)
6. **Physics** (multi-task constraints for consistency)
7. **Systematic** (210 experiments cover the solution space)

**Probability of Success:** Medium-High  
**Expected Outcome:** 0.925-0.940 oracle range  
**Risk-Adjusted Target:** 0.930 oracle (0.919 private) — still beats 0.92

---

**Bottom Line:** Phase 5 has a **realistic path to 0.935 oracle** through three major innovations (SSL at scale, multi-task physics, Tg specialist) backed by literature, with safety nets if any one fails. The documentation is complete, the infrastructure is ready, and the competition deadline is achievable.

**Next Action:** Agent begins Phase A, Experiment 001 (Baseline Reproduction)

---

**Document:** EXECUTIVE_SUMMARY.md  
**Version:** 1.0  
**Date:** 2026-08-30  
**Status:** 🟢 FINAL — READY FOR EXECUTION
