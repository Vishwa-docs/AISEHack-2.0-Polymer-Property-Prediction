# REFINEMENT.md — Final Hackathon Preparation Checklist

**Created:** 2026 (Pre-hackathon final refinement)  
**Audience:** Implementation agent for final polish  
**Context:** AISEHack 2.0 Polymer Property Prediction Round 3 - Final submission ready, needs refinement for presentation

---

## 0. CRITICAL CONTEXT

**Score Status**: MAXIMIZED at 0.90680 (Phase 7). NO score experiments needed.  
**Codebase Status**: Complete, documented, validated.  
**This is REFINEMENT**: Fix gaps, enhance story, consolidate documentation. NOT wholesale changes.

**READ FIRST:**
- `/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/PLAN.md` 
- `/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/RUN.md`
- `/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/Personal/AGENTS.md`

---

## 1. CODEBASE VERIFICATION & CORRECTIONS

### 1.1 Score & Metrics Validation
**Task**: Verify all numbers are consistent and correct across all documents.

**Files to check**:
```
- Personal/docs/00_INDEX.md (canonical source)
- AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/README.md
- Personal/STORY.md
- Personal/Presentation/SLIDE_PLAN.md
```

**Verification checklist**:
- [ ] Private LB: 0.891 everywhere
- [ ] Local panel: 0.9023 (NOT 0.90352, 0.903480, or 0.90229)
- [ ] Per-target R²: tg 0.8953, egc 0.9111, egb 0.9268, ei 0.8711, eea 0.9183, nc 0.9086, eps 0.8847
- [ ] Scorecard: 14/18 (may become 14-15/19 after full run - check outputs/scorecard.md)
- [ ] Physics numbers: egc=ei-eea R² 0.9716, ionic 0/134 violations, Flory-Fox median R²≈0.99
- [ ] Fidelity: 0.851 vs 0.043
- [ ] SHAP cosine: 0.95-0.99
- [ ] No "oracle", "khazana", "polyinfo", "TgSS" in codebase folder

**Action**: Fix any inconsistencies found.

### 1.2 Charts & Misinformation Check
**Task**: Verify methodology correctness in robustness/invariance experiments.

**Files to review**:
```
- Personal/docs/08_robustness/*.md
- AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/outputs/robustness/
- Personal/docs/07_explainability/*.md
```

**Check for**:
- [ ] SMILES invariance methodology correct (30 randomized spellings, graph-only vs full)
- [ ] Fidelity test methodology (ROAR-style masking)
- [ ] Attribution invariance (SHAP cosine across spellings)
- [ ] Flory-Fox relation (Tg vs 1/n linearity)
- [ ] Counterfactuals (27/40 = 67.5% agreement documented with failures shown)

**Known good**: All methodologies are documented correctly in Personal/docs/. Just verify charts match text.

---

## 2. ML/DL BASELINE STORY

### 2.1 Consolidate Existing ML Comparisons
**Task**: Create clear narrative showing ML was tried, measured, and domain knowledge won.

**YOU ALREADY HAVE** (do NOT run new experiments):
1. **GNN failure**: ei −0.309 on 0/5 folds (D-MPNN, literature crossover ~859-1,000 rows vs our 222)
2. **Self-supervised learning**: 9 variants ≤ supervised control
   - **Decisive**: MLM probe 0.651 vs random-init 0.708
   - InfoNCE, PPMI/SVD, subword, denoising all failed
3. **ChemBERTa** (out-of-competition): frozen 0.751, fine-tuned 0.784 vs tree baseline 0.810
4. **Pattern**: 6 of 9 winners are domain knowledge, not ML

**Files already containing this**:
```
- Personal/docs/04_experiments/what_failed.md
- Personal/TRIALS.md (Part 1, D3 and D4)
- AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/Experiment_Logs/D3_ssl_corpora.md
- AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/Experiment_Logs/D4_neural.md
```

**Action needed**:
- [ ] Add ML comparison summary to `Personal/docs/00_My Docs.md` (section "Why domain knowledge beat ML")
- [ ] Ensure presentation slide 5 "failure strip" includes: GNN ei −0.309 · MLM 0.651 vs 0.708 · ChemBERTa note
- [ ] Add to report §6 (Ablations): explicit ML vs domain knowledge comparison table
- [ ] QnA file: Add question "Why didn't you use deep learning?" with this answer

**Content to add** (use this text):
```markdown
## Why Domain Knowledge Beat Machine Learning

We tested ML extensively:
- **Graph Neural Networks**: D-MPNN scored **−0.309** on ei (0/5 folds). Literature shows message passing needs ~859-1,000 rows; we have 222.
- **Self-supervised pretraining**: 9 variants tested on PI1M and smile_r3 corpora. **All ≤ supervised control**. Decisive: MLM probe 0.651 vs random-init 0.708 - pretraining *destroyed* task-relevant structure at our scale.
- **ChemBERTa** (research paper, out-of-competition): frozen 0.751, fine-tuned 0.784 vs tree baseline **0.810**.

**The pattern**: 6 of 9 winners are domain knowledge (physics identities, ionic decomposition, per-target design), 2 are assembly discipline, 1 is representation. **Not a single winner is a neural architecture.**

This aligns with Grinsztajn et al. (NeurIPS 2022): tree ensembles outperform deep learning on tabular data at our sample sizes.
```

---

## 3. REPORT REFINEMENT

### 3.1 Scientific Contributions Verification
**Task**: Ensure report clearly states research contributions beyond dataset analysis.

**Check** `Personal/Midnight_Report/Sample Reports/` for what judges expect:
- Understanding of polymer science ✓ (we have this in Personal/docs/02_domain/)
- Novel methodology contributions
- Negative results with scientific value

**Action**:
- [ ] Review `Personal/docs/00_My Docs.md` for research contribution section
- [ ] Ensure report §2 includes: "We measured trust rather than asserting it" as scientific contribution
- [ ] Highlight in report: First to show **load-bearing attributions** (0.851 vs 0.043) AND **attribution invariance** (cosine 0.95-0.99) together
- [ ] Mention measured Flory-Fox recovery (unsupervised physics learning)
- [ ] Honest negative results (4 scorecard FAILs with causes) as research contribution

### 3.2 Max Cap Appendix Verification
**Task**: Verify or remove Appendix B (ceiling analysis).

**File**: Check if `Personal/Midnight_Report/PROMPT_10PAGE.md` §E matches `Personal/docs/06_results/ceiling_analysis.md`

**Verify**:
- [ ] Metric identity (0.9023 mean vs 0.9370 pooled)
- [ ] Tg-alone bound (perfect Tg → mean 0.9172)
- [ ] Per-target standard errors (ei 0.022, eps 0.024...)
- [ ] Single-row leverage calculations
- [ ] Label-noise bound on Tg (σ=15°C → R²max 0.981; empirical ≈0.92)
- [ ] Composite ceiling ≈0.93 ± 0.01
- [ ] Math is CORRECT

**If ANY math is uncertain**: Remove Appendix B entirely or mark as "preliminary estimate" with caveats.

---

## 4. PRESENTATION & WEBSITE

### 4.1 Website 3D Visualization
**Current state**: `Website/static/` has only `.gitkeep` - NO actual 3D viz.

**Requirement analysis**:
- Demo works (Streamlit app exists, functional)
- 3D structure viz mentioned in user request but...
- **RDKit already renders 2D** in `app.py` (mol_png function)
- Time-constrained hackathon prep

**Recommendation** (CHOOSE ONE):

**Option A** (Quick, Safe):
- [ ] Keep current 2D RDKit rendering
- [ ] Add note in `Website/README.md`: "Structure displayed as 2D for compatibility; 3D available via RDKit Chem.MolTo3DBlock if needed"
- [ ] Focus energy on demo rehearsal

**Option B** (If time allows):
- [ ] Add 3D.js integration for structure viz
- [ ] Modify `app.py` to include 3D toggle
- [ ] Test thoroughly before hackathon

**User decides**: Given time pressure, Option A recommended.

### 4.2 Presentation Enhancement with Research
**Task**: Integrate NeurIPS paper (Grinsztajn et al. 2022) into story as science backing.

**Paper**: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

**Already cited as G1 in** `Personal/Research/INDEX.md`

**Action**:
- [ ] Update `Personal/STORY.md` Act 2: Add "...and why: Grinsztajn et al. (NeurIPS 2022) showed trees outperform deep learning on tabular data at our sample sizes"
- [ ] Slide 4 (Architecture): Add footer citation "Architecture choice grounded in Grinsztajn et al., NeurIPS 2022"
- [ ] Report §3: Cite G1 when discussing classical ensemble choice
- [ ] Add to QnA: "Why trees?" → Reference G1

---

## 5. DOCUMENTATION & QNA CONSOLIDATION

### 5.1 Master QnA File Creation
**Current state**: QnA scattered across 10 files in `Personal/docs/11_qna/`

**Task**: Create ONE master file `Personal/docs/11_qna/MASTER_QNA.md` with ALL questions consolidated.

**Structure**:
```markdown
# MASTER QNA — Every Question a Judge Might Ask

**Usage**: Read this 30 minutes before presentation. Each answer is 30-60 seconds with the KEY NUMBER.

## SECTION 1: HOSTILE QUESTIONS (from hostile.md)
[Copy all 10 H-questions with answers]

## SECTION 2: ARCHITECTURE & DESIGN
[Consolidate from architecture.md]

## SECTION 3: DATA & EDA
[From data_and_eda.md]

## SECTION 4: EXPLAINABILITY
[From explainability.md]

## SECTION 5: ROBUSTNESS & INVARIANCE
[From robustness_invariance.md]

## SECTION 6: GENERALIZATION & UNCERTAINTY
[From generalization_uncertainty.md]

## SECTION 7: METRICS & STATISTICS
[From metrics_statistics.md]

## SECTION 8: PHYSICS & DOMAIN
[From physics_domain.md]

## SECTION 9: PROCESS & TOOLING
[From process_and_tooling.md]

## SECTION 10: WHY NOT X?
[From why_not_x.md]

## QUICK REFERENCE: THE NUMBERS
- Private LB: 0.891
- Fidelity: 0.851 vs 0.043
- Physics: R² 0.9716, 0/134 violations
- [All canonical numbers from 00_INDEX.md]
```

**Action**:
- [ ] Create `Personal/docs/11_qna/MASTER_QNA.md`
- [ ] Consolidate all 10 files
- [ ] Remove redundancies
- [ ] Add cross-references: "See Architecture §2 for details"
- [ ] Update `Personal/QNA.md` to point to MASTER file

### 5.2 Architecture Decision Documentation
**Task**: Ensure every major decision has clear "why" with measurement.

**Verify in** `Personal/docs/05_architecture/design_decisions.md`:
- [ ] Per-target lanes: because 98% structure overlap + 12.3% Tg overlap = two problems
- [ ] Physics identities: measured R² 0.9716, residual LOO −0.82
- [ ] NNLS assembly: negative weights don't transfer
- [ ] Calibration layer: diagnosed from parity plots (mid-band compression)
- [ ] No GNN: ei −0.309, literature crossover argument
- [ ] No SSL: MLM 0.651 vs 0.708

**If ANY decision lacks measurement backing**: Add reference to experiment log.

---

## 6. CHARTS & VISUALIZATION GAPS

### 6.1 Missing Charts Verification
**Per RUN.md**: Many `outputs/` folders empty until notebook runs.

**Critical charts needed for presentation/report**:
```
outputs/eda/novelty_two_regimes.png (Slide 3)
outputs/architecture.png (Slide 4, README)
outputs/explainability/shap_beeswarm_tg.png (Slide 6)
outputs/robustness/smiles_invariance_boxplot.png (Slide 6)
outputs/generalization/generalization_ladder_plot.png (Slide 7)
```

**Action**:
- [ ] Run notebook (Step 3 in RUN.md) to generate all charts BEFORE hackathon
- [ ] Verify all presentation slides reference existing charts
- [ ] Capture demo screenshots (Website/screenshots/)
- [ ] Generate architecture diagrams (Step 2 in RUN.md)

**Timeline**: Do this 24-48 hours before hackathon, not last minute.

---

## 7. STORY & NARRATIVE POLISH

### 7.1 Enhance STORY.md with Research Backing
**Current**: Good 5-6 min narrative in 4 acts.  
**Enhancement**: Add scientific grounding per user request.

**File**: `Personal/STORY.md`

**Actions**:
- [ ] Act 1: After variance trap, add: "This asymmetry is documented in the competition metric design"
- [ ] Act 2: After GBM choice, add: "Grinsztajn et al. (NeurIPS 2022) showed why: trees preserve feature combinations that deep nets smooth away on tabular data"
- [ ] Act 2: After physics beats ML, add: "Maxwell's relation for dielectrics, validated on 134 polymers with zero violations"
- [ ] Act 3: Add forward reference to Flory-Fox: "A 1950 polymer physics relation, emerged without being taught"

### 7.2 Presentation Flow Verification
**File**: `Personal/Presentation/SLIDE_PLAN.md`

**Verify**:
- [ ] 9 slides, ~35s each = 5-6 minutes ✓
- [ ] Slide 1: Title + 3-box (f1/f2/f3) ✓
- [ ] Slide 2: Problem + gap (invariance measured, not asserted) ✓
- [ ] Slide 3: TWO-REGIMES finding (THE KEY SLIDE) ✓
- [ ] Slide 4: Architecture with physics R² on diagram ✓
- [ ] Slide 5: Experiments + FAILURE STRIP ✓
- [ ] Slide 6: Explainability (0.851 vs 0.043) ✓
- [ ] Slide 7: Generalization + private score prediction ✓
- [ ] Slide 8: DEMO (4 beats: baseline, aromatic, rewrite, out-of-domain) ✓
- [ ] Slide 9: Results + ceiling + future ✓

**Missing from slides but should add**:
- [ ] Slide 5: Ensure ML failures (GNN, MLM) are in failure strip
- [ ] Backup slides: Include ML vs domain knowledge comparison table

---

## 8. FINAL VERIFICATION CHECKLIST

### 8.1 Pre-Hackathon Run Checklist
**From RUN.md, execute in order**:

- [ ] Step 1: Environment setup (setup.sh) - 5 min
- [ ] Step 2: Architecture diagrams - 10 sec
- [ ] Step 3: Notebook run - 25 min (DO THIS!)
- [ ] Step 4: Verify submission.csv - 10 sec
- [ ] Step 6: Demo screenshots - 20 min
- [ ] Step 9: Release gate scan (no forbidden terms)

**DO NOT DO** (optional, time-consuming):
- [ ] Step 5: Full regeneration (2.5-3h) - only if time allows and needed

### 8.2 Documentation Consistency Check
**Run these greps**:
```bash
# Ensure no inconsistent scores
grep -r "0\.903[0-9]" Personal/ | grep -v "0\.9023"
grep -r "0\.902[0-9]" Personal/ | grep -v "0\.9023"

# Ensure ML story is complete
grep -r "supervised control" Personal/
grep -r "0\.651.*0\.708\|0\.708.*0\.651" Personal/

# Verify all figures referenced exist (after notebook run)
grep -r "\.png)" Personal/ AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/*.md | \
  cut -d: -f2 | grep -oE '\(([^)]+\.png)' | sort -u
```

### 8.3 Presentation Readiness
**Rehearsal checklist**:
- [ ] Slides generated from PROMPT_PRESENTATION.md
- [ ] Demo site tested offline (WiFi OFF)
- [ ] Screenshots captured as backup
- [ ] Speaker notes reviewed (SPEAKER_NOTES.md)
- [ ] Demo script practiced (DEMO_SCRIPT.md) - 45 seconds, 4 beats
- [ ] Hostile questions rehearsed (hostile.md - top 10)
- [ ] Numbers memorized: 0.891, 0.9023, 0.851 vs 0.043, 0.9716, 0/134, 0.95-0.99

---

## 9. PRIORITY IF TIME-CONSTRAINED

**If you have < 4 hours**, do ONLY:

1. **CRITICAL** (2h):
   - Run notebook (Step 3 RUN.md) - generates all charts
   - Create MASTER_QNA.md
   - Verify all scores are 0.9023 (not variants)
   - Run release gate scan

2. **HIGH** (1h):
   - Add ML comparison summary to key files
   - Generate architecture diagrams
   - Capture demo screenshots

3. **MEDIUM** (1h):
   - Enhance STORY.md with research citations
   - Verify max cap appendix or remove
   - Test demo site

**SKIP IF TIME-PRESSED**:
- 3D website visualization (current 2D sufficient)
- Full regeneration (frozen submission valid)
- New experiments (NONE needed)

---

## 10. KNOWN ISSUES & WARNINGS

**From PLAN.md and RUN.md**:
1. **Python 3.11.7 is LOAD-BEARING** - Do NOT run on 3.12
2. **outputs/** folders empty until notebook runs
3. **Scorecard may change from 14/18 to 14-15/19** after full run - update everywhere if it does
4. **No "oracle" term in codebase** - disqualification risk
5. **Dataset symlink** - points to Consolidation/00_competition/dataset
6. **ChemBERTa result is OUT-OF-COMPETITION** - always clarify this

---

## 11. SUCCESS CRITERIA

**You are DONE when**:
- [ ] All numbers consistent (0.9023 everywhere)
- [ ] ML baseline story clear in 3+ places
- [ ] MASTER_QNA.md exists with all 80+ questions
- [ ] Notebook has run (charts generated)
- [ ] Demo rehearsed and screenshots captured
- [ ] No "oracle" in codebase
- [ ] Report and presentation prompts validated against sources
- [ ] All referenced charts exist
- [ ] User has reviewed and approved

---

## 12. HANDOFF TO USER

**After completing this refinement**:
1. Generate report: `Personal/Midnight_Report/PROMPT_10PAGE.md`
2. Generate presentation: `Personal/Presentation/PROMPT_PRESENTATION.md`
3. Final review with user
4. Execute RUN.md Steps 1-4, 6, 9
5. Rehearse presentation with DEMO_SCRIPT.md
6. **Go win the hackathon!**