# Round 2 (Polymer Property Prediction) — Experiment-History Analysis

Source: READ-ONLY analysis of `/tmp/r2dump` (logs, research, final_submissions,
submissions, experiments/final_submission_runs, experiments/ORACLE_ASSISTED_RESEARCH_ONLY).
No files were modified.

Competition identity: "Polymer Prediction Challenge Round 2" / Kaggle competition
source `aisehack-2-0`. Objective throughout: an official-data-only, single-notebook
pipeline reaching an unweighted **seven-target mean R² of 0.95**, with **0.93** as
the intermediate milestone. Seven targets: `tg`, `egc`, `egb`, `ei`, `eea`, `nc`, `eps`.

---

## 1. FINAL RESULTS (public leaderboard)

**Important accuracy note:** this dump contains **no Round 2 final leaderboard**
beyond a single early observation. The value "0.891 non-archive" that appears in the
task prompt is **NOT present anywhere in this dump**; I found no public/private
with_archive or without_archive final scores and no competing-team scores. What the
dump actually records:

| Observation | Value | Source |
|---|---|---|
| C001 public score (user-reported, 2026-08-03) | **0.8590000000** | `logs/LEADERBOARD_LOG.md`, `logs/leaderboard_observations.jsonl` |
| C001 frozen proxy expectation | 0.8560283011 (4,905/4,940 coverage) | same |
| C001 clean OOF mean | 0.8658425762 | same |
| public − proxy | +0.0029716989 | same |
| public − clean OOF | −0.0068425762 | same |
| gap to 0.93 goal | 0.0710000000 | same |
| Prior public incumbent ("0.916 barrier") | **0.916** (test-side proxy 0.9084) | `research/second-wave-representation-research-20260805.md`, `Fable_prompt.md`, `per_target_best_leaderboard.json` |

- C001 candidate: `submissions/Sandman_ppp_round2_initial_reference_20260803.csv`,
  SHA-256 `55eabfa7933765aeff8cf0d6ed9da758a39864569cc59ba216afe42722bfc4a1`.
- C001 was the only submission in `LEADERBOARD_LOG.md`; the log explicitly states
  "No submission has been made by an agent or command from this workspace" and that
  the 0.859 was reported by the user after an **external** submission.
- The **0.916** figure is the pre-existing user-submitted incumbent (the "Breaking
  the 0.916 Barrier" reset target). It is the artifact that the clean C260 compound
  and the V1 "per-target best" compound were numerically identical to.
- **Private LB scores: none present.** **Competing-team scores: none mentioned.**
- The only other "submission-adjacent" records are Kaggle kernel packaging metadata
  under `kaggle_push_20260810/` for `Sandman_Version_50_8th_Aug_with_archive` and
  `Sandman_Version_52_8th_Aug_without_archive` (`is_private: true`, CPU-only,
  internet off) — packaging only, with no recorded Kaggle score.

---

## 2. BEST EXPERIMENTS (final-submission package and per-target bests)

### 2.1 The delivered package (Versions 46–57)

**Notebook-backed strict one-run finals (2026-08-08, `final_submissions/`):**

| Version | Branch | verified mean R² | proxy mean R² |
|---|---|---|---|
| V46 | with_archive | 0.914066206676 | 0.908105904640 |
| V47 | with_archive | 0.911124257767 | 0.904964306421 |
| V48 | without_archive | 0.835736287443 | 0.834550521581 |
| V49 | without_archive | 0.833579564738 | 0.832521108644 |

**Notebook-backed V50–V53 (the "Sandman Version 50-57" set, 2026-08-08; these are
the arithmetic leaders assembled from target splices):**

| Version | Branch | Source experiment | verified mean R² | proxy mean R² |
|---|---|---|---|---|
| V50 | with_archive | R2-C1579 (archive combined target-leader) | 0.934272624216806 | 0.928442303497511 |
| V51 | with_archive | R2-C1577 (archive current-only EI/EPS arms) | 0.934271971645240 | 0.928441650925945 |
| V52 | without_archive | R2-C1572 (noarchive Egc splice over C1567) | 0.902756451432471 | 0.901683132031365 |
| V53 | without_archive | R2-C1570 (noarchive joint-physics grid) | 0.902755125174725 | 0.901681805773619 |

**CSV-only oracle-assisted hybrids V54–V57 (2026-08-09, targetwise tail/shrinkage
+ character-TFIDF residual selection; no notebook generated):**

| Version | Branch | Over | verified mean R² | proxy mean R² | Δ vs base |
|---|---|---|---|---|---|
| V54 | with_archive | V50 | 0.934694027625333 | 0.928863706906037 | +0.000421403409 |
| V55 | with_archive | V51 | 0.934693180142146 | 0.928862859422850 | +0.000421208497 |
| V56 | without_archive | V52 | 0.904148817822538 | 0.903045752701492 | +0.001392366390 |
| V57 | without_archive | V53 | 0.904149561414815 | 0.903046496293769 | +0.001394436240 |

V54/V55 selection map: `tg=base`, `egc/egb/eea/nc/eps=char-residual`, `ei=mild-spread`.
V56/V57 map: `tg/egc/egb/nc/eps=char-residual`, `ei/eea=mild-spread`.

### 2.2 Per-target best R² (`research/per_target_best_leaderboard.json`)

Source = clean C260 compound (identical to the 0.916 public artifact), the
`INCUMBENT_R2-BEST-DEFENSIBLE-COMPOSITE-SUBMISSION`:

| Target | clean OOF R² | post-freeze proxy R² | post-freeze verified R² | verified rows |
|---|---|---|---|---|
| tg | 0.9088768072 | 0.9591877550 | 1.0 | 1,641 |
| egc | 0.9115043879 | 0.9627557988 | 0.9627557988 | 1,352 |
| egb | 0.9221467344 | 0.9353070670 | 0.9353070670 | 224 |
| ei | 0.8454440895 | 0.8168122947 | 0.8168122947 | 148 |
| eea | 0.9008357940 | 0.9416769211 | 0.9416769211 | 147 |
| nc | 0.8397322432 | 0.8977058633 | 0.8977058633 | 153 |
| eps | 0.7835054390 | 0.8451876725 | 0.8451876725 | 153 |
| **compound** | — | **0.9083761961** | **0.9142065168** | coverage 0.7728744939 |

`best_component_registry.yaml` (oracle-first, 2026-08-08) later registers per-target
leaders that exceed these: archive `tg=1.0`, `egc=0.9629498648`, `egb=0.9458960909`,
`ei=0.8699895477`, `eea=0.9542465920`, `nc=0.9198719897`, `eps=0.8869542844`
(via C1577/C1579); noarchive leaders `tg=0.9017902573`, `egc=0.9089264190`,
`egb=0.9294798840`, `ei=0.8699895477`, `eea=0.9138271650`, `nc=0.9083276028`,
`eps=0.8869542844` (via C1535/C1572).

### 2.3 Diagnostic ceilings (oracle-assisted, NOT submission-eligible)

- Archive signed-source diagnostic: C1588 = **0.9407446681** verified; C1596 =
  **0.9462132000** verified (11,795 non-oracle sources); C1580 (splicing the
  oracle-assisted noarchive C1565 weak arms) = **0.9506182266** verified / 0.9447879059 proxy.
- Noarchive signed-source diagnostic: C1565 = **0.9506018086** verified / 0.9447715416
  proxy (crossed 0.93 and 0.95 in the diagnostic lane only).
- These depend on oracle-selected source/alpha choices and prior oracle-assisted
  source files (C660/C623/C1441/C1473); anti-oracle replay audits found **0
  clean-replayable source rows**, so none is promotable to a clean notebook/submission.

---

## 3. BLEND SWEEP RESULTS (from final_submission_runs manifests + oracle summaries)

Sweep manifests are `schema_version ppp.round2.c355.target-blend-sweep.v1`; each row
records base, source, `target`, `weight_on_source`. Scores live in the paired
`-scores.jsonl` / `-summary.json` under `ORACLE_ASSISTED_RESEARCH_ONLY`.

### C378 — archive bounded portfolio sweep (post-C369)
- Parameters: 22 predeclared branch-local sources; targets `ei,eps,nc,egb,eea,tg`;
  weights `0.025, 0.050, 0.075, 0.100, 0.125, 0.150`; **522 frozen CSVs**.
- Best single candidate: `egb w=0.075` over `F06-PI1M-with_archive` → verified
  **0.9251298123** / proxy 0.9192994916.
- Target winners banked: EEA 0.9479338810, EGB 0.9433392070, EI 0.8548806206,
  EPS 0.8603747533, NC 0.9074457229 (Tg worse → not banked).
- → C379 assembled = verified **0.9252471405** / proxy 0.9194168198 (staged V27).

### C393 — archive fine-source blend (post-C379)
- Parameters: 6 non-Tg targets; **1,125 CSVs**.
- Best single candidate: `eps w=0.125` over `C388-ARCHIVE-WEAK-TARGET-MODEL-ZOO-COMPACT`
  → verified **0.9254827518** / proxy 0.9196524311.
- Target winners: EGB 0.9433413804, EI 0.8553013532, EEA 0.9480932614,
  NC 0.9074927846, EPS 0.8620240321.
- → C395 assembled = verified **0.9255726587** / proxy 0.9197423379 (staged V28).

### C394 — noarchive fine-source blend (post-C377)
- Parameters: all 7 targets; **1,455 CSVs**.
- Best single candidate: `egb w=0.200` over `F06-PI1M-without_archive` → verified
  **0.8831852996** / proxy 0.8821233718.
- Target winners: Tg 0.8991124313, Egc 0.9040882668, Egb 0.9207815030,
  EI 0.8233626005, EEA 0.9054358105, NC 0.8847655137, EPS 0.8500092567.
- → C396 assembled = verified **0.8839364832** / proxy 0.8828639711 (staged V35).

### Follow-on micro-sweeps (bookkeeping gains only)
- C403 (noarchive 6 coordinate sweeps, 98 CSVs) → C404 verified 0.8843203461.
- C405 (archive fine-weight sweeps, 106 CSVs) → C406 verified 0.9255755494.
- C414–C433 reflected-source blends lifted noarchive C404→C433 to 0.8862028368
  (final EPS winner via reflected C402 at w=0.075 = 0.8568736127).
- C423–C428 reflected blends lifted archive C413→C428 to 0.9257418512.
- C355/C358/C368 archive EI/EPS/NC blend sweeps drove C334→C369 (0.9191921045 proxy /
  0.9250224252 verified); C356/C370 noarchive sweeps drove C351→C371
  (0.8811770185 proxy / 0.8822403849 verified).

---

## 4. TIMELINE / NARRATIVE

### 2026-08-03 — bootstrap and first public score
- EDA + Round-1 distillation; C000 failed (RDKit float32 overflow); C001 (deterministic
  sanitizer) completed. Clean OOF per-target: Tg 0.908877, Egc 0.911504, Egb 0.922147,
  Ei 0.806897, Eea 0.888235, Nc 0.839732, EPS 0.783505; mean 0.865843. 2,445/4,940
  rows are exact current/archive overrides.
- User reports C001 public **0.859** (+0.002972 vs proxy).
- C050 "mixed gap-components" became the clean incumbent (0.8731493565 mean).

### 2026-08-04 — first wave (many small residuals)
- C025–C049: per-target residual attempts; most regressed. Big clean wins were
  physical-coordinate routes: Eea Flory–Fox C189 (0.9008→0.9163), ionic EPS
  C190/C214 (0.7835→0.8501), selected-EPS→Nc ionic C252 (0.8397→0.8832), guarded
  Egc C207 (0.9115→0.9221), guarded Ei C199 (0.8454→0.8567).
- C111/C113 oracle first-portfolio diagnostics: 0.869075 / 0.870955 verified (oracle-assisted).

### 2026-08-05 — clean component audit loop; 0.916 barrier reset; Fable engines
- Deterministic audit-only assemblers C209–C257 raised clean compound
  0.8731 → 0.8852 → **0.8941972740** (C257). Selected targets: Tg/Egb/Nc on C050,
  Egc=C207, Ei=C199, Eea=C189, EPS=C190/C214.
- User resets goal to "Breaking the 0.916 Barrier"; the prior public incumbent is 0.916.
- Fable engines (F01 chain / F02 ionic / F03 polymer-genome):
  - F01 A3 shift-matched: ei 0.835230 (+0.113525), eea 0.902779, egb 0.923908; archive-enabled
    ei 0.890515 / eea 0.925451 / egb 0.949574. Materializing F01 failed transfer (proxy 0.882266).
  - F02 B2: eps 0.839768, nc 0.864697 (NC failed its ≥0.93 partner-observed kill → F03 not chained).
  - F03 mean 0.800624; rejected.
  - F04 GPR (0.720538 proxy), F05 multitask (0.714488), F06 PI1M (archive 0.813288 /
    noarchive 0.753491 proxy) all rejected.

### 2026-08-07 — dual-branch local loop resumes; archive/noarchive split
- Rescored archive V2 (proxy 0.9083761961 / verified 0.9142065168) and noarchive V3
  (0.8117214458 / 0.8128479612).
- Built F10 portfolios (archive 0.9148605203; noarchive 0.8560804191, hitting the 0.85 milestone).
- F11 noarchive 0.8565410956; F12 archive weak-target equal ensemble 0.9172547190;
  F14 noarchive 0.8654538643; F17 0.8654881501; F18 0.8672680895; F19 0.9173440140;
  F20 0.9212236322; F21 0.8676981425; F22 0.9212236322 (C270-free);
  F23 cross-property 0.9222572912; F24 0.8692701162; F25 ionic co-test 0.9229779035;
  F26 0.8712399659. Dual-0.93 gate still unmet.

### 2026-08-08 — the main compound/splice/sweep marathon (C291→C1596, C900+/reflect era)
- C291–C322 target-wise compound loop; C312 noarchive 0.8774919626; C327/C331–C345
  co-test meta-calibrators; C334 archive 0.9239797059; C343 noarchive 0.8786849906.
- C346–C351 joint EPS/NC consistency (train-side ionic OOF was strong — 0.949/0.927 —
  but full-candidate transfer often regressed).
- C355–C375 target-blend + EHT: archive C369 0.9250224252; noarchive C375 0.8823683935.
- **C376–C379** (the APPEND_NOTE record): C376 target-leader scan; C377 noarchive
  = 0.8823759497 (V34); C378 522-CSV archive sweep; C379 archive = 0.9252471405 (V27).
- C393/C394/C395/C396 fine-source blends (archive 0.9255726587, noarchive 0.8839364832).
- C401–C433 second-order microblends + "reflected source" (2×base − source) blends:
  archive C428 0.9257418512; noarchive C433 0.8862028368.
- Later same-day the loop re-indexed to a C539 leader (0.9303381375) and the "C900+"
  clean current-only B3/identity-grid era: noarchive C1230 0.9008992081 → C1284
  0.9011820382 → C1319 0.9012519846 → C1349 0.9012766081; archive C1360 0.9312645729
  → C1364 0.9312854114 → C1427 0.9315773363 → C1449 0.9316574237 → C1495 0.9316991456
  → C1523 0.9317400396 → C1529 0.9319756556 → C1534 0.9320703730 → C1544 0.9320960229
  → C1569 0.9320965056 → C1573 0.9321257115 → C1577 0.9342719716 → **C1579 0.9342726242**
  (the V50 base). Noarchive: C1496 0.9026539633 → C1532 0.9026836218 → C1535
  0.9027530659 → C1572 0.9027564514 (the V52 base).
- C1580 diagnostic ceiling 0.9506182266; C1581–C1596 signed-source diagnostics
  (0.9407446681 → 0.9462132000) confirmed source diversity but stayed non-promotable.
- **C984** (noarchive PI1M full-model bank) stalled after >12 CPU-hours and was SIGTERMed.
- F01 chain materialization (C1363) failed badly (0.8952605632 vs C1360 0.9312645729).

### 2026-08-09/10 — final packaging and Kaggle staging
- Targetwise tail/shrinkage hybrids V54–V57 built and scored (Section 2.1).
- Kaggle kernel metadata for V50 (with_archive) and V52 (without_archive) created
  (private notebooks, no internet/GPU). No Kaggle score recorded in the dump.

**Bottom line:** the archive branch reached ~0.934 verified (clean arithmetic leader
C1579/V50) and ~0.9506 only in the oracle-assisted diagnostic lane; the noarchive
branch reached ~0.9028 verified clean and ~0.9506 only oracle-assisted. The dual
0.93/0.95 gates were **never met by a clean, notebook-eligible pipeline**.

---

## 5. KEY LESSONS

### Biggest durable gains (clean, transferable)
1. **Physical-coordinate / chemistry-motivated features**, not generic descriptors:
   Flory–Fox/oligomer Eea (+0.0154), ionic-coordinate EPS (+0.0666 clean OOF),
   selected-EPS→Nc ionic projection (+0.0434), guarded Egc/Ei transfer.
2. **Cross-property overlays and joint EPS/NC "co-test" consistency** on the archive
   branch (F23/F25, C332/C333) were the single most reliable archive lever
   (0.9223 → 0.9230 → 0.9239).
3. **Target-wise splicing / compounding**: assembling only the target each candidate
   improved (never mixing whole CSVs) produced steady monotone, if small, gains.
4. **Post-freeze local-oracle scoring discipline**: every candidate frozen and hashed
   before oracle read; the proxy panel (99.3% coverage) tracked public LB to ~0.35%
   error (C001), making it a trustworthy early signal.

### Repeated failures / cooled families
1. **PI1M self-supervised representation learning** failed every time: C010, C119,
   C131, C157, C158, C169, C181, C185, C261 (0/7 targets positive vs control), F06,
   C284/C285 SVD, C391 model-zoo, C984 (stalled), C1445 distillation. Repeatedly
   cooled; only the F06 EGB component ever banked as a blend source.
2. **OOF→test transfer failures** were the dominant theme: many candidates passed
   clean OOF gates (e.g., C346 Egc/EPS, C366 EPS +0.0526 OOF, C1576 ridge
   +0.00246 OOF, F01 chain +0.1688 EI OOF) yet regressed on the frozen test candidate.
3. **Direct weak-target model-zoo / stacker replacement** (C388, C391, C1502, C1536–C1539)
   collapsed transfer; train OOF "is not predictive enough for this test distribution."
4. **Identity/physics overlays at low/mixed weights** (C1503–C1506) and **NC-only
   physics** (C325/C326) were negative.
5. **Oracle-assisted signed-source diagnostics cannot be promoted**: anti-oracle replay
   found 0 clean-replayable source rows (C1444, C1489/C1490), so the 0.938–0.951
   diagnostic ceilings were never convertible to a clean notebook.

### What worked but only at bookkeeping scale
- **Blend/splice micro-sweeps** (weights 0.002–0.6) and **reflected-source blends**
  (2×base − source) yielded +0.000001 to +0.0015 mean R² — real but not path-to-0.93.

### Left unexplored / unresolved
- A clean conversion of the noarchive weak-target arms (Ei/Nc/EPS are the bottleneck:
  noarchive clean stuck ~0.90 vs 0.93 gate; archive clean stuck ~0.934 vs 0.95).
- The ~0.9506 diagnostic ceiling was reached only by splicing oracle-assisted noarchive
  weak arms (C1580) — never reproduced cleanly.
- The "0.891 non-archive" figure in the prompt is unverifiable from this dump.

---

## 6. ROW COUNTS / DATA FACTS

| Item | Value | Source |
|---|---|---|
| Current train rows | **7,409** | EDA, research-log, second-wave doc |
| Current test rows | **4,940** | EDA, all manifests |
| Archive (Round 1) train rows | **6,171** | EDA (`3722/6171` verbatim in current train) |
| Archive test rows | **4,115** | EDA, research-state |
| Alternate row count user reported | 4,497 (no official file found) | research-state |
| Verified oracle coverage | **3,818/4,940** (77.29%); 1,122 null, all Tg | EDA, research-state |
| Proxy diagnostic coverage | **4,905/4,940** (99.29%); 35 unresolved, all Tg | EDA, research-state |
| Exact current/archive label overrides | **2,445/4,940** | EXPERIMENT_LOG |
| Archive test rows also in current test | 1,666/4,115 | EDA |
| PI1M | ~1M polymers (paper); local supplied copy only; experiments used 49,834–200,000 hash-ranked subsets | Fable_prompt, research-state |
| Per-target train/test | tg 4143/2763, egc 2028/1352, egb 337/224, ei 222/148, eea 221/147, nc 229/153, eps 229/153 | EDA |
| Scores in experiments.jsonl | 562 records (107 result, 97 allocation, 36 incident, …) | experiments.jsonl |
| oracle_scores.jsonl | 3 records (C001, C111, C113) | logs |

---

*End of report. Prepared read-only from `/tmp/r2dump`; all scores quoted verbatim from
source JSON/Markdown files.*
