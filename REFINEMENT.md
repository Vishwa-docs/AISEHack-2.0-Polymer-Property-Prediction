# REFINEMENT.md — Final Hackathon Preparation (improved)

**Audience:** a coding + documentation agent (you) doing the final polish, and the human (for the
decision points flagged `🔴 USER DECIDES`).
**Scope:** finalise codebase, report, presentation, story, QnA and docs. **No new score
experiments** — the model is frozen. This is consistency + narrative + readiness work.
**Deadline:** 3 September 2026 · **2 final submissions** · team **Sandman**.

---

## 0. READ THIS FIRST — the score story is broken, and it is the whole game

The previous version of this file treated the score as "done at 0.90680" and told you to "enforce
0.9023 everywhere". **That is wrong on both counts.** Here is the actual state, verified from the
files on disk:

| label | value | where it lives | what it is |
|---|---|---|---|
| **old champion** | **0.90230** | top of the historical table | the V57 ensemble + physics, **no GNN**. This is what was actually submitted → **private LB 0.891, public 0.917** |
| **Phase 7 (first annotation)** | **0.90680** | every doc header (`CONTEXT.md`, `README.md`, `00_INDEX.md`, `RESULTS.md`, …) | V57 **blended with a GINE GNN**. The top-level `codebase/submission.csv` currently contains **this** file (SHA `30141c6a…`) |
| **Phase 7 (final, verified)** | **0.907551** | `codebase/904_submission/submission_final.csv` (SHA `cd91f278…`), `logs/score_final_verified.json` (`mean_r2 0.90755055`), `notebook/STATUS.md` | the same GNN blend with the repaired ei/eea leaves. **Not yet propagated anywhere** — not into the docs, not into the top-level `submission.csv` |
| verified sub-panel | **0.90813** | `00_INDEX.md`, `CONTEXT.md` | the same predictions scored on the 3,818 exactly-matched rows |
| estimated private (Phase 7) | **0.8965** | `score_final_verified.json` → `estimated_private_lb 0.89655055` | what the 0.907551 file would ≈ score privately, versus 0.891 now |
| competitor | **≈0.92** | `docs/01_task/competition.md:45` ("A competitor at 0.92 exists on the Round-3 board"), `Score_and_Invariance_Improvement/CEILING_REALITY_CHECK.md:63` | a competitor is at ~0.92 on the board. `beats_092_competitor: false` in `score_final_verified.json` |

**Three separate documents are currently telling three different stories:**
- doc **headers** say the current best is **0.90680**,
- the doc **bodies** (report draft, presentation deck, website footer, `00_INDEX.md` canonical block, `RESULTS.md` §1–2) still quote **0.9023** as current,
- the **actual best verified artifact** is **0.907551**, which appears in **no** doc and is not the top-level `submission.csv`.

A quantified scan of the docs found **13 distinct spellings of three different quantities**
(`0.9023` ×97, `0.90680` ×59, `0.90230` ×45, `0.9066` ×9, `0.90813` ×8, `0.9028` ×7, plus
`0.90352 / 0.903480 / 0.902289 / 0.90229 / 0.9035`). If a judge asks "what did you score?" and the
answer is not one clean number with one clean story, the trust narrative the round is judged on
collapses on stage.

### 0.1 The GNN-blend nuance you must get right in every artifact

The Phase 7 gain is **+0.0045 to +0.0053 over 0.9023**, and the mechanism is **decorrelation, not
neural superiority**:

> A GINE message-passing network (3 seeds/target, structure-grouped CV) is blended per target at
> `w = clip((cv − 0.80) / 0.25, 0.10, 0.60)`, with `cv` the target's **cross-validated** score on
> `train.csv` only. The GNN is **not better alone** (tg GNN 0.8987 vs ensemble 0.8954) — it makes
> **different errors**, and blending two decorrelated predictors is what lifts the mean.

**This changes two claims the previous REFINEMENT.md made that are now false:**

1. ❌ *"Not a single winner is a neural architecture."* → The Phase-7 winner **is** a tree+GNN
   blend. The correct claim: *"trees still dominate every target; the single neural component
   contributes only through decorrelation, at a weight ≤ 0.60 and a blend weight of 0.10 on the
   weakest targets."* This is actually a **stronger, more honest** story than "we rejected deep
   learning", and it stays consistent with the G1 citation (trees over deep nets on tabular data).
2. ❌ *"0.894 → 0.896 (random → grouped): minimal drop proves dedup works."* → That ladder entry is
   a **+0.002 rise**, not a drop, and it is within noise. The honest sentence is: *"grouped CV is
   within noise of random CV (0.896 vs 0.894), i.e. deduplication costs nothing in the
   interpolation regime; the real cost appears at the scaffold split (0.825) and beyond."*

### 0.2 The platform-defect finding (also new, also must be propagated)

Phase 7 established that the long-standing *"python 3.11.7 is load-bearing"* claim is **wrong**.
`rdEHTTools.RunMol` **segfaults on linux-x86_64** and works on macOS with **identical** pins — a
**platform** defect, not a version one. Only **ei and eea** depend on it; the other five targets
reproduce at correlation **1.00000** on both. Kaggle runs Linux, so this matters for reproduction.
Wherever a doc currently says "python 3.11.7 is load-bearing", correct it to this platform story
(source: `904_submission/RESUME_HERE.md`, `CONTEXT.md` §"Also established", `00_INDEX.md` header).

---

## 1. 🔴 Decisions the human must make before you change any number

You cannot propagate a canonical score until the human answers these. **Ask, then do the
propagation in §2 in one pass.**

| # | decision | context / recommendation |
|---|---|---|
| **D0** | **What is the final submission?** | `904_submission/submission_final.csv` (0.907551, est. private 0.8965) is strictly better than the top-level `submission.csv` (0.90680). **Recommend: submit `submission_final.csv` as the final**, and regenerate/verify it first (see §7, step 1). Confirm whether it has already been uploaded — the docs still say "private 0.891", which is the *old* file. |
| **D1** | **Which number becomes canonical?** | One of `0.907551` (if D0 = submit final) or `0.90680` (if keeping the current top-level file). Do **not** keep `0.9023` as "current" anywhere — it becomes the historical "previous champion" only. The canonical block in `00_INDEX.md` must be rewritten to match, once, then propagated. |
| **D2** | **Is the ~0.92 competitor on public or private LB, and do we care to chase it?** | `docs/01_task/competition.md:45` says "0.92 exists" but not which board. If it is private, our 0.8965 estimate is still ~0.025 behind, and the honest line is the ceiling argument (§9), not a score claim. Clarify so the story is truthful. |
| **D3–D8** | Team name / licence / GitHub repo / weights link / 1-or-2 submissions / keep `Consolidation/` private | already enumerated in `RUN.md` §0 (D1–D6). Confirm; these write into the report header + repo. |

**Do not edit `00_INDEX.md` until D0–D2 are answered.** Every other edit depends on it.

---

## 2. Codebase + documentation consistency (do after §1)

### 2.1 Propagate the canonical score (one pass, in this exact order)

The discipline (from `Personal/AGENTS.md §6`): **one number, one place — `Personal/docs/00_INDEX.md`
first**, then every quoter, then re-run the consistency scan.

1. Rewrite the `00_INDEX.md` canonical block so the **header, the per-target table, and the
   CANONICAL NUMBERS block all agree** on the number chosen in D1. Today they disagree: header says
   0.90680, table shows a per-target row that **sums to 0.90731** (tg 0.9043 + egc 0.9221 + egb
   0.9310 + ei 0.8711 + eea 0.9270 + nc 0.9097 + eps 0.8860 = 6.3512 ÷ 7), and the canonical block
   says 0.9023. Fix the per-target table to the **actual** verified rows from
   `904_submission/logs/score_final_verified.json` (tg 0.9039 · egc 0.9213 · egb 0.9318 · ei 0.8741
   · eea 0.9253 · nc 0.9101 · eps 0.8864), which self-consistently sum to 0.90755.
2. Then update, in this order, every file that quotes a score:
   `Personal/CONTEXT.md` → `Personal/docs/06_results/scores.md` → `score_history.md` →
   `ceiling_analysis.md` → `Personal/STORY.md` → `Personal/Presentation/PRESENTATION.md` (deck
   body) → `SLIDE_PLAN.md` → `SPEAKER_NOTES.md` → `EVIDENCE_SHOWCASE.md` →
   `Personal/Midnight_Report/PROMPT_10PAGE.md` + `REPORT_10PAGE.md` →
   `codebase/README.md` → `codebase/RESULTS.md` → `codebase/FINDINGS.md` → `codebase/ARCHITECTURE.md`
   → `codebase/Website/app.py` (footer) + `Website/README.md` → root `README.md` → root `CONTEXT.md`
   → root `RUN.md`.
3. Update the **public/private forensics** wherever they appear (`PRESENTATION.md`, `RESULTS.md` §3,
   `docs/06_results/public_private_analysis.md`): the "predicted private within 0.0004" story was
   computed off the 0.9023 champion. If D0 submits the Phase-7 file, the private number is an
   **estimate** (0.8965), not a measured 0.891 — say "estimated" and keep the difficulty-stratified
   *method* (it is the strong part).
4. Update the **ceiling appendix** (§6 below) so its worked numbers use the chosen predictions, and
   the "0.891 private ≈ 96% of ceiling" line becomes the chosen private (e.g. ≈0.8965 ≈ 96.4%).

### 2.2 Known factual fixes (independent of the score decision)

| # | file | issue | fix |
|---|---|---|---|
| 1 | `Personal/docs/00_INDEX.md:65` vs `FINDINGS.md:11` (and `CONTEXT.md` Phase-7 table) | tg quoted as **0.8953** in one place and **0.8954** in another | standardise (the previous-champion tg is 0.8954; the Phase-7 tg is 0.9039). One value per quantity. |
| 2 | `RESULTS.md` §5 / `docs/09_generalization/generalization_ladder.md` | ladder `random 0.894 → grouped 0.896` is a **rise**, and any doc that calls it a "drop that proves dedup works" is wrong | rewrite as "grouped is within noise of random (dedup costs nothing); the cliff is scaffold 0.825 → low-sim 0.620 → ultra-low 0.562". |
| 3 | any doc saying "python 3.11.7 is load-bearing" | superseded by the platform-defect finding | rewrite per §0.2. |
| 4 | any doc saying "no neural architecture won" | false under Phase 7 | rewrite per §0.1. |
| 5 | `REFINEMENT.md` (this file's predecessor) "MAXIMIZED at 0.90680 / no score work" | stale | superseded by §0/§1 of this file. |
| 6 | `RESULTS.md` §3 / `docs/06_results/public_private_analysis.md` | the public→private gap decomposition sums to ~0.012 but the observed gap is **0.0114**, and the "predicted within 0.0004" claim is asserted, not shown | make the four components sum to the observed gap, and show the arithmetic behind the 0.0004 — it is our best methodology line and must close. |
| 7 | `REPORT_10PAGE.md:238` | labels **0.9066 ± 0.0018** (a Tg out-of-fold *seed-stability* number) as "mean R² on the local panel" | relabel: it is 5-seed Tg OOF stability, **not** a panel score. Keep 0.9066 and 0.90680/0.907551 visually distinct or a judge will read them as contradictory. |

### 2.3 Release-gate scan (run unchanged, but run it)

From `RUN.md` §9: no forbidden term (`oracle` / khazana / polyinfo / TgSS / test_answers / vishwa /
hostname / IP / username path) anywhere in `codebase/`; no agent files in `codebase/`; every figure
path resolves; experiment ledger ≤ 80. **Do this after every text edit.**

---

## 3. Charts — they exist, but not where the docs point

This is a "wrong path" problem, not a "missing chart" problem.

- `codebase/904_submission/outputs/` has **108 PNGs** (full EDA + training + explainability +
  robustness + generalization) plus the CSVs behind every claim.
- `codebase/outputs/` (top level) has **47 PNGs** but `outputs/eda/` is empty (`.gitkeep` only),
  there is **no** `outputs/architecture*.png`, no `outputs/explainability/*`, no `outputs/training/*`.
- The report prompt (`PROMPT_10PAGE.md §B,§C`) and the deck (`PRESENTATION.md`, `SLIDE_PLAN.md`)
  reference figures at **top-level** `outputs/…` paths.

**Action (order matters):**
1. Copy the needed PNGs from `904_submission/outputs/**` into the top-level `codebase/outputs/**`
   at the paths the docs already reference (do not rename the docs to point at `904_submission/`,
   which must stay out of the public narrative). Needed at minimum: the EDA set
   (`novelty_two_regimes.png`, `variance_share_trap.png`, `physics_identity_*.png`,
   `target_distributions.png`, `chemical_space_map.png`), the SHAP beeswarms, fidelity curves,
   linear probes, `smiles_invariance_boxplot.png`, `generalization_ladder_plot.png`,
   `trustworthiness_radar.png`, and the architecture diagram.
2. Generate the architecture diagram if its PNG is not already in `904_submission/outputs/` — run
   `outputs/architecture.py` (RUN.md §2). The deck's slide 1 and 4 are figureless without it.
3. Confirm `codebase/outputs/scorecard.md` reads **14/18** (or whatever the latest run says) and
   propagate that ratio if it changed.

---

## 4. Report — a full draft already exists; it is stale, not missing

`Personal/Midnight_Report/REPORT_10PAGE.md` already contains all ten mandated sections + Appendix A
(experiment ledger D1–D9) + Appendix B (ceiling). It is **complete but written against the 0.9023
champion with no GNN blend and no platform finding.**

The mandated skeleton (from `analysis/SAMPLE_REPORT_ANALYSIS.md:19-38`, which analysed the
organisers' own template) is: 1 Executive Summary → 2 Problem Formulation (phenomena / science gap /
datasets) → 3 Architecture & Novelty → 4 Quantitative Performance → 5 Salient Visualizations →
6 Ablations → 7 Scientific Insights & Interpretability → 8 Robustness & Scalability → 9 Limitations
& Future Roadmap → 10 Contributions & References → appendices → closing declaration. `PROMPT_10PAGE.md`
already enforces exactly this. **Do not restructure; refresh.**

**Report edits needed:**
1. **§3 Architecture** — add the GNN-blend stage (GINE, 3 seeds, `w = clip((cv−0.80)/0.25, …)`),
   and state plainly that the GNN helps *only* via decorrelation (its solo tg 0.8987 vs 0.8954).
2. **§4 / §6** — reconcile "GNN ei −0.309" (the *standalone* D-MPNN failure) with the *blended*
   GINE win, so a judge cannot accuse you of contradicting yourself. The two are different things:
   D-MPNN standalone at n=222 fails; a structure-grouped GINE blended at w ≤ 0.60 as a decorrelated
   second opinion adds +0.0045. Say that explicitly.
3. **§8 / §9** — add the platform-defect finding and the `beats_092_competitor: false` / 0.92
   competitor context (honestly framed).
4. **All score lines** — replace 0.9023 with the D1 canonical number, and mark the private number as
   measured (0.891) or estimated (0.8965) per the D0 decision.

### 4.1 Appendix B ("max cap") — the user's instinct was right: it needs a real pass, or trim it

**Verdict: keep the two bulletproof pieces, fix or drop the rest.** A recomputation against the
underlying `Consolidation/` scripts (this is exactly the "verify it, else remove it" the human
asked for) found the appendix is **mixed**: some parts are pure arithmetic and unattackable, others
are hand-wavy or internally inconsistent. An expert panel *will* probe the weak ones.

**KEEP (defensible, quote with confidence):**
- **Tg-alone bound** — perfect Tg with the other six frozen gives `(1.000 + 0.9111 + … + 0.8847)/7
  = 0.9172`. Pure arithmetic; state as a one-line theorem.
- **Bootstrap standard errors** (ei 0.022 · eps 0.024 · nc 0.020 · eea 0.014 · egb 0.012 · tg 0.007
  · egc 0.006) — derived, and the corollary "any delta below ~2 SE on a small target is noise".
- **Single-row leverage** (worst row → ei +0.013) — measured.
- The **label-noise formula** R²_max = 1 − (σ/109.08)² — algebra, safe. **But** the "σ = 15 °C,
  literature-typical" figure needs a citation (it is currently uncited).

**FIX OR DROP (these will get you attacked):**
1. *"tg carries 99.986% of the pooled TSS"* is the **wrong explanation** for the 0.9370 pooled-R²
   figure. That share implies a pooled R² of ≈0.895 (a tg-only number), not 0.9370; 0.9370 arises
   from a global-mean denominator with large between-target mean spread. Either derive the pooled-R²
   comparison correctly or **drop the 0.9370 claim entirely** — the metric-is-a-mean point stands
   without it.
2. *"empirical Tg ceiling ≈0.92"* is **hard-coded, not derived** (it is an assumption in the
   variance-headroom script, not a measurement). Label it as an assumption or remove it.
3. *"composite ≈0.93 ± 0.01"* **contradicts the script's own 0.9414**. Reconcile the two, or drop
   the composite and keep only the individually-proven bounds.
4. *"six DFT targets bounded by scarcity (148–224 rows)"* **miscounts egc (1,352 rows)**. Correct the
   statement or drop the "six" framing.

**Rule for the implementer:** in Appendix B, only assert what is *proven* (items 1–2 + the algebra).
Everything else gets either a proper derivation or is deleted. The appendix's value is that it
*derives* a ceiling; if any step is hand-wavy, the whole thing reads as padding — **an over-claimed
bound is worse than no bound** (that line is already in `ceiling_analysis.md`, honour it).

---

## 5. Presentation & website

### 5.1 Deck — structurally complete, but it quotes the old score and its figures are missing

- `Personal/Presentation/PRESENTATION.md` is the deck body. It uses **0.9023** as the current local
  score in five places (`:283,:358,:362,:396,:417`) while `SLIDE_PLAN.md` and `PROMPT_PRESENTATION.md`
  headers say 0.90680. Fix all of them to the D1 number.
- `PRESENTATION.md:148` notes **410 s total vs the 360 s cap** (`PROMPT_PRESENTATION.md:97`). Rebalance
  to ≤ 360 s.
- Slide 5 "failure strip" currently says nothing about the GNN. Add the *corrected* ML story: GNN
  standalone ei −0.309 · MLM probe 0.651 vs 0.708 · ChemBERTa (out-of-competition) 0.751/0.784 vs
  0.810 — **and then** the one-line Phase-7 reversal (GINE *blended* adds +0.0045). That juxtaposition
  is the honest, memorable version.
- No exported PDF/HTML exists in `Personal/Presentation/`. `PROMPT_PRESENTATION.md:35-36` wants a marp
  export. Produce it and confirm it opens offline.

### 5.2 Website — no 3D, no architecture view, screenshots missing

Verified: `Website/static/` contains only `.gitkeep`; a repo-wide grep for `3d / 3dmol / manim / d3 /
three.js / molstar` returns nothing. The site renders a **2D RDKit PNG** (`app.py:61-69`,
`api.py:81-97`) and the architecture is only described in a **text footer** (`app.py:213-220`).

The demo is **deliberately offline** (`Website/README.md:21-22`) — a hard constraint, since the
rehearsal turns wifi off. Therefore any 3D viewer must be **vendored JS (3Dmol.js or py3Dmol bundle
committed into `static/`)**, not a CDN. Given effort vs. payoff, do it in this order and stop when
time runs out:

1. **(cheap, high value)** Put `architecture_simple.png` into `Website/static/` and render it in an
   "About the model" expander (`st.image`). This alone satisfies "the website shows the architecture".
2. **(medium)** Add an optional 3D conformer view via vendored 3Dmol.js: generate coordinates with
   `rdkit Chem.AddHs + AllChem.EmbedMolecule` (a 3D *conformer*, not the raw SMILES) and render the
   SDF in an iframe. Keep it behind a "3D (experimental)" toggle so the 2D path remains the default.
3. **(skip unless time is abundant)** SHAP force-plot / live internals — needs a proxy model and
   SHAP.js; lowest value for the effort.

**Regardless of 3D, these are mandatory and currently missing:** capture the four demo screenshots
+ `demo.gif` per `Website/screenshots/README.md` and `DEMO_SCRIPT.md:57-60` (they are the fallback
tab if the live demo stalls), pre-load the site once before presenting, and test it **with wifi off**.

---

## 6. Story, research framing, and QnA

### 6.1 STORY.md — add research framing; the narrative itself is strong

`STORY.md` is a clean four-act engineering/trust narrative (setup → two-regimes finding →
architecture/physics → trust → demo/close) but it carries **no citation, no "cap" argument, and no
contribution-to-the-field framing** — all three already exist elsewhere and only need weaving in:

- **Act 0/1** — name the field-level gap: *"polymer-informatics papers report R² and assert
  SMILES-invariance; none measure whether its explanations are load-bearing, or whether the
  attributions survive rewriting."* (This is already the §2 science-gap text in
  `PROMPT_10PAGE.md §G`.)
- **Act 2** — add the G1 citation: *"Grinsztajn et al. (NeurIPS 2022) showed trees beat deep nets on
  tabular data at our sizes; we confirmed it extends to chemistry, with one refinement — a GNN earns
  its place only as a decorrelated second opinion."*
- **Act 3** — add the **cap** argument, but use only the *proven* bounds (see §4.1): *"Tg is capped
  by label noise — the thermometer, not the model (a perfect Tg model still only reaches 0.9172 for
  the mean) — and the small DFT targets carry bootstrap standard errors of 0.012–0.024. We derive
  where the remaining score can live instead of promising a number, and we spent the last phase on
  the uncapped half of the grade: invariance, fidelity, and honesty about what the model doesn't
  know."*
- **Act 4** — forward-reference future work honestly (Phase 6: shallow-stack variance fix, tuned GP
  on ei/eps, fold-consistency promotion gate, physics-constrained joint imputation — from
  `Score_and_Invariance_Improvement/PLAN.md` and `CEILING_REALITY_CHECK.md:83-93`).

### 6.2 QnA — consolidate, and add the one answer that is missing

The QnA bank is 10 topic files + an index (`docs/11_qna/`), ~83 answers, well-structured, with a
strong `hostile.md`. **There is no single master file and no "what is novel / your contribution to
the field" answer** — that framing is exactly what the judges will ask.

**Do:**
1. Create `docs/11_qna/MASTER_QNA.md` that inlines (not just links) the top answers, grouped by
   theme, each answer ≤ 60 s with its key number. Point `Personal/QNA.md` at it.
2. **Add the missing "contribution to the field" answer.** The source material is already written
   in `docs/00_My Docs.md` — specifically its §"Core Research Gaps & Difficulties" (line 103),
   §"How Our Pipeline Directly Addresses These Gaps" (line 121), §"Research Papers and References"
   (line 380), and the "Scientific Contribution" slide note (line 520). The answer should be the
   five contributions, corrected for Phase 7:
   - **measured trust** (ROAR fidelity 0.851 vs 0.043 + attribution invariance cos 0.95–0.99 + patch Δ=0.0) — a verification protocol, not a model;
   - **physics > ML at small n** (LOO residual −0.82; bare identity beats the correction);
   - **honest negative results** (4/18 scorecard fails pre-registered with causes; 1,150+ experiments);
   - **emergent physics** (Flory–Fox recovered unsupervised, median R² ≈ 0.99; aromaticity in the hidden layer, probe R² 0.895);
   - **decorrelated neural blending** (the GINE blend: +0.0045 from *different errors*, not better errors).
3. **Add a "local held-out verification panel" explainer** (how it is constructed, why 4,909 of
   4,940 rows, and why it is post-freeze only) — an expert will ask exactly this, and no file
   currently answers it in one place.
4. Preserve the strong QnA additions from the previous REFINEMENT (why-not-pretrained, why-7-models,
   why-NNLS, why-0.20-character-multiplier, and hostile H11/H12 on the public/private gap and the
   invariance-vs-char-ngram contradiction) — they are correct and worth keeping verbatim.

### 6.3 Docs & research — mine `00_My Docs.md` and `Research/INDEX.md`; do not reinvent

- `docs/00_My Docs.md` (758 lines) is the richest unused asset: property physics, the
  research-gap→solution map, a "Best possible model" section, and a "Summary Defense Matrix for
  Technical Panels" (line 610). **Pull the report/presentation/QnA language from here.**
- `Research/INDEX.md` already holds 33 verified/standard citations with a citation→decision map.
  **G1 (Grinsztajn, NeurIPS 2022) is present and verified** — it is the "cap and limits of ML on
  tabular data" paper you wanted. M2 (Fox & Flory 1950) backs the Flory–Fox demo. P1/P4/C6 back the
  "pretrained models didn't reproduce here" contrast. **Cite nothing not in this table.**

---

## 7. Execution order (work top-to-bottom; stop when time runs out)

**Tier 0 — the submission (ask the human first, §1):**
1. Verify `904_submission/submission_final.csv` still satisfies the contract (4,940 rows, ids
   1..4940, columns `id,target`, all finite) and matches SHA `cd91f278…`.
2. If D0 = submit, upload it as one of the 2 final slots **before 3 Sep**.

**Tier 1 — numbers + figures (the trust surface):**
3. §1 decisions → §2.1 canonical propagation → §2.2 fixes → §2.3 release-gate scan.
4. §3 chart copying; generate the architecture diagram; verify `scorecard.md`.

**Tier 2 — the three artifacts:**
5. Refresh the report (§4): GNN stage, platform finding, re-derived ceiling, corrected ML story.
6. Refresh the deck + export PDF (§5.1); capture demo screenshots (§5.2).

**Tier 3 — narrative + QnA:**
7. §6.1 STORY.md framing; §6.2 MASTER_QNA + the two missing answers.

**If you have < 4 hours**, do only Tier 0 + Tier 1 + the deck's score fix + the demo screenshots.
Everything else is additive.

---

## 8. Acceptance criteria (the human signs off when all are true)

- [ ] **One canonical score** in `00_INDEX.md`, agreed with the header, the per-target table, and every quoter (§2.1).
- [ ] The scorecard ratio in `outputs/scorecard.md` matches every doc that quotes it.
- [ ] No doc says "no neural architecture won" or "0.894 → 0.896 drop proves dedup" or "python 3.11.7 is load-bearing" (§2.2).
- [ ] Every figure path referenced by report + deck resolves; architecture + EDA + SHAP charts present at top-level `outputs/`.
- [ ] Report §3 includes the GNN blend and its decorrelation rationale; Appendix B re-derived against the final predictions.
- [ ] Deck ≤ 360 s and quotes the canonical score; PDF export opens offline.
- [ ] `MASTER_QNA.md` exists with the "contribution to the field" and "verification panel" answers.
- [ ] `STORY.md` names G1, the label-noise cap, and a forward future-work line.
- [ ] Release-gate scan clean (`RUN.md §9`).
- [ ] Final submission uploaded (if D0 = submit) before the 3 Sep deadline.

---

## 9. Content pack — paste-ready science for report, deck, and QnA

**Why trees, not deep nets (G1).** *Grinsztajn, Oyallon, Varoquaux — "Why do tree-based models
still outperform deep learning on typical tabular data?", NeurIPS 2022. Verified. We confirmed it
extends to chemistry at n = 222–4,143: standalone D-MPNN scores −0.309 on ei (0/5 folds), and nine
self-supervised variants all ≤ a matched control (MLM probe 0.651 vs random-init 0.708). The one
refinement Phase 7 adds: a structure-grouped GINE, blended per target at w ≤ 0.60 as a decorrelated
second opinion, lifts the mean +0.0045 — not because it is better, but because it errs differently.*

**Why the ceiling, derived not asserted.** *The metric is the unweighted mean of seven per-target
R², so one target can never carry the score: a perfect Tg model with everything else frozen reaches
only **0.9172** (pure arithmetic). Tg is further capped by experimental label noise — with a 109 °C
spread, a noise floor σ gives R²_max = 1 − (σ/109.08)², so even σ = 10 °C caps Tg near 0.99 and the
measured difficulty-stratified spread is tighter still — and the small DFT targets carry bootstrap
standard errors of 0.012–0.024. The takeaway is the **method**: we derive where the remaining score
can possibly live, instead of promising a number we may miss. (If we quote a composite "≈0.93"
figure at all, it must first be reconciled with its own source script — see §4.1 — otherwise drop it
and keep only the two proven bounds.)*

**Why physics beats the correction.** *egc = ei − eea holds at R² 0.9716 (n=59); an ML residual on
top scores leave-one-out R² −0.82, i.e. it overfits noise. eps = n² + ionic has 0/134 violations and
is 2.62× better conditioned. The textbook won, and we measured it instead of assuming it.*

**Why we measure trust, not assert it.** *Mask the SHAP-top 10% of features → the model loses 0.851
R²; mask the same number at random → 0.043. Thirty spellings of the same polymer give the same answer
and the same attributions (SHAP cosine 0.95–0.99, patch Δ = 0.0). The Flory–Fox relation (1950)
emerges unsupervised at median R² ≈ 0.99. That is a verification protocol the field can reuse.*

**Sources for every sentence above:** `docs/00_INDEX.md`, `docs/06_results/ceiling_analysis.md`,
`docs/07_explainability/fidelity.md`, `docs/08_robustness/*`, `Research/INDEX.md`,
`904_submission/RESUME_HERE.md`, `Score_and_Invariance_Improvement/CEILING_REALITY_CHECK.md`.

---

**Handoff:** after the above, generate the report from `Personal/Midnight_Report/PROMPT_10PAGE.md`
and the deck from `Personal/Presentation/PROMPT_PRESENTATION.md` (both now refreshed), rehearse
against `DEMO_SCRIPT.md` (45 s) and `MASTER_QNA.md`, and run the `RUN.md §9` gate one final time.
