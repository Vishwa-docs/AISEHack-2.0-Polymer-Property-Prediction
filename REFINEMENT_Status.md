# REFINEMENT_Status.md — live status tracker

> Handled by the coding/documentation agent per `REFINEMENT.md`. Updated after every task.
> **Canonical number decision (D0/D1, answered by the human):** submit
> `904_submission/submission_final.csv` → **canonical local verification panel score 0.907551**,
> per-target tg 0.9039 · egc 0.9213 · egb 0.9318 · ei 0.8741 · eea 0.9253 · nc 0.9101 · eps 0.8864.
> `0.90230` becomes the *historical previous champion* only.
>
> Other defaults applied: private LB = **estimated** 0.89655055 (difficulty-stratified method; the
> measured 0.891 is the *old* 0.9023 file); competitor ~0.92 is treated with the **ceiling
> argument** (`beats_092_competitor: false`); public/private "within 0.0004" story is re-based on the
> measured 0.9023 champion with the arithmetic shown, and the Phase-7 private figure is labelled
> "estimated".

## Progress log

- [x] **Tier 0 — submission verified.** `submission_final.csv`: 4,940 rows, ids 1..4940, columns
  `id,target`, all finite; SHA256 `cd91f2785dd9b7…` ✓ (contract PASS). Top-level `submission.csv`
  was SHA `30141c6a…` (= 0.90680, the older file). Environment: root `.venv` python 3.11.7,
  rdkit 2026.03.5, pandas 3.0.5 ✓. Scorecard reads 14/18 ✓.
- [x] **§1 decisions recorded.** D0/D1 answered by the human (above); D2 defaulted to the
  ceiling-argument framing with `beats_092_competitor: false` stated honestly.

- [x] **§2.1 Canonical propagation (score).** Decision D0/D1 = submit `submission_final.csv`,
  canonical **0.907551**. Updated the shared "CURRENT BEST ARCHITECTURE" header block (6 files:
  Personal/CONTEXT, docs/00_INDEX, Personal/FINDINGS, codebase README/RESULTS/FINDINGS) with the
  verified per-target rows (tg 0.9039 · egc 0.9213 · egb 0.9318 · ei 0.8741 · eea 0.9253 · nc 0.9101
  · eps 0.8864, sum 0.907551). Rewrote the `00_INDEX.md` canonical block (SCORES/PER TARGET/MAE/RMSE
  to verified values), the environment note (platform defect, not "python 3.11.7 load-bearing"), and
  the vocab rule. Updated CONTEXT.md (Personal+root), scores.md, score_history.md (added a Phase 7
  row), and mechanically replaced `0.90680 → 0.907551` across all Personal + codebase markdown
  (remaining 0.90680 hits are only the `submission_0.90680.csv` filename + the notebook's correct
  "beats 0.90680" comparison, both left intact).
- [x] **§4.1 Ceiling re-derived (honest).** ceiling_analysis.md: Tg-alone bound recomputed to
  **0.9213** under Phase-7 values (corollary "Tg alone can't reach 0.92" correctly retired);
  retired the contested pooled-R² `0.9370`; retired the hard-coded empirical "≈0.92" Tg ceiling and
  the `≈0.93 ± 0.01` composite (contradicted its own 0.9414 script + demanded ~0.028 elsewhere);
  labelled the uncited σ=15 °C input as an assumption; relabelled "Fixing"→"Dropping" for single-row
  leverage. public_private_analysis.md: gap components now sum exactly to observed **0.0114** with
  the 0.0004 arithmetic shown; Phase-7 private marked **estimated ≈0.89655** (not measured).
- [x] **§3 Charts copied + architecture generated.** Top-level `codebase/outputs/` now has
  `eda/`, `explainability/`, `training/`, `robustness/`, `generalization/` populated from
  `904_submission/outputs/**` at the paths the docs reference; `architecture*.png` generated via
  `outputs/architecture.py`; every `outputs/…` figure path referenced by REPORT_10PAGE.md,
  PRESENTATION.md and SLIDE_PLAN.md resolves (verified by an automated existence scan — zero
  MISSING). Scorecard ratio 14/18 propagated consistently.
- [x] **§4 Report refreshed.** REPORT_10PAGE.md: header/front matter updated (canonical 0.907551,
  weights as URL front matter), §3 GNN-blend stage added (GINE, 3 seeds,
  `w = clip((cv−0.80)/0.25, 0.10, 0.60)`, decorrelation rationale, GINE-vs-D-MPNN distinction),
  §4/§6 reconciled "standalone D-MPNN fails / blended GINE adds +0.0045", §8/§9 platform-defect
  finding + 0.92-competitor honesty, Appendix B reduced to the four proven bounds. PROMPT_10PAGE.md
  updated (hyperparameter mandate in §3, word budgets). PROMPT_3PAGE.md "must survive" block
  re-based on the canonical numbers (0.891 measured old / ≈0.8965 estimated Phase 7 / 0.907551).
- [x] **§5.1 Deck refreshed + exported.** PRESENTATION.md: all five 0.9023 quotes → 0.907551,
  timing rebalanced 410 s → **360 s** (table now sums to the cap), slide-5 failure strip rewritten
  with the corrected GNN story (standalone ei −0.309 · MLM probe 0.651 vs 0.708 · ChemBERTa
  0.751/0.784 vs 0.810, then the Phase-7 blended +0.0045 reversal); SLIDE_PLAN.md and
  SPEAKER_NOTES.md synced; **PRESENTATION.pdf exported** via marp-cli v4.5.0.
- [x] **§5.2 Website.** `architecture_simple.png` vendored into `Website/static/` and rendered in
  an "About the model" expander (footer text updated); fixed a real `%`-format-string bug in
  app.py; captured all four demo stills (`01_prediction.png`, `02_out_of_domain.png`,
  `03_invariance.png` + demo fallback) and **demo.gif** into `Website/screenshots/`. Live-checked
  via headless streamlit (HTTP 200), then shut the server down. Remaining human task: rehearse with
  wifi off (cannot be automated here).
- [x] **§6.1 STORY.md framing.** Retitled "five short acts"; Act 0/1 science-gap sentence, Act 2
  G1 citation with the Phase-7 refinement (decorrelation, not superiority), Act 3 cap argument
  using only the proven bounds, Act 4 honest Phase-6 forward line.
- [x] **§6.2 QnA consolidated.** `docs/11_qna/MASTER_QNA.md` created (inlined top answers by
  theme) with the "contribution to the field" answer (five corrected contributions), the
  "local held-out verification panel" explainer (4,909/4,940 = 99.4%, post-freeze), the trivial-
  baseline answer, the published-numbers comparison, and the full-ablation answer; `Personal/QNA.md`
  points at it; prior strong additions (why-not-pretrained, why-7-models, why-NNLS,
  why-0.20-multiplier, hostile H11/H12) preserved.
- [x] **§6.3 Citations whitelisted.** `Personal/Research/INDEX.md` extended with **Krogh &
  Vedelsby (NIPS 1994)** (primary backing for the decorrelated blend) and **Hu et al. (ICLR 2020)**
  (GNN-pretraining contrast alongside C4/C6); the σ=15 °C figure stays labelled as an assumption
  rather than cited (no verified polymer-noise citation exists in the whitelist — honoured the
  "cite nothing not in this table" rule by dropping the number from claims instead).
- [x] **§2.2 factual fixes + stale-claim sweep (final pass).** "python 3.11.7 is load-bearing"
  rewritten to the platform-defect story in `docs/01_task/constraints.md`,
  `docs/10_gaps_and_future/limitations.md`, `docs/11_qna/process_and_tooling.md` (30-s answer
  reworded), and `Presentation/SLIDE_PLAN.md` (remaining "load-bearing" hits are the *corrected*
  "the claim is wrong" phrasing — intentional). No doc asserts "no neural architecture won" or a
  "0.894 → 0.896 drop proves dedup" (ladder retold as within-noise + scaffold cliff). `STORY.md`
  act-count fixed. tg 0.8953/0.8954 drift standardised. Rounding drift standardised to 4-sig-fig
  forms in `00_INDEX.md`. REPORT_10PAGE.md:238 relabelled as 5-seed Tg OOF stability, not a panel
  score. Competition doc clarified: the ~0.92 competitor is on the Round-3 (public) board.
- [x] **§2.3 Release-gate scan (final).** Clean: zero hits for oracle/khazana/polyinfo/TgSS/
  test_answers/vishwa/username paths anywhere in `codebase/` (904_submission artifacts redacted
  mechanically); no agent files in the codebase; experiment ledger = 11 curated entries (≤ 80);
  every referenced figure path resolves; scorecard 14/18 consistent across all quoters.
- [ ] **Human-only residual items:** (1) actually upload `submission_final.csv` as a final slot
  before 3 Sep if not already done (D0); (2) rehearse the website demo with wifi off (screenshots
  + demo.gif are the fallback tab and are in place); (3) reconcile the ROUNDING of the two
  estimated-vs-measured private numbers on stage so the "0.891 measured vs ≈0.8965 estimated"
  pairing is said deliberately, not read as a contradiction.