# Round 3 Experiment Loop — Polymer Property Prediction

Read `AGENTS.md` first. This file defines the research→experiment→execution→analysis
feedback loop for Round 3. It does NOT authorize Kaggle compute, upload, or
submission — those are human-only actions (see AGENTS.md).

## Objective

Beat the current Round 3 leaderboard (a submission at **0.92** exists) and win the
hackathon with an official-data-only, from-scratch, single-notebook pipeline.

Quantified targets (all on the **no-archive** lane; `archive/` no longer exists):

| Lane | Metric | Target |
|---|---|---|
| Local post-freeze **verified oracle** mean R² (3,818-row panel) | ≥ **0.935** | hard goal |
| Local **proxy** panel mean R² (4,905-row panel) | ≥ 0.928 | diagnostic |
| Expected Kaggle **public LB** | ≥ **0.93** | win condition (public ran ≈ 0.013 below local oracle in R2) |
| Floor (must never go below) | public ≥ 0.92 | beat the current leader |

The metric gives every target equal weight despite unequal row counts, so per-target
incumbents and target-wise assembly remain mandatory (Round 1 + Round 2 lesson).

Frozen starting evidence (Round 2, no-archive): best clean local composite
**0.9042 verified / 0.9030 proxy** (Sandman V52/V57), user-reported public **0.891**.
Summed gap to a 0.935 verified mean is ≈ **0.215 R² points across seven targets**.
This is too large for cosmetic global calibration or micro-blends — seek prospective,
target-specific signal (weak targets: **eps, nc, ei, tg** in that order).

## Frozen data identity (verified 2026-08-26 on the Mac)

| File | Rows | SHA-256 |
|---|---:|---|
| `Dataset/train.csv` | 7,409 | `609b0f48a95fb5151a8e7fbf0d90755e42216a931d71bb31b5d2263f660f9ba2` |
| `Dataset/test.csv` | 4,940 (4,497 unique SMILES) | `d8a0da2669b2ec41275f0fb42f08e29f1100220874464f23c129a76ab811cf2d` |
| `Dataset/PI1M.csv` | 995,799 | `c5e1017b61bad9642f09c3e85be22ee1d9926fd32a0a77d8258ba41b41cd9cd8` |
| `Dataset/smile_r3.csv` | 5,973,369 (all unique, no overlap with train/test/PI1M) | `c64f96eecb01f8ff5fe5ba0619dbf4ed882e825d34494a803ac1376e55184ac3` |

**PI1M note:** `PI1M.csv` is official Round 3 data (user confirmed the
organizers provided everything in `Dataset/`) even though the Dataset
Description omits it. Sanity-check its presence in the Kaggle input dir at
notebook time; prefer `smile_r3.csv` for representation learning (6× larger).

Per-target train/test rows: tg 4,143/2,763 · egc 2,028/1,352 · egb 337/224 ·
ei 222/148 · eea 221/147 · nc 229/153 · eps 229/153. `archive/` is gone — no Round-1
labeled data may be used in Round 3 training (it is not in the Round 3 data section).

## Stage 0 — bootstrap (one-time, do not skip)

1. Verify the hashes above against the local `Dataset/` files.
2. Verify the oracle files under `Oracle/` (hashes in `Oracle/NOTES_R3.md`) and confirm
   no code path outside `Oracle/` references them.
3. Freeze: target order, seeds, fold assignments, canonicalizer, similarity clustering,
   and evaluation code (`scripts/r2_reference/fable_engines/fable_common.py` is the
   R2 reference for grouped folds + shift-matched R² — port and freeze an R3 copy).
4. Initialize append-only `logs/experiments.jsonl` (R3-C000 is the first id; never
   rewrite existing lines; never recycle ids).
5. Confirm the oracle path is not imported, opened, mentioned, or discoverable by the
   clean pipeline or the submission notebook.

## Stage 1 — recover the Round 2 no-archive baseline (R3-C000)

Port the Round 2 best no-archive pipeline into this repo and reproduce its frozen
diagnostic before inventing anything new:

- Source: `scripts/r2_reference/Sandman_Version_52_8th_Aug_without_archive.ipynb`
  (and V53) + `scripts/r2_reference/fable_engines/`. The R2 codebase on the GPU
  laptop (see AGENTS.md for SSH details) holds the full history — search it
  read-only, copy what you use into this repo.
- Gate: reproduce the V52/V57 local oracle score (verified 0.90415 / proxy 0.90305)
  to within 0.0005, with a clean source scan (no oracle references) and a
  4,940-row `id,target` output. If reproduction fails, debug before proceeding —
  a broken baseline makes every later comparison meaningless.

Only after C000 reproduces may new experiments begin.

## Stage 2 — improvement ladder (one bounded experiment at a time)

Candidate families are ordered by expected value. Any family may be skipped if the
Round 2 logs (search them first — see "Cooled families" below) already falsified it.

1. **Validation realism / selection repair** — close the gap between oracle-selected
   and cleanly-selected components (R2: 0.904 clean vs 0.951 oracle-assisted). Test
   selection criteria that do not see the oracle: shift-matched R² (OOF reweighted to
   the test nearest-neighbour similarity histogram), co-test agreement, scaffold
   bootstrap, availability-masked scoring.
2. **Tg group-contribution & ensemble breadth** — Tg is the largest target
   (2,763 test rows) and sat at ≈0.90 without archive. Test: richer group/substructure
   counts, Bicerano-style rigidity/free-volume features, multi-seed bagged GBM/Ridge
   stacks, monomer-vs-repeat-unit feature splits.
3. **EPS/Nc ionic-coordinate deepening** — the biggest R2 win (EPS +0.0666, Nc +0.0434
   clean OOF). Extend the F02 engine family (scripts/r2_reference/fable_engines/):
   polarizability-weighted coordinates, Clausius-Mossotti/Lorentz-Lorenz residuals,
   co-test consistency, and joint eps-nc² constraints as soft losses.
4. **Ei/Eea electronic deepening** — Ei is the weakest transfer target. Test:
   EHT/quantum-chemistry descriptors, conjugation/donor-acceptor features, identity
   coordinates Ei = Eea + Egc (soft), 3D-conformer-derived HOMO/LUMO proxies
   (bounded compute).
5. **smile_r3 self-supervised representation ladder** (new in R3, official data,
   from scratch only): a fixed hash-ranked ladder of 100k → 1M → 6M SMILES.
   Representation candidates in order of cost: Morgan/substructure-count SVD/PPMI,
   char-level masked-language modeling (tiny transformer, e.g., 4 layers × 256 width),
   word2vec-style token embeddings, then (only if a probe passes) contrastive/multi-view
   variants, and pseudo-labeling on unlabeled polymers for the sparse targets (only
   after a probe passes, fold-local, oracle-free). Every probe must compare against an
   equal-budget, same-fold official-only control and pass the weak-target gates.
   **R2 history: PI1M SSL failed every time (C010/C119/C131/C157/C158/C169/C181/C185/C261/F06)** —
   do not repeat those exact designs; a new probe must differ in representation AND scale AND head.
6. **Proper multi-task learning** (new for R3, bounded): Khazana's own paper found
   multi-task beats single-task on sparse targets, but R2 only tried low-rank/linear
   variants. A shared-encoder MLP with 7 heads, target-balanced sampling, and soft
   physics constraints (Egc ≈ Ei − Eea, eps ≥ nc²) is admissible once; kill if no
   weak-target gain over the incumbent.
6. **Invariance & test-time augmentation** — canonical + randomized-SMILES training
   augmentation, multi-view consistency regularization, and TTA (average predictions
   over K valid SMILES permutations of each test polymer). Free, often +0.002…0.01.
7. **Deep-from-scratch models** (last, bounded): a small GNN or SMILES transformer
   trained from scratch, only if families 1–5 have been exhausted and gates remain
   unmet. R2 found generic GNN/CNN/Transformer weak — require a materially different
   design (e.g., periodic-graph recurrence + physics heads) and a strict kill gate.
8. **Explainability build-out** (required by Round 3 theme; no score impact): per-target
   SHAP/permutation importance for the final ensemble, global + local explanations,
   invariance robustness demonstrations. Package into the submission notebook and
   `FINAL_REPORT.md` as results arrive.

## Inner loop — one bounded experiment

After a completed experiment, run the five-role review (subagents where available;
the main runner must verify every claimed metric itself):

1. **Historian** — check for duplicate methods and source hashes; search the R2 logs
   on the GPU laptop (`~/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/logs/EXPERIMENT_LOG.md`,
   `research/research-log.md`, `research/findings.md`, `research/RESEARCH_NOVELTY_LEDGER.md`)
   for the same idea before running it. Record the verdict.
2. **Adversary** — propose leakage, fold-luck, shortcut, and distribution-shift
   explanations. Must answer, literally: *"Why are we not at 0.935 verified? Which
   exact target-level increments supply the missing sum? Is this proposal a credible
   path to ≥ +0.01 on one target under exact baseline parity, grouped folds, bootstrap,
   similarity, scaffold, and availability panels — or is it another small residual?"*
3. **Property researcher** — investigate exactly one target/residual mechanism
   (primary sources; record URLs + content hashes in the novelty ledger).
4. **Planner** — choose exactly one discriminating next run: expected signal, resource
   bound, pass gate, stop gate.
5. **Notebook auditor** — for any new incumbent: clean-input scan (no `oracle`/`Oracle`
   string, no local file reads beyond `Dataset/`), parity check against the local run.

All five roles are disjoint and read-only. **Sidecar subagents must never read the
oracle** — oracle reads are post-freeze and main-runner-only, via the scoring script.

Every inner-loop protocol freezes before execution: one primary hypothesis, one
changed factor, compute limit, expected target/slice, clean pass gate, stop gate, and
notebook impact. Complete metrics, decision, hashes, and research state before starting
another experiment. Do not revisit a cooled method without a new falsifiable reason.
After three consecutive weak ideas, change target or representation instead of
continuing a hyperparameter crawl.

## Promotion gates (from Round 2, unchanged — they prevented false convergence)

- **No-op rule (new, R3)**: a run whose output CSV is byte-identical to the
  incumbent, or whose verified score is within 1e-5 of the incumbent without a
  real model change, is a **no-op** — record `state=no_op`, never promote, and
  the slot must be re-run with a real pipeline. (The first R3 batch of 100
  experiments violated this — placeholder V52+noise scripts — and one no-op was
  wrongly promoted; see PLAN.md §2.)
- **Component gate**: grouped target gain ≥ `0.01`, at least 4 of 5 folds in the same
  direction, group-bootstrap lower bound above zero, adjacent/paired-target loss
  ≤ `0.003`, non-negative missing-auxiliary and low-similarity slices.
- **Shrinkage lane (R2 retrospective fix — "shrink, don't reject", Fable F09)**: a
  component that misses the +0.01 gate ONLY because of magnitude but shows 4–5/5
  positive folds and a positive group-bootstrap lower bound is NOT a dead end. It may
  enter the compound as a *shrunk* arm (small fixed weight toward the incumbent,
  e.g., 0.05–0.25, chosen by fold-local NNLS on OOF), and its per-target effect is
  re-measured in the compound audit. R2's +0.01/bootstrap gate rejected ≈ +0.06 of
  real summed signal on the small targets (n≈222) — do not repeat that mistake.
  The full +0.01 gate still applies for *replacing* a target's incumbent arm.
- **Full-incumbent gate**: prospective seven-target clean gain ≥ `0.002`, no target
  grouped loss worse than `0.003`, all transfer panels pass, notebook parity passes.
- **Oracle post-freeze checkpoint**: after a clean candidate passes all gates, fit the
  unchanged frozen configuration on all official training rows, write all 4,940
  predictions, hash the CSV, THEN score against `Oracle/oracle.csv` (verified panel)
  and the proxy panel separately. Report verified/proxy per-target and mean R².
  The oracle may inform the NEXT candidate's aggregate component choice
  (`oracle-observed`), never the frozen candidate's rows, weights, or routing.

Never pool targets; always report the seven R² values, their mean, fold std, MAE,
and coverage. Same-OOF screening is evidence only — the shift-matched/transfer panels
decide.

## Cooled families (search the R2 logs before touching any of these)

Generic GNN/CNN/Transformer/MLM from scratch · rich OOF stacking · broad read-across ·
forced residual routers · Mordred/trimer/generic-3D/AutoGluon sweeps · PI1M PPMI/
density/denoising/contrastive/subword probes as previously designed · naive
Lorentz-Lorenz hard equalities · abstention gates without scaffold transfer ·
unconstrained missing-partner routing · micro-blend weight sweeps (0.002–0.6) as a
primary search strategy. A Round 3 repeat must carry a genuinely new mechanism and a
pre-registered kill gate.

## Outer loop — synthesis and direction

Reflect after four valid experiments, three consecutive non-improvements, a surprising
clean/proxy/public divergence, or any integrity/rule incident. Read the full
trajectory and update `research/findings.md` (synthesize, do not copy logs). Choose
exactly one direction — `deepen`, `broaden`, or `pivot` — and record it in
`research/research-state.yaml`. Append `research/research-log.md`.

## Submission regeneration policy (hard requirement of this round)

- Every promoted candidate must be packaged as a **single, end-to-end, self-contained
  notebook or `.py` file** that reads only `Dataset/` files and writes
  `submission.csv` (4,940 rows, `id,target`) in one run with fixed seeds. It must
  never read `Oracle/`, `scripts/`, `experiments/`, or any precomputed artifact.
- When a new best is promoted: regenerate the notebook + CSV pair, store it in
  `final_submissions/`, **delete the superseded pair**, and update
  `final_submissions/README.md` with the verified/proxy scores and hashes.
- Experiments may be messy internally but must always be runnable so the submission
  notebook can be regenerated from the same frozen configuration without
  side-dependencies.

## Stop conditions

Stop a branch when it violates rules, leaks a held-out structure, cannot reproduce,
exceeds notebook resources, fails three relevant panels, or repeats a cooled
hypothesis. Preserve all artifacts; record the reason. A runtime-invalid run is
`failed`, a valid negative run is `rejected` — neither is ever a candidate.

## Compute policy

Mac = source of truth + orchestration + small runs. GPU laptop = long-running
inference/representation jobs only, via copied scripts, with results copied back into
this repo (see AGENTS.md — never modify the laptop's files). Kaggle compute only with
explicit user authorization.

Budget tiers (per experiment, declared in the protocol BEFORE execution):
smoke ≤ 15 min / ≤ 5% of the machine; pilot ≤ 60 min / ≤ 25%; confirm ≤ 4 h / full
allocation + 3 seeds; final full-data run ≤ 8 h and ≤ 75% of Kaggle notebook limits.
At most ONE heavy GPU process at a time; reserve ≥ 20% RAM/VRAM headroom (fail at
> 80% peak); reserve 2 CPU cores; never force OMP/OPENBLAS/MKL thread variables when
a run is replay-gated at 1e-12 (R2 C188-v2 incident).
