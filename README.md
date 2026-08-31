# AISEHack 2.0 · Polymer Property Prediction · Round 3

Everything for this project, in three folders.

| folder | what is in it | open first |
|---|---|---|
| **`Personal/`** | your operating base — docs, findings, story, trials, research, report and presentation prompts, 60+ QnA answers | `Personal/AGENTS.md` |
| **`AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/`** | the public submission codebase — pipeline, the annotated notebook, 47 curated charts, the evidence bundle, the offline demo site | `README.md` inside it |
| **`Consolidation/`** | the archive — Round 1, Round 2, Round 3, every phase, every submission, the quarantined verification data | `Consolidation/AGENTS.md` |

Plus at this level: **`RUN.md`** (everything still to run, in order), `CONTEXT.md` (one-page
portable context), `AGENTS.md` (router for agents), `PLAN.md` (the contract this was built from).

## Status

| | |
|---|---|
| private LB | **0.891** |
| public LB | 0.917 |
| local held-out verification panel | 0.9023 |
| evidence scorecard | **14 / 18 requirement groups PASS** |
| submission file | frozen and shipped — `<codebase>/submission.csv` |
| deadline | **3 September 2026** |

## What to do next, in order

1. **Read `RUN.md`.** Nothing in this delivery has been executed — the notebook, the diagram
   renderer and the demo screenshots all still need one run each. `RUN.md` has every command
   with its expected output and its known failure mode.
2. **Decide the six open questions** listed at the top of `RUN.md` (team name on public
   artifacts, licence, GitHub repository, model-weights link, one final submission or two, and
   whether `Consolidation/` stays private).
3. **Generate the report** — `Personal/Midnight_Report/PROMPT_10PAGE.md`.
4. **Generate the deck** — `Personal/Presentation/PROMPT_PRESENTATION.md`.
5. **Rehearse the demo** — `Personal/Presentation/DEMO_SCRIPT.md` (45 seconds).
6. **Queue the next research phase** — `Personal/Score_and_Invariance_Improvement/`: 38
   experiments across score, invariance, generalization, interpretability and reliability, with a
   ready-to-paste brief for the agent that runs them on the GPU laptop.

## The one-paragraph version

Seven polymer properties from one SMILES string. We measured the data before choosing an
architecture and found the competition is really two problems — for the six DFT targets 98% of
the evaluation polymers are already in the training file under a *different* property, while Tg
has almost no overlap — so we built one lane per target and used exact physics wherever it
exists. Private leaderboard **0.891**. What makes it a Round-3 answer is that we *measured*
trust instead of claiming it: masking the features our explanations point at costs 0.851 R²
against 0.043 at random, thirty spellings of the same polymer give the same answer *and the same
attributions*, and difficulty-stratified performance predicted our own private score to within
0.0004.
