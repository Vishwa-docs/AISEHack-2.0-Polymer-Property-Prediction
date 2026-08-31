# AGENTS.md — root router

**You are in `AISEHack 2.0 Polymr Property Prediction Round 3`** (the folder name is misspelled;
that is expected and is not a typo to fix — it is the live working directory). Everything for
this project lives here, in **three folders**.

Read this file, then open the `AGENTS.md` **inside** the folder you need. Do not work from this
page alone.

| folder | what it is | its own router |
|---|---|---|
| `Personal/` | the operating base: docs, findings, story, trials, research, report + presentation prompts, QnA. **Not a codebase.** | `Personal/AGENTS.md` |
| `AISEHack-2.0-Sandman-Polymer-Property-Prediction-Codebase/` | the clean, public submission codebase: pipeline, notebook, evidence, website. **No agent files in here — keep it that way.** | `<codebase>/README.md` §15 "For reviewers" |
| `Consolidation/` | the historical archive: every artifact from Round 1, Round 2 and Round 3. Things here are *found*, not run. | `Consolidation/AGENTS.md` |

Root also holds: `CONTEXT.md` (portable one-page context), `README.md` (human map),
`PLAN.md` (the consolidation contract this delivery was built from), and **`RUN.md`** (every
command that still needs running, with its expected output).

## Route by request

| the request | go to |
|---|---|
| "prepare my presentation" | `Personal/Presentation/PROMPT_PRESENTATION.md` + `SLIDE_PLAN.md` |
| "write the midnight report" | `Personal/Midnight_Report/PROMPT_10PAGE.md` (or `PROMPT_3PAGE.md`) |
| "what do I say if they ask X" | `Personal/docs/11_qna/` |
| "what is our score / any number" | `Personal/docs/00_INDEX.md` — the canonical block |
| "run the notebook / verify something" | **`RUN.md`** |
| "explain the model" | `<codebase>/ARCHITECTURE.md` |
| "what did we try" | `<codebase>/Experiment_Logs/` (curated 72) or `Personal/TRIALS.md` (everything) |
| "what do we run next / Phase 6" | `Personal/Score_and_Invariance_Improvement/PLAN.md` + `PROMPT.md` |
| "where did file X come from" | `Consolidation/MANIFEST.md` |
| "start the demo" | `<codebase>/Website/README.md` |

## Five rules that apply everywhere

1. **Never write "oracle"** (or khazana / polyinfo / TgSS / test_answers) into `<codebase>/` or
   anything that could be pasted publicly. Say **"local held-out verification panel"**.
2. **Never invent a number.** The canonical block is `Personal/docs/00_INDEX.md`. If a number is
   not there and not in a named file, it does not exist.
3. **Never touch `Personal/Obsidian/`**, `Obsidian.zip` or `.obsidian/`. They are the user's own.
4. **Never modify the GPU laptop.** Read-only, always.
5. **python 3.11.7 is load-bearing.** On 3.12 the ei target collapses 0.871 → 0.512 regardless of
   package versions. A far-below-0.90 score with identical code is an environment mismatch.
