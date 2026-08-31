# Consolidation_Handoff — read me first

Produced 2026-08-31 by the planning agent. **Nothing outside this folder was created,
moved or modified.** The source repo's git working tree was clean at HEAD `9db3154` before
and after this session (there was nothing to commit).

| file | what it is |
|---|---|
| **`PLAN.md`** | **The execution contract.** ~1,890 lines. Read it completely before acting. |
| `research/EDA_VERIFIED_FACTS.md` | Every dataset statistic, computed live from `train.csv`/`test.csv` with the pinned env and verified. Novelty, partner availability, physics identities, correlations with sample sizes, variance shares, tail concentration, replicate structure. **Cite it; do not recompute it.** |
| `research/POLYMER_DOMAIN_PRIMER.md` | The polymer science behind all seven targets, why each needs a different model, the five research gaps, and ten QnA answers. |
| `research/RESULTS_ANALYSIS.md` | The canonical scoreboard, the public/private gap decomposition, the mathematical ceiling, the full experiment ledger (~1,150 runs), what worked and what failed with numbers, and the Round-3 evidence results. |
| `research/SOURCE_INVENTORY.md` | Every path on the Mac and the GPU laptop, what is in it, its size, and where it should end up. |

## What the planning agent did and did not do

**Did:** read the whole Round-3 repo, both Mac folders, and the GPU laptop (read-only);
ran three EDA probes with the pinned interpreter to verify dataset facts; ran eight web
searches for the bibliography seed; wrote the plan and the four research documents.

**Did not:** move any file, modify any existing file, create anything in the destination
folder (the file sandbox denied writes there — the executing agent will need a wider scope),
run any pipeline, change any git repo, or touch the GPU laptop.

## Three things the executing agent should read first

1. `PLAN.md` §2 — the canonical numbers and the D1–D9 taxonomy. Everything downstream
   depends on these being used consistently.
2. `PLAN.md` §15.2 — the ten known contradictions in the existing material, with their
   resolutions. Fixing these is most of the credibility work.
3. `PLAN.md` §17 — the execution order. The notebook must be built and **run** before any
   prose that quotes a number is written.

## Suggested first action

Create `PROGRESS.md` next to this file and append one timestamped line per completed step.
If your session ends, the next agent resumes from it.
