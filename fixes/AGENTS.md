# Fixes workspace handoff

`fixes/` contains isolated, non-public workstreams. Do not copy artifacts into the
submission codebase without an explicit verification step.

| Folder | Purpose | Run policy |
|---|---|---|
| `isolated_runs/` | User-operated full notebook reproduction | Do not run unless the user explicitly asks; write only inside this folder. |
| `pure_ml/` | Independent classical baseline | Outputs must remain under `pure_ml/outputs*`. |
| `experiments/` | Research-only experiments | Never promote pretrained-model results as plain ML. |
| `qualitative_evidence/` | Audits and presentation figures | Treat recorded evidence as provisional until an isolated rerun confirms it. |

Never read, print, commit, or transmit secret files such as `experiments/tabpfn/.env`.
Use Python 3.11.7 from `isolated_runs/.venv` for runnable work.

## Active hackathon-rubric plan

Follow these stages in order. Record material status changes in `STATUS.md` before
handing this work to another agent.

1. **Evidence inventory — in progress.** Map every feature, architecture choice,
   qualitative claim, chart, and citation to a named source artifact. Reject claims
   that lack a reproducible table or a source in `Personal/Research/INDEX.md`.
2. **Independent qualitative checks — planned.** Use scripts in
   `qualitative_evidence/`, never the notebook the user is running, to consolidate
   invariance, robustness, generalizability, and explainability results. Preserve
   boundary cases rather than optimizing away failures.
3. **Feature/architecture rationale — planned.** Audit feature families, per-target
   lanes, physical relations, and known failures. Produce a claim-to-evidence map
   suitable for the report and speaker notes.
4. **Demo and website — planned.** Audit the existing website before modifying it.
   If changed, add a live polymer prediction, uncertainty/applicability, explanation,
   and same-polymer representation comparison without exposing any secret labels.
5. **Story and presentation — planned.** Update only after stages 1–4 yield verified
   material: keep slides sparse, place methods/references in speaker notes, and use
   the qualitative scorecard plus live demo as proof.
6. **Promotion gate — blocked on isolated run.** Compare new outputs from
   `isolated_runs/` with recorded evidence, then and only then copy verified work to
   the public codebase or final report/deck.

TabPFN is explicitly paused. Do not resume it without a fresh user request.
