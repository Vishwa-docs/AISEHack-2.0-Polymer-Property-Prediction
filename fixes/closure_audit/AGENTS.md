# Closure-audit handoff

This directory is an isolated, read-only release-readiness audit. It must not modify
the active notebook, datasets, model artifacts, or the public codebase.

## Scope

1. Trace public claims to executable code and named artifacts.
2. Check data-flow and release hygiene for leakage, hidden labels, private paths,
   credentials, and non-compliant dependencies.
3. Distinguish reproduced facts from historical claims and planned work.
4. Record gaps and a smallest safe closure checklist in `AUDIT.md` and `STATUS.md`.

## Rules

- Read `../AGENTS.md` before working in `fixes/`.
- Do not run, edit, restart, or clear `../isolated_runs/` while the user is running it.
- Do not read or expose any `.env` file.
- Do not use or copy data outside the official competition inputs.
- Treat every score as unverified unless a submission identity, code version,
  environment, and result artifact are all linked.
- Never promote a finding from this directory into the public codebase without the
  user's explicit instruction and a fresh verification run.
