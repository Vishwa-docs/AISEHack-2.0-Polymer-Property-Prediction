# Closure audit status

**Started:** 2026-09-03

## Scope

Release-readiness audit of the public codebase and supporting isolated evidence.
The user-operated `fixes/isolated_runs/` notebook is active and is deliberately
out of scope for execution or modification.

## Current stage

- [x] Audit workspace and handoff instructions created.
- [x] Public-release hygiene and leakage review.
- [x] Reproducibility / execution-path review.
- [x] Claim, score, citation, and evidence-artifact traceability review.
- [x] Promised-work completion review.
- [x] Prioritised close-out checklist.

## Known starting cautions

- The live full evidence run has not yet been accepted as a promotion gate.
- The website's strict repeat demonstration is intentionally limited to a declared
  linear-repeat grammar; it is not a universal PSMILES proof.
- The portable website predictor is not the same as the full assembled submission
  pipeline, so its score and capability must be labelled separately.

## Result

See `AUDIT.md`. Principal blockers: the documented command does not execute the
advertised GNN Phase-7 system; the headline score is not tied to the canonical CSV
hash; selected model-selection paths use ordinary KFold despite broad grouped-CV
claims; evidence is stale/mixed and proxy scoped; and interval calibration fails its
own criterion. No hidden-label access was found in reviewed readable source, but
transductive feature construction and exact train-label routing need rule clearance.
