# Invariance and qualitative evidence workstream

This is an isolated, non-public validation workspace. It must not modify or execute
`../isolated_runs/`, including the user-operated notebook.

## Objective

Validate two distinct polymer-representation claims before they are shown in the demo:

1. **Translation/cut-point invariance:** equivalent linear repeat windows map to one
   primitive repeat.
2. **Repetition invariance:** monomer, dimer and trimer spelling maps to that same repeat.

The scope is intentionally limited to the validated linear-repeat grammar. Unsupported
PSMILES must be reported as unsupported, never normalised speculatively.

## Rules

- Keep every generated file under this directory.
- No training, no hidden labels, no external data and no model changes.
- The compact portable predictor may be read only to evaluate a fixed demo panel.
- A public website integration may happen only after `run_panel.py` passes. It must use
  the tested implementation and label its supported scope.
- Update `STATUS.md` after each material validation or integration step.

## Required checks

- Every panel row has exactly two star endpoints.
- All variants in a family produce the same primitive normalised PSMILES.
- The compact model receives the same normalised input for each strict variant.
- All seven target predictions are finite and have zero numerical spread, subject only
  to deterministic floating-point equality.
