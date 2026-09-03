# Status — invariance and qualitative workstream

## Plan

1. Implement a conservative primitive-repeat normaliser for a declared linear PSMILES
   grammar.
2. Validate translated/cut-point, monomer, dimer and trimer forms on a fixed PEO panel.
3. Evaluate the compact portable predictor on the normalised forms and write an auditable
   result table and figure inside this folder.
4. Only if the panel passes, add the same implementation to the offline website as a
   labelled `Strict representation check` interaction.
5. Smoke-test the website module and retain ordinary prediction for arbitrary user input.

## Completed validation

- `run_panel.py` passed on 2026-09-03 using the pinned Python 3.11.7 environment at
  `../isolated_runs/.venv`. It performs no training.
- The fixed PEO panel covers `*CCO*`, translated `*OCC*`, dimer `*CCOCCO*`, and trimer
  `*CCOCCOCCO*`.
- Every form normalised to `*CCO*`; all seven compact-model predictions had maximum range
  exactly `0.0`. See `results/manifest.json` and `results/invariance_panel_summary.csv`.
- The result CSV SHA-256 is
  `bf8e8a53c329b1958690ce379b6bd65456e141024b1ee3f72571cab0e30736e6`.
- The tested normaliser was copied into the offline website as
  `Website/repeat_invariance.py`. The website interaction leaves ordinary arbitrary-input
  inference unchanged and clearly labels unsupported strict-repeat input.
- Python syntax checks passed for the isolated scripts, website module, Streamlit app and API.
- A predictor-backed website module smoke test passed. API runtime smoke testing remains
  pending only because FastAPI is not installed in the isolated notebook environment; it is
  already declared in `Website/requirements-web.txt`.

## Next action

Install the existing website requirements in the codebase's own demo environment and launch
Streamlit for an interaction check. Do not install packages into `../isolated_runs/.venv` while
the user notebook is active.

## Scope boundary

The strict check is a validated linear-repeat demonstration, not a universal PSMILES parser.
Complex, branched, stereochemical, copolymer or ambiguous-attachment input must be shown as
outside the strict panel while remaining eligible for ordinary model inference.
