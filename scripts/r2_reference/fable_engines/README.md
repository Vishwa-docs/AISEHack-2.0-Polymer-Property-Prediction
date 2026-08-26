# Fable research session — reference implementations (NOT executed)

Created 2026-08-05 by the Claude (Fable) research session. These files are
**specifications in code form** for the F-series experiment ladder described in
`../../Fable_Findings.md`. They were written but deliberately **not executed** —
the user runs all experiments. No existing file in the repository was modified.

Contents:

| File | Role |
|---|---|
| `fable_common.py` | Shared harness: official-input hash checks, canonicalization, structure-grouped and Butina-cluster folds, and the **shift-matched R² metric** (OOF reweighted to the test NN-similarity histogram) that every F-experiment uses as its decision metric. |
| `F01_ei_eea_egb_chain_engine.py` | Availability-stratified chained identity engine for ei / eea / egb (arms A0–A3). |
| `F02_eps_nc_ionic_engine.py` | eps/nc joint physics engine in ionic coordinates (arms B0–B2 + eps ≥ nc²+0.02 constraint). |

Compliance notes:

- Reads only `ppp-round-2/train.csv`, `ppp-round-2/test.csv`,
  `ppp-round-2/archive/train.csv`, and `ppp-round-2/PI1M.csv` — hashes verified
  at load. `archive/train.csv` is a permitted official bundle file and supplies
  exact same-property labels for 1,645 `tg` and 804 `egc` test rows;
  `exact_lookup_table` exposes per-group conflict spread so a caller can abstain
  on the 6 conflicting `tg` groups.
- Never reads the oracle, any prior prediction artifact, or any file under
  `ORACLE_ASSISTED_RESEARCH_ONLY/`.
- Partner **labels** (a different property measured on the same canonical
  structure, present in official train/archive) are legitimate test-time
  features and are never masked. Partner **predictions** used to fill missing
  labels are cross-fitted with the evaluated row's structure excluded — the
  fix for the C132/C139 circular-fallback incident.
- Reports are written to a new versioned directory under
  `experiments/CLEAN_OFFICIAL_ONLY/`; existing artifacts are never overwritten
  (`save_report` refuses to overwrite).

Other `F0*.py` files in this directory were added by a later session and are not
part of this reference set. Note that `load_data()` includes `archive/train.csv`
labels in `wide`/`all_labels`; any runner written against an archive-free variant
of this harness has changed meaning and must be re-run.

Read `Fable_Findings.md` (folder root) first — it contains the full experiment
ladder (F01–F10), preregistered gates, and expected per-target outcomes.
