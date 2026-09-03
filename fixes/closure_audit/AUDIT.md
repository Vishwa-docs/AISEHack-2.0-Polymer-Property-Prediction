# Release-readiness audit

This file will contain only results traceable to a path, command, code location, or
research source. It is a closure checklist, not a replacement for the competition
submission documentation.

## Status key

- **PASS** — directly verified in this audit.
- **CONDITIONAL** — supported, but dependent on a stated scope or missing runtime check.
- **GAP** — unverified, inconsistent, or not release-ready.

## Findings

**Audited:** 2026-09-03  
**Decision:** **not release-ready as one reproducible public submission.** No hidden-label
access was found in reviewed readable source, but the headline score, documented executable
path, evidence bundle, and website do not yet describe one identical artefact.

## Verification summary

| Area | Status | Evidence |
|---|---|---|
| Python syntax | **PASS** | `ast.parse` passed all 15 public Python files. This is syntax only, not an end-to-end run. |
| Source hygiene | **PASS, bounded** | No prohibited hidden-label identifiers, local user paths, credentials, or remote host names found in scanned readable public source/docs. `os.environ` matches were normal configuration reads. |
| Submitted file identity | **PARTIAL** | `submission.csv` and `904_submission/submission_final.csv` are byte-identical: SHA-256 `cd91f2785dd9b7704d5db1a9f59f251a25937726d48522d1c39f66be504f66f5`. |
| Strict repeat demo | **PASS, narrow scope** | The PEO primitive/translated/dimer/trimer panel normalises to `*CCO*` and produces exact zero compact-model range. Result CSV SHA-256: `bf8e8a53c329b1958690ce379b6bd65456e141024b1ee3f72571cab0e30736e6`. |
| Website runtime | **GAP** | Code and Predictor-backed strict-repeat smoke passed; Streamlit/FastAPI have not been launched in a dedicated website environment. |

## P0 — release blockers

### P0.1 Documented command does not execute the advertised Phase-7 system

README/ARCHITECTURE call the 0.907551 method “V57 + GNN”. The documented command runs
`src/pipeline_final.py --mode full`; its `run_v57` path contains no GINE/GNN execution, and
there are no GNN dependencies in `requirements.txt`. A source scan finds no GNN runtime in
the public pipeline.

**Close:** either integrate the checkpointed GNN blend into the documented deterministic
command (code, weights, environment, blend rule, output hash), or call the public command
**V57 only** and scope the Phase-7 claim to a separate reproducible artefact. Never replace
this with cached predictions.

### P0.2 Score provenance is not tied to the final CSV

`score_final_verified.json` records mean R² 0.9075505507 but no CSV hash. The later
`notebook_metrics.json` records a different submission hash (`d48793b1…f250`), while the
public root evidence tables still report the older 0.9023 verification values. Thus the
0.907551 claim may be genuine but is not currently attributable to the canonical `cd91…f66f`
CSV.

**Close:** write one immutable `release_manifest.json` beside the chosen CSV: CSV SHA-256,
git revision, command, interpreter/platform/package versions, dataset-schema digest,
per-target local-panel metrics, scoring-script digest, timestamp, and platform submission ID
plus measured public result. Keep estimates explicitly labelled estimates.

### P0.3 Not every selection-bearing validation path is grouped

`fit_targets` uses shuffled ordinary `KFold` at `src/pipeline_final.py:333`; `run_v57`
invokes C282/C284/C285 through that path. Its final character-residual tail uses ordinary
`KFold` at line 9395. This conflicts with the broad “canonical-SMILES GroupKFold” claim.

This is **not hidden-label leakage**, but identical canonical structures can straddle the
internal OOF split and make selection optimistic.

**Close:** group every fit, residual, calibration, and stack-selection split by canonical
SMILES, or withdraw the global grouped-CV claim and state exactly which evidence proxy is
grouped.

### P0.4 Dashboard intervals are not currently verified 90% intervals

`conformal_coverage_table.csv` has maximum absolute coverage error 0.089, and
`error_uncertainty_correlation.csv` has correlation ≥0.30 for only one target. The shipped
scorecard marks both FAIL, but the website displays a “90% conformal interval”.

**Close:** until a fresh correctly split audit passes, label it “experimental uncertainty
interval — calibration pending”. The applicability tier can remain, but cannot be presented
as repairing coverage.

### P0.5 Oligomer / Flory–Fox wording exceeds the artefacts

The archived oligomer table contains substantial monomer–dimer changes and only declares a
permissive \|delta\| < 3 training-standard-deviation rate. Root `outputs/` lacks the
homologous-series files linked by README. The fresh exact result is only the declared linear
PEO grammar with the compact predictor.

**Close:** demo the exact PEO result with its narrow scope. Do not say all PSMILES or all
polymers are oligomer invariant. Treat Flory–Fox as molecular-weight literature motivation,
not validation from arbitrary repeated text strings.

## P1 — credibility and hand-off gaps

| Gap | Evidence | Required closure |
|---|---|---|
| Mixed evidence tree | Root scorecard is 14/18 from 2026-08-31; root is missing relation/augmentation artifacts, while later artifacts live under `904_submission`. The active isolated run has not yet emitted its late gate CSVs. | Finish it without interruption, validate it using the existing scripts, and publish one matched output tree only. |
| Proxy-model evidence | `evidence_engine.py` trains Ridge/ExtraTrees/LightGBM proxies. Fidelity masks a sampled **training** subset (e.g. Tg baseline 0.978), not the full V57/GNN DAG. | Label all such plots “proxy-model diagnostic”; do not call them final-pipeline explanations or causal chemistry. |
| Proxy generalization failures | The ladder includes negative family-split results (e.g. Tg G3 −0.409). | Show expected degradation / AD boundaries, not blanket claims of new-family generalization. |
| Duplicate label transfer | `apply_official_overrides` copies a unique train label for exact raw/canonical structure–target matches. Train-label only, not answer-key access. | Retain organiser rule approval, route counts, and label it duplicate-measurement routing rather than ordinary model inference. |
| Transductive construction | Features use the union of official train + test structures; this is disclosed in `weights/README.md`. | Confirm rules allow unlabeled test structures at fit time; otherwise refit train-only. Separate fixed-panel performance from novel-polymer deployment. |
| Incomplete requirements | Source conditionally imports CatBoost, Mordred, and `polymer_property_prediction`; requirements do not pin them. | Lock the exact final environment or delete unused paths. State macOS/Linux rdEHT limitation plainly. |
| Public-tree hygiene | Tracked notebook symlinks point outside the repository; `Dataset` is local; an internal notebook `STATUS.md` remains. | Export a clean cloneable tree with no data/local symlinks/agent hand-off metadata. |
| Website score mismatch | Root README says historical public 0.917; Website README says 0.920. | Use one provenance-bound measured number only. |
| Literature panel | Only PS/PMMA illustrative anchors found; no reproducible cited external panel. | Keep them illustrative or build a sourced table with conditions, mapping, units, and model version. |

## Research-backing check

- **PolyGNN** supports testing repeat-unit translation, addition and subtraction, and motivates
  periodic graph construction. It does not prove this model is invariant without its own test.
  Gurnani et al., *Chemistry of Materials* (2023), DOI `10.1021/acs.chemmater.2c02991`.
- **Flory–Fox** supports an empirical Tg–molecular-weight relation within a homologous polymer
  family; it does not justify arbitrary string-repetition experiments as physical measurement.
  Fox & Flory, *Journal of Polymer Science* (1954), DOI `10.1002/pol.1954.120147514`.
- **TreeSHAP** supports tree-feature attribution and aggregation of local explanations, not a
  causal chemical claim. Lundberg et al., *Nature Machine Intelligence* 2, 56–67 (2020), DOI
  `10.1038/s42256-019-0138-9`.
- **Conformal prediction** offers coverage guarantees only under its assumptions; the project's
  failed empirical coverage diagnostic overrides any generic-theory claim. Angelopoulos &
  Bates (2021), arXiv:2107.07511.

## Promises-to-proof matrix

| Promise | Honest current state |
|---|---|
| Reproducible full Phase-7 pipeline | **GAP** — code + GNN + CSV + score manifest not bound. |
| Random-SMILES robustness | **CONDITIONAL** — stable graph proxy features; final assembled procedure awaits fresh evidence. |
| Translation and dimer/trimer invariance | **CONDITIONAL** — exact only for the scoped PEO compact-model panel. |
| Explainability | **CONDITIONAL** — proxy SHAP/fidelity exists; cross-model agreement fails and final-model scope is absent. |
| Generalization / robustness | **CONDITIONAL** — proxy scaffold/similarity ladder exists; final-pipeline structural validation is absent. |
| Pure-ML deployable alternative | **PARTIAL** — valid 0.816344 grouped-CV baseline, not a score replacement. |
| Interactive dashboard | **PARTIAL** — implementation/smoke only; browser runtime pending. |
| Literature panel | **GAP** — anchors only. |

## Smallest safe closure order

1. Do not touch the active notebook. After it finishes, run the existing isolated qualitative
   validation and retain its manifest + scorecard together.
2. Select one final CSV, bind it to a scored release manifest, and reconcile all score text.
3. Decide whether the GNN is public; do not claim it before its documented command reproduces
   the selected file.
4. Correct validation scope: group all selection paths or narrow the claim.
5. Refresh only evidence belonging to that exact release and retain all FAIL results.
6. Launch the website once; add the experimental-interval caveat and visible PEO-scope label.
7. Export a clean public tree without data, local symlinks, or agent metadata.
