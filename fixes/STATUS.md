# Fixes status — hackathon-rubric pass

Last updated: 2026-09-03 (completed isolated run staged)

## Current state

- `isolated_runs/Sandman_Polymer_Property_Prediction_2_906.ipynb` completed under its isolated
  Python 3.11.7 environment. Its notebook, full outputs, both emitted CSVs and aggregate GNN
  artifact were copied intact to the submission codebase at
  `904_submission/final_notebook/`; source outputs remain untouched in `isolated_runs/`.
- The GNN cell's fold loop was repaired without execution: the late submission-engine
  cell returns fold IDs, so the GNN cell now converts them into local train/validation
  index pairs. Existing notebook outputs and isolated artifacts were retained.
- The isolated environment is Python 3.11.7 with verified pinned core dependencies.
- The completed run produced 184 output artifacts and a later scorecard with 17/19 checks PASS.
  The two remaining scorecard failures are cross-model explanation agreement and post-freeze
  external verification. The newly emitted final CSV is schema-valid but still awaits a score-to-
  hash binding before it can be called the final released candidate.
- The prior recorded full-run evidence has been consolidated into
  `qualitative_evidence/figures/qualitative_scorecard.{csv,png,pdf}`. It is
  presentation material only until the isolated run reconfirms it.
- `pure_ml/outputs_full/` is a completed independent ExtraTrees baseline; its mean
  target-wise grouped-CV R² is 0.816344, so it is not a replacement candidate.
  Its output schema, finite predictions and model bundle are recorded in
  `pure_ml/VERIFICATION.md`.
- `experiments/tabpfn/` is **approved to resume**. A delegated, self-contained
  smoke-then-full run is pending; do not read or display its `.env`.

## Active plan and ownership

| Stage | Status | Output / next action |
|---|---|---|
| 1. Evidence inventory | COMPLETE (recorded run) | Feature/design evidence, qualitative CSVs, website capability and allowed citations audited. Fresh notebook confirmation remains pending. |
| 2. Independent qualitative checks | COMPLETE — scoped evidence staged | Completed scorecard, CSVs and figures were copied with the final notebook. Retain scope boundaries in `904_submission/final_notebook/RUN_RECORD.md`. |
| 3. Feature and architecture rationale | COMPLETE, release-gated | `qualitative_evidence/claim_evidence_map.md` links claims to artifacts and records the unresolved archive conflicts. Reconfirm from the isolated run before promotion. |
| 4. Website demo audit | COMPLETE | All website numbers verified against evidence tables. Invariance caption corrected to exact values (0.947-0.996 SHAP cosine, 0.23% graph-feature std, 0.0000 violation rate). Literature anchors, visual explanation, candidate screening, AD tiers, conformal intervals, and honest footer all evidence-backed. |
| 5. Report/presentation/story | PARTIAL | `finals/Report.md` is aligned to the official four-section check-in template. Presentation and website implementation remain separate work. |
| 6. Promotion | PARTIAL | Notebook/output promotion is complete. Final release remains blocked only on selecting one CSV and attaching a score record to that exact hash. |

## Rubric evidence already available

| Theme | Recorded evidence | Presentation-safe phrasing |
|---|---|---|
| Invariance | Graph representation: zero tested violations; mean explanation cosine 0.979737 | Equivalent polymer spellings preserve prediction and explanation. |
| Explainability | Feature-removal fidelity is recorded; the scorecard and raw table differ slightly | Present only with its source/proxy scope; refresh after the rerun. |
| Robustness | Tg error rises 14.77 → 43.56 °C across similarity tiers; seed stability is recorded | The applicability-domain boundary is verified. Do **not** claim calibrated coverage until rerun. |
| Generalizability | Raw ladder means: 0.823783 (canonical-group), 0.660064 (scaffold); all 7 targets positive | Performance persists under stricter structural splits, with expected degradation. |

## Boundaries to retain

- Raw cross-model feature-rank agreement is 0.472223; present it as a secondary
  sensitivity diagnostic, not as the primary explanation-fidelity test.
- The archived conformal-coverage and error--uncertainty tables contradict the old
  favorable scorecard. They are a release gate, not a public success claim, until
  `validate_archive_evidence.py` has run on the completed isolated output.
- Extreme family/low-similarity and tail extrapolation are not the claimed operating
  regime. The demo must show an applicability tier and uncertainty rather than imply
  reliable extrapolation.
- No external labels, secret artifacts, or unverified new scores may be used in public
  material. Use the phrase “local held-out verification panel” where a protocol name
  is needed.

## Handoff commands

```bash
# Rebuild qualitative presentation evidence from a completed output directory
fixes/isolated_runs/.venv/bin/python fixes/qualitative_evidence/summarize_qualitative.py \
  --outputs <completed-output-dir> \
  --output-dir fixes/qualitative_evidence/figures

# Validate the four release-gate qualitative CSVs
fixes/isolated_runs/.venv/bin/python fixes/qualitative_evidence/validate_archive_evidence.py \
  --outputs <completed-output-dir>
```
