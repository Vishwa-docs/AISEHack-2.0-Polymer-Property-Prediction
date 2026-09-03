# TabPFN Experiment Status

## Current State: COMPLETED ✅

Both smoke test and full evaluation completed successfully.

## Summary

| Stage | Status | Exit Code | Output |
|-------|--------|-----------|--------|
| Environment Creation | ✅ Done | 0 | `.venv/` created with Python 3.11.7 |
| Package Installation | ✅ Done | 0 | tabpfn 8.5.0 + dependencies |
| Smoke Test | ✅ Done | 0 | `outputs/smoke/tabpfn_grouped_cv.csv` |
| Full Evaluation | ✅ Done | 0 | `outputs/full/tabpfn_grouped_cv.csv` |
| Manifest Written | ✅ Done | — | `run_manifest.md` |
| Status Written | ✅ Done | — | `STATUS.md` |

## Key Results

**Full Evaluation (Grouped-CV R², 3 folds per target):**

| Target | Samples | R² |
|--------|---------|-----|
| egb | 337 | 0.8861 |
| ei | 222 | 0.8008 |
| eea | 221 | 0.8701 |
| nc | 229 | 0.8263 |
| eps | 229 | 0.7831 |
| **Mean** | 1238 | **0.8333** |

## Compliance

- ✅ All caches/outputs confined to `fixes/experiments/tabpfn/`
- ✅ Python 3.11.7 used (load-bearing per AGENTS.md rule 5)
- ✅ `.env` never exposed
- ✅ No authentication bypass
- ✅ Results labeled as **pretrained, research-only**
- ✅ Not promoted to contest pipeline
- ✅ Not directly comparable to held-out 0.907551 score

## Next Steps

None required. Experiment complete. Results recorded in:
- `outputs/full/tabpfn_grouped_cv.csv` (primary result)
- `run_manifest.md` (full execution record)
- `logs/full.log` (stdout/stderr)