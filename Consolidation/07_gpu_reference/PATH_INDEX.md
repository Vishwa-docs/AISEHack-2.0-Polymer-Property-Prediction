# GPU laptop — path index

**Read-only.** Nothing on that machine may be modified, except a scratch directory you create
under `/tmp` or the Phase-6 workspace the user has authorised at
`~/Desktop/r3_runtime/Phase_6/`.

## Hardware

RTX 5090 laptop GPU (24 GB VRAM) · 62 GB RAM · 24 cores · ~600 GB free disk.
**One heavy job at a time**, 20% RAM/VRAM headroom.

## Paths

| path | what is there | copied to the Mac? |
|---|---|---|
| `~/Desktop/AISEHack-2.0/` | the Round-1 and Round-2 codebase (a git repo with a safety tag) | **read-only reference.** The relevant Round-2 material is in `../02_round2/` |
| `~/Desktop/AISEHack-2.0/.venv-polymer/bin/python` | python 3.12.3 with numpy 2.4.6, sklearn 1.9.0, torch 2.11 + cu128, torch-geometric | usable for GPU work — **but see the warning below** |
| `~/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/` | 375 clean Round-2 experiments, logs, research, 368 tools | referenced, not bulk-copied |
| `~/Desktop/r3_runtime/Phase_2/` | 151 mechanism-sweep scripts | scripts mirrored in `../04_phases/phase2_mechanism_sweep/` |
| `~/Desktop/r3_runtime/Phase_3/` | 282 clean-stack scripts | scripts mirrored in `../04_phases/phase3_clean_stack/` |
| `~/Desktop/r3_runtime/Phase_4_Explainability/` | the evidence suite: 38 analysis scripts, ~2.5 GB of outputs | **scripts and docs only**; the curated outputs are in `../04_phases/phase4_explainability/` |
| `~/Desktop/r3_runtime/Phase_4_Explainability/.venv/` | **numpy 2.5.2 — collapses ei/eea. Do not use for the submission path** | — |
| `/tmp/r3_py311_venv` | a python **3.11.7** venv (numpy 2.4.6, pandas 3.0.5, sklearn 1.9.0, rdkit 2026.03.5, scipy 1.17.1, lightgbm 4.7.0) — **the correct environment for the submission path** | a `/tmp` path: may be wiped on reboot, recreate from `requirements.txt` |
| `~/Desktop/r3_runtime/Phase_6/` | **the Phase-6 workspace** (see `../../Personal/Score_and_Invariance_Improvement/PROMPT.md`) | results copied back to `Personal/Score_and_Invariance_Improvement/results/` |
| `~/Desktop/AISE Full Codebase.zip` | 66.7 GB | **never copy** |

## The environment warning, once more

The submission path's ei/eea leaf models collapse from R² **0.871 to 0.512** under python 3.12.x
*regardless of package versions*, and to 0.516 under numpy ≥ 2.5. **Use `/tmp/r3_py311_venv`
(python 3.11.7) for anything that regenerates a submission.** The evidence suite is
version-robust and may use either.

## What we deliberately do not copy back

Bulk outputs, model blobs, the corpora, and anything above ~50 MB. Scripts, documents, metrics,
logs and small result files only.
