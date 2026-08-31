# NEXT STEPS — how to regenerate the submission with the correct environment

> **Status (2026-08-31):** the deliverable is complete and committed. One
> environment-sensitive step remains (the V57 submission regeneration on the
> GPU laptop) because the ei/eea leaf models are **python-build sensitive**.
> Everything below is logged so the next agent (or the user) can finish in
> one command without re-deriving anything.

---

## 1. Why a manual regeneration step exists (do not skip)

The **frozen submission is already valid**: `CODEBASE/submission_v57.csv`
scores verified **0.90348** / final_oracle **0.90229** (hash
`bfa1dc3b…`), and that file is what gets submitted.  It was generated under
**python 3.11.7**.

Trying to regenerate it on the laptop with python 3.12.x **collapses ei**
(0.871 → 0.512; mean 0.9023 → 0.8469), *regardless of numpy/rdkit/pandas
versions* — two python-3.12 envs with different package versions collapsed
identically.  **python 3.11.7 is the load-bearing factor.**

The Round-3 **evidence suite (Part B) is version-robust** — proxy models
reproduce on any of these envs.  Only the V57 submission path (Part A) is
python-build sensitive.

## 2. The one command (already staged on the GPU laptop)

The **python 3.11.7 venv already exists** on the GPU laptop at:

```
/tmp/r3_py311_venv   (uv-created; numpy 2.4.6, pandas 3.0.5, sklearn 1.9.0,
                      rdkit 2026.03.5, scipy 1.17.1, lightgbm 4.7.0 — verified)
```

Regenerate the submission + full evidence bundle in ONE run:

```bash
# from the Mac repo, copy the final file (if not already there):
scp CODEBASE/pipeline_final.py vishwa@100.116.22.29:/tmp/r3_final_run/

# on the laptop:
cd /tmp/r3_final_run
nohup /tmp/r3_py311_venv/bin/python -u pipeline_final.py \
  --mode full \
  --data-dir ~/Desktop/r3_runtime/Phase_4_Explainability/Dataset \
  --out /tmp/r3_final_run/submission_final.csv \
  --out-dir /tmp/r3_final_run/outputs \
  > /tmp/r3_final_run/run_py311.log 2>&1 &
# ~2.5–3 h. Then copy results back:
scp vishwa@100.116.22.29:/tmp/r3_final_run/submission_final.csv .
```

**Expected result:** submission hash ≈ `bfa1dc3b…` (byte-parity with the
frozen 0.9035) — this closes DEFECT-3.  If the hash matches, update
`final_submissions/README.md`; if not, the remaining delta is DEFECT-1
(chain leaf rebuild variance, documented) — the score will still be ≈ 0.9023.

## 3. Sanity checks after the run

```bash
# score the regenerated submission (Mac):
.venv/bin/python Oracle/score_round2_ORACLE_ASSISTED_RESEARCH_ONLY.py \
  --candidate submission_final.csv --verified Oracle/oracle.csv \
  --proxy Oracle/final_oracle.csv --output /tmp/score_py311.json
# expect verified_mean_r2 ≈ 0.9035 (ei ≈ 0.871, NOT 0.512)

# verify the evidence scorecard:
cat outputs/scorecard.md     # expect 14–15/19 PASS incl. AUG + REL
```

## 4. If you are converting to ipynb (user plan)

Follow `CODEBASE/README.md` → "Converting pipeline_final.py to a Kaggle
notebook": Part A as one big cell, Part B as one cell, CLI last.  The Kaggle
image runs python 3.11.x — if it ships numpy ≥ 2.5, `pip install numpy==2.4.6`
in cell 1 and verify the written `submission.csv` per-target means against
this repo.

## 5. What is already done (do not redo)

- `CODEBASE/pipeline_final.py` — single standalone file: V57 submission
  engine (byte-identical core, 570,044 chars verified) + full Round-3
  evidence engine (R1 explainability, R2 invariance, R3 reliability,
  R4 generalization, AUG data-augmentation, REL homologous-series).
- `CODEBASE/evidence_engine.py` — standalone source of Part B.
- `CODEBASE/outputs/` — full-data evidence from the laptop Phase-4 run
  (160 artifacts) + regenerated scorecard/HTML; upgraded technique numbers
  come from the evidence-v2 run already in flight on the laptop.
- `CODEBASE/README.md`, `ARCHITECTURE.md`, `outputs/SESSION_SUMMARY.md`,
  `FINAL_REPORT.md`, `CONTEXT.md`, `AGENTS.md` — all updated.
- Env pins: `CODEBASE/requirements.txt` (python 3.11.7 load-bearing).
