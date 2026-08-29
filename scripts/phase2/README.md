# Phase_2 — Round 3 Experiment Suite (150 real experiments)

Everything in this folder runs on the GPU machine only
(`~/Desktop/r3_runtime/Phase_2/`).  Every experiment is a REAL standalone
script that reads ONLY the official `Dataset/` inputs, trains from scratch with
fixed seeds, evaluates with structure-grouped folds + panels, and writes its
outputs.  No placeholders, no oracle references, no old-file references.

## Layout

    Phase_2/
      r3_core/                  shared library (data/features/models/metrics/physics/panels/engine)
      experiments/              exp001_*.py .. exp150_*.py   (one real experiment each)
      tests/                    pytest suite verifying every experiment is real
      run.sh                    sequential runner (visible status in terminal)
      outputs_and_logs/
        output/<exp_name>/      VALUES ONLY: metrics.json, predictions.csv, oof_values.csv, decision.md
        logs/<exp_name>.log     full run log
        logs/summary.tsv        one row per experiment (status + mean R2)

## How to run

    ./run.sh                 # run experiments 001..150 in order
    ./run.sh 1 5             # run only experiments 001..005
    ./run.sh 1 1 --smoke     # smoke mode for one experiment (fast)
    PYTHON=python3 ./run.sh  # override the interpreter

While running you will see, in the terminal, each experiment's status as it
starts and finishes:

    ----------------------------------------------------------------------
    [007/150] RUNNING  exp007_bB07
               started 12:31:05
    ----------------------------------------------------------------------
    [007/150] COMPLETED exp007_bB07  (142s)  mean R2 = 0.4213

## Verify the suite

    python3 -m pytest tests/ -q     # imports + smoke-run every experiment

## Experiment inventory (150)

Phases per PLAN.md: A (V57 reproduction, 5) · B (selection repair, 12) ·
C (weak-target physics, 20) · D (Tg push, 12) · E (SSL at scale, 20) ·
F (TTA/invariance/explainability, 12) · H (GBM breadth/tuning, 15) ·
I (weak-target zoo, 10) · J (Tg deep push 2, 10) · K (multi-task, 10) ·
L (data curation, 5) · G (compound audits/packaging, 19).

## Rules compliance

- Every `.py` is standalone: it reads ONLY official `Dataset/` files.
- No oracle, no `Oracle/`, no old R2 CSVs, no hashes of historical files,
  no experiment records — by construction and verified by `tests/`.
- Fixed seeds; reproducible.
