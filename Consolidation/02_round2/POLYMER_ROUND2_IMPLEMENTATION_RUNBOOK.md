# Round 2 implementation runbook

Run all commands from this folder. Never invoke a Kaggle kernel or submission
command.

## Environment

```bash
.venv/bin/python --version
.venv/bin/kaggle --version
```

The local `.venv/` contains Kaggle CLI, pandas, NumPy, SciPy, scikit-learn, RDKit,
and notebook parsing dependencies. Record exact versions in each experiment.

## Starting or resuming the research loop

The loop is currently paused. When the user explicitly starts it:

1. read `AGENTS.md`, `research/research-state.yaml`, `research/findings.md`, and
   `POLYMER_ROUND2_EXPERIMENT_LOOP.md`;
2. change only the state field from paused to active and freeze the recorded next
   protocol before execution;
3. run one local experiment at a time and update the research state/log/findings
   after its decision;
4. keep oracle and public observations post-freeze and monitoring-only;
5. do not create a recurring loop, Kaggle run, upload, submission, Git commit, or
   deletion unless separately authorized.

The first experiment is `R2-C002`, a model-free validation audit. Do not change
folds or models to reproduce the user-reported public score `0.859`.

## Re-download official files only when required

```bash
.venv/bin/kaggle competitions files -c ppp-round-2
.venv/bin/kaggle competitions download -c ppp-round-2 -p ppp-round-2
unzip -n ppp-round-2/ppp-round-2.zip -d ppp-round-2
```

`unzip -n` preserves existing files. If hashes differ, use a new versioned data
directory and update the rule/data audit; never overwrite or delete the old copy.

## EDA

```bash
.venv/bin/python tools/round2_eda.py
```

Expected sanitized outputs:

- `analysis/eda/round2_eda.json`
- `analysis/eda/round2_eda.md`

## Oracle lane

Oracle operations run only against
`scraped/ORACLE_ASSISTED_RESEARCH_ONLY/`. The clean pipeline must not import its
tools or accept its paths as arguments. Validate every mapping transform on
Round 2 train before applying it to test and store per-row provenance.

## Clean run

For each experiment:

1. allocate a new `R2-C###-YYYYMMDD-HHMM-slug` directory;
2. freeze `config.json`, official hashes, folds, seed, and command;
3. run smoke, then full OOF/full fit;
4. validate candidate schema and hash outputs;
5. append a JSONL log record;
6. if it is a train-side incumbent, generate and execute the notebook;
7. compare notebook/local predictions;
8. only then score the frozen candidate against the local oracle.

Initial reference commands:

```bash
.venv/bin/python tools/initial_reference_pipeline.py \
  --data-dir ppp-round-2 \
  --output submissions/<versioned-name>.csv \
  --run-dir experiments/CLEAN_OFFICIAL_ONLY/<new-experiment-id>

.venv/bin/python tools/build_initial_reference_notebook.py \
  --source tools/initial_reference_pipeline.py \
  --output notebooks/<versioned-name>.ipynb

.venv/bin/python tools/execute_notebook_local.py \
  --notebook notebooks/<versioned-name>.ipynb \
  --work-dir experiments/CLEAN_OFFICIAL_ONLY/<experiment-id>/<new-execution-dir> \
  --executed-output experiments/CLEAN_OFFICIAL_ONLY/<experiment-id>/<executed-name>.ipynb
```

## Candidate validation invariants

- 4,940 rows exactly;
- `id,target` only;
- IDs unique and equal to `test.csv` in the same order;
- every target finite;
- no oracle or external-source path in code, notebook JSON, environment, or logs;
- no precomputed predictions, features, weights, caches, or checkpoints loaded;
- fixed seeds and bounded resources;
- notebook/local output values equal within the declared tolerance.

## Actions that remain prohibited

Do not run `kaggle kernels push`, `kaggle kernels update`, `kaggle competitions
submit`, an upload API, or a final-selection action without a new exact user
instruction.
