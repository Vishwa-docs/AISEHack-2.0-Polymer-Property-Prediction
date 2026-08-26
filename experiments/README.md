# experiments/

One directory per experiment: `R3-C###-YYYYMMDD-HHMM-<short-slug>`.

Required artifacts per experiment (see EXPERIMENT_LOOP.md and AGENTS.md §8):
- `config.json` — frozen configuration (seeds, folds, features, hyperparameters)
- `command.txt` — exact command used
- `run.log` — console output
- `metrics.json` — seven per-target R², mean, fold std, MAE, coverage
- `predictions.csv` — 4,940-row `id,target` output (git-ignored)
- `artifact_manifest.sha256` — hashes of all inputs/outputs
- `decision.md` — pass/fail vs gates + next action

Clean lane (official data only) and oracle-assisted lane (post-freeze verification
only) must stay in separate namespaces. Oracle-scored outputs never enter the
clean lane.
