# Isolated full-run notebook

This folder is a self-contained launch point for the submitted 904 notebook. It contains immutable copies of the `.ipynb` and percent-format `.py` source, a symlink to the official project dataset, and a dedicated `.venv` created with Python **3.11.7**.

## Run it

```bash
cd fixes/isolated_runs
bash bootstrap.sh          # once; installs the pinned environment
bash launch_lab.sh
```

In Jupyter, choose **PPP isolated (Python 3.11.7)** and select **Run All**. For a headless run:

```bash
bash run_all.sh
```

`data/` is a read-only symlink to the project’s official dataset directory; the notebook should not be moved without preserving that link. `PPP_RUN_DIR`, `PPP_OUTPUT_DIR` and `PPP_CHECKPOINT_DIR` force all generated files to remain inside this folder: `outputs/` for charts/tables/submissions and `checkpoints/` for restartable stages. Before copying any regenerated result to the public codebase, compare the submission schema, checksum, recorded local held-out score, and each qualitative evidence table.

If VS Code already had the notebook open, reload it from disk before Run All so it
uses the self-locating startup cell. The copied notebook SHA-256 is
`d9e4ec36e62db4d18ed1a350494ced4402e221e1b95ba458e3342a08c0d26d02`.
