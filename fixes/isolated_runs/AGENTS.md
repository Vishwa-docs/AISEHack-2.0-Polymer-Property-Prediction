# Isolated notebook handoff

This folder is a self-contained user-run reproduction environment. `data/` is a
read-only source symlink; all generated artifacts must remain under this folder,
especially `outputs/`, `checkpoints/`, `.matplotlib/`, and Jupyter state folders.

The user is currently running the notebook. Do not execute, overwrite, or regenerate
it without explicit approval. After it finishes, verify its output schema, recorded
metrics, and qualitative CSVs before copying anything elsewhere. Python 3.11.7 is
required; use `.venv/bin/python` when a command is authorized.
