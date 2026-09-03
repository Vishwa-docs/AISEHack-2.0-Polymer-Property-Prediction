# Qualitative evidence handoff

This folder audits recorded evidence and creates presentation figures from CSVs. It
does not train the submission model. Outputs currently summarize the recorded full run
and must be labelled provisional until `isolated_runs/` completes and produces a fresh
output directory.

Keep generated scorecards and figures under `qualitative_evidence/figures/`. Retain
limitations honestly: cross-model raw rank agreement is a sensitivity diagnostic, not
the primary explanation-fidelity result. Do not inspect or alter the running notebook.
