# Controlled experiment workspace

This folder keeps exploratory work out of the public submission codebase until it is reproducible and independently reviewed.

## Experiment rules

1. Use the official data through `../isolated_runs/data`.
2. Report target-wise grouped-CV R², never a pooled score as a substitute.
3. Save the command, environment, random seed, metrics CSV, and model/checkpoint.
4. Promote a result only if it improves the appropriate matched protocol and does not weaken the qualitative evidence suite.
5. Do not edit the submission codebase while an experiment is still exploratory.

`tabpfn/` is an optional scarce-target comparison. It is not part of the submitted architecture unless it earns that status through the same validation protocol.
