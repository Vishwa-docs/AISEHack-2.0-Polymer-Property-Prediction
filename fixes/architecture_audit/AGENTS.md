# Feature and architecture evidence audit

This folder contains claim-to-evidence maps, not a new training pipeline. Every claim must name
an executed artifact and an allowed literature identifier from `Personal/Research/INDEX.md`.
Do not convert a rationale into a performance claim without a matching structured-validation
result. Treat the user-operated isolated run as the promotion gate.

The public-facing story is: **evaluate on a held-out panel, then retrain on all training data
for the final submission.** A post-retrain submission has no new held-out score by itself.
