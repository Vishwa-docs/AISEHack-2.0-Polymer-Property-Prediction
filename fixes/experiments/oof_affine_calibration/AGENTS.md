# OOF affine calibration experiment

Purpose: test whether a small, target-wise affine correction improves prediction calibration
without accessing test labels or changing any submission artifact.

Inputs are read-only completed-run OOF and proxy test predictions from `fixes/isolated_runs/`.
All generated files stay in this directory's `outputs/` folder.

Protocol:

1. Split each target's OOF rows by canonical structure using five GroupKFold folds.
2. Fit the affine correction on four folds and score it only on the held-out fold.
3. Retain it only when its mean target R² gain is at least 0.005 and it wins at least four
   of five folds. Otherwise keep identity.
4. Fit the retained correction on all OOF rows and apply it once to the matching proxy-test
   predictions. The resulting CSV is an isolated candidate, never a replacement submission.

Do not tune against leaderboard feedback or change `submission.csv` from this experiment.
