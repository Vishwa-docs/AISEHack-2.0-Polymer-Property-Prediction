# Polymer Round 2 experiment loop

Read `AGENTS.md` first. This file defines the method order and feedback loop; it
does not authorize Kaggle compute, upload, or submission.

## Objective

Reach mean seven-target R² ≥ 0.93 with an official-only, from-scratch,
single-notebook pipeline. Maintain separate per-target incumbents and a final
target router because the metric gives every target equal weight despite unequal
row counts.

Frozen starting evidence: C001 clean OOF `0.865843`, proxy expectation `0.856028`,
and user-reported public score `0.859000`. The proxy missed public by only
`0.002972`, while clean OOF was `0.006843` optimistic. The remaining public-to-goal
gap is `0.071`, equal to `0.497` summed R² points across seven targets. This is too
large for cosmetic global calibration; seek prospective target-specific signal.

## Stage 0 — bootstrap and invariants

Before the first run:

1. Verify the official hashes recorded in `AGENTS.md`.
2. Run `tools/round2_eda.py` and review both reports.
3. Freeze target order, seeds, fold assignments, canonicalizer, similarity
   clustering, and evaluation code.
4. Initialize append-only `logs/experiments.jsonl` without rewriting existing
   lines.
5. Confirm the oracle path is not imported, opened, mentioned, or discoverable by
   the clean pipeline or notebook.

## Validation panels

Use the same folds for comparable runs:

- Main: repeated target-stratified folds with fixed seeds.
- Canonical group: identical canonical structures stay together.
- Scaffold/family: scaffolds or chemistry clusters stay together.
- Similarity cluster: Butina/Tanimoto clusters stay together.
- Low similarity: report performance by nearest-train Tanimoto bin.
- Mapping simulation: hold complete canonical groups out before evaluating an
  archive/current exact-lookup policy.
- Cross-property availability: reproduce which other target labels would be
  available for the held-out row at test time.

Report seven R² values, their mean, fold standard deviations, MAE, and coverage.
Never pool targets.

## Experiment sequence

### Executed bootstrap records

- `R2-C000-20260803-1642-initial-reference`: runtime-invalid before predictions;
  preserved after an extreme RDKit descriptor exceeded a tree model's internal
  float32 range.
- `R2-C001-20260803-1645-initial-reference-repaired`: completed initial reference.
  It deliberately bundles the official archive lookup, Morgan/descriptor/SMILES/
  Tanimoto carriers, other-property official covariates, and target-wise OOF NNLS.
  Clean mean OOF R² is `0.865843`; the high-coverage post-freeze diagnostic mean is
  `0.856028`; user-reported public mean is `0.859000`. Treat it as the fixed
  comparison reference, not as proof that all transfer gates pass.

### R2-C002 — validation hardening and fixed carrier table

Reproduce `R2-C001` on fixed canonical-group, scaffold/family, similarity-cluster,
low-similarity, exact-lookup simulation, and property-availability panels. Persist
per-target fold assignments and an arm-by-panel table. No hyperparameter search.
Pass only if every panel is leakage-free and the current notebook candidate is
reproduced; otherwise repair validation before testing a new model.

### R2-C003 — EPS/Nc paired-property specialist

EPS is the weakest clean OOF target (`0.783505`), while paired train labels have
strong EPS/Nc correlation. Compare the frozen C001 carrier with one low-variance
specialist using polarizability/density descriptors plus the other official label
when genuinely available. Mask availability at structure-group level. Retain only
if EPS improves by at least `0.01` mean grouped R² without Nc falling more than
`0.003`, the gain survives at least four of five folds, the group-bootstrap lower
bound exceeds zero, and missing-auxiliary plus low-similarity slices are
non-negative. Run one preregistered comparison and one negative control; gate
failure cools the branch and advances to the next property.

### R2-C004 — Ei/Eea electronic specialist

Test one preregistered donor/acceptor and conjugation descriptor block with Ridge,
ExtraTrees, and a small boosted-tree control. Evaluate Ei and Eea separately;
cross-property labels are permitted only under the frozen availability mask. Stop
after one bounded model comparison if neither target gains `0.01` grouped R².

### R2-C005 — Egc/Egb coupled specialist

Exploit the strong paired Egc/Egb relationship with a low-variance residual model,
but keep an independent molecular carrier for missing-covariate rows. Require
improvement in the low-gap and no-other-label slices, not merely the easy paired
rows. Reject a forced router that cannot pass the availability simulation.

### R2-C006 — Tg Round 1 portable reproduction

Reproduce the strongest clean Round 1 portable carrier on the expanded official
current+archive label pool: RDKit/physical descriptors, Morgan counts at radii
1/2/3, character n-grams, Ridge and Tanimoto KRR/kNN. Compare it against C001 on
identical folds and similarity bins. Do not port Round 1 weights, embedded
predictions, staged additions, or oracle-selected routes.

### R2-C007 — nested target-wise blend

Only arms that passed their target transfer gate enter a nested OOF simplex/NNLS
blend. Shrink sparse-target weights toward the best single carrier and require
residual diversity. Generate a new notebook only if the seven-target prospective
mean improves and no target suffers an unacceptable grouped regression.

### R2-C008 — residual diagnosis and one mechanistic branch

Slice errors by target, similarity, scaffold, wildcard count, SMILES length,
functional motifs, label-availability pattern, and model disagreement. Choose one
smallest falsifiable experiment. Do not launch a broad sweep or revisit cooled
Round 1 families without new evidence.

### R2-C009 — PI1M equal-budget control

Only after the supervised baseline is stable, test a from-scratch PI1M
representation against an equal-budget randomly initialized/no-pretraining control.
Reject unless it improves the preregistered target and transfer panels and remains
feasible in the single end-to-end notebook. Imported embeddings or checkpoints are
never eligible.

## Promotion threshold

Promote only a train-side prospective gain that survives transfer panels. Freeze
the clean candidate hash before oracle scoring. Every promoted run requires a
self-contained notebook and local parity report.

Component gate: grouped target gain ≥ `0.01`, at least four of five folds positive,
group-bootstrap lower bound above zero, adjacent/paired-target loss ≤ `0.003`, and
non-negative missing-auxiliary plus low-similarity behavior. Full-incumbent gate:
prospective seven-target clean gain ≥ `0.002`, no target grouped loss worse than
`0.003`, all transfer panels pass, and notebook parity passes. Stable smaller
components may wait for C007; they do not create a new candidate.

Do not use public `0.859` as an optimization target. It is one aggregate observation
with no per-target decomposition. A future candidate must first improve the frozen
clean comparison; its oracle/proxy and any later user-reported public score are
post-freeze monitoring checks. Record prediction correlation and changed-target
scope so a public movement can later be interpreted without score inversion.
Never algebraically back-solve a hidden property score from the rounded aggregate
and local oracle components. Do not use submissions merely to estimate calibration;
three or more method-diverse observations would still support aggregate monitoring
only, never target or row corrections.

### Full-data situational oracle checkpoint

For future experiments, preregister a separate post-freeze checkpoint when the user
wants a practical estimate of where the full-data model stands. After the clean OOF
run is complete, fit the unchanged frozen configuration on all available official
training rows, generate predictions for all 4,940 official test IDs, and freeze the
prediction path, bytes, configuration hash, and artifact manifest before opening
the isolated oracle lane. The resulting oracle/proxy value is a situational
diagnostic only: it may be reported to show the current ceiling or gap, but it may
not change features, seeds, weights, routing, row values, candidate selection,
promotion, packaging, upload, or submission. A protocol with a stricter oracle
policy remains stricter, and no oracle diagnostic is retroactively applied to an
already-running experiment. Full-data fitting is therefore part of the final
prediction-generation step, not a license to train on oracle values.

## Inner loop — one bounded experiment

After a completed experiment:

1. Historian checks for duplicate methods and source hashes.
2. Adversary proposes leakage, fold-luck, shortcut, and distribution-shift
   explanations.
3. Property researcher investigates exactly one target/residual mechanism and
   records new sources in the novelty ledger.
4. Planner chooses one discriminating next run, expected signal, resource bound,
   pass gate, and stop gate.
5. Notebook auditor checks clean inputs and parity for a new incumbent.

Do not revisit a cooled method/source without a new falsifiable reason. After three
consecutive weak ideas, change target or representation rather than continuing a
hyperparameter crawl.

Every inner-loop protocol freezes one primary hypothesis, one changed factor,
compute limit, expected target/slice, clean pass gate, stop gate, and notebook
impact before execution. Complete the metrics, decision, hashes, research state,
and one next action before starting another experiment.

## Outer loop — synthesis and direction

Reflect after four valid experiments, three consecutive non-improvements, a
surprising clean/proxy/public divergence, or any integrity/rule incident. Read the
full trajectory and update `research/findings.md` rather than merely copying logs.
Choose exactly one direction:

- `deepen`: a supported mechanism deserves a stricter or adjacent test;
- `broaden`: current carriers are stable but another target/representation is more promising;
- `pivot`: assumptions or validation failed and the current branch should stop.

Update `research/research-state.yaml` and append `research/research-log.md`. Search
one property and residual mechanism only when existing evidence cannot discriminate
the next action; record URLs and content hashes in the novelty ledger. The loop is
configured but remains paused until the user explicitly asks to start it.

## Stop conditions

Stop a branch when it violates rules, leaks a held-out structure, cannot reproduce,
exceeds notebook resources, fails three relevant panels, or repeats a cooled
hypothesis. Preserve all artifacts and record the reason.
