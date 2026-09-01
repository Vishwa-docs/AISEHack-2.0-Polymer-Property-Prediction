# Polymer Property Prediction — Round 2 Progress Report

**Date:** 2026-08-04  
**Scope:** local-only, official-data-only Round 2 research loop  
**Target:** unweighted mean R² ≥ 0.93 across seven properties

## Executive status

The 0.93 target has not yet been reached. The strongest defensible clean Round 2 incumbent is C050 at mean R² **0.8731493565**. No Kaggle compute, upload, submission, or hidden-answer-driven model selection has been used. The five-role review closed C108/C109 before any scientific metric and allocated C110, a fresh compact mixed portfolio.

The main explanation for the apparent regression from Round 1 is that the scores are not measuring the same task. The protected Round 1 result was a two-property Tg/Egc result, recorded in the logs at approximately 0.9229 and remembered operationally as about 0.924–0.925. Round 2 evaluates seven properties, including three much weaker sparse targets. A seven-target average of 0.93 therefore requires solving substantially more than the two Round 1 targets.

## What produced the Round 1 result

The Round 1 work established the useful strategy that the Round 2 loop has been trying to preserve:

- use a different carrier or model family per property rather than forcing one universal model;
- combine target-specific tree/booster models with molecular descriptors and similarity features;
- use Tanimoto/kernel or read-across-style models where structural similarity is predictive;
- use target-specific post-processing and conservative blends only when they improve clean out-of-fold evidence;
- retain strong existing predictions for unsupported rows instead of replacing the full test set with a weaker specialist.

For Round 1, the strongest remembered carriers were a Tanimoto/KRR-style Tg route and a tree/booster-style Egc route. Several later experiments also tested ExtraTrees, HistGradientBoosting, LightGBM-style models, descriptor families, residual corrections, and gated blends. The important lesson was the deployment pattern—separate pipelines and selective routing—not that one of those two-target models can be copied unchanged into the seven-target problem.

## Round 2 setup and baseline

Round 2 was rebuilt around the official files and a clean/oracle separation:

- **7 target types:** Tg, Egc, Egb, Ei, Eea, Nc, and EPS;
- **official training rows:** 7,409;
- **official test rows:** 4,940;
- **archive rows:** 6,171;
- **training-only rule:** no external target rows, pretrained molecular models, imported embeddings, or learned caches;
- **execution rule:** local hardware only; Kaggle execution remains prohibited;
- **submission rule:** all 4,940 test IDs must be generated in order.

The initial repaired reference C001 reached clean mean R² **0.8658425762**. C050 became the best clean incumbent after target-specific gap/component routing and conservative fallback logic.

| Target | C050 clean R² |
|---|---:|
| Tg | 0.9088768072 |
| Egc | 0.9115043879 |
| Egb | 0.9221467344 |
| Ei | 0.8454440895 |
| Eea | 0.9008357940 |
| Nc | 0.8397322432 |
| EPS | 0.7835054390 |
| **Mean** | **0.8731493565** |

This makes the bottleneck visible: EPS, Nc, and Ei are much lower than the stronger Tg/Egc/Egb/Eea components. Even raising only those three weak targets to 0.93 would leave the mean around 0.919, so the route to 0.93 must also improve at least some of the stronger properties.

## What has been tried in the current Round 2 loop

The loop has tested target specialists, mixed pipelines, cross-property residuals, similarity/read-across routes, tree zoos, physical/chemometric descriptors, graph models, and nonlinear variants. The broad progression was:

1. **Reference and validation:** official-data baseline, canonical grouping, scaffold/family holdouts, similarity bins, low-similarity tails, and property-availability panels.
2. **Round 1 transfers:** Tg mobility/free-volume, Egc electronic/conjugation, finite-chain, and target-specific tree/booster carriers.
3. **Selective deployment:** target-specific blends, similarity-gated routes, direction-consistent corrections, and the C019 three-target route.
4. **Kernel and fingerprint families:** Tanimoto variants, Morgan/count fingerprints, AtomPair/TopologicalTorsion, WL/topological features, endpoint-path features, and structure-key encodings.
5. **Cross-target residuals:** strict cross-fitted predicted-property corrections for Eea, Egb, EPS, Nc, and Ei. These were rejected when the gain was below threshold, transfer panels were negative, or parent parity was not exact.
6. **Physical and nonlinear branches:** periodic distance spectra, Lorentz–Lorenz-style Nc, gap/identity features, RBF/spline/affine calibration, and nonlinear heads.
7. **Graph branches:** graph grammar, shared periodic graph multitask, and target-specific periodic graph models. These either regressed the incumbent or failed the full parent/fold/replay gates.

Representative measured results:

| Experiment | Result | Interpretation |
|---|---:|---|
| C050 | mean 0.8731493565 | Only current defensible clean incumbent |
| C098 target-routed QSPR | mean 0.8748045537, gain +0.0016552 | Best research-only near-miss; EPS +0.0073381 and Nc +0.0042483, but below component and complete-candidate gates |
| C105 shared periodic graph | mean 0.8729048898 | Negative shared graph route; EPS/Nc regressed |
| C106 target-specific periodic graph | mean 0.8731175464, gain -0.0000318 | Complete 4,940-row clean output, but active-target and replay/panel gates failed |

The loop has also recorded many smaller positive-looking results that were correctly rejected because they were not comparable to the exact C050 parent, used incomplete support, selected a route after inspecting the same out-of-fold bins, had negative transfer panels, or missed the fixed +0.01 component gate. Those negative results are why the loop has not been allowed to convert a fragile local improvement into a leaderboard claim.

## Oracle lane status

The oracle/test-answer material is isolated under the explicitly marked oracle-assisted research namespace. It has not been used to train, tune, select, assemble, package, upload, or submit a clean candidate. The available oracle diagnostic has verified values for 3,818 of 4,940 rows; the 4,905-row high-coverage file is a proxy with unanswered rows filled from an existing prediction, not ground truth. Earlier frozen-candidate diagnostics were approximately 0.8644 verified and 0.8578 proxy for C019, and approximately 0.8687 verified and 0.8621 proxy for C050.

Those numbers are monitoring diagnostics only. They do not establish a clean score, and no current candidate has been oracle-scored because the required clean gates have not passed.

## Why convergence slowed

The experiments initially explored many variants of the same underlying descriptor/kernel/graph assumptions. The adversarial reviews identified three recurring failure modes:

- a local gain was measured against a weaker or mismatched parent rather than the exact C050 route;
- a model improved one sparse target but was not stable across folds, scaffolds, similarity tails, or missing-label panels;
- execution metadata sometimes omitted independent parent replay or exact fold-map evidence, making an otherwise interesting result ineligible.

The recent loop therefore tightened the protocol: regenerate C050 from official inputs in memory, use identical target-local folds and masks, require full 4,940-row output, require independent replay, and run a five-role review after each result. This is slower per experiment but prevents false convergence from incomparable metrics.

## Recent runtime failures and current next experiment

C108 used official directed edge-conditioned graph states, but the full-universe graph scope exceeded the bounded local runtime before writing metrics. C109 restricted that same mechanism to target-local graph unions, but still ended without metrics or predictions. They are runtime-invalid/pre-metric records, not negative scientific scores. The five-role council then identified a chronology defect in the C109 append-only timestamps and required a fresh runner.

The next preregistered direction is **C110: compact PLS/Ridge residual portfolio**. It uses separate fixed heads for EPS/Nc/Ei/Tg, preserves C050 for Egc/Egb/Eea, rebuilds the parent from official files, and runs three fixed-seed replicas with no sweep. It has a fresh versioned runner and full 4,940-row output contract. Promotion requires the fixed component, fold, bootstrap, panel, mean-gain, replica, and no-regression gates.

## Full-data oracle checkpoint

The experiment loop now explicitly preregisters a post-freeze situational checkpoint: after clean OOF gates, full official training data are used to refit the unchanged frozen configuration and produce all 4,940 predictions; only then may the isolated oracle be read. That value is reported as a ceiling/standing diagnostic only and cannot tune, select, promote, package, upload, or submit a candidate. Existing C050 diagnostics remain approximately **0.8687 verified** on 3,818 rows and **0.8621 proxy** on 4,905 rows; they are not clean scores and are not a route to selection.

## Files containing the evidence

- `research/findings.md` — synthesized findings and rejection reasons;
- `research/research-log.md` — chronological experiment and council log;
- `research/research-state.yaml` — current incumbent, hypotheses, controls, and next action;
- `experiments/CLEAN_OFFICIAL_ONLY/` — per-experiment protocols and local metrics;
- `experiments/oracle_guided_program/ORACLE_ASSISTED_RESEARCH_ONLY/` — isolated oracle diagnostics;
- `POLYMER_ROUND2_EXPERIMENT_LOOP.md` and `AGENTS.md` — governing Round 2 methodology and operating rules.

**Bottom line:** the Round 2 loop has not yet solved the seven-target problem. The evidence explains the discrepancy with Round 1 and has narrowed the search to the weak targets plus exact parent/replay discipline. C050 remains the clean fallback while C110 is the next bounded test; the 0.93 claim will be made only if a clean, reproducible, rules-compliant candidate actually reaches it.
