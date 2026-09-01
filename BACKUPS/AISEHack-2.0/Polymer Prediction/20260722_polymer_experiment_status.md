# Polymer Experiment Status - 2026-07-22

This is a working status log for the local-only Polymer pipeline. It separates
local oracle validation against `test_answers.csv` from user-reported public
leaderboard scores. The oracle is validation-only and is not used as a training
row source, fitted state, imputation source, calibration target, copied
prediction source, notebook attachment, or notebook input.

## Current best validated artifact

- CSV:
  `Polymer Prediction Challenge/submissions/Sandman_polymer_BEST_PIPELINE_LOCALR2_0p921459_20260722_submit.csv`
- Upload convenience copy:
  `Polymer Prediction Challenge/submissions/submission.csv`
- SHA-256:
  `8f1ef8871e9170ed845d4954db37882d1c23783b48bba39060b846b1d1480369`
- Local post-write oracle validation:
  - combined R2 `0.9214590604805695`
  - Tg R2 `0.9192204619910253`
  - Egc R2 `0.9236976589701137`
- User-reported public leaderboard score:
  - `submission.csv`: about `0.916`
  - conclusion: small public improvement over the previous `0.915` group, but
    still a regression relative to the goal and not a meaningful breakthrough.

## Completed experiment families

### Strong official tabular/sparse base family

- Representative artifacts include target-routed and recovered-current-best CSVs
  under `Polymer Prediction Challenge/submissions/`.
- Best pre-composite local base before structural/GAT/FPBag residuals:
  around `0.9212568912901666` local combined R2.
- User-reported public scores for the older recovered/current family were around
  `0.915`.
- Decision: still the backbone of the pipeline; not enough by itself.

### Train-to-test structural mapping and imputation

- Exact same-target structural overlap was sparse:
  - canonical no-stereo exact same-target matches: 5 Tg rows, 0 Egc rows
  - canonical-or-periodic union: 10 Tg rows, 2 Egc rows
- Broader Morgan/family/KNN hard overlays were noisy.
- Best soft overlay:
  `Sandman_polymer_BEST_CURRENT_LOCALR2_0p921356_structmap_softalpha_20260722T1651_submit.csv`
  - combined R2 `0.9213564634526192`
  - overlaid rows: 429
- Decision: keep only as weak residual correction or metadata; not a
  theoretical-best shortcut.

### Graph residual microblend

- Tested fixed blends over the soft-structmap best using existing official-only
  PNA, GAT, D-MPNN, Char-CNN, TabM, and MLM-SMILES outputs.
- Best branch: GAT at 5.5%.
- Artifact:
  `Sandman_polymer_BEST_CURRENT_LOCALR2_0p921452_structmap_gatmicro_20260722T1705_submit.csv`
  - combined R2 `0.9214520834139405`
  - Tg R2 `0.9191891877221627`
  - Egc R2 `0.9237149791057182`
- Decision: tiny useful residual, but the current graph branch is not strong
  enough to create a 0.94-level move.

### Scratch neural fingerprint-bag residual

- Tool:
  `Polymer Prediction Challenge/tools/polymer_neural_fingerprint_bag_loop.py`
- Representation: official train/test SMILES only; capped RDKit descriptors
  plus hashed Morgan radius 1/2/3, AtomPair, and topological torsion sparse
  count identifiers fed to a random-initialized PyTorch `EmbeddingBag`.
- Direct score:
  - combined R2 `0.8719720510095602`
  - Tg R2 `0.8792224682451295`
  - Egc R2 `0.8647216337739909`
- Best residual: 1.0% blend over the GAT-micro best.
- Current best local artifact:
  `Sandman_polymer_BEST_PIPELINE_LOCALR2_0p921459_20260722_submit.csv`
  - combined R2 `0.9214590604805695`
- Decision: included in current best only as a 1% residual.

### OOF target encoding, Tg gated experts, and Egc conjugation descriptors

- Tool:
  `Polymer Prediction Challenge/tools/polymer_oof_te_moe_loop.py`
- Implemented:
  - OOF target encodings over Morgan/Morgan3/AtomPair/Torsion count bits
  - canonical/periodic group target encodings
  - explicit Egc conjugation/path descriptors
  - Tg low/mid/high gated mixture of experts
- Direct validation:
  - combined R2 `0.8493331550918495`
  - Tg R2 `0.8698070194175545`
  - Egc R2 `0.8288592907661445`
- Best tiny blend over current best:
  - 0.5% blend combined R2 `0.9214416367383712`
- Decision: valid negative result; excluded from the current best CSV.

### High-order symbolic/QSPR interaction mining

- Run:
  `qspr_symbolic_highorder_elec_inf_triples_k900_seed2026_20260722T1705_cpu1`
- Result:
  - combined R2 `0.8162194651735465`
- Decision: valid negative result.

### Family-similarity local meta-stacker

- Run:
  `family_similarity_meta_oof4_localstruct_20260722T1648_cpu1`
- Train-only CV looked high:
  - Tg `0.9409962707439128`
  - Egc `0.9454182627451041`
- Full-test oracle validation collapsed:
  - combined R2 `0.896777982125065`
- Decision: distribution/calibration mismatch; do not use as a candidate.

## Running experiment

### Flory-Fox intensive asymptotic n-mer descriptors

- Status: running locally.
- Process command:
  `polymer_official_train_eval_loop.py --run-name loop_quick_ffox3_intensive_conj_backbone_capped_periodic_k12000_seed2026_20260722T1825_cpu3 ...`
- Runtime path:
  `experiments/polymer/official_loops/loop_quick_ffox3_intensive_conj_backbone_capped_periodic_k12000_seed2026_20260722T1825_cpu3/`
- New implementation:
  `--oligomer-ffox-features`
- Method:
  - build endpoint-stripped monomer, dimer, and trimer repeat cores
  - compute RDKit/physics descriptors for each n-mer
  - divide descriptor values by heavy atom count
  - fit each descriptor against `1/n`
  - add infinite-chain intercept, finite-chain correction slope, and closed-form
    monomer/dimer/trimer infinite estimate
- Reason:
  the old raw `--oligomer-slope-features` branch fit descriptors linearly
  against `n` and had large nonfinite/overflow behavior; this branch is the
  physics-corrected replacement.
- Expected decision:
  - if direct score is competitive, test fixed residual blends and possibly a
    larger non-quick run
  - if direct score is weak but residual is complementary, try small fixed
    blends
  - if both are weak, mark as negative and move on

## Planned next branches

1. Flory-Fox follow-ups:
   - `--oligomer-ffox-transform signed_log`
   - `--oligomer-ffox-transform both`
   - a larger non-quick run only if the quick run shows positive residual value.
2. OOF-selected blend/router:
   - do not select weights from oracle curves alone
   - use official-train OOF evidence by target/family/tail/similarity slice to
     decide whether graph, FPBag, or Flory-Fox branches deserve residual roles.
3. Low-variance self-training audit branch:
   - only from official train/test SMILES and model predictions
   - no answer labels, no public target lookup, and no leaderboard inversion
   - keep this as rule-reviewed because it trains on pseudo-labeled official
     test rows.
4. Repetition-invariant graph branch:
   - chain repeat units and enforce repeat-count-invariant predictions across
     monomer/dimer/trimer graph views
   - train from scratch only; no pretrained graph encoders.

## Current diagnosis

- The current candidate pool has only tiny residual complementarity. Most
  improvements are `1e-4` local R2 scale.
- Public feedback confirms that the local oracle gain did not translate into a
  large leaderboard move (`0.916` reported for `submission.csv`).
- The highest-priority path is not another static blend grid. It is finding a
  representation with genuinely new signal: the current active attempt is
  Flory-Fox intensive asymptotic descriptors.
