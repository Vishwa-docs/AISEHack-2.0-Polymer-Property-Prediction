# AGENTS.md — AISEHack 2.0 Polymer Property Prediction, Round 3

You are a coding/research agent working on **Round 3 of the AISEHack 2.0 Polymer
Property Prediction Challenge** (Kaggle). Read this entire file at the start of
every session. It is the operating contract. Then read `PLAN.md` and
`EXPERIMENT_LOOP.md` before doing any work.

## 1. Mission

Predict seven polymer properties from SMILES strings and **win the hackathon**.
Round 2 ended with a user-submitted public score of **0.891** (no-archive lane).
A competitor submission at **0.92** already exists on the Round 3 leaderboard.
We must beat 0.92 — target public **≥ 0.93** — while fully complying with the
competition rules, and this round additionally requires **model explainability**
and **polymer-invariance robustness** as judged themes.

Seven targets (metric = unweighted mean of per-target R² — never pool rows):
`tg` (glass transition), `egc` (chain bandgap), `egb` (bulk bandgap),
`ei` (ionisation energy), `eea` (electron affinity), `eps` (dielectric constant),
`nc` (refractive index).

Timeline: competition closes **3 September 2026**. Max **3 submissions/day**,
**2 final submissions**. Every submission must be backed by a Kaggle notebook
(see §10) and reproduced end-to-end after the competition.

## 2. Repository map (this Mac — the ONLY source of truth)

Working directory: `/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3`

| Path | Purpose |
|---|---|
| `Competition_Details/` | Official Round 3 context. **Read these before planning anything**: `Overview.txt`, `Dataset Description.txt`, `Competition Rules.txt`, `AISEHack 2.0 Polymer Property Prediction: Round 3 | Kaggle.html` |
| `Dataset/` | Official Round 3 data: `train.csv` (7,409 rows), `test.csv` (4,940 rows), `PI1M.csv` (995,799 unlabeled polymer SMILES), `smile_r3.csv` (5,973,369 unlabeled molecular SMILES), `sample_submission.csv`, `base_line_model.ipynb` |
| `Oracle/` | Local verification answers. **Git-ignored. Verification only — see §6.** Contains `oracle.csv` (3,818/4,940 exact values), `oracle_proxy_DIAGNOSTIC_ONLY.csv` (4,905/4,940), build/score scripts, and `NOTES_R3.md` (verification report: Round 3 train/test/PI1M are byte-identical to Round 2, so the oracle is unchanged) |
| `analysis/` | EDA and dataset analysis (incl. the new `smile_r3.csv`). Not done yet — the user will assign it; do not duplicate completed work, fill it in as it happens |
| `experiments/` | One directory per experiment (`R3-C###-YYYYMMDD-HHMM-<slug>`). Every experiment writes config, metrics, decision, hashes. Follow `EXPERIMENT_LOOP.md` |
| `final_submissions/` | Best 2 CSV + their generating notebook/`.py` pairs. See §10 |
| `logs/` | Append-only `experiments.jsonl`, `LEADERBOARD_LOG.md`, oracle score records |
| `scripts/` | Working scripts; `scripts/r2_reference/` holds the Round 2 no-archive submission notebooks (V52/V53) and fable engine code **copied here for Round 3 use** |
| `research/` | Research notes, novelty ledger, research-state |
| `PLAN.md` | The experiment plan for this round — follow it |
| `EXPERIMENT_LOOP.md` | The research→experiment→analysis loop with gates — follow it |
| `TRIALS.md` | Catalog of everything tried in Round 1/2 and whether it worked — consult before proposing an experiment |
| `CONTEXT.md` | Portable, self-contained context — paste it (with TRIALS.md) to any agent without repo access |
| `FINAL_REPORT.md` | End-of-round deliverable (write at the end, not now) |

**Everything for Round 3 lives in this repo on this Mac** — code, experiment
logs, results, and submissions. Never store Round 3 artifacts primarily on the
GPU laptop (§5).

## 3. Every-run checklist (do this at session start, every session)

1. Read `AGENTS.md` (this file), `PLAN.md`, `EXPERIMENT_LOOP.md`, `TRIALS.md`, `CONTEXT.md`.
2. Read `Competition_Details/Overview.txt` + `Competition_Details/Competition Rules.txt` (rules change — verify the deadline, submission limits, and host-sharing list are still as stated here).
3. **Load your available skills and use them.** Previous agents ignored skills and burned context. At minimum: load any skill for context economy / long-task management (in past sessions these were named **headroom** and **ponytail**) and follow their instructions faithfully; load any skill for experiment hygiene, file management, or subagent orchestration that applies to this work. If a skill's description matches a task you are about to do, load it BEFORE acting.
4. Verify the frozen data hashes (see `EXPERIMENT_LOOP.md` Stage 0) before trusting any cached feature or prediction.
5. Confirm the oracle is not importable/reachable from the code you are about to run (grep for `oracle`, `Oracle`, `ORACLE_ASSISTED` in the clean path — must be absent).
6. State in your first message: current best verified-oracle score, the next planned experiment id, and what you will do this session.

Use subagents for bulk reading/research/EDA so the main context stays lean; the
main runner must verify every claimed metric itself (never accept a subagent's
score as evidence).

## 4. Rules compliance (violations = disqualification — non-negotiable)

- **Official competition data only — NO external data, NO pretrained models.** No external datasets (public or private, including any Kaggle/other competition data, any web-scraped SMILES, any literature Tg/property datasets), no pretrained models/weights/embeddings/checkpoints/vocabularies (including HuggingFace, ChemBERTa, MolBERT, Uni-Mol, Graphormer, **TabPFN**, any LLM/VLM/GNN pretrained on molecules/polymers), no transfer learning from any model trained outside the notebook, no artifacts created outside notebook execution. `archive/` from Round 2 is NOT available in Round 3 and must not be used. `smile_r3.csv` (5,973,369 rows) and `PI1M.csv` (995,799 rows) are the **only** additional official Round 3 data (user confirmed both were provided by the organizers) and are allowed **only** for representation learning **from scratch, inside the notebook, with no external weights** — every embedding/vocabulary/SVD/MLM must be fitted from random initialization inside the single notebook run. At notebook time, sanity-check that `train.csv`, `test.csv`, `PI1M.csv`, `smile_r3.csv` exist in the Round 3 Kaggle input dir (`/kaggle/input/ppp-round-3` or `/kaggle/input/aisehack-2-0`) before relying on them; if any file is missing, fall back to training without it (still rules-compliant).
- **Notebook/code-only.** The entire pipeline (load → features → train → infer →
  write `submission.csv`) must run in ONE Kaggle notebook run, no manual
  intervention, fixed seeds, reproducible.
- Public code (e.g., model architecture code) is allowed if it brings no data,
  weights, or artifacts and runs reproducibly in the notebook. **Attach nothing
  to the notebook** (no wheels, no datasets, no checkpoints) — the Kaggle image
  preinstalls RDKit, torch, transformers, sklearn, xgboost, lightgbm, catboost,
  and shap, which is enough. A wheel containing anything learned (weights,
  vocabularies, embeddings) is a prohibited artifact, and even pure-source
  wheels are an audit risk; if one ever seems unavoidable, ask the user first.
- The notebook linked to each submission must be shared (view) with the hosts
  (Rohit Batra IITM, Rahulsundar, LaksmanN, VIJITH P, shreyasri0301) and the
  pinned version must reproduce the submitted score exactly.
- **No cheating**: no hand labeling, no leaderboard back-solving, no repeat-submit
  probing, no test-answer leakage. The oracle (§6) exists to verify locally — using
  it in any submitted artifact is disqualification.

## 5. GPU laptop access (read-only reference + long-running compute ONLY)

- Host: `vishwa@100.116.22.29` (Tailscale). Password: `kumaresh@123`.
- Connect from this Mac. The Mac may lack `sshpass`; a working pattern:

  ```bash
  cat > /tmp/dsh_askpass.sh <<'EOF'
  #!/bin/sh
  echo "kumaresh@123"
  EOF
  chmod +x /tmp/dsh_askpass.sh
  SSH_ASKPASS=/tmp/dsh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
    ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
        -o PubkeyAuthentication=no vishwa@100.116.22.29 '<command>'
  ```

- Hardware: RTX 5090 laptop GPU (24 GB), 62 GB RAM, 24 cores, ~612 GB free disk.
- Round 2 codebase: `~/Desktop/AISEHack-2.0/` (git repo; safety snapshot commit
  `d75bd74`, tag `round3-before-start-20260826-2230` exists).
  Round 2 details: `~/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/`
  — read `AGENTS.md`, `POLYMER_ROUND2_EXPERIMENT_LOOP.md`, `POLYMER_ROUND2_FINAL_REPORT_20260804.md`,
  `logs/EXPERIMENT_LOG.md`, `research/research-log.md`, `research/findings.md`,
  `research/best_component_registry.yaml`, `research/per_target_best_leaderboard.json`,
  `experiments/CLEAN_OFFICIAL_ONLY/` (375 experiments), `final_submissions/`,
  `tools/` (368 scripts).
- Python env on the laptop: `~/Desktop/AISEHack-2.0/.venv-polymer/bin/python`
  (Python 3.12.3; rdkit 2026.3.4, numpy 2.4.6, pandas 2.2.3, scikit-learn 1.9.0,
  scipy 1.17.1, torch 2.11.0+cu128, torch-geometric 2.8.0.post1, xgboost 3.3.0,
  lightgbm 4.7.0, mordredcommunity 2.0.7).
- **STRICT — the laptop files are READ-ONLY reference.** You may read/search/copy
  FROM them, and you may run compute jobs ON the machine, but you must NEVER
  modify, create, or delete anything inside `~/Desktop/AISEHack-2.0/` (or anywhere
  on the laptop except a scratch dir you create for runtime, e.g. `/tmp/r3_runtime/`).
  The user has a backup and a safety commit — do not touch it.
- If a long job must run on the laptop GPU: (a) write the script HERE in this repo,
  (b) copy it to laptop scratch (`scp` to `/tmp/r3_runtime/`), (c) run it there,
  (d) copy ALL results/logs back into this repo's `experiments/` or `logs/`,
  (e) clean up the scratch dir. Round 3 experiment records live in THIS repo only.
- Keep at most ONE heavy GPU process at a time; respect the machine's other users.

## 6. Oracle policy (verification only — never more)

The oracle (`Oracle/oracle.csv` + proxy) is a local, incomplete answer panel
(3,818 exact / 4,905 proxy of 4,940 test rows) used to verify frozen candidates.
It is the ONLY exception to external knowledge, and it is verification-only:

- Read it ONLY after a candidate CSV is fully written, frozen, and hashed
  (post-freeze lane), and only via `Oracle/score_round2_ORACLE_ASSISTED_RESEARCH_ONLY.py`
  (adapted to Round 3 paths).
- Allowed after freeze: aggregate component selection for the NEXT candidate
  (label it `oracle-observed`).
- FORBIDDEN: any oracle value in training, features, transforms, calibration,
  routing, blend-weight selection before freeze, or per-row predictions; copying
  oracle rows into any submission; model-filling unresolved rows and calling them
  answers; any reference to `Oracle/` from a notebook/script/CSV that gets
  uploaded or submitted.
- Before any upload, scan the notebook for `oracle|Oracle|ORACLE_ASSISTED` — must
  be absent. Also scan that it reads nothing outside the official Kaggle data dir.

## 7. Data facts you must not rediscover

- `Dataset/train.csv` 7,409 rows; `Dataset/test.csv` 4,940 rows (4,497 unique
  SMILES); byte-identical to Round 2 (hashes in `EXPERIMENT_LOOP.md`).
- `smile_r3.csv`: 5,973,369 rows, all unique, zero overlap with train/test/PI1M,
  mean SMILES length 54.
- Per-target train/test counts: tg 4,143/2,763 · egc 2,028/1,352 · egb 337/224 ·
  ei 222/148 · eea 221/147 · nc 229/153 · eps 229/153.
- 457 SMILES appear in both train and test — always use structure-grouped
  validation (a canonical structure must never straddle train/val folds).
- Round 2 no-archive best: verified 0.9042 / proxy 0.9030 local, public 0.891.
  Round 2 weak targets were eps, nc, ei (and tg without archive).
- The "4,497" figure on the Kaggle Data page counts unique test SMILES, not rows.
  The submission must have 4,940 rows, ids 1..4940, exactly `id,target`.

## 8. Experiment discipline (details in EXPERIMENT_LOOP.md)

- IDs: `R3-C###-YYYYMMDD-HHMM-<short-slug>`, sequential, never recycled.
- One experiment dir under `experiments/` per run with: config, command,
  metrics.json (seven per-target R² + mean + fold std + MAE + coverage),
  predictions.csv, artifact hashes, `decision.md`. Append one line to
  `logs/experiments.jsonl` (append-only).
- Clean lane vs oracle lane are separate namespaces. Oracle-scored outputs never
  enter the clean lane.
- Validation panels (mandatory): target-stratified grouped folds, canonical-group
  folds, scaffold/family folds, similarity clusters, low-similarity bins,
  availability masking. Promotion gates are fixed (see EXPERIMENT_LOOP.md) —
  do not relax them to keep a flattering number.
- **Before proposing any experiment, search `TRIALS.md` and the Round 2 logs on
  the GPU laptop for the same idea.** Cooled families need a genuinely new
  mechanism plus a pre-registered kill gate.

## 9. Skills & tools (mandatory)

- At session start, list your available skills and LOAD the applicable ones:
  context-economy / long-task skills (historically `headroom`, `ponytail`),
  plus anything for experiment hygiene or subagent orchestration. Follow their
  instructions — do not just acknowledge them.
- Use subagents for parallel reading/research/EDA; keep the main context for
  decisions and metric verification.
- Write everything durable to files immediately (research notes, decisions,
  metrics). The session's working memory is not a log.

## 10. Submission file policy (hard requirement — this is what Round 2 failed)

- **STANDALONE-REPRODUCTION RULE (user-mandated, 2026-08-27, non-negotiable):**
  EVERY `.py` file that generates a submission — the V57 reproduction included —
  must be a **single standalone file** that fully reproduces its output by itself:
  it reads ONLY the official `Dataset/` inputs (`train.csv`, `test.csv`, and —
  only when the experiment's protocol says so — `PI1M.csv` / `smile_r3.csv`),
  trains every model from scratch with fixed seeds, and writes the 4,940-row
  `id,target` `submission.csv` in one run. At runtime it must NEVER:
  - read, open, or import any precomputed prediction CSV, model artifact, cached
    feature file, old submission, or Round-2 CSV (including the R2 `v52_bundle`
    CSVs, `v53_base.csv`, the V57 arm CSVs, or `latest_submission.csv`);
  - reference, compare against, or depend on any historical file hash, experiment
    id, manifest, or `logs/experiments.jsonl` record;
  - reference any file under `Oracle/`, `experiments/`, `final_submissions/`
    (other than writing its own output), or the GPU laptop's Round-2 tree.
  In other words: **no old-file references at runtime, period** — the .py is the
  complete recipe. Verification of "reproduces V57" is done by the agent/user
  AFTER the run by scoring the freshly generated CSV against the oracle, never by
  the .py reading old CSVs or hashes. (All R2 reference code may still be READ
  and PORTED into the .py as source while building it — the prohibition is on
  runtime references, not on reading reference material during development.)

- Every candidate promoted to "best" must come with a **single, end-to-end,
  self-contained notebook (`.ipynb`) or `.py` file** that: reads only the official
  Kaggle data dir, trains everything from scratch with fixed seeds, predicts all
  4,940 test ids, and writes `submission.csv` (`id,target`) — in ONE run, no
  manual steps, no local file dependencies, no `__file__`, no oracle references.
- Store the pair (`<notebook/py>` + its `submission.csv`) in `final_submissions/`.
  Keep only the current best TWO pairs (the competition allows 2 final
  submissions). When a new best is promoted: generate the new pair, verify local
  parity (identical CSV from notebook vs pipeline), score it post-freeze against
  the oracle, **delete the superseded pair**, and update
  `final_submissions/README.md` with scores + hashes.
- Experiments must be run so the submission file can be regenerated from the same
  frozen configuration at any time with no side dependencies (no imports from
  `scripts/`, no cached features/weights/CSVs, no GPU-only paths).
- A notebook that copies a CSV, embeds predictions, or has regeneration disabled
  is NOT a submission notebook (Round 2 disqualification lesson).

## 11. Cleanliness rules (this repo must not become Round 2's mess)

- Keep the layout of §2. Delete nothing that is a record (experiment dirs,
  logs, research notes). Superseded predictions/CSVs live in the experiment dirs;
  only `final_submissions/` carries the current-best policy of §10.
- **Reference vs copy:** you may reference the Round 2 codebase and logs on the
  GPU laptop without copying them (they are history). But any code you actually
  build on, run, or reuse for Round 3 — including code from Round 2 — must be
  COPIED into this repo first (e.g., `scripts/r2_reference/`) and worked on here.
  Anything that generates a submission lives in this repo. All Round 3 code,
  logs, and results live in this repo, not on the laptop.
- Do not commit `Oracle/` (git-ignored). Do not commit large prediction blobs
  (git-ignored patterns in `.gitignore`). Do not create files outside the
  defined layout without updating this map.
- No `rm`/overwrite of existing files except the explicit §10 supersede policy;
  versioned new paths are the default.

## 12. Reporting

- After every experiment: one short summary with the seven R² values + mean,
  vs the incumbent, gate verdict, and the single next action.
- After every session: update `logs/` and `research/` state files; leave the
  repo in a state where a fresh agent can pick up by reading AGENTS → PLAN →
  EXPERIMENT_LOOP → logs.
- `FINAL_REPORT.md` is the end-of-round deliverable (accuracy + explainability +
  invariance narrative). Build it up as results arrive; complete it before the
  3 September 2026 deadline.

## 13. Hard prohibitions (summary)

- Never modify the GPU laptop's files. Never run Kaggle uploads/submissions or
  `kaggle kernels push` without explicit user authorization (authentication ≠
  permission; the user submits).
- **Never use external labeled data, pretrained weights, or oracle values in any submitted artifact.** This includes: no external SMILES, no literature Tg/property datasets, no pretrained ChemBERTa/MolBERT/Uni-Mol/Graphormer/TabPFN/LLM/VLM/GNN weights or vocabularies, no transfer learning from external models, no artifacts (weights, vocabularies, embeddings, SVD matrices) created outside the notebook execution. Every representation (TF-IDF, SVD, word2vec, MLM, contrastive) must be fitted **from scratch, inside the single notebook run**, on official data only. Never submit without the matching end-to-end notebook that regenerates all artifacts from scratch.
- Never relax seeds, folds, or gates to make a number look better. Never report
  an oracle/proxy number as a public score.
- Never exceed one GPU job on the laptop at a time; respect 20% RAM/VRAM headroom.
- Never print or copy credentials (Kaggle OAuth, passwords) into any file in
  this repo.
