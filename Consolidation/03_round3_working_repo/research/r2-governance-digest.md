# Governance Knowledge Extraction — Polymer Property Prediction Hackathon

**Sources analyzed (read in full):**
- `/tmp/r2dump/AGENTS.md` (1,148 lines) — Round 2 operating contract ("Polymer Property Prediction Challenge — Round 2 Operating Contract", frozen evidence date 2026-08-03 Asia/Kolkata, with later amendments through 2026-08-09).
- `/tmp/r2root/AGENTS.md` (1,718 lines) — repo-root Round 1 operating contract ("AISEHack 2.0 Experiment Operating Contract", frozen evidence date 2026-07-18 Asia/Kolkata, later amendments through 2026-08-05).

Both are "executable contracts" for coding agents, not result claims. The Round 2 file is the more recent and more specific of the two and governs all Round 2 work; the repo-root file still supplies credential safety, data security, and the standing local-only Kaggle directive (r2dump §1).

---

## 1. ORACLE POLICY

### What "the oracle" is
The oracle is a **local, provenance-backed answer/verification panel** for scoring a completed candidate's predictions against recovered ground-truth target values. It is explicitly an **incomplete verification panel**, not a complete answer key, and it is **never** part of clean model training or submission generation.

Two historical names refer to the same concept:
- Round 1: the user-provided validation file `Polymer Prediction Challenge/scraped/scraped/test_answers.csv` (r2root lines 186, 221).
- Round 2: `scraped/ORACLE_ASSISTED_RESEARCH_ONLY/oracle.csv` (r2dump lines 241, 620, 736).

### Where the oracle dataset lives (paths)
Round 2 directory contract (r2dump §7, lines 219–244) places oracle material under:
- `scraped/ORACLE_ASSISTED_RESEARCH_ONLY/` — contains `oracle.csv`, provenance, and external answer sources.
- `experiments/ORACLE_ASSISTED_RESEARCH_ONLY/` — oracle-assisted diagnostic runs, "never eligible for submission".

The clean side lives separately under `experiments/CLEAN_OFFICIAL_ONLY/`. Oracle file schema (r2dump lines 338–342):

```
id,smiles,target_type,target,oracle_status,source_id,match_method,source_sha256
```

`oracle_status` is `verified` or `unresolved`; unresolved `target` values remain empty. Per-row provenance is stored only in the oracle file; only aggregate coverage may enter clean documentation (lines 344–346).

Round 1 namespaces (r2root lines 187–188): `experiments/polymer/oracle_guided_program/ORACLE_ASSISTED_RESEARCH_ONLY/` (every oracle-assisted file name MUST contain `ORACLE_ASSISTED_RESEARCH_ONLY`) and `experiments/polymer/oracle_guided_program/CLEAN_OFFICIAL_ONLY/`.

### What sources it was built from
Audited evidence (r2dump lines 348–355):
- **Khazana `export.csv`** resolves every Round 2 test row for `egc`, `egb`, `ei`, `eea`, `eps`, and `nc`: 2,177 exact rows; its canonical mapping reproduces all 3,266 corresponding Round 2 train values with max absolute error 0.
- **Official bundled Round 1 train** gives 1,641 exact `tg` values by unambiguous raw row identity.
- **Older public Tg recovery** supplies a high-fidelity proxy for another 1,087 rows (exact on 1,329/1,604 newly revealed Round 2 train labels = 82.855%, MAE 1.6979 °C, RMSE 7.3757 °C, R² 0.995399). This is a *proxy*, not truth.

Result: `oracle.csv` has **verified truth for 3,818/4,940 rows** and leaves **1,122 Tg values null**; the proxy reaches 4,905/4,940 diagnostic coverage with 35 unresolved rows (lines 352–353). "A perfect 4,940-row oracle is not currently substantiated." Never hide the limitation by model-filling nulls (line 354). Nominally different Tg mirrors are often byte-identical — hash every source (line 355).

### How it was used — verification-only role
- It is a **separate post-freeze verification lane** and must be "explicitly identified in internal records" (r2dump lines 28–30).
- It may be read **only after a candidate CSV has been completely written and hashed** (line 316).
- Scoring must report "oracle coverage, verified-panel R² by target, aggregate mean R², and the proxy-panel result separately"; null rows and coverage must never be hidden or silently model-filled (lines 48–51).
- It may guide **aggregate** research decisions (model-family choice, target routes, splice targets, blend weights, architecture settings) after completed candidates are scored; every such choice is labeled `oracle-observed` (lines 32–41, 105–110).

### Strictly allowed vs strictly forbidden
Allowed (aggregate, post-freeze only):
- Score a completed/hashed candidate CSV against the oracle.
- Use aggregate coverage-disclosed results to guide model-family, target-pipeline, architecture, and compound-selection decisions, labeled `oracle-observed`.

Forbidden (quoted from r2dump §9.2 lines 323–335 and the 2026-08-09 clarification lines 32–41):
- Supply a **training row, target statistic, transform, calibration target, row-level routing label, or per-row prediction**.
- Be **referenced by the final submission script or notebook**.
- Be **copied to `submissions/`**.
- Be **packaged, uploaded, or described as a valid submission artifact**.
- **Fill an unresolved answer with a model prediction and label it truth.**
- "Row-level oracle values may never be copied, imputed, memorized, embedded as constants, or used to set an individual test-row prediction."
- Round 1 (r2root line 230): never use "a row's own oracle label to directly set that row's prediction, choose that row's carrier, tune that row's blend weight, or validate a clean official-only method."

### Keeping oracle usage out of final submissions/notebooks
- The final notebook "must not read, mention, attach, reconstruct, or embed oracle/answer rows, scoring files, prior prediction CSVs, local experiment manifests, frozen component outputs, caches, or checkpoints" (r2dump lines 36–40).
- "Never rename an oracle-assisted output into the clean or submission namespace" (line 215).
- `test_answers.csv` is "a scoring target only and MUST NOT be used as a training row source, fitted state, imputation source, calibration target, copied prediction source, notebook attachment, or submission-construction input" (r2root line 186).

---

## 2. EXPERIMENT CONVENTIONS

### Naming
- Round 2 recommended ID form: `R2-C###-YYYYMMDD-HHMM-<short-slug>` (r2dump §6 line 211). Example IDs in the file: `R2-C000-20260803-1642-initial-reference`, `R2-C001-20260803-1645-initial-reference-repaired`, `R2-C002` … `R2-C009`.
- Round 1 (repo-root) IDs: `POLY-P###-YYYYMMDD-HHMM-<slug>` and `SAR-S###-YYYYMMDD-HHMM-<slug>`; `YYYYMMDD-HHMM` is Asia/Kolkata proposal time rounded down to the minute; slug is lowercase ASCII letters/digits/single hyphens; sequence `000` reserved for the canonical root baseline; IDs are never recycled (r2root §7 lines 370–396).
- The Round 2 file also refers to "F-series run[s]" — every F-series run must record branch, input hashes, and result under the matching `with_archive`/`without_archive` directory (r2dump lines 8–17).

### Where runs are stored
Round 2 (r2dump §7 lines 219–244, §12 lines 434–444):
```
experiments/
  CLEAN_OFFICIAL_ONLY/<experiment_id>/   # runnable clean experiments
  ORACLE_ASSISTED_RESEARCH_ONLY/          # diagnostics never eligible for submission
logs/experiments.jsonl                    # append-only compact run records
scraped/ORACLE_ASSISTED_RESEARCH_ONLY/    # oracle.csv, provenance, external sources
submissions/                              # local candidates only; no automatic upload
notebooks/                                # self-contained generating notebooks
final_submissions/with_archive/           # active submission-candidate branch
final_submissions/without_archive/        # control branch
```

### Manifest / decision / score files each run must produce
Each executed clean run gets (r2dump §12 lines 434–443):
```
experiments/CLEAN_OFFICIAL_ONLY/<experiment_id>/
  config.json
  command.txt
  environment.txt
  run.log
  metrics.json
  predictions.csv
  artifact_manifest.sha256
  decision.md
```
Plus one compact append-only record in `logs/experiments.jsonl` recording: ID, parent, hypothesis, target scope, official input hashes, seed, folds, config hash, runtime, per-target OOF scores, mean score, mapping coverage, artifact hashes, decision, and next action (lines 446–449).

Round 2 states (line 451): `planned`, `running`, `completed`, `promoted`, `rejected`, `failed`, `blocked_rule`. A runtime-invalid result is `failed`; a valid negative is `rejected`.

Round 1 (repo-root) used a far heavier ledger: immutable `preregistration.yaml` (62 envelopes), append-only `events.jsonl`, `incidents.jsonl`, `gate_reports/{compliance,integrity}.yaml`, `promotion_record.yaml`, `final_record.yaml`, `artifact_manifest.sha256`, plus track-level `index.jsonl`, `submissions.jsonl`, `progress.md`, `incumbents.yaml` (r2root §§6–9, 20).

### How scores are recorded — oracle scoring vs CV
- **Clean prospective validation (CV/OOF)** is the only clean evidence used for promotion/selection: "deterministic repeated K-fold or group K-fold main panel", plus canonical-structure group, scaffold/family, Tanimoto-cluster/low-similarity, target-specific sparse-data uncertainty, exact/archive mapping leave-group-out, and multi-property availability panels (r2dump §10 lines 384–393).
- **Oracle scoring** is strictly a post-freeze diagnostic lane, always reported separately with coverage; it must never be mixed with clean CV (r2dump §9, §17). "Oracle scores may guide later aggregate research choices… but the final notebook must contain no oracle mention or dependency" (lines 332–336).
- Clean-only vs oracle-assisted results are always reported separately (r2root line 188).
- Never compute a pooled R² across targets; always per-target R² plus unweighted mean (r2dump lines 381–383).

### Watchdog / queue system
Round 2 requires a persistent local supervisor (r2dump §23 "Continuous execution and overnight supervision", lines 772–805):
1. a pre-created **protocol-only queue** of versioned child experiments;
2. a **single-instance lock** so two heavy runs cannot launch;
3. an **atomic heartbeat/state file** (active run, PID, queue position, last heartbeat, terminal history);
4. per-run stdout/stderr logs and exit status;
5. **adoption of a visible existing child** after supervisor restart;
6. a versioned recovery child after a process exits without metrics;
7. **automatic advancement** after success/valid-rejection/runtime-failure;
8. a persistent user-level service with restart-on-failure and user lingering.

The supervisor must not kill a healthy experiment to refresh its heartbeat, must not retry an ID in place after a config/resource/runtime change, and must not force `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` when parent replay is gated at `1e-12` (the C188-v2 parity incident — thread overrides caused a runtime-invalid mismatch). A queue reaching `idle` while the objective is unmet is an automation incident (lines 766–768).

### CLEAN_OFFICIAL_ONLY vs ORACLE_ASSISTED_RESEARCH_ONLY
- `CLEAN_OFFICIAL_ONLY` = runs that use only official competition files for fitting; eligible to become candidates/submissions.
- `ORACLE_ASSISTED_RESEARCH_ONLY` = local diagnostics that may touch answer data; **never** eligible for submission, promotion, packaging, final selection, or leaderboard action (r2dump §7, §9.2; r2root lines 187–188).

---

## 3. SUBMISSION / NOTEBOOK REQUIREMENTS

### What the Kaggle notebook must do
A real submission must be a **single end-to-end Kaggle notebook run** (r2dump §4 lines 156–158; §11 lines 415–425):
- discover local vs Kaggle input paths **without using `__file__`**;
- read **only** the official competition directory (`train.csv`, `test.csv`, `PI1M.csv`, optionally official `archive/` files);
- derive target names and full test order dynamically;
- initialize every model from code and **fixed seeds**;
- contain all feature, fit, inference, override, blend, and CSV-generation logic;
- **never read a local prediction, OOF table, weight, embedding, cache, oracle, or checkpoint**;
- write exactly **4,940 rows in current test order** with columns `id,target` and no null/non-finite values;
- use bounded CPU/GPU memory and Kaggle-compatible runtime;
- make network access unnecessary after Kaggle-preinstalled dependencies load.

Required notebook H1/H2 sections (r2dump §23 "Final standalone notebook gate", lines 940–970): competition objective/rules/provenance/seed; EDA for all seven targets (distributions, missingness, duplicates, canonical structures, archive overlap, feature/label availability, train/test shift); research findings and per-target rationale; all feature construction + fold-safe preprocessing (+ PI1M-from-scratch if used); architecture/hyperparameters/routing/assembly/fallback; clean validation summaries + ablations **without oracle values**; full-train fit + inference on every test ID; finite/ID-order/schema/row-count/duplicate checks; direct creation of the final `id,target` CSV.

The repo-root Round 1 contract adds: "All preprocessing, split creation, model initialization, training, inference, and `submission.csv` generation must happen in a single Kaggle notebook run" and "Set and record every stochastic seed. A reproduced notebook must match the submitted result" (r2root §4 lines 156–158); initialize every learned model from random weights inside the final notebook; derive every descriptor/fingerprint/split/preprocessing state/model/prediction/submission row from official data during that run (r2root lines 184–185).

### with_archive / without_archive variants
Round 2 maintains two branches (r2dump lines 5–17):
- `final_submissions/with_archive/`: official current inputs plus the permitted `ppp-round-2/archive/train.csv` label pool — the **active submission-candidate branch**.
- `final_submissions/without_archive/`: current `train.csv`/`test.csv` only, no archive labels — the **control**.

A candidate from one branch must never be silently mixed with the other. Every F-series run records branch, input hashes, and result under the matching directory. Branch gates (lines 744–752): `with_archive` → keep improving until verified-oracle mean R² ≥ 0.95, then generate + locally execute the standalone `.ipynb`; `without_archive` → first reach and preserve a clean CSV at ≥ 0.93 verified-oracle mean R², then improve toward 0.95; don't generate a final notebook before the branch gate unless separately requested.

### What went wrong last time (missing end-to-end ipynb)
- **2026-08-09 user correction (r2dump lines 83–86):** "local chain-replay notebooks are not acceptable final artifacts. The user will upload only the `.ipynb` files to Kaggle, so every active final notebook must be fully standalone and must not refer to, require, or replay local experiment artifacts."
- §9.3 (lines 361–373): do not place a notebook in `final_submissions/` if it reads repository-local run outputs, prior prediction CSVs, frozen component artifacts, manifests, trace inventories, local checkpoints, or local caches; do not describe a replay/wrapper/reconstruction/local-artifact-dependent notebook as final.
- Specific bad Round 1 notebooks to NOT reuse as templates (r2dump §13 lines 481–485): `official_only_public_baseline_repro.ipynb` (unexecuted, uses `__file__`); `best_current_composite_pipeline_20260722.ipynb` (can copy an existing CSV); old C106c option notebooks embed precomputed artifacts.
- §11 line 427–428: "A notebook that copies a CSV, embeds predictions, or has regeneration disabled is not a submission notebook."
- §5 (r2root line 33–36 and r2dump lines 198–202): never present a locally generated file as Kaggle-notebook-generated; never create/attach/submit a dummy or non-generating notebook.

### Lessons learned (stated in the file)
- The notebook is the deliverable, not the local pipeline; it must be standalone, deterministic (fixed seeds), parity-verified (identical IDs/order and numerically matching targets, r2dump §11 lines 407–413), and free of `__file__`, CSV-copy, and precomputed-artifact dependencies.
- Round 1's failure mode was OOF overfitting: "Rich OOF stackers 0.936–0.940 collapsed to about 0.887 test/oracle" → cooldown unless a new transfer gate is demonstrated (r2dump §14 line 498).

---

## 4. MODEL / PIPELINE ARCHITECTURE

> Note on terminology: the exact component labels **"engines F01/F02/F03" and "blend sweeps" do not appear in either file.** What these files document is a per-target "Best Compound" portfolio assembled from target-specific "carriers", per-target OOF NNLS/simplex blends, and an F-series run convention. I report the architecture as it is actually written.

### Round 2 winning-pipeline shape ("Best Compound", user-approved 2026-08-05)
`Best Compound` = a **seven-property portfolio, not one model selected by a single global score** (r2dump lines 993–1021). Maintain a per-target leaderboard for `tg, egc, egb, ei, eea, nc, eps`; a candidate that improves one target stays eligible for that target even if it worsens another, so the final portfolio may use up to **seven distinct target-specific pipelines** (§21 lines 657–665; §23 lines 874–895).

Allowed per-target pipelines (lines 874–885):
- **Tg**: mobility/free-volume or tree carrier;
- **Egc**: conjugation/electronic carrier;
- **Egb**: route specialized to available band-gap labels;
- **Ei**: electronic/charge/topology route;
- **Eea**: Flory-Fox, electronic, or identity-safe route;
- **Nc**: refractivity/volume/ionic-coordinate route;
- **EPS**: dielectric/ionic-coordinate route.

### Component vocabulary actually used in the files
- **Carriers**: `R2-C001` (the initial reference) "combines the official current/archive label pool, conflict-abstaining raw/canonical lookup, **RDKit/Morgan/SMILES/Tanimoto carriers**, other-property official covariates, and per-target OOF NNLS" (r2dump line 515). "Routers or row-wise experts only after two independently useful carriers exist and transfer gates pass" (line 527). Cross-property carrier + similarity route + fallback per target (line 660).
- **Blends**: "target-specific OOF NNLS/simplex blends" (Round 1 portables, line 476); `R2-C007` = "nested target-wise blend using only arms that passed transfer panels" (line 524). Compound assembly = select strongest eligible full pipeline per target, predict every test ID exactly once, combine only the seven target columns, record component map + hashes (lines 1013–1021).
- **Per-target specialists**: `R2-C003` EPS/Nc paired-property specialist (EPS is the weakest clean OOF target); `R2-C004` Ei/Eea donor/acceptor + conjugation specialist; `R2-C005` Egc/Egb coupled residual specialist; `R2-C006` clean Round 1 Tg carrier (lines 517–529). Sparse targets (`ei, eea, egb, eps, nc`) prefer low-variance models, repeated folds, shrinkage, and mechanistically related official labels (lines 529–531).
- **F-series runs**: convention naming the with_archive/without_archive final-candidate runs that must record branch + input hashes + result (lines 8–17).

### Data files used
Official clean inputs (r2dump §4 lines 161–183 and §8 fingerprints lines 251–257):
- `ppp-round-2/train.csv` — 7,409 rows; cols `smiles,target,target_type`; SHA-256 `609b0f48…f9ba2`.
- `ppp-round-2/test.csv` — 4,940 rows; cols `id,smiles,target_type`; SHA-256 `d8a0da26…cf2d`.
- `ppp-round-2/PI1M.csv` — 995,799 unique unlabeled SMILES; SHA-256 `c5e1017b…cd8cd`.
- `ppp-round-2/archive/train.csv` — Round 1 labeled train, 6,171 rows; SHA-256 `b12cadb3…6864f68`.
- `ppp-round-2/archive/test.csv` — 4,115 rows unlabeled; SHA-256 `e03e0659…21216dd`.
- `ppp-round-2/archive/sample_submission.csv` (10 illustrative rows only — never use for coverage) and `ppp-round-2/archive/base_line_model.ipynb`.

Per-target counts (lines 259–269): train/test = tg 4,143/2,763; egc 2,028/1,352; egb 337/224; ei 222/148; eea 221/147; nc 229/153; eps 229/153.

`archive/train.csv` is an official Round 2 bundle file and is treated as clean competition data (line 174–178). `PI1M.csv` may support representation learning **only if learned from scratch inside the final notebook** (lines 180–183; §23 lines 903–912).

### Key modeling ingredients
Round 1 portable methods to carry forward (r2dump §13 lines 470–479): RDKit descriptors and physical/count features; Morgan count/bit fingerprints with target-specific radii; Tanimoto kernel or nearest-neighbor arms; SMILES character n-gram Ridge; target-specific OOF NNLS/simplex blends; grouped and cluster validation; bounded electronic/conjugation features for gap targets; bounded rigidity/mobility features for Tg.

Frozen EDA findings (lines 274–282): `archive/train.csv` adds ~2,445 unambiguous raw lookups into current test (Tg/Egc dominated); `egc`–`egb` Pearson ≈ 0.963 and `nc`–`eps` ≈ 0.918 (justify cross-property baselines, not leakage); 885 canonical train structures have ≥2 labeled properties; PI1M overlaps 295 train + 174 test rows.

### Score history / evidence hierarchy
- C001 initial reference OOF R² (line 613): Tg 0.908877, Egc 0.911504, Egb 0.922147, Ei 0.806897, Eea 0.888235, Nc 0.839732, EPS 0.783505; **mean 0.865843**. Public score 0.859000 (line 616).
- Strongest clean arithmetic recorded: corrected-parent/C162 line ≈ **0.919134** (line 1057). C187-v2 (C050-relative EPS reproduction) 0.879899 (line 1058). C050 clean-OOF mean 0.8731 vs test-side proxy mean 0.8621 (line 708–709).
- Second-wave (lines 1114–1141): public 0.916 artifact = frozen practical leaderboard incumbent; C257/C253 composite clean reference ≈ 0.894197; next order = corrected C272 EHT-response screen → repeat-unit recutting/reversal/periodic/doubled/tripled views (equivalent views grouped in one fold) → common-fold nested portfolio → random-init periodic graph/multiview multitask control → PI1M graph self-supervision with full encoder fine-tuning + deterministic EHT-response distillation.
- Historical Round 1 tiers (lines 487–492): portable clean public baseline OOF ≈ 0.9062; best public-observed staged C09 ≈ 0.9224 (public ≈ 0.917); highest oracle-selected diagnostic C106c ≈ 0.9232 (not clean-promoted); strongest formally reproduced P011 main panel ≈ 0.8504.

---

## 5. RULES AND GUARDRAILS IMPOSED ON AGENTS

### Local-only execution (standing directive, r2root lines 7–39)
- All training/inference/scoring/reproduction/candidate generation MUST execute **only on local hardware** until the user gives a new explicit per-action authorization.
- Do not call `kaggle kernels push`, `kaggle kernels update`, kernel-session APIs, or any operation that starts/restarts Kaggle compute.
- Read-only listing/status/metadata/log retrieval allowed for audit; cancelling an already-active session allowed only when directed; cancellation does not authorize a replacement run.

### No Kaggle runs/upload/submission without exact human authorization
- "Never start or update Kaggle compute, upload a notebook, upload a file, create a submission, or select a final submission unless the user gives a later explicit authorization for that exact action" (r2dump §2 line 113).
- Prohibited until exact user authorization (r2dump §5 lines 195–202): `kaggle kernels push`/`update`, any Kaggle session/run API, notebook/file/dataset upload, `kaggle competitions submit`, final-submission selection, and presenting a local file as Kaggle-notebook-generated.
- "Authentication is not permission to act" (line 203). Upload authorization ≠ run authorization ≠ submission/final-selection authorization (r2root lines 26–31, 1128–1129).
- The user owns the actual Kaggle notebook version creation, upload, submission, and final-selection (r2root line 1127).

### No modifying / deleting certain paths
- Work only inside `/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2` (Round 2 root) unless a task explicitly requires read-only consultation of a Round 1 artifact (r2dump §1 lines 90–92). Note this reveals the GPU machine is a **Linux** box (`/home/vishwa/...`), not macOS.
- Do not edit/move/rename/delete/clean/reset/overwrite anything outside this folder; do not delete files inside it (lines 91–92).
- Never use `rm`, `unlink`, destructive globbing, `git clean`, `git reset --hard`, or overwrite redirection on a prior result (r2dump §6 line 209).
- Hard Round 2 artifact-location boundary (lines 1080–1112): every Round 2 artifact must live below `Polymer Prediction Challenge Round 2/`; forbidden locations are repo root, root-level `research/`/`experiments/`/`notebooks/`/`submissions/`, the sibling `Polymer Prediction Challenge/` (Round 1) folder, the SAR folder, and temp/home paths used as final handoffs.
- Never rename an oracle-assisted output into the clean/submission namespace (line 215).

### Budget / GPU constraints (r2root §12 lines 819–842)
Resource tiers: smoke ≤15 min / ≤5% rows, 1 seed + 1 fold; pilot ≤60 min / ≤25% support, 1 seed + 2 folds; confirm ≤4h / all rows, 3 Polymer seeds (5 SAR); final ≤8h / clean reproduction within 75% of verified live Kaggle runtime+disk.
- Run **at most one GPU training process at a time** across both tracks; at most two concurrent SAR raster readers.
- Reserve ≥20% measured RAM and ≥20% VRAM headroom; a run fails if peak RAM/VRAM exceeds 80% of measured available.
- Reserve ≥2 physical CPU cores for OS/monitoring; cap numerical-library threads per worker.
- Measure available RAM/VRAM immediately before each run (`free -h`, `df -h`, accelerator status command).

### Other guardrails
- **Credential safety** (r2root §5): never print/log/commit credentials (`kaggle.json`, tokens, cookies, auth headers, private keys, signed URLs); read only from a secret store; never run an access-token print command (r2dump line 203–205).
- **No cheating**: no hand-labeling test rows, leaderboard inversion, row impulses, repeated submissions to infer hidden targets, or recovering targets from another team's output (r2dump §4 line 159; r2root §17).
- **Determinism & seeds**: set/record every stochastic seed; a reproduced notebook must match the submitted result.
- **No external data/weights**: no pretrained weights, checkpoints, embeddings, processed features, cached tensors, or artifacts produced outside the final notebook (r2dump §4 line 154).
- **Git**: Round 2 requires no commits unless the user asks (r2dump §1 line 93); Round 1 root contract was commit-heavy (commit metadata between gates).

---

## 6. ANYTHING ELSE A ROUND 3 CODING AGENT NEEDS TO KNOW

### Target properties & metric
- Seven targets: `tg` (glass transition), `egc` (chain bandgap), `egb` (bulk bandgap), `ei` (ionisation energy), `eea` (electron affinity), `eps` (dielectric constant), `nc` (refractive index) — r2dump §3 lines 134–144.
- Metric: **arithmetic mean of independent R² per target** — `mean(R²_tg, R²_egc, R²_egb, R²_ei, R²_eea, R²_nc, R²_eps)`. Never pool rows across targets (lines 130, 377–383).
- Round 2 objective: mean seven-target R² ≥ **0.95** (0.93 is an intermediate branch gate, not the goal) — lines 67–71, 727–732.
- Submission filename on Kaggle: `submission.csv` (line 129).

### Evaluation / coverage / submission mechanics (Round 2)
- Competition `ppp-round-2`, Kaggle ID 157637, enabled 2026-07-29; deadline 2026-08-12T18:30Z = 2026-08-13 00:00 IST (lines 122–125).
- Team limit 5; final submissions selectable 2; rules page says 3 submissions/day but live API/team control say 5/day — zero submissions without exact authorization (lines 126–128).
- Official `test.csv` has **4,940 rows** (Overview text said 4,497; the file is authoritative) — line 131.
- Round 1 (repo-root §4 line 181): evaluation coverage public 37.5% / private 62.5%; daily Polymer cap conservative 3/day; scored finals 2; 6,171 train / 4,115 test rows; two development-robust final choices under `POLY-AUDIT-CONTINGENCY-v1`.
- The retired `POLY-SPLIT-v1` local audit is not a pristine holdout — never claim "pristine/untouched/unopened/unbiased" (r2root lines 191–192, 989–1000).

### Timeline
- Round 1 (repo-root) frozen deadline 2026-07-24 23:55 IST; reserve 07-23 for clean notebook reproduction + first submission, 07-24 for correction/final-selection buffer; no new architecture families on 07-24 (r2root §21).
- Round 2 deadline 2026-08-13 00:00 IST (r2dump line 125).

### Tools / venvs / GPU machine
> **Factual caveat:** neither file documents a specific venv name, conda environment, Python version, or GPU model (no RTX/A100/CUDA specifics). The only machine facts are: the Round 2 work root is a **Linux path** `/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2`; each run records `environment.txt` (Round 2) or `environment.lock` (Round 1) plus hardware fields `{accelerator, vram_gb, driver}`; session-start commands are `free -h`, `df -h`, and "the available accelerator status command" (r2root §24 line 1509). So the "GPU laptop" specifics (exact venv paths, package lists) are **not contained in these two AGENTS.md files** — a Round 3 agent should read the actual `environment.txt`/`environment.lock`/runbook files in the workspace instead.

### Structure of the Round 2 folder (r2dump §7) — files a new agent should read first
`AGENTS.md`, `CLAUDE_RUNS.md`, `ROUND2_COMPETITION_DETAILS.md`, `POLYMER_ROUND2_EXPERIMENT_LOOP.md`, `POLYMER_ROUND2_IMPLEMENTATION_RUNBOOK.md`, `README_EXPERIMENTS.md`, plus `research/research-state.yaml`, `research/findings.md`, `research/research-log.md`, `research/RESEARCH_NOVELTY_LEDGER.md`, `research/best_component_registry.yaml`, `analysis/eda/round2_eda.md`, and `analysis/prior_round/ROUND1_SYNTHESIS.md`.

### Self-improving multi-agent loop + review council (r2dump §16, §21)
Five roles after every experiment: **Historian** (prevent duplicate work), **Property researcher** (one target + one residual slice, not generic ML), **Adversary** (explain leakage/fold-luck/conflict/shift), **Planner** (one smallest discriminating next experiment with a stop condition), **Notebook auditor** (official-only inputs, no artifacts, deterministic, local parity). Use subagents with disjoint read-only scopes; sidecars are advisory and MUST NOT read oracle/answer values to choose experiments.

### Promotion gates (r2dump §17)
A target component is retained only with ≥0.01 grouped gain, ≥4/5 folds same direction, group-bootstrap lower bound >0, adjacent target loss ≤0.003, non-negative missing-auxiliary/low-similarity behavior. A full incumbent needs ≥0.002 prospective clean mean gain, no target grouped loss worse than 0.003, all transfer panels, notebook parity, frozen hash before any oracle/public observation. Public score can never satisfy a promotion gate. Submission-eligible replacement additionally must beat the incumbent on the frozen full-test post-freeze transfer diagnostic (OOF gain alone is not banked).

### Cooldown list (do not blindly retry, r2dump §14)
Rich OOF stackers, generic GNN/CNN/Transformer/MLM, broad mapping/read-across (unless official archive overlap proves it), forced residual overlays/routers, Mordred/trimer/generic-3D/AutoGluon sweeps, target transforms, and (bounded-only) Tg rigidity/mobility and Egc electronic/conjugation specialists.

### Overall posture for a new agent
Keep oracle and clean lanes strictly separate; never name an improvement from CV alone (require oracle scoring before calling anything an improvement, r2dump lines 43–51); always disclose Tg/oracle incompleteness (~60% Tg coverage); keep the loop bounded to one experiment at a time; never abandon the loop when the objective is unmet; and never submit/upload/run Kaggle without exact user authorization.
