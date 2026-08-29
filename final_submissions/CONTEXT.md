# CONTEXT.md — V57 Standalone Reproduction (AISEHack 2.0 Round 3 / Round 2 no-archive)

## 1. What this task is

We must deliver ONE completely standalone Python file that reproduces the Round-2
no-archive V57 submission (local verified oracle score ≈ 0.90415) **from scratch**.
The file may read ONLY the official competition inputs — train.csv, test.csv,
PI1M.csv — and must regenerate every intermediate prediction, every model, every
feature, and every blend/splice inside a single run, then write submission.csv
(4,940 rows, columns exactly id,target).

- Competition metric: unweighted mean of per-target R² over the 7 targets
  (tg, egc, egb, ei, eea, nc, eps) — never pool rows across targets.
- Round 3 data is byte-identical to Round 2, so the Round-2 oracle and scores apply.
- Target bar for THIS task: verified-oracle score ≥ 0.9041 (V57 reference = 0.90414956).
  An earlier regenerated-base run scored 0.9041546653908089 verified.

## 2. The deliverable file

final_submissions/v57_reproduction_standalone.py — a single .py (~560 KB) that:

1. Reads ONLY train.csv / test.csv / PI1M.csv (via --data-dir, or auto-detect
   ppp-round-2/ walking up from cwd).
2. Rebuilds every model/feature from scratch with fixed seeds (no caches, no weights,
   no pretrained artifacts, no old CSVs).
3. Chains the full Round-2 no-archive candidate tree in memory (DataFrames only),
   ending with the V53 compound (C1572 base + 7 weighted arms), then applies the
   char arm (Ridge on character n-grams of C282 OOF residuals) and the spread arm
   (ei/eea train-median scaling) and splices them per target.
4. Writes submission.csv.

Run it with (any directory containing the three official files):

    python v57_reproduction_standalone.py --data-dir /path/to/ppp-round-2 --out submission.csv

If --data-dir is omitted it searches ppp-round-2/ from cwd upward.

## 3. Where the oracle is, and how to verify (MUST read before judging)

Oracle lives on the Mac repo at Oracle/ (git-ignored, verification ONLY):

- Oracle/oracle.csv — 3,818/4,940 exact test answers
- Oracle/oracle_proxy_DIAGNOSTIC_ONLY.csv — 4,905/4,940 proxy answers
- Oracle/score_round2_ORACLE_ASSISTED_RESEARCH_ONLY.py — the ONLY sanctioned scorer.

Scoring a candidate (run from the repo root, on the Mac):

    python Oracle/score_round2_ORACLE_ASSISTED_RESEARCH_ONLY.py         --candidate <path/to/submission.csv>         --verified Oracle/oracle.csv         --proxy Oracle/oracle_proxy_DIAGNOSTIC_ONLY.csv         --output out.json

Look at verified_oracle.score.unweighted_mean_r2 in out.json. The oracle is
verification-only: it is read only AFTER a candidate is fully generated, and its
values are NEVER used to train, tune, or select anything inside the .py.

## 4. Non-negotiable rules (user-mandated — do not violate)

1. **Standalone.** The .py is the complete recipe. It must run in an isolated
   environment with ONLY the official dataset present, and reproduce its output.
2. **Official inputs only at runtime.** The ONLY CSVs the file may read are
   train.csv, test.csv, PI1M.csv. It must NEVER read, open, or import any
   precomputed prediction CSV, model artifact, cached feature file, old
   submission, Round-2 CSV, experiment record, manifest, or historical hash.
3. **No old-file references, period.** No experiments/..., no
   final_submission_runs/..., no Oracle/, no R2 paths, no hashes (no sha256,
   no hashlib), no TOOL_SOURCES/VARIANT_CONFIGS/RECIPE JSON blobs, no base64
   bundles, no subprocess calls, no argv lists referencing old runs.
4. **No oracle wording.** The words oracle, Oracle, ORACLE_ASSISTED must not
   appear in the .py or its README (docstrings/comments included).
5. **Everything from scratch, fixed seeds.** Every model/feature/embedding is fitted
   inside the single run with deterministic seeds.
6. **Exact computation.** The .py must reproduce the numeric computation of the
   Round-2 no-archive chain exactly (model classes, hyperparameters, seeds, folds,
   weights, clip margins, routing) — a faithful port, not a re-derivation.
7. **The user verifies the file structure BEFORE any heavy run.** Do not burn GPU
   hours until the user has approved the structure.

## 5. Where the code/files were generated from

- **Reference source of truth (read-only):** the Round-2 no-archive chain in the
  V53/V57 submission notebook and the R2 codebase on the GPU laptop
  (~/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2/).
- **Clean, sanitized copies** of every tool/engine used to build the .py are under
  this repo's scripts/r2_reference/ (copies of the R2 code, read-only reference).
- **Intermediate components inside the .py** were traced back to their producers:
  each intermediate candidate CSV that the original chain consumed is now computed
  by an in-memory function in the .py (leaf models, builders, overlays, blenders).
  The backward trace (required_sequence.json / closure_calls.json artifacts in
  /tmp/ on the Mac during development) maps every intermediate to its generator.

## 6. The pipeline inside the file (architecture)

All functions live in one module; helpers are prefixed by their tool name to avoid
collisions. The driver run_v57(data_dir, out_path) does (in order):

1. **Leaves (official data only):**
   - c282 (current-only reference ensemble + OOF)
   - c284 / c285 (PI1M-SVD reference / weak residual, read PI1M.csv)
   - F03 clean candidate; fable engines F01–F06 (chain-identity, ionic, GPR,
     multitask-MLP, PI1M-distill students)
   - F11/F14 portfolios, F15 weak aggregate, F18 fixed blends, F10 portfolio
   - C287 weak-model zoo (28 arms: per-target × 7 arm types), C286v4 artifact
     stack, C340 C282-polymer-genome wrapper, C927 repeat-view wrapper,
     C391 PI1M model-zoo (c289)
2. **F21/F24/F26 stack** (broad equal combo → cross-property overlay → ionic
   co-test overlay).
3. **The candidate spine** (~75 blend/splice/reflect/overlay calls in the exact
   order and with the exact weights of the original chain), ending at C1572.
4. **V53 compound (noarchive_rank2):** base C1572 + 7 weighted arms (weights and
   arm sources exactly as the original variant config):
   - eea +0.055858956052114814 → C287v3-eea-dense_huber
   - egb −0.2280466904798347 → C565
   - egc −0.039664646178507984 → C1370
   - ei  −1.1545974766202354 → C1349
   - eps −0.37638616522376744 → C488
   - nc  −1.6702239063784179 → C1345
   - tg  +0.15926696690525582 → C927
   formula per target: value = base + weight * (arm − base).
5. **Char arm:** per target, Ridge(α=40) on CountVectorizer character n-grams
   (2–7, 65536 features, lowercase=False) fitted on C282 OOF residuals with
   5-fold KFold(seed 2026, shuffle); prediction averaged over folds; applied as
   base + 0.20 * pred to targets tg/egc/egb/nc/eps.
6. **Spread arm:** for ei/eea only, median + 1.05 * (base − median) using train
   medians (ei ≈ 6.16795, eea ≈ 2.2723).
7. **Final splice:** tg/egc/egb/nc/eps → char arm; ei/eea → spread arm; write
   submission.csv.

## 7. Verification checklist for the next agent

1. **Static checks on the .py** (before any run):
   - grep -n "oracle" file.py → must be empty (also case-insensitive).
   - grep -nE "sha256|hashlib|subprocess|base64|tarfile|pickle.load" file.py →
     must be empty (a couple of docstring mentions of sha256/hashlib are allowed,
     but no executable code — verify with grep for the actual calls).
   - grep -n "final_submission_runs|experiments/|Oracle/|/home/vishwa" file.py
     → must be empty.
   - python -c "import ast; ast.parse(open('file.py').read())" → OK.
   - Check the ONLY pd.read_csv calls reference train.csv/test.csv/PI1M.csv.
2. **Runtime on the GPU laptop** (python:
   ~/Desktop/AISEHack-2.0/.venv-polymer/bin/python, rdkit 2026.3.4, sklearn
   1.9.0, lightgbm 4.7.0):

       cd /tmp/r3_runtime && mkdir -p v57_final && cd v57_final
       cp <repo>/final_submissions/v57_reproduction_standalone.py .
       cp -r <official ppp-round-2 dir with train/test/PI1M> ./ppp-round-2
       python v57_reproduction_standalone.py --data-dir ./ppp-round-2 --out submission.csv

   Expect 30–120+ minutes (C284/C285/C391 read 100–120k PI1M rows and fit SVD;
   the chain trains dozens of models). Monitor run.log; the script prints
   progress markers [1/6]..[6/6] and per-target summary at the end.
3. **Verify the CSV**: 4,940 rows, ids 1..4940 sequential, all finite, columns
   exactly id,target.
4. **Oracle score** (on the Mac): use the command in §3; expect
   verified_oracle.score.unweighted_mean_r2 ≥ 0.9041.
5. **Reproducibility**: rerun once more; identical CSV (byte-for-byte) expected
   given fixed seeds.

## 8. What to do if there is an issue

- **ImportError / missing name at runtime:** the file inlines ~40 tools with
  prefixed helpers; a missing prefix usually means a cross-tool reference was not
  rewritten. Search the failing function name in the file, find which tool it
  belongs to, and fix the call to the prefixed name (pattern <file>_<fn>).
- **SyntaxError:** the file is generated by concatenating sanitized sources; check
  for duplicated from __future__ (only one allowed, at the very top), stray
  top-level try: with stripped body (import guards), or double-prefixed defs
  (e.g. f03_clean_f03_clean_build_f03).
- **Wrong score but runs:** compare per-target summary against the reference
  (tg ≈ 0.95+, egc ≈ 0.94+, egb/ei/eea/nc/eps lower). Check that the char-arm
  residuals use C282 OOF prediction (not target), that the compound weights
  match §6 exactly, and that splice routing (char vs spread) is per-target.
- **Missing intermediate (None passed to blend/splice):** an output variable was
  not produced because its generator call was skipped; trace the intermediate
  name in the chain and add the producing call (see §5 trace artifacts).
- **Memory/time:** reduce nothing that changes numbers; instead run on the GPU
  laptop (62 GB RAM) and let it finish.
- If a fix changes ANY numeric computation, re-run and re-verify from step §7.

## 9. Supporting artifacts (on the Mac, mostly under /tmp during development)

- /tmp/inmem_out/ — one clean in-memory port per tool (.py files); these were
  concatenated (with prefix renaming) into the standalone.
- /tmp/inmem_lib/ — clean shared modules (reference_lib.py, fable_common.py,
  round-2 tool copies) used as sources.
- /tmp/bundle_check/ — the extracted R2 V53/V57 notebook bundle (raw reference).
- /tmp/closure_calls.json, /tmp/var_map2.json, /tmp/main_body_gen.txt — the
  backward trace of the chain and the generated driver body.
- scripts/r2_reference/ — Round-2 reference notebooks/copies (read-only).

## 10. Current status

- Standalone v3 assembled (373 functions, ~565 KB), byte-compiles cleanly, all static checks pass (no oracle word, no sha256/hashlib/subprocess/base64, no Round-2 path references; the only CSV reads are train/test/PI1M). Cross-tool name resolution and
  the char/spread/compound tail are in place. hashlib replaced by a pure-Python
  deterministic hash for the two EHT conformer-seed functions (c374/c380).
- NOT yet run end-to-end — user approval of the structure is required before the
  first heavy run on the GPU laptop.
