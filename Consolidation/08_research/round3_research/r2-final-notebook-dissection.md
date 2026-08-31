# Reverse-Engineering Report — Sandman V50–V53 (Round 2 final submissions)

Scope: four final notebooks (V50/V51 "with archive", V52/V53 "without archive"), their embedded
source bundles, the C257 clean-compound audit experiment, and the surrounding tool scripts.
All analysis is read-only. No CSV submission outputs were present under `/tmp/r2dump` (the
directories contain only the four `.ipynb` files), so output sha256/min/max values are quoted
from code/logs, not from re-reading a CSV.

---

## 1. NOTEBOOK STRUCTURE

The notebooks are **thin wrappers**, not self-contained pipelines. Each is only 5–6 cells:

- **V50 / V52** (`with_archive` / `without_archive` rank-1): 6 cells = 3 markdown + 3 code.
- **V51 / V53** (rank-2): 5 cells = 2 markdown + 3 code (they omit the "NOTE TO ORGANIZERS" cell).

Cell layout (V50/V52):
1. **Markdown** — title + one-paragraph summary ("Single-run … pipeline … discovers the official
   competition input bundle, performs EDA, rebuilds descriptors and target models from scratch …").
2. **Markdown** — "NOTE TO ORGANIZERS": author admits ~2000+ experiments, that the final notebook
   is "a bit tainted" and not fully reproducible from scratch, and points to a private GitHub repo
   (`vishwakumaresh@gmail.com`) and a draft paper on Google Drive.
3. **Markdown** — `## Architecture and route` (bulleted capability list + runtime estimate).
4. **Code (~61 lines)** — bootstrap: `SEED = 20260809`, `TARGETS = ['tg','egc','egb','ei','eea','nc','eps']`,
   `locate_bundle()` (searches `ppp-round-2` dir + `/kaggle/input/*`), sha256 of `train.csv`/`test.csv`,
   EDA prints of per-target `count/mean/std/min/max`, and column assertions
   (`train: [smiles,target,target_type]`, `test: [id,smiles,target_type]`).
5. **Code (~9704 lines)** — the actual body: a **base64-encoded compressed tarball** stored as a
   Python string `BUNDLE_B64`, decoded and extracted, then run as a subprocess:
   ```python
   with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64)), mode='r:xz') as tar: ...   # V50/V52
   with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64)), mode='r:gz') as tar: ...   # V51/V53
   env['SANDMAN_VARIANT_KEY'] = 'archive_rank1' | 'archive_rank2' | 'noarchive_rank1' | 'noarchive_rank2'
   subprocess.run([sys.executable, str(BUNDLE_ROOT / 'scripts' / 'sandman_runner.py')], ...)
   ```
6. **Code (~15 lines)** — validation of the written output CSV: columns `['id','target']`, 4940 rows,
   unique ids `np.arange(1,4941)`, all finite; prints output sha256/min/max.

The extracted bundle (`BUNDLE_ROOT`) contains **381 files** (~300+ Python scripts) — the entire
experiment codebase is embedded, not the notebook. The single entry point is
`scripts/sandman_runner.py`, which re-uses `tools/reconstruct_v50_v53_standalone_noarchive_20260809.py`
and `recipes/variant_configs.json` + `recipes/recipe_records.json` (a 489-entry manifest DAG).

**End-to-end?** Yes — one pass via subprocess. The notebook does not iterate or checkpoint at the
notebook level; all orchestration is inside `sandman_runner.py`.

**"Estimated cell count"**: the notebook is ~6 cells; the embedded bundle is the real "notebook"
(381 files; core runtime scripts `sandman_runner.py` 181–201 lines, `final_compound.py` 608 lines,
`baseline_defensible.py` ~3.3k lines, `polymer_official_train_eval_loop.py` 6415 lines).

---

## 2. DATA INPUTS

`locate_bundle()` (in the notebook bootstrap and again in `sandman_runner.py`) finds the first
directory containing both `train.csv` and `test.csv`, checking `ppp-round-2` next to CWD/ancestors,
then `/kaggle/input/ppp-round-2`, `/kaggle/input/polymer-property-prediction-round-2/ppp-round-2`, etc.

Official bundle (asserted in `baseline_defensible.py` / `final_compound.py`):
- `train.csv` — **7409** rows (`smiles, target, target_type`)
- `test.csv` — **4940** rows (`id, smiles, target_type`), ids `1..4940`
- `archive/train.csv` — **6171** rows (same schema as train)
- `PI1M.csv` — optional (only opened by `final_compound.py`/`baseline_defensible.py` for a row-count
  print; not mounted in the "without archive" lane).

**Archive vs current-only lanes:**
- **V50/V51 (with archive)** set `SANDMAN_VARIANT_KEY = archive_rank1/rank2`. Their base/source CSVs
  live under `experiments/final_submission_runs/with_archive/...`, whose leaf scripts
  (`final_compound.py`, `baseline_defensible.py`, and the archive-branch builders) read
  `archive/train.csv`. Archive rows are pooled into the label matrix as `labels` (archive first, then
  train overwrites).
- **V52/V53 (without archive)** set `noarchive_rank1/rank2`. Their base/source CSVs are all
  `experiments/final_submission_runs/without_archive/...`; the active route never opens
  `archive/train.csv` (the notebook markdown says exactly this: "the archive label file is not loaded
  by the active route").

Note the with-archive *blend* is actually a **hybrid**: its `selected` table mixes `with_archive`
sources (ei=C1386, eps=C513) with `without_archive` sources (eea=C483, egb=C1378, egc=C1369,
nc=C1530). So "with archive" = archive data feeds the base and the ei/eps arms; the other arms are
current-only even in the archive notebooks.

---

## 3. FEATURE PIPELINE

Features are produced by `polymer_official_train_eval_loop.py:build_features()` (Round-1 code reused
by all Round-2 components) and by the simpler standalone block in `final_compound.py`. Every structure
is canonicalized first (`Chem.MolToSmiles(canonical=True, isomericSmiles=True)`, `[*]→*`).

### Dense blocks
- **`descriptor_matrix`** — every `rdkit.Chem.Descriptors._descList` entry (the full 200+ descriptor
  set, minus nothing here; `final_compound.py` drops `Ipc`), plus 20 hand-written extras
  (`smiles_len, star_count, atom_count, heavy_atom_count, dummy_atom_count, ring_count,
  aromatic_atom_count, hetero_atom_count, halogen_count, n/o/s/si/f/cl/br counts, double/triple bond
  counts, branch_count ('(' count), bracket_count ('[' count)`), plus the **EState** fingerprint
  (79 min + 79 max values).
- **`physics_feature_matrix`** (26 features) — Gasteiger charge min/max/mean/std/abs-mean/abs-max,
  formal charge sum & abs-sum, radical electrons, avg valence/degree, sp/sp2/sp3 hybridization
  fractions, conjugated/aromatic/single/double/triple bond fractions, and **polymer-endpoint
  features** (dummy `*` neighbor atomic numbers, aromatic/ring/degree sums, endpoint path length,
  direct-bond-present).
- **`backbone_sidechain_matrix`** — backbone (between the two `*` endpoints) vs pendant side-chain
  decomposition, with region rotatable-bond/atom-count/charge stats and side-distance/component stats.
- **`conjugation_feature_matrix`** — conjugated-atom count and longest conjugated path (aromatic/sp/sp2
  subgraph diameter).
- **`mobility_feature_matrix`** — free-volume/topological-shape features.
- **`electronic_tail_feature_matrix`** — low-gap SMARTS counters (cyano, imide, sulfone, quinone-like,
  thiophene, triazine, acceptor-carbonyl, vinylene, ethynylene …) plus acceptor/path counts.
- **`topological_autocorr_feature_matrix`**, **`huckel_spectrum_feature_matrix`** (Hückel spectrum),
  **`infinite_chain_proxy_feature_matrix`**, **`bicerano_feature_matrix`**, **`mordred_descriptor_matrix`**,
  **`motif_dense_matrix`** (atom/bond/triplet tokens `S|…`, `P|…`, `T|…`), and **3D descriptors**
  (`rdkit_3d`, `oligomer_3d`, ETKDG + UFF).

### Sparse / hashed blocks
- `maccs_bit`, `morgan_count_r1..r5` (radius 1–5, count, log1p), `morgan_bit_r1..r5`,
  `atom_pair_count`, `topological_torsion_count`, `fcfp_count_r2`, `fcfp_bit_r2`, `rdk_bit`,
  `char_text` (HashingVectorizer char n-grams 2–5, n_features=32768), `motif_hash_count`,
  `map4_like_count`, `endpoint_path_ngram_count`, `rooted_smiles_text`, `random_smiles_text`,
  `kekule_smiles_text`, region (backbone/side) sparse, `wl_subtree`, `exact_morgan_count`.
- **`n_bits`** default 2048 (fingerprint width) in the Round-2 carrier builds; 512/1024 in
  `final_compound.py`.

### Periodic / capped / oligomer / Flory-Fox representations
- **`periodic_closure_mol`** — wraps the `*…*` repeat unit into a ring (periodic boundary) for
  periodic fingerprints/descriptors.
- **`capped_descriptor_mols` / `cap_polymer_smiles`** — replaces `*` endpoints with explicit H (or C)
  before re-computing descriptors (`capped_*` features).
- **`oligomer_mols`** — concatenates the repeat unit n times (1-mer/2-mer/3-mer) and recomputes the
  descriptor + fingerprint set for each oligomer.
- **`oligomer_ffox_descriptor_matrix`** — **Flory-Fox-style asymptotic carriers**: builds n-mer
  descriptors for n=1..max_repeats, normalizes each by heavy-atom count, then **extrapolates against
  the coordinate `1/n`** (both raw and slope transforms; `oligomer_ffox_transform="both"`,
  `oligomer_ffox_max_repeats=3`).
- **`oligomer_slope_descriptor_matrix`** — per-descriptor slope across repeat counts.

### final_compound.py / baseline_defensible.py composite matrix
`X_TREE = hstack([ handcrafted(desc + topo + polymer-genome), SVD(64 of [morgan-count(1024) + char]),
morgan_r2(512), morgan_r3(512) ])` where:
- `topology_block` adds Gasteiger stats, conjugation, backbone/side atoms, spectral gap/top/bottom
  of the adjacency matrix, weighted bond sum.
- `polymer_genome_block` = `S|`/`P|`/`T|` token counts (log1p), top-384 vocabulary.

### Cross-property covariate construction (`context_block` in final_compound.py)
For target `j` it appends every other target's best estimate + availability flag, plus these
physics identities:
```python
context = [
    (ei + eea) / 2.0, ei - eea, eea + egc, ei - egc,
    egb - egc, egb, egc, nc ** 2, eps - nc ** 2,
    nc, eps, np.sqrt(np.maximum(eps - 0.65, 0.05)),
    np.maximum(nc ** 2 + 0.65, 0.0), availability.sum(axis=1), base_predictions[:, exclude_index],
]
```

---

## 4. MODELS PER TARGET

There are **two distinct lineages** that must not be conflated:

### (A) The "clean compound" audit lineage (C050 → … → C257) — provenance of the *defensible* compound
From `metrics.json` of `R2-C257` (`selected_components`, with per-target OOF R², rows, and source):

| target | component run | candidate R² | parent R² | rows | mechanism |
|---|---|---|---|---|---|
| tg   | C050 parent (unchanged) | 0.9089 | — | 5781 | 7-target mixed parent |
| egb  | C050 parent (unchanged) | 0.9221 | — | 337  | 7-target mixed parent |
| egc  | R2-C207 (egc c180 transfer-guard) | 0.9221 | 0.9115 | 2832 | Flory-Fox carrier + transfer guard |
| ei   | R2-C199 (ei c196 transfer-guard) | 0.8567 | 0.8454 | 222  | Flory-Fox carrier, 0.75 shrink + guard |
| eea  | R2-C189 (ffox eea confirmation) | 0.9163 | 0.9008 | 221  | Flory-Fox direct carrier |
| nc   | R2-C252 (nc eps-ionic projection) | 0.8832 | 0.8397 | 229  | nc = sqrt(eps − ionic) |
| eps  | R2-C214 (eps ionic full-amplitude) | 0.8501 | 0.7835 | 229  | eps = nc² + exp(ionic) |

The C257 `component_priority` lists the full frozen fallback stack per target (e.g. eea:
C204→C189→C188→C192; ei: C224→C222→C220→C199→C196→C194→C188→C192; nc: C252→C242→C240→C236→…→C188→C192;
eps: C238→C224→C222→C216→C214→C190→C188→C192; tg: C254→C244→C232→C228→C208). Many of those were
`skipped_components` with reason `target_not_banked` in the C257 audit because the target was already
banked from a prior/fallback component.

**Component mechanics (all re-run the C050 parent for parity, then add one target's residual):**
- **C050 parent** (`run_round2_mixed_candidate_v7.py`): 7-target mixed candidate. For `ei`/`eea` it
  uses a "route" with a gap model + nearest-neighbour similarity barrier (`SIMILARITY_BARRIER=0.70`,
  blocked scaffold `c1ccsc1`); other targets carry the C001-preserved panel. Grouped 5-fold.
- **C189 eea / C180 carriers** (`round2_c127_round1_carrier_factory.py`): `DIRECT_BLOCKS` =
  maccs + morgan count r1–r5 + morgan bit r2 + atom-pair + torsion + char + periodic morgan r2/r3 +
  capped morgan r2. Two direct arms: **Ridge(α=30, lsqr)** on `sparse + scaled-dense`, and
  **ExtraTrees(160, leaf=2, max_features=0.65)** on dense. Blend arms = `[parent, ridge_oof, tree_oof]`
  via NNLS `blend_from_oof`.
- **C207 egc / C199 ei**: C180 direct carrier, then `transfer_guard` reverts to parent on
  pre-declared failure slices (scaffold `c1ccccc1` and similarity 0.50–0.70 for ei; negative scaffolds
  + similarity < threshold for egc). C199 additionally applies `SHRINK_ALPHA=0.75`.
- **C214 eps**: C187/C190 ionic runner with `HALF_PARENT` raised 0.50→**1.00** (full amplitude):
  fit `log(eps − nc²)` with Ridge(α=50)/ExtraTrees(300)/HistGB(220), then
  `eps = nc² + mean(exp(pred_log_ionic))`.
- **C252 nc**: fit the same ionic models, then `nc = (1−0.5)·nc_parent + 0.5·sqrt(max(eps_selected − ionic, 0.05²))`
  using the **C214-selected eps** as the source (`PROJECTION_WEIGHT=0.50`, `NC_FLOOR=0.05`).

### (B) The *final submission* blend lineage (`variant_configs.json`) — what V50–V53 actually write
The submitted file is a **target-wise signed residual blend**, computed per target as:
```python
values[mask] = base_values[mask] + weight * (source_values[mask] - base_values[mask])
```
(only rows whose `target_type` matches the target are touched). The recipe:

**archive_rank1 (V50)** — base `R2-C1577-ARCHIVE-CURRENT-ONLY-EI-EPS-ARMS-OVER-C1573`:
eea −0.14268, egb +0.13298, egc +0.03699, ei +0.25125, eps +0.33165, nc +0.17132, tg 0.0.

**archive_rank2 (V51)** — base `R2-C1579-ARCHIVE-COMBINED-TARGET-LEADER-OVER-C1577`;
only egc (+0.036979) and nc (+0.170883) weights differ (tiny re-tune), everything else identical.

**noarchive_rank1 (V52)** — base `R2-C1570-NOARCHIVE-JOINT-PHYSICS-GRID-OVER-C1567`, with a **two-level
blend**: `internal_noarchive_blend1` (base `R2-C1572…`, arms eea +0.05586, egb −0.22805, egc −0.03966,
ei −1.15460, eps −0.37639, nc −1.67022, tg +0.15927), then the top blend uses egb =
`__internal_noarchive_blend1__` (weight 1.00236) and re-uses the other arms with slightly different
weights (eea 0.05598, ei −1.17557).

**noarchive_rank2 (V53)** — base `R2-C1572-NOARCHIVE-SPLICE-C1570-EGC-OVER-C1567`; the single-level
blend whose `selected` is essentially V52's internal arms directly (egb = `R2-C565` at −0.22805).

### The blending arithmetic is "REFLECT / SPLICE / BLEND" recursion
The 489 manifest records encode schemas like `ppp.round2.target-splice.v1`,
`ppp.round2.branch-target-blend.v1` (per-target weight-on-source),
`ppp.round2.c415.reflected-source.v1` (base + (source−base) reflection),
`ppp.round2.clean-current-epsnc-ionic-overlay.v1`, `…joint-physics-grid.v1`, etc. The `Rebuilder`
walks this DAG (see §6) so the final CSV is a long chain of per-target signed-residual overlays.

---

## 5. VALIDATION & FOLDS

- **Grouped CV**: `GroupKFold(n_splits=5)` grouped by **canonical no-stereo SMILES**
  (`canonical_no_stereo`) — same polymer with duplicate measurements is kept in one fold
  (`carrier.grouped_folds`, `c187`, `c252`). The C050 parent uses 4/5-fold grouping with a
  nearest-neighbour similarity barrier. `final_compound.py`'s zoo uses plain `KFold(3 or 5,
  shuffle=True, random_state=SEED+len(tag))` (3 folds when rows ≥ 1200).
- **OOF**: each arm is predicted fold-locally; `direct_oof`/`pair_oof` arrays are filled on
  validation rows only, then a weighted blend (`blend_from_oof` / NNLS `choose_weights`) with a
  fitted intercept is scored against the full OOF.
- **Banking gate** (used by every component): candidate R² vs parent R² must satisfy
  `delta_r2 ≥ 0.01`, `positive_folds ≥ 4/5`, **group-bootstrap lower 2.5% quantile > 0**
  (2000 resamples of groups), and `minimum_panel_delta ≥ 0` (panels = similarity buckets, quantile
  buckets, and per-scaffold slices ≥ 10 rows). Full-compound gate: `mean_gain ≥ 0.002` and
  `max_target_loss ≥ −0.003`.
- **Blend-weight choice**: in-component weights come from NNLS (non-negative, normalized, with
  single-arm fallbacks). The *final submission* weights (`variant_configs.json`) are **frozen
  signed weights chosen before the notebook run** (the "REFLECT" recipes), not re-fit at runtime.
- **Parity**: every component re-runs the C050 parent and asserts bit-exact replay
  (`source_parity`, tolerance `1e-12`). C257 reported `oof_max_abs = 1.14e-13`,
  `test_max_abs = 1.14e-13`, `pass: true`.

---

## 6. RUNTIME & FEASIBILITY

- Notebook markdown: **"about 30-90 minutes for the archive notebooks and 20-60 minutes for the
  current-only notebooks"** on the dev workstation; "No network access is required after Python
  dependencies are present."
- **CPU only.** No GPU/TPU. `n_jobs=2`/`n_jobs=4` on ExtraTrees; Ridge via `lsqr`; lightgbm/xgb/catboost
  on CPU. Environment: `python=3.12.3, numpy=2.5.1, pandas=3.0.5, sklearn=1.9.0, rdkit=2026.03.5`
  (from the C257 `environment.txt`). Kaggle flags (`kaggle_compute/submission/upload`) are all `false`.
- **Implied compute**: the "notebook" is a recipe-replay engine. `recipe_records.json` holds **489
  manifest records**; `Rebuilder.materialize()` recursively re-runs tool scripts to regenerate every
  CSV in the dependency DAG from official inputs (header: "no prior prediction CSV is read"). The final
  blend materializes 1 base + up to 7 source CSVs (V52 also materializes the internal blend first),
  each chaining back through many splice/blend/reflect steps to leaf scripts (C050, C214, C187, C199,
  C252, C282, C284, C285, C340, C391, C287 zoo, F01–F06 fable, etc.) that each re-fit several models
  × 5 folds. So the true cost is **hundreds of small Ridge/ExtraTrees/HistGB/LGBM fits**, not one big
  model. The C257 audit alone logged `elapsed_seconds=143.8` (audit-only, reusing/replaying C050).
- Leaf-script caching: `Rebuilder` short-circuits any `work_path` CSV that already exists and skips
  completed sub-DAGs via `*_done` flags, so a warm cache is much faster.

---

## 7. WHAT CHANGED BETWEEN V50/V51 (with archive) AND V52/V53 (without archive)

Identical machinery (`sandman_runner.py` + `reconstruct` + `variant_configs.json`); the only
differences are the recipe key and the data lane:

| | V50 (archive_rank1) | V51 (archive_rank2) | V52 (noarchive_rank1) | V53 (noarchive_rank2) |
|---|---|---|---|---|
| bundle compression | tar.xz | tar.gz | tar.xz | tar.gz |
| branch | with_archive | with_archive | without_archive | without_archive |
| base CSV | C1577 (archive current-only EI/EPS arms) | C1579 (combined target leader over C1577) | C1570 (joint physics grid) | C1572 (splice C1570 egc over C1567) |
| archive/train.csv read | yes (via base + ei/eps arms) | yes | **no** | **no** |
| ei source | C1386 (with_archive) | C1386 (with_archive) | C1349 (without_archive) | C1349 (without_archive) |
| eps source | C513 (with_archive) | C513 (with_archive) | C488 (without_archive) | C488 (without_archive) |
| egb source | C1378 | C1378 | internal two-level blend (→C565) | C565 directly |
| two-level blend | no | no | yes (`internal_noarchive_blend1`) | no |
| weight magnitudes | small (±0.03–0.33) | small | large (ei −1.18, nc −1.67) | large (ei −1.15, nc −1.67) |
| V50↔V51 / V52↔V53 diff | — | only egc/nc weight + base | — | base + egb arm + ei/eea weights |

Concretely, removing archive changes: (1) the base is a **current-only** composite (C1570/C1572,
which never open `archive/train.csv`), (2) the ei and eps arms switch from archive-branch sources to
noarchive sources, and (3) the blend weights grow in magnitude and become negative for ei/egc/nc/eps
(the noarchive lane must *correct* the current-only base with larger signed residuals, including a
nested two-level blend for egb in V52).

---

## 8. SEEDS & REPRODUCIBILITY

- Notebook bootstrap: `SEED = 20260809`, `np.random.seed(SEED)` (cosmetic; the real work happens in
  the subprocess bundle).
- `final_compound.py` / `baseline_defensible.py`: `SEED = 20260804`, `np.random.seed(SEED)`.
- `round2_c127_round1_carrier_factory.py` / `round2_c180_flory_fox_oligomer_carriers.py` /
  `round2_c189_ffox_eea_confirmation.py`: `SEED = 2026`.
- `round2_c187_ionic_eps_only.py`: `SEED = 20260804` (reused by C214).
- `round2_c199_ei_c196_transfer_guard.py`, `round2_c207_egc_c180_transfer_guard.py`,
  `round2_c252_nc_eps_ionic_projection.py`, `round2_c238…`: `SEED = 20260805`.
- Model seeds are derived (`SEED + fold`, `base_seed + 101/131/…`); SVD/QuantileTransformer take
  `random_state=SEED`; the group-bootstrap uses `np.random.default_rng(SEED)` (2000 resamples).
- **Determinism**: GroupKFold is deterministic; `final_compound.py`'s `KFold(shuffle=True)` is seeded.
  Every component asserts **exact parent replay parity at 1e-12** (`source_parity`), and the
  reconstruct harness compares regenerated V52/V53 sources against reference sha256
  (`EXPECTED_HASHES`), tolerance 1e-12.
- **Known nondeterminism/taint concerns**: the author's "NOTE TO ORGANIZERS" explicitly states the
  final notebook "is a bit tainted and its difficult for me to produce a clean version fully
  reproducable from scratch", after ~2000+ experiments. ETKDG/UFF 3D conformers (if enabled) and
  LightGBM/XGBoost/CatBoost tree growth are seeded but not bit-reproducible across library builds;
  the runtime records library versions and source-hash manifests (`source_hashes`,
  `artifact_manifest.sha256`) to pin provenance.

---

## 9. KEY COMPONENT FORMULAS

These are the physical identities/coordinates threaded through the components, and how they enter:

1. **Ionic coordinate** (`c187`/`c214`/`c252`, `final_compound.py`):
   ```python
   ionic_y = eps - nc**2          # must be > 0; floor 0.05
   log_ionic = log(ionic_y)       # regression target
   eps_raw  = nc**2 + mean(exp(clip(pred_log_ionic, -8, 4)))
   ```
   C214 uses full amplitude: `eps = eps_raw` (HALF_PARENT=1.0); C187 uses `(1−0.5)·parent + 0.5·eps_raw`.

2. **Optical coordinate**: `optical = nc**2` (the refractive-index-like part of the dielectric).

3. **Nc projection from eps** (`c252`):
   ```python
   raw_nc = sqrt(max(selected_eps - ionic, 0.05**2))
   nc = (1−0.5)·nc_parent + 0.5·raw_nc          # PROJECTION_WEIGHT=0.50
   ```

4. **Electronic coordinates** (`c156`):
   ```python
   chi = (Ei + Eea) / 2      # electron affinity mid-point
   gap = Ei - Eea            # ionization-energy gap
   ```

5. **Identity routes** (`c160`, `final_compound.py`):
   ```python
   Ei  = Eea + Egc           # direct or 0.5·parent + 0.5·raw (c160: 0.75/0.25)
   Eea = Ei  − Egc           # direct or 0.5·parent + 0.5·raw
   Egb = 1.1178·Egc − 0.9221 # linear identity (c160), blended 0.75·parent + 0.25·raw
   ```

6. **Flory-Fox carriers** (`c180`/`c189`): n-mer descriptor values, normalized by heavy-atom count,
   extrapolated against the asymptotic coordinate **1/n** (n = repeat-unit count 1..3).

7. **Cross-property context** (`final_compound.py` stage-2): `(ei+eea)/2, ei−eea, eea+egc, ei−egc,
   egb−egc, egb, egc, nc², eps−nc², nc, eps, sqrt(max(eps−0.65,0.05)), max(nc²+0.65,0), availability
   sum, and the target's own stage-1 prediction`.

8. **Final submission blend** (all four notebooks):
   ```python
   values[target_rows] = base[target_rows] + weight * (source[target_rows] − base[target_rows])
   ```
   where `weight` is the frozen per-target coefficient from `variant_configs.json`.

9. **Consistency override** (`final_compound.py`): for test structures carrying both eps and nc rows,
   `eps = max(eps, nc**2 + 0.02)`; plus exact canonical overrides from unique labelled values, and a
   final per-target clip to `[min − 0.02·span, max + 0.02·span]`.

---

## Key file references (inside the extracted bundles)

- `scripts/sandman_runner.py` — orchestrator; locates data, loads `variant_configs.json` +
  `recipe_records.json`, drives `Rebuilder`, computes the final target blend.
- `tools/reconstruct_v50_v53_standalone_noarchive_20260809.py` — the `Rebuilder`/`materialize`/
  `run_schema` recipe-replay engine (1803 lines).
- `recipes/variant_configs.json` — the 4 frozen blend recipes (base + selected + weights).
- `recipes/recipe_records.json` — 489 manifest records (schema, base/blends/sources deps, sha256).
- `scripts/final_compound.py` (608 lines) and `scripts/baseline_defensible.py` — the 7-target
  "defensible" compound (stage1/stage2 zoo + ionic + identity routes).
- `tools/polymer_official_train_eval_loop.py` (6415 lines) — the shared `build_features` +
  `model_specs` + CV/OOF machinery.
- Component scripts: `round2_c127_round1_carrier_factory.py`, `round2_c180_flory_fox_oligomer_carriers.py`,
  `round2_c187_ionic_eps_only.py` (+`round2_c214_eps_ionic_full_amplitude.py`),
  `round2_c252_nc_eps_ionic_projection.py`, `round2_c199_ei_c196_transfer_guard.py`,
  `round2_c207_egc_c180_transfer_guard.py`, `round2_c189_ffox_eea_confirmation.py`,
  `round2_c160_observed_physical_identities.py`, `round2_c156_latent_physical_coordinates.py`,
  `run_round2_mixed_candidate_v7.py` (C050 parent).
