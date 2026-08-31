# scripts/

Working scripts for Round 3. Everything here lives in THIS repo and may be
modified freely.

- `r2_reference/` — Round 2 code copied here because Round 3 builds on it:
  - `Sandman_Version_52_8th_Aug_without_archive.ipynb` — the best notebook-backed
    no-archive submission notebook from Round 2 (base of the R3-C000 port).
  - `Sandman_Version_53_8th_Aug_without_archive.ipynb` — sibling variant.
  - `fable_engines/` — F01 (ei/eea/egb chain engine), F02 (eps/nc ionic engine),
    `fable_common.py` (grouped folds + shift-matched R²), plus the fable README.
- Anything new you write goes in `scripts/` (or an experiment dir), not in the
  repo root.

GPU laptop jobs: copy scripts to laptop scratch (`/tmp/r3_runtime/`), run, copy
results back here. Never edit files on the laptop itself.
