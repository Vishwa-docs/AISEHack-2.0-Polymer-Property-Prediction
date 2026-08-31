# Phase_3 runtime mirror (GPU laptop) — 2026-08-30

This directory mirrors the experiment suite that runs on the GPU laptop at
`~/Desktop/r3_runtime/Phase_3/` (authoritative location for running).

- `run.sh` — master parallel runner (resource-aware dual-pool scheduler,
  smile_r3-first ordering, resumable, auto-detects experiment count 282).
- `gen_experiments.py` — registry/generator for all 282 experiment scripts.
- `r3_core/` — shared library with the NEW capabilities added 2026-08-30:
  - ssl.py: fixed PPMI-SVD broadcast bug; new atom-level PPMI-SVD (`atom_ppmi`)
    and atom-token MLM (`atom_mlm`) — chemically-aware tokenization (ask §3.2).
  - nn.py: `AtomTokenizer` (regex: Cl/Br/Si/[ * ]/bonds/ring-digits as one token).
  - features.py: `polar_moieties_block` (C=O/ester/ether/sulfone/amide/C-F/
    siloxane counts), `gasteiger_separation_block` (sigma/pi separation),
    `conformer3d_block` (ETKDG+UFF 3D descriptors).
  - harness.py: physics coordinates mulliken (chi) + gapres (ei-eea / egb-egc);
    new kinds mixture / recalib / shiftweight / matfac / uncertainty; curation
    (tg_median, drop_overlap, drop_near_dup, overlap_weight); kmeans folds;
    row weights (sample_weight threading); audit now reports the 457-overlap
    audit + shift-matched reweighted R².
- New experiments: indices 251–282 (Phases L/M/N/O, see Phase_3 README on the
  laptop). smile_r3 atom-SSL (251-262) run first per the user request.

Overnight command (from the laptop):
```bash
cd ~/Desktop/r3_runtime/Phase_3 && nohup ./run.sh > run_log.txt 2>&1 &
```

