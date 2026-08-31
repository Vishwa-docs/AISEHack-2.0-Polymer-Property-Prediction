"""r3_core — shared, official-data-only core for the Round 3 Phase_2 experiment suite.

Every module here reads ONLY the official Dataset/ inputs (train.csv, test.csv,
and — only where a protocol explicitly allows it — PI1M.csv / smile_r3.csv).
Nothing here imports, reads, or references Oracle/, old R2 CSVs, hashes, or
experiment records.  All models are trained from scratch with fixed seeds.

Modules
-------
data     : official data loading, canonical structure keys, grouped folds,
           target statistics, and the frozen 457 train/test overlap audit.
features : RDKit descriptors, Morgan counts/bits, char n-grams, polymer-genome
           fingerprint, physics blocks, SVD — all fit from scratch in-process.
metrics  : per-target R2 / MAE / RMSE, unweighted mean, grouped fold statistics,
           similarity and availability panels, shift-matched weighting.
models   : per-target Ridge / ExtraTrees / HistGB / LGBM / Tanimoto-KRR arms and
           a simple OOF NNLS blend, all with fixed seeds and grouped CV.
physics  : the R2-bankable coordinate identities (ionic = eps - nc^2,
           chi = (ei+eea)/2, ei = egc+eea, egb ~ a*egc+b) as feature builders.
panels   : scaffold / family / similarity-cluster fold builders and helpers.
"""
