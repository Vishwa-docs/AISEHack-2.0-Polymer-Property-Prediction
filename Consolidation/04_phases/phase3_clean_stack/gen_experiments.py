#!/usr/bin/env python3
"""Generate the 250 Phase 3 experiment scripts from the PLAN.md registry.

Run on the GPU laptop:  python gen_experiments.py
Writes experiments/exp001_p3A01.py ... exp250_p3K25.py (thin config wrappers
around r3_core.harness.run_config). Every script reads ONLY the official
Dataset/ inputs passed via --data-dir. No oracle, no external data, no
pretrained artifacts — all representations fit from scratch in-process.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments"

TEMPLATE = '''#!/usr/bin/env python3
"""Phase 3 experiment {idx:03d} [{name}] — {desc}

Generated from score_discrepancy/PLAN.md. Official-data-only; {long_note}
no oracle, no external data, no pretrained artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r3_core import harness

CFG = {cfg}

CFG["name"] = "{name}"
CFG["exp_id"] = "R3-P3-{idx:03d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs_and_logs/output/exp{idx:03d}_{name}")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    metrics = harness.run_config(
        CFG, Path(args.output), smoke=args.smoke,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )
    print(json.dumps({{"exp": CFG["name"], "mean_r2": metrics.get("mean_r2")}}, default=float))


if __name__ == "__main__":
    main()
'''


def G(kind="gbm", **kw):
    cfg = {"kind": kind}
    cfg.update(kw)
    return cfg


# Default model / feature presets ------------------------------------------------
LGBM = {"type": "lgbm"}
BASE = {"base": True, "svd_dim": 64}

# =============================================================================
# Phase C — smile_r3 / PI1M SSL representation learning (031-060)
# =============================================================================
def _ssl(method, corpus="smile_r3", n=100_000, dim=128, **kw):
    return {"base": True, "ssl": {"method": method, "corpus": corpus, "n": n, "dim": dim, **kw}}


EXPERIMENTS: list[tuple[int, str, str, dict]] = []

EXPERIMENTS += [
    (31, "p3C01-r3-svd-100k", "smile_r3 TF-IDF+SVD (100k sample, 128d) added to baseline",
     G(features=_ssl("svd", n=100_000))),
    (32, "p3C02-r3-svd-500k", "smile_r3 TF-IDF+SVD (500k)", G(features=_ssl("svd", n=500_000))),
    (33, "p3C03-r3-svd-2m", "smile_r3 TF-IDF+SVD (2M)", G(features=_ssl("svd", n=2_000_000))),
    (34, "p3C04-r3-svd-full", "smile_r3 TF-IDF+SVD FULL 5.97M, 256d",
     G(features=_ssl("svd", n=None, dim=256), long="LONG RUN (~2-4h)")),
    (35, "p3C05-r3-svd-dims-512", "smile_r3 SVD-512 on 2M sample",
     G(features=_ssl("svd", n=2_000_000, dim=512))),
    (36, "p3C06-r3-morgan-idf", "Morgan substructure-IDF vocabulary from 300k smile_r3",
     G(features=_ssl("morgan_idf", n=300_000))),
    (37, "p3C07-r3-w2v-100k", "PPMI-SVD word2vec-analogue on 100k smile_r3",
     G(features=_ssl("ppmi", n=100_000))),
    (38, "p3C08-r3-w2v-1m", "PPMI-SVD on 1M smile_r3", G(features=_ssl("ppmi", n=1_000_000))),
    (39, "p3C09-r3-w2v-5m", "PPMI-SVD on 5M smile_r3",
     G(features=_ssl("ppmi", n=5_000_000, dim=256), long="LONG RUN (~2h)")),
    (40, "p3C10-r3-w2v-window8", "PPMI-SVD with window=8 (long-range co-occurrence)",
     G(features=_ssl("ppmi", n=1_000_000, window=8))),
    (41, "p3C11-r3-ppmi-2m", "PPMI-SVD on 2M smile_r3", G(features=_ssl("ppmi", n=2_000_000))),
    (42, "p3C12-r3-mlm-tiny-100k", "tiny MLM transformer (2L, 64d) pretrained on 100k smile_r3",
     G(features=_ssl("mlm", n=100_000, dim=64, layers=2))),
    (43, "p3C13-r3-mlm-tiny-1m", "tiny MLM on 1M smile_r3",
     G(features=_ssl("mlm", n=1_000_000, dim=64, layers=2), long="LONG RUN (~2h GPU)")),
    (44, "p3C14-r3-mlm-small-1m", "4-layer 128d MLM on 1M smile_r3",
     G(features=_ssl("mlm", n=1_000_000, dim=128, layers=4), long="LONG RUN (~6h GPU)")),
    (45, "p3C15-r3-mlm-small-5m", "4-layer 128d MLM on 5M smile_r3",
     G(features=_ssl("mlm", n=5_000_000, dim=128, layers=4), long="LONG RUN (overnight GPU)")),
    (46, "p3C16-r3-mlm-medium-5m", "6-layer 256d MLM on 5M smile_r3",
     G(features=_ssl("mlm", n=5_000_000, dim=256, layers=6), long="LONG RUN (overnight GPU)")),
    (47, "p3C17-r3-pi1m-svd", "PI1M polymer SVD (995k)", G(features=_ssl("svd", corpus="pi1m", n=None))),
    (48, "p3C18-r3-pi1m-w2v", "PI1M PPMI-SVD", G(features=_ssl("ppmi", corpus="pi1m", n=None))),
    (49, "p3C19-r3-combined-svd", "combined PI1M + smile_r3 2M SVD",
     G(features=_ssl("svd", corpus="combined", n=2_000_000))),
    (50, "p3C20-r3-knn-idf", "Morgan-IDF 4096-bit corpus weighting",
     G(features=_ssl("morgan_idf", n=500_000, bits=4096))),
    (51, "p3C21-pi1m-mlm", "tiny MLM on PI1M polymers",
     G(features=_ssl("mlm", corpus="pi1m", n=None, dim=64, layers=2), long="LONG RUN (~1h GPU)")),
    (52, "p3C22-combined-ppmi", "combined corpus PPMI-SVD 2M",
     G(features=_ssl("ppmi", corpus="combined", n=2_000_000))),
    (53, "p3C23-svd-combined-256", "combined corpus SVD-256 2M",
     G(features=_ssl("svd", corpus="combined", n=2_000_000, dim=256))),
    (54, "p3C24-mlm-128d", "MLM(100k) at 128 dims + base stack",
     G(features=_ssl("mlm", n=100_000, dim=128, layers=2))),
    (55, "p3C25-pseudo-tg-pi1m", "pseudo-label Tg on 20k PI1M polymers (self-training round 1)",
     G("pseudo", targets=["tg"], pseudo={"corpus": "pi1m", "n": 20000, "top_k": 2000})),
    (56, "p3C26-pseudo-egc-pi1m", "pseudo-label egc on 20k PI1M",
     G("pseudo", targets=["egc"], pseudo={"corpus": "pi1m", "n": 20000, "top_k": 1500})),
    (57, "p3C27-svd-500k-egc", "SVD-500k features focused check on egc",
     G(features=_ssl("svd", n=500_000), targets=["egc"])),
    (58, "p3C28-svd-500k-ei", "SVD-500k features on ei",
     G(features=_ssl("svd", n=500_000), targets=["ei"])),
    (59, "p3C29-svd-500k-eps-nc", "SVD-500k features on eps+nc",
     G(features=_ssl("svd", n=500_000), targets=["eps", "nc"])),
    (60, "p3C30-ssl-ladder-compound", "NNLS assembly of SSL arms (500k SVD, PPMI-1M, MLM-100k)",
     G("assembly", arms=["exp032_p3C02-r3-svd-500k", "exp038_p3C08-r3-w2v-1m",
                          "exp042_p3C12-r3-mlm-tiny-100k"])),
]

# =============================================================================
# Phase D — physics-informed routes (061-085)
# =============================================================================
EXPERIMENTS += [
    (61, "p3D01-ei-eea-joint", "joint ei+eea multi-task MLP with ei-eea-egc soft constraint",
     G("mlp", targets=["ei", "eea", "egc"], mlp={"physics": True, "physics_lambda": 0.05})),
    (62, "p3D02-ei-gpr", "Gaussian-process regression for ei (222 rows)",
     G(targets=["ei"], model={"type": "gpr", "length_scale": 2.0})),
    (63, "p3D03-ei-gpr-combo", "NNLS blend of GPR-ei and XGB-ei",
     G("assembly", arms=["exp062_p3D02-ei-gpr", "exp004_p3A04-xgboost"], targets=["ei"])),
    (64, "p3D04-eps-nc-joint", "partner-feature injection: eps <- nc pred, nc <- eps",
     G("physics", partner={"eps": ["nc"], "nc": ["eps"]}, targets=["eps", "nc"])),
    (65, "p3D05-eps-ionic-direct", "ionic coordinate: predict eps - nc_pred^2 then invert",
     G("physics", partner={"eps": ["nc"]}, coordinate={"eps": "ionic"}, targets=["eps", "nc"])),
    (66, "p3D06-eps-catboost", "CatBoost on eps (raw scale; log(ionic) known dead)",
     G(targets=["eps"], model={"type": "cat", "iterations": 1600})),
    (67, "p3D07-egb-egc-identity", "egb via egc partner feature",
     G("physics", partner={"egb": ["egc"]}, targets=["egb", "egc"])),
    (68, "p3D08-eea-egc-ei", "eea via egc & ei partner features",
     G("physics", partner={"eea": ["egc", "ei"]}, targets=["eea"])),
    (69, "p3D09-full-physics-routing", "all identity routes: eps<-nc, eea<-egc,ei, egb<-egc",
     G("physics", partner={"eps": ["nc"], "eea": ["egc", "ei"], "egb": ["egc"]})),
    (70, "p3D10-multi-physics-mlp", "7-target MLP with all physics soft constraints",
     G("mlp", mlp={"physics": True, "physics_lambda": 0.1, "epochs": 250})),
    (71, "p3D11-ei-electronic-feats", "ei: Mordred electronic descriptors + base",
     G(targets=["ei"], features={"base": True, "mordred": True})),
    (72, "p3D12-ei-eht-proxy", "ei: conjugation/HOMO-proxy via Mordred-only set",
     G(targets=["ei"], features={"base": False, "mordred": True})),
    (73, "p3D13-eps-nc-polarizability", "eps/nc: polarizability-proxy Mordred block",
     G(targets=["eps", "nc"], features={"base": True, "mordred": True})),
    (74, "p3D14-nc-lorentz-residual", "nc: partner route via eps",
     G("physics", partner={"nc": ["eps"]}, targets=["nc", "eps"])),
    (75, "p3D15-physics-compound", "NNLS compound of D-phase physics arms",
     G("assembly", arms=["exp064_p3D04-eps-nc-joint", "exp065_p3D05-eps-ionic-direct",
                          "exp067_p3D07-egb-egc-identity", "exp068_p3D08-eea-egc-ei"])),
    (76, "p3D16-ei-svr", "SVR for ei", G(targets=["ei"], model={"type": "svr", "C": 10.0})),
    (77, "p3D17-ei-knn", "kNN for ei", G(targets=["ei"], model={"type": "knn", "k": 5})),
    (78, "p3D18-ei-bayesridge", "Bayesian Ridge for ei",
     G(targets=["ei"], model={"type": "bayesridge"})),
    (79, "p3D19-eea-physics-audit", "audit: physics-route vs direct eea arms",
     G("audit", arms=["exp068_p3D08-eea-egc-ei", "exp002_p3A02-clean-stack-v2"])),
    (80, "p3D20-eps-nc-system", "full eps/nc joint system with ionic coordinate",
     G("physics", partner={"eps": ["nc"], "nc": ["eps"]}, coordinate={"eps": "ionic"},
         targets=["eps", "nc"])),
    (81, "p3D21-ei-krr", "kernel ridge (laplacian) for ei",
     G(targets=["ei"], model={"type": "krr", "alpha": 1.0})),
    (82, "p3D22-eeg-arm-xgb", "ei+eea+egc XGB with Mordred block",
     G(targets=["ei", "eea", "egc"], model={"type": "xgb"}, features={"base": True, "mordred": True})),
    (83, "p3D23-egb-catboost", "CatBoost on egb with Mordred",
     G(targets=["egb"], model={"type": "cat"}, features={"base": True, "mordred": True})),
    (84, "p3D24-nc-catboost", "CatBoost on nc with Mordred",
     G(targets=["nc"], model={"type": "cat"}, features={"base": True, "mordred": True})),
    (85, "p3D25-small-targets-vote", "ei/eea/eps/nc vote ensemble with Mordred",
     G(targets=["ei", "eea", "eps", "nc"], model={"type": "vote", "models": [
         {"type": "lgbm"}, {"type": "cat"}, {"type": "ridge", "alpha": 12.0}]},
         features={"base": True, "mordred": True})),
]

# =============================================================================
# Phase E — GNNs from scratch (086-105)
# =============================================================================
EXPERIMENTS += [
    (86, "p3E01-gcn-tg", "GCN (3-layer, 128h) for Tg",
     G("gnn", targets=["tg"], gnn={"style": "gcn", "hidden": 128, "layers": 3})),
    (87, "p3E02-gat-tg", "GATv2-style attention GNN for Tg",
     G("gnn", targets=["tg"], gnn={"style": "gat", "hidden": 128, "layers": 3})),
    (88, "p3E03-mpnn-tg", "MPNN-style edge-attention GNN for Tg",
     G("gnn", targets=["tg"], gnn={"style": "mpnn", "hidden": 128, "layers": 3})),
    (89, "p3E04-gnn-tg-regularised", "GCN with higher dropout + longer training",
     G("gnn", targets=["tg"], gnn={"style": "gcn", "hidden": 192, "layers": 4, "dropout": 0.3, "epochs": 90})),
    (90, "p3E05-gnn-egc", "GCN for egc",
     G("gnn", targets=["egc"], gnn={"style": "gcn", "hidden": 128, "layers": 3})),
    (91, "p3E06-gnn-multitask", "GCN on all 7 targets (shared encoder)",
     G("gnn", gnn={"style": "gcn", "hidden": 128, "layers": 3})),
    (92, "p3E07-gnn-blend", "NNLS blend of GCN-Tg and vote baseline",
     G("assembly", arms=["exp086_p3E01-gcn-tg", "exp002_p3A02-clean-stack-v2"], targets=["tg"])),
    (93, "p3E08-gnn-electronic", "GCN small electronic targets (ei/eea/egc)",
     G("gnn", targets=["ei", "eea", "egc"], gnn={"style": "gcn", "hidden": 128, "layers": 3})),
    (94, "p3E09-gnn-deeper", "GCN 4-layer deeper stack (Tg)",
     G("gnn", targets=["tg"], gnn={"style": "gcn", "hidden": 160, "layers": 4, "epochs": 80})),
    (95, "p3E10-gnn-dimer", "GCN on dimer-expanded chain graphs (Tg)",
     G("gnn", targets=["tg"], gnn={"style": "gcn", "expand": "dimer", "hidden": 128, "layers": 3})),
    (96, "p3E11-gcn-tg-deep6", "6-layer GCN (Tg)",
     G("gnn", targets=["tg"], gnn={"style": "gcn", "hidden": 128, "layers": 6, "dropout": 0.3})),
    (97, "p3E12-gnn-polymer-chain", "GCN on trimer chain graphs (Tg)",
     G("gnn", targets=["tg"], gnn={"style": "gcn", "expand": "trimer", "hidden": 128, "layers": 3})),
    (98, "p3E13-gnn-tg-scaffold-cv", "GCN-Tg with scaffold CV",
     G("gnn", targets=["tg"], cv={"type": "scaffold", "n_splits": 5},
         gnn={"style": "gcn", "hidden": 128, "layers": 3})),
    (99, "p3E14-gnn-small-electronic", "GCN on egb+eea+ei",
     G("gnn", targets=["egb", "eea", "ei"], gnn={"style": "gcn", "hidden": 128, "layers": 3})),
    (100, "p3E15-gnn-compound", "NNLS compound: GCN-Tg arms + vote baseline",
     G("assembly", arms=["exp086_p3E01-gcn-tg", "exp089_p3E04-gnn-tg-regularised",
                          "exp002_p3A02-clean-stack-v2"], targets=["tg"])),
    (101, "p3E16-gnn-capacity", "GCN-Tg with higher-capacity encoder (256h)",
     G("gnn", targets=["tg"], gnn={"style": "gcn", "hidden": 256, "layers": 3})),
    (102, "p3E17-sage-tg", "GraphSAGE-style mean-aggregation GNN (Tg)",
     G("gnn", targets=["tg"], gnn={"style": "sage", "hidden": 128, "layers": 3})),
    (103, "p3E18-gnn-tg-noise", "GCN-Tg with heavy dropout regularisation",
     G("gnn", targets=["tg"], gnn={"style": "gcn", "hidden": 128, "layers": 3, "dropout": 0.4})),
    (104, "p3E19-gnn-tg-ensemble", "ensemble of GCN/GAT/MPNN Tg arms (NNLS)",
     G("assembly", arms=["exp086_p3E01-gcn-tg", "exp087_p3E02-gat-tg",
                          "exp088_p3E03-mpnn-tg"], targets=["tg"])),
    (105, "p3E20-gnn-final", "final GNN compound: best Tg GNN + blend arm",
     G("assembly", arms=["exp086_p3E01-gcn-tg", "exp092_p3E07-gnn-blend"])),
]

# =============================================================================
# Phase F — SMILES transformers from scratch (106-125)
# =============================================================================
EXPERIMENTS += [
    (106, "p3F01-mlm-smoke", "transformer pipeline smoke: tiny MLM 50k + GBM head",
     G(features=_ssl("mlm", n=50_000, dim=64, layers=2))),
    (107, "p3F02-mlm-500k", "tiny MLM 500k + GBM head",
     G(features=_ssl("mlm", n=500_000, dim=64, layers=2), long="LONG RUN (~1h GPU)")),
    (108, "p3F03-mlm-5m", "tiny MLM 5M + GBM head",
     G(features=_ssl("mlm", n=5_000_000, dim=64, layers=2), long="LONG RUN (overnight GPU)")),
    (109, "p3F04-bert-small-5m", "BERT-small (4L/256d) MLM on 5M + per-target GBM heads",
     G(features=_ssl("mlm", n=5_000_000, dim=256, layers=4), long="LONG RUN (overnight GPU)")),
    (110, "p3F05-bert-multitask", "BERT-small embeddings + multi-task MLP head",
     G("mlp", features=_ssl("mlm", n=5_000_000, dim=256, layers=4),
         mlp={"hidden": 256, "epochs": 200}, long="LONG RUN (overnight GPU)")),
    (111, "p3F06-bert-blend", "NNLS blend: BERT arm + GBM baseline",
     G("assembly", arms=["exp109_p3F04-bert-small-5m", "exp002_p3A02-clean-stack-v2"])),
    (112, "p3F07-mlm-pi1m-only", "MLM pretrained on PI1M only (polymer-specific)",
     G(features=_ssl("mlm", corpus="pi1m", n=None, dim=128, layers=4), long="LONG RUN (~3h GPU)")),
    (113, "p3F08-mlm-long-len", "MLM with max_len=384 (long SMILES)",
     G(features=_ssl("mlm", n=500_000, dim=128, layers=2, max_len=384), long="LONG RUN (~3h GPU)")),
    (114, "p3F09-ppmi-3m-192", "PPMI-SVD on 3M smile_r3 (permutation-robust analogue)",
     G(features=_ssl("ppmi", n=3_000_000, dim=192), long="LONG RUN (~1h)")),
    (115, "p3F10-mlm-mask25", "MLM with higher mask rate 0.25 (span-flavour)",
     G(features=_ssl("mlm", n=1_000_000, dim=128, layers=2), long="LONG RUN (~2h GPU)")),
    (116, "p3F11-mlm-combined", "MLM combined corpus (PI1M + 2M smile_r3)",
     G(features=_ssl("mlm", corpus="combined", n=2_000_000, dim=128, layers=2),
         long="LONG RUN (~4h GPU)")),
    (117, "p3F12-mlm-tta", "MLM(500k) arm with randomized-SMILES TTA K=5",
     G(features=_ssl("mlm", n=500_000, dim=64, layers=2), tta={"test_views": 5},
         long="LONG RUN (~1h GPU)")),
    (118, "p3F13-mlm-4epochs", "MLM(1M) longer pretraining epochs=4",
     G(features=_ssl("mlm", n=1_000_000, dim=128, layers=2, epochs=4), long="LONG RUN (~4h GPU)")),
    (119, "p3F14-ppmi-highdim", "PPMI-SVD 2M + high-dim fingerprints concat",
     G(features={"base": False, "high_dim": True,
                 "ssl": {"method": "ppmi", "corpus": "smile_r3", "n": 2_000_000, "dim": 128}})),
    (120, "p3F15-transformer-compound", "NNLS compound of best F arms + baseline",
     G("assembly", arms=["exp107_p3F02-mlm-500k", "exp109_p3F04-bert-small-5m",
                          "exp002_p3A02-clean-stack-v2"])),
    (121, "p3F16-seq-mlp", "char-SVD 512 + MLP head (sequence arm)",
     G(features={"char_only": True, "char_n_features": 4000, "char_svd": 512},
         model={"type": "mlp", "hidden": [256, 128], "epochs": 150})),
    (122, "p3F17-seq-lgbm", "char-SVD 512 + LGBM head",
     G(features={"char_only": True, "char_n_features": 4000, "char_svd": 512})),
    (123, "p3F18-seq-blend", "blend char-SVD arms (MLP + LGBM)",
     G("assembly", arms=["exp121_p3F16-seq-mlp", "exp122_p3F17-seq-lgbm"])),
    (124, "p3F19-mlm-tg-specialist", "MLM(500k) focused on Tg",
     G(targets=["tg"], features=_ssl("mlm", n=500_000, dim=128, layers=2), long="LONG RUN (~1h GPU)")),
    (125, "p3F20-transformer-final", "final F-phase compound",
     G("assembly", arms=["exp111_p3F06-bert-blend", "exp120_p3F15-transformer-compound"])),
]

# =============================================================================
# Phase G — global assembly & invariance (126-150)
# =============================================================================
EXPERIMENTS += [
    (126, "p3G01-global-assembly-v1", "assemble A/B/C/D phase arms (NNLS per target)",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp030_p3B20-tg-final-compound",
                          "exp060_p3C30-ssl-ladder-compound", "exp075_p3D15-physics-compound"])),
    (127, "p3G02-global-assembly-v2", "G01 + best transformer arm",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp030_p3B20-tg-final-compound",
                          "exp060_p3C30-ssl-ladder-compound", "exp075_p3D15-physics-compound",
                          "exp120_p3F15-transformer-compound"])),
    (128, "p3G03-tta-global", "TTA K=10 baseline arm for global blends",
     G("assembly", arms=["exp016_p3B06-tg-tta-k10", "exp030_p3B20-tg-final-compound",
                          "exp002_p3A02-clean-stack-v2"])),
    (129, "p3G04-tta-k20", "TTA K=20 baseline arm",
     G(tta={"train_aug": 20, "test_views": 20}, targets=["tg"])),
    (130, "p3G05-invariance-audit", "SMILES-permutation invariance audit (20 views)",
     G("invariance", features={"base": True, "svd_dim": 64}, invariance={"views": 20})),
    (131, "p3G06-no-override-clean", "pure-model vote baseline (no overrides)",
     G(model={"type": "vote", "models": [{"type": "lgbm"}, {"type": "xgb"}, {"type": "cat"}]})),
    (132, "p3G07-tanimoto-cv-final", "audit of baseline with Tanimoto binned OOF R2",
     G("audit", arms=["exp002_p3A02-clean-stack-v2", "exp001_p3A01-clean-stack-v1"])),
    (133, "p3G08-calibration-nnls", "NNLS blend as fold-local calibration proxy",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp003_p3A03-catboost",
                          "exp004_p3A04-xgboost"])),
    (134, "p3G09-calibration-mean", "mean blend (vs NNLS) for calibration comparison",
     G("assembly", blend="mean", arms=["exp002_p3A02-clean-stack-v2", "exp003_p3A03-catboost",
                                        "exp004_p3A04-xgboost"])),
    (135, "p3G10-seed-bagging", "LGBM with different seed (bagging diversity arm)",
     G(seed=77, model={"type": "lgbm", "num_leaves": 96, "learning_rate": 0.03})),
    (136, "p3G11-global-randsmiles", "randomized-SMILES augmentation K=3 global",
     G(tta={"train_aug": 3, "test_views": 3})),
    (137, "p3G12-canonical-vs-random", "TTA-only (no train aug) K=5 ablation",
     G(tta={"test_views": 5})),
    (138, "p3G13-global-assembly-v3", "best-so-far assembly incl. GNN arm",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp030_p3B20-tg-final-compound",
                          "exp060_p3C30-ssl-ladder-compound", "exp075_p3D15-physics-compound",
                          "exp100_p3E15-gnn-compound"])),
    (139, "p3G14-shap-analysis", "SHAP explainability on GBM arms (per-target top features)",
     G("shap", features={"base": True, "svd_dim": 64})),
    (140, "p3G15-per-target-audit", "per-target audit of main arms",
     G("audit", arms=["exp002_p3A02-clean-stack-v2", "exp075_p3D15-physics-compound",
                       "exp060_p3C30-ssl-ladder-compound"])),
    (141, "p3G16-global-v3-tta", "global v3 with TTA arms in blend",
     G("assembly", arms=["exp128_p3G03-tta-global", "exp137_p3G12-canonical-vs-random",
                          "exp002_p3A02-clean-stack-v2"])),
    (142, "p3G17-blend-search", "mean-blend search across diverse arms",
     G("assembly", blend="mean", arms=["exp001_p3A01-clean-stack-v1", "exp002_p3A02-clean-stack-v2",
                                        "exp007_p3A07-lgbm-deep", "exp009_p3A09-mordred"])),
    (143, "p3G18-oracle-candidate-1", "candidate 1: widest NNLS assembly",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp030_p3B20-tg-final-compound",
                          "exp060_p3C30-ssl-ladder-compound", "exp075_p3D15-physics-compound",
                          "exp100_p3E15-gnn-compound", "exp120_p3F15-transformer-compound"])),
    (144, "p3G19-oracle-candidate-2", "candidate 2: candidate-1 arms + TTA baseline",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp016_p3B06-tg-tta-k10",
                          "exp060_p3C30-ssl-ladder-compound", "exp075_p3D15-physics-compound",
                          "exp100_p3E15-gnn-compound"])),
    (145, "p3G20-per-target-specialist", "specialist assembly: physics + Tg compound arms",
     G("assembly", arms=["exp075_p3D15-physics-compound", "exp030_p3B20-tg-final-compound",
                          "exp085_p3D25-small-targets-vote"])),
    (146, "p3G21-low-sim-final-audit", "low-similarity audit of candidate-1",
     G("audit", arms=["exp143_p3G18-oracle-candidate-1"])),
    (147, "p3G22-global-v4", "v4: adds pseudo-labeling arm",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp060_p3C30-ssl-ladder-compound",
                          "exp055_p3C25-pseudo-tg-pi1m", "exp075_p3D15-physics-compound"])),
    (148, "p3G23-global-v5-max", "v5 max: everything available",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp030_p3B20-tg-final-compound",
                          "exp060_p3C30-ssl-ladder-compound", "exp075_p3D15-physics-compound",
                          "exp100_p3E15-gnn-compound", "exp120_p3F15-transformer-compound",
                          "exp147_p3G22-global-v4"])),
    (149, "p3G24-explainability-pkg", "SHAP package on Mordred-augmented model",
     G("shap", features={"base": True, "mordred": True})),
    (150, "p3G25-final-prep-audit", "audit of the top candidates before Phase H",
     G("audit", arms=["exp143_p3G18-oracle-candidate-1", "exp144_p3G19-oracle-candidate-2",
                       "exp148_p3G23-global-v5-max"])),
]

# =============================================================================
# Phase H — target-specific deep dives (151-175)
# =============================================================================
EXPERIMENTS += [
    (151, "p3H01-tg-scaffold-cv-deep", "Tg: Mordred+base with scaffold CV",
     G(targets=["tg"], features={"base": True, "mordred": True},
         cv={"type": "scaffold", "n_splits": 5})),
    (152, "p3H02-tg-similarity-knn", "Tg: kNN k=3 (similarity-weighted analogue)",
     G(targets=["tg"], model={"type": "knn", "k": 3})),
    (153, "p3H03-tg-outlier-removal", "Tg: 3-sigma outlier removal before fit",
     G(targets=["tg"], model_opts={"outlier_sigma": 3.0})),
    (154, "p3H04-tg-pi1m-pseudo", "Tg: PI1M pseudo-label augmentation (soft labels)",
     G("pseudo", targets=["tg"], pseudo={"corpus": "pi1m", "n": 50000, "top_k": 5000})),
    (155, "p3H05-tg-ridge-highdim", "Tg: very high-dim fingerprints + Ridge",
     G(targets=["tg"], features={"base": False, "high_dim": True},
         model={"type": "ridge", "alpha": 8.0})),
    (156, "p3H06-ei-all-features", "ei: every feature block stacked",
     G(targets=["ei"], features={"base": True, "mordred": True, "high_dim": True},
         model={"type": "vote", "models": [{"type": "lgbm"}, {"type": "cat"}]})),
    (157, "p3H07-ei-gpr-wide", "ei: GPR with wider length scale",
     G(targets=["ei"], model={"type": "gpr", "length_scale": 4.0, "noise": 0.05})),
    (158, "p3H08-eps-nc-mordred", "eps+nc: Mordred + base + partner injection",
     G("physics", targets=["eps", "nc"], partner={"eps": ["nc"], "nc": ["eps"]},
         features={"base": True, "mordred": True})),
    (159, "p3H09-nc-molar-refraction", "nc: Mordred-only route",
     G(targets=["nc"], features={"base": False, "mordred": True})),
    (160, "p3H10-eea-deep-catboost", "eea: deeper CatBoost (2600 iters)",
     G(targets=["eea"], model={"type": "cat", "iterations": 2600, "learning_rate": 0.03})),
    (161, "p3H11-egc-mordred", "egc: Mordred electronic descriptors",
     G(targets=["egc"], features={"base": True, "mordred": True})),
    (162, "p3H12-eps-ionic-hard", "eps: ionic coordinate + partner features",
     G("physics", targets=["eps", "nc"], partner={"eps": ["nc"]}, coordinate={"eps": "ionic"},
         features={"base": True, "mordred": True})),
    (163, "p3H13-tg-family-audit", "audit: Tg error structure across arms",
     G("audit", arms=["exp011_p3B01-tg-group-contrib", "exp015_p3B05-tg-randsmiles-aug",
                       "exp028_p3B18-tg-mixup"])),
    (164, "p3H14-all-targets-mordred", "all targets: full Mordred set + LGBM",
     G(features={"base": False, "mordred": True})),
    (165, "p3H15-catboost-all", "CatBoost-only all targets, tuned-ish params",
     G(model={"type": "cat", "iterations": 1800, "learning_rate": 0.045, "depth": 7})),
    (166, "p3H16-lgbm-dart", "LGBM DART boosting all targets",
     G(model={"type": "lgbm", "boosting": "dart", "drop_rate": 0.12, "n_estimators": 800})),
    (167, "p3H17-xgb-hist", "XGBoost histogram all targets",
     G(model={"type": "xgb", "tree_method": "hist", "n_estimators": 1400})),
    (168, "p3H18-rf-extra-trees", "ExtraTrees all targets",
     G(model={"type": "et", "n_estimators": 800})),
    (169, "p3H19-blended-stack", "wide NNLS stack of diverse model families",
     G("assembly", arms=["exp003_p3A03-catboost", "exp004_p3A04-xgboost",
                          "exp006_p3A06-extratrees", "exp005_p3A05-ridge",
                          "exp166_p3H16-lgbm-dart"])),
    (170, "p3H20-van-krevelen", "Tg: group-count-heavy features (Mordred + char)",
     G(targets=["tg"], features={"mordred": True, "char_only": True})),
    (171, "p3H21-label-spreading", "Tg: pseudo-labels from smile_r3 sample",
     G("pseudo", targets=["tg"], pseudo={"corpus": "smile_r3", "n": 30000, "top_k": 3000})),
    (172, "p3H22-mean-teacher-pseudo", "Tg: bigger-corpus self-training",
     G("pseudo", targets=["tg"], pseudo={"corpus": "pi1m", "n": 40000, "top_k": 6000})),
    (173, "p3H23-self-training-egc", "egc: self-training on PI1M",
     G("pseudo", targets=["egc"], pseudo={"corpus": "pi1m", "n": 30000, "top_k": 3000})),
    (174, "p3H24-tg-latent-mixup", "Tg: mixup + noise combined regularisation",
     G(targets=["tg"], model_opts={"mixup": 0.3, "noise": 0.02})),
    (175, "p3H25-h-phase-compound", "NNLS compound of best H arms",
     G("assembly", arms=["exp164_p3H14-all-targets-mordred", "exp165_p3H15-catboost-all",
                          "exp169_p3H19-blended-stack"])),
]

# =============================================================================
# Phase I — hyperparameter systematic sweep (176-200)
# =============================================================================
EXPERIMENTS += [
    (176, "p3I01-randsearch-lgbm-tg", "random-search LGBM for Tg (30 trials, grouped CV)",
     G("optuna", targets=["tg"], search={"model_kind": "lgbm", "trials": 30})),
    (177, "p3I02-randsearch-lgbm-ei", "random-search LGBM for ei",
     G("optuna", targets=["ei"], search={"model_kind": "lgbm", "trials": 25})),
    (178, "p3I03-randsearch-lgbm-eps", "random-search LGBM for eps",
     G("optuna", targets=["eps"], search={"model_kind": "lgbm", "trials": 25})),
    (179, "p3I04-randsearch-xgb-tg", "random-search XGB for Tg",
     G("optuna", targets=["tg"], search={"model_kind": "xgb", "trials": 30})),
    (180, "p3I05-randsearch-cat-tg", "random-search CatBoost for Tg",
     G("optuna", targets=["tg"], search={"model_kind": "cat", "trials": 25})),
    (181, "p3I06-feature-select-tg", "Tg: slim feature set (colsample 0.5, fewer leaves)",
     G(targets=["tg"], model={"type": "lgbm", "colsample_bytree": 0.5, "num_leaves": 31})),
    (182, "p3I07-feature-select-ei", "ei: slim feature set",
     G(targets=["ei"], model={"type": "lgbm", "colsample_bytree": 0.5, "num_leaves": 31})),
    (183, "p3I08-feature-select-eps", "eps: slim feature set",
     G(targets=["eps"], model={"type": "lgbm", "colsample_bytree": 0.5, "num_leaves": 31})),
    (184, "p3I09-n-folds-3", "Tg: 3-fold CV variance check",
     G(targets=["tg"], cv={"type": "grouped", "n_splits": 3})),
    (185, "p3I10-n-folds-10", "Tg: 10-fold CV variance check",
     G(targets=["tg"], cv={"type": "grouped", "n_splits": 10})),
    (186, "p3I11-gbm-cat-final", "LGBM+Cat mean ensemble with tuned params",
     G(model={"type": "vote", "models": [
         {"type": "lgbm", "num_leaves": 96, "learning_rate": 0.03},
         {"type": "cat", "iterations": 1500, "learning_rate": 0.04}]})),
    (187, "p3I12-stochastic-blend", "mean-blend of tuned GBM arms",
     G("assembly", blend="mean", arms=["exp003_p3A03-catboost", "exp004_p3A04-xgboost",
                                        "exp007_p3A07-lgbm-deep"])),
    (188, "p3I13-feature-interactions", "Tg: Mordred + SVD-256 (interaction-rich block)",
     G(targets=["tg"], features={"base": True, "mordred": True, "svd_dim": 256})),
    (189, "p3I14-noise-regularization", "Tg: feature-noise regularization (sigma 0.01)",
     G(targets=["tg"], model_opts={"noise": 0.01})),
    (190, "p3I15-target-mordred-only", "all targets: Mordred + char only (transform check)",
     G(features={"mordred": True, "char_only": True})),
    (191, "p3I16-seed-2028", "seed sensitivity check (seed 2028)",
     G(seed=2028)),
    (192, "p3I17-lgbm-tg-deeper", "Tg: 4000 trees, lr 0.008",
     G(targets=["tg"], model={"type": "lgbm", "n_estimators": 4000, "learning_rate": 0.008})),
    (193, "p3I18-catboost-tg-more-trees", "Tg: CatBoost 4000 iters",
     G(targets=["tg"], model={"type": "cat", "iterations": 4000, "learning_rate": 0.03})),
    (194, "p3I19-randsearch-lgbm-egc", "random-search LGBM for egc",
     G("optuna", targets=["egc"], search={"model_kind": "lgbm", "trials": 25})),
    (195, "p3I20-tuned-compound", "compound of tuned per-target GBMs (NNLS)",
     G("assembly", arms=["exp003_p3A03-catboost", "exp004_p3A04-xgboost",
                          "exp007_p3A07-lgbm-deep", "exp166_p3H16-lgbm-dart"])),
    (196, "p3I21-diversity-ensemble", "diverse-family vote (GBM+ET+Ridge+kNN)",
     G(model={"type": "vote", "models": [
         {"type": "lgbm"}, {"type": "et", "n_estimators": 600},
         {"type": "ridge", "alpha": 14.0}, {"type": "knn", "k": 7}]})),
    (197, "p3I22-blend-calibration", "NNLS blend of seed-bagging arms",
     G("assembly", arms=["exp135_p3G10-seed-bagging", "exp002_p3A02-clean-stack-v2"])),
    (198, "p3I23-final-audit", "final validation panel audit of best arms",
     G("audit", arms=["exp143_p3G18-oracle-candidate-1", "exp148_p3G23-global-v5-max"])),
    (199, "p3I24-mordred-nc-eps-vote", "eps/nc vote with Mordred block",
     G(targets=["eps", "nc"], model={"type": "vote", "models": [
         {"type": "lgbm"}, {"type": "cat"}]}, features={"base": True, "mordred": True})),
    (200, "p3I25-decision-point", "mid-suite audit: where is the remaining gap?",
     G("audit", arms=["exp143_p3G18-oracle-candidate-1", "exp175_p3H25-h-phase-compound",
                       "exp148_p3G23-global-v5-max"])),
]

# =============================================================================
# Phase J — extended SSL at maximum scale (201-210) — LONG RUNS
# =============================================================================
EXPERIMENTS += [
    (201, "p3J01-svd-full-512", "FULL 5.97M SVD at 512 dims",
     G(features=_ssl("svd", n=None, dim=512), long="LONG RUN (~3-5h)")),
    (202, "p3J02-w2v-full-combined", "PPMI-SVD on combined 6.97M corpus",
     G(features=_ssl("ppmi", corpus="combined", n=None, dim=256), long="LONG RUN (~4h)")),
    (203, "p3J03-mlm-bert-base-5m", "BERT-base-scale (8L/512d) MLM on 5M",
     G(features=_ssl("mlm", n=5_000_000, dim=512, layers=8, heads=8),
         long="LONG RUN (~24h GPU)")),
    (204, "p3J04-pi1m-pseudo-tg-large", "pseudo-label Tg on 100k PI1M, keep 20k",
     G("pseudo", targets=["tg"], pseudo={"corpus": "pi1m", "n": 100000, "top_k": 20000})),
    (205, "p3J05-pseudo-all-targets", "pseudo-label all targets on PI1M (2k each)",
     G("pseudo", pseudo={"corpus": "pi1m", "n": 30000, "top_k": 2000})),
    (206, "p3J06-ssl-finetune-multitask", "MLM(2M) embeddings + multitask MLP head",
     G("mlp", features=_ssl("mlm", n=2_000_000, dim=256, layers=4),
         mlp={"hidden": 256, "epochs": 250}, long="LONG RUN (~8h GPU)")),
    (207, "p3J07-structural-curriculum", "Tg: scaffold-CV curriculum-style regularised LGBM",
     G(targets=["tg"], cv={"type": "scaffold", "n_splits": 5},
         model={"type": "lgbm", "num_leaves": 64, "learning_rate": 0.03})),
    (208, "p3J08-scaffold-correction", "Tg: scaffold-CV LGBM with Mordred",
     G(targets=["tg"], cv={"type": "scaffold", "n_splits": 5},
         features={"base": True, "mordred": True})),
    (209, "p3J09-large-ensemble", "large NNLS ensemble of best diverse arms",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp165_p3H15-catboost-all",
                          "exp166_p3H16-lgbm-dart", "exp167_p3H17-xgb-hist",
                          "exp168_p3H18-rf-extra-trees"])),
    (210, "p3J10-final-ssl-compound", "compound best J SSL arms",
     G("assembly", arms=["exp034_p3C04-r3-svd-full", "exp202_p3J02-w2v-full-combined",
                          "exp109_p3F04-bert-small-5m"])),
]

# =============================================================================
# Phase K — final push & packaging (211-250)
# =============================================================================
EXPERIMENTS += [
    (211, "p3K01-final-assembly", "final assembly: best arms, NNLS, TTA inputs",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp016_p3B06-tg-tta-k10",
                          "exp060_p3C30-ssl-ladder-compound", "exp075_p3D15-physics-compound",
                          "exp100_p3E15-gnn-compound", "exp120_p3F15-transformer-compound",
                          "exp175_p3H25-h-phase-compound"])),
    (212, "p3K02-final-audit", "final audit of the top-3 candidates",
     G("audit", arms=["exp143_p3G18-oracle-candidate-1", "exp148_p3G23-global-v5-max",
                       "exp211_p3K01-final-assembly"])),
    (213, "p3K03-submission-prep-audit", "submission-prep audit: verify candidate CSV shape",
     G("audit", arms=["exp211_p3K01-final-assembly"])),
    (214, "p3K04-parity-check-audit", "determinism/parity audit of candidate",
     G("audit", arms=["exp211_p3K01-final-assembly"])),
    (215, "p3K05-standalone-validation", "standalone validation audit of candidate",
     G("audit", arms=["exp211_p3K01-final-assembly"])),
    (216, "p3K06-shap-final", "final SHAP analysis (explainability theme)",
     G("shap", features={"base": True, "mordred": True})),
    (217, "p3K07-invariance-report", "final invariance report (robustness theme)",
     G("invariance", features={"base": True, "svd_dim": 64}, invariance={"views": 20})),
    (218, "p3K08-final-report-data", "audit: all data needed for FINAL_REPORT",
     G("audit", arms=["exp211_p3K01-final-assembly", "exp132_p3G07-tanimoto-cv-final"])),
    (219, "p3K09-candidate-a", "final candidate A (wide assembly)",
     G("assembly", arms=["exp002_p3A02-clean-stack-v2", "exp030_p3B20-tg-final-compound",
                          "exp060_p3C30-ssl-ladder-compound", "exp075_p3D15-physics-compound",
                          "exp100_p3E15-gnn-compound"])),
    (220, "p3K10-candidate-b", "final candidate B (TTA-heavy alternative)",
     G("assembly", arms=["exp016_p3B06-tg-tta-k10", "exp060_p3C30-ssl-ladder-compound",
                          "exp075_p3D15-physics-compound", "exp137_p3G12-canonical-vs-random"])),
]

# Reserve slots 221-250: recomputations of the strongest families with alternate
# seeds/params, to be re-pointed by analysis after the first 220 land.
for i in range(221, 251):
    vs = 3000 + i
    if i % 5 == 1:
        EXPERIMENTS.append((i, f"p3K{11 + (i - 221)}-reserve-vote",
                            f"reserve: vote ensemble, seed {vs}",
                            G(seed=vs, model={"type": "vote", "models": [
                                {"type": "lgbm"}, {"type": "xgb"}, {"type": "cat"}]})))
    elif i % 5 == 2:
        EXPERIMENTS.append((i, f"p3K{11 + (i - 221)}-reserve-svd",
                            f"reserve: SVD-128 SSL(500k), seed {vs}",
                            G(seed=vs, features=_ssl("svd", n=500_000, dim=128))))
    elif i % 5 == 3:
        EXPERIMENTS.append((i, f"p3K{11 + (i - 221)}-reserve-tg-tta",
                            f"reserve: Tg TTA K=5, seed {vs}",
                            G(seed=vs, targets=["tg"], tta={"train_aug": 5, "test_views": 5})))
    elif i % 5 == 4:
        EXPERIMENTS.append((i, f"p3K{11 + (i - 221)}-reserve-mordred",
                            f"reserve: Mordred+base, seed {vs}",
                            G(seed=vs, features={"base": True, "mordred": True})))
    else:
        EXPERIMENTS.append((i, f"p3K{11 + (i - 221)}-reserve-audit",
                            "reserve: audit of the current top assembly",
                            G("audit", arms=["exp143_p3G18-oracle-candidate-1",
                                              "exp211_p3K01-final-assembly"])))



# --- long-arm propagation: any experiment whose arms reference a LONG-RUN arm
# is itself treated as LONG so the default ./run.sh stays clean and consistent.
LONG_IDX = {idx for idx, _n, _d, c in EXPERIMENTS if "long" in str(c).lower()}
for _i, (_idx, _nn, _dd, _cc) in enumerate(EXPERIMENTS):
    _arms = _cc.get("arms")
    if not _arms:
        continue
    _hit = []
    for _a in _arms:
        _m = re.match(r"exp(\d{3})", str(_a).split("/")[-1])
        if _m and int(_m.group(1)) in LONG_IDX:
            _hit.append(str(_a).split("/")[-1])
    if _hit and "long" not in str(_cc).lower():
        _cc["long"] = "depends on long arms: " + ",".join(_hit)

def main() -> None:
    OUT.mkdir(exist_ok=True)
    seen = set()
    for idx, name, desc, cfg in EXPERIMENTS:
        assert idx not in seen, f"duplicate index {idx}"
        seen.add(idx)
        long_note = cfg.pop("long", "standard runtime (<30 min)")
        cfg = dict(sorted(cfg.items()))
        path = OUT / f"exp{idx:03d}_{name}.py"
        path.write_text(TEMPLATE.format(idx=idx, name=name, desc=desc,
                                        cfg=repr(cfg), long_note=long_note))
    print(f"wrote {len(seen)} experiment scripts to {OUT}")


# =============================================================================
# Phase A — strong baselines (001-010)
# =============================================================================
EXPERIMENTS += [
    (1, "p3A01-clean-stack-v1", "in-run 5-model vote ensemble (LGBM+XGB+Cat+Ridge+ET), base features",
     G(model={"type": "vote", "models": [
         {"type": "lgbm"}, {"type": "xgb"}, {"type": "cat"},
         {"type": "ridge", "alpha": 18.0}, {"type": "et", "n_estimators": 400}]})),
    (2, "p3A02-clean-stack-v2", "vote ensemble + SVD-64 on counts+char",
     G(model={"type": "vote", "models": [
         {"type": "lgbm"}, {"type": "xgb"}, {"type": "cat"},
         {"type": "ridge", "alpha": 18.0}, {"type": "et", "n_estimators": 400}]},
         features={"base": True, "svd_dim": 64})),
    (3, "p3A03-catboost", "CatBoost only, base features", G(model={"type": "cat"})),
    (4, "p3A04-xgboost", "XGBoost only, base features", G(model={"type": "xgb"})),
    (5, "p3A05-ridge", "Ridge only, base features", G(model={"type": "ridge", "alpha": 18.0})),
    (6, "p3A06-extratrees", "ExtraTrees only, base features",
     G(model={"type": "et", "n_estimators": 600})),
    (7, "p3A07-lgbm-deep", "LGBM with deeper trees / lower lr",
     G(model={"type": "lgbm", "num_leaves": 127, "learning_rate": 0.02, "n_estimators": 1600})),
    (8, "p3A08-svd128", "LGBM with SVD-128 text block",
     G(features={"base": True, "svd_dim": 128})),
    (9, "p3A09-mordred", "LGBM with full Mordred descriptor block added",
     G(features={"base": True, "mordred": True})),
    (10, "p3A10-vote-mordred", "vote ensemble on base+Mordred features",
     G(model={"type": "vote", "models": [{"type": "lgbm"}, {"type": "cat"}, {"type": "et"}]},
         features={"base": True, "mordred": True})),
]

# =============================================================================
# Phase B — Tg specialists (011-030)
# =============================================================================
_B_TG = {"targets": ["tg"]}
EXPERIMENTS += [
    (11, "p3B01-tg-group-contrib", "Tg: base + Mordred (explicit group-count features)",
     G(features={"base": True, "mordred": True}, **_B_TG)),
    (12, "p3B02-tg-backbone-sidechain", "Tg: Mordred + SVD-128 representation",
     G(features={"base": True, "mordred": True, "svd_dim": 128}, **_B_TG)),
    (13, "p3B03-tg-dimer", "Tg: dimer repeat-unit text expansion features",
     G(features={"base": True, "rep_text": 2}, **_B_TG)),
    (14, "p3B04-tg-trimer", "Tg: trimer repeat-unit text expansion features",
     G(features={"base": True, "rep_text": 3}, **_B_TG)),
    (15, "p3B05-tg-randsmiles-aug", "Tg: randomized-SMILES augmentation K=5 + TTA",
     G(tta={"train_aug": 5, "test_views": 5}, **_B_TG)),
    (16, "p3B06-tg-tta-k10", "Tg: randomized-SMILES augmentation K=10 + TTA",
     G(tta={"train_aug": 10, "test_views": 10}, **_B_TG)),
    (17, "p3B07-tg-scaffold-cv", "Tg: scaffold-stratified CV folds (honest novel-structure estimate)",
     G(cv={"type": "scaffold", "n_splits": 5}, **_B_TG)),
    (18, "p3B08-tg-knn", "Tg: Tanimoto kNN regression (neighbor-label signal)",
     G(model={"type": "knn", "k": 5}, **_B_TG)),
    (19, "p3B09-tg-multitask-egc", "Tg+egc multi-task MLP with masked loss",
     G("mlp", targets=["tg", "egc"], mlp={"hidden": 256, "epochs": 200})),
    (20, "p3B20-tg-catboost-text", "Tg: CatBoost on char-ngram SVD text features",
     G(model={"type": "cat", "iterations": 1600}, features={"char_only": True}, **_B_TG)),
    (21, "p3B11-tg-flory-fox", "Tg: copolymer-composition stats + base features",
     G(features={"base": True, "char_only": False, "svd_dim": 128}, **_B_TG)),
    (22, "p3B12-tg-3d-descriptors", "Tg: 3D-shape proxy via Mordred + base",
     G(features={"base": True, "mordred": True}, model={"type": "cat"}, **_B_TG)),
    (23, "p3B13-tg-mordred-only", "Tg: Mordred-only feature set with LGBM",
     G(features={"base": False, "mordred": True}, **_B_TG)),
    (24, "p3B14-tg-wl-kernel", "Tg: higher-order Morgan r3/r4 counts + Ridge (WL-flavour)",
     G(features={"base": False, "high_dim": True}, model={"type": "ridge", "alpha": 8.0}, **_B_TG)),
    (25, "p3B15-tg-ridge-char", "Tg: high-dim char n-grams + Ridge",
     G(features={"char_only": True, "char_ngram": [1, 4], "char_n_features": 2000},
         model={"type": "ridge", "alpha": 10.0}, **_B_TG)),
    (26, "p3B16-tg-ensemble-B", "Tg: NNLS assembly of B-phase Tg arms",
     G("assembly", arms=["exp013_p3B03-tg-dimer", "exp015_p3B05-tg-randsmiles-aug",
                          "exp019_p3B09-tg-multitask-egc", "exp024_p3B14-tg-wl-kernel"],
         targets=["tg"])),
    (27, "p3B17-tg-lowsim-audit", "Tg: low-similarity audit of the K=5 TTA arm",
     G("audit", arms=["exp015_p3B05-tg-randsmiles-aug"])),
    (28, "p3B18-tg-mixup", "Tg: feature-space mixup augmentation",
     G(model_opts={"mixup": 0.5}, **_B_TG)),
    (29, "p3B19-tg-quantile", "Tg: quantile-loss LGBM (median prediction)",
     G(model={"type": "lgbm", "objective": "quantile", "alpha": 0.5}, **_B_TG)),
    (30, "p3B20-tg-final-compound", "Tg: NNLS compound of best B arms",
     G("assembly", arms=["exp011_p3B01-tg-group-contrib", "exp015_p3B05-tg-randsmiles-aug",
                          "exp016_p3B06-tg-tta-k10", "exp028_p3B18-tg-mixup"],
         targets=["tg"])),
]




# =============================================================================
# Phase L — smile_r3 / PI1M ATOM-LEVEL SSL (251-262) — PRIORITY (user request:
# start with the new smile_r3.csv experiments; chemically-aware atom tokenization)
# =============================================================================
EXPERIMENTS += [
    (251, "p3L01-r3-atom-ppmi-100k", "atom-level PPMI-SVD on 100k smile_r3 (chemically-aware tokens)",
     G(features=_ssl("atom_ppmi", n=100_000))),
    (252, "p3L02-r3-atom-ppmi-1m", "atom-level PPMI-SVD on 1M smile_r3",
     G(features=_ssl("atom_ppmi", n=1_000_000))),
    (253, "p3L03-r3-atom-ppmi-2m", "atom-level PPMI-SVD on 2M smile_r3",
     G(features=_ssl("atom_ppmi", n=2_000_000))),
    (254, "p3L04-r3-atom-mlm-100k", "atom-token MLM (2L/128d) pretrained on 100k smile_r3",
     G(features=_ssl("atom_mlm", n=100_000, dim=128, layers=2))),
    (255, "p3L05-r3-atom-mlm-500k", "atom-token MLM (2L/128d) on 500k smile_r3",
     G(features=_ssl("atom_mlm", n=500_000, dim=128, layers=2), long="LONG RUN (~1h GPU)")),
    (256, "p3L06-r3-atom-mlm-1m", "atom-token MLM (2L/128d) on 1M smile_r3",
     G(features=_ssl("atom_mlm", n=1_000_000, dim=128, layers=2), long="LONG RUN (~2h GPU)")),
    (257, "p3L07-r3-combined-atom-ppmi-2m", "atom-level PPMI-SVD on combined PI1M+smile_r3 2M",
     G(features=_ssl("atom_ppmi", corpus="combined", n=2_000_000))),
    (258, "p3L08-r3-atom-mlm-tg-specialist", "atom-token MLM(500k) focused on Tg",
     G(targets=["tg"], features=_ssl("atom_mlm", n=500_000, dim=128, layers=2),
         long="LONG RUN (~1h GPU)")),
    (259, "p3L09-r3-atom-ppmi-window8-1m", "atom PPMI-SVD window=8 on 1M smile_r3 (long-range context)",
     G(features=_ssl("atom_ppmi", n=1_000_000, window=8))),
    (260, "p3L10-r3-char-ppmi-5m", "FIXED char PPMI-SVD at full 5M scale (the exp039 bugfix rerun)",
     G(features=_ssl("ppmi", n=5_000_000, dim=256), long="LONG RUN (~2-3h)")),
    (261, "p3L11-r3-atom-ppmi-5m", "atom-level PPMI-SVD at 5M smile_r3",
     G(features=_ssl("atom_ppmi", n=5_000_000, dim=256), long="LONG RUN (~2-3h)")),
    (262, "p3L12-r3-atom-ssl-compound", "NNLS compound of atom-SSL arms",
     G("assembly", arms=["exp251_p3L01-r3-atom-ppmi-100k", "exp252_p3L02-r3-atom-ppmi-1m",
                          "exp254_p3L04-r3-atom-mlm-100k"])),
]

# =============================================================================
# Phase M — advanced physics / feature blocks (263-270)
# =============================================================================
EXPERIMENTS += [
    (263, "p3M01-ei-mulliken", "ei via Mulliken electronegativity chi=(ei+eea)/2 with eea partner",
     G("physics", partner={"ei": ["eea"]}, coordinate={"ei": {"type": "mulliken", "partner": "eea"}},
         targets=["ei", "eea"])),
    (264, "p3M02-eea-mulliken", "eea via Mulliken chi with ei partner",
     G("physics", partner={"eea": ["ei"]}, coordinate={"eea": {"type": "mulliken", "partner": "ei"}},
         targets=["eea", "ei"])),
    (265, "p3M03-ei-gap-residual", "ei = eea + residual (gap identity, ask sec 1.2)",
     G("physics", partner={"ei": ["eea"]}, coordinate={"ei": {"type": "gapres", "partner": "eea"}},
         targets=["ei", "eea"])),
    (266, "p3M04-egb-gap-residual", "egb = egc + residual (bulk gap affine identity)",
     G("physics", partner={"egb": ["egc"]}, coordinate={"egb": {"type": "gapres", "partner": "egc"}},
         targets=["egb", "egc"])),
    (267, "p3M05-eps-nc-polar-ionic", "eps/nc ionic coordinate + polar-moiety + Gasteiger blocks",
     G("physics", targets=["eps", "nc"], partner={"eps": ["nc"], "nc": ["eps"]},
         coordinate={"eps": "ionic"}, features={"base": True, "polar": True, "gasteiger": True})),
    (268, "p3M06-eps-nc-3d-conformers", "eps/nc with 3D-conformer (ETKDG+UFF) + polar + Gasteiger",
     G(targets=["eps", "nc"], features={"base": True, "polar": True, "gasteiger": True,
         "conformer3d": True}, long="LONG RUN (3D conformers ~20-40 min)")),
    (269, "p3M07-tg-3d-rigidity", "Tg with 3D conformer + polar + base (rigidity/free-volume)",
     G(targets=["tg"], features={"base": True, "polar": True, "conformer3d": True},
         long="LONG RUN (3D conformers ~20-40 min)")),
    (270, "p3M08-ei-electronic-blocks", "ei with polar + Gasteiger + Mordred electronic block",
     G(targets=["ei"], features={"base": True, "polar": True, "gasteiger": True, "mordred": True},
         model={"type": "lgbm", "num_leaves": 31})),
]

# =============================================================================
# Phase N — advanced modeling (271-276)
# =============================================================================
EXPERIMENTS += [
    (271, "p3N01-mixture-electronic", "grouped multi-output trees: electronic group (ei,eea,egc,egb)",
     G("mixture", mixture={"groups": {"electronic": ["ei", "eea", "egc", "egb"]}},
         targets=["ei", "eea", "egc", "egb"])),
    (272, "p3N02-mixture-optical-thermal", "grouped multi-output trees: optical (eps,nc) + thermal (tg)",
     G("mixture", mixture={"groups": {"optical": ["eps", "nc"], "thermal": ["tg"]}},
         targets=["eps", "nc", "tg"])),
    (273, "p3N03-matfac-lowrank", "prediction-matrix factorization: low-rank completion as features",
     G("matfac")),
    (274, "p3N04-recalib-affine", "per-target affine recalibration on OOF (R2 >= raw guaranteed)",
     G("recalib")),
    (275, "p3N05-shiftweight-r3", "covariate-shift importance weighting vs smile_r3 sample",
     G("shiftweight", shiftweight={"corpus": "smile_r3", "n": 60000})),
    (276, "p3N06-uncertainty-spread", "quantile-spread (q90-q10) as uncertainty feature for Tg",
     G("uncertainty", targets=["tg"])),
]

# =============================================================================
# Phase O — curation & validation discipline (277-282)
# =============================================================================
EXPERIMENTS += [
    (277, "p3O01-tg-median-smooth", "Tg: median-smooth replicate labels by canonical structure",
     G(curation={"tg_median": True}, targets=["tg"])),
    (278, "p3O02-neardup-drop", "drop train rows with >0.99 Tanimoto to any test row",
     G(curation={"drop_near_dup": 0.99})),
    (279, "p3O03-overlap-weighted", "down-weight the 457 train/test shared structures (w=0.3)",
     G(curation={"overlap_weight": 0.3})),
    (280, "p3O04-fold-kmeans-tg", "Tg with structural-similarity (k-means) folds",
     G(targets=["tg"], cv={"type": "kmeans", "n_splits": 5})),
    (281, "p3O05-overlap-shift-audit", "audit candidate-1: 457-overlap + shift-matched reweighted R2",
     G("audit", arms=["exp143_p3G18-oracle-candidate-1"])),
    (282, "p3O06-fold-design-compare", "fold-design comparison: grouped vs scaffold vs kmeans (Tg)",
     G("audit", arms=["exp017_p3B07-tg-scaffold-cv", "exp280_p3O04-fold-kmeans-tg",
                      "exp002_p3A02-clean-stack-v2"])),
]

if __name__ == "__main__":
    main()
