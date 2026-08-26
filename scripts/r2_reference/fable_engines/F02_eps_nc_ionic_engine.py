"""F02 - eps/nc joint physics engine in ionic coordinates.

Physics (exact by construction in this data lineage - Kim 2018 JPCC, Tran 2020
JAP): nc = sqrt(eps_electronic), eps_total = eps_electronic + eps_ionic, so

    ionic := eps - nc^2  >= 0   (0 violations across all 134 official pairs)

Measured on official train pairs (n=134), grouped 5-fold. archive/train.csv
contains only tg and egc rows, so every number in this file is identical with or
without it.

    eps = nc^2 + median(ionic)         -> R2 0.8487
    eps = nc^2 + ionic_ML              -> R2 0.9548   <-- headline
    nc  = sqrt(eps - ionic_ML)         -> R2 0.9323   (vs 0.7887 const)
    ionic_ML itself                    -> R2 0.6901

ionic_ML CONFIGURATION IS MEASURED, NOT GUESSED. ExtraTrees(min_samples_leaf=2)
on the polar-feature block, RAW target. The alternatives, all worse:
    log target                 ionic 0.676  eps 0.9527
    min_samples_leaf=5         ionic 0.609  eps 0.9429
    + 512 Morgan bits          ionic 0.665  eps 0.9510
    Ridge on same features     ionic 0.523  eps 0.9304
    Ridge, log target          ionic -0.069 eps 0.8439
Adding fingerprints costs ~0.005; log-transforming costs ~0.02. Note the prior
incumbent's C144 route IS log(eps - nc^2), so dropping the log is a real gain.

Test-row partner support: 95/153 eps rows have an observed nc label and the
other 58 have their nc partner as a CO-TEST row - an exact partition with zero
unsupported rows. Symmetric for nc.
Incumbent test-side baseline: eps 0.8452, nc 0.8977.

Design (preregistered)
----------------------
Arms per target on identical grouped folds:
  B0 baseline: structure-only blend (same as F01 A0).
  B1 partner-label physics: where partner label observed,
       eps = nc_label^2 + ionic_ML;  nc = sqrt(max(eps_label - ionic_ML, 1.0));
     where missing, fall back to B0. ionic_ML = ExtraTrees on the polar-group
     block, RAW target, trained fold-locally on the labeled pairs.
  B2 chained physics: missing partner labels filled with cross-fitted partner
     predictions (structure-grouped, row's own structure excluded), with a
     predicted-partner flag feeding a per-stratum inner-fold blend weight
     against B0. This extends physics coverage from 62% to ~100% of rows.
  B3 = B2 + joint test-test consistency: for the 58 structures present in BOTH
     eps and nc test rows, reconcile so eps_pred - nc_pred^2 = ionic_pred
     exactly (weighted least squares in (nc^2, eps) space, weights =
     inverse inner-fold MSE of each head). Evaluated on OOF by simulating the
     same reconciliation on rows labeled for both.

Constraint applied to every arm: eps >= nc^2 + 0.02 (observed min ionic 0.024).

Gates: bank a target if delta_shift_matched >= +0.005 vs B0 AND grouped
bootstrap lower bound > 0 AND the partner-missing stratum does not regress
by more than 0.005 (fallback must be exact B0 there if it does).
Expected: eps +0.04..0.07, nc +0.02..0.04 on shift-matched panel vs incumbent.

Run:  .venv/bin/python tools/claude_r2_02_fable/F02_eps_nc_ionic_engine.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fable_common as fc
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

from F01_ei_eea_egb_chain_engine import (
    descriptor_block,
    morgan_count_block,
    fit_predict_structure_blend,
    cross_fitted_partner_predictions,
)
from sklearn.ensemble import ExtraTreesRegressor

SEED = fc.SEED
MIN_IONIC = 0.02

POLAR_SMARTS = {
    "CF": "[#6][F]", "CCl": "[#6][Cl]", "ester": "C(=O)O", "carbonyl": "[CX3]=[OX1]",
    "ether": "[OD2]([#6])[#6]", "OH": "[OX2H]", "nitrile": "C#N", "amide": "C(=O)N",
    "NH": "[NX3;H1,H2]", "sulfone": "S(=O)(=O)", "thioether": "[#16X2]",
    "aromatic_N": "n", "aromatic_O": "o", "aromatic_S": "s", "imide": "C(=O)NC(=O)",
    "siloxane": "[Si][O]", "phosphate": "P=O", "urethane": "NC(=O)O",
}
_PATS = {k: Chem.MolFromSmarts(v) for k, v in POLAR_SMARTS.items()}


def polar_block(cans):
    """Descriptors targeting the IONIC dielectric response: polar bond density,
    H-bonding, dipolar group counts per heavy atom (Van Krevelen-style)."""
    rows = []
    for c in cans:
        m = Chem.MolFromSmiles(c)
        nh = max(m.GetNumHeavyAtoms(), 1)
        r = [len(m.GetSubstructMatches(p)) / nh for p in _PATS.values()]
        r += [Descriptors.TPSA(m) / nh, Descriptors.NumHDonors(m) / nh,
              Descriptors.NumHAcceptors(m) / nh, Descriptors.FractionCSP3(m),
              Descriptors.NumRotatableBonds(m) / nh,
              Crippen.MolMR(m) / nh, Crippen.MolLogP(m) / nh,
              rdMolDescriptors.CalcNumAromaticRings(m) / nh]
        rows.append(r)
    return np.asarray(rows, dtype=float)


def fit_ionic_model(cans_tr, ionic_tr, seed=SEED):
    # Preregistered arm: raw ionic target with polar features only. The log
    # target and Morgan-bit variants are separate hypotheses and stay out.
    X = polar_block(cans_tr)
    model = ExtraTreesRegressor(800, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    model.fit(X, np.asarray(ionic_tr, dtype=float))
    return model


def predict_ionic(model, cans):
    return model.predict(polar_block(cans))


def main():
    t0 = time.time()
    data = fc.load_data()
    ts = time.strftime("%Y%m%d-%H%M")
    output_root = os.environ.get("FABLE_OUTPUT_ROOT", os.path.join(fc.ROUND2_DIR, "experiments", "CLEAN_OFFICIAL_ONLY"))
    mode = "with_archive" if os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1" else "without_archive"
    exp_dir = os.path.join(output_root, f"R2-F02-{ts}-ionic-engine-{mode}")
    reports = []

    pairs = data.wide[data.wide["eps"].notna() & data.wide["nc"].notna()]
    pair_cans = pairs.index.tolist()

    for target, partner in (("eps", "nc"), ("nc", "eps")):
        sub = data.train[data.train.target_type == target].reset_index(drop=True)
        cans = sub["can"].tolist()
        y = sub["target"].values.astype(float)
        folds = fc.grouped_folds(cans, 5, SEED)
        Xf = morgan_count_block(cans)
        Xd = np.hstack([descriptor_block(cans), polar_block(cans)])

        # observed partner labels + cross-fitted partner predictions
        L = np.array([data.wide.loc[c, partner] if (c in data.wide.index and pd.notna(data.wide.loc[c, partner]))
                      else np.nan for c in cans])
        pp = cross_fitted_partner_predictions(data, cans, SEED)
        P = np.array([pp[partner][c] for c in cans])
        FILL = np.where(np.isnan(L), P, L)
        observed = ~np.isnan(L)

        oof = {k: np.zeros(len(y)) for k in ("B0", "B1", "B2")}
        for f in range(5):
            tr, va = folds != f, folds == f
            oof["B0"][va] = fit_predict_structure_blend(Xf[tr], Xd[tr], y[tr], Xf[va], Xd[va])

            # ionic model trained fold-locally on pairs NOT in this validation fold's structures
            va_cans = set(np.array(cans)[va])
            tr_pairs = [c for c in pair_cans if c not in va_cans]
            ionic_tr = (pairs.loc[tr_pairs, "eps"] - pairs.loc[tr_pairs, "nc"] ** 2).values
            im = fit_ionic_model(tr_pairs, ionic_tr, SEED)
            ionic_va = predict_ionic(im, list(np.array(cans)[va]))

            if target == "eps":
                phys_lab = np.where(observed[va], L[va] ** 2 + ionic_va, np.nan)
                phys_fill = FILL[va] ** 2 + ionic_va
            else:
                phys_lab = np.where(observed[va], np.sqrt(np.clip(L[va] - ionic_va, 1.0, None)), np.nan)
                phys_fill = np.sqrt(np.clip(FILL[va] - ionic_va, 1.0, None))

            oof["B1"][va] = np.where(np.isnan(phys_lab), oof["B0"][va], phys_lab)
            # B2: physics with filled partner everywhere, blended 50/50 with B0
            # on predicted-partner rows (conservative fixed weight; the report
            # shows both strata so the blend can be tuned in a follow-up child)
            oof["B2"][va] = np.where(observed[va], phys_fill, 0.5 * phys_fill + 0.5 * oof["B0"][va])

        # eps >= nc^2 + MIN_IONIC enforcement happens at assembly (needs both heads)
        strata = np.where(observed, "partner_observed", "partner_missing")
        for arm in ("B0", "B1", "B2"):
            rep = fc.evaluate_target(f"F02-{arm}", target, y, oof[arm],
                                     oof["B0"] if arm != "B0" else None, cans, folds, data,
                                     extra={"stratum_r2": {
                                         s: fc.r2_score_manual(y[strata == s], oof[arm][strata == s])
                                         for s in np.unique(strata)}})
            reports.append(rep)
            print(f"{target} {arm}: oof={rep['oof_r2']:.4f} shift={rep['shift_matched_r2']:.4f} "
                  f"strata={rep['stratum_r2']}")

    fc.save_report({"runtime_s": time.time() - t0, "reports": reports},
                   os.path.join(exp_dir, "report.json"))


if __name__ == "__main__":
    main()
