"""F01 - Availability-stratified chained identity engine for ei, eea, egb.

Hypothesis
----------
Numbers below are measured on official train.csv + archive/train.csv (both are
permitted Round 2 bundle files). Train-only equivalents are in brackets, since
archive/ contributes only tg and egc rows and therefore shrinks the identity
fitting sets without changing the mechanism.

The DFT identities are generation identities, not correlations:
  ei = egc + eea   grouped-5-fold R2 0.9276 raw / 0.9278 affine  (n=92) [0.9629, n=59]
  eea = ei - egc   grouped-5-fold R2 0.9546 raw / 0.9552 affine  (n=92) [0.9710, n=59]
  egb = a*egc + b  grouped-5-fold R2 0.9205, rising to 0.9478 with a
                   full-weight ExtraTrees residual              (n=268) [0.9443, n=175]

The incumbent underuses them in two ways:
1. It abstains instead of CHAINING when a partner label is missing. Test ei rows
   have eea labelled for 98/148 and egc for 115/148, with the remainder
   available as CO-TEST rows (49 and 62) and only 1 row lacking eea entirely.
   A missing partner label is a second unknown, not an absence.
2. It never reconciles ei/eea/egc predictions to be mutually consistent.

MEASURED NEGATIVE RESULT - do not "improve" on this:
The identity residual ei - (egc + eea) has leave-one-out ExtraTrees R2 of -0.82
(Ridge on fingerprints: -0.18). It is pure noise at this sample size, and adding
a modelled residual to the ei/eea identity COSTS ~0.06 R2. Hence IDENTITY_RESID
below is 0.0 for ei/eea and 1.0 for egb, where the residual is the real
physical interchain-screening term and IS predictable.

Design (preregistered)
----------------------
For each target t in {ei, eea, egb} evaluate FOUR nested arms on identical
grouped folds, using ONLY official inputs:

  A0 baseline: structure-only blend (Morgan-count KRR-Tanimoto + LightGBM on
     RDKit descriptors), the C050-comparable control.
  A1 partner-label arm: A0 features + observed partner LABELS (official
     train.csv + archive/train.csv, same canonical structure) + missingness
     flags. Deployment-matched:
     labels are never masked (they exist at test time).
  A2 chained arm: A1, but missing partner labels are filled with CROSS-FITTED
     partner predictions (partner model trained with the current row's
     structure excluded - the C132/C139 circularity fix), plus a
     "partner_is_predicted" flag so the model can discount imputed values.
  A3 physics arm: fold-local affine identity, with the residual weight fixed
     per target by IDENTITY_RESID (0.0 for ei/eea, 1.0 for egb):
       ei_hat  = Huber_affine(egc_fill + eea_fill)
       eea_hat = Huber_affine(ei_fill  - egc_fill)
       egb_hat = Huber_affine(egc_fill) + 1.0 * ExtraTrees_residual
     Fills are observed labels where available, else cross-fitted predictions.
     Final A3 = blend of the physics prediction and A2, with the blend weight
     selected on INNER folds of the training part only (never on the outer
     validation fold) to avoid same-OOF selection optimism.

Availability strata are reported separately (both/one/none partners) because
test-time availability is known exactly and differs from train.

Decision metric: shift-matched R2 (see fable_common). Gates per target:
  bank for compound if delta_shift_matched >= +0.005 AND grouped bootstrap
  lower bound > 0 AND no stratum degrades by more than 0.01.
Incumbent test-side baseline: ei 0.8168 / eea 0.9417 / egb 0.9353.
Expected outcome on the shift-matched panel:
  ei  0.8168 -> 0.88..0.91
  eea 0.9417 -> 0.95..0.96
  egb 0.9353 -> 0.945..0.955

Run:  .venv/bin/python tools/claude_r2_02_fable/F01_ei_eea_egb_chain_engine.py
Writes: experiments/CLEAN_OFFICIAL_ONLY/R2-F01-<ts>/report.json (+ test preds)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fable_common as fc
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import HuberRegressor

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

SEED = fc.SEED
PARTNERS = {
    "ei": ["egc", "eea", "egb", "nc", "eps"],
    "eea": ["egc", "ei", "egb", "nc", "eps"],
    "egb": ["egc", "eea", "ei", "nc", "eps"],
}
IDENTITY = {
    # target: (list of components, sign vector) -> t ~ affine(sum_i sign_i * comp_i)
    "ei": (["egc", "eea"], [1.0, 1.0]),
    "eea": (["ei", "egc"], [1.0, -1.0]),
    "egb": (["egc"], [1.0]),
}
# Weight on the ML residual added to the affine identity. MEASURED, not guessed:
# the ei/eea identity residual has leave-one-out R2 = -0.82, so any nonzero
# weight costs score; the egb residual is the real interchain-screening term and
# full weight lifts grouped R2 0.9205 -> 0.9478 (n=268, train+archive).
IDENTITY_RESID = {"ei": 0.0, "eea": 0.0, "egb": 1.0}


def descriptor_block(cans):
    rows = []
    for c in cans:
        m = Chem.MolFromSmiles(c)
        nh = m.GetNumHeavyAtoms()
        rows.append([
            nh, Descriptors.MolWt(m), Crippen.MolMR(m), Crippen.MolLogP(m),
            Descriptors.TPSA(m), Descriptors.NumRotatableBonds(m),
            rdMolDescriptors.CalcNumAromaticRings(m), Descriptors.FractionCSP3(m),
            Descriptors.NumHAcceptors(m), Descriptors.NumHDonors(m),
            rdMolDescriptors.CalcNumRings(m), Descriptors.BertzCT(m),
            Descriptors.MaxPartialCharge(m, force=True) or 0.0,
            Descriptors.MinPartialCharge(m, force=True) or 0.0,
        ])
    # Some official polymer SMILES yield undefined partial charges. Keep the
    # descriptor block deterministic and finite without using labels or fold
    # statistics; zero is the neutral sentinel for an undefined charge.
    return np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def morgan_count_block(cans, radius=2, nbits=2048):
    X = np.zeros((len(cans), nbits), dtype=np.float32)
    for i, c in enumerate(cans):
        fp = AllChem.GetHashedMorganFingerprint(Chem.MolFromSmiles(c), radius, nbits)
        for k, v in fp.GetNonzeroElements().items():
            X[i, k] = v
    return X


def tanimoto_kernel(A, B):
    """MinMax (generalized Tanimoto) kernel on count vectors."""
    K = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    for i in range(A.shape[0]):
        mins = np.minimum(A[i], B).sum(axis=1)
        maxs = np.maximum(A[i], B).sum(axis=1)
        K[i] = np.where(maxs > 0, mins / maxs, 0.0)
    return K


def fit_predict_structure_blend(Xf_tr, Xd_tr, y_tr, Xf_va, Xd_va, seed=SEED):
    """Structure-only blend: Tanimoto KRR + (LightGBM | ExtraTrees)."""
    K_tr = tanimoto_kernel(Xf_tr, Xf_tr)
    K_va = tanimoto_kernel(Xf_va, Xf_tr)
    best_pred, best_err = None, np.inf
    for alpha in (1e-3, 1e-2, 1e-1):  # small inner grid, fixed
        kr = KernelRidge(alpha=alpha, kernel="precomputed")
        kr.fit(K_tr, y_tr)
        # quick inner estimate via 3-fold on training part
        inner = fc.grouped_folds(range(len(y_tr)), 3, seed)
        errs = []
        for f in range(3):
            tr, va = inner != f, inner == f
            if va.sum() < 3:
                continue
            kr_i = KernelRidge(alpha=alpha, kernel="precomputed")
            kr_i.fit(K_tr[np.ix_(tr, tr)], y_tr[tr])
            errs.append(np.mean((kr_i.predict(K_tr[np.ix_(va, tr)]) - y_tr[va]) ** 2))
        e = float(np.mean(errs))
        if e < best_err:
            best_err = e
            best_pred = kr.predict(K_va)
    krr_pred = best_pred
    if HAVE_LGBM:
        gbm = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                            colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                            min_child_samples=8, random_state=seed, verbosity=-1)
    else:
        gbm = ExtraTreesRegressor(600, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    gbm.fit(np.hstack([Xd_tr, Xf_tr]), y_tr)
    gbm_pred = gbm.predict(np.hstack([Xd_va, Xf_va]))
    return 0.5 * krr_pred + 0.5 * gbm_pred


def cross_fitted_partner_predictions(data, target_cans_all, seed=SEED):
    """For every canonical structure of interest, produce partner-property
    predictions from models whose training data excluded that structure.
    Returns: dict prop -> (dict can -> prediction)."""
    out = {}
    for prop in fc.ELECTRONIC:
        rows = data.wide[data.wide[prop].notna()]
        cans = rows.index.tolist()
        y = rows[prop].values
        need = sorted(set(target_cans_all))
        folds = fc.grouped_folds(cans, 5, seed)
        Xf = morgan_count_block(cans)
        Xd = descriptor_block(cans)
        Xf_need = morgan_count_block(need)
        Xd_need = descriptor_block(need)
        preds = np.zeros((len(need), 5))
        labeled = {c: i for i, c in enumerate(cans)}
        for f in range(5):
            tr = folds != f
            p = fit_predict_structure_blend(Xf[tr], Xd[tr], y[tr], Xf_need, Xd_need, seed)
            preds[:, f] = p
        # for a structure that is itself labeled for prop: use only folds that
        # excluded it; else average all folds
        res = {}
        for i, c in enumerate(need):
            if c in labeled:
                f_own = folds[labeled[c]]
                use = [preds[i, f] for f in range(5) if f != f_own]
            else:
                use = preds[i]
            res[c] = float(np.mean(use))
        out[prop] = res
    return out


def main():
    t0 = time.time()
    data = fc.load_data()
    ts = time.strftime("%Y%m%d-%H%M")
    output_root = os.environ.get("FABLE_OUTPUT_ROOT", os.path.join(fc.ROUND2_DIR, "experiments", "CLEAN_OFFICIAL_ONLY"))
    mode = "with_archive" if os.environ.get("FABLE_INCLUDE_ARCHIVE", "1") == "1" else "without_archive"
    exp_dir = os.path.join(output_root, f"R2-F01-{ts}-chain-engine-{mode}")
    reports = []

    for target in ("ei", "eea", "egb"):
        sub = data.train[data.train.target_type == target].reset_index(drop=True)
        cans = sub["can"].tolist()
        y = sub["target"].values.astype(float)
        folds = fc.grouped_folds(cans, 5, SEED)
        Xf = morgan_count_block(cans)
        Xd = descriptor_block(cans)

        partner_cols = PARTNERS[target]
        # observed partner labels (never masked - deployment-matched)
        L = np.full((len(cans), len(partner_cols)), np.nan)
        for j, p in enumerate(partner_cols):
            L[:, j] = [data.wide.loc[c, p] if (c in data.wide.index and pd.notna(data.wide.loc[c, p])) else np.nan
                       for c in cans]
        # cross-fitted partner predictions for chaining (fold-safe fills)
        pp = cross_fitted_partner_predictions(data, cans, SEED)
        P = np.array([[pp[p][c] for p in partner_cols] for c in cans])
        FILL = np.where(np.isnan(L), P, L)
        IS_PRED = np.isnan(L).astype(float)

        oof = {k: np.zeros(len(y)) for k in ("A0", "A1", "A2", "A3")}
        for f in range(5):
            tr, va = folds != f, folds == f
            # A0 structure-only
            oof["A0"][va] = fit_predict_structure_blend(Xf[tr], Xd[tr], y[tr], Xf[va], Xd[va])
            # A1 + observed labels (nan -> column median of training fold)
            med = np.nanmedian(L[tr], axis=0)
            L_tr = np.where(np.isnan(L[tr]), med, L[tr])
            L_va = np.where(np.isnan(L[va]), med, L[va])
            X1_tr = np.hstack([Xd[tr], L_tr, np.isnan(L[tr]).astype(float)])
            X1_va = np.hstack([Xd[va], L_va, np.isnan(L[va]).astype(float)])
            oof["A1"][va] = fit_predict_structure_blend(Xf[tr], X1_tr, y[tr], Xf[va], X1_va)
            # A2 chained fills
            X2_tr = np.hstack([Xd[tr], FILL[tr], IS_PRED[tr]])
            X2_va = np.hstack([Xd[va], FILL[va], IS_PRED[va]])
            oof["A2"][va] = fit_predict_structure_blend(Xf[tr], X2_tr, y[tr], Xf[va], X2_va)
            # A3 physics identity. The ei/eea residual is deliberately not
            # modeled; only egb gets the preregistered interchain residual.
            comps, signs = IDENTITY[target]
            idx = [partner_cols.index(c) for c in comps]
            base_tr = sum(s * FILL[tr][:, i] for s, i in zip(signs, idx))
            base_va = sum(s * FILL[va][:, i] for s, i in zip(signs, idx))
            hub = HuberRegressor().fit(base_tr.reshape(-1, 1), y[tr])
            id_tr = hub.predict(base_tr.reshape(-1, 1))
            id_va = hub.predict(base_va.reshape(-1, 1))
            if IDENTITY_RESID[target] > 0:
                resid_model = ExtraTreesRegressor(500, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
                resid_model.fit(np.hstack([Xd[tr], FILL[tr], IS_PRED[tr]]), y[tr] - id_tr)
                phys_va = id_va + IDENTITY_RESID[target] * resid_model.predict(
                    np.hstack([Xd[va], FILL[va], IS_PRED[va]]))
            else:
                phys_va = id_va
            # blend weight chosen on INNER grouped folds of the training part
            # only (never on this outer validation fold) to avoid same-OOF
            # selection optimism.
            inner = fc.grouped_folds([c for c, m in zip(cans, tr) if m], 3, SEED + 1)
            tr_idx = np.where(tr)[0]
            inner_phys = np.zeros(len(tr_idx))
            inner_a2 = np.zeros(len(tr_idx))
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 3:
                    continue
                a, b = tr_idx[itr], tr_idx[iva]
                hub_i = HuberRegressor().fit(
                    sum(s * FILL[a][:, i] for s, i in zip(signs, idx)).reshape(-1, 1), y[a])
                base_b = sum(s * FILL[b][:, i] for s, i in zip(signs, idx))
                if IDENTITY_RESID[target] > 0:
                    rm_i = ExtraTreesRegressor(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
                    rm_i.fit(np.hstack([Xd[a], FILL[a], IS_PRED[a]]),
                             y[a] - hub_i.predict(sum(s * FILL[a][:, i] for s, i in zip(signs, idx)).reshape(-1, 1)))
                    inner_phys[iva] = hub_i.predict(base_b.reshape(-1, 1)) + \
                        IDENTITY_RESID[target] * rm_i.predict(
                            np.hstack([Xd[b], FILL[b], IS_PRED[b]]))
                else:
                    inner_phys[iva] = hub_i.predict(base_b.reshape(-1, 1))
                inner_a2[iva] = fit_predict_structure_blend(
                    Xf[a], np.hstack([Xd[a], FILL[a], IS_PRED[a]]), y[a],
                    Xf[b], np.hstack([Xd[b], FILL[b], IS_PRED[b]]))
            grid = np.linspace(0.0, 1.0, 11)
            errs = [np.mean((w * inner_phys + (1 - w) * inner_a2 - y[tr_idx]) ** 2) for w in grid]
            w_phys = float(grid[int(np.argmin(errs))])
            oof["A3"][va] = w_phys * phys_va + (1 - w_phys) * oof["A2"][va]

        strata = np.where(np.isnan(L[:, [partner_cols.index(c) for c in IDENTITY[target][0]]]).sum(1) == 0,
                          "full_support", "partial_or_none")
        for arm in ("A0", "A1", "A2", "A3"):
            rep = fc.evaluate_target(f"F01-{arm}", target, y, oof[arm], oof["A0"] if arm != "A0" else None,
                                     cans, folds, data,
                                     extra={"stratum_r2": {
                                         s: fc.r2_score_manual(y[strata == s], oof[arm][strata == s])
                                         for s in np.unique(strata)}})
            reports.append(rep)
            print(f"{target} {arm}: oof={rep['oof_r2']:.4f} shift={rep['shift_matched_r2']:.4f}")

    fc.save_report({"runtime_s": time.time() - t0, "reports": reports},
                   os.path.join(exp_dir, "report.json"))


if __name__ == "__main__":
    main()
