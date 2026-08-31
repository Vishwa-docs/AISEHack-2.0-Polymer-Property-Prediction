"""Physics/identity coordinate builders — the R2-bankable identities.

All identities are computed from official features only:
  * ionic   = eps - nc^2          (reparametrization of the dielectric identity)
  * chi     = (ei + eea) / 2      (Mulliken electronegativity / gap centre)
  * ei_identity  = egc + eea      (band-edge identity, affine, no ML residual)
  * egb_identity = a*egc + b      (chain->bulk gap affine fit on official pairs)
These are used as covariate blocks / soft constraints inside experiments.
"""
from __future__ import annotations

import numpy as np


def ionic_coordinate(eps: np.ndarray, nc: np.ndarray) -> np.ndarray:
    """Raw ionic term eps - nc^2 (never log-transform — R2 lesson)."""
    return eps - np.square(nc)


def chi_coordinate(ei: np.ndarray, eea: np.ndarray) -> np.ndarray:
    return 0.5 * (ei + eea)


def fit_egb_affine(egc: np.ndarray, egb: np.ndarray) -> tuple[float, float]:
    """Fit egb ~ a*egc + b on official paired rows (identity carrier)."""
    mask = np.isfinite(egc) & np.isfinite(egb)
    if mask.sum() < 3:
        return 1.0, 0.0
    a, b = np.polyfit(egc[mask], egb[mask], 1)
    return float(a), float(b)


def identity_residual_features(egc, eea, ei, egb):
    """Derived covariate columns from the DFT identities (NaN-safe)."""
    chi = 0.5 * (np.asarray(ei, float) + np.asarray(eea, float))
    ei_identity = np.asarray(egc, float) + np.asarray(eea, float)
    a, b = fit_egb_affine(np.asarray(egc, float), np.asarray(egb, float))
    egb_identity = a * np.asarray(egc, float) + b
    return {
        "chi": chi,
        "ei_identity": ei_identity,
        "egb_identity": egb_identity,
        "egb_slope": a,
        "egb_intercept": b,
    }
