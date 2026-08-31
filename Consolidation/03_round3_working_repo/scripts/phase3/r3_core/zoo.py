"""Model zoo: config-driven model factories + built-in random hyperparameter search.

Official-data-only, fixed seeds, trained from scratch in-process. No external
weights, no oracle, no cached artifacts.
"""
from __future__ import annotations

import numpy as np

SEED = 2026

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None
try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def _lgbm(params: dict, seed: int):
    if LGBMRegressor is None:
        return None
    base = dict(
        n_estimators=params.get("n_estimators", 900),
        learning_rate=params.get("learning_rate", 0.035),
        num_leaves=params.get("num_leaves", 63),
        min_child_samples=params.get("min_child_samples", 12),
        subsample=params.get("subsample", 0.85),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        random_state=seed,
        n_jobs=8,
        verbose=-1,
    )
    if params.get("boosting") == "dart":
        base["boosting_type"] = "dart"
        base["drop_rate"] = params.get("drop_rate", 0.1)
        base["n_estimators"] = params.get("n_estimators", 600)
        base.pop("subsample", None)
    if params.get("objective") == "quantile":
        base["objective"] = "quantile"
        base["alpha"] = params.get("alpha", 0.5)
    return LGBMRegressor(**base)


def _xgb(params: dict, seed: int):
    if XGBRegressor is None:
        return None
    return XGBRegressor(
        n_estimators=params.get("n_estimators", 900),
        learning_rate=params.get("learning_rate", 0.04),
        max_depth=params.get("max_depth", 7),
        subsample=params.get("subsample", 0.85),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        tree_method=params.get("tree_method", "hist"),
        random_state=seed,
        n_jobs=8,
        verbosity=0,
    )


def _cat(params: dict, seed: int):
    if CatBoostRegressor is None:
        return None
    return CatBoostRegressor(
        iterations=params.get("iterations", 1200),
        learning_rate=params.get("learning_rate", 0.05),
        depth=params.get("depth", 6),
        l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=8,
    )


class SkFallback:
    """HistGB fallback when a GBM library is unavailable."""

    def __init__(self, params: dict, seed: int):
        self.m = HistGradientBoostingRegressor(
            max_iter=params.get("n_estimators", 600),
            learning_rate=params.get("learning_rate", 0.05),
            max_leaf_nodes=params.get("num_leaves", 63),
            random_state=seed,
        )

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.m.fit(X, y)
        else:
            self.m.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.m.predict(X)


class VoteModel:
    """Fits several model specs and averages predictions (in-run ensemble)."""

    def __init__(self, specs, seed=SEED):
        self.specs = specs
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        self.models = [make_model(dict(s), seed=self.seed + i) for i, s in enumerate(self.specs)]
        for m in self.models:
            if sample_weight is not None:
                try:
                    m.fit(X, y, sample_weight=sample_weight)
                    continue
                except TypeError:
                    pass
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def make_model(spec: dict, seed: int = SEED):
    """spec: {"type": "lgbm"|"xgb"|"cat"|"ridge"|"et"|"rf"|"svr"|"knn"|"gpr"|"bayesridge"|"hgb"|"krr", ...params}"""
    spec = dict(spec)
    kind = spec.pop("type", "lgbm")
    params = {k: v for k, v in spec.items() if k != "type"}
    if kind == "lgbm":
        m = _lgbm(params, seed)
        if m is None:
            m = SkFallback(params, seed)
        return m
    if kind == "xgb":
        m = _xgb(params, seed)
        if m is None:
            m = SkFallback(params, seed)
        return m
    if kind == "cat":
        m = _cat(params, seed)
        if m is None:
            m = SkFallback(params, seed)
        return m
    if kind == "ridge":
        return Ridge(alpha=params.get("alpha", 10.0), random_state=seed)
    if kind == "bayesridge":
        return BayesianRidge()
    if kind == "et":
        return ExtraTreesRegressor(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", 0.6),
            n_jobs=8,
            random_state=seed,
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", 0.5),
            n_jobs=8,
            random_state=seed,
        )
    if kind == "svr":
        return SVR(C=params.get("C", 10.0), epsilon=params.get("epsilon", 0.1), gamma=params.get("gamma", "scale"))
    if kind == "knn":
        return KNeighborsRegressor(
            n_neighbors=params.get("k", 5), weights=params.get("weights", "distance"), n_jobs=-1
        )
    if kind == "gpr":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        length = params.get("length_scale", 1.0)
        kernel = ConstantKernel(1.0) * RBF(length_scale=length) + WhiteKernel(noise_level=params.get("noise", 0.1))
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=1)
    if kind == "hgb":
        return SkFallback(params, seed)
    if kind == "vote":
        return VoteModel(params.get("models", []), seed=seed)
    if kind == "mlp":
        from . import nn as nnmod

        return nnmod.SkMLP(params=params, seed=seed)
    raise ValueError(f"unknown model type: {kind}")


# ---------------------------------------------------------------------------
# Mixup / noise augmentation wrappers (feature-space regularisation)
# ---------------------------------------------------------------------------

def augment_xy(X: np.ndarray, y: np.ndarray, *, mixup: float = 0.0, noise: float = 0.0, seed: int = SEED):
    """Return optionally augmented (X, y). mixup: fraction of extra mixed rows."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if noise > 0:
        X = X + rng.normal(0.0, noise * (X.std(axis=0) + 1e-9), size=X.shape)
    if mixup > 0 and len(X) > 4:
        n_extra = int(len(X) * mixup)
        idx_a = rng.integers(0, len(X), n_extra)
        idx_b = rng.integers(0, len(X), n_extra)
        lam = rng.beta(2.0, 2.0, size=(n_extra, 1))
        X_extra = lam * X[idx_a] + (1 - lam) * X[idx_b]
        y_extra = lam.ravel() * y[idx_a] + (1 - lam.ravel()) * y[idx_b]
        X = np.vstack([X, X_extra])
        y = np.concatenate([y, y_extra])
    return X, y


# ---------------------------------------------------------------------------
# Built-in random search (no optuna dependency; deterministic seed)
# ---------------------------------------------------------------------------

LGBM_SPACE = {
    "num_leaves": ("int", 31, 383),
    "learning_rate": ("log", 0.008, 0.2),
    "min_child_samples": ("int", 5, 80),
    "feature_fraction": ("uniform", 0.4, 1.0),
    "bagging_fraction": ("uniform", 0.5, 1.0),
    "reg_lambda": ("log", 1e-3, 10.0),
}
XGB_SPACE = {
    "max_depth": ("int", 4, 12),
    "learning_rate": ("log", 0.01, 0.3),
    "subsample": ("uniform", 0.5, 1.0),
    "colsample_bytree": ("uniform", 0.5, 1.0),
    "reg_lambda": ("log", 1e-3, 10.0),
}
CAT_SPACE = {
    "depth": ("int", 4, 10),
    "learning_rate": ("log", 0.01, 0.3),
    "l2_leaf_reg": ("log", 0.1, 10.0),
}

KIND_TO_CLS = {"lgbm": (_lgbm, LGBM_SPACE), "xgb": (_xgb, XGB_SPACE), "cat": (_cat, CAT_SPACE)}


def sample_params(space: dict, rng) -> dict:
    out = {}
    for key, spec in space.items():
        kind = spec[0]
        if kind == "int":
            out[key] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == "log":
            out[key] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            out[key] = float(rng.uniform(spec[1], spec[2]))
    # map feature_fraction -> colsample_bytree, bagging_fraction -> subsample for lgbm ctor
    if "feature_fraction" in out:
        out["colsample_bytree"] = out.pop("feature_fraction")
    if "bagging_fraction" in out:
        out["subsample"] = out.pop("bagging_fraction")
    return out


def random_search(model_kind: str, X: np.ndarray, y: np.ndarray, folds, n_trials: int = 25, seed: int = SEED):
    """Small-n random search over grouped folds. Returns (best_params, best_cv_r2, trials)."""
    from sklearn.metrics import r2_score

    ctor, space = KIND_TO_CLS[model_kind]
    rng = np.random.default_rng(seed)
    trials = []
    best = (None, -np.inf)
    for t in range(n_trials):
        params = sample_params(space, rng)
        scores = []
        for tr_idx, va_idx in folds:
            m = ctor(params, seed)
            m.fit(X[tr_idx], y[tr_idx])
            scores.append(r2_score(y[va_idx], m.predict(X[va_idx])))
        mean_s = float(np.mean(scores)) if scores else -np.inf
        trials.append({"trial": t, "params": params, "cv_r2": mean_s})
        if mean_s > best[1]:
            best = (params, mean_s)
    return best[0], best[1], trials
    return None
