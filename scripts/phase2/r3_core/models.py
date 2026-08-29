"""Per-target model arms, all trained from scratch with fixed seeds.

Every fit happens inside the run; nothing is cached or loaded from disk.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SEED = 2026

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


class PerTargetEnsemble:
    """Ridge + ExtraTrees + GBM (LGBM if available else HistGB) per target."""

    def __init__(
        self,
        seed: int = SEED,
        gbm_rounds: int = 280,
        lr: float = 0.04,
        leaves: int = 25,
        ridge_alpha: float = 18.0,
        et_estimators: int = 120,
    ) -> None:
        self.seed = seed
        self.gbm_rounds = gbm_rounds
        self.lr = lr
        self.leaves = leaves
        self.ridge_alpha = ridge_alpha
        self.et_estimators = et_estimators
        self.gbm = None
        self.et = None
        self.ridge = None
        self.ridge_scaler = None

    def fit(self, X, y) -> "PerTargetEnsemble":
        n = len(y)
        min_child = max(8, n // 35)
        if HAVE_LGB:
            self.gbm = lgb.LGBMRegressor(
                n_estimators=self.gbm_rounds, learning_rate=self.lr, num_leaves=self.leaves,
                min_child_samples=min_child, subsample=0.85, colsample_bytree=0.40,
                reg_lambda=1.0, random_state=self.seed, n_jobs=2, verbosity=-1,
            )
        else:
            self.gbm = HistGradientBoostingRegressor(
                max_iter=160, learning_rate=0.05, max_leaf_nodes=17,
                min_samples_leaf=min_child, l2_regularization=1.0, random_state=self.seed,
            )
        self.et = ExtraTreesRegressor(
            n_estimators=self.et_estimators, max_features=0.25, min_samples_leaf=2,
            n_jobs=2, random_state=self.seed,
        )
        self.ridge_scaler = StandardScaler()
        X_scaled = self.ridge_scaler.fit_transform(X)
        self.ridge = Ridge(alpha=self.ridge_alpha).fit(X_scaled, y)
        self.gbm.fit(X, y)
        self.et.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        p_gbm = self.gbm.predict(X)
        p_et = self.et.predict(X)
        p_ridge = self.ridge.predict(self.ridge_scaler.transform(X))
        return 0.40 * p_gbm + 0.35 * p_et + 0.25 * p_ridge


class FastRidgeArm:
    """Cheap standardized Ridge arm for ablations / baselines."""

    def __init__(self, alpha: float = 10.0, seed: int = SEED) -> None:
        self.alpha = alpha
        self.seed = seed
        self.model = None
        self.scaler = None

    def fit(self, X, y) -> "FastRidgeArm":
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = Ridge(alpha=self.alpha, random_state=self.seed).fit(Xs, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))


class ExtraTreesArm:
    def __init__(self, n_estimators: int = 160, min_samples_leaf: int = 2, seed: int = SEED) -> None:
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            max_features=0.25, n_jobs=2, random_state=seed,
        )

    def fit(self, X, y) -> "ExtraTreesArm":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


class HistGBArm:
    def __init__(self, seed: int = SEED, max_iter: int = 200, lr: float = 0.06) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=lr, max_leaf_nodes=23,
            min_samples_leaf=8, l2_regularization=1.0, random_state=seed,
        )

    def fit(self, X, y) -> "HistGBArm":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
