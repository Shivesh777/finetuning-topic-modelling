"""Tree ensembles for return forecasts: gradient boosting (XGBoost) and a random forest.

Conventions
-----------
* X is a float32 array (n, p) without NaN, y a float array (n,); predict returns float64.
* Both models minimise squared error. Hyperparameters live in the dict params; unknown
  names raise a TypeError. seed fixes all randomness: with the same seed, data and n_jobs
  a fit is reproducible.
* Xgb uses the validation set only to stop early: boosting ends when the validation RMSE
  has not improved for early_stopping_rounds rounds, predictions use the trees up to the
  best round, and rounds_ is the number of those trees. Without validation data (or with
  early_stopping_rounds=None) exactly n_estimators rounds are trained.
* Forest ignores validation data. Each tree is grown on a bootstrap sample of at most
  max_rows rows, so the cost of a tree does not grow with the panel.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class _Tuned:
    """Hyperparameter storage shared by the model classes of this module."""

    name = ""
    defaults: dict = {}

    def __init__(self, **params) -> None:
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise TypeError(f"{type(self).__name__} has no hyperparameter {unknown}")
        self.params = {**self.defaults, **params}


class Xgb(_Tuned):
    """Gradient-boosted trees: XGBRegressor with histogram splits (64 bins), squared error.

    Shrinkage 0.05, row and column subsampling of one half per tree and at least 50
    observations per leaf; max_depth is the tuned complexity (1 = additive stumps).
    n_estimators is the maximum number of rounds.
    """

    name = "xgb"
    defaults = {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 600, "subsample": 0.5,
                "colsample_bytree": 0.5, "min_child_weight": 50, "reg_lambda": 1.0,
                "early_stopping_rounds": 50, "n_jobs": 4, "seed": 0}

    @staticmethod
    def grid() -> list[dict]:
        """Tree depths from additive stumps to five-way interactions."""
        return [{"max_depth": depth} for depth in (1, 2, 3, 5)]

    def fit(self, X: np.ndarray, y: ArrayLike, X_val: np.ndarray | None = None,
            y_val: ArrayLike | None = None) -> Xgb:
        """Boost on (X, y); with validation data stop at the best validation RMSE."""
        p = self.params
        patience = p["early_stopping_rounds"] if X_val is not None else None
        self.model_ = XGBRegressor(
            tree_method="hist", max_bin=64, objective="reg:squarederror", eval_metric="rmse",
            max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            n_estimators=p["n_estimators"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
            reg_lambda=p["reg_lambda"], early_stopping_rounds=patience or None,
            n_jobs=p["n_jobs"], random_state=p["seed"])
        if patience:
            self.model_.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            self.rounds_ = max(1, int(self.model_.best_iteration) + 1)
        else:
            self.model_.fit(X, y, verbose=False)
            self.rounds_ = int(p["n_estimators"])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecasts from the first rounds_ trees, as float64 (n,)."""
        return self.model_.predict(X, iteration_range=(0, self.rounds_)).astype(np.float64)

    def refit_params(self) -> dict:
        """params with n_estimators = rounds_ (best round + 1) and early stopping off."""
        return {**self.params, "n_estimators": self.rounds_, "early_stopping_rounds": None}

    def feature_importance(self) -> np.ndarray:
        """Share of the total split gain due to each feature, (p,), summing to one.

        The gain of a split is the fall in the regularised squared-error objective it
        produces; it is summed over the splits on a feature in the first rounds_ trees.
        All zeros if no tree has a split.
        """
        booster = self.model_.get_booster()[: self.rounds_]
        gain = np.zeros(self.model_.n_features_in_)
        for feature, value in booster.get_score(importance_type="total_gain").items():
            gain[int(feature[1:])] = value
        total = gain.sum()
        return gain / total if total > 0 else gain


class Forest(_Tuned):
    """Random forest: RandomForestRegressor with bootstrap samples of at most max_rows rows.

    max_samples = min(1, max_rows / n), at least min_samples_leaf observations per leaf and
    max_features of the columns tried at each split (a share, or "sqrt" for the square root
    of their number); max_depth is the tuned complexity.
    """

    name = "rf"
    defaults = {"max_depth": 6, "max_features": "sqrt", "n_estimators": 100,
                "min_samples_leaf": 50, "max_rows": 30_000, "n_jobs": 4, "seed": 0}

    @staticmethod
    def grid() -> list[dict]:
        """Shallow to deep trees."""
        return [{"max_depth": depth} for depth in (3, 6, 10)]

    def fit(self, X: np.ndarray, y: ArrayLike, X_val: np.ndarray | None = None,
            y_val: ArrayLike | None = None) -> Forest:
        """Grow the forest on (X, y); validation data are ignored."""
        p = self.params
        self.model_ = RandomForestRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
            max_samples=min(1.0, p["max_rows"] / X.shape[0]), bootstrap=True,
            n_jobs=p["n_jobs"], random_state=p["seed"])
        self.model_.fit(X, np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average of the trees' forecasts, as float64 (n,)."""
        return np.asarray(self.model_.predict(X), dtype=np.float64)

    def refit_params(self) -> dict:
        """The hyperparameters as they are: nothing depends on a validation set."""
        return dict(self.params)
