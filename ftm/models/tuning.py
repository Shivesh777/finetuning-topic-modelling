"""Validation-based tuning of the return-forecasting models, and the registry of models.

The protocol
------------
A model class (see Model) has a class attribute name, a constructor taking hyperparameters
as keywords, a static grid() listing a few settings from the MOST regularised to the
least, fit(X, y, X_val=None, y_val=None), predict(X) and refit_params(). Validation data
passed to fit may only stop boosting early; the linear models, the forest and IPCA ignore
them. refit_params() returns the hyperparameters to use when the model is fitted again
without a validation set, on training plus validation data: for XGBoost the number of
rounds that early stopping kept replaces the maximum and early stopping is switched off.
Optional parts of the protocol:

* refit = False on a class that cannot be fitted again that way (the EBM, see ebm.py):
  fit_for_test then fits it as tune did, on the training rows with the validation rows
  for early stopping, and refit_params() returns the setting unchanged;

* tune_path(X_train, y_train, X_val, y_val, tolerance), a classmethod returning what tune
  returns, to score a whole regularisation path at once (the penalised linear models);
* rounds_, the boosting rounds kept by a fitted model, which tune reports;
* needs_groups = True on a class whose fit takes groups, the period of every row (IPCA).
  Rows must then be sorted by period.

The classes of linear.py, trees.py and ebm.py reject unknown hyperparameters with a
TypeError and keep the values, defaults included, in the dict params.

Conventions
-----------
* X is a float array (n, p) without NaN and y a float array (n,): the next-period excess
  return minus its cross-sectional mean. Rows are observations; nothing here knows about
  dates, so the caller makes the training rows precede the validation rows in time.
* mse is the mean of squared forecast errors. r2_zero = 1 - sum((y - pred)^2) / sum(y^2)
  compares the forecast with a forecast of zero, not with the sample mean of y.
* The tuning rule: the setting with the lowest validation MSE or, with tolerance > 0,
  the FIRST setting in grid order whose validation MSE is at most (1 + tolerance) times
  the lowest, i.e. the simpler one when two are close. The tolerance is relative to the
  MSE, and the MSEs of return forecasts differ by thousandths of a percent: a tolerance
  as large as the best validation R2 (about 1e-4 to 1e-3 on our panels) always picks the
  first setting. config.TUNE_TOLERANCE is 1e-5.
* Tuning never sees test data and fit_final never sees a validation set. The validation
  forecasts that tune returns come from a model fitted on the training rows alone.
* The model that forecasts the test period comes from fit_for_test: the chosen setting
  fitted on training plus validation rows, or, for a class with refit = False, the
  validated model itself.
"""
from __future__ import annotations

from time import perf_counter
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ftm.models.ebm import Ebm
from ftm.models.ipca import Ipca, SparseIpca
# simplest_within lives in linear.py because that module cannot import this one: the
# registry below imports it.
from ftm.models.linear import ElasticNet, Lasso, Ols, Ridge, simplest_within
from ftm.models.trees import Forest, Xgb


class Model(Protocol):
    """What tune and fit_final need from a model (duck-typed; see the module docstring)."""

    name: str

    @staticmethod
    def grid() -> list[dict]: ...

    def fit(self, X: np.ndarray, y: ArrayLike, X_val: np.ndarray | None = None,
            y_val: ArrayLike | None = None) -> Model: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def refit_params(self) -> dict: ...


MODELS: dict[str, type] = {cls.name: cls for cls in (Ols, Ridge, Lasso, ElasticNet, Xgb,
                                                     Forest, Ebm, Ipca, SparseIpca)}


def get_model(name: str) -> type:
    """The model class registered under name ("ols", "ridge", "lasso", "enet", ...)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name]


def mse(y: ArrayLike, pred: ArrayLike) -> float:
    """Mean squared error mean((y - pred)^2)."""
    error = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(error @ error) / error.size


def r2_zero(y: ArrayLike, pred: ArrayLike) -> float:
    """R2 against a zero forecast: 1 - sum((y - pred)^2) / sum(y^2). NaN if y is all zero."""
    y = np.asarray(y, dtype=np.float64)
    error = y - np.asarray(pred, dtype=np.float64)
    total = float(y @ y)
    return 1.0 - float(error @ error) / total if total > 0 else float("nan")


def _fit_keywords(model_cls: type, groups: ArrayLike | None) -> dict:
    """The keywords fit takes beyond the data: groups, for classes with needs_groups only."""
    return {"groups": groups} if getattr(model_cls, "needs_groups", False) else {}


def tune(model_cls: type, X_train: np.ndarray, y_train: ArrayLike, X_val: np.ndarray,
         y_val: ArrayLike, tolerance: float = 0.0, grid: Sequence[dict] | None = None,
         groups: ArrayLike | None = None) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Fit every setting on the training data and choose one on the validation data.

    Each setting of grid (default model_cls.grid()) is fitted on the training data, with
    the validation data passed for early stopping, and scored on the validation data. The
    chosen setting is the first in grid order whose validation MSE is at most
    (1 + tolerance) times the lowest. Returns (best_params, table, val_pred): best_params
    is the chosen model's refit_params() merged over its setting, ready for fit_final;
    table has one row per setting with its parameters, rounds (models with rounds_ only),
    val_mse, val_r2, train_r2, seconds (fitting and scoring) and chosen; val_pred (n_val,)
    holds the chosen model's forecasts of the validation rows. groups, the period of every
    training row, reaches fit only if model_cls.needs_groups. Without an explicit grid a
    model class that defines tune_path is tuned by it instead.
    """
    if grid is None and hasattr(model_cls, "tune_path"):
        return model_cls.tune_path(X_train, y_train, X_val, y_val, tolerance)
    settings = model_cls.grid() if grid is None else list(grid)
    if not settings:
        raise ValueError(f"{model_cls.__name__} has no settings to tune")
    keywords = _fit_keywords(model_cls, groups)
    rows, refits, forecasts = [], [], []
    for setting in settings:
        start = perf_counter()
        model = model_cls(**setting).fit(X_train, y_train, X_val, y_val, **keywords)
        pred = np.asarray(model.predict(X_val), dtype=np.float64)
        row = dict(setting)
        if hasattr(model, "rounds_"):
            row["rounds"] = model.rounds_
        row.update(val_mse=mse(y_val, pred), val_r2=r2_zero(y_val, pred),
                   train_r2=r2_zero(y_train, model.predict(X_train)),
                   seconds=perf_counter() - start)
        rows.append(row)
        refits.append({**setting, **model.refit_params()})
        forecasts.append(pred)
    table = pd.DataFrame(rows)
    best = simplest_within(table["val_mse"].to_numpy(), tolerance)
    table["chosen"] = np.arange(len(table)) == best
    return refits[best], table, forecasts[best]


def chosen_setting(table: pd.DataFrame) -> str:
    """The hyperparameters of the chosen row of a tuning table, as "name=value, ..."
    (empty for a model without hyperparameters)."""
    scores = ("rounds", "n_nonzero", "val_mse", "val_r2", "train_r2", "seconds", "chosen")
    row = table[table["chosen"]].iloc[0]

    def text(value) -> str:
        number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
        return f"{value:g}" if number else str(value)

    return ", ".join(f"{name}={text(row[name])}" for name in table.columns if name not in scores)


def fit_final(model_cls: type, params: dict, X: np.ndarray, y: ArrayLike,
              groups: ArrayLike | None = None) -> Model:
    """model_cls(**params) fitted on (X, y) without a validation set.

    groups, the period of every row, reaches fit only if model_cls.needs_groups.
    """
    return model_cls(**params).fit(X, y, **_fit_keywords(model_cls, groups))


def fit_for_test(model_cls: type, params: dict, X: np.ndarray, y: ArrayLike, train: np.ndarray,
                 val: np.ndarray, groups: ArrayLike | None = None) -> Model:
    """The model that forecasts the test period, given the setting that tune chose.

    train and val are boolean masks over the rows of X. The model is fitted on the rows of
    both; a class with refit = False is instead fitted exactly as tune fitted it, on the
    training rows with the validation rows deciding where boosting stops, so that the model
    tested is the model validated. groups is the period of every row of X.
    """
    y = np.asarray(y)
    period = (lambda rows: None) if groups is None else (lambda rows: np.asarray(groups)[rows])
    if getattr(model_cls, "refit", True):
        rows = train | val
        return fit_final(model_cls, params, X[rows], y[rows], period(rows))
    return model_cls(**params).fit(X[train], y[train], X[val], y[val],
                                   **_fit_keywords(model_cls, period(train)))
