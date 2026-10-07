"""Linear return forecasts fitted from sufficient statistics: OLS, ridge, lasso, elastic net.

One pass over the data builds a Gram (n, sum x, sum y, X'X, X'y, y'y in float64); after
that every fit, every regularisation path and every mean squared error is an operation on
p x p matrices, so a whole path costs milliseconds and the statistics of two samples add up
(gram_train + gram_validation is the Gram of the stacked sample).

Conventions
-----------
* X is a float array (n, p) without NaN, y a float array (n,). Gram.from_data raises a
  ValueError if the statistics are not finite; nothing is imputed.
* Every model fits an unpenalised intercept: the slopes solve the problem on the centred
  moments Sxx = cov(x), Sxy = cov(x, y) (both divided by n, not n - 1) and
  intercept = mean(y) - mean(x)' coef.
* Penalties are per observation, so an alpha tuned on one sample keeps its meaning on a
  larger one:
    Ridge       minimises RSS / (2n) + (alpha / 2) * sum(b^2), i.e. b = (Sxx + alpha I)^-1 Sxy.
                scikit-learn's Ridge(alpha=a) minimises RSS + a * sum(b^2): a = n * alpha.
    Lasso       minimises RSS / (2n) + alpha * sum(|b|)                    (as scikit-learn).
    ElasticNet  minimises RSS / (2n) + alpha * l1_ratio * sum(|b|)
                + 0.5 * alpha * (1 - l1_ratio) * sum(b^2)                  (as scikit-learn).
  alpha has the units of cov(x, y) for the lasso and of var(x) for ridge. The default
  alpha = 1.0 is only a placeholder: choose alpha with tune_path.
* Features are not standardised here: they are expected on a common scale (ranks).
* Paths run from the most regularised setting to the least. Ridge: 25 alphas log-spaced
  from 1e2 down to 1e-4 times trace(Sxx) / p. Lasso and elastic net: 40 alphas log-spaced
  from alpha_max = max|Sxy| / l1_ratio (the smallest alpha at which every slope is zero)
  down to alpha_max / 1000, so the first setting is the constant forecast mean(y).
* fit(X, y) and tune_path(X_train, y_train, X_val, y_val, tolerance) take data;
  fit_gram(gram) and tune_gram(gram_train, gram_val, tolerance) take the statistics, so
  several models (or a refit on training plus validation data) can share one pass.
* R2 is measured against a zero forecast: 1 - sum((y - pred)^2) / sum(y^2).
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import enet_path

PREDICT_CHUNK = 100_000
# Eigenvalues of the centred Gram below this share of the largest count as zero in OLS.
# A float32 column that is the sum of two others leaves an eigenvalue of about 5e-16 of the
# largest: numpy's default cutoff (machine precision times p) misses it in one sample out
# of eight when p = 3, and the slopes are then of the order of 1e5.
RCOND = 1e-10


def _forecast(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """intercept + X coef as float64 (n,), converting X to float64 a chunk at a time."""
    out = np.empty(X.shape[0], dtype=np.float64)
    for start in range(0, X.shape[0], PREDICT_CHUNK):
        stop = start + PREDICT_CHUNK
        out[start:stop] = np.asarray(X[start:stop], dtype=np.float64) @ coef
    return out + intercept


def simplest_within(losses: ArrayLike, tolerance: float = 0.0) -> int:
    """Index of the first loss that is at most (1 + tolerance) * min(losses).

    With settings ordered from the most regularised to the least this keeps the simpler
    of two settings that are close; tolerance = 0 gives the (first) minimum. NaN losses
    are never chosen. The tolerance is a fraction of the loss itself: when the loss is the
    MSE of a return forecast whose R2 is a fraction of a percent, all settings lie within
    a fraction of a percent of each other and useful tolerances are of the order of 1e-5
    (config.TUNE_TOLERANCE).
    """
    losses = np.asarray(losses, dtype=np.float64)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not np.isfinite(losses).any():
        raise ValueError("no finite loss to choose from")
    return int(np.flatnonzero(losses <= (1.0 + tolerance) * np.nanmin(losses))[0])


@dataclass(eq=False)
class Gram:
    """Sufficient statistics of a regression sample: n, sum x, sum y, X'X, X'y and y'y."""

    n: int
    sum_x: np.ndarray
    sum_y: float
    XtX: np.ndarray
    Xty: np.ndarray
    yty: float

    @classmethod
    def from_data(cls, X: np.ndarray, y: ArrayLike, chunk: int = 100_000) -> Gram:
        """Accumulate the statistics of (X, y) in float64, chunk rows at a time.

        Only one chunk of X is converted to float64 at a time, so a float32 X is never
        copied whole. Raises ValueError if the shapes disagree or any statistic is not
        finite (NaN or infinite inputs).
        """
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        if y.shape != (n,):
            raise ValueError(f"X has {n} rows but y has shape {y.shape}")
        sum_x, XtX, Xty = np.zeros(p), np.zeros((p, p)), np.zeros(p)
        for start in range(0, n, chunk):
            block = np.asarray(X[start:start + chunk], dtype=np.float64)
            sum_x += block.sum(axis=0)
            XtX += block.T @ block
            Xty += block.T @ y[start:start + chunk]
        gram = cls(n, sum_x, float(y.sum()), XtX, Xty, float(y @ y))
        if not (np.isfinite(XtX).all() and np.isfinite(Xty).all() and np.isfinite(gram.yty)):
            raise ValueError("X or y contain NaN or infinite values")
        return gram

    def __add__(self, other: Gram) -> Gram:
        """Statistics of the two samples stacked (each statistic is a sum over rows)."""
        if self.XtX.shape != other.XtX.shape:
            raise ValueError("the two samples have different numbers of features")
        return Gram(self.n + other.n, self.sum_x + other.sum_x, self.sum_y + other.sum_y,
                    self.XtX + other.XtX, self.Xty + other.Xty, self.yty + other.yty)

    def centred(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
        """Centred second moments divided by n and the means: (Sxx, Sxy, syy, x_mean, y_mean).

        Sxx = X'X / n - x_mean x_mean', Sxy = X'y / n - x_mean * y_mean and
        syy = y'y / n - y_mean^2.
        """
        x_mean, y_mean = self.sum_x / self.n, self.sum_y / self.n
        Sxx = self.XtX / self.n - np.outer(x_mean, x_mean)
        Sxy = self.Xty / self.n - x_mean * y_mean
        return Sxx, Sxy, self.yty / self.n - y_mean ** 2, x_mean, y_mean

    def mse(self, coef: ArrayLike, intercept: ArrayLike) -> float | np.ndarray:
        """Mean squared error of y = intercept + x'coef on this sample, from the statistics.

        mse = syy - 2 b'Sxy + b'Sxx b + d^2 with d = intercept + x_mean'b - y_mean, the
        mean forecast error. coef may be (p,) with a scalar intercept (returns a float) or
        a path (k, p) with intercepts (k,) (returns an array (k,)). Rounding can leave a
        perfect fit a hair below zero, so the result is floored at zero.
        """
        coef = np.asarray(coef, dtype=np.float64)
        B = np.atleast_2d(coef)
        Sxx, Sxy, syy, x_mean, y_mean = self.centred()
        bias = np.asarray(intercept, dtype=np.float64) + B @ x_mean - y_mean
        out = np.maximum(syy - 2.0 * B @ Sxy + ((B @ Sxx) * B).sum(axis=1) + bias ** 2, 0.0)
        return float(out[0]) if coef.ndim == 1 else out

    def r2_zero(self, coef: ArrayLike, intercept: ArrayLike) -> float | np.ndarray:
        """R2 against a zero forecast, 1 - n * mse / y'y, with the shapes of mse.

        NaN if y is identically zero.
        """
        mean_square = self.yty / self.n if self.yty > 0 else np.nan
        return 1.0 - self.mse(coef, intercept) / mean_square


class _Linear:
    """Shared plumbing of the linear models; subclasses provide _solve on centred moments."""

    name = ""
    defaults: dict = {}

    def __init__(self, **params) -> None:
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise TypeError(f"{type(self).__name__} has no hyperparameter {unknown}")
        self.params = {**self.defaults, **params}

    @staticmethod
    def grid() -> list[dict]:
        """Empty: the settings depend on the data, so tuning goes through tune_path."""
        return []

    def _solve(self, Sxx: np.ndarray, Sxy: np.ndarray, syy: float) -> np.ndarray:
        raise NotImplementedError

    def fit_gram(self, gram: Gram) -> _Linear:
        """Fit from sufficient statistics: sets coef_ (p,) and intercept_."""
        Sxx, Sxy, syy, x_mean, y_mean = gram.centred()
        self.coef_ = self._solve(Sxx, Sxy, syy)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def fit(self, X: np.ndarray, y: ArrayLike, X_val: np.ndarray | None = None,
            y_val: ArrayLike | None = None) -> _Linear:
        """Fit on (X, y); validation data are ignored (there is nothing to stop early)."""
        return self.fit_gram(Gram.from_data(X, y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """intercept_ + X coef_ as float64 (n,), converting X to float64 a chunk at a time."""
        return _forecast(X, self.coef_, self.intercept_)

    def refit_params(self) -> dict:
        """The hyperparameters as they are: nothing depends on a validation set."""
        return dict(self.params)


class Ols(_Linear):
    """Ordinary least squares: coef = pinv(Sxx) Sxy, the minimum-norm least-squares slopes.

    The pseudo-inverse (lstsq on the centred Gram, dropping eigenvalues below RCOND times
    the largest) keeps collinear or constant columns from failing: they share or lose their
    coefficient instead. Working on X'X squares the condition number of X, so the slopes
    of nearly collinear columns are less accurate than from a QR of X.
    """

    name = "ols"

    @staticmethod
    def grid() -> list[dict]:
        """One setting without hyperparameters."""
        return [{}]

    def _solve(self, Sxx: np.ndarray, Sxy: np.ndarray, syy: float) -> np.ndarray:
        return np.linalg.lstsq(Sxx, Sxy, rcond=RCOND)[0]


class _Penalised(_Linear):
    """A penalised linear model with a data-dependent alpha path and Gram-based tuning."""

    def __init__(self, **params) -> None:
        super().__init__(**params)
        if self.params["alpha"] < 0:
            raise ValueError("alpha must be non-negative")

    def _alphas(self, Sxx: np.ndarray, Sxy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _path(self, Sxx: np.ndarray, Sxy: np.ndarray, syy: float,
              alphas: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _solve(self, Sxx: np.ndarray, Sxy: np.ndarray, syy: float) -> np.ndarray:
        return self._path(Sxx, Sxy, syy, np.array([self.params["alpha"]], dtype=float))[0]

    @classmethod
    def _tune(cls, train: Gram, val: Gram, tolerance: float,
              fixed: dict) -> tuple[dict, pd.DataFrame, np.ndarray, float]:
        """tune_gram, returning also the slopes and the intercept at the chosen alpha."""
        start = perf_counter()
        model = cls(**fixed)
        Sxx, Sxy, syy, x_mean, y_mean = train.centred()
        alphas = model._alphas(Sxx, Sxy)
        coefs = model._path(Sxx, Sxy, syy, alphas)
        intercepts = y_mean - coefs @ x_mean
        val_mse = val.mse(coefs, intercepts)
        best = simplest_within(val_mse, tolerance)
        columns = {"alpha": alphas}
        if "l1_ratio" in model.params:
            columns["l1_ratio"] = model.params["l1_ratio"]
        table = pd.DataFrame({
            **columns,
            "n_nonzero": np.count_nonzero(coefs, axis=1),
            "val_mse": val_mse,
            "val_r2": val.r2_zero(coefs, intercepts),
            "train_r2": train.r2_zero(coefs, intercepts),
            "seconds": (perf_counter() - start) / len(alphas),
            "chosen": np.arange(len(alphas)) == best,
        })
        best_params = {**model.params, "alpha": float(alphas[best])}
        return best_params, table, coefs[best], float(intercepts[best])

    @classmethod
    def tune_gram(cls, train: Gram, val: Gram, tolerance: float = 0.0,
                  **fixed) -> tuple[dict, pd.DataFrame]:
        """Fit the whole alpha path on train, score it on val and choose by the rule.

        Returns (best_params, table). table has one row per alpha, from the most
        regularised to the least: alpha (and l1_ratio for the elastic net), n_nonzero,
        val_mse, val_r2, train_r2, seconds (the time of the path divided by its length)
        and chosen. The chosen row is the first whose val_mse is at most (1 + tolerance)
        times the lowest (see simplest_within). fixed are hyperparameters other than alpha.
        """
        return cls._tune(train, val, tolerance, fixed)[:2]

    @classmethod
    def tune_path(cls, X_train: np.ndarray, y_train: ArrayLike, X_val: np.ndarray,
                  y_val: ArrayLike,
                  tolerance: float = 0.0) -> tuple[dict, pd.DataFrame, np.ndarray]:
        """tune_gram on the statistics of the two samples (one pass over each).

        Returns (best_params, table, val_pred): val_pred (n_val,) holds the forecasts of
        the validation rows at the chosen alpha. seconds then includes the time of the two
        passes, spread evenly over the rows.
        """
        start = perf_counter()
        train, val = Gram.from_data(X_train, y_train), Gram.from_data(X_val, y_val)
        best_params, table, coef, intercept = cls._tune(train, val, tolerance, {})
        table["seconds"] = (perf_counter() - start) / len(table)
        return best_params, table, _forecast(X_val, coef, intercept)


class Ridge(_Penalised):
    """Ridge regression: b = (Sxx + alpha I)^-1 Sxy, alpha per observation.

    Solved through the eigendecomposition Sxx = V diag(w) V', b = V diag(1 / (w + alpha)) V'
    Sxy, which gives the whole path at the cost of one decomposition. Directions with
    w + alpha = 0 (alpha = 0 on collinear columns) get a zero coefficient.
    """

    name = "ridge"
    defaults = {"alpha": 1.0}
    n_alphas = 25

    def _alphas(self, Sxx: np.ndarray, Sxy: np.ndarray) -> np.ndarray:
        return np.trace(Sxx) / len(Sxx) * np.logspace(2.0, -4.0, self.n_alphas)

    def _path(self, Sxx: np.ndarray, Sxy: np.ndarray, syy: float,
              alphas: np.ndarray) -> np.ndarray:
        eigenvalues, vectors = np.linalg.eigh(Sxx)
        denominator = np.maximum(eigenvalues, 0.0)[None, :] + np.asarray(alphas)[:, None]
        safe = np.where(denominator > 0, denominator, 1.0)
        weights = np.where(denominator > 0, (vectors.T @ Sxy) / safe, 0.0)
        return weights @ vectors.T


class ElasticNet(_Penalised):
    """Elastic net in scikit-learn's (alpha, l1_ratio) parametrisation, on the centred Gram.

    Solved by scikit-learn's coordinate descent (enet_path with precompute and Xy), warm
    started along the path. tol is the duality-gap tolerance relative to sum(y^2): it is
    set far below scikit-learn's default of 1e-4 because the explained share of return
    variance is itself of the order of 1e-3.
    """

    name = "enet"
    defaults = {"alpha": 1.0, "l1_ratio": 0.5, "tol": 1e-8, "max_iter": 20_000}
    n_alphas = 40

    def __init__(self, **params) -> None:
        super().__init__(**params)
        if not 0 < self._l1_ratio() <= 1:
            raise ValueError("l1_ratio must be in (0, 1]: use Ridge for a pure L2 penalty")

    def _l1_ratio(self) -> float:
        return float(self.params["l1_ratio"])

    def _alphas(self, Sxx: np.ndarray, Sxy: np.ndarray) -> np.ndarray:
        alpha_max = max(float(np.abs(Sxy).max()) / self._l1_ratio(), np.finfo(float).tiny)
        return alpha_max * np.logspace(0.0, -3.0, self.n_alphas)

    def _path(self, Sxx: np.ndarray, Sxy: np.ndarray, syy: float,
              alphas: np.ndarray) -> np.ndarray:
        # enet_path sorts alphas in decreasing order, which is the order of every caller.
        features, target = _moment_matched_sample(Sxx, Sxy, syy)
        m = len(target)
        _, coefs, _ = enet_path(features, target, l1_ratio=self._l1_ratio(), alphas=alphas,
                                precompute=m * Sxx, Xy=m * Sxy, tol=self.params["tol"],
                                max_iter=self.params["max_iter"])
        return np.ascontiguousarray(coefs.T)


class Lasso(ElasticNet):
    """Lasso: the elastic net with l1_ratio = 1, RSS / (2n) + alpha * sum(|b|)."""

    name = "lasso"
    defaults = {"alpha": 1.0, "tol": 1e-8, "max_iter": 20_000}

    def _l1_ratio(self) -> float:
        return 1.0


def _moment_matched_sample(Sxx: np.ndarray, Sxy: np.ndarray,
                           syy: float) -> tuple[np.ndarray, np.ndarray]:
    """A sample of p + 1 rows whose uncentred moments per row equal (Sxx, Sxy, syy).

    With M = [[Sxx, Sxy], [Sxy', syy]] = R'R (from the eigendecomposition, negative
    rounding eigenvalues set to zero) the rows of sqrt(p + 1) * R are returned as (X, y):
    X'X / m = Sxx, X'y / m = Sxy and y'y / m = syy with m = p + 1. Any least-squares
    objective per observation is therefore the same on this small sample as on the centred
    data, which lets scikit-learn's solver run without the n original rows.
    """
    p = len(Sxy)
    moments = np.empty((p + 1, p + 1))
    moments[:p, :p], moments[:p, p], moments[p, :p], moments[p, p] = Sxx, Sxy, Sxy, syy
    eigenvalues, vectors = np.linalg.eigh(moments)
    root = np.sqrt((p + 1) * np.maximum(eigenvalues, 0.0))[:, None] * vectors.T
    return np.asfortranarray(root[:, :p]), np.ascontiguousarray(root[:, p])
