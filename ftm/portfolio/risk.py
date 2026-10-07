"""Covariance matrices of asset returns for portfolio construction.

Conventions
-----------
* ``returns`` is a (T, n) array of simple returns as decimal fractions (0.01 is 1%), one row per
  day and one column per asset. NaN (or an infinite value) marks a day without a return.
* Variances and covariances are per period of the input and in its squared units: daily returns
  give daily variances. Nothing is annualised here (multiply by 252 for an annual figure).
* Sample variances divide by the number of observations minus one.
* Missing days: every asset is demeaned over the days on which it is observed and its missing
  days are then set to zero, so they add nothing to any cross-product.
* A covariance is over all n assets in the column order of ``returns``; the estimators never
  return NaN, so a caller decides separately which assets are investable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

__all__ = ["FactorCovariance", "factor_covariance", "ledoit_wolf_covariance"]


@dataclass(frozen=True, eq=False)
class FactorCovariance:
    """Covariance with factor structure, S = B diag(f) B' + diag(s).

    ``loadings`` B is (n, k), ``factor_var`` f is (k,) and non-negative, ``specific_var`` s is
    (n,) and strictly positive, which makes S positive definite. All methods work on B, f and s
    and cost O(n k^2); only ``matrix`` forms the n x n array.
    """

    loadings: np.ndarray
    factor_var: np.ndarray
    specific_var: np.ndarray

    def __post_init__(self) -> None:
        b = np.asarray(self.loadings, dtype=float)
        f = np.asarray(self.factor_var, dtype=float)
        s = np.asarray(self.specific_var, dtype=float)
        if b.ndim != 2 or f.shape != (b.shape[1],) or s.shape != (b.shape[0],):
            raise ValueError("expected loadings (n, k), factor_var (k,) and specific_var (n,)")
        if not (np.isfinite(b).all() and np.isfinite(f).all() and np.isfinite(s).all()):
            raise ValueError("loadings, factor_var and specific_var must be finite")
        if (f < 0).any() or (s <= 0).any():
            raise ValueError("factor_var must be non-negative and specific_var strictly positive")
        for name, value in (("loadings", b), ("factor_var", f), ("specific_var", s)):
            object.__setattr__(self, name, value)

    def matrix(self) -> np.ndarray:
        """The dense (n, n) covariance B diag(f) B' + diag(s)."""
        out = (self.loadings * self.factor_var) @ self.loadings.T
        out[np.diag_indices_from(out)] += self.specific_var
        return out

    def solve(self, b: np.ndarray) -> np.ndarray:
        """S^-1 b for b of shape (n,) or (n, m), by the Woodbury identity.

        With C = B diag(sqrt(f)) and D = diag(s): S^-1 b = D^-1 b - D^-1 C (I + C' D^-1 C)^-1
        C' D^-1 b. Only a k x k system is solved; this form also allows a zero factor variance.
        """
        b = np.asarray(b, dtype=float)
        scaled = self.loadings * np.sqrt(self.factor_var)
        weighted = scaled / self.specific_var[:, None]
        core = np.eye(scaled.shape[1]) + scaled.T @ weighted
        direct = b / (self.specific_var if b.ndim == 1 else self.specific_var[:, None])
        return direct - weighted @ np.linalg.solve(core, weighted.T @ b)

    def variance(self, w: np.ndarray) -> float:
        """Portfolio variance w' S w = sum_j f_j (B' w)_j^2 + sum_i s_i w_i^2; a NaN weight is 0."""
        w = np.asarray(w, dtype=float)
        w = np.where(np.isnan(w), 0.0, w)
        exposure = self.loadings.T @ w
        return float(self.factor_var @ exposure**2 + self.specific_var @ w**2)

    def subset(self, idx: np.ndarray) -> FactorCovariance:
        """The same model restricted to the assets at positions ``idx`` (integers or a mask)."""
        return FactorCovariance(self.loadings[idx], self.factor_var, self.specific_var[idx])


def _demean_zero_fill(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns demeaned per column over the observed days, zero where missing, and the mask."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be a (T, n) array")
    observed = np.isfinite(r)
    mean = np.where(observed, r, 0.0).sum(axis=0) / np.maximum(observed.sum(axis=0), 1)
    return np.where(observed, r - mean, 0.0), observed


def _component_scores(z: np.ndarray, k: int) -> np.ndarray:
    """Scores (T, k) of the top k principal components of z (T, n), largest first.

    These are the columns u_j * sigma_j of the singular value decomposition z = U diag(sigma) V'.
    They are read off the eigenvectors of the smaller of z z' and z' z, which gives the same
    leading components as a full decomposition of z at a fraction of its cost.
    """
    t, n = z.shape
    if k == 0:
        return np.zeros((t, 0))
    if t <= n:
        value, vector = eigh(z @ z.T, subset_by_index=(t - k, t - 1))
        return vector[:, ::-1] * np.sqrt(np.clip(value[::-1], 0.0, None))
    _, vector = eigh(z.T @ z, subset_by_index=(n - k, n - 1))
    return z @ vector[:, ::-1]


def factor_covariance(
    returns: np.ndarray,
    n_factors: int = 5,
    min_obs: int = 60,
    specific_floor: float = 0.1,
) -> FactorCovariance:
    """Statistical factor model of the covariance of daily returns with missing values.

    Method. (1) Demean each column over its observed days, set its missing days to zero and
    divide it by sqrt(v_i), where v_i = sum_t x_ti^2 / (T_i - 1) is the variance of asset i over
    its T_i observed days. (2) The factor return series F (T, k) are the top ``n_factors``
    principal components of that standardised (T, n) matrix (the columns u_j * sigma_j of its
    singular value decomposition), with variances f_j = sum_t F_tj^2 / (T - 1). (3) The loadings
    of asset i are the coefficients of a regression without intercept of its standardised returns
    on F over its observed days, multiplied by sqrt(v_i) to return units. Without missing days
    these are exactly the principal-component loadings; with missing days they are not shrunk
    towards zero, as the loadings of the zero-filled matrix would be. (4) The specific variance
    is the residual variance over the observed days, sum_t e_ti^2 / (T_i - 1), floored at
    ``specific_floor`` * v_i. Assets with fewer than ``min_obs`` observations (or without any
    variation) take no part in the estimation and receive the median loading of the other assets
    on each factor and their median specific variance. Each factor is signed so that its loadings
    sum to a non-negative number. The result is positive definite because every specific variance
    is strictly positive. Fewer than ``n_factors`` factors are returned when T or the number of
    estimated assets is smaller.
    """
    if n_factors < 0 or not specific_floor > 0:
        raise ValueError("n_factors must be non-negative and specific_floor positive")
    x, observed = _demean_zero_fill(returns)
    t, n = x.shape
    count = observed.sum(axis=0)
    total_var = (x**2).sum(axis=0) / np.maximum(count - 1, 1)
    estimated = np.flatnonzero((count >= max(min_obs, 2)) & (total_var > 0))
    if estimated.size == 0:
        raise ValueError("no asset has min_obs observations and a positive variance")
    seen = observed[:, estimated]
    var = total_var[estimated]
    z = x[:, estimated] / np.sqrt(var)

    k = min(n_factors, t, estimated.size)
    factors = _component_scores(z, k)
    factor_var = (factors**2).sum(axis=0) / (t - 1)

    # One k x k regression per asset: its Gram matrix sums F_t F_t' over its observed days only.
    products = (factors[:, :, None] * factors[:, None, :]).reshape(t, k * k)
    gram = (seen.T.astype(float) @ products).reshape(estimated.size, k, k)
    beta = np.einsum("ijl,il->ij", np.linalg.pinv(gram, hermitian=True), z.T @ factors)
    residual = np.where(seen, z - factors @ beta.T, 0.0)
    # In standardised units the total variance is 1, so the floor is specific_floor itself.
    residual_share = np.maximum((residual**2).sum(axis=0) / (count[estimated] - 1), specific_floor)

    estimated_loadings = beta * np.sqrt(var)[:, None]
    estimated_loadings *= np.where(estimated_loadings.sum(axis=0) < 0, -1.0, 1.0)
    loadings = np.tile(np.median(estimated_loadings, axis=0), (n, 1))
    loadings[estimated] = estimated_loadings
    specific_var = np.full(n, np.median(residual_share * var))
    specific_var[estimated] = residual_share * var
    return FactorCovariance(loadings, factor_var, specific_var)


def ledoit_wolf_covariance(returns: np.ndarray) -> np.ndarray:
    """Dense Ledoit-Wolf covariance (n, n) of the demeaned, zero-filled returns.

    The estimator is (1 - d) * X'X / T + d * trace(X'X / T) / n * I with the shrinkage intensity d
    chosen by ``sklearn.covariance.LedoitWolf``. Meant for small universes and tests: it divides
    by T, so the variance of an asset with missing days is understated by its share of missing
    days. Use ``factor_covariance`` for panels with gaps.
    """
    x, _ = _demean_zero_fill(returns)
    return LedoitWolf(assume_centered=True).fit(x).covariance_
