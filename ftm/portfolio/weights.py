"""Portfolio weights from one cross-section of scores.

Every function maps ONE cross-section to a weight vector with the length of its input.

Conventions
-----------
* ``score`` is a 1-D float array in which a higher value means a more attractive asset. A NaN
  (or infinite) score marks an asset that is not investable: it takes no part in the calculation
  and its weight is exactly 0. An asset with a NaN in its row of ``exposures`` or ``neutral`` is
  treated the same way.
* Weights are fractions of capital: positive is long, negative is short, and ``w @ r`` is the
  return per unit of capital. The sorted and ranked long-short portfolios have longs that sum to
  +1 and shorts that sum to -1 (gross exposure 2, net exposure 0); ``top_quantile`` is long only
  and sums to +1; ``mean_variance`` has the gross exposure that is asked for.
* In the sorted portfolios ties in ``score`` are broken by position: of two assets with the same
  score, the one that comes first in the array is treated as the one with the lower score.
  ``rank_weighted`` gives tied scores their average rank and so the same weight.
* Nothing is annualised. ``cov`` may be in any unit of time, because the scale of a covariance
  cancels when the mean-variance weights are scaled to the gross exposure.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata

from .risk import FactorCovariance

__all__ = ["quantile_long_short", "top_quantile", "rank_weighted", "tilted", "neutralise",
           "mean_variance"]

_CAP_PASSES = 50
_CAP_TOL = 1e-6
# A constrained solution this small relative to the unconstrained one is rounding error.
_DEGENERATE = 1e-10


def _ascending(score: np.ndarray) -> np.ndarray:
    """Positions of the finite scores, from the lowest to the highest score (stable sort)."""
    valid = np.flatnonzero(np.isfinite(score))
    return valid[np.argsort(score[valid], kind="stable")]


def _leg(members: np.ndarray, cap: np.ndarray | None) -> np.ndarray:
    """Weights of the assets at ``members`` within one leg, positive and summing to 1.

    Equal weights 1 / m, or cap_i / sum(cap) when ``cap`` is given. A cap that is missing or not
    positive is replaced by the median of the usable caps of the leg; a leg without any usable
    cap is equally weighted.
    """
    if cap is not None:
        value = np.asarray(cap, dtype=float)[members]
        usable = np.isfinite(value) & (value > 0)
        if usable.any():
            value = np.where(usable, value, np.median(value[usable]))
            return value / value.sum()
    return np.full(members.size, 1.0 / members.size)


def _columns(exposures: np.ndarray, n: int) -> np.ndarray:
    """``exposures`` of shape (n,) or (n, k) as a float array of shape (n, k)."""
    x = np.asarray(exposures, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError(f"exposures must have shape ({n},) or ({n}, k)")
    return x


def quantile_long_short(
    score: np.ndarray, q: int = 10, cap: np.ndarray | None = None
) -> np.ndarray:
    """Long the top 1/q of the valid scores and short the bottom 1/q.

    Each leg holds n_leg = max(1, floor(n_valid / q)) assets, equally weighted (+1 / n_leg and
    -1 / n_leg) or weighted by ``cap`` (market value) within the leg, so that longs sum to +1 and
    shorts to -1. All zeros with fewer than 2 valid scores. With tied scores the legs are still
    filled, by position, so a constant score gives a portfolio and not zeros.
    """
    if q < 2:
        raise ValueError("q must be at least 2 for the two legs not to overlap")
    score = np.asarray(score, dtype=float)
    w = np.zeros(score.shape[0])
    order = _ascending(score)
    if order.size < 2:
        return w
    n_leg = max(1, int(order.size // q))
    w[order[-n_leg:]] = _leg(order[-n_leg:], cap)
    w[order[:n_leg]] = -_leg(order[:n_leg], cap)
    return w


def top_quantile(score: np.ndarray, q: int = 10, cap: np.ndarray | None = None) -> np.ndarray:
    """Long-only portfolio of the top 1/q of the valid scores, summing to 1.

    Holds the max(1, floor(n_valid / q)) assets with the highest score, equally weighted or
    weighted by ``cap`` (market value): the long leg of ``quantile_long_short``. q = 1 holds every
    valid asset. All zeros only when no score is valid.
    """
    if q < 1:
        raise ValueError("q must be at least 1")
    score = np.asarray(score, dtype=float)
    w = np.zeros(score.shape[0])
    order = _ascending(score)
    if order.size == 0:
        return w
    members = order[-max(1, int(order.size // q)):]
    w[members] = _leg(members, cap)
    return w


def tilted(score: np.ndarray, cap: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Long-only weights proportional to cap_i * (1 + strength * s_i), summing to 1.

    s_i is the rank of the score among the valid scores scaled to [-1, 1] (average rank for
    ties), so strength = 1 holds the highest score at twice and the lowest at zero times
    its market weight. Assets without a valid score keep their market weight (s = 0); a
    cap that is missing or not positive is replaced by the median usable cap. A constant
    score gives the market weights.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must lie in [0, 1] for the weights to stay long only")
    score = np.asarray(score, dtype=float)
    value = np.asarray(cap, dtype=float)
    usable = np.isfinite(value) & (value > 0)
    if not usable.any():
        value = np.ones(score.shape[0])
    else:
        value = np.where(usable, value, np.median(value[usable]))
    s = np.zeros(score.shape[0])
    valid = np.isfinite(score)
    if valid.sum() >= 2:
        r = rankdata(score[valid])
        s[valid] = 2.0 * (r - 1.0) / (valid.sum() - 1.0) - 1.0
    w = value * (1.0 + strength * s)
    return w / w.sum()


def rank_weighted(score: np.ndarray) -> np.ndarray:
    """Weights proportional to the rank of the score minus the mean rank.

    w_i = (rank_i - (n_valid + 1) / 2) / sum_j max(rank_j - (n_valid + 1) / 2, 0), with ranks 1
    to n_valid among the valid scores and the average rank for ties, so that longs sum to +1 and
    shorts to -1. All zeros with fewer than 2 valid scores or when all valid scores are equal.
    """
    score = np.asarray(score, dtype=float)
    w = np.zeros(score.shape[0])
    valid = np.isfinite(score)
    if valid.sum() < 2:
        return w
    rank = rankdata(score[valid], method="average")
    centred = rank - rank.mean()
    longs = centred[centred > 0].sum()
    if longs > 0:
        w[valid] = centred / longs
    return w


def neutralise(w: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Remove from ``w`` its Euclidean projection on the columns of ``exposures``.

    Returns w - X (X'X)^+ X'w for X = exposures of shape (n,) or (n, k), so that X'w = 0
    afterwards; the result is not rescaled, so the legs no longer sum to +1 and -1. Collinear
    columns are allowed (least squares with the pseudo-inverse). The projection moves the weight
    of every asset with a non-zero exposure, including assets that had weight 0. An asset with a
    NaN exposure or a NaN weight is left out of the projection and gets weight 0, which is the
    way to keep an asset that is not investable out of the result.
    """
    w = np.asarray(w, dtype=float)
    x = _columns(exposures, w.shape[0])
    valid = np.isfinite(w) & np.isfinite(x).all(axis=1)
    out = np.zeros(w.shape[0])
    coefficient = np.linalg.lstsq(x[valid], w[valid], rcond=None)[0]
    out[valid] = w[valid] - x[valid] @ coefficient
    return out


def _inverse_times(
    cov: np.ndarray | FactorCovariance, idx: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """S^-1 b for the covariance S of the assets at positions ``idx``."""
    if isinstance(cov, FactorCovariance):
        return cov.subset(idx).solve(b)
    dense = np.asarray(cov, dtype=float)
    return cho_solve(cho_factor(dense[np.ix_(idx, idx)], lower=True), b)


def _cap_weights(
    w: np.ndarray, gross: float, max_abs: float, exposures: np.ndarray | None
) -> np.ndarray:
    """Alternate (clip to +-max_abs, neutralise, rescale to gross) until the cap holds.

    ``w`` comes in neutral and with the right gross exposure, and neutralising and rescaling
    restore both after every clip, so the cap is the only condition left to test. If it still
    fails after the last pass, the weights are scaled down until it holds.
    """
    for _ in range(_CAP_PASSES):
        if np.abs(w).max() <= max_abs + _CAP_TOL:
            return w
        w = np.clip(w, -max_abs, max_abs)
        if exposures is not None:
            w = neutralise(w, exposures)
        total = np.abs(w).sum()
        if total == 0.0:
            return w
        w = w * (gross / total)
    peak = np.abs(w).max()
    return w if peak <= max_abs + _CAP_TOL else w * (max_abs / peak)


def mean_variance(
    score: np.ndarray,
    cov: np.ndarray | FactorCovariance,
    gross: float = 2.0,
    max_abs: float | None = None,
    neutral: np.ndarray | None = None,
) -> np.ndarray:
    """Mean-variance weights with linear neutrality constraints, scaled to a gross exposure.

    Solves min_w 0.5 w'S w - w'mu subject to N'w = 0 with mu = score, S = cov and N = neutral of
    shape (n,) or (n, k): w = S^-1 (mu - N lam), lam = (N' S^-1 N)^-1 N' S^-1 mu (lam = 0 without
    ``neutral``), and then multiplies w by gross / sum(abs(w)). ``cov`` is a dense (n, n)
    positive definite array or a ``FactorCovariance``. Assets with a NaN score or a NaN in
    ``neutral`` are excluded from the problem and get weight 0. Returns zeros when the problem is
    degenerate: fewer than 3 valid assets, all valid scores equal, or a score that lies in the
    span of ``neutral``.

    With ``max_abs``, abs(w_i) <= max_abs is enforced by repeating (clip, ``neutralise``, rescale
    to gross) for at most 50 passes or until the cap holds to 1e-6. This is a heuristic and NOT
    the exact box-constrained optimum. N'w = 0 holds on return in every case. If the passes do
    not converge, or the cap cannot be met at this gross exposure (max_abs * n_valid < gross),
    the weights are scaled down so that the cap holds exactly and the gross exposure is below
    ``gross``.
    """
    if gross <= 0 or (max_abs is not None and max_abs <= 0):
        raise ValueError("gross and max_abs must be positive")
    mu = np.asarray(score, dtype=float)
    n = mu.shape[0]
    is_factor = isinstance(cov, FactorCovariance)
    if (cov.specific_var.shape * 2 if is_factor else np.shape(cov)) != (n, n):
        raise ValueError(f"cov must be a FactorCovariance or an (n, n) array for n = {n} assets")
    out = np.zeros(n)
    valid = np.isfinite(mu)
    x = None
    if neutral is not None:
        x = _columns(neutral, n)
        valid &= np.isfinite(x).all(axis=1)
        x = x[valid]
    idx = np.flatnonzero(valid)
    if idx.size < 3 or np.ptp(mu[idx]) == 0.0:
        return out

    if x is None:
        unconstrained = w = _inverse_times(cov, idx, mu[idx])
    else:
        solved = _inverse_times(cov, idx, np.column_stack([mu[idx], x]))
        unconstrained, inverse_x = solved[:, 0], solved[:, 1:]
        lam = np.linalg.lstsq(x.T @ inverse_x, x.T @ unconstrained, rcond=None)[0]
        w = unconstrained - inverse_x @ lam
    total = np.abs(w).sum()
    if not total > _DEGENERATE * np.abs(unconstrained).sum():
        return out
    w = w * (gross / total)
    if max_abs is not None:
        w = _cap_weights(w, gross, max_abs, x)
    out[idx] = w
    return out
