"""Performance metrics for return series and accuracy metrics for stock-level forecasts.

Conventions
-----------
* Returns are simple per-period returns in decimal units, 1-D, and already in excess of
  the risk-free rate unless a function says otherwise.
* Annualisation uses periods_per_year (see PERIODS_PER_YEAR): mean * periods_per_year,
  std(ddof=1) * sqrt(periods_per_year), and the Sharpe ratio is their ratio. In
  performance the mean, volatility, Sharpe ratio and its standard error, alpha, active
  mean, tracking error and information ratio are annualised; worst, best and the
  drawdowns are per-period or cumulative figures.
* Drawdowns are non-positive: -0.25 is a fall of 25 percent from the previous peak.
* Panel functions take one row per stock and period: y is the realised return, pred the
  forecast made for it and groups the period label of the row. Rows need not be sorted.
* Inputs are numpy arrays or pandas Series and are used by position: there is no index
  alignment. Rows in which any input is NaN (or the group label is missing) are dropped
  before anything else, so outputs with one value per observation refer to the rows kept.
* Degenerate data (too few observations, zero variance) give NaN, not an error; invalid
  arguments (shapes, q < 1) raise ValueError.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ftm.evaluation.inference import (complete_cases, mean_test, ols_hac, sample_moments,
                                      sharpe_se)

PERIODS_PER_YEAR = {"M": 12, "2W": 26, "W": 52, "D": 252}

_STATS = ("n", "mean", "vol", "sharpe", "sharpe_se", "t_mean", "p_mean", "skew",
          "excess_kurtosis", "max_drawdown", "calmar", "hit_rate", "worst", "best")
_RELATIVE = ("alpha", "beta", "t_alpha", "active_mean", "tracking_error", "information_ratio")
_EXACT = 1e-9          # a residual volatility below this share of the volatility is rounding


def sharpe(r: ArrayLike, periods_per_year: float) -> float:
    """Annualised Sharpe ratio mean(r) / std(r, ddof=1) * sqrt(periods_per_year).

    NaN with fewer than two observations or zero variance.
    """
    _, mean, sd, _, _ = sample_moments(r)
    return mean / sd * sqrt(periods_per_year) if sd > 0 else np.nan


def drawdown(r: ArrayLike) -> np.ndarray:
    """Drawdown of the wealth path W_t = cumprod(1 + r) from its running peak.

    drawdown_t = W_t / max(1, max_{s<=t} W_s) - 1, so the starting wealth of 1 counts as a
    peak and every value is <= 0. One value per non-NaN return.
    """
    (r,) = complete_cases(r)
    wealth = np.cumprod(1 + r)
    return wealth / np.maximum(np.maximum.accumulate(wealth), 1.0) - 1


def max_drawdown(r: ArrayLike) -> float:
    """Largest peak-to-trough loss: the minimum of drawdown(r), a non-positive number."""
    path = drawdown(r)
    return float(path.min()) if path.size else np.nan


def performance(r: ArrayLike, periods_per_year: float, benchmark: ArrayLike | None = None,
                lags: int | None = None) -> dict:
    """Summary statistics of a return series, optionally relative to a benchmark.

    Always: n, mean and vol (annualised), sharpe, sharpe_se (Mertens), t_mean and p_mean
    (Newey-West test of a zero mean), skew, excess_kurtosis, max_drawdown, calmar
    (annualised mean / |max_drawdown|, NaN if there is no drawdown), hit_rate (share of
    r > 0), worst and best (per-period extremes). With a benchmark of the same length
    also: alpha (intercept of the OLS of r on a constant and the benchmark, annualised),
    beta, t_alpha (Newey-West), active_mean and tracking_error (annualised mean and
    volatility of r - benchmark) and information_ratio (their ratio). Degenerate cases
    give NaN instead of rounding noise: t_alpha when r is an exact linear function of the
    benchmark, information_ratio when r is the benchmark plus a constant, alpha and beta
    as well when r or the benchmark is constant. Periods in which r or the benchmark is
    NaN are dropped from every statistic.
    """
    names = _STATS if benchmark is None else _STATS + _RELATIVE
    if benchmark is None:
        (r,) = complete_cases(r)
    else:
        r, bench = complete_cases(r, benchmark)
    n, mean, sd, skew, kurt = sample_moments(r)
    out = dict.fromkeys(names, np.nan)
    out["n"] = n
    if n == 0:
        return out
    test = mean_test(r, lags)
    trough = max_drawdown(r)
    annual = sqrt(periods_per_year)
    out.update(
        mean=mean * periods_per_year, vol=sd * annual, sharpe=sharpe(r, periods_per_year),
        sharpe_se=sharpe_se(r, periods_per_year), t_mean=test["t"], p_mean=test["p"],
        skew=skew, excess_kurtosis=kurt - 3, max_drawdown=trough,
        calmar=mean * periods_per_year / -trough if trough < 0 else np.nan,
        hit_rate=float(np.mean(r > 0)), worst=float(r.min()), best=float(r.max()),
    )
    if benchmark is None:
        return out
    _, active, active_sd, _, _ = sample_moments(r - bench)
    out.update(active_mean=active * periods_per_year, tracking_error=active_sd * annual)
    if active_sd > _EXACT * sd:
        out["information_ratio"] = active / active_sd * annual
    fit = ols_hac(r, np.column_stack([np.ones(n), bench]), lags)
    alpha, beta = (float(c) for c in fit["coef"])
    if np.isnan(alpha) or not sd > 0:          # constant benchmark or constant r: no regression
        return out
    if np.std(fit["resid"]) > _EXACT * sd:
        out.update(alpha=alpha * periods_per_year, beta=beta, t_alpha=float(fit["t"][0]))
    else:                 # r is a + b * benchmark up to rounding: coefficients, but no test
        out.update(alpha=round(alpha, 12) * periods_per_year + 0.0, beta=round(beta, 12) + 0.0)  # no -0.0
    return out


def r2_oos(y: ArrayLike, pred: ArrayLike) -> float:
    """Out-of-sample R squared against a zero forecast: 1 - sum((y - pred)^2) / sum(y^2).

    The denominator is not demeaned (Gu, Kelly and Xiu 2020), so a forecast of zero scores
    exactly 0 and the historical mean is not given for free.
    """
    y, pred = complete_cases(y, pred)
    total = float(y @ y)
    if not total > 0:
        return np.nan
    error = y - pred
    return 1 - float(error @ error) / total


def _grouped(groups: ArrayLike,
             *arrays: ArrayLike) -> tuple[np.ndarray, pd.Index, list[np.ndarray]]:
    """Group codes, sorted unique group labels and float arrays without incomplete rows.

    codes[i] is the position of the group of row i in the labels. Rows with a NaN in any
    array or a missing label are removed; labels keeps every label seen in groups.
    """
    codes, labels = pd.Index(groups).factorize(sort=True)
    values = [np.asarray(a, dtype=float) for a in arrays]
    if any(v.ndim != 1 or v.size != codes.size for v in values):
        raise ValueError("inputs must be 1-D and as long as groups")
    keep = codes >= 0
    for v in values:
        keep &= ~np.isnan(v)
    return codes[keep], labels.rename(getattr(groups, "name", None)), [v[keep] for v in values]


def r2_within(y: ArrayLike, pred: ArrayLike, groups: ArrayLike) -> float:
    """Cross-sectional R squared: r2_oos after removing group means.

    The mean of y and the mean of pred inside each group are subtracted first, so only the
    ability to tell stocks apart within a period counts.
    """
    codes, labels, (y, pred) = _grouped(groups, y, pred)
    count = np.maximum(np.bincount(codes, minlength=len(labels)), 1)
    y = y - (np.bincount(codes, y, minlength=len(labels)) / count)[codes]
    pred = pred - (np.bincount(codes, pred, minlength=len(labels)) / count)[codes]
    return r2_oos(y, pred)


def _order_within(codes: np.ndarray, x: np.ndarray, count: np.ndarray,
                  stable: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows sorted by group and then by x: their order, group and position inside the group.

    count is the number of rows per group and positions start at 0. With stable=True equal
    values of x keep their input order. x is sorted once; the stable sort of the group
    codes that follows is a linear-time radix sort in numpy when the codes fit in 16 bits.
    """
    order = np.argsort(x, kind="stable" if stable else None)
    key = codes[order]
    if count.size <= 2 ** 16:
        key = key.astype(np.uint16)
    order = order[np.argsort(key, kind="stable")]
    group = np.repeat(np.arange(count.size), count)
    position = np.arange(group.size) - np.repeat(np.cumsum(count) - count, count)
    return order, group, position


def _average_ranks(codes: np.ndarray, x: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Rank of x inside its group, starting at 0, with ties sharing their average rank."""
    order, _, position = _order_within(codes, x, count, stable=False)
    value = x[order]
    new = position == 0                              # first row of each run of tied values
    new[1:] |= value[1:] != value[:-1]
    first = np.flatnonzero(new)
    length = np.diff(np.append(first, value.size))
    ranks = np.empty(value.size)
    ranks[order] = np.repeat(position[first] + (length - 1) / 2, length)
    return ranks


def rank_ic(y: ArrayLike, pred: ArrayLike, groups: ArrayLike) -> pd.Series:
    """Rank information coefficient of every group.

    Spearman correlation between y and pred inside each group: the Pearson correlation of
    the within-group average ranks. Indexed by the sorted unique group labels; NaN for a
    group with fewer than 3 complete rows or with a constant y or pred.
    """
    codes, labels, (y, pred) = _grouped(groups, y, pred)
    k = len(labels)
    count = np.bincount(codes, minlength=k)
    centre = (count[codes] - 1) / 2                  # mean rank of the group
    a = _average_ranks(codes, y, count) - centre
    b = _average_ranks(codes, pred, count) - centre
    cov = np.bincount(codes, a * b, minlength=k)
    var_a = np.bincount(codes, a * a, minlength=k)
    var_b = np.bincount(codes, b * b, minlength=k)
    ok = (count >= 3) & (var_a > 0) & (var_b > 0)
    ic = np.full(k, np.nan)
    ic[ok] = cov[ok] / np.sqrt(var_a[ok] * var_b[ok])
    return pd.Series(ic, index=labels)


def ic_summary(ic: ArrayLike, lags: int | None = None) -> dict:
    """Summary of a series of information coefficients.

    n, mean, std (ddof=1), t = mean / Newey-West standard error, icir = mean / std (not
    annualised) and share_positive = share of ic > 0, all over the non-NaN values.
    """
    (ic,) = complete_cases(ic)
    n, mean, sd, _, _ = sample_moments(ic)
    return {"n": n, "mean": mean, "std": sd, "t": mean_test(ic, lags)["t"],
            "icir": mean / sd if sd > 0 else np.nan,
            "share_positive": float(np.mean(ic > 0)) if n else np.nan}


def quantile_returns(y: ArrayLike, pred: ArrayLike, groups: ArrayLike, q: int = 10,
                     weights: ArrayLike | None = None) -> pd.DataFrame:
    """Mean realised y in each prediction quantile of each group.

    Inside a group the rows are ranked by pred from 0 (ties in input order) and a row
    belongs to quantile floor(rank * q / n_group) + 1, so column 1 holds the lowest
    predictions and column q the highest. Each cell is the mean of y, or the average
    weighted by the non-negative weights (NaN if they sum to zero). Indexed by group label;
    groups with fewer than q complete rows are dropped.
    """
    if q < 1:
        raise ValueError("q must be at least 1")
    arrays = (y, pred) if weights is None else (y, pred, weights)
    codes, labels, values = _grouped(groups, *arrays)
    y, pred = values[0], values[1]
    w = np.ones(y.size) if weights is None else values[2]
    k = len(labels)
    count = np.bincount(codes, minlength=k)
    order, group, position = _order_within(codes, pred, count, stable=True)
    cell = group * q + position * q // count[group]
    total = np.bincount(cell, (w * y)[order], minlength=k * q).reshape(k, q)
    weight = np.bincount(cell, w[order], minlength=k * q).reshape(k, q)
    keep = count >= q
    with np.errstate(divide="ignore", invalid="ignore"):
        means = total[keep] / weight[keep]
    return pd.DataFrame(means, index=labels[keep], columns=np.arange(1, q + 1))


def monotonicity(quantile_means: ArrayLike) -> float:
    """Spearman correlation between the quantile number and the mean return of the quantile.

    +1 when the means rise from the first to the last quantile and -1 when they fall. A
    2-D table (periods by quantiles, as returned by quantile_returns) is averaged over the
    periods first. NaN means are skipped; NaN with fewer than two or with constant means.
    """
    means = np.asarray(quantile_means, dtype=float)
    if means.ndim == 2:
        seen = ~np.isnan(means)
        with np.errstate(invalid="ignore"):
            means = np.where(seen, means, 0.0).sum(axis=0) / seen.sum(axis=0)
    means = means[~np.isnan(means)]
    if means.size < 2 or means.max() == means.min():
        return np.nan
    number = np.arange(means.size, dtype=float)
    ranks = _average_ranks(np.zeros(means.size, dtype=np.intp), means, np.array([means.size]))
    return float(np.corrcoef(number, ranks)[0, 1])
