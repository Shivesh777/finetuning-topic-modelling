"""Statistical inference for return series, forecast losses and strategy comparisons.

Conventions
-----------
* Returns are simple per-period returns in decimal units (0.01 is one percent), 1-D, and
  already in excess of the risk-free rate unless a function says otherwise.
* periods_per_year is 12, 26, 52 or 252. Annualised mean = mean * periods_per_year,
  annualised volatility = std(ddof=1) * sqrt(periods_per_year), and the annualised Sharpe
  ratio is their ratio, that is the per-period ratio times sqrt(periods_per_year).
  Functions that work in per-period units say so in their docstring.
* Skewness and kurtosis are the moment estimators m3 / m2^1.5 and m4 / m2^2 with
  m_k = mean((x - mean(x))^k). Kurtosis is not in excess of 3 (a normal has 3).
* Inputs are numpy arrays or pandas Series and are used by position: there is no index
  alignment. Rows in which any input is NaN are dropped before anything else, so the
  corrections for serial correlation treat the remaining observations as consecutive.
* Degenerate data (too few observations, zero variance) give NaN, not an error. Invalid
  arguments (shapes, negative lags, block < 1) raise ValueError, and so does
  reality_check with fewer than three complete rows.
* Signs: a positive mean, t statistic or Sharpe difference favours the first argument.
  For losses (diebold_mariano) a negative statistic favours the first argument.
* Small samples. A Newey-West standard error estimated from a few dozen periods is too
  small and too variable for normal critical values: with 21 observations a nominal 5
  percent test rejects 10 percent of the time. t statistics are therefore judged by the
  fixed-b critical values of Kiefer and Vogelsang (2005): critical_value gives the 5
  percent two-sided value (2.39 for 21 observations, 2.11 for 81, 1.96 in the limit) and
  p_value the matching p-value. p-values are two-sided unless they come from a bootstrap
  or a function says otherwise.
* Randomness enters only through the seed argument: equal seeds give equal results.
"""
from __future__ import annotations

from math import floor, log, sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri


def complete_cases(*arrays: ArrayLike) -> tuple[np.ndarray, ...]:
    """Float copies of the inputs restricted to the positions where none of them is NaN.

    All inputs must be 1-D and of equal length; pandas objects are used by position.
    """
    out = [np.asarray(a, dtype=float) for a in arrays]
    if any(a.ndim != 1 for a in out) or len({a.size for a in out}) > 1:
        raise ValueError("inputs must be 1-D and of equal length")
    keep = ~np.isnan(out[0])
    for a in out[1:]:
        keep &= ~np.isnan(a)
    return tuple(a[keep] for a in out)


def sample_moments(x: ArrayLike) -> tuple[int, float, float, float, float]:
    """Number of observations, mean, standard deviation, skewness and kurtosis of x.

    The standard deviation uses ddof=1; skewness m3 / m2^1.5 and kurtosis m4 / m2^2 use
    m_k = mean((x - mean)^k). The standard deviation is NaN with fewer than two
    observations and exactly 0 for a constant series; skewness and kurtosis are NaN in
    both cases.
    """
    (x,) = complete_cases(x)
    n = x.size
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    mean = float(x.mean())
    if n < 2:
        return n, mean, np.nan, np.nan, np.nan
    if x.max() == x.min():
        return n, mean, 0.0, np.nan, np.nan
    dev = x - mean
    dev2 = dev * dev
    m2 = float(dev2.sum()) / n
    skew = float(dev2 @ dev) / n / m2 ** 1.5
    kurt = float(dev2 @ dev2) / n / m2 ** 2
    return n, mean, sqrt(m2 * n / (n - 1)), skew, kurt


def _ratio(num: float, den: float) -> float:
    """num / den, or NaN when den is not strictly positive (including NaN)."""
    return num / den if den > 0 else np.nan


def default_lags(n: int) -> int:
    """Rule-of-thumb truncation lag of Newey and West (1994): floor(4 * (n / 100) ** (2 / 9))."""
    return int(floor(4 * (n / 100) ** (2 / 9)))


def _resolve_lags(lags: int | None, n: int) -> int:
    """Truncation lag that is used: default_lags(n) when lags is None, never above n - 1."""
    if lags is None:
        lags = default_lags(n)
    if lags < 0:
        raise ValueError("lags must be non-negative")
    return int(min(lags, n - 1))


# Kiefer and Vogelsang (2005, table 1), Bartlett kernel: the quantile of the t statistic is
# a0 + a1 b + a2 b^2 + a3 b^3 with b = (lags + 1) / n. Columns: 90, 95, 97.5, 99 percent.
_FIXED_B = np.array([[1.2816, 1.6449, 1.9600, 2.3263],
                     [1.3040, 2.1859, 2.9694, 4.1618],
                     [0.5135, 0.3142, 0.4160, 0.5368],
                     [-0.3386, -0.3427, -0.5324, -0.9060]])


def _fixed_b_quantiles(n: int, lags: int | None) -> np.ndarray:
    """90, 95, 97.5 and 99 percent quantiles of a Newey-West t statistic of n observations."""
    b = (_resolve_lags(lags, n) + 1) / n
    return np.array([1.0, b, b * b, b ** 3]) @ _FIXED_B


def critical_value(n: int, lags: int | None = None) -> float:
    """|t| beyond which a Newey-West t statistic is significant at 5 percent, two-sided.

    Kiefer and Vogelsang (2005): with the bandwidth a fixed share b = (lags + 1) / n of the
    sample, the 97.5 percent quantile is 1.96 + 2.9694 b + 0.4160 b^2 - 0.5324 b^3 for the
    Bartlett kernel. NaN with fewer than two observations.
    """
    return float(_fixed_b_quantiles(n, lags)[2]) if n >= 2 else np.nan


def p_value(t: float, n: int, lags: int | None = None, two_sided: bool = True) -> float:
    """p-value of a Newey-West t statistic of n observations under fixed-b asymptotics.

    |t| is mapped to the normal deviate with the same tail probability by linear
    interpolation through the tabulated quantiles (and through the origin; beyond the 99
    percent quantile the last ratio is kept), so the p-value is exact at 10, 5, 2.5 and 1
    percent per tail and close in between. two_sided=False gives the probability of a
    larger t (the test against a positive mean). NaN for a NaN t.
    """
    if np.isnan(t) or n < 2:
        return np.nan
    fixed = np.r_[0.0, _fixed_b_quantiles(n, lags)]
    normal = np.r_[0.0, _FIXED_B[0]]
    size = abs(t)
    z = np.interp(size, fixed, normal) if size <= fixed[-1] else size * normal[-1] / fixed[-1]
    return float(2 * ndtr(-z)) if two_sided else float(ndtr(-z if t > 0 else z))


def _bartlett_sum(h: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-weighted sum of the cross products of the rows of h, shape (k, k).

    S = sum_t h_t h_t' + sum_{l=1..L} (1 - l / (L + 1)) * sum_t (h_t h_{t-l}' + h_{t-l} h_t').
    """
    total = h.T @ h
    for lag in range(1, lags + 1):
        cross = h[lag:].T @ h[:-lag]
        total += (1 - lag / (lags + 1)) * (cross + cross.T)
    return total


def newey_west_variance(x: ArrayLike, lags: int | None = None) -> float:
    """Newey-West variance of the sample mean of x.

    (g_0 + 2 * sum_{l=1..L} (1 - l / (L + 1)) * g_l) / n with the autocovariances
    g_l = (1 / n) * sum_t (x_t - xbar) * (x_{t-l} - xbar). lags=None uses default_lags(n)
    and L never exceeds n - 1. NaN with fewer than two observations and exactly 0 for a
    constant series.
    """
    (x,) = complete_cases(x)
    n = x.size
    if n < 2:
        return np.nan
    if x.max() == x.min():
        return 0.0
    dev = (x - x.mean())[:, None]
    return float(_bartlett_sum(dev, _resolve_lags(lags, n))[0, 0]) / n ** 2


def mean_test(x: ArrayLike, lags: int | None = None) -> dict:
    """Test that the mean of x is zero with a Newey-West standard error.

    Returns mean, se = sqrt(newey_west_variance(x, lags)), t = mean / se, the two-sided
    p-value p = p_value(t, n, lags), critical = critical_value(n, lags) (|t| above it is
    significant at 5 percent) and n. t and p are NaN when x is constant (se is zero).
    """
    (x,) = complete_cases(x)
    n = x.size
    mean = float(x.mean()) if n else np.nan
    se = sqrt(newey_west_variance(x, lags)) if n >= 2 else np.nan
    t = _ratio(mean, se)
    return {"mean": mean, "se": se, "t": t, "p": p_value(t, n, lags),
            "critical": critical_value(n, lags), "n": n}


def ols_hac(y: ArrayLike, X: ArrayLike, lags: int | None = None) -> dict:
    """Ordinary least squares of y on X with Newey-West standard errors.

    coef = (X'X)^-1 X'y and cov(coef) = (X'X)^-1 S (X'X)^-1, where S is the sum of
    (1 - |t - s| / (L + 1)) * h_t h_s' over |t - s| <= L with h_t = x_t * resid_t. There is
    no small-sample correction, so lags=0 is White's HC0 estimator. X is used as given:
    add a column of ones for an intercept. Rows with a NaN in y or X are dropped and
    resid refers to the remaining rows. Returns coef, se, t = coef / se and resid; a
    rank-deficient X gives NaN throughout.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("y must be 1-D and X must have one row per element of y")
    keep = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y, X = y[keep], X[keep]
    n, k = X.shape
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < k:
        return {"coef": np.full(k, np.nan), "se": np.full(k, np.nan), "t": np.full(k, np.nan),
                "resid": np.full(n, np.nan)}
    resid = y - X @ coef
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ _bartlett_sum(X * resid[:, None], _resolve_lags(lags, n)) @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, coef / se, np.nan)
    return {"coef": coef, "se": se, "t": t, "resid": resid}


def default_block(n: int) -> float:
    """Default mean block length of the stationary bootstrap: max(1, n ** (1 / 3))."""
    return max(1.0, n ** (1 / 3))


def stationary_bootstrap(n: int, n_boot: int, block: float, seed: int) -> np.ndarray:
    """Index arrays of the stationary bootstrap of Politis and Romano (1994).

    Returns integers in [0, n) of shape (n_boot, n). In every row the first position is a
    uniformly random index; each later position starts a new block at a uniformly random
    index with probability 1 / block and otherwise takes the previous index plus one,
    wrapping from n - 1 to 0. Block lengths are geometric with mean block, and block=1 is
    the ordinary bootstrap.
    """
    if n < 1 or block < 1:
        raise ValueError("n and block must be at least 1")
    rng = np.random.default_rng(seed)
    position = np.arange(n)
    starts = rng.random((n_boot, n)) < 1.0 / block
    # position at which the block covering each position began (0 before the first start)
    origin = np.maximum.accumulate(np.where(starts, position, 0), axis=1)
    index = np.take_along_axis(rng.integers(0, n, size=(n_boot, n)), origin, axis=1)
    index += position - origin
    index %= n
    return index


def _resolve_block(block: float | None, n: int) -> float:
    """Mean block length that is used: default_block(n) when block is None."""
    return default_block(n) if block is None else block


def _resample_shares(index: np.ndarray) -> np.ndarray:
    """Share of every observation in each resample described by index, shape (n_boot, n).

    Row b times a (n, m) matrix gives the column means of resample b, which avoids
    materialising n_boot resampled copies of the matrix.
    """
    n_boot, n = index.shape
    cell = index + n * np.arange(n_boot)[:, None]
    return np.bincount(cell.ravel(), minlength=n_boot * n).reshape(n_boot, n) / n


def _row_sharpes(draws: np.ndarray) -> np.ndarray:
    """Per-period Sharpe ratio (ddof=1) of every row of draws; NaN for a constant row."""
    constant = draws.max(axis=1) == draws.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(constant, np.nan, draws.mean(axis=1) / draws.std(axis=1, ddof=1))


def _sharpe_variance(sr: float, skew: float, kurt: float) -> float:
    """n times the asymptotic variance of an estimated per-period Sharpe ratio (Mertens 2002).

    1 + SR^2 / 2 - skew * SR + (kurt - 3) / 4 * SR^2 = 1 - skew * SR + (kurt - 1) / 4 * SR^2,
    which cannot be negative because kurt >= 1 + skew^2; rounding below zero is clipped.
    """
    value = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    return 0.0 if value < 0 else value


def sharpe_se(r: ArrayLike, periods_per_year: float) -> float:
    """Annualised standard error of the Sharpe ratio without assuming normality.

    Mertens (2002): sqrt((1 + SR^2 / 2 - skew * SR + (kurt - 3) / 4 * SR^2) / n) with the
    per-period Sharpe ratio SR, times sqrt(periods_per_year). Assumes no serial correlation.
    """
    n, mean, sd, skew, kurt = sample_moments(r)
    if not sd > 0:
        return np.nan
    return sqrt(_sharpe_variance(mean / sd, skew, kurt) / n * periods_per_year)


def sharpe_ci(r: ArrayLike, periods_per_year: float, level: float = 0.95, n_boot: int = 2000,
              block: float | None = None, seed: int = 0) -> tuple[float, float]:
    """Percentile confidence interval of the annualised Sharpe ratio.

    The Sharpe ratio is recomputed on n_boot stationary-bootstrap resamples (mean block
    length block, default_block(n) if None) and the interval runs from the (1 - level) / 2
    to the (1 + level) / 2 quantile of those values.
    """
    (r,) = complete_cases(r)
    if r.size < 2 or r.max() == r.min():
        return np.nan, np.nan
    index = stationary_bootstrap(r.size, n_boot, _resolve_block(block, r.size), seed)
    draws = _row_sharpes(r[index]) * sqrt(periods_per_year)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return np.nan, np.nan
    low, high = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def sharpe_difference(r1: ArrayLike, r2: ArrayLike, periods_per_year: float, n_boot: int = 2000,
                      block: float | None = None, seed: int = 0) -> dict:
    """Bootstrap test that two return series have the same Sharpe ratio.

    diff = annualised SR(r1) - SR(r2). The two series are resampled with the same
    stationary-bootstrap indices, which keeps their correlation; with d_b the difference in
    resample b, se = std(d_b, ddof=1), ci = the 2.5 and 97.5 percent quantiles of d_b, and
    p = share of |d_b - diff| >= |diff|, the two-sided p-value of equal Sharpe ratios from
    the centred bootstrap distribution.
    """
    a, b = complete_cases(r1, r2)
    out = {"diff": np.nan, "se": np.nan, "ci": (np.nan, np.nan), "p": np.nan}
    _, mean_a, sd_a, _, _ = sample_moments(a)
    _, mean_b, sd_b, _, _ = sample_moments(b)
    if not (sd_a > 0 and sd_b > 0):
        return out
    scale = sqrt(periods_per_year)
    diff = (mean_a / sd_a - mean_b / sd_b) * scale
    out["diff"] = diff
    index = stationary_bootstrap(a.size, n_boot, _resolve_block(block, a.size), seed)
    draws = (_row_sharpes(a[index]) - _row_sharpes(b[index])) * scale
    draws = draws[np.isfinite(draws)]
    if draws.size < 2:
        return out
    low, high = np.quantile(draws, [0.025, 0.975])
    out["se"] = float(draws.std(ddof=1))
    out["ci"] = (float(low), float(high))
    out["p"] = float(np.mean(np.abs(draws - diff) >= abs(diff)))
    return out


def sharpe_difference_hac(r1: ArrayLike, r2: ArrayLike, periods_per_year: float,
                          lags: int | None = None) -> dict:
    """Ledoit and Wolf (2008) test that two return series of the same periods have the same
    Sharpe ratio, with a HAC standard error from the delta method.

    With v = (mean_1, mean_2, E[r_1^2], E[r_2^2]) and f(v) = mean_1 / sqrt(E[r_1^2] - mean_1^2)
    - mean_2 / sqrt(E[r_2^2] - mean_2^2), the per-period difference d = f(v_hat) has
    variance grad' Psi grad / n, where Psi is the Bartlett (Newey-West) long-run covariance
    of y_t = (r_1t - mean_1, r_2t - mean_2, r_1t^2 - E[r_1^2], r_2t^2 - E[r_2^2]) with lags
    default_lags(n) when None. Returns diff and se annualised by sqrt(periods_per_year),
    t = diff / se, the two-sided p-value under fixed-b critical values (p_value) and n.
    Ledoit and Wolf use a prewhitened quadratic-spectral kernel; the Bartlett kernel keeps
    this consistent with mean_test, and the fixed-b p-value allows for short samples.
    """
    a, b = complete_cases(r1, r2)
    n = a.size
    out = {"diff": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": n}
    if n < 3:
        return out
    m1, m2, g1, g2 = a.mean(), b.mean(), (a * a).mean(), (b * b).mean()
    v1, v2 = g1 - m1 * m1, g2 - m2 * m2
    if not (v1 > 0 and v2 > 0):
        return out
    grad = np.array([g1 / v1 ** 1.5, -g2 / v2 ** 1.5, -0.5 * m1 / v1 ** 1.5, 0.5 * m2 / v2 ** 1.5])
    y = np.column_stack([a - m1, b - m2, a * a - g1, b * b - g2])
    psi = _bartlett_sum(y, _resolve_lags(lags, n)) / n
    variance = float(grad @ psi @ grad) / n
    scale = sqrt(periods_per_year)
    diff = (m1 / sqrt(v1) - m2 / sqrt(v2)) * scale
    out["diff"] = float(diff)
    if not variance > 0:
        return out
    se = sqrt(variance) * scale
    t = diff / se
    out.update({"se": se, "t": t, "p": p_value(t, n, lags)})
    return out


def probabilistic_sharpe(r: ArrayLike, benchmark_sr: float = 0.0) -> float:
    """Probability that the true Sharpe ratio exceeds benchmark_sr (Bailey and Lopez de Prado 2012).

    Phi((SR - SR*) * sqrt(n - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2)), where SR and
    the benchmark SR* are per-period Sharpe ratios (not annualised).
    """
    n, mean, sd, skew, kurt = sample_moments(r)
    if not sd > 0:
        return np.nan
    sr = mean / sd
    variance = _sharpe_variance(sr, skew, kurt)
    if not variance > 0:
        return np.nan
    return float(ndtr((sr - benchmark_sr) * sqrt(n - 1) / sqrt(variance)))


def expected_max_sharpe(n_trials: float, sd_trials: float) -> float:
    """Expected maximum Sharpe ratio of n_trials unskilled strategies.

    sd_trials * ((1 - g) * Phi^-1(1 - 1 / N) + g * Phi^-1(1 - 1 / (N * e))) with g the
    Euler-Mascheroni constant (Bailey and Lopez de Prado 2014), in the units of sd_trials.
    Returns 0 for n_trials <= 1.
    """
    if n_trials <= 1:
        return 0.0
    g = float(np.euler_gamma)
    quantiles = (1 - g) * ndtri(1 - 1 / n_trials) + g * ndtri(1 - 1 / (n_trials * np.e))
    return float(sd_trials * quantiles)


def deflated_sharpe(r: ArrayLike, n_trials: float, sd_trials: float) -> dict:
    """Deflated Sharpe ratio: the probabilistic Sharpe ratio against the best of many trials.

    sharpe is the per-period Sharpe ratio of r, threshold = expected_max_sharpe(n_trials,
    sd_trials) and dsr = probabilistic_sharpe(r, threshold). sd_trials is the standard
    deviation of the per-period Sharpe ratios across the trials that were run.
    """
    _, mean, sd, _, _ = sample_moments(r)
    threshold = expected_max_sharpe(n_trials, sd_trials)
    return {"sharpe": _ratio(mean, sd), "threshold": threshold,
            "dsr": probabilistic_sharpe(r, threshold)}


def diebold_mariano(loss_a: ArrayLike, loss_b: ArrayLike, lags: int | None = None) -> dict:
    """Diebold-Mariano test of equal expected loss of two forecasts.

    With the loss differential d_t = loss_a_t - loss_b_t of each period, mean_diff = mean(d),
    stat = mean(d) / sqrt(newey_west_variance(d, lags)) and p is the two-sided p-value of
    mean_test. A negative stat means that A has the lower loss.
    """
    a, b = complete_cases(loss_a, loss_b)
    test = mean_test(a - b, lags)
    return {"mean_diff": test["mean"], "stat": test["t"], "p": test["p"]}


def reality_check(returns: ArrayLike, benchmark: ArrayLike | None = None, n_boot: int = 2000,
                  block: float | None = None, seed: int = 0) -> dict:
    """White's Reality Check and Hansen's test for superior predictive ability (SPA).

    returns is (T, m) with one column per strategy, d = returns - benchmark[:, None]
    (benchmark zero if None) after dropping rows with a NaN, dbar_k = mean(d_k), and d* is
    a stationary-bootstrap resample of the rows of d. The null hypothesis is that no
    strategy has a positive expected d.

    White (2000): V = max_k sqrt(T) * dbar_k, compared with the bootstrap distribution of
    V* = max_k sqrt(T) * (mean(d*_k) - dbar_k).

    Hansen (2005), SPA_c: omega_k^2 = mean over the resamples of
    (sqrt(T) * (mean(d*_k) - dbar_k))^2, t_k = sqrt(T) * dbar_k / omega_k and the statistic
    is T_spa = max(0, max_k t_k). A strategy that is significantly worse than the benchmark
    keeps its own mean under the null, mu_k = dbar_k * 1{t_k <= -sqrt(2 * log(log(T)))},
    and the bootstrap statistic is
    T* = max(0, max_k sqrt(T) * (mean(d*_k) - dbar_k + mu_k) / omega_k).

    Each p-value is the share of resamples whose statistic is >= the observed one. Hansen
    writes >, which differs only when T_spa = 0, where >= gives p = 1. Strategies with
    omega_k = 0 are left out of the SPA test. Returns p_reality_check, p_spa, best (the
    column with the largest dbar_k), stat (V), stat_spa (T_spa) and expected_stat, the mean
    of V* over the resamples: what the best of these strategies is expected to show when
    none has a positive mean, given how they move together.

    With every column divided by its standard deviation, V / sqrt(T) is the best per-period
    Sharpe ratio and expected_stat / sqrt(T) the best Sharpe ratio expected without skill.
    """
    d = np.asarray(returns, dtype=float)
    if d.ndim != 2:
        raise ValueError("returns must be a (T, m) array")
    if benchmark is not None:
        d = d - np.asarray(benchmark, dtype=float)[:, None]
    d = d[~np.isnan(d).any(axis=1)]
    t, m = d.shape
    if t < 3 or m < 1:
        raise ValueError("reality_check needs at least 3 complete rows and 1 column")
    root = sqrt(t)
    dbar = d.mean(axis=0)
    index = stationary_bootstrap(t, n_boot, _resolve_block(block, t), seed)
    centred = root * (_resample_shares(index) @ (d - dbar))
    stat = root * float(dbar.max())
    p_rc = float(np.mean(centred.max(axis=1) >= stat))

    omega = np.sqrt(np.mean(centred * centred, axis=0))
    ok = omega > 0
    stat_spa, p_spa = np.nan, np.nan
    if ok.any():
        t_k = root * dbar[ok] / omega[ok]
        stat_spa = max(0.0, float(t_k.max()))
        mu = np.where(t_k <= -sqrt(2 * log(log(t))), root * dbar[ok], 0.0)
        boot = np.maximum(((centred[:, ok] + mu) / omega[ok]).max(axis=1), 0.0)
        p_spa = float(np.mean(boot >= stat_spa))
    return {"p_reality_check": p_rc, "p_spa": p_spa, "best": int(np.argmax(dbar)),
            "stat": stat, "stat_spa": stat_spa,
            "expected_stat": float(centred.max(axis=1).mean())}


def benjamini_hochberg(p: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, which control the false discovery rate.

    With the m non-NaN p-values sorted in ascending order, the adjusted value of the i-th
    is min(1, min_{j >= i} m * p_(j) / j). NaN entries stay NaN and do not count towards m;
    the output has the shape of the input and is monotone in p.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    values = p[valid]
    m = values.size
    order = np.argsort(values)
    scaled = values[order] * (m / np.arange(1, m + 1))     # factor >= 1: never below p
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    out[valid] = adjusted
    return out
