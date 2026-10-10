"""Tests of ftm.evaluation (inference and metrics) on synthetic data.

Every number is checked against an independent calculation: a closed form, an explicit
loop, scipy, scikit-learn, pandas, or a simulation with a fixed seed and a wide tolerance.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats
from sklearn.metrics import r2_score

from ftm.evaluation import inference as inf
from ftm.evaluation import metrics as met

# degenerate inputs must give NaN quietly, without floating-point warnings
pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def with_gaps(x: np.ndarray, where: list[int]) -> np.ndarray:
    """Copy of x with NaN at the given positions."""
    out = np.array(x, dtype=float)
    out[where] = np.nan
    return out


def bartlett_sum_brute(h: np.ndarray, lags: int) -> np.ndarray:
    """sum over |t - s| <= lags of (1 - |t - s| / (lags + 1)) * h_t h_s', by a double loop."""
    n, k = h.shape
    total = np.zeros((k, k))
    for t in range(n):
        for s in range(max(0, t - lags), min(n, t + lags + 1)):
            total += (1 - abs(t - s) / (lags + 1)) * np.outer(h[t], h[s])
    return total


def reality_check_brute(d: np.ndarray, n_boot: int, block: float, seed: int) -> dict:
    """White's and Hansen's p-values from explicitly resampled data, one resample at a time."""
    t = d.shape[0]
    root = np.sqrt(t)
    dbar = d.mean(axis=0)
    index = inf.stationary_bootstrap(t, n_boot, block, seed)
    boot = np.array([d[rows].mean(axis=0) for rows in index])
    v_star = (root * (boot - dbar)).max(axis=1)
    omega = np.sqrt(np.mean((root * (boot - dbar)) ** 2, axis=0))
    t_k = root * dbar / omega
    mu = np.where(t_k <= -np.sqrt(2 * np.log(np.log(t))), dbar, 0.0)
    t_star = np.maximum(0.0, (root * (boot - dbar + mu) / omega).max(axis=1))
    stat_spa = max(0.0, t_k.max())
    return {"p_reality_check": np.mean(v_star >= root * dbar.max()),
            "p_spa": np.mean(t_star >= stat_spa), "stat": root * dbar.max(),
            "stat_spa": stat_spa, "recentred": int((mu != 0).sum())}


# ----------------------------------------------------------------------------- inference


def test_default_lags_and_default_block():
    assert inf.default_lags(100) == 4
    assert inf.default_lags(130) == 4
    assert inf.default_lags(600) == 5
    assert inf.default_lags(10) == 2
    assert inf.default_block(1) == 1.0
    assert inf.default_block(0) == 1.0
    assert inf.default_block(216) == pytest.approx(6.0)
    assert inf.default_block(1000) == pytest.approx(10.0)


def test_complete_cases_drops_rows_pairwise_and_validates_shapes():
    a, b = inf.complete_cases([1.0, np.nan, 3.0, 4.0], pd.Series([1.0, 2.0, np.nan, 5.0]))
    np.testing.assert_array_equal(a, [1.0, 4.0])
    np.testing.assert_array_equal(b, [1.0, 5.0])
    with pytest.raises(ValueError):
        inf.complete_cases([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        inf.complete_cases(np.ones((3, 2)))


def test_sample_moments_match_scipy():
    x = rng(0).gamma(2.0, size=200)
    n, mean, sd, skew, kurt = inf.sample_moments(x)
    assert n == 200
    assert mean == pytest.approx(x.mean())
    assert sd == pytest.approx(x.std(ddof=1))
    assert skew == pytest.approx(stats.skew(x))
    assert kurt == pytest.approx(stats.kurtosis(x, fisher=False))
    assert inf.sample_moments([])[0] == 0
    assert np.isnan(inf.sample_moments([0.3])[2])
    assert inf.sample_moments(np.full(7, 0.01))[2] == 0.0


def test_newey_west_with_zero_lags_is_variance_over_n():
    x = rng(1).standard_normal(250) * 0.04 + 0.01
    assert inf.newey_west_variance(x, lags=0) == pytest.approx(x.var(ddof=0) / x.size, rel=1e-12)


@pytest.mark.parametrize("lags", [1, 3, 7])
def test_newey_west_matches_explicit_autocovariances(lags):
    x = np.cumsum(rng(2).standard_normal(90)) * 0.1
    n = x.size
    dev = x - x.mean()
    g = [sum(dev[t] * dev[t - lag] for t in range(lag, n)) / n for lag in range(lags + 1)]
    expected = (g[0] + 2 * sum((1 - lag / (lags + 1)) * g[lag] for lag in range(1, lags + 1))) / n
    assert inf.newey_west_variance(x, lags) == pytest.approx(expected, rel=1e-12)


def test_newey_west_default_and_capped_lags():
    x = rng(3).standard_normal(130)
    assert inf.newey_west_variance(x) == inf.newey_west_variance(x, inf.default_lags(130))
    short = x[:6]
    assert inf.newey_west_variance(short, 50) == inf.newey_west_variance(short, 5)
    assert np.isnan(inf.newey_west_variance([0.1]))
    with pytest.raises(ValueError):
        inf.newey_west_variance(x, -1)


def test_newey_west_recovers_long_run_variance_of_an_ar1():
    n, phi, lags = 200_000, 0.5, 60
    x = signal.lfilter([1.0], [1.0, -phi], rng(4).standard_normal(n))
    weights = 1 - np.arange(1, lags + 1) / (lags + 1)
    # population value of the Bartlett estimator: g0 * (1 + 2 * sum_l w_l * phi^l)
    smoothed = (1 + 2 * np.sum(weights * phi ** np.arange(1, lags + 1))) / (1 - phi ** 2)
    estimate = n * inf.newey_west_variance(x, lags)
    assert estimate == pytest.approx(smoothed, rel=0.08)
    assert estimate == pytest.approx(1 / (1 - phi) ** 2, rel=0.12)   # true long-run variance 4
    assert x.var() < 0.4 * estimate                                  # the iid formula is far off


def test_mean_test_matches_hand_calculation():
    x = rng(5).standard_normal(180) * 0.03 + 0.004
    out = inf.mean_test(x, lags=3)
    se = np.sqrt(inf.newey_west_variance(x, 3))
    assert out["mean"] == pytest.approx(x.mean())
    assert out["se"] == pytest.approx(se)
    assert out["t"] == pytest.approx(x.mean() / se)
    assert out["p"] == inf.p_value(out["t"], 180, 3) and out["n"] == 180
    assert out["critical"] == inf.critical_value(180, 3)
    assert np.isnan(inf.mean_test(np.zeros(10))["t"])
    flat = inf.mean_test(np.full(6, 0.1))        # in floating point their mean is not 0.1
    assert flat["se"] == 0.0 and np.isnan(flat["t"]) and np.isnan(flat["p"])
    assert np.isnan(inf.mean_test([])["mean"])


def test_critical_values_and_p_values_allow_for_few_observations():
    # Kiefer and Vogelsang's polynomial at b = (2 + 1) / 21, and its large-sample limit.
    b = 3 / 21
    assert inf.critical_value(21, 2) == pytest.approx(1.96 + 2.9694 * b + 0.4160 * b ** 2
                                                      - 0.5324 * b ** 3)
    assert inf.critical_value(21) == inf.critical_value(21, 2)        # the default lags of n = 21
    assert inf.critical_value(10 ** 7, 0) == pytest.approx(1.96, abs=1e-5)
    assert inf.critical_value(7) > inf.critical_value(21) > inf.critical_value(81) > 1.96
    # The p-value is 5 percent at the critical value, the normal one in large samples.
    assert inf.p_value(inf.critical_value(21), 21) == pytest.approx(0.05, abs=1e-4)
    assert inf.p_value(-inf.critical_value(21), 21) == pytest.approx(0.05, abs=1e-4)
    assert inf.p_value(2.0, 10 ** 7, 0) == pytest.approx(2 * stats.norm.sf(2.0), abs=1e-5)
    assert inf.p_value(0.0, 21) == 1.0 and np.isnan(inf.p_value(np.nan, 21))
    assert inf.p_value(2.0, 21) > 2 * stats.norm.sf(2.0)              # less significant than it looks
    one = [inf.p_value(t, 21, two_sided=False) for t in (-3.0, -1.0, 0.0, 1.0, 3.0, 9.0)]
    assert one == sorted(one, reverse=True) and one[2] == 0.5 and one[0] + one[4] == pytest.approx(1)
    assert inf.p_value(1.5, 21, two_sided=False) == pytest.approx(inf.p_value(1.5, 21) / 2)


def test_newey_west_t_has_its_nominal_size_only_with_the_small_sample_critical_value():
    for n in (14, 21):
        t = np.array([inf.mean_test(x)["t"] for x in rng(60 + n).standard_normal((4000, n))])
        assert np.mean(np.abs(t) > 1.96) > 0.08                       # the normal value over-rejects
        assert 0.035 < np.mean(np.abs(t) > inf.critical_value(n)) < 0.065


def test_ols_hac_with_zero_lags_is_white_by_hand():
    g = rng(6)
    n = 150
    X = np.column_stack([np.ones(n), g.standard_normal((n, 2))])
    y = X @ np.array([0.2, 1.0, -0.5]) + g.standard_normal(n) * (1 + np.abs(X[:, 1]))
    out = inf.ols_hac(y, X, lags=0)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ coef
    xtx_inv = np.linalg.inv(X.T @ X)
    white = xtx_inv @ (X.T @ np.diag(u ** 2) @ X) @ xtx_inv
    np.testing.assert_allclose(out["coef"], coef, rtol=1e-10)
    np.testing.assert_allclose(out["resid"], u, atol=1e-10)
    np.testing.assert_allclose(out["se"], np.sqrt(np.diag(white)), rtol=1e-10)
    np.testing.assert_allclose(out["t"], coef / np.sqrt(np.diag(white)), rtol=1e-10)


def test_ols_hac_matches_explicit_bartlett_sandwich():
    g = rng(7)
    n, lags = 120, 4
    x = signal.lfilter([1.0], [1.0, -0.6], g.standard_normal(n))
    X = np.column_stack([np.ones(n), x])
    y = 0.1 + 0.8 * x + signal.lfilter([1.0], [1.0, -0.6], g.standard_normal(n))
    out = inf.ols_hac(y, X, lags=lags)
    u = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    xtx_inv = np.linalg.inv(X.T @ X)
    cov = xtx_inv @ bartlett_sum_brute(X * u[:, None], lags) @ xtx_inv
    np.testing.assert_allclose(out["se"], np.sqrt(np.diag(cov)), rtol=1e-10)
    assert np.all(out["se"] > inf.ols_hac(y, X, lags=0)["se"])   # positive serial correlation


def test_ols_hac_on_a_constant_is_the_mean_test():
    x = np.cumsum(rng(8).standard_normal(100)) * 0.01
    for lags in (None, 0, 6):
        fit = inf.ols_hac(x, np.ones(x.size), lags)
        test = inf.mean_test(x, lags)
        assert fit["coef"][0] == pytest.approx(test["mean"])
        assert fit["se"][0] == pytest.approx(test["se"], rel=1e-10)
        assert fit["t"][0] == pytest.approx(test["t"], rel=1e-10)


def test_ols_hac_drops_nan_rows_and_flags_rank_deficiency():
    g = rng(9)
    X = np.column_stack([np.ones(60), g.standard_normal(60)])
    y = X @ np.array([1.0, 2.0]) + g.standard_normal(60)
    y_gap, X_gap = with_gaps(y, [3]), X.copy()
    X_gap[10, 1] = np.nan
    keep = np.setdiff1d(np.arange(60), [3, 10])
    a, b = inf.ols_hac(y_gap, X_gap, 2), inf.ols_hac(y[keep], X[keep], 2)
    for key in ("coef", "se", "t", "resid"):
        np.testing.assert_allclose(a[key], b[key])
    assert a["resid"].size == 58
    collinear = inf.ols_hac(y, np.column_stack([np.ones(60), np.ones(60)]))
    assert np.isnan(collinear["coef"]).all() and np.isnan(collinear["se"]).all()
    with pytest.raises(ValueError):
        inf.ols_hac(y, X[:50])


def test_stationary_bootstrap_shape_range_and_seed():
    index = inf.stationary_bootstrap(37, 50, 4.0, seed=5)
    assert index.shape == (50, 37)
    assert np.issubdtype(index.dtype, np.integer)
    assert index.min() >= 0 and index.max() <= 36
    np.testing.assert_array_equal(index, inf.stationary_bootstrap(37, 50, 4.0, seed=5))
    assert not np.array_equal(index, inf.stationary_bootstrap(37, 50, 4.0, seed=6))
    with pytest.raises(ValueError):
        inf.stationary_bootstrap(0, 5, 2.0, 0)
    with pytest.raises(ValueError):
        inf.stationary_bootstrap(10, 5, 0.5, 0)


@pytest.mark.parametrize("block", [1.0, 4.0, 12.5])
def test_stationary_bootstrap_mean_block_length(block):
    n = 1000
    index = inf.stationary_bootstrap(n, 300, block, seed=3)
    continues = index[:, 1:] == (index[:, :-1] + 1) % n
    n_blocks = index.shape[0] + (~continues).sum()
    assert index.size / n_blocks == pytest.approx(block, rel=0.04)


def test_stationary_bootstrap_is_uniform_and_wraps_around():
    n, block = 20, 5.0
    index = inf.stationary_bootstrap(n, 20_000, block, seed=1)
    share = np.bincount(index.ravel(), minlength=n) / index.size
    np.testing.assert_allclose(share, 1 / n, rtol=0.08)
    previous, following = index[:, :-1].ravel(), index[:, 1:].ravel()
    # after the last index a block continues at 0 unless a new block starts elsewhere
    expected = 1 - 1 / block + 1 / (block * n)
    assert np.mean(following[previous == n - 1] == 0) == pytest.approx(expected, abs=0.02)


def test_sharpe_se_is_the_mertens_formula():
    r = 0.02 - rng(10).gamma(2.0, 0.01, size=300) + 0.01
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    per_period = np.sqrt((1 + sr ** 2 / 2 - skew * sr + (kurt - 3) / 4 * sr ** 2) / r.size)
    assert inf.sharpe_se(r, 12) == pytest.approx(per_period * np.sqrt(12), rel=1e-10)
    assert inf.sharpe_se(r, 52) == pytest.approx(inf.sharpe_se(r, 12) * np.sqrt(52 / 12))
    assert np.isnan(inf.sharpe_se(np.full(20, 0.01), 12))
    assert np.isnan(inf.sharpe_se([0.01], 12))


def test_sharpe_se_matches_monte_carlo_for_skewed_fat_tailed_returns():
    n, sims, sr = 500, 4000, 0.3
    # one minus an exponential: mean 0, variance 1, skewness -2, kurtosis 9
    r = sr + 1.0 - rng(11).exponential(1.0, size=(sims, n))
    sharpes = np.array([met.sharpe(row, 12) for row in r])
    estimated = np.mean([inf.sharpe_se(row, 12) for row in r])
    monte_carlo = sharpes.std(ddof=1)
    assert estimated == pytest.approx(monte_carlo, rel=0.08)
    theory = np.sqrt((1 + sr ** 2 / 2 + 2 * sr + (9 - 3) / 4 * sr ** 2) / n * 12)
    assert monte_carlo == pytest.approx(theory, rel=0.08)
    normal_theory = np.sqrt((1 + sr ** 2 / 2) / n * 12)
    assert monte_carlo > 1.15 * normal_theory          # the normal formula is far too small


def test_sharpe_ci_matches_explicit_resampling():
    r = 0.008 + 0.05 * rng(12).standard_t(5, size=150)
    index = inf.stationary_bootstrap(r.size, 500, inf.default_block(r.size), seed=7)
    draws = np.array([met.sharpe(r[rows], 12) for rows in index])
    low, high = inf.sharpe_ci(r, 12, level=0.9, n_boot=500, seed=7)
    np.testing.assert_allclose([low, high], np.quantile(draws, [0.05, 0.95]), rtol=1e-9)
    wide = inf.sharpe_ci(r, 12, level=0.99, n_boot=500, seed=7)
    assert wide[0] < low < met.sharpe(r, 12) < high < wide[1]
    assert inf.sharpe_ci(with_gaps(np.r_[r, 0.0], [150]), 12, 0.9, 500, seed=7) == (low, high)
    assert np.isnan(inf.sharpe_ci(np.full(30, 0.01), 12)).all()

    tiny = np.array([0.1, 0.2, -0.15])                 # one resample in nine is constant
    index = inf.stationary_bootstrap(3, 300, 1.0, seed=3)
    draws = np.array([met.sharpe(tiny[rows], 12) for rows in index])
    assert np.isnan(draws).mean() > 0.05               # those have no Sharpe ratio: skipped
    expected = np.quantile(draws[~np.isnan(draws)], [0.005, 0.995])
    interval = inf.sharpe_ci(tiny, 12, level=0.99, n_boot=300, block=1.0, seed=3)
    np.testing.assert_allclose(interval, expected, rtol=1e-9)


def test_sharpe_ci_width_and_coverage_on_iid_data():
    r = 0.01 + 0.04 * rng(13).standard_normal(600)
    low, high = inf.sharpe_ci(r, 12, block=1.0)
    assert (high - low) / (2 * 1.96) == pytest.approx(inf.sharpe_se(r, 12), rel=0.12)
    true_sharpe, covered, sims = 0.25 * np.sqrt(12), 0, 250
    for i in range(sims):
        sample = 0.25 + rng(1000 + i).standard_normal(240)
        low, high = inf.sharpe_ci(sample, 12, n_boot=400, seed=i)
        covered += low <= true_sharpe <= high
    assert 0.89 <= covered / sims <= 0.995


def test_sharpe_difference_matches_explicit_resampling():
    g = rng(14)
    common = g.standard_normal(200)
    r1 = 0.010 + 0.04 * (0.7 * common + 0.7 * g.standard_normal(200))
    r2 = 0.002 + 0.03 * (0.7 * common + 0.7 * g.standard_normal(200))
    out = inf.sharpe_difference(r1, r2, 26, n_boot=400, block=6.0, seed=2)
    index = inf.stationary_bootstrap(200, 400, 6.0, seed=2)
    draws = np.array([met.sharpe(r1[rows], 26) - met.sharpe(r2[rows], 26) for rows in index])
    diff = met.sharpe(r1, 26) - met.sharpe(r2, 26)
    assert out["diff"] == pytest.approx(diff, rel=1e-12)
    assert out["se"] == pytest.approx(draws.std(ddof=1), rel=1e-9)
    np.testing.assert_allclose(out["ci"], np.quantile(draws, [0.025, 0.975]), rtol=1e-9)
    assert out["p"] == pytest.approx(np.mean(np.abs(draws - diff) >= abs(diff)), abs=1e-12)

    swapped = inf.sharpe_difference(r2, r1, 26, n_boot=400, block=6.0, seed=2)
    assert swapped["diff"] == pytest.approx(-out["diff"])
    assert swapped["se"] == pytest.approx(out["se"])
    assert swapped["p"] == pytest.approx(out["p"])
    np.testing.assert_allclose(swapped["ci"], [-out["ci"][1], -out["ci"][0]])

    same = inf.sharpe_difference(r1, r1, 26, n_boot=200)
    assert same["diff"] == 0.0 and same["p"] == 1.0 and same["se"] == pytest.approx(0.0)


def test_sharpe_difference_se_matches_closed_form_and_detects_a_gap():
    n, rho, s1, s2 = 1500, 0.6, 0.3, 0.1
    z = rng(15).standard_normal((n, 2))
    r1 = s1 + z[:, 0]
    r2 = s2 + rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
    out = inf.sharpe_difference(r1, r2, 12, block=1.0)
    # Jobson and Korkie (1981) with the correction of Memmel (2003), iid normal returns
    variance = (2 * (1 - rho) + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / n
    assert out["se"] == pytest.approx(np.sqrt(variance * 12), rel=0.12)
    assert out["diff"] == pytest.approx((s1 - s2) * np.sqrt(12), abs=4 * out["se"])
    assert out["p"] < 0.01 and out["ci"][0] > 0


def test_probabilistic_sharpe():
    r = 0.01 - rng(16).gamma(2.0, 0.02, size=240) + 0.04
    sr = r.mean() / r.std(ddof=1)
    assert inf.probabilistic_sharpe(r, benchmark_sr=sr) == pytest.approx(0.5, abs=1e-12)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    z = (sr - 0.05) * np.sqrt(r.size - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    assert inf.probabilistic_sharpe(r, 0.05) == pytest.approx(stats.norm.cdf(z), rel=1e-10)
    values = [inf.probabilistic_sharpe(r, sr + shift) for shift in (-0.5, -0.05, 0.0, 0.05, 0.5)]
    assert values == sorted(values, reverse=True) and values[0] > 0.99 > 0.01 > values[-1]
    assert np.isnan(inf.probabilistic_sharpe(np.zeros(12)))


def test_expected_max_sharpe():
    assert inf.expected_max_sharpe(1, 0.3) == 0.0
    assert inf.expected_max_sharpe(0, 0.3) == 0.0
    values = [inf.expected_max_sharpe(n, 0.1) for n in (2, 5, 20, 100, 1000)]
    assert all(a < b for a, b in zip(values, values[1:])) and values[0] > 0
    ratio = inf.expected_max_sharpe(40, 0.25) / inf.expected_max_sharpe(40, 0.1)
    assert ratio == pytest.approx(2.5)
    g, n = np.euler_gamma, 50
    by_hand = (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * np.e))
    assert inf.expected_max_sharpe(n, 1.0) == pytest.approx(by_hand, rel=1e-12)
    simulated = rng(17).standard_normal((100_000, n)).max(axis=1).mean()
    assert inf.expected_max_sharpe(n, 1.0) == pytest.approx(simulated, rel=0.025)


def test_deflated_sharpe_threshold_rises_with_the_number_of_trials():
    r = 0.012 + 0.05 * rng(18).standard_normal(300)
    results = [inf.deflated_sharpe(r, n, sd_trials=0.05) for n in (1, 2, 10, 100, 1000)]
    thresholds = [res["threshold"] for res in results]
    scores = [res["dsr"] for res in results]
    assert all(a < b for a, b in zip(thresholds, thresholds[1:]))
    assert all(a > b for a, b in zip(scores, scores[1:]))
    assert thresholds[0] == 0.0
    assert scores[0] == pytest.approx(inf.probabilistic_sharpe(r, 0.0))
    assert results[2]["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1))
    assert results[2]["threshold"] == pytest.approx(inf.expected_max_sharpe(10, 0.05))
    assert results[2]["dsr"] == pytest.approx(inf.probabilistic_sharpe(r, thresholds[2]))


def test_diebold_mariano_sign_and_symmetry():
    g = rng(19)
    truth = g.standard_normal(250)
    loss_a = (truth - (truth + 0.5 * g.standard_normal(250))) ** 2     # the better forecast
    loss_b = (truth - (truth + 1.0 * g.standard_normal(250))) ** 2
    ab = inf.diebold_mariano(loss_a, loss_b)
    ba = inf.diebold_mariano(loss_b, loss_a)
    assert ab["stat"] < 0 and ab["mean_diff"] < 0 and ab["p"] < 0.01
    assert ba["stat"] == pytest.approx(-ab["stat"])
    assert ba["mean_diff"] == pytest.approx(-ab["mean_diff"])
    assert ba["p"] == pytest.approx(ab["p"])
    test = inf.mean_test(loss_a - loss_b, lags=4)
    out = inf.diebold_mariano(loss_a, loss_b, lags=4)
    assert out == {"mean_diff": test["mean"], "stat": test["t"], "p": test["p"]}
    assert np.isnan(inf.diebold_mariano(loss_a, loss_a)["stat"])


def test_reality_check_rejects_when_one_of_twenty_strategies_is_good():
    returns = 0.05 * rng(20).standard_normal((300, 20))
    returns[:, 7] += 0.02
    out = inf.reality_check(returns)
    assert out["best"] == 7
    assert out["p_reality_check"] < 0.01 and out["p_spa"] < 0.01
    assert out["stat"] == pytest.approx(np.sqrt(300) * returns[:, 7].mean())


def test_reality_check_does_not_reject_pure_noise():
    # the legacy generator has a frozen stream, so this sample is the same on every numpy
    returns = 0.05 * np.random.RandomState(4).standard_normal((300, 20))
    out = inf.reality_check(returns, seed=0)
    assert out["p_reality_check"] > 0.1 and out["p_spa"] > 0.1


def test_expected_best_without_skill_depends_on_how_the_strategies_move_together():
    g = rng(28)
    t, m = 400, 50
    independent = g.standard_normal((t, m))
    out = inf.reality_check(independent / independent.std(axis=0, ddof=1), n_boot=2000, seed=1)
    # Best of 50 independent unskilled strategies: sqrt(T) * mean is standard normal for each,
    # and the expected maximum of 50 standard normals is 2.25.
    assert out["expected_stat"] == pytest.approx(2.25, abs=0.15)
    one = g.standard_normal(t)
    copies = np.column_stack([one + 0.01 * g.standard_normal(t) for _ in range(m)])
    same = inf.reality_check(copies / copies.std(axis=0, ddof=1), n_boot=2000, seed=1)
    assert abs(same["expected_stat"]) < 0.15            # fifty copies of one strategy are one trial
    # The p-value of the best is the share of resampled maxima above it.
    assert 0 < out["p_reality_check"] <= 1


def test_spa_is_not_hurt_by_one_very_bad_strategy():
    returns = rng(25).standard_normal((300, 10))
    returns = (returns - returns.mean(axis=0)) / returns.std(axis=0)   # exact sample moments
    returns[:, 0] += 0.22                              # a winner: t = sqrt(300) * 0.22 = 3.8
    returns[:, 9] = 10 * returns[:, 9] - 2             # a noisy and very poor strategy
    out = inf.reality_check(returns, seed=0)
    assert out["p_spa"] <= out["p_reality_check"] + 1e-12
    assert out["p_spa"] < 0.05 < out["p_reality_check"]
    without = inf.reality_check(returns[:, :9], seed=0)
    assert without["p_reality_check"] < 0.05           # the bad strategy alone hid the winner


def strategies_with_t_stats(t_stats: list[float], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (160, m) and a benchmark such that the excess returns have these t-statistics.

    The t-statistic is sqrt(T) * mean / std of returns minus benchmark, exact in the sample.
    """
    g = rng(seed)
    t = 160
    benchmark = 0.03 * g.standard_normal(t)
    noise = g.standard_normal((t, len(t_stats)))
    noise = (noise - noise.mean(axis=0)) / noise.std(axis=0)
    return benchmark[:, None] + 0.02 * (noise + np.array(t_stats) / np.sqrt(t)), benchmark


# The SPA rule recentres strategies with t_k <= -sqrt(2 * log(log(160))) = -1.80. In the second
# case the bootstrap is iid with many resamples, so omega_k is within about one percent of
# the sample standard deviation and the strategies at -1.71 and -1.89 lie on either side of
# that threshold.
@pytest.mark.parametrize("t_stats, block, n_boot, recentred", [
    ([1.4, 0.5, -2.4, 0.9, -12.0, 0.0], 5.0, 500, 2),
    ([1.4, 0.5, -1.71, 0.9, -1.89, 0.0], 1.0, 4000, 1),
])
def test_reality_check_matches_explicit_resampling(t_stats, block, n_boot, recentred):
    returns, benchmark = strategies_with_t_stats(t_stats, seed=21)
    d = returns - benchmark[:, None]
    expected = reality_check_brute(d, n_boot, block, seed=9)
    assert expected["recentred"] == recentred
    out = inf.reality_check(returns, benchmark, n_boot=n_boot, block=block, seed=9)
    for key in ("p_reality_check", "p_spa", "stat", "stat_spa"):
        assert out[key] == pytest.approx(expected[key], rel=1e-9, abs=1e-9)
    assert 0 < out["p_spa"] < 1 and 0 < out["p_reality_check"] < 1
    assert out["best"] == 0


def test_reality_check_default_block_benchmark_and_nan_rows():
    returns, benchmark = strategies_with_t_stats([1.4, 0.5, -2.4, 0.9, -12.0, 0.0], seed=26)
    d = returns - benchmark[:, None]
    out = inf.reality_check(returns, benchmark, n_boot=500, seed=9)
    by_hand = reality_check_brute(d, 500, 160 ** (1 / 3), seed=9)
    assert out["p_spa"] == pytest.approx(by_hand["p_spa"], abs=1e-9)
    assert out["p_reality_check"] == pytest.approx(by_hand["p_reality_check"], abs=1e-9)
    assert out == inf.reality_check(d, n_boot=500, seed=9)
    assert out == inf.reality_check(pd.DataFrame(returns), pd.Series(benchmark), 500, seed=9)

    gap_returns, gap_benchmark = returns.copy(), benchmark.copy()
    gap_returns[5, 2] = np.nan
    gap_benchmark[40] = np.nan
    keep = np.setdiff1d(np.arange(160), [5, 40])
    assert (inf.reality_check(gap_returns, gap_benchmark, n_boot=200, seed=1)
            == inf.reality_check(returns[keep], benchmark[keep], n_boot=200, seed=1))


def test_reality_check_with_a_strategy_identical_to_the_benchmark():
    returns, benchmark = strategies_with_t_stats([1.4, 0.5, -0.3], seed=27)
    plain = inf.reality_check(returns, benchmark, n_boot=300, seed=4)
    extra = inf.reality_check(np.column_stack([returns, benchmark]), benchmark, n_boot=300, seed=4)
    # omega = 0: left out of the SPA test. It does enter the maximum, which can only rise.
    assert {k: v for k, v in extra.items() if k != "expected_stat"} == \
           {k: v for k, v in plain.items() if k != "expected_stat"}
    assert extra["expected_stat"] >= plain["expected_stat"]
    nothing = inf.reality_check(np.column_stack([benchmark, benchmark]), benchmark, n_boot=50)
    assert nothing["p_reality_check"] == 1.0 and nothing["stat"] == 0.0
    assert np.isnan(nothing["p_spa"]) and np.isnan(nothing["stat_spa"])


def test_reality_check_when_every_strategy_is_worse_than_the_benchmark():
    returns = rng(22).standard_normal((200, 5)) - 0.5
    out = inf.reality_check(returns, n_boot=500)
    assert out["stat_spa"] == 0.0 and out["p_spa"] == 1.0
    assert out["stat"] < 0 and out["p_reality_check"] > 0.99
    with pytest.raises(ValueError, match="array"):
        inf.reality_check(returns[:, 0])
    with pytest.raises(ValueError, match="at least 3 complete rows"):
        inf.reality_check(returns[:2])
    with pytest.raises(ValueError, match="at least 3 complete rows"):
        inf.reality_check(with_gaps(returns[:, 0], list(range(198)))[:, None])
    with pytest.raises(ValueError, match="1 column"):
        inf.reality_check(returns[:, :0])


def test_reality_check_size_under_the_null():
    g = rng(23)
    sims, rejections = 400, np.zeros(2)
    for i in range(sims):
        out = inf.reality_check(g.standard_normal((120, 5)), n_boot=300, block=1.0, seed=i)
        rejections += [out["p_reality_check"] <= 0.10, out["p_spa"] <= 0.10]
    rc, spa = rejections / sims
    assert 0.05 <= rc <= 0.17
    assert 0.05 <= spa <= 0.18


def test_benjamini_hochberg_hand_computed():
    # sorted p: 0.005, 0.01, 0.03, 0.04 -> p * 4 / rank: 0.02, 0.02, 0.04, 0.04
    np.testing.assert_allclose(inf.benjamini_hochberg([0.01, 0.04, 0.03, 0.005]),
                               [0.02, 0.04, 0.04, 0.02])
    # p * 3 / rank: 0.03, 0.0165, 0.5 -> the running minimum from the right lowers the first
    np.testing.assert_allclose(inf.benjamini_hochberg([0.01, 0.011, 0.5]), [0.0165, 0.0165, 0.5])
    out = inf.benjamini_hochberg([0.005, 0.5, np.nan, 0.02, 0.9])
    np.testing.assert_allclose(out, [0.02, 2 / 3, np.nan, 0.04, 0.9])
    np.testing.assert_allclose(inf.benjamini_hochberg([0.9, 1.0, 0.95]), [1.0, 1.0, 1.0])
    # the largest p-value is returned unchanged, to the last bit (0.1 * 3 / 3 != 0.1)
    assert inf.benjamini_hochberg([0.1, 0.01, 0.02])[0] == 0.1
    assert inf.benjamini_hochberg([]).size == 0
    assert inf.benjamini_hochberg(pd.Series([0.2])).tolist() == [0.2]


def test_benjamini_hochberg_is_monotone_and_matches_scipy():
    p = rng(24).uniform(size=200) ** 2
    adjusted = inf.benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= p) and adjusted.max() <= 1
    assert inf.benjamini_hochberg(p.reshape(20, 10)).shape == (20, 10)
    if hasattr(stats, "false_discovery_control"):
        np.testing.assert_allclose(adjusted, stats.false_discovery_control(p, method="bh"))


# ------------------------------------------------------------------------------- metrics


def test_periods_per_year():
    assert met.PERIODS_PER_YEAR == {"M": 12, "2W": 26, "W": 52, "D": 252}


def test_sharpe():
    r = 0.01 + 0.05 * rng(30).standard_normal(120)
    assert met.sharpe(r, 12) == pytest.approx(r.mean() * 12 / (r.std(ddof=1) * np.sqrt(12)))
    assert met.sharpe(r, 52) == pytest.approx(met.sharpe(r, 12) * np.sqrt(52 / 12))
    assert np.isnan(met.sharpe([0.01], 12))
    assert np.isnan(met.sharpe([], 12))
    assert np.isnan(met.sharpe(np.full(6, 0.1), 12))       # their computed mean is not 0.1
    assert met.sharpe([0.01, np.nan, 0.03], 12) == pytest.approx(0.02 / np.sqrt(2e-4) * np.sqrt(12))
    series = pd.Series(r, index=np.arange(r.size)[::-1])     # the index is ignored
    assert met.sharpe(series, 12) == met.sharpe(r, 12)


def test_drawdown_on_a_hand_computed_path():
    # wealth 1.1, 0.88, 0.924, 1.2012, 1.08108 with peaks 1.1, 1.1, 1.1, 1.2012, 1.2012
    r = [0.10, -0.20, 0.05, 0.30, -0.10]
    np.testing.assert_allclose(met.drawdown(r), [0.0, -0.2, -0.16, 0.0, -0.1], atol=1e-12)
    assert met.max_drawdown(r) == pytest.approx(-0.2)
    # a loss in the first period is measured from the starting wealth of 1
    np.testing.assert_allclose(met.drawdown([-0.10, 0.05, 0.20]), [-0.10, -0.055, 0.0], atol=1e-12)
    assert met.max_drawdown([0.01, 0.02]) == 0.0
    assert np.isnan(met.max_drawdown([]))
    np.testing.assert_allclose(met.drawdown([0.10, np.nan, -0.20]), [0.0, -0.2], atol=1e-12)
    path = met.drawdown(0.05 * rng(31).standard_normal(500))
    assert path.max() <= 0 and path.min() == met.max_drawdown(0.05 * rng(31).standard_normal(500))


def test_performance_matches_direct_calculation():
    r = 0.006 + 0.04 * rng(32).standard_t(6, size=260)
    out = met.performance(r, 26, lags=3)
    assert list(out) == ["n", "mean", "vol", "sharpe", "sharpe_se", "t_mean", "p_mean", "skew",
                         "excess_kurtosis", "max_drawdown", "calmar", "hit_rate", "worst", "best"]
    wealth = np.cumprod(1 + r)
    trough = (wealth / np.maximum(np.maximum.accumulate(wealth), 1) - 1).min()
    t_mean = r.mean() / np.sqrt(inf.newey_west_variance(r, 3))
    expected = {
        "n": 260, "mean": r.mean() * 26, "vol": r.std(ddof=1) * np.sqrt(26),
        "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(26), "sharpe_se": inf.sharpe_se(r, 26),
        "t_mean": t_mean, "p_mean": inf.p_value(t_mean, 260, 3), "skew": stats.skew(r),
        "excess_kurtosis": stats.kurtosis(r), "max_drawdown": trough,
        "calmar": r.mean() * 26 / abs(trough), "hit_rate": (r > 0).mean(),
        "worst": r.min(), "best": r.max(),
    }
    for key, value in expected.items():
        assert out[key] == pytest.approx(value, rel=1e-10), key


def test_performance_alpha_and_beta_equal_lstsq():
    g = rng(33)
    bench = 0.005 + 0.04 * g.standard_normal(200)
    r = 0.003 + 0.7 * bench + 0.02 * g.standard_normal(200)
    out = met.performance(r, 12, benchmark=bench, lags=2)
    X = np.column_stack([np.ones(200), bench])
    intercept, slope = np.linalg.lstsq(X, r, rcond=None)[0]
    assert out["alpha"] == pytest.approx(intercept * 12, rel=1e-10)
    assert out["beta"] == pytest.approx(slope, rel=1e-10)
    assert out["t_alpha"] == pytest.approx(inf.ols_hac(r, X, 2)["t"][0], rel=1e-10)
    active = r - bench
    assert out["active_mean"] == pytest.approx(active.mean() * 12)
    assert out["tracking_error"] == pytest.approx(active.std(ddof=1) * np.sqrt(12))
    ratio = active.mean() / active.std(ddof=1) * np.sqrt(12)
    assert out["information_ratio"] == pytest.approx(ratio)
    assert list(out)[-6:] == ["alpha", "beta", "t_alpha", "active_mean", "tracking_error",
                              "information_ratio"]

    keep = np.setdiff1d(np.arange(200), [4, 77])
    gaps = met.performance(with_gaps(r, [4]), 12, benchmark=with_gaps(bench, [77]), lags=2)
    assert gaps == met.performance(r[keep], 12, benchmark=bench[keep], lags=2)
    assert gaps["n"] == 198


def test_performance_gives_no_test_for_an_exact_fit():
    bench = 0.005 + 0.04 * rng(34).standard_normal(120)
    same = (bench * 3 + 0.1) / 3 - 0.1 / 3                # the benchmark, with rounding error
    assert not np.array_equal(same, bench)
    out = met.performance(same, 12, benchmark=bench)
    assert np.isnan(out["t_alpha"]) and np.isnan(out["information_ratio"])
    assert out["beta"] == 1.0 and out["alpha"] == 0.0 and abs(out["active_mean"]) < 1e-12
    assert out["tracking_error"] == pytest.approx(0, abs=1e-12)

    shifted = met.performance(bench + 0.001, 12, benchmark=bench)
    assert shifted["beta"] == 1.0 and shifted["alpha"] == pytest.approx(0.012)
    assert np.isnan(shifted["t_alpha"]) and np.isnan(shifted["information_ratio"])

    scaled = met.performance(0.5 * bench + 0.003, 12, benchmark=bench)       # an exact line
    assert scaled["beta"] == 0.5 and scaled["alpha"] == pytest.approx(0.036)
    assert np.isnan(scaled["t_alpha"])
    assert scaled["information_ratio"] == pytest.approx(
        (scaled["active_mean"] / scaled["tracking_error"]))                  # the two do differ

    constant = met.performance(np.full(120, 0.01), 12, benchmark=bench)
    assert all(np.isnan(constant[k]) for k in ("alpha", "beta", "t_alpha", "sharpe"))
    # A real, tiny difference is still tested.
    noisy = met.performance(bench + 1e-6 * rng(35).standard_normal(120), 12, benchmark=bench)
    assert np.isfinite(noisy["t_alpha"]) and np.isfinite(noisy["information_ratio"])


def test_performance_on_degenerate_inputs():
    rising = met.performance([0.01, 0.02, 0.03], 12)
    assert rising["max_drawdown"] == 0.0 and np.isnan(rising["calmar"])
    assert rising["hit_rate"] == 1.0 and rising["worst"] == 0.01 and rising["best"] == 0.03
    empty = met.performance([np.nan], 12, benchmark=[0.01])
    assert empty["n"] == 0 and len(empty) == 20
    assert all(np.isnan(v) for k, v in empty.items() if k != "n")
    flat_benchmark = met.performance([0.01, -0.02, 0.03, 0.0], 12, benchmark=np.zeros(4))
    assert np.isnan(flat_benchmark["alpha"]) and np.isnan(flat_benchmark["beta"])
    assert flat_benchmark["active_mean"] == pytest.approx(0.005 * 12)
    assert flat_benchmark["hit_rate"] == 0.5               # a return of zero is not a hit


def test_r2_oos():
    # squared errors 0.25 + 0.25 + 1 = 1.5 against sum(y^2) = 6
    assert met.r2_oos([1.0, -1.0, 2.0], [0.5, -0.5, 1.0]) == pytest.approx(0.75)
    g = rng(34)
    y = g.standard_normal(500) + 0.3
    pred = 0.4 * y + 0.5 * g.standard_normal(500)
    assert met.r2_oos(y, np.zeros(500)) == 0.0
    assert met.r2_oos(y, y) == 1.0
    assert met.r2_oos(y, pred) == pytest.approx(1 - np.sum((y - pred) ** 2) / np.sum(y ** 2))
    centred = y - y.mean()                 # with a zero-mean target it is the usual R squared
    assert met.r2_oos(centred, pred) == pytest.approx(r2_score(centred, pred))
    assert met.r2_oos(y, pred) != pytest.approx(r2_score(y, pred), abs=1e-3)
    assert met.r2_oos(with_gaps(y, [0]), with_gaps(pred, [1])) == met.r2_oos(y[2:], pred[2:])
    assert np.isnan(met.r2_oos(np.zeros(5), np.ones(5)))


def test_r2_within_equals_r2_oos_on_demeaned_data():
    g = rng(35)
    n = 3000
    groups = g.integers(0, 25, size=n)
    level = g.standard_normal(25)[groups]                 # a period effect in y and in pred
    y = level + g.standard_normal(n)
    pred = 0.5 * level + 0.2 * (y - level) + 0.3 * g.standard_normal(n)
    frame = pd.DataFrame({"y": y, "pred": pred, "g": groups})
    demeaned = frame[["y", "pred"]] - frame.groupby("g")[["y", "pred"]].transform("mean")
    expected = met.r2_oos(demeaned["y"], demeaned["pred"])
    assert met.r2_within(y, pred, groups) == pytest.approx(expected, rel=1e-10)
    assert met.r2_within(y, pred, groups) != pytest.approx(met.r2_oos(y, pred), abs=1e-3)
    shifted = met.r2_within(y + 3 * level, pred - 7 * level, groups)
    assert shifted == pytest.approx(expected, rel=1e-9)
    labels = pd.Series(groups).map(lambda k: f"p{k:02d}")   # string labels, same answer
    assert met.r2_within(y, pred, labels) == pytest.approx(expected, rel=1e-10)


def test_rank_ic_equals_scipy_spearman_group_by_group_with_ties():
    g = rng(36)
    dates = pd.date_range("2015-01-31", periods=30, freq="D")
    sizes = g.integers(3, 60, size=30)
    groups = np.repeat(dates, sizes)
    y = np.round(g.standard_normal(groups.size), 1)        # one decimal: many ties
    pred = np.round(0.4 * y + g.standard_normal(groups.size), 1)
    shuffle = g.permutation(groups.size)
    groups, y, pred = groups[shuffle], y[shuffle], pred[shuffle]
    assert max(pd.Series(y).groupby(groups).agg(lambda v: v.duplicated().sum())) > 0

    ic = met.rank_ic(y, pred, groups)
    assert isinstance(ic, pd.Series) and list(ic.index) == list(dates)
    for date in dates:
        rows = groups == date
        assert ic[date] == pytest.approx(stats.spearmanr(y[rows], pred[rows])[0], abs=1e-12)

    y_gap, pred_gap = with_gaps(y, [0, 5]), with_gaps(pred, [9])
    keep = np.setdiff1d(np.arange(groups.size), [0, 5, 9])
    pd.testing.assert_series_equal(met.rank_ic(y_gap, pred_gap, groups),
                                   met.rank_ic(y[keep], pred[keep], groups[keep]))


def test_rank_ic_is_nan_for_small_or_constant_groups():
    groups = ["a"] * 4 + ["b"] * 2 + ["c"] * 4 + ["d"] * 3 + ["e"] * 3
    y = [1, 2, 3, 4] + [1, 2] + [5, 5, 5, 5] + [1, 2, 3] + [1, 2, 3]
    pred = [1, 3, 2, 4] + [1, 2] + [1, 2, 3, 4] + [3, 2, 1] + [np.nan, 1, 2]
    ic = met.rank_ic(y, pred, pd.Series(groups, name="period"))
    assert list(ic.index) == ["a", "b", "c", "d", "e"] and ic.index.name == "period"
    assert ic["a"] == pytest.approx(0.8)                   # 1 - 6 * 2 / (4 * 15)
    assert ic["d"] == pytest.approx(-1.0)
    assert np.isnan(ic["b"]) and np.isnan(ic["c"]) and np.isnan(ic["e"])
    assert met.rank_ic([], [], []).empty


@pytest.mark.parametrize("labels", [
    ["a", "a", "a", None, "b", "b", "b", "b"],
    [2.0, 2.0, 2.0, np.nan, 7.0, 7.0, 7.0, 7.0],
    list(pd.to_datetime(["2020-01-31"] * 3 + [None] + ["2020-02-29"] * 4)),
])
def test_rows_with_a_missing_group_label_are_dropped(labels):
    y = np.array([1.0, 2.0, 6.0, 100.0, 1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 3.0, 2.0, -100.0, 4.0, 3.0, 2.0, 1.0])
    keep = np.array([0, 1, 2, 4, 5, 6, 7])
    kept = [labels[i] for i in keep]
    ic = met.rank_ic(y, pred, labels)
    np.testing.assert_allclose(ic.to_numpy(), [0.5, -1.0])
    assert list(ic.index) == [labels[0], labels[4]]
    assert met.r2_within(y, pred, labels) == met.r2_within(y[keep], pred[keep], kept)
    # first group by rising pred: y = 1, 6 | 2; second group: y = 4, 3 | 2, 1
    table = met.quantile_returns(y, pred, labels, q=2)
    np.testing.assert_allclose(table.to_numpy(), [[3.5, 2.0], [3.5, 1.5]])


def test_panel_functions_accept_a_group_without_complete_rows():
    groups = np.repeat(["a", "b", "c"], 4)
    y = np.array([1.0, 2.0, 3.0, 5.0, np.nan, np.nan, np.nan, np.nan, 4.0, 1.0, 2.0, 3.0])
    pred = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 4.0, 3.0])
    rows = groups != "b"
    ic = met.rank_ic(y, pred, groups)
    assert list(ic.index) == ["a", "b", "c"] and np.isnan(ic["b"])
    np.testing.assert_allclose(ic[["a", "c"]], met.rank_ic(y[rows], pred[rows], groups[rows]))
    assert met.r2_within(y, pred, groups) == met.r2_within(y[rows], pred[rows], groups[rows])
    table = met.quantile_returns(y, pred, groups, q=2)
    assert list(table.index) == ["a", "c"]
    np.testing.assert_allclose(table.to_numpy(), [[2.0, 3.5], [2.5, 2.5]])


def test_rank_ic_is_fast_and_correct_on_a_large_panel():
    g = rng(37)
    n_groups, n_stocks = 600, 1000
    y = g.standard_normal((n_groups, n_stocks))
    pred = np.round(0.05 * y + g.standard_normal((n_groups, n_stocks)), 2)
    groups = np.repeat(np.arange(n_groups), n_stocks)
    start = time.perf_counter()
    ic = met.rank_ic(y.ravel(), pred.ravel(), groups)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
    ranks_y = stats.rankdata(y, axis=1) - (n_stocks + 1) / 2
    ranks_p = stats.rankdata(pred, axis=1) - (n_stocks + 1) / 2
    expected = (ranks_y * ranks_p).sum(axis=1) / np.sqrt((ranks_y ** 2).sum(axis=1)
                                                         * (ranks_p ** 2).sum(axis=1))
    np.testing.assert_allclose(ic.to_numpy(), expected, atol=1e-12)
    assert 0.03 < ic.mean() < 0.07


def test_rank_ic_with_more_groups_than_fit_in_16_bits():
    g = rng(43)
    n_groups = 2 ** 16 + 1
    y = g.standard_normal((n_groups, 4))
    pred = np.round(y + g.standard_normal((n_groups, 4)))          # integers: ties
    groups = np.repeat(np.arange(n_groups), 4)
    shuffle = g.permutation(groups.size)
    ic = met.rank_ic(y.ravel()[shuffle], pred.ravel()[shuffle], groups[shuffle])
    ranks_y = stats.rankdata(y, axis=1) - 2.5
    ranks_p = stats.rankdata(pred, axis=1) - 2.5
    cross = (ranks_y * ranks_p).sum(axis=1)
    scale = np.sqrt((ranks_y ** 2).sum(axis=1) * (ranks_p ** 2).sum(axis=1))
    expected = np.divide(cross, scale, out=np.full(n_groups, np.nan), where=scale > 0)
    np.testing.assert_allclose(ic.to_numpy(), expected, atol=1e-12)
    assert np.isnan(expected).any() and not np.isnan(expected).all()


def test_ic_summary():
    ic = pd.Series(0.03 + 0.1 * rng(38).standard_normal(150))
    ic.iloc[[3, 50]] = np.nan
    clean = ic.dropna().to_numpy()
    out = met.ic_summary(ic, lags=2)
    assert out["n"] == 148
    assert out["mean"] == pytest.approx(clean.mean())
    assert out["std"] == pytest.approx(clean.std(ddof=1))
    assert out["t"] == pytest.approx(clean.mean() / np.sqrt(inf.newey_west_variance(clean, 2)))
    assert out["icir"] == pytest.approx(clean.mean() / clean.std(ddof=1))
    assert out["share_positive"] == pytest.approx((clean > 0).mean())
    assert met.ic_summary([])["n"] == 0 and np.isnan(met.ic_summary([])["mean"])


def test_quantile_returns_on_a_hand_made_example():
    # group a (7 rows, shuffled): floor(rank * 3 / 7) = 0, 0, 0, 1, 1, 2, 2
    pred_a = [5, 1, 7, 3, 2, 6, 4]
    y_a = [50, 10, 70, 30, 20, 60, 40]
    w_a = [1, 1, 2, 2, 1, 2, 1]
    # group b (4 tied rows): ranks follow the row order, floor(rank * 3 / 4) = 0, 0, 1, 2
    pred_b, y_b, w_b = [9, 9, 9, 9], [1, 2, 3, 4], [3, 1, 1, 1]
    # group c has fewer rows than quantiles and is dropped; so is the incomplete row of b
    pred_c, y_c, w_c = [1, 2], [5, 6], [1, 1]
    groups = pd.Series(["a"] * 7 + ["b"] * 4 + ["c"] * 2 + ["b"], name="period")
    y = np.array(y_a + y_b + y_c + [np.nan], dtype=float)
    pred = np.array(pred_a + pred_b + pred_c + [0.0], dtype=float)
    weights = np.array(w_a + w_b + w_c + [1.0], dtype=float)

    equal = met.quantile_returns(y, pred, groups, q=3)
    assert list(equal.index) == ["a", "b"] and list(equal.columns) == [1, 2, 3]
    assert equal.index.name == "period"
    np.testing.assert_allclose(equal.to_numpy(), [[20.0, 45.0, 65.0], [1.5, 3.0, 4.0]])

    weighted = met.quantile_returns(y, pred, groups, q=3, weights=weights)
    # a: (10 + 20 + 2 * 30) / 4, (40 + 50) / 2, (2 * 60 + 2 * 70) / 4; b: (3 * 1 + 2) / 4
    np.testing.assert_allclose(weighted.to_numpy(), [[22.5, 45.0, 65.0], [1.25, 3.0, 4.0]])
    with pytest.raises(ValueError):
        met.quantile_returns(y, pred, groups, q=0)
    # weights that sum to zero in a cell leave it empty: NaN, then (3 * 1 + 4 * 3) / 4
    idle = met.quantile_returns([1, 2, 3, 4], [1, 2, 3, 4], ["a"] * 4, q=2, weights=[0, 0, 1, 3])
    np.testing.assert_allclose(idle.to_numpy(), [[np.nan, 3.75]])


def test_quantile_returns_matches_pandas_groupby():
    g = rng(39)
    n, q = 4000, 10
    frame = pd.DataFrame({"g": g.integers(0, 30, size=n), "y": g.standard_normal(n),
                          "w": g.uniform(0.1, 5.0, size=n)})
    frame["pred"] = np.round(0.3 * frame["y"] + g.standard_normal(n), 1)     # ties
    rank = frame.groupby("g")["pred"].rank(method="first") - 1
    size = frame.groupby("g")["pred"].transform("size")
    frame["bucket"] = (rank * q // size).astype(int) + 1
    frame["wy"] = frame["w"] * frame["y"]
    cells = frame.groupby(["g", "bucket"])
    equal = cells["y"].mean().unstack()
    weighted = (cells["wy"].sum() / cells["w"].sum()).unstack()

    out = met.quantile_returns(frame["y"], frame["pred"], frame["g"], q=q)
    assert list(out.index) == list(equal.index) and list(out.columns) == list(range(1, q + 1))
    np.testing.assert_allclose(out.to_numpy(), equal.to_numpy(), atol=1e-12)
    out_w = met.quantile_returns(frame["y"], frame["pred"], frame["g"], q=q, weights=frame["w"])
    np.testing.assert_allclose(out_w.to_numpy(), weighted.to_numpy(), atol=1e-12)
    assert out[q].mean() - out[1].mean() > 0.5
    by_scipy = stats.spearmanr(np.arange(q), out.mean().to_numpy())[0]
    assert met.monotonicity(out) == pytest.approx(by_scipy) and by_scipy > 0.9


def test_monotonicity():
    assert met.monotonicity([0.1, 0.2, 0.3, 0.5]) == pytest.approx(1.0)
    assert met.monotonicity([0.5, 0.3, 0.2, 0.1]) == pytest.approx(-1.0)
    means = np.round(rng(40).standard_normal(10), 1)
    means[3] = means[8]                                      # a tie
    assert met.monotonicity(means) == pytest.approx(stats.spearmanr(np.arange(10), means)[0])
    gaps = [0.1, np.nan, 0.3, 0.2, 0.4]
    assert met.monotonicity(gaps) == pytest.approx(stats.spearmanr([0, 1, 2, 3],
                                                                   [0.1, 0.3, 0.2, 0.4])[0])
    table = pd.DataFrame([[0.0, 0.3, 0.1], [0.2, np.nan, 0.5]], columns=[1, 2, 3])
    assert met.monotonicity(table) == pytest.approx(met.monotonicity([0.1, 0.3, 0.3]))
    assert np.isnan(met.monotonicity([0.2, 0.2, 0.2]))
    assert np.isnan(met.monotonicity([0.2]))


@pytest.mark.parametrize("function", [
    lambda x: inf.newey_west_variance(x, 2),
    lambda x: inf.mean_test(x, 2),
    lambda x: inf.sharpe_se(x, 12),
    lambda x: inf.probabilistic_sharpe(x, 0.1),
    lambda x: inf.deflated_sharpe(x, 10, 0.1),
    lambda x: inf.sharpe_ci(x, 12, n_boot=100),
    lambda x: met.sharpe(x, 12),
    lambda x: met.max_drawdown(x),
    lambda x: met.performance(x, 12),
    lambda x: met.ic_summary(x),
])
def test_nan_observations_are_dropped_by_single_series_functions(function):
    x = 0.01 + 0.05 * rng(41).standard_normal(80)
    gaps = np.insert(x, [0, 17, 17, 80], np.nan)
    assert function(pd.Series(gaps)) == function(x)


@pytest.mark.parametrize("function", [
    lambda a, b: inf.diebold_mariano(a, b, 2),
    lambda a, b: inf.sharpe_difference(a, b, 12, n_boot=100),
    lambda a, b: met.r2_oos(a, b),
    lambda a, b: met.performance(a, 12, benchmark=b),
])
def test_nan_observations_are_dropped_pairwise_by_two_series_functions(function):
    g = rng(42)
    a, b = 0.01 + 0.05 * g.standard_normal(80), 0.05 * g.standard_normal(80)
    keep = np.setdiff1d(np.arange(80), [2, 30, 31])
    assert function(with_gaps(a, [2, 30]), with_gaps(b, [30, 31])) == function(a[keep], b[keep])


def test_sharpe_difference_hac_matches_closed_form_for_independent_normals():
    # iid normal, independent: n Var(SR_1 - SR_2) = 2 + (SR_1^2 + SR_2^2) / 2 per period
    rng = np.random.default_rng(3)
    n = 20_000
    a, b = rng.normal(0.1, 1.0, n), rng.normal(0.0, 2.0, n)
    out = inf.sharpe_difference_hac(a, b, periods_per_year=1, lags=0)
    assert out["se"] == pytest.approx(np.sqrt((2 + 0.1 ** 2 / 2) / n), rel=0.05)
    assert out["diff"] == pytest.approx(a.mean() / a.std(ddof=0) - b.mean() / b.std(ddof=0))
    assert out["t"] > 5 and out["p"] < 1e-6


def test_sharpe_difference_hac_annualises_sees_correlation_and_agrees_with_the_bootstrap():
    rng = np.random.default_rng(4)
    common = rng.normal(0.0, 1.0, 400)
    a = 0.02 + common + 0.3 * rng.normal(size=400)
    b = common + 0.3 * rng.normal(size=400)
    weekly = inf.sharpe_difference_hac(a, b, periods_per_year=52)
    single = inf.sharpe_difference_hac(a, b, periods_per_year=1)
    assert weekly["diff"] == pytest.approx(single["diff"] * np.sqrt(52))
    assert weekly["t"] == pytest.approx(single["t"])
    # highly correlated series: the difference is far more precise than either Sharpe ratio
    assert single["se"] < 0.5 * inf.sharpe_se(a, 1)
    boot = inf.sharpe_difference(a, b, periods_per_year=1, n_boot=4000, seed=1)
    assert single["se"] == pytest.approx(boot["se"], rel=0.25)
    # swapping the series flips the sign and keeps the p-value; NaN rows are dropped
    swapped = inf.sharpe_difference_hac(b, a, periods_per_year=1)
    assert swapped["t"] == pytest.approx(-single["t"]) and swapped["p"] == pytest.approx(single["p"])
    a_nan = a.copy(); a_nan[5] = np.nan
    assert inf.sharpe_difference_hac(a_nan, b, 1)["n"] == 399
    assert np.isnan(inf.sharpe_difference_hac(a[:2], b[:2], 1)["diff"])
