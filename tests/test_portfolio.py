"""Tests of ftm.portfolio (weights, covariance models, backtest) on synthetic data.

Every expected value comes from a calculation that does not use the code under test: numbers
worked out by hand, a brute-force loop, a closed form in numpy, or scikit-learn / pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from ftm.portfolio.backtest import align, run_backtest
from ftm.portfolio.risk import FactorCovariance, factor_covariance, ledoit_wolf_covariance
from ftm.portfolio.weights import (
    mean_variance,
    neutralise,
    quantile_long_short,
    rank_weighted,
    top_quantile,
)

NAN = np.nan


def _random_cov(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n))
    return a @ a.T / n + 0.5 * np.eye(n)


def _random_factor_cov(rng: np.random.Generator, n: int, k: int) -> FactorCovariance:
    return FactorCovariance(
        rng.normal(size=(n, k)), rng.uniform(0.5, 2.0, size=k), rng.uniform(0.1, 1.0, size=n)
    )


def _project_out(vectors: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    """Columns of ``vectors`` made orthogonal to the columns of ``neutral`` (QR, not ftm code)."""
    q, _ = np.linalg.qr(neutral)
    return vectors - q @ (q.T @ vectors)


def _one_factor_returns(rng: np.random.Generator, t: int, n: int):
    """Daily returns r_ti = beta_i f_t + e_ti with one common factor."""
    beta = rng.uniform(0.5, 1.5, size=n)
    factor = rng.normal(0.0, 0.015, size=t)
    sigma = rng.uniform(0.01, 0.02, size=n)
    returns = np.outer(factor, beta) + rng.normal(size=(t, n)) * sigma
    return returns, beta, factor, sigma


# ---------------------------------------------------------------- sorted and ranked portfolios


def test_quantile_long_short_sums_signs_and_members():
    score = np.random.default_rng(1).normal(size=100)
    order = np.argsort(score)
    w = quantile_long_short(score, q=10)
    assert w.shape == (100,)
    assert set(np.flatnonzero(w > 0)) == set(order[-10:])
    assert set(np.flatnonzero(w < 0)) == set(order[:10])
    np.testing.assert_allclose(w[order[-10:]], 0.1)
    np.testing.assert_allclose(w[order[:10]], -0.1)
    assert w[w > 0].sum() == pytest.approx(1.0)
    assert w[w < 0].sum() == pytest.approx(-1.0)

    terciles = quantile_long_short(score, q=3)
    assert (terciles > 0).sum() == 33 and (terciles < 0).sum() == 33
    assert terciles[terciles > 0].sum() == pytest.approx(1.0)
    assert terciles[terciles < 0].sum() == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        quantile_long_short(score, q=1)


def test_quantile_long_short_ignores_nan_scores():
    score = np.array([0.3, NAN, -1.0, 2.0, NAN, 0.5, 0.1, -0.2, 0.9, 1.5, -0.7, NAN])
    # 9 valid scores and q = 4 give 2 names a leg: 2.0 and 1.5 against -1.0 and -0.7.
    expected = np.zeros(12)
    expected[[3, 9]] = 0.5
    expected[[2, 10]] = -0.5
    np.testing.assert_array_equal(quantile_long_short(score, q=4), expected)

    # floor(9 / 10) = 0, so each leg still holds one name.
    single = np.zeros(12)
    single[3], single[2] = 1.0, -1.0
    np.testing.assert_array_equal(quantile_long_short(score, q=10), single)

    assert not quantile_long_short(np.array([NAN, 1.0, NAN])).any()
    assert not quantile_long_short(np.array([NAN, NAN])).any()
    assert quantile_long_short(np.array([])).shape == (0,)


def test_quantile_long_short_breaks_ties_by_position():
    score = np.array([5.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 3.0])
    # Stable ascending order of the positions: 1 2 3 | 7 0 | 4 5 6; two names a leg.
    expected = np.array([0.0, -0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 0.0])
    np.testing.assert_array_equal(quantile_long_short(score, q=4), expected)
    np.testing.assert_array_equal(quantile_long_short(np.ones(4), q=2), [-0.5, -0.5, 0.5, 0.5])
    np.testing.assert_array_equal(top_quantile(score, q=4), np.clip(expected, 0.0, None))


def test_quantile_long_short_cap_weights_by_hand():
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.5, 0.4])
    cap = np.array([30.0, 10.0, 1.0, 3.0, 100.0, 100.0])
    # Long 30 and 10 (weights 3/4 and 1/4), short 1 and 3 (weights 1/4 and 3/4).
    expected = np.array([0.75, 0.25, -0.25, -0.75, 0.0, 0.0])
    np.testing.assert_allclose(quantile_long_short(score, q=3, cap=cap), expected)
    np.testing.assert_allclose(top_quantile(score, q=3, cap=cap), [0.75, 0.25, 0.0, 0.0, 0.0, 0.0])


def test_missing_cap_takes_the_median_cap_of_its_leg():
    score = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    cap = np.array([NAN, 10.0, 30.0, 8.0, NAN, 2.0])
    # Long leg: NaN -> median(10, 30) = 20, total 60. Short leg: NaN -> median(8, 2) = 5, total 15.
    expected = np.array([20 / 60, 10 / 60, 30 / 60, -8 / 15, -5 / 15, -2 / 15])
    np.testing.assert_allclose(quantile_long_short(score, q=2, cap=cap), expected)

    # A leg without any cap falls back to equal weights.
    no_cap = np.array([NAN, NAN, NAN, 8.0, 5.0, 2.0])
    np.testing.assert_allclose(quantile_long_short(score, q=2, cap=no_cap)[:3], 1 / 3)


def test_top_quantile_is_the_long_leg_and_sums_to_one():
    rng = np.random.default_rng(2)
    score = rng.normal(size=57)
    score[[4, 20, 21]] = NAN
    cap = rng.lognormal(size=57)
    for weights_by in (None, cap):
        top = top_quantile(score, q=5, cap=weights_by)
        long_short = quantile_long_short(score, q=5, cap=weights_by)
        np.testing.assert_allclose(top, np.clip(long_short, 0.0, None))
        assert (top >= 0).all() and (top > 0).sum() == 10
        assert top.sum() == pytest.approx(1.0)

    everything = top_quantile(score, q=1)
    np.testing.assert_allclose(everything[~np.isnan(score)], 1 / 54)
    assert not everything[np.isnan(score)].any()
    np.testing.assert_array_equal(top_quantile(np.array([NAN, 2.0, NAN])), [0.0, 1.0, 0.0])
    assert not top_quantile(np.array([NAN, NAN])).any()


def test_rank_weighted_by_hand():
    score = np.array([10.0, 20.0, 20.0, 40.0, NAN, 30.0])
    # Ranks of the five valid scores: 1, 2.5, 2.5, 5, 4 with mean 3. Rank minus mean:
    # -2, -0.5, -0.5, 2, 1, whose positive part sums to 3.
    expected = np.array([-2 / 3, -1 / 6, -1 / 6, 2 / 3, 0.0, 1 / 3])
    np.testing.assert_allclose(rank_weighted(score), expected)


def test_rank_weighted_against_pandas_ranks():
    rng = np.random.default_rng(3)
    score = rng.integers(0, 15, size=200).astype(float)
    score[rng.choice(200, size=17, replace=False)] = NAN
    rank = pd.Series(score).rank(method="average")
    centred = rank - rank.mean()
    expected = (centred / centred[centred > 0].sum()).fillna(0.0).to_numpy()

    w = rank_weighted(score)
    np.testing.assert_allclose(w, expected)
    assert w[w > 0].sum() == pytest.approx(1.0)
    assert w[w < 0].sum() == pytest.approx(-1.0)
    assert not w[np.isnan(score)].any()
    valid = ~np.isnan(score)
    assert (np.diff(w[valid][np.argsort(score[valid], kind="stable")]) >= 0).all()

    assert not rank_weighted(np.full(6, 2.0)).any()
    assert not rank_weighted(np.array([NAN, 1.0])).any()


def test_neutralise_is_the_orthogonal_projection():
    rng = np.random.default_rng(4)
    w = rng.normal(size=50)
    x = rng.normal(size=(50, 3))
    out = neutralise(w, x)
    np.testing.assert_allclose(out, w - x @ np.linalg.solve(x.T @ x, x.T @ w))
    np.testing.assert_allclose(x.T @ out, 0.0, atol=1e-12)
    np.testing.assert_allclose(neutralise(out, x), out)

    np.testing.assert_allclose(neutralise(w, np.ones(50)), w - w.mean())
    collinear = np.column_stack([x[:, 0], 2.0 * x[:, 0]])
    np.testing.assert_allclose(neutralise(w, collinear), neutralise(w, x[:, 0]))
    with pytest.raises(ValueError):
        neutralise(w, np.ones(49))


def test_neutralise_gives_zero_weight_without_an_exposure():
    w = np.array([0.5, 0.5, -0.5, -0.5, 0.2])
    x = np.array([1.0, 2.0, 1.0, 3.0, NAN])
    # Over the first four assets x'w = -0.5 and x'x = 15, so w - x * (-0.5 / 15).
    expected = np.append(w[:4] + x[:4] / 30.0, 0.0)
    np.testing.assert_allclose(neutralise(w, x), expected)


# ------------------------------------------------------------------------------ mean variance


def test_mean_variance_matches_the_closed_form():
    rng = np.random.default_rng(5)
    n = 40
    cov = _random_cov(rng, n)
    mu = rng.normal(size=n)
    neutral = np.column_stack([np.ones(n), rng.normal(1.0, 0.3, size=n)])

    # Stationarity and feasibility of min 0.5 w'Sw - w'mu s.t. N'w = 0 as one linear system.
    kkt = np.block([[cov, neutral], [neutral.T, np.zeros((2, 2))]])
    optimum = np.linalg.solve(kkt, np.concatenate([mu, np.zeros(2)]))[:n]
    w = mean_variance(mu, cov, neutral=neutral)
    np.testing.assert_allclose(w, 2.0 * optimum / np.abs(optimum).sum(), rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(neutral.T @ w, 0.0, atol=1e-10)
    assert np.abs(w).sum() == pytest.approx(2.0)

    # The same optimum from the formula with explicit inverses.
    inverse = np.linalg.inv(cov)
    lam = np.linalg.inv(neutral.T @ inverse @ neutral) @ neutral.T @ inverse @ mu
    formula = inverse @ (mu - neutral @ lam)
    np.testing.assert_allclose(w, 2.0 * formula / np.abs(formula).sum(), rtol=1e-8, atol=1e-11)

    # Among neutral portfolios none has a higher score per unit of risk.
    def ratio(v: np.ndarray) -> float:
        return float(v @ mu / np.sqrt(v @ cov @ v))

    others = _project_out(rng.normal(size=(n, 300)), neutral)
    assert max(ratio(v) for v in others.T) < ratio(w)

    free = np.linalg.solve(cov, mu)
    np.testing.assert_allclose(mean_variance(mu, cov, gross=1.0), free / np.abs(free).sum())
    one = mean_variance(mu, cov, neutral=np.ones(n))
    assert abs(one.sum()) < 1e-10 and np.abs(one).sum() == pytest.approx(2.0)


def test_mean_variance_excludes_nan_scores():
    rng = np.random.default_rng(6)
    n = 30
    cov = _random_cov(rng, n)
    mu = rng.normal(size=n)
    mu[[2, 11, 29]] = NAN
    neutral = np.column_stack([np.ones(n), rng.normal(size=n)])
    neutral[7, 1] = NAN
    keep = ~np.isnan(mu) & ~np.isnan(neutral[:, 1])

    w = mean_variance(mu, cov, neutral=neutral)
    assert not w[~keep].any()
    reduced = mean_variance(mu[keep], cov[np.ix_(keep, keep)], neutral=neutral[keep])
    np.testing.assert_allclose(w[keep], reduced, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(neutral[keep].T @ w[keep], 0.0, atol=1e-10)
    assert np.abs(w).sum() == pytest.approx(2.0)


def test_mean_variance_with_factor_covariance_equals_dense():
    rng = np.random.default_rng(7)
    n = 80
    cov = _random_factor_cov(rng, n, 4)
    mu = rng.uniform(-1.0, 1.0, size=n)
    mu[[0, 40]] = NAN
    neutral = np.column_stack([np.ones(n), cov.loadings[:, 0]])
    for max_abs in (None, 0.06):
        structured = mean_variance(mu, cov, neutral=neutral, max_abs=max_abs)
        dense = mean_variance(mu, cov.matrix(), neutral=neutral, max_abs=max_abs)
        np.testing.assert_allclose(structured, dense, rtol=1e-7, atol=1e-10)
        assert not structured[[0, 40]].any()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_mean_variance_cap_keeps_neutrality_and_gross(seed):
    rng = np.random.default_rng(seed)
    n = 60
    cov = _random_cov(rng, n)
    mu = rng.uniform(-1.0, 1.0, size=n)
    neutral = np.column_stack([np.ones(n), rng.normal(1.0, 0.3, size=n)])
    max_abs = 0.05
    assert np.abs(mean_variance(mu, cov, neutral=neutral)).max() > max_abs + 0.01

    w = mean_variance(mu, cov, gross=2.0, max_abs=max_abs, neutral=neutral)
    assert np.abs(w).max() <= max_abs + 1e-6
    assert np.abs(neutral.T @ w).max() <= 1e-6
    assert abs(np.abs(w).sum() - 2.0) <= 1e-6
    assert (np.abs(w) >= max_abs - 1e-6).sum() >= 2

    only_cap = mean_variance(mu, cov, gross=2.0, max_abs=max_abs)
    assert np.abs(only_cap).max() <= max_abs + 1e-6
    assert abs(np.abs(only_cap).sum() - 2.0) <= 1e-6


def test_mean_variance_cap_on_a_large_universe_with_factor_covariance():
    rng = np.random.default_rng(8)
    n = 1000
    cov = FactorCovariance(
        rng.normal(1.0, 0.4, size=(n, 3)) * 0.01,
        np.array([1.0, 0.2, 0.1]),
        rng.uniform(1e-4, 9e-4, size=n),
    )
    mu = rng.uniform(-1.0, 1.0, size=n)
    neutral = np.column_stack([np.ones(n), cov.loadings[:, 0]])
    assert (np.abs(mean_variance(mu, cov, neutral=neutral)) > 0.005).sum() > 50

    w = mean_variance(mu, cov, gross=2.0, max_abs=0.005, neutral=neutral)
    assert np.abs(w).max() <= 0.005 + 1e-6
    assert np.abs(neutral.T @ w).max() <= 1e-6
    assert abs(np.abs(w).sum() - 2.0) <= 1e-6
    assert w @ mu > 0


def test_mean_variance_cap_that_cannot_be_met_lowers_gross():
    rng = np.random.default_rng(9)
    n = 20
    cov = _random_cov(rng, n)
    mu = rng.normal(size=n)
    # 20 names of at most 0.05 each cannot reach a gross exposure of 2.
    w = mean_variance(mu, cov, gross=2.0, max_abs=0.05, neutral=np.ones(n))
    assert np.abs(w).max() <= 0.05 + 1e-12
    assert abs(w.sum()) <= 1e-9
    assert 0.0 < np.abs(w).sum() <= 1.0 + 1e-9


def test_mean_variance_degenerate_problems_give_zeros():
    rng = np.random.default_rng(10)
    n = 12
    cov = _random_cov(rng, n)
    beta = rng.normal(size=n)
    assert not mean_variance(np.full(n, 0.3), cov).any()
    assert not mean_variance(np.full(n, 0.3), cov, neutral=beta).any()

    two_valid = np.full(n, NAN)
    two_valid[[1, 5]] = [0.2, -0.4]
    assert not mean_variance(two_valid, cov).any()

    in_span = 2.0 + 3.0 * beta
    assert not mean_variance(in_span, cov, neutral=np.column_stack([np.ones(n), beta])).any()

    with pytest.raises(ValueError):
        mean_variance(rng.normal(size=n), cov[:5, :5])
    with pytest.raises(ValueError):
        mean_variance(rng.normal(size=n), cov, gross=0.0)


# --------------------------------------------------------------------------- covariance models


def test_factor_covariance_solve_matches_numpy():
    rng = np.random.default_rng(11)
    n, k = 60, 4
    cov = _random_factor_cov(rng, n, k)
    dense = cov.matrix()
    expected = cov.loadings @ np.diag(cov.factor_var) @ cov.loadings.T + np.diag(cov.specific_var)
    np.testing.assert_allclose(dense, expected)

    vector = rng.normal(size=n)
    block = rng.normal(size=(n, 5))
    assert cov.solve(vector).shape == (n,) and cov.solve(block).shape == (n, 5)
    np.testing.assert_allclose(cov.solve(vector), np.linalg.solve(dense, vector), rtol=1e-9)
    np.testing.assert_allclose(cov.solve(block), np.linalg.solve(dense, block), rtol=1e-9)

    # A factor without variance and a model without factors are both allowed.
    flat = FactorCovariance(cov.loadings, np.array([1.0, 0.0, 2.0, 0.0]), cov.specific_var)
    np.testing.assert_allclose(flat.solve(vector), np.linalg.solve(flat.matrix(), vector))
    diagonal = FactorCovariance(np.zeros((n, 0)), np.zeros(0), cov.specific_var)
    np.testing.assert_allclose(diagonal.solve(vector), vector / cov.specific_var)
    np.testing.assert_allclose(diagonal.matrix(), np.diag(cov.specific_var))


def test_factor_covariance_variance_and_subset():
    rng = np.random.default_rng(12)
    cov = _random_factor_cov(rng, 30, 3)
    dense = cov.matrix()
    w = rng.normal(size=30)
    assert cov.variance(w) == pytest.approx(w @ dense @ w)
    with_gap = w.copy()
    with_gap[3] = NAN
    zeroed = np.where(np.arange(30) == 3, 0.0, w)
    assert cov.variance(with_gap) == pytest.approx(zeroed @ dense @ zeroed)

    idx = np.array([5, 0, 17, 3])
    np.testing.assert_allclose(cov.subset(idx).matrix(), dense[np.ix_(idx, idx)])
    mask = rng.random(30) < 0.5
    np.testing.assert_allclose(cov.subset(mask).matrix(), dense[np.ix_(mask, mask)])
    np.testing.assert_allclose(
        cov.subset(idx).solve(w[idx]), np.linalg.solve(dense[np.ix_(idx, idx)], w[idx])
    )


def test_factor_covariance_rejects_invalid_parts():
    with pytest.raises(ValueError):
        FactorCovariance(np.ones((3, 1)), np.ones(1), np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        FactorCovariance(np.ones((3, 1)), np.array([-1.0]), np.ones(3))
    with pytest.raises(ValueError):
        FactorCovariance(np.ones((3, 2)), np.ones(1), np.ones(3))
    with pytest.raises(ValueError):
        FactorCovariance(np.ones((3, 1)), np.ones(1), np.array([1.0, NAN, 1.0]))


def test_factor_covariance_recovers_a_one_factor_structure():
    rng = np.random.default_rng(13)
    returns, beta, factor, sigma = _one_factor_returns(rng, 252, 300)
    cov = factor_covariance(returns, n_factors=1)
    assert cov.loadings.shape == (300, 1) and cov.factor_var.shape == (1,)
    assert np.corrcoef(cov.loadings[:, 0], beta)[0, 1] > 0.9

    # Systematic volatility beta_i * sd(f), specific variance and total variance, per asset.
    systematic = cov.loadings[:, 0] * np.sqrt(cov.factor_var[0])
    assert np.median(systematic / (beta * factor.std(ddof=1))) == pytest.approx(1.0, abs=0.05)
    assert np.median(cov.specific_var / sigma**2) == pytest.approx(1.0, abs=0.05)
    np.testing.assert_allclose(np.diag(cov.matrix()), returns.var(axis=0, ddof=1), rtol=1e-9)

    # The model covariance is close to the true one, far closer than a diagonal matrix is.
    truth = np.outer(beta, beta) * 0.015**2 + np.diag(sigma**2)
    error = np.linalg.norm(cov.matrix() - truth) / np.linalg.norm(truth)
    assert error < 0.2 < np.linalg.norm(np.diag(np.diag(truth)) - truth) / np.linalg.norm(truth)


@pytest.mark.parametrize("t, n", [(120, 40), (60, 150)])
def test_factor_covariance_without_gaps_is_the_pca_decomposition(t, n):
    rng = np.random.default_rng(14)
    exposures = rng.normal(size=(n, 3))
    returns = 0.01 * (rng.normal(size=(t, 3)) @ exposures.T + 2.0 * rng.normal(size=(t, n)))
    sd = returns.std(axis=0, ddof=1)
    pca = PCA(n_components=3, svd_solver="full").fit((returns - returns.mean(axis=0)) / sd)
    # Rank-3 part of the correlation matrix, back in return units, plus the residual variances.
    expected = (pca.components_.T * pca.explained_variance_) @ pca.components_ * np.outer(sd, sd)
    np.fill_diagonal(expected, sd**2)

    cov = factor_covariance(returns, n_factors=3, min_obs=10, specific_floor=1e-9)
    np.testing.assert_allclose(cov.factor_var, pca.explained_variance_, rtol=1e-8)
    np.testing.assert_allclose(cov.matrix(), expected, rtol=1e-7, atol=1e-12)
    assert (cov.loadings.sum(axis=0) >= 0).all()


def test_factor_covariance_with_missing_data_is_positive_definite():
    rng = np.random.default_rng(15)
    complete, _, _, _ = _one_factor_returns(rng, 252, 120)
    returns = complete.copy()
    returns[rng.random(returns.shape) < 0.2] = NAN
    returns[:150, 100:110] = NAN
    returns[:, 110:115] = NAN
    returns[-20:, 110:113] = complete[-20:, 110:113]
    returns[:, 115] = 0.0
    short = np.arange(110, 116)
    estimated = np.setdiff1d(np.arange(120), short)

    cov = factor_covariance(returns, n_factors=3)
    dense = cov.matrix()
    assert np.isfinite(dense).all()
    assert np.linalg.eigvalsh(dense).min() > 0
    np.linalg.cholesky(dense)

    # Too few observations (or no variation): the medians of the estimated assets.
    medians = np.median(cov.loadings[estimated], axis=0)
    np.testing.assert_allclose(cov.loadings[short], np.tile(medians, (6, 1)))
    np.testing.assert_allclose(cov.specific_var[short], np.median(cov.specific_var[estimated]))
    total_var = np.nanvar(returns[:, estimated], axis=0, ddof=1)
    assert (cov.specific_var[estimated] >= 0.1 * total_var * (1 - 1e-12)).all()

    floored = factor_covariance(returns, n_factors=3, specific_floor=0.9)
    np.testing.assert_allclose(floored.specific_var[estimated], 0.9 * total_var)
    with pytest.raises(ValueError):
        factor_covariance(returns[:30], n_factors=3)


def test_factor_covariance_loadings_are_not_shrunk_by_missing_days():
    rng = np.random.default_rng(16)
    returns, beta, factor, sigma = _one_factor_returns(rng, 252, 400)
    gaps = rng.random((252, 200)) < 0.5
    returns[:, :200][gaps] = NAN
    cov = factor_covariance(returns, n_factors=1)

    # Zero-filled loadings would be about half as large for the assets that miss half the days.
    systematic = cov.loadings[:, 0] * np.sqrt(cov.factor_var[0]) / (beta * factor.std(ddof=1))
    assert np.median(systematic[:200]) == pytest.approx(1.0, abs=0.1)
    assert np.median(systematic[200:]) == pytest.approx(1.0, abs=0.1)
    implied = np.diag(cov.matrix()) / (beta**2 * 0.015**2 + sigma**2)
    assert np.median(implied[:200]) == pytest.approx(1.0, abs=0.1)


def test_ledoit_wolf_covariance_matches_the_published_formula():
    rng = np.random.default_rng(17)
    t, n = 90, 12
    returns = 0.01 * rng.normal(size=(t, n)) @ rng.normal(size=(n, n))
    x = returns - returns.mean(axis=0)
    sample = x.T @ x / t
    mean_var = np.trace(sample) / n
    # Ledoit and Wolf (2004): shrinkage = min(b^2, d^2) / d^2 with d^2 = ||S - m I||^2 and
    # b^2 = sum_t ||x_t x_t' - S||^2 / T^2 (squared Frobenius norms).
    d2 = ((sample - mean_var * np.eye(n)) ** 2).sum()
    b2 = sum(((np.outer(row, row) - sample) ** 2).sum() for row in x) / t**2
    shrinkage = min(b2, d2) / d2
    expected = (1.0 - shrinkage) * sample + shrinkage * mean_var * np.eye(n)
    assert 0.0 < shrinkage < 1.0

    cov = ledoit_wolf_covariance(returns)
    np.testing.assert_allclose(cov, expected, rtol=1e-8, atol=1e-14)

    returns[rng.random(returns.shape) < 0.3] = NAN
    with_gaps = ledoit_wolf_covariance(returns)
    assert np.isfinite(with_gaps).all()
    np.testing.assert_allclose(with_gaps, with_gaps.T)
    assert np.linalg.eigvalsh(with_gaps).min() > 0


# ----------------------------------------------------------------------------------- backtest


def test_run_backtest_three_periods_three_assets_by_hand():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    weights = pd.DataFrame(
        [[0.5, 0.5, -1.0], [0.5, 0.5, -1.0], [1.0, NAN, -0.5]], index=dates, columns=list("ABC")
    )
    returns = pd.DataFrame(
        [[0.10, -0.10, 0.05], [0.20, NAN, -0.10], [-0.02, 0.50, NAN]],
        index=dates,
        columns=list("ABC"),
    )
    out = run_backtest(weights, returns, cost_bps=25.0)

    # Period 1: gross = 0.05 - 0.05 - 0.05; the whole portfolio is bought, 0.5 * (0.5 + 0.5 + 1).
    # Period 2: capital is 0.95, so the positions drifted to 0.55, 0.45 and -1.05 over 0.95;
    #           the trades back to target are (0.075 + 0.025 + 0.10) / 0.95. B has no return.
    # Period 3: capital grew by 1.20, so the positions drifted to 0.60, 0.50 (B earned 0) and
    #           -0.90 over 1.20 = 0.5, 5/12 and -0.75; trades 0.5 + 5/12 + 0.25. C has no return.
    gross = np.array([-0.05, 0.20, -0.02])
    turnover = np.array([1.0, 0.5 * 0.2 / 0.95, 0.5 * (0.5 + 5 / 12 + 0.25)])
    cost = turnover * 2.0 * 25.0 / 1e4
    expected = pd.DataFrame(
        {
            "gross": gross,
            "turnover": turnover,
            "cost": cost,
            "net": gross - cost,
            "long": [1.0, 1.0, 1.0],
            "short": [-1.0, -1.0, -0.5],
            "n_long": [2, 2, 1],
            "n_short": [1, 1, 1],
            "missing": [0, 1, 1],
        },
        index=dates,
    )
    assert list(out.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(out, expected, check_dtype=False, rtol=1e-12, atol=1e-15)
    assert turnover[1] == pytest.approx(2 / 19) and turnover[2] == pytest.approx(7 / 12)
    assert out["cost"].iloc[0] == pytest.approx(0.005)

    free = run_backtest(weights, returns)
    np.testing.assert_array_equal(free["cost"], 0.0)
    np.testing.assert_array_equal(free["net"], free["gross"])
    np.testing.assert_allclose(free["turnover"], turnover)


def test_run_backtest_matches_a_loop_over_dates():
    rng = np.random.default_rng(18)
    t, n = 40, 25
    weights = rng.normal(size=(t, n)) / n
    weights[rng.random((t, n)) < 0.2] = 0.0
    returns = rng.normal(0.005, 0.08, size=(t, n))
    returns[rng.random((t, n)) < 0.1] = NAN
    frame = run_backtest(pd.DataFrame(weights), pd.DataFrame(returns), cost_bps=10.0)

    drifted = np.zeros(n)
    for row in range(t):
        earned = np.where(np.isnan(returns[row]), 0.0, returns[row])
        gross = float(weights[row] @ earned)
        turnover = 0.5 * float(np.abs(weights[row] - drifted).sum())
        assert frame["gross"].iloc[row] == pytest.approx(gross, abs=1e-14)
        assert frame["turnover"].iloc[row] == pytest.approx(turnover, abs=1e-13)
        assert frame["net"].iloc[row] == pytest.approx(gross - 2 * turnover * 1e-3, abs=1e-13)
        assert frame["missing"].iloc[row] == (np.isnan(returns[row]) & (weights[row] != 0)).sum()
        drifted = weights[row] * (1.0 + earned) / (1.0 + gross)


@pytest.mark.parametrize("long_short", [False, True])
def test_buy_and_hold_has_no_turnover_after_the_first_period(long_short):
    rng = np.random.default_rng(19)
    t, n = 30, 8
    returns = rng.normal(0.01, 0.05, size=(t, n))
    start = rng.dirichlet(np.ones(n))
    if long_short:
        start = np.concatenate([start[:4] / start[:4].sum(), -start[4:] / start[4:].sum()])

    # Hold the starting positions without trading: position values grow with the asset, and
    # capital is 1 plus the profit made so far.
    growth = np.vstack([np.ones(n), np.cumprod(1.0 + returns, axis=0)])
    value = start * growth
    capital = 1.0 + (value - start).sum(axis=1)
    weights = (value / capital[:, None])[:-1]

    out = run_backtest(pd.DataFrame(weights), pd.DataFrame(returns), cost_bps=10.0)
    assert out["turnover"].iloc[0] == pytest.approx(0.5 * np.abs(start).sum())
    np.testing.assert_allclose(out["turnover"].iloc[1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["cost"].iloc[1:], 0.0, atol=1e-14)
    np.testing.assert_allclose(np.cumprod(1.0 + out["gross"].to_numpy()), capital[1:], rtol=1e-12)


def test_run_backtest_checks_its_inputs_and_survives_a_total_loss():
    dates = pd.to_datetime(["2021-01-29", "2021-02-26"])
    weights = pd.DataFrame([[2.0, 0.0], [0.5, 0.5]], index=dates, columns=["A", "B"])
    returns = pd.DataFrame([[-0.6, 0.1], [0.1, 0.1]], index=dates, columns=["A", "B"])
    out = run_backtest(weights, returns)
    # The first period loses 120% of capital; the next portfolio is built from nothing.
    np.testing.assert_allclose(out["gross"], [-1.2, 0.1])
    np.testing.assert_allclose(out["turnover"], [1.0, 0.5])

    with pytest.raises(ValueError):
        run_backtest(weights, returns[["B", "A"]])
    with pytest.raises(ValueError):
        run_backtest(weights, returns.iloc[:1])
    with pytest.raises(ValueError):
        run_backtest(weights.iloc[::-1], returns.iloc[::-1])

    empty = run_backtest(weights.iloc[:0], returns.iloc[:0])
    assert empty.shape == (0, 9)


def test_align_builds_the_weights_frame():
    later, earlier = pd.Timestamp("2021-02-26"), pd.Timestamp("2021-01-29")
    by_date = {
        later: pd.Series({"B": 0.4, "C": -0.4}),
        earlier: pd.Series({"C": NAN, "A": 1.0, "Z": 0.0}),
    }
    frame = align(by_date, ["A", "B", "C", "D"])
    expected = pd.DataFrame(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.4, -0.4, 0.0]],
        index=pd.DatetimeIndex([earlier, later]),
        columns=["A", "B", "C", "D"],
    )
    pd.testing.assert_frame_equal(frame, expected)

    returns = pd.DataFrame(0.1, index=frame.index, columns=frame.columns)
    np.testing.assert_allclose(run_backtest(frame, returns)["gross"], [0.1, 0.0], atol=1e-15)

    with pytest.raises(KeyError):
        align({later: pd.Series({"B": 0.4, "Z": -0.4})}, ["A", "B", "C", "D"])
