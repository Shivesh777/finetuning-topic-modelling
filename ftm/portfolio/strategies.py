"""The portfolio rules that turn one period's forecasts into weights, and their backtest.

    ls_ew     long the top decile of forecasts, short the bottom decile, equal weights
    ls_vw     the same two legs, weighted by market value
    ls_buf    ls_ew with a buffer: a stock enters a leg in the top (bottom) decile and
              stays until it leaves the top (bottom) fifth, so a signal that moves a
              stock within the fifth does not trade (Novy-Marx and Velikov 2016)
    rank      every stock, weight proportional to its forecast rank minus the mean rank
    mvo       mean-variance weights for the forecasts under a factor covariance of the
              last year's daily returns, dollar neutral, with a cap per stock
    long_ew   the top decile only, equal weights
    long_vw   the top decile only, weighted by market value
    univ_ew   benchmark: every stock of the universe, equal weights
    univ_vw   benchmark: every stock of the universe, weighted by market value

The long-short rules have longs summing to +1 and shorts to -1 and their return
is an excess return as it stands; the long-only rules and the benchmarks are
fully invested and their excess return is the return minus the risk-free rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .backtest import run_backtest
from .risk import factor_covariance
from .weights import mean_variance, quantile_long_short, rank_weighted, top_quantile

LONG_SHORT = ("ls_ew", "ls_vw", "rank", "mvo", "ls_buf")
LONG_ONLY = ("long_ew", "long_vw")
BENCHMARKS = ("univ_ew", "univ_vw")
STRATEGIES = LONG_SHORT + LONG_ONLY


def _untied(score: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 of the scores, stocks with equal scores ranked by `order` (NaN stays NaN)."""
    position = np.lexsort((order, score))                          # NaN sorts last
    rank = np.empty(len(score))
    rank[position] = np.arange(len(score))
    return np.where(np.isfinite(score), rank, np.nan)


def weights_for(strategy: str, score: np.ndarray, cap: np.ndarray, cov=None,
                q: int = config.QUANTILES, order: np.ndarray | None = None) -> np.ndarray:
    """Weights of one strategy for one cross-section of forecasts.

    A forecast that is the same for every stock expresses no view: the long-short rules
    then hold nothing and the long-only rules hold the universe.

    Ties. A coarse forecast (a boosted model with a single split has two values) leaves
    many stocks tied at the edge of a decile. They are ranked by `order`, one arbitrary
    number per stock that backtests() keeps fixed through time, so a forecast that does
    not change does not trade. Array position would not do (stocks are stored by
    identifier, roughly by listing date), and a fresh draw every period would make a
    coarse forecast trade at random."""
    n = len(score)
    if strategy in BENCHMARKS:
        return top_quantile(np.zeros(n), 1, cap if strategy == "univ_vw" else None)
    cap = cap if strategy.endswith("_vw") else None
    finite = score[np.isfinite(score)]
    if finite.size < 2 or finite.max() == finite.min():
        return np.zeros(n) if strategy in LONG_SHORT else top_quantile(np.zeros(n), 1, cap)
    if strategy == "rank":
        return rank_weighted(score)
    if strategy == "mvo":
        return mean_variance(score - finite.mean(), cov, gross=2.0,
                             max_abs=config.MVO_MAX_WEIGHT, neutral=np.ones(n))
    ranked = _untied(score, np.random.default_rng(config.SEED).random(n) if order is None else order)
    if strategy in ("ls_ew", "ls_vw"):
        return quantile_long_short(ranked, q, cap)
    if strategy in ("long_ew", "long_vw"):
        return top_quantile(ranked, q, cap)
    raise ValueError(f"unknown strategy {strategy!r}")


def buffered_long_short(score: np.ndarray, was_long: np.ndarray, was_short: np.ndarray,
                        order: np.ndarray, q: int = config.QUANTILES,
                        q_hold: int = config.BUFFER_QUANTILES):
    """Weights of ls_buf for one cross-section, and its two legs as boolean masks.

    A stock is long if it is among the floor(n / q) highest forecasts, or if it was long
    in the previous period (was_long) and is still among the floor(n / q_hold) highest;
    the short leg mirrors this at the bottom. n counts the stocks with a forecast, ties
    are ranked by `order` as in weights_for, and each leg is equally weighted, +1 and -1
    in total. With nothing held before, the legs are those of ls_ew. A flat forecast
    holds nothing and forgets the previous legs."""
    n = len(score)
    none = np.zeros(n, dtype=bool)
    finite = np.isfinite(score)
    if finite.sum() < 2 or np.nanmax(score) == np.nanmin(score):
        return np.zeros(n), none, none.copy()
    rank = _untied(score, order)                    # 0 = lowest forecast, NaN without one
    valid = int(finite.sum())
    enter, hold = max(1, valid // q), max(1, valid // q_hold)
    with np.errstate(invalid="ignore"):
        long = (rank >= valid - enter) | (was_long & (rank >= valid - hold))
        short = (rank < enter) | (was_short & (rank < hold))
    w = np.zeros(n)
    w[long] = 1.0 / long.sum()
    w[short] = -1.0 / short.sum()
    return w, long, short


def covariances(frame: pd.DataFrame, market) -> list:
    """Factor covariance of the stocks of each formation date, from the daily returns of the
    config.MVO_WINDOW trading days up to that date. One entry per date, in date order.

    frame: the rows of the panel index (date, form_day, col), sorted by date."""
    row = np.unique(frame["date"].to_numpy(), return_inverse=True)[1]
    bounds = np.flatnonzero(np.r_[True, row[1:] != row[:-1], True])
    form, col = frame["form_day"].to_numpy(), frame["col"].to_numpy()
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        days = slice(max(0, int(form[a]) - config.MVO_WINDOW + 1), int(form[a]) + 1)
        daily = np.where(market.observed[days][:, col[a:b]], market.ret[days][:, col[a:b]], np.nan)
        out.append(factor_covariance(daily, config.MVO_FACTORS))
    return out


def backtests(frame: pd.DataFrame, strategies=STRATEGIES, covs: list | None = None,
              cost_bps: float = config.COST_BPS, q: int = config.QUANTILES) -> pd.DataFrame:
    """Per-period results of each strategy, starting from cash.

    frame: one row per (formation date, stock), sorted by date, with the columns date,
    permno, pred (the forecast), ret (the return of the holding period that follows), rf
    and me. covs: the output of covariances() for the same rows, needed by "mvo".
    Returns the columns of run_backtest plus strategy, date, rf and excess: the net
    return, minus the risk-free rate for the fully invested portfolios. The result of a
    strategy does not depend on which other strategies are evaluated with it."""
    dates, row = np.unique(frame["date"].to_numpy(), return_inverse=True)
    stocks, col = np.unique(frame["permno"].to_numpy(), return_inverse=True)
    returns = np.full((len(dates), len(stocks)), np.nan)
    returns[row, col] = frame["ret"].to_numpy()
    pred, cap = frame["pred"].to_numpy(dtype=float), frame["me"].to_numpy(dtype=float)
    weights = {s: np.zeros_like(returns) for s in strategies}
    bounds = np.flatnonzero(np.r_[True, row[1:] != row[:-1], True])
    order = np.random.default_rng(config.SEED).random(len(stocks))      # how ties are broken
    was_long = np.zeros(len(stocks), dtype=bool)                         # legs of ls_buf
    was_short = np.zeros(len(stocks), dtype=bool)
    for t, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        cols = col[a:b]
        for s in strategies:
            if s == "ls_buf":
                w, long, short = buffered_long_short(pred[a:b], was_long[cols], was_short[cols],
                                                     order[cols], q)
                was_long[:], was_short[:] = False, False
                was_long[cols], was_short[cols] = long, short
                weights[s][t, cols] = w
                continue
            weights[s][t, cols] = weights_for(s, pred[a:b], cap[a:b],
                                              covs[t] if s == "mvo" else None, q, order[cols])
    rf = frame["rf"].to_numpy()[bounds[:-1]]
    held = pd.DataFrame(returns, index=dates, columns=stocks)
    out = []
    for s in strategies:
        result = run_backtest(pd.DataFrame(weights[s], index=dates, columns=stocks), held, cost_bps)
        result["rf"] = rf
        result["excess"] = result["net"] - (0.0 if s in LONG_SHORT else rf)
        result.insert(0, "strategy", s)
        out.append(result.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)
