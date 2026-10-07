"""Features from past returns and market value: the no-text benchmark.

All use data up to and including the formation day `f` and observed daily
returns only (never filled ones).
"""
from __future__ import annotations

import numpy as np

NAMES = ["beta", "ivol", "vol_21", "mom_12_1", "rev_21", "rev_5", "size", "max_21"]


def _compound(ret: np.ndarray, min_obs: int) -> np.ndarray:
    n = (~np.isnan(ret)).sum(axis=0)
    out = np.expm1(np.nansum(np.log1p(ret), axis=0))
    return np.where(n >= min_obs, out, np.nan)


def price_features(market, f: int) -> np.ndarray:
    """(N, 8) array in the order of NAMES for a portfolio formed at the close of day f.

    beta, ivol   slope and residual volatility of daily excess returns on the market
                 excess return over the last 252 days (at least 120 observations)
    vol_21       volatility of the last 21 daily returns
    mom_12_1     return from day f-252 to day f-21
    rev_21, rev_5  return of the last 21 and 5 days
    size         log market value
    max_21       largest daily return of the last 21 days
    """
    lo = max(0, f - 251)
    ret = np.where(market.observed[lo:f + 1], market.ret[lo:f + 1], np.nan).astype(np.float64)
    mkt = market.factors["mktrf"].to_numpy()[lo:f + 1]
    rf = market.rf[lo:f + 1]
    ok = ~np.isnan(ret)
    n = ok.sum(axis=0)
    x = np.where(ok, mkt[:, None], 0.0)
    y = np.where(ok, ret - rf[:, None], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xm, ym = x.sum(axis=0) / n, y.sum(axis=0) / n
        xc, yc = (x - xm) * ok, (y - ym) * ok
        beta = (xc * yc).sum(axis=0) / (xc * xc).sum(axis=0)
        resid = (yc - beta * xc) * ok
        ivol = np.sqrt((resid ** 2).sum(axis=0) / (n - 2))
        beta, ivol = np.where(n >= 120, beta, np.nan), np.where(n >= 120, ivol, np.nan)
        last21, last5 = ret[-21:], ret[-5:]
        n21 = (~np.isnan(last21)).sum(axis=0)
        dev = np.where(np.isnan(last21), 0.0, last21 - np.nansum(last21, axis=0) / n21)
        vol21 = np.where(n21 >= 15, np.sqrt((dev ** 2).sum(axis=0) / (n21 - 1)), np.nan)
        max21 = np.where(n21 >= 15, np.where(np.isnan(last21), -np.inf, last21).max(axis=0), np.nan)
        size = np.log(market.me[f].astype(np.float64))
    return np.column_stack([
        beta, ivol, vol21, _compound(ret[:-21], 150), _compound(last21, 15), _compound(last5, 3),
        size, max21]).astype(np.float32)
