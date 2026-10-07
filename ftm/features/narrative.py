"""Narrative exposures: how a stock's return co-moves with shocks to topic attention.

Attention is the share of a trading day's articles in each topic; its shock is
the deviation from the mean of the previous days. A stock's exposure to topic
k is the exponentially weighted covariance of its daily return with that shock
over the trailing window (Bybee, Kelly and Su, 2023). With `neutral=True` the
shocks are first made orthogonal to the market return inside the window, so
that the exposure is not a disguised market beta.

Article dates carry no time of day. News of day d is therefore paired with the
return of day d, and a portfolio formed at the close of day f uses pairs up to
day f - lag only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

MIN_ARTICLES = 20        # a trading day with fewer articles has no attention reading


def trading_day_attention(sums: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Attention shares on trading days from calendar-day sums of topic weights.

    News of a non-trading day counts on the next trading day. Days with fewer than
    MIN_ARTICLES articles are NaN."""
    day = np.searchsorted(dates.values, sums.index.values, side="left")
    inside = day < len(dates)
    total = sums[inside].groupby(day[inside]).sum().reindex(range(len(dates)), fill_value=0.0)
    topics = total.drop(columns="n_articles")
    theta = topics.div(topics.sum(axis=1).where(total["n_articles"] >= MIN_ARTICLES), axis=0)
    theta.index = dates
    return theta


def attention_shocks(theta: pd.DataFrame, window: int = config.SHOCK_WINDOW) -> np.ndarray:
    """(D, K) attention minus its mean over the previous `window` trading days."""
    trailing = theta.rolling(window, min_periods=max(2, window - 2)).mean().shift(1)
    return np.array(theta - trailing, dtype=np.float64)       # a writable copy


def exposures(market, shocks: np.ndarray, f: int, window: int = config.EXPOSURE_WINDOW,
              halflife: float = config.EXPOSURE_HALFLIFE, min_obs: int = config.EXPOSURE_MIN_OBS,
              lag: int = config.NEWS_LAG_DAYS, neutral: bool = True) -> np.ndarray:
    """(N, K) exposures for a portfolio formed at the close of day f."""
    end = f - lag + 1                                   # days lo .. end-1
    lo = max(0, end - window)
    z = shocks[lo:end]
    valid = ~np.isnan(z).any(axis=1)
    w = 0.5 ** ((end - 1 - np.arange(lo, end)) / halflife) * valid
    z = np.where(valid[:, None], z, 0.0)
    if w.sum() <= 0:
        return np.full((market.ret.shape[1], shocks.shape[1]), np.nan, dtype=np.float32)
    basis = [np.ones(len(z))]
    if neutral:
        basis.append(market.factors["mktrf"].to_numpy()[lo:end])
    b = np.column_stack(basis)
    bw = b * w[:, None]
    z = z - b @ np.linalg.lstsq(bw.T @ b, bw.T @ z, rcond=None)[0]     # weighted residual
    obs = market.observed[lo:end] & valid[:, None]
    r = np.where(obs, market.ret[lo:end], 0.0).astype(np.float64)
    wo = w[:, None] * obs
    wsum = wo.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rbar = (wo * r).sum(axis=0) / wsum
        cov = ((wo * (r - rbar)).T @ z) / wsum[:, None]
    cov[obs.sum(axis=0) < min_obs] = np.nan
    return cov.astype(np.float32)
