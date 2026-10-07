"""Backtest of a sequence of target weights against holding-period returns.

Conventions
-----------
* Row t of ``weights`` holds the target weights formed at date t and held over the period that
  STARTS at t; row t of ``returns`` holds each asset's simple return over that same period, as a
  decimal fraction. Both frames have the same index (sorted ascending) and the same columns.
* Weights are fractions of capital at the start of the period (positive long, negative short).
  A NaN weight is a weight of 0.
* A held asset with a NaN return earns 0 over that period and is counted in ``missing``.
* All outputs are per period and as decimal fractions of capital; nothing is compounded or
  annualised here.
* Costs: ``cost_bps`` is the one-way cost in basis points per dollar traded, charged on every
  dollar bought or sold, including the build-up of the first portfolio.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = ["run_backtest", "align"]


def run_backtest(
    weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 0.0
) -> pd.DataFrame:
    """Per-period return, turnover and cost of holding ``weights`` over ``returns``.

    One row per date with the columns: gross = sum_i w_it r_it; turnover = 0.5 * sum_i
    abs(w_it - d_it), where d_it = w_i,t-1 (1 + r_i,t-1) / (1 + gross_t-1) is last period's
    position after it drifted with returns, as a fraction of current capital (d = 0 in the first
    period, and after a period that lost all capital, gross <= -1); cost = 2 * turnover *
    cost_bps / 1e4; net = gross - cost; long and short = the sums of the positive and of the
    negative weights; n_long and n_short = their counts; missing = the number of held assets
    without a return. The drift uses gross returns, so the cost paid is not taken out of the
    capital base.
    """
    if not (weights.index.equals(returns.index) and weights.columns.equals(returns.columns)):
        raise ValueError("weights and returns must have the same index and the same columns")
    if not weights.index.is_monotonic_increasing:
        raise ValueError("the index of formation dates must be sorted in ascending order")
    w = weights.to_numpy(dtype=float)
    w = np.where(np.isnan(w), 0.0, w)
    r = returns.to_numpy(dtype=float)
    absent = ~np.isfinite(r)
    r = np.where(absent, 0.0, r)

    gross = np.einsum("ti,ti->t", w, r)
    growth = 1.0 + gross[:-1]
    per_capital = np.divide(1.0, growth, out=np.zeros_like(growth), where=growth > 0)
    drifted = np.zeros_like(w)
    np.multiply(w[:-1], 1.0 + r[:-1], out=drifted[1:])
    drifted[1:] *= per_capital[:, None]
    traded = np.subtract(w, drifted, out=drifted)
    turnover = 0.5 * np.abs(traded, out=traded).sum(axis=1)
    cost = 2.0 * turnover * cost_bps / 1e4
    return pd.DataFrame(
        {
            "gross": gross,
            "turnover": turnover,
            "cost": cost,
            "net": gross - cost,
            "long": np.maximum(w, 0.0).sum(axis=1),
            "short": np.minimum(w, 0.0).sum(axis=1),
            "n_long": np.count_nonzero(w > 0, axis=1),
            "n_short": np.count_nonzero(w < 0, axis=1),
            "missing": np.count_nonzero(absent & (w != 0), axis=1),
        },
        index=weights.index,
    )


def align(
    weights_by_date: Mapping[Hashable, pd.Series], columns: Iterable[Hashable]
) -> pd.DataFrame:
    """Stack {date: Series of weights indexed by asset id} into the frame ``run_backtest`` takes.

    Rows are the dates in ascending order and columns are ``columns`` in the given order. An
    asset that a Series does not mention, or mentions with NaN, gets weight 0. A non-zero weight
    on an asset that is not in ``columns`` raises KeyError, because dropping it would change the
    portfolio without notice.
    """
    columns = pd.Index(columns)
    dates = sorted(weights_by_date)
    out = np.zeros((len(dates), len(columns)))
    for row, date in enumerate(dates):
        series = weights_by_date[date]
        value = series.to_numpy(dtype=float)
        value = np.where(np.isnan(value), 0.0, value)
        position = columns.get_indexer(series.index)
        known = position >= 0
        if value[~known].any():
            unknown = list(series.index[~known & (value != 0)][:5])
            raise KeyError(f"weights at {date} on assets that are not in columns: {unknown}")
        out[row, position[known]] = value[known]
    return pd.DataFrame(out, index=pd.Index(dates), columns=columns)
