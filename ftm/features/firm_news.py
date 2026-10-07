"""Firm-level news features: how much news a firm had, and about what.

For a portfolio formed at the close of day f the features describe the
articles filed under the firm's ticker during the `window` trading days up to
day f - lag: the log number of articles and the share of them in each topic
(zero when the firm had no news). Tickers are matched to stocks through the
stock's latest ticker, because the news sources file old articles under the
ticker a company has today.

Coverage. The news was collected ticker by ticker at the end of the sample: a
company that had disappeared by then has no articles at all, and the most
written-about companies have only their most recent ones. "No news" therefore
often means "not in the source", and for a company that later disappeared that
is information about the future. A firm without a single article in the last
`coverage` trading days is treated as not covered: its features are missing
(NaN), and the panel then gives it another firm's features at random.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


class FirmNews:
    """Article counts by (trading day, stock, topic), ready for window sums."""

    def __init__(self, market, firm_daily: pd.DataFrame, n_topics: int):
        col = firm_daily["ticker"].map(market.tickers)
        day = np.searchsorted(market.dates.values, firm_daily["date"].to_numpy(), side="left")
        keep = col.notna().to_numpy() & (day < len(market.dates))
        order = np.argsort(day[keep], kind="stable")
        self.day = day[keep][order]
        self.stock = col.to_numpy()[keep][order].astype(np.int64)
        self.topic = firm_daily["topic"].to_numpy()[keep][order].astype(np.int64)
        self.n = firm_daily["n"].to_numpy()[keep][order].astype(np.float64)
        self.shape = (len(market.permnos), n_topics)
        self.matched = float(keep.mean())

    def names(self) -> list[str]:
        return ["news_count"] + [f"news_t{k:02d}" for k in range(self.shape[1])]

    def features(self, f: int, window: int = config.FIRM_NEWS_WINDOW,
                 lag: int = config.NEWS_LAG_DAYS,
                 coverage: int = config.FIRM_COVERAGE_WINDOW) -> np.ndarray:
        """(N, 1 + K): log(1 + articles) and topic shares over days f-lag-window+1 .. f-lag;
        NaN for firms without an article in the `coverage` days up to f-lag."""
        last = f - lag
        a = np.searchsorted(self.day, last - window + 1, side="left")
        b = np.searchsorted(self.day, last, side="right")
        counts = np.zeros(self.shape)
        np.add.at(counts, (self.stock[a:b], self.topic[a:b]), self.n[a:b])
        total = counts.sum(axis=1, keepdims=True)
        shares = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
        out = np.column_stack([np.log1p(total), shares]).astype(np.float32)
        a = np.searchsorted(self.day, last - coverage + 1, side="left")
        out[np.bincount(self.stock[a:b], minlength=self.shape[0]) == 0] = np.nan
        return out
