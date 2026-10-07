"""From article-topic weights to daily attention and firm-level topic counts.

Daily attention is the share of the day's articles that belongs to each topic
(the mean of the article-topic weights). Tables are stored as daily SUMS of
weights plus the number of articles, per calendar day, so that they can be
re-aggregated exactly to trading days or longer windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def topic_columns(k: int) -> list[str]:
    return [f"t{j:02d}" for j in range(k)]


def daily_sums(weights: np.ndarray, dates, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Per calendar day: number of articles and the sum of their topic weights.

    weights (n, k); dates (n,) sorted ascending; mask selects the articles to count."""
    dates = pd.DatetimeIndex(dates)
    if mask is not None:
        weights, dates = weights[mask], dates[mask]
    days, first = np.unique(dates.values, return_index=True)
    sums = np.add.reduceat(np.asarray(weights, dtype=np.float64), first, axis=0)
    out = pd.DataFrame(sums, index=pd.DatetimeIndex(days, name="date"),
                       columns=topic_columns(weights.shape[1]))
    out.insert(0, "n_articles", np.diff(np.r_[first, len(dates)]))
    return out


def save_daily_sums(sums: pd.DataFrame, path) -> None:
    sums.reset_index().to_parquet(path, index=False)


def load_daily_sums(path) -> pd.DataFrame:
    """Daily sums indexed by calendar date, as written by save_daily_sums."""
    return pd.read_parquet(path).set_index("date")


def attention(sums: pd.DataFrame) -> pd.DataFrame:
    """Daily attention shares (rows sum to one) from a table of daily sums."""
    topics = sums.drop(columns="n_articles")
    return topics.div(topics.sum(axis=1), axis=0)


def sample_per_day(dates, per_day: int, seed: int = 0) -> np.ndarray:
    """Mask keeping at most per_day randomly chosen articles of each date."""
    dates = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(seed)
    draw = rng.random(len(dates))
    rank = pd.Series(draw).groupby(dates.values).rank(method="first").to_numpy()
    return rank <= per_day


def random_halves(dates, seed: int = 0) -> np.ndarray:
    """Boolean array splitting each day's articles into two random halves."""
    dates = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(seed)
    rank = pd.Series(rng.random(len(dates))).groupby(dates.values).rank(method="first")
    return (rank.to_numpy().astype(np.int64) % 2).astype(bool)


def firm_daily(labels: np.ndarray, dates, links: pd.DataFrame) -> pd.DataFrame:
    """Number of articles by (date, ticker, topic); labels are the articles' main topics.

    links: article_id, ticker, with article_id indexing labels and dates."""
    ids = links["article_id"].to_numpy()
    frame = pd.DataFrame({"date": pd.DatetimeIndex(dates).values[ids],
                          "ticker": links["ticker"].to_numpy(),
                          "topic": labels[ids].astype(np.int16)})
    out = frame.groupby(["date", "ticker", "topic"], sort=True).size().rename("n").reset_index()
    out["n"] = out["n"].astype(np.int32)
    return out
