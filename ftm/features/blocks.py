"""The feature blocks of a panel, by name.

    price                  beta, volatility, momentum, reversal, size (no text)
    narr_<model>           exposure of the stock's return to each topic's attention shock
    narr_<model>_placebo   the same exposures to the attention shocks of another date: what
                           an exposure looks like when the news carries no information
    firm_<model>           amount and topic mix of the firm's own news

<model> is a folder of data/topics/. A block is a function from a formation day to the raw
features of every stock, NaN where they cannot be computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..topics import attention
from . import firm_news, narrative, price


def makers(market, topics, placebo=(), with_price: bool = True) -> dict:
    """name -> (maker, feature names), in the order in which a panel stores the blocks:
    price, then for each topic model its narrative block, the placebo of that block if it
    is in `placebo`, and its firm-news block."""
    unknown = sorted(set(placebo) - set(topics))
    if unknown:
        raise ValueError(f"a placebo needs its topic model among the topics: {unknown}")
    out = {}
    if with_price:
        out["price"] = (lambda f: price.price_features(market, f), list(price.NAMES))
    for model in topics:
        sums = attention.load_daily_sums(config.TOPICS / model / "daily_sums.parquet")
        k = sums.shape[1] - 1
        shocks = narrative.attention_shocks(narrative.trading_day_attention(sums, market.dates))
        out[f"narr_{model}"] = (lambda f, s=shocks: narrative.exposures(market, s, f),
                                [f"narr_{model}_t{j:02d}" for j in range(k)])
        if model in placebo:                          # the same shocks, half a year out of step
            shifted = np.roll(shocks, config.PLACEBO_SHIFT, axis=0)
            out[f"narr_{model}_placebo"] = (lambda f, s=shifted: narrative.exposures(market, s, f),
                                            [f"narr_{model}_placebo_t{j:02d}" for j in range(k)])
        firm_path = config.TOPICS / model / "firm_daily.parquet"
        if firm_path.exists():
            news = firm_news.FirmNews(market, pd.read_parquet(firm_path), k)
            out[f"firm_{model}"] = (news.features, [f"firm_{model}_{n}" for n in news.names()])
    return out


def of_panel(market, names) -> dict:
    """The makers of the blocks that a stored panel holds, from their names."""
    narr = [n[len("narr_"):] for n in names if n.startswith("narr_")]
    placebo = [n[:-len("_placebo")] for n in narr if n.endswith("_placebo")]
    topics = [n for n in narr if not n.endswith("_placebo")]
    built = makers(market, topics, placebo, with_price="price" in names)
    return {name: built[name] for name in names}
