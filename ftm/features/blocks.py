"""The feature blocks of a panel, by name.

    price                  beta, volatility, momentum, reversal, size (no text)
    narr_<model>           exposure of the stock's return to each topic's attention shock
    narr_<model>_placebo   the same exposures to the attention shocks of another date: what
                           an exposure looks like when the news carries no information
    firm_<model>           amount and topic mix of the firm's own news
    narr_<model>_s<w>      narrative exposures with attention shocks over a w-day window
                           (scripts/05b_tune_shocks.py) instead of config.SHOCK_WINDOW
    tone_fb, tone_lm       tone of the firm's own news (FinBERT, Loughran-McDonald)
    emb_fb                 FinBERT headline vectors since the last rebalancing
    npx                    past returns of stocks with and without news, and times the tone

<model> is a folder of data/topics/. A block is a function from a formation day to the raw
features of every stock, NaN where they cannot be computed. The last four come from
ftm/features/tone.py and need data/sentiment/ (scripts/03d_sentiment.py).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import config
from ..topics import attention
from . import firm_news, narrative, price, tone


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


def of_panel(market, names, freq: str | None = None) -> dict:
    """The makers of the blocks that a stored panel holds, from their names (freq is
    needed for the tone blocks, whose embedding window is one holding period)."""
    narr = [n[len("narr_"):] for n in names if n.startswith("narr_")]
    placebo = [n[:-len("_placebo")] for n in narr if n.endswith("_placebo")]
    tuned = {n: re.fullmatch(r"(.+)_s(\d+)", n) for n in narr if not n.endswith("_placebo")}
    topics = [n for n, m in tuned.items() if m is None]
    built = makers(market, topics, placebo, with_price="price" in names)
    for m in filter(None, tuned.values()):
        name, maker, features = narrative_maker(market, m.group(1), int(m.group(2)))
        built[name] = (maker, features)
    if any(n in names for n in ("tone_fb", "tone_lm", "emb_fb", "npx")):
        if freq is None:
            raise ValueError("the tone blocks need the frequency of the panel")
        built.update(tone_makers(market, freq))
    return {name: built[name] for name in names}


def narrative_maker(market, model: str, window: int):
    """(name, maker, names) of the narrative block of a topic model with shocks over
    `window` days; the name is narr_<model> for config.SHOCK_WINDOW, else narr_<model>_s<w>."""
    sums = attention.load_daily_sums(config.TOPICS / model / "daily_sums.parquet")
    shocks = narrative.attention_shocks(narrative.trading_day_attention(sums, market.dates), window)
    name = f"narr_{model}" if window == config.SHOCK_WINDOW else f"narr_{model}_s{window}"
    return name, (lambda f: narrative.exposures(market, shocks, f)), \
        [f"{name}_t{j:02d}" for j in range(shocks.shape[1])]


def tone_makers(market, freq: str, folder=None) -> dict:
    """name -> (maker, feature names) of the blocks of ftm/features/tone.py."""
    news = tone.FirmTone.load(market, folder)
    window = tone.PERIOD_DAYS[freq]
    k = news.emb.shape[1]
    return {
        "tone_fb": (news.finbert, list(tone.FB_NAMES)),
        "tone_lm": (news.lm, list(tone.LM_NAMES)),
        "emb_fb": (lambda f: news.embedding(f, window), [f"emb_fb_{j:02d}" for j in range(k)]),
        "npx": (lambda f: news.interactions(market, f), list(tone.NPX_NAMES)),
    }
