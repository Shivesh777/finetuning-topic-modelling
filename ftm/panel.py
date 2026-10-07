"""The stock-period panel that every forecasting model is trained on.

One row per (formation day, investable stock). Targets:

    ret     return of the stock over the holding period that follows
    y       ret minus the risk-free return
    y_rel   y minus its mean over the stocks of that formation day

Models are fitted on y_rel: with features that only rank stocks against each
other, the market's own return is noise for them. Feature blocks are stored
separately, as cross-sectional ranks scaled to [-1, 1], so that a new topic
model adds a block without rebuilding the others.

Incomplete periods. From 2017 our return file has daily returns only for the
stocks of each calendar year's June list. Between January and the day the list
changes, the stocks with data are those that will still be on the next list,
so every feature of those days is observed for a sample selected on the
future. Formation days in that part of the year are left out (see
config.DAILY_RETURNS_COMPLETE_UNTIL and Market.complete).

Missing features. Where data are missing for reasons that may depend on the
future (the news files have no articles for companies that no longer existed
when the news was collected), a model that can see that a feature is missing
sees the future. So a stock whose block has a missing value receives the
complete feature row of another, randomly drawn stock of the same day
(hot-deck imputation): its features carry no information, and nothing marks it
as missing. This hides the stocks without data; it cannot undo the selection
of the stocks with data, which is why the incomplete periods are dropped as
well. The column `stale` of the index records which stocks had no daily return
in the last 21 days; it is for diagnostics and must never be used as a feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import zlib

import numpy as np
import pandas as pd

from . import config

STALE_DAYS = 21          # a stock without a daily return for this long has stale price data


def rank_transform(x: np.ndarray) -> np.ndarray:
    """Column-wise ranks scaled to [-1, 1]; ties get their average rank, NaN becomes 0."""
    frame = pd.DataFrame(x)
    n = frame.notna().sum(axis=0).to_numpy()
    ranks = frame.rank(axis=0, method="average").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0
    scaled[:, n < 2] = 0.0
    return np.nan_to_num(scaled, nan=0.0).astype(np.float32)


def hot_deck(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rows with a missing value replaced by complete rows drawn at random from x.

    Unchanged if no row is complete (then no stock can be told apart by it either)."""
    missing = np.isnan(x).any(axis=1)
    donors = np.flatnonzero(~missing)
    if not missing.any() or not len(donors):
        return x
    x = x.copy()
    x[missing] = x[rng.choice(donors, int(missing.sum()))]
    # A copy would tie with its donor in every feature; a relative change of one in ten
    # million keeps the two apart without moving either.
    x[missing] *= 1.0 + 1e-7 * rng.standard_normal((int(missing.sum()), x.shape[1]))
    return x


@dataclass
class Panel:
    freq: str
    index: pd.DataFrame                       # date, hold_start, period, permno, col, ret, y, ...
    blocks: dict = field(default_factory=dict)        # name -> float32 (rows, features)
    names: dict = field(default_factory=dict)         # name -> feature names

    def X(self, blocks) -> np.ndarray:
        return np.hstack([self.blocks[b] for b in blocks])

    def feature_names(self, blocks) -> list[str]:
        return [n for b in blocks for n in self.names[b]]

    def rows(self, split: str) -> np.ndarray:
        return (self.index["split"] == split).to_numpy()

    # ------------------------------------------------------------- construction
    @classmethod
    def skeleton(cls, market, freq: str, start: str | None = None, end: str | None = None,
                 complete_only: bool = True):
        """Rows and targets, without features.

        complete_only: leave out formation days on which the return file does not hold
        every stock of the universe (Market.complete)."""
        forms = market.formation_days(freq)
        held, rf = market.holding_returns(forms)
        start = pd.Timestamp(start or config.SPLITS["train"][0])
        end = pd.Timestamp(end or config.SPLITS["test"][1])
        parts = []
        for p, f in enumerate(forms[:-1]):
            hold_start = market.dates[f + 1]
            if hold_start < start or hold_start > end or (complete_only and not market.complete(f)):
                continue
            cols = np.flatnonzero(market.eligible(f))
            if len(cols) < 10:
                continue
            ret = np.nan_to_num(held[p, cols], nan=0.0)
            y = ret - rf[p]
            parts.append(pd.DataFrame({
                "date": market.dates[f], "hold_start": hold_start, "period": p, "form_day": f,
                "permno": market.permnos[cols], "col": cols, "ret": ret, "y": y,
                "y_rel": y - y.mean(), "rf": rf[p], "me": market.me[f, cols].astype(np.float64),
                "stale": ~market.observed[max(0, f - STALE_DAYS + 1):f + 1][:, cols].any(axis=0)}))
        index = pd.concat(parts, ignore_index=True)
        index["split"] = config.split_of(index["hold_start"]).to_numpy()
        return cls(freq, index)

    def add_block(self, name: str, maker, names: list[str]) -> None:
        """Compute a feature block: maker(form_day) -> (N, p) raw features of all stocks.

        Stocks with a missing value in the block get another stock's row (see above)."""
        out = np.empty((len(self.index), len(names)), dtype=np.float32)
        day = self.index["form_day"].to_numpy()
        col = self.index["col"].to_numpy()
        rng = np.random.default_rng([config.SEED, zlib.crc32(name.encode())])
        bounds = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            raw = maker(int(day[a]))[col[a:b]]
            out[a:b] = rank_transform(hot_deck(raw, rng) if config.MISSING_FEATURES == "donor"
                                      else raw)
        self.blocks[name], self.names[name] = out, list(names)

    # ------------------------------------------------------------------ storage
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.index.to_parquet(folder / "index.parquet", index=False)
        for name, block in self.blocks.items():           # ranks of about a thousand stocks:
            np.save(folder / f"{name}.npy", block.astype(np.float16))    # half precision is enough
        meta = folder / "names.json"                       # keep blocks saved earlier
        names = json.loads(meta.read_text())["names"] if meta.exists() else {}
        meta.write_text(json.dumps({"freq": self.freq, "names": {**names, **self.names}}))

    @classmethod
    def load(cls, folder: str | Path, blocks=None) -> "Panel":
        folder = Path(folder)
        meta = json.loads((folder / "names.json").read_text())
        wanted = list(meta["names"]) if blocks is None else list(blocks)
        return cls(meta["freq"], pd.read_parquet(folder / "index.parquet"),
                   {b: np.load(folder / f"{b}.npy").astype(np.float32) for b in wanted},
                   {b: meta["names"][b] for b in wanted})
