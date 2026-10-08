"""Firm-level news tone: whether a firm's recent news was good or bad, and how it reads.

The topic features say what a firm is in the news about. These blocks say how the
news sounded, from the article scores of ftm/topics/sentiment.py:

    tone_fb   FinBERT tone of the headlines, P(positive) - P(negative) averaged over
              the firm's articles of the last 1, 5, 21 and 252 trading days, the
              surprise (21-day tone minus 252-day tone), and the amount of news: log
              articles of the last 5 and 21 days and the abnormal amount (21-day count
              against the firm's 252-day rate)
    tone_lm   Loughran-McDonald tone of headline and lead, (positive - negative words)
              / (positive + negative words) over the same windows, and its surprise
    emb_fb    FinBERT's headline vectors (principal components) averaged over the
              articles since the last rebalancing: the inputs of the embedding
              regression of Chen, Kelly and Xiu (2022), with the models of
              ftm/models/ in the role of their linear forecaster
    npx       news and past returns together: the 5- and 21-day return of stocks with
              news in that window and of stocks without (Chan 2003: large moves on
              news drift, moves without news reverse), and the returns times the tone

Timing and coverage are those of ftm/features/firm_news.py: a portfolio formed at the
close of day f uses articles dated up to day f - lag (news of a weekend counts on the
next trading day), filed under at most config.FIRM_MAX_TICKERS tickers and matched to
a stock through its latest ticker; a firm without a single article in the
config.FIRM_COVERAGE_WINDOW days up to f - lag is not covered and gets NaN (the
panel then gives it another stock's features). A covered firm without news in a
window has a tone of zero, the tone of a neutral article.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import price

WINDOWS = (1, 5, 21)
BASELINE = 252
PERIOD_DAYS = {"W": 5, "2W": 10, "M": 21}     # trading days of a holding period

FB_NAMES = [f"fb_tone_{w}" for w in WINDOWS] + [f"fb_tone_{BASELINE}", "fb_surprise_21",
                                                 "news_log_n5", "news_log_n21", "news_abnormal_21"]
LM_NAMES = [f"lm_tone_{w}" for w in WINDOWS] + [f"lm_tone_{BASELINE}", "lm_surprise_21"]
NPX_NAMES = ["rev5_news", "rev5_quiet", "rev21_news", "rev21_quiet", "rev5_x_fbtone5",
             "rev21_x_fbtone21"]


class FirmTone:
    """Article scores by (trading day, stock), ready for window sums."""

    def __init__(self, market, dates, links: pd.DataFrame, fb_tone: np.ndarray,
                 lm: np.ndarray, emb: np.ndarray | None = None):
        """dates (n,) of the scored articles; links: position (0..n-1 into the scored
        articles) and ticker; fb_tone (n,); lm (n, 3) words, positive, negative; emb (n, k)."""
        col = links["ticker"].map(market.tickers)
        article = links["position"].to_numpy()
        day = np.searchsorted(market.dates.values, np.asarray(dates)[article], side="left")
        keep = col.notna().to_numpy() & (day < len(market.dates))
        order = np.argsort(day[keep], kind="stable")
        self.day = day[keep][order]
        self.stock = col.to_numpy()[keep][order].astype(np.int64)
        self.article = article[keep][order]
        self.fb = np.asarray(fb_tone, dtype=np.float64)
        self.lm_pos = np.asarray(lm[:, 1], dtype=np.float64)
        self.lm_neg = np.asarray(lm[:, 2], dtype=np.float64)
        self.emb = emb
        self.n = len(market.permnos)
        self.matched = float(keep.mean())

    @classmethod
    def load(cls, market, folder: str | Path | None = None, with_embeddings: bool = True):
        from ..data import news

        folder = Path(folder) if folder is not None else config.SENTIMENT
        ids = np.load(folder / "ids.npy")
        index = news.load_articles(columns=["article_id", "date"])
        dates = index.set_index("article_id").loc[ids, "date"].to_numpy()
        links = pd.read_parquet(config.NEWS / "links.parquet")
        links = links[np.isin(links["article_id"].to_numpy(), ids)]
        links = pd.DataFrame({"position": np.searchsorted(ids, links["article_id"].to_numpy()),
                              "ticker": links["ticker"].to_numpy()})
        probs = np.load(folder / "finbert_probs.npy").astype(np.float64)
        emb = np.load(folder / "finbert_pca.npy") if with_embeddings else None
        return cls(market, dates, links, probs[:, 0] - probs[:, 1], np.load(folder / "lm.npy"),
                   emb)

    # ----------------------------------------------------------------- helpers
    def _rows(self, last: int, window: int) -> slice:
        a = np.searchsorted(self.day, last - window + 1, side="left")
        b = np.searchsorted(self.day, last, side="right")
        return slice(a, b)

    def _sum(self, rows: slice, values=None) -> np.ndarray:
        return np.bincount(self.stock[rows], weights=values, minlength=self.n).astype(np.float64)

    def _mean(self, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def _uncovered(self, last: int) -> np.ndarray:
        return self._sum(self._rows(last, config.FIRM_COVERAGE_WINDOW)) == 0

    def _windows(self, last: int, score):
        """{window: (count, mean score)} for WINDOWS and BASELINE; score(rows) -> (num, den)."""
        out = {}
        for w in WINDOWS + (BASELINE,):
            rows = self._rows(last, w)
            num, den = score(rows)
            out[w] = (self._sum(rows), self._mean(num, den))
        return out

    # ----------------------------------------------------------------- blocks
    def finbert(self, f: int, lag: int = config.NEWS_LAG_DAYS) -> np.ndarray:
        """(N, 8) in the order of FB_NAMES."""
        last = f - lag
        w = self._windows(last, lambda r: (self._sum(r, self.fb[self.article[r]]), self._sum(r)))
        n5, n21, n252 = w[5][0], w[21][0], w[BASELINE][0]
        surprise = np.where(n21 > 0, w[21][1] - w[BASELINE][1], 0.0)
        abnormal = np.log1p(n21) - np.log1p(n252 * 21 / BASELINE)
        out = np.column_stack([w[1][1], w[5][1], w[21][1], w[BASELINE][1], surprise,
                               np.log1p(n5), np.log1p(n21), abnormal])
        out[self._uncovered(last)] = np.nan
        return out.astype(np.float32)

    def lm(self, f: int, lag: int = config.NEWS_LAG_DAYS) -> np.ndarray:
        """(N, 5) in the order of LM_NAMES."""
        last = f - lag

        def score(rows):
            pos = self._sum(rows, self.lm_pos[self.article[rows]])
            neg = self._sum(rows, self.lm_neg[self.article[rows]])
            return pos - neg, pos + neg

        w = self._windows(last, score)
        surprise = np.where(w[21][0] > 0, w[21][1] - w[BASELINE][1], 0.0)
        out = np.column_stack([w[1][1], w[5][1], w[21][1], w[BASELINE][1], surprise])
        out[self._uncovered(last)] = np.nan
        return out.astype(np.float32)

    def embedding(self, f: int, window: int, lag: int = config.NEWS_LAG_DAYS) -> np.ndarray:
        """(N, k) mean headline vector of the last `window` trading days (zero without news)."""
        last = f - lag
        rows = self._rows(last, window)
        vectors = np.asarray(self.emb[self.article[rows]], dtype=np.float64)
        total = np.column_stack([self._sum(rows, vectors[:, j]) for j in range(vectors.shape[1])])
        count = self._sum(rows)
        out = np.divide(total, count[:, None], out=np.zeros_like(total), where=count[:, None] > 0)
        out[self._uncovered(last)] = np.nan
        return out.astype(np.float32)

    def interactions(self, market, f: int, lag: int = config.NEWS_LAG_DAYS) -> np.ndarray:
        """(N, 6) in the order of NPX_NAMES."""
        p = price.price_features(market, f).astype(np.float64)
        rev21, rev5 = p[:, price.NAMES.index("rev_21")], p[:, price.NAMES.index("rev_5")]
        fb = self.finbert(f, lag).astype(np.float64)
        n5, n21 = np.expm1(fb[:, 5]), np.expm1(fb[:, 6])
        tone5, tone21 = fb[:, 1], fb[:, 2]
        return np.column_stack([np.where(n5 > 0, rev5, 0.0), np.where(n5 > 0, 0.0, rev5),
                                np.where(n21 > 0, rev21, 0.0), np.where(n21 > 0, 0.0, rev21),
                                rev5 * tone5, rev21 * tone21]).astype(np.float32)
