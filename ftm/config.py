"""Project settings: paths, sample periods and the parameters the stages share.

Not here: the hyperparameters of a model (the `defaults` and `grid()` of its class in
ftm/models/) and the constants of the report (top of ftm/evaluation/report.py).
REFERENCE.md lists all three.
"""
from __future__ import annotations

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
# FTM_DATA and FTM_RESULTS move the two output folders (the tests use this).
DATA = Path(os.environ.get("FTM_DATA", ROOT / "data"))
# Downloads and extraction shards (several GB) live outside the repository.
RAW = Path(os.environ.get("FTM_CACHE", Path.home() / ".cache" / "ftm"))
NEWS = DATA / "news"               # de-duplicated article tables (third-party text)
TOPICS = DATA / "topics"           # one folder per topic model
CRSP = DATA / "crsp"               # licensed return data (never committed)
PANELS = DATA / "panels"           # stock-period feature panels (derived from CRSP)
RESULTS = Path(os.environ.get("FTM_RESULTS", ROOT / "results"))

# --- sample --------------------------------------------------------------------------
NEWS_START = "2012-01-01"          # Nasdaq.com coverage is too thin before 2012
NEWS_END = "2024-12-31"
# Scrape dumps and holes of the source: days whose article count says nothing about that day.
NEWS_BAD_DAYS = ("2023-12-16",)
# A period belongs to the split of the day its return starts.
SPLITS = {
    "train": ("2013-01-01", "2019-12-31"),
    "validation": ("2020-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2024-12-31"),
}
FREQUENCIES = ("M", "2W", "W")     # monthly, biweekly, weekly rebalancing
# Our return file (scripts/00_prepare_crsp.py) has daily returns of every stock up to this
# year. In later years it has them only for the stocks of that calendar year's June list.
# Between January and the day the list changes, the stocks with data are then exactly those
# that will still be on the next list: any use of those days selects on the future, so the
# panels leave them out. Set to None with a complete file (scripts/00_pull_crsp.py).
DAILY_RETURNS_COMPLETE_UNTIL = 2016
NEWS_LAG_DAYS = 1                  # article dates carry no time of day: use news up to t-1

# --- text ----------------------------------------------------------------------------
LEAD_CHARS = 500                   # an article is its title plus the first LEAD_CHARS of body
MIN_BODY_CHARS = 400
VOCAB_SIZE = 10_000                # unigrams and bigrams, built on training-period text only
VOCAB_MIN_DF = 20
VOCAB_MAX_DF = 0.4
FIT_ARTICLES_PER_DAY = 60          # per-day sample of training articles used to FIT topic models
N_TOPICS = 60
SEED = 0
ATTENTION_ARTICLES_PER_DAY = 150   # per-day sample of articles that enter daily attention
ENCODER = "sentence-transformers/paraphrase-MiniLM-L3-v2"
ENCODER_MAX_TOKENS = 64            # headline and the first sentences of the lead
TOPIC_TEMPERATURE = 0.07           # softmax temperature of the article-to-topic weights

# --- features ------------------------------------------------------------------------
UNIVERSE_SIZE = 1000               # largest stocks by market value at the end of June
EXPOSURE_WINDOW = 252              # trading days of the covariance with attention shocks
EXPOSURE_HALFLIFE = 60
EXPOSURE_MIN_OBS = 120
SHOCK_WINDOW = 5                   # attention shock = attention minus its trailing mean
FIRM_NEWS_WINDOW = 21              # trading days of a firm's own news
FIRM_MAX_TICKERS = 3               # firm news = articles filed under at most this many tickers
FIRM_COVERAGE_WINDOW = 126         # a firm with no article for this long is not in the news source
# What a stock with a missing feature gets: "donor" = the complete feature row of a random
# other stock of the day, so that nothing marks it; "median" = the median rank, which lets a
# tree model see that data are missing and, in our data, see the future (see ftm/panel.py).
MISSING_FEATURES = "donor"
PLACEBO_SHIFT = 126                # trading days by which placebo attention shocks are displaced

# --- models --------------------------------------------------------------------------
TUNE_TOLERANCE = 1e-5              # keep the simpler setting unless validation MSE falls by more
                                   # (0.001 percentage points of R2)

# --- portfolios ----------------------------------------------------------------------
QUANTILES = 10
COST_BPS = 10.0                    # one-way cost per dollar traded
MVO_WINDOW = 252                   # trading days of the covariance used by mean-variance weights
MVO_FACTORS = 5
MVO_MAX_WEIGHT = 0.01              # cap on a stock's absolute weight (gross exposure is 2)


def split_of(dates) -> "pd.Series":
    """Split name ("train", "validation", "test" or "") of each date."""
    import pandas as pd

    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    out = pd.Series("", index=dates.index, dtype=object)
    for name, (start, end) in SPLITS.items():
        out[(dates >= start) & (dates <= end)] = name
    return out
