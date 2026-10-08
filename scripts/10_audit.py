"""Properties of the data that decide what can be believed (results/data_audit/).

    python scripts/10_audit.py [--topics lda_all] [--freq M]

firm_coverage.csv   At the end of each June and December: the share of the universe (by
                    number, by market value, and among the 50 largest stocks) with at least
                    one article filed under its ticker in the last 21 and 252 trading days.
                    The news was collected ticker by ticker at the end of the sample, so
                    companies that had disappeared by then are missing altogether and the
                    most covered ones have only their latest articles.
stale_rows.csv      Per frequency and split, separately for the formation days on which the
                    return file holds the whole universe and those on which it does not
                    (Market.complete): the share of stock-periods without a daily return in
                    the last 21 trading days, and their average return relative to the
                    cross-section. On the incomplete days these are the stocks about to
                    leave the list of the largest stocks; the panels leave those days out.
donor_rows.csv      Per feature block of the stored panel and split: the share of
                    stock-periods whose raw features were missing, so that the block holds
                    the features of another stock (ftm/panel.py). For those rows the block
                    is noise by construction.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config
from ftm.data.returns import Market
from ftm.features import blocks
from ftm.features.firm_news import FirmNews
from ftm.panel import Panel


def firm_coverage(market: Market, firm: pd.DataFrame) -> pd.DataFrame:
    news = FirmNews(market, firm, n_topics=int(firm["topic"].max()) + 1)
    daily = np.zeros((len(market.dates), len(market.permnos)), dtype=np.float32)
    np.add.at(daily, (news.day, news.stock), news.n)
    total = np.vstack([np.zeros((1, daily.shape[1]), dtype=np.float32), np.cumsum(daily, axis=0)])
    listed = np.isin(np.arange(len(market.permnos)), market.tickers.to_numpy())
    rows = []
    for f in market.formation_days("M"):
        date = market.dates[f]
        if date < pd.Timestamp(config.SPLITS["train"][0]) or date.month not in (6, 12):
            continue
        members = market.eligible(f)
        last = f - config.NEWS_LAG_DAYS
        month = total[last + 1] - total[max(0, last - config.FIRM_NEWS_WINDOW + 1)] > 0
        year = total[last + 1] - total[max(0, last - 252 + 1)] > 0
        value = np.where(members, np.nan_to_num(market.me[f]), 0.0)
        largest = np.argsort(-value)[:50]
        rows.append({"date": date.date(), "stocks": int(members.sum()),
                     "with_a_ticker": listed[members].mean(),
                     "news_in_21_days": month[members].mean(),
                     "news_in_252_days": year[members].mean(),
                     "market_value_with_news_in_252_days": value[year].sum() / value.sum(),
                     "largest_50_with_news_in_252_days": year[largest].mean()})
    return pd.DataFrame(rows)


def stale_rows(market: Market) -> pd.DataFrame:
    rows = []
    for freq in config.FREQUENCIES:
        index = Panel.skeleton(market, freq, complete_only=False).index
        days = index["form_day"].unique()
        index["complete"] = index["form_day"].map(dict(zip(days, map(market.complete, days))))
        for (split, complete), g in index.groupby(["split", "complete"]):
            rows.append({"freq": freq, "split": split,
                         "return_file": "complete" if complete else "incomplete",
                         "periods": g["date"].nunique(), "rows": len(g),
                         "share_stale": g["stale"].mean(),
                         "mean_relative_return_stale": g.loc[g["stale"], "y_rel"].mean(),
                         "mean_relative_return_others": g.loc[~g["stale"], "y_rel"].mean()})
    return pd.DataFrame(rows)


def donor_rows(market: Market, freq: str) -> pd.DataFrame:
    folder = config.PANELS / freq
    names = list(json.loads((folder / "names.json").read_text())["names"])
    index = pd.read_parquet(folder / "index.parquet", columns=["form_day", "col", "split"])
    day, col = index["form_day"].to_numpy(), index["col"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    rows = []
    for name, (maker, _) in blocks.of_panel(market, names, freq).items():
        missing = np.concatenate([np.isnan(maker(int(day[a]))[col[a:b]]).any(axis=1)
                                  for a, b in zip(bounds[:-1], bounds[1:])])
        for split in config.SPLITS:
            part = missing[(index["split"] == split).to_numpy()]
            rows.append({"freq": freq, "block": name, "split": split, "rows": len(part),
                         "share_donor": part.mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", default="lda_all",
                    help="topic model whose firm_daily.parquet counts the articles "
                         "(lda_all counts every article)")
    ap.add_argument("--freq", default="M", help="the panel whose donor rows are counted")
    a = ap.parse_args()
    out = config.RESULTS / "data_audit"
    out.mkdir(parents=True, exist_ok=True)
    market = Market.load()
    coverage = firm_coverage(market, pd.read_parquet(config.TOPICS / a.topics / "firm_daily.parquet"))
    coverage.to_csv(out / "firm_coverage.csv", index=False, float_format="%.4f")
    stale = stale_rows(market)
    stale.to_csv(out / "stale_rows.csv", index=False, float_format="%.5f")
    print(coverage.round(3).to_string(index=False))
    print(stale.round(4).to_string(index=False))
    if (config.PANELS / a.freq / "names.json").exists():
        donors = donor_rows(market, a.freq)
        donors.to_csv(out / "donor_rows.csv", index=False, float_format="%.4f")
        print(donors.pivot(index="block", columns="split", values="share_donor")
              [list(config.SPLITS)].round(3).to_string())


if __name__ == "__main__":
    main()
