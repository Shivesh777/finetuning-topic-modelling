"""A small artificial market in the file layout of data/crsp/, for tests.

Mimics the features of the real files that the loaders must handle: stocks
enter and leave the yearly lists, the daily file of the later years has rows
only for the stocks listed in that calendar year (so leavers lack January to
June), a monthly file covers every stock, and some stocks stop trading.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def make_market(folder, n_stocks=60, list_size=40, seed=0, start="2012-01-03", end="2015-12-31",
                full_until=2013, drift=0.0):
    """drift > 0 plants a signal: stocks differ in volatility and a stock's expected daily
    return is drift times its volatility, so volatility features forecast returns."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(start, end)
    days = days[~((days.month == 12) & (days.day == 25))]
    permnos = np.arange(10001, 10001 + n_stocks)
    mkt = rng.normal(0.0004, 0.01, len(days))
    beta = rng.uniform(0.5, 1.5, n_stocks)
    ret = beta[None, :] * mkt[:, None] + rng.normal(0, 0.015, (len(days), n_stocks))
    if drift:
        sigma = rng.uniform(0.4, 1.6, n_stocks)
        ret = beta[None, :] * mkt[:, None] + (ret - beta[None, :] * mkt[:, None]) * sigma
        ret += drift * 0.015 * sigma
    dead = permnos[-2]                                   # stops trading in mid-2014
    ret[(days >= "2014-09-10").nonzero()[0][:, None], [n_stocks - 2]] = np.nan
    me = 1e6 * np.exp(rng.normal(0, 1, n_stocks))[None, :] * np.cumprod(1 + np.nan_to_num(ret), axis=0)

    lists = {}
    for year in range(days.year.min() - 1, days.year.max() + 1):
        june = np.flatnonzero((days.year == year) & (days.month == 6))
        size = me[june[-1]] if len(june) else me[0]
        lists[year] = permnos[np.argsort(-size)[:list_size]]
    universe = pd.concat([pd.DataFrame({"formed": pd.Timestamp(y, 6, 30), "permno": m,
                                        "me": 1.0, "rank": np.arange(1, len(m) + 1)})
                          for y, m in lists.items()], ignore_index=True)

    long = pd.DataFrame({"date": np.repeat(days.values, n_stocks),
                         "permno": np.tile(permnos, len(days)),
                         "ret": ret.ravel().astype(np.float32), "me": me.ravel()})
    long = long.dropna(subset=["ret"])
    year = long["date"].dt.year
    in_own_list = pd.Series(list(zip(year, long["permno"]))).isin(
        {(y, p) for y, m in lists.items() for p in m}).to_numpy()
    daily = long[(year <= full_until).to_numpy() | in_own_list]

    filled = pd.DataFrame(np.nan_to_num(ret), index=days, columns=permnos)
    monthly = (1 + filled).groupby([days.year, days.month]).prod() - 1
    month_end = pd.DataFrame(index=days).groupby([days.year, days.month]).apply(lambda g: g.index[-1])
    monthly.index = pd.DatetimeIndex(month_end.to_numpy())
    monthly = monthly.stack().rename("ret").reset_index()
    monthly.columns = ["date", "permno", "ret"]
    alive = ~((monthly["permno"] == dead) & (monthly["date"] > "2014-09-30"))
    monthly = monthly[alive]

    factors = pd.DataFrame({"date": days, "mktrf": mkt, "smb": 0.0, "hml": 0.0, "rf": 0.0001})
    tickers = pd.DataFrame({"permno": permnos, "ticker": [f"T{p}" for p in permnos],
                            "last_date": days[-1]})
    universe.to_parquet(folder / "universe.parquet", index=False)
    daily.to_parquet(folder / "daily.parquet", index=False)
    monthly.to_parquet(folder / "monthly.parquet", index=False)
    factors.to_parquet(folder / "ff_daily.parquet", index=False)
    tickers.to_parquet(folder / "tickers.parquet", index=False)
    return {"days": days, "permnos": permnos, "ret": ret, "lists": lists, "mkt": mkt, "dead": dead}


def make_news(days, permnos, n_topics=5, per_day=80, seed=1):
    """(daily sums of topic weights by calendar day, firm-day-topic article counts)."""
    rng = np.random.default_rng(seed)
    calendar = pd.date_range(days[0], days[-1])
    n = rng.poisson(per_day, len(calendar))
    share = rng.dirichlet(np.ones(n_topics) * 5, len(calendar))
    sums = pd.DataFrame(share * n[:, None], index=pd.DatetimeIndex(calendar, name="date"),
                        columns=[f"t{k:02d}" for k in range(n_topics)])
    sums.insert(0, "n_articles", n)
    m = 6000
    firm = pd.DataFrame({"date": rng.choice(calendar, m), "ticker": [f"T{p}" for p in rng.choice(permnos, m)],
                         "topic": rng.integers(0, n_topics, m), "n": rng.integers(1, 4, m)})
    firm = firm.groupby(["date", "ticker", "topic"], as_index=False)["n"].sum()
    return sums, firm


def make_sentiment(data, days, permnos, per_day=6, dim=4, seed=2):
    """Articles, ticker links and their scores in the layout of data/news/ and
    data/sentiment/ (scripts/03d_sentiment.py), for the stocks T<permno>; pure noise.
    A few articles are filed under four tickers (too many for firm news) and some under
    a ticker that is not in our universe."""
    rng = np.random.default_rng(seed)
    data = Path(data)
    calendar = pd.date_range(days[0], days[-1])
    date = np.sort(rng.choice(calendar, per_day * len(calendar)))
    n = len(date)
    tickers = np.array([f"T{p}" for p in permnos] + ["ZZZ"])
    first = rng.choice(tickers, n)
    many = rng.random(n) < 0.05
    links = [pd.DataFrame({"article_id": np.arange(n), "ticker": first})]
    for _ in range(3):
        links.append(pd.DataFrame({"article_id": np.flatnonzero(many),
                                   "ticker": rng.choice(tickers, many.sum())}))
    links = pd.concat(links).drop_duplicates().sort_values(["article_id", "ticker"],
                                                          ignore_index=True)
    n_tickers = links.groupby("article_id").size().to_numpy().astype(np.int32)
    (data / "news" / "articles").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"article_id": np.arange(n), "date": date, "title": "headline",
                  "lead": "lead", "n_tickers": n_tickers}).to_parquet(
        data / "news" / "articles" / "all.parquet", index=False)
    links.to_parquet(data / "news" / "links.parquet", index=False)
    ids = np.flatnonzero((n_tickers <= 3) & np.isin(np.arange(n), links.loc[
        links["ticker"] != "ZZZ", "article_id"]))
    out = data / "sentiment"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ids.npy", ids)
    probs = rng.dirichlet(np.ones(3), len(ids)).astype(np.float16)
    np.save(out / "finbert_probs.npy", probs)
    words = rng.integers(20, 90, len(ids))
    np.save(out / "lm.npy", np.column_stack([words, rng.binomial(words, 0.02),
                                             rng.binomial(words, 0.03)]).astype(np.int32))
    np.save(out / "finbert_pca.npy", rng.normal(0, 1, (len(ids), dim)).astype(np.float16))
    return {"ids": ids, "links": links, "date": date, "probs": probs}
