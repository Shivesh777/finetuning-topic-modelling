"""Return matrices, universe, holding periods, features and the panel, on an artificial market."""
import copy

import numpy as np
import pandas as pd
import pytest

from ftm.data.returns import Market
from ftm.features import firm_news, narrative, price
from ftm.panel import Panel, hot_deck, rank_transform

from synthetic import make_market, make_news


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp("crsp")
    truth = make_market(folder)
    market = Market.load(folder)
    sums, firm = make_news(truth["days"], truth["permnos"])
    return truth, market, sums, firm


def test_load_and_fill(world):
    truth, market, _, _ = world
    assert market.ret.shape == (len(truth["days"]), len(truth["permnos"]))
    missing = ~market.observed & ~np.isnan(market.ret)           # filled from the monthly file
    assert missing.any()
    d, p = np.argwhere(missing)[0]
    month = (market.dates.year == market.dates[d].year) & (market.dates.month == market.dates[d].month)
    assert not market.observed[month, p].any()
    true_month = np.prod(1 + truth["ret"][month, p]) - 1
    assert np.isclose(np.prod(1 + market.ret[month, p]) - 1, true_month, atol=1e-5)
    assert np.allclose(market.ret[month, p], market.ret[month, p][0])     # spread evenly
    dead = np.searchsorted(market.permnos, truth["dead"])
    assert np.isnan(market.ret[market.dates >= "2014-11-01", dead]).all()  # no fill after delisting


def test_formation_days(world):
    _, market, _, _ = world
    monthly = market.formation_days("M")
    for f in monthly[:-1]:
        assert market.dates[f].month != market.dates[f + 1].month
    assert market.dates[monthly[-1]] == market.dates[-1]          # December 2015 is complete
    weekly, biweekly = market.formation_days("W"), market.formation_days("2W")
    assert set(biweekly) <= set(weekly) and abs(len(weekly) - 2 * len(biweekly)) <= 2
    for f in weekly[:-1]:
        assert market.dates[f + 1].weekday() < market.dates[f].weekday() or \
            (market.dates[f + 1] - market.dates[f]).days > 2


def test_holding_returns(world):
    _, market, _, _ = world
    forms = market.formation_days("M")
    held, rf = market.holding_returns(forms)
    p, col = 7, 3
    days = slice(forms[p] + 1, forms[p + 1] + 1)
    assert np.isclose(held[p, col], np.prod(1 + np.nan_to_num(market.ret[days, col])) - 1)
    assert np.isclose(rf[p], np.prod(1 + market.rf[days]) - 1)


def test_universe_is_known_at_formation(world):
    truth, market, _, _ = world
    june_end = max(i for i, d in enumerate(market.dates) if d.year == 2013 and d.month == 6)
    assert market.list_year(june_end) == 2013 and market.list_year(june_end - 1) == 2012
    assert market.list_year(june_end + 100) == 2013
    held = market.permnos[market.eligible(june_end)]
    assert set(held) <= set(truth["lists"][2013])
    dead = np.searchsorted(market.permnos, truth["dead"])
    late = int(np.flatnonzero(market.dates >= "2014-10-15")[0])
    assert not market.eligible(late)[dead]


def _shocks(market, sums):
    return narrative.attention_shocks(narrative.trading_day_attention(sums, market.dates))


def test_trading_day_attention(world):
    _, market, sums, _ = world
    theta = narrative.trading_day_attention(sums, market.dates)
    assert np.allclose(theta.dropna().sum(axis=1), 1.0)
    monday = next(i for i, d in enumerate(market.dates) if d.weekday() == 0 and i > 10
                  and (d - market.dates[i - 1]).days == 3)
    days = pd.date_range(market.dates[monday] - pd.Timedelta(days=2), market.dates[monday])
    block = sums.loc[days].drop(columns="n_articles").sum()
    assert np.allclose(theta.iloc[monday], block / block.sum())      # weekend news counts on Monday


def test_exposure_matches_direct_computation(world):
    _, market, sums, _ = world
    shocks = _shocks(market, sums)
    f, col, k = 600, 5, 2
    got = narrative.exposures(market, shocks, f, neutral=False)[col, k]
    days = np.arange(f - 252, f)                                    # lag of one day
    ok = market.observed[days, col] & ~np.isnan(shocks[days]).any(axis=1)
    w = 0.5 ** ((f - 1 - days) / 60.0)
    valid = ~np.isnan(shocks[days]).any(axis=1)
    zbar = (w * valid * np.nan_to_num(shocks[days, k])).sum() / (w * valid).sum()
    r = market.ret[days, col]
    rbar = (w[ok] * r[ok]).sum() / w[ok].sum()
    direct = (w[ok] * (r[ok] - rbar) * (shocks[days, k][ok] - zbar)).sum() / w[ok].sum()
    assert np.isclose(got, direct, rtol=1e-4, atol=1e-10)


def test_neutral_exposure_is_orthogonal_to_beta(world):
    truth, market, sums, _ = world
    shocks = _shocks(market, sums)
    shocks[:, 0] = np.nan_to_num(shocks[:, 0]) + 20.0 * truth["mkt"]  # a topic that tracks the market
    f = 700
    raw = narrative.exposures(market, shocks, f, neutral=False)[:, 0]
    net = narrative.exposures(market, shocks, f, neutral=True)[:, 0]
    beta = price.price_features(market, f)[:, 0]
    ok = ~np.isnan(raw) & ~np.isnan(beta)
    assert np.corrcoef(raw[ok], beta[ok])[0, 1] > 0.8
    assert abs(np.corrcoef(net[ok], beta[ok])[0, 1]) < 0.35


def test_firm_news_window(world):
    _, market, _, firm = world
    fn = firm_news.FirmNews(market, firm, n_topics=5)
    f, window = 500, 21
    got = fn.features(f, window=window)
    col = 4
    ticker = market.tickers.index[market.tickers.to_numpy() == col][0]
    day = np.searchsorted(market.dates.values, firm["date"].to_numpy(), side="left")
    mine = firm[(firm["ticker"] == ticker).to_numpy() & (day >= f - window) & (day <= f - 1)]
    assert np.isclose(got[col, 0], np.log1p(mine["n"].sum()))
    if mine["n"].sum():
        share = mine.groupby("topic")["n"].sum() / mine["n"].sum()
        assert np.allclose(got[col, 1 + share.index.to_numpy()], share.to_numpy())


def test_no_look_ahead(world):
    """Features of day f do not change when everything after day f is altered."""
    _, market, sums, firm = world
    f = 650
    before = [price.price_features(market, f),
              narrative.exposures(market, _shocks(market, sums), f),
              firm_news.FirmNews(market, firm, n_topics=5).features(f)]
    rng = np.random.default_rng(3)
    future = copy.deepcopy(market)
    future.ret[f + 1:] = rng.normal(0, 0.05, future.ret[f + 1:].shape)
    future.observed[f + 1:] = True
    future.me[f + 1:] *= 7.0
    future.factors.iloc[f + 1:] = 0.3
    late = sums.index >= market.dates[f]                     # news dated f or later
    sums2 = sums.astype(float)
    sums2[late] = sums2[late] * rng.uniform(0.1, 9.0, (int(late.sum()), sums.shape[1]))
    firm2 = firm.copy()
    firm2.loc[firm2["date"] >= market.dates[f], "n"] += 50
    after = [price.price_features(future, f),
             narrative.exposures(future, _shocks(future, sums2), f),
             firm_news.FirmNews(future, firm2, n_topics=5).features(f)]
    for a, b in zip(before, after):
        assert np.array_equal(a, b, equal_nan=True)


def test_rank_transform():
    x = np.array([[1.0, np.nan], [3.0, 2.0], [3.0, 5.0], [7.0, np.nan]])
    r = rank_transform(x)
    assert np.allclose(r[:, 0], [-1.0, 0.0, 0.0, 1.0])
    assert np.allclose(r[:, 1], [0.0, -1.0, 1.0, 0.0])


def test_panel(world, tmp_path):
    _, market, sums, firm = world
    panel = Panel.skeleton(market, "M", start="2013-01-01", end="2015-12-31")
    shocks = _shocks(market, sums)
    panel.add_block("price", lambda f: price.price_features(market, f), price.NAMES)
    panel.add_block("narr", lambda f: narrative.exposures(market, shocks, f),
                    [f"narr_t{k:02d}" for k in range(5)])
    index = panel.index
    assert np.allclose(index.groupby("date")["y_rel"].sum(), 0.0, atol=1e-9)
    assert (index["hold_start"] > index["date"]).all()
    assert panel.X(["price", "narr"]).shape == (len(index), 13)
    assert np.abs(panel.blocks["price"]).max() <= 1.0
    per_date = index.groupby("date").size()
    assert per_date.max() <= 40 and per_date.min() >= 30
    panel.save(tmp_path / "p")
    again = Panel.load(tmp_path / "p", blocks=["narr"])
    assert len(again.index) == len(index)
    assert np.allclose(again.blocks["narr"], panel.blocks["narr"], atol=1e-3)      # half precision


def test_missing_data_cannot_be_seen(world):
    """A stock without recent returns gets another stock's features: nothing marks it."""
    _, market, _, firm = world
    panel = Panel.skeleton(market, "M", start="2013-01-01", end="2015-12-31")
    panel.add_block("price", lambda f: price.price_features(market, f), price.NAMES)
    index, block = panel.index, panel.blocks["price"]
    assert 0.01 < index["stale"].mean() < 0.3
    for date in index.loc[index["stale"], "date"].unique()[:5]:
        rows = np.flatnonzero((index["date"] == date).to_numpy())
        stale = rows[index["stale"].to_numpy()[rows]]
        fresh = rows[~index["stale"].to_numpy()[rows]]
        step = 2.0 / (len(rows) - 1)                    # one rank
        for row in stale:                               # next to a stock with data in every feature
            assert (np.abs(block[fresh] - block[row]).max(axis=1) <= 2.5 * step).any()
        ties = (block[rows][:, None, :] == block[rows][None, :, :]).all(axis=2).sum() - len(rows)
        assert ties == 0                                # and tied with none
    assert not (block[index["stale"].to_numpy()] == 0).all(axis=1).any()


def test_hot_deck():
    rng = np.random.default_rng(0)
    x = np.array([[1.0, 2.0], [np.nan, 5.0], [3.0, 4.0], [np.nan, np.nan]])
    filled = hot_deck(x, rng)
    assert np.array_equal(filled[[0, 2]], x[[0, 2]])
    for row in filled[[1, 3]]:                          # a donor's row, moved by a hair
        assert any(np.allclose(row, donor, rtol=1e-5) and not np.array_equal(row, donor)
                   for donor in x[[0, 2]])
    empty = np.full((3, 2), np.nan)
    assert hot_deck(empty, rng) is empty


def test_firm_without_coverage_is_missing(world):
    _, market, _, firm = world
    ticker = firm["ticker"].iloc[0]
    gone = firm[firm["ticker"] != ticker]               # a company the source never had
    col = int(market.tickers[ticker])
    out = firm_news.FirmNews(market, gone, n_topics=5).features(500)
    assert np.isnan(out[col]).all() and not np.isnan(out).all(axis=1).all()
    late = firm[(firm["ticker"] != ticker) | (firm["date"] > market.dates[420])]
    fn = firm_news.FirmNews(market, late, n_topics=5)
    assert np.isnan(fn.features(400)[col]).all() and not np.isnan(fn.features(700)[col]).any()
