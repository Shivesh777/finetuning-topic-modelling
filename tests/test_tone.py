"""Firm-level tone features (ftm/features/tone.py) and their article scores."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ftm import config, experiments
from ftm.data.returns import Market
from ftm.features import tone
from ftm.topics import sentiment

from synthetic import make_market, make_sentiment


def fake_market(n_days=300):
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    return SimpleNamespace(dates=dates, permnos=np.array([1, 2, 3]),
                           tickers=pd.Series([0, 1, 2], index=["AAA", "BBB", "CCC"]))


def build(market, rows):
    """rows: (trading day, ticker, finbert tone, lm positive, lm negative, vector)."""
    dates = [market.dates[d] for d, *_ in rows]
    links = pd.DataFrame({"position": np.arange(len(rows)), "ticker": [r[1] for r in rows]})
    lm = np.array([[50, r[3], r[4]] for r in rows])
    emb = np.array([r[5] for r in rows], dtype=np.float16)
    return tone.FirmTone(market, dates, links, np.array([r[2] for r in rows]), lm, emb)


def test_windows_counts_and_coverage():
    market = fake_market()
    news = build(market, [(200, "AAA", 0.5, 2, 0, [1.0, 0.0]),
                          (204, "AAA", -0.1, 0, 3, [0.0, 2.0]),
                          (206, "BBB", 0.9, 1, 0, [5.0, 5.0]),        # dated on the formation day
                          (10, "CCC", 0.3, 1, 1, [1.0, 1.0])])        # long ago: CCC not covered
    f = 206                                   # formation day; lag 1 -> news up to day 205
    fb = news.finbert(f)
    a = dict(zip(tone.FB_NAMES, fb[0]))
    assert a["fb_tone_1"] == 0.0 and a["fb_tone_5"] == pytest.approx(-0.1)
    assert a["fb_tone_21"] == pytest.approx(0.2) and a["fb_tone_252"] == pytest.approx(0.2)
    assert a["fb_surprise_21"] == pytest.approx(0.0)
    assert a["news_log_n5"] == pytest.approx(np.log1p(1)) and a["news_log_n21"] == pytest.approx(np.log1p(2))
    assert a["news_abnormal_21"] == pytest.approx(np.log1p(2) - np.log1p(2 * 21 / 252))
    assert np.isnan(fb[1]).all() and np.isnan(fb[2]).all()   # BBB only has news of day f
    assert not np.isnan(news.finbert(f + 1)[1]).any()           # ... which counts one day later
    lm = dict(zip(tone.LM_NAMES, news.lm(f)[0]))
    assert lm["lm_tone_5"] == pytest.approx(-1.0) and lm["lm_tone_21"] == pytest.approx(-1 / 5)
    emb = news.embedding(f, window=21)
    assert emb[0] == pytest.approx([0.5, 1.0]) and np.isnan(emb[2]).all()
    assert news.embedding(f, window=1)[0] == pytest.approx([0.0, 0.0])   # covered, no news


def test_weekend_news_counts_on_the_next_trading_day():
    market = fake_market()
    saturday = market.dates[50] + pd.offsets.Week(weekday=5)
    links = pd.DataFrame({"position": [0], "ticker": ["AAA"]})
    news = tone.FirmTone(market, [saturday], links, np.array([0.7]), np.array([[10, 1, 0]]))
    day = int(np.searchsorted(market.dates, saturday))
    assert market.dates[day] > saturday and market.dates[day - 1] < saturday
    assert news.finbert(day + 1)[0][0] == pytest.approx(0.7)      # tone of the last day
    assert np.isnan(news.finbert(day)[0]).all()                     # not yet known at day


def test_load_and_interactions_on_synthetic_files(tmp_path, monkeypatch):
    truth = make_market(tmp_path / "crsp", n_stocks=30, list_size=20, end="2014-12-31")
    made = make_sentiment(tmp_path, truth["days"], truth["permnos"])
    monkeypatch.setattr(config, "NEWS", tmp_path / "news")
    market = Market.load(tmp_path / "crsp")
    news = tone.FirmTone.load(market, tmp_path / "sentiment")
    assert 0.9 < news.matched <= 1.0                     # ZZZ links are not in the market
    f = len(market.dates) - 10
    fb, lm, npx = news.finbert(f), news.lm(f), news.interactions(market, f)
    assert fb.shape == (30, len(tone.FB_NAMES)) and lm.shape == (30, len(tone.LM_NAMES))
    assert npx.shape == (30, len(tone.NPX_NAMES))
    covered = ~np.isnan(fb).any(axis=1)
    assert covered.mean() > 0.8
    assert np.all(np.abs(fb[covered, :4]) <= 1) and np.all(np.abs(lm[covered]) <= 1)
    # a stock has a 5-day return either in the "news" or in the "quiet" column
    ok = ~np.isnan(npx).any(axis=1)
    assert np.all((npx[ok, 0] == 0) | (npx[ok, 1] == 0))
    assert news.embedding(f, 5).shape == (30, 4)


def test_lm_counts_and_word_lists():
    positive, negative = sentiment.lm_words()
    assert "GAIN" in positive and "LOSS" in negative and len(negative) > len(positive) > 300
    counts = sentiment.lm_counts(["Profit gains beat estimates", "Losses widen as sales decline", ""],
                                 positive, negative)
    assert counts[:, 0].tolist() == [4, 5, 0]
    assert counts[0, 1] >= 1 and counts[1, 2] >= 2 and counts[2].tolist() == [0, 0, 0]


def test_pca_is_fitted_on_the_rows_given():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (500, 6)).astype(np.float16)
    x[:250, 0] *= 10                                    # only the first half varies along axis 0
    mean, components = sentiment.pca_fit(x, np.arange(250), dim=2, block=100)
    assert abs(components[0, 0]) > 0.99
    out = sentiment.pca_apply(x, mean, components)
    assert out.shape == (500, 2) and out.dtype == np.float16


def test_feature_sets_and_kinds_of_the_news_blocks():
    blocks = ["price", "narr_nolbert", "firm_nolbert", "narr_nolbert_s1", *experiments.TONE_BLOCKS]
    sets = experiments.feature_sets(blocks)
    assert set(experiments.NEWS_SETS) <= set(sets)
    assert sets["px_all"][-2:] == ["narr_nolbert_s1", "firm_nolbert"]
    assert experiments.kind("narr_nolbert_s1") == "narr_s"
    assert experiments.topic_model("narr_nolbert_s1") is None
    assert experiments.kind("px_news") == "px" and experiments.topic_model("tone_fb") is None
    assert experiments.topic_model("all_nolbert20") == "nolbert20"
    assert "px_all" not in experiments.feature_sets(["price", *experiments.TONE_BLOCKS])
