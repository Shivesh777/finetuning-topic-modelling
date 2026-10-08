"""The portfolio rules as the backtests call them: ties, flat forecasts, excess returns."""
import numpy as np
import pandas as pd

from ftm.portfolio import strategies


def test_flat_forecast_takes_no_view():
    score, cap = np.full(50, 0.3), np.arange(1.0, 51.0)
    for name in strategies.LONG_SHORT:
        assert not strategies.weights_for(name, score, cap).any()
    assert np.allclose(strategies.weights_for("long_ew", score, cap), 1 / 50)
    assert np.allclose(strategies.weights_for("long_vw", score, cap), cap / cap.sum())
    assert np.allclose(strategies.weights_for("tilt_vw", score, cap), cap / cap.sum())


def test_ties_are_broken_by_a_fixed_order_not_by_position():
    score = np.r_[np.zeros(80), np.ones(20)]            # the bottom decile lies inside a tie
    order = np.random.default_rng(3).random(100)
    w = strategies.weights_for("ls_ew", score, None, order=order)
    assert np.isclose(w[w > 0].sum(), 1) and np.isclose(w[w < 0].sum(), -1)
    assert (w[:80] <= 0).all() and (w[80:] >= 0).all()
    # Short: the ten tied stocks with the lowest order. Long: the ten highest of the top group.
    assert set(np.flatnonzero(w < 0)) == set(np.argsort(order[:80])[:10])
    assert set(np.flatnonzero(w > 0)) == set(80 + np.argsort(order[80:])[-10:])
    assert set(np.flatnonzero(w < 0)) != set(range(10))                 # not the first ten
    # Without an order the default is the same on every call.
    again = [strategies.weights_for("ls_ew", score, None) for _ in range(2)]
    np.testing.assert_array_equal(*again)


def tied_frame(periods=8, stocks=60, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    n = periods * stocks
    return pd.DataFrame({"date": np.repeat(dates, stocks), "permno": np.tile(np.arange(stocks), periods),
                         "pred": np.tile(np.r_[np.zeros(40), np.ones(20)], periods),     # two values
                         "ret": rng.normal(0.01, 0.05, n), "rf": 0.001, "me": rng.uniform(1, 10, n)})


def test_a_forecast_that_does_not_change_does_not_trade():
    out = strategies.backtests(tied_frame(), ("ls_ew", "long_ew"), cost_bps=10.0)
    for name, g in out.groupby("strategy"):
        held = g.sort_values("date")
        assert held["turnover"].iloc[0] > 0.49                      # the portfolio is built once
        # afterwards it only rebalances the drift of its weights with returns
        assert held["turnover"].iloc[1:].max() < 0.1, name


def test_a_strategy_does_not_depend_on_the_others_evaluated_with_it():
    frame = tied_frame(seed=5)
    alone = strategies.backtests(frame, ("ls_ew",))
    together = strategies.backtests(frame, ("long_vw", "rank", "ls_ew", "long_ew"))
    together = together[together["strategy"] == "ls_ew"].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone, together)


def test_backtests_excess_and_benchmarks():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    frame = pd.DataFrame({"date": np.repeat(dates, 30), "permno": np.tile(np.arange(30), 6),
                          "pred": rng.normal(size=180), "ret": rng.normal(0.01, 0.05, 180),
                          "rf": 0.001, "me": rng.uniform(1, 10, 180)})
    out = strategies.backtests(frame, ("ls_ew", "long_ew", "univ_ew"), cost_bps=10.0)
    by = {name: g.set_index("date") for name, g in out.groupby("strategy")}
    assert np.allclose(by["ls_ew"]["excess"], by["ls_ew"]["net"])
    assert np.allclose(by["long_ew"]["excess"], by["long_ew"]["net"] - 0.001)
    assert np.allclose(by["univ_ew"]["gross"], frame.groupby("date")["ret"].mean())
    assert np.isclose(by["ls_ew"]["cost"].iloc[0], 2 * 1.0 * 10 / 1e4)   # builds 2 of gross from cash


def test_buffer_starts_as_the_decile_rule_and_keeps_what_stays_in_the_fifth():
    n = 100
    score = np.arange(n, dtype=float)                      # stock i has rank i
    order = np.random.default_rng(0).random(n)
    none = np.zeros(n, dtype=bool)
    w, long, short = strategies.buffered_long_short(score, none, none, order)
    np.testing.assert_array_equal(w, strategies.weights_for("ls_ew", score, None, order=order))
    assert set(np.flatnonzero(long)) == set(range(90, 100)) and set(np.flatnonzero(short)) == set(range(10))
    # Stock 99 falls to rank 85 (still in the top fifth) and stock 98 to rank 70 (out of it).
    moved = score.copy()
    moved[[99, 85]], moved[[98, 70]] = moved[[85, 99]], moved[[70, 98]]
    w2, long2, _ = strategies.buffered_long_short(moved, long, short, order)
    assert long2[99] and not long2[98]                     # kept; sold
    assert long2[85] and long2[70]                         # the stocks that rose enter
    assert np.isclose(w2[w2 > 0].sum(), 1) and np.isclose(w2[w2 < 0].sum(), -1)
    # A flat forecast holds nothing and forgets the legs.
    w3, long3, short3 = strategies.buffered_long_short(np.ones(n), long, short, order)
    assert not w3.any() and not long3.any() and not short3.any()


def test_buffer_trades_less_than_the_decile_rule():
    rng = np.random.default_rng(4)
    periods, stocks = 30, 200
    signal = np.cumsum(rng.normal(size=(periods, stocks)), axis=0)     # slowly moving forecast
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    frame = pd.DataFrame({"date": np.repeat(dates, stocks), "permno": np.tile(np.arange(stocks), periods),
                          "pred": signal.ravel(), "ret": rng.normal(0.01, 0.05, periods * stocks),
                          "rf": 0.001, "me": rng.uniform(1, 10, periods * stocks)})
    out = strategies.backtests(frame, ("ls_ew", "ls_buf"), cost_bps=10.0)
    turnover = out.groupby("strategy")["turnover"].mean()
    assert turnover["ls_buf"] < 0.7 * turnover["ls_ew"]
    first = out[out["date"] == dates[0]].set_index("strategy")
    assert np.isclose(first.loc["ls_buf", "gross"], first.loc["ls_ew", "gross"])  # same start
    assert (out.loc[out["strategy"] == "ls_buf", "excess"] == out.loc[out["strategy"] == "ls_buf", "net"]).all()


def test_tilt_holds_the_universe_tilted_towards_the_forecast():
    cap, score = np.array([10.0, 20, 30, 40]), np.array([0.1, 0.4, 0.2, 0.3])
    w = strategies.weights_for("tilt_vw", score, cap)
    # ranks scaled to [-1, 1]: -1, 1, -1/3, 1/3 -> cap * (0, 2, 2/3, 4/3)
    expected = np.array([0.0, 40.0, 20.0, 160 / 3])
    assert np.allclose(w, expected / expected.sum()) and (w >= 0).all()
    # a stock without a forecast keeps its market weight relative to the others' average
    w = strategies.weights_for("tilt_vw", np.array([0.1, np.nan, 0.2, 0.3]), cap)
    assert np.isclose(w.sum(), 1) and w[1] > 0 and w[0] == 0
    assert "tilt_vw" in strategies.LONG_ONLY
