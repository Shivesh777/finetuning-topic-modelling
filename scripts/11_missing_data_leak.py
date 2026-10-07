"""How much "predictability" comes from missing data: a demonstration.

    python scripts/11_missing_data_leak.py [--freq M 2W W]

From 2017 our return file has no daily returns, between January and the June list
change, for the stocks that are about to leave the list of the 1,000 largest (see
REFERENCE.md, section 6). The same model (XGBoost on the eight price features, tuned
on validation as everywhere) is fitted on four versions of the panel:

    missing features   median   a missing feature is set to the median rank, the usual
                                convention: a tree model can see which stocks lack data
                       donor    a stock with a missing feature gets the features of a
                                random other stock of the day: nothing marks it
    periods            all      every formation day
                       complete only the days on which the file holds the whole universe
                                (what the repository uses, together with "donor")

and scored on the test period: rank IC overall, in the incomplete and in the complete
periods, among the stocks that do have recent data, and the long-short Sharpe ratio.
Writes results/data_audit/missing_data_leak.csv.

The last version is the experiment "price, xgb" of scripts/06_run_models.py on a panel
that differs from the stored one only in which donors were drawn (the random numbers are
consumed by the incomplete days as well). Where its result differs, validation has chosen
another setting: see scripts/12_tuning_noise.py.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config
from ftm.data.returns import Market
from ftm.evaluation import metrics
from ftm.features import price
from ftm.models.tuning import chosen_setting, fit_for_test, get_model, tune
from ftm.panel import Panel
from ftm.portfolio import strategies


def ic(frame: pd.DataFrame) -> tuple[float, float]:
    if not len(frame):
        return np.nan, np.nan
    summary = metrics.ic_summary(metrics.rank_ic(frame["y_rel"], frame["pred"], frame["date"]))
    return summary["mean"], summary["t"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--model", default="xgb")
    a = ap.parse_args()
    market = Market.load()
    cls = get_model(a.model)
    rows = []
    for freq in a.freq:
        ppy = metrics.PERIODS_PER_YEAR[freq]
        panel = Panel.skeleton(market, freq, complete_only=False)
        index = panel.index
        days = index["form_day"].unique()
        complete = index["form_day"].map(dict(zip(days, map(market.complete, days)))).to_numpy()
        y, groups = index["y_rel"].to_numpy(), index["period"].to_numpy()
        for mode in ("median", "donor"):
            config.MISSING_FEATURES = mode
            panel.add_block("price", lambda f: price.price_features(market, f), price.NAMES)
            X = panel.blocks["price"]
            for periods, keep in (("all", np.ones(len(index), dtype=bool)), ("complete", complete)):
                train, val, test = (panel.rows(s) & keep for s in ("train", "validation", "test"))
                params, table, _ = tune(cls, X[train], y[train], X[val], y[val],
                                        config.TUNE_TOLERANCE, groups=groups[train])
                model = fit_for_test(cls, params, X, y, train, val, groups)
                frame = index[test].assign(pred=model.predict(X[test]), complete=complete[test])
                spread = strategies.backtests(frame, ("ls_ew",))
                stale = frame["stale"]
                row = {"freq": freq, "missing_features": mode, "periods": periods,
                       "test_periods": frame["date"].nunique(), "setting": chosen_setting(table),
                       "share_without_recent_data": stale.mean(),
                       "their_mean_relative_return": frame.loc[stale, "y_rel"].mean(),
                       "their_mean_forecast": frame.loc[stale, "pred"].mean()}
                for name, part in (("", frame), ("_incomplete_periods", frame[~frame["complete"]]),
                                   ("_complete_periods", frame[frame["complete"]]),
                                   ("_stocks_with_recent_data", frame[~stale])):
                    row[f"ic_mean{name}"], row[f"ic_t{name}"] = ic(part)
                row.update(r2_oos_pct=100 * metrics.r2_oos(frame["y_rel"], frame["pred"]),
                           sharpe_gross=metrics.sharpe(spread["gross"], ppy),
                           sharpe_net=metrics.sharpe(spread["excess"], ppy))
                rows.append(row)
                print({k: (round(float(v), 4) if isinstance(v, float) else v)
                       for k, v in row.items()}, flush=True)
    out = config.RESULTS / "data_audit"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "missing_data_leak.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
