"""Turn futures_daily.parquet (UTC-day bars, including the short Sunday evening session) into
one return per trading day, and attach Databento's data-quality flags.

    python3 build_returns.py

Writes futures_trading_daily.parquet and degraded_days.csv next to the inputs.

Choices (all fixed here, none tuned on returns):
  - A Sunday bar is the first ~2 hours of Monday's trading day, so its return is compounded into Monday's.
  - Bars close at 00:00 UTC, which is after the CME settlement for these contracts; the Friday bar's close is
    therefore the weekly cutoff used for every asset.
  - Days Databento marks "degraded" are kept but flagged, so they can be dropped as a sensitivity check.
"""
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

import pull_futures  # loads ../.env and defines OUT

OUT = pull_futures.OUT


def main():
    d = pd.read_parquet(OUT / "futures_daily.parquet")
    d["trade_date"] = d.date + pd.to_timedelta((d.date.dt.dayofweek == 6).astype(int), unit="D")
    g = d.groupby(["root", "trade_date"])
    out = g.agg(ret=("ret", lambda x: np.nan if x.isna().all() else (1 + x.dropna()).prod() - 1),
                volume=("volume", "sum"), carry=("carry", "last"), held_expiry=("held_expiry", "last"),
                bars=("date", "size")).reset_index().rename(columns={"trade_date": "date"})

    cond = db.Historical().metadata.get_dataset_condition(
        dataset=pull_futures.DATASET, start_date="2015-01-01", end_date="2025-01-01")
    bad = pd.DataFrame([c for c in cond if c["condition"] != "available"])
    bad["date"] = pd.to_datetime(bad["date"])
    bad.to_csv(OUT / "degraded_days.csv", index=False)
    # a trading day is flagged if any of the UTC days that make it up is flagged
    flagged = set(bad.date) | {x + pd.Timedelta(days=1) for x in bad.date if x.dayofweek == 6}
    out["degraded"] = out.date.isin(flagged)
    out.to_parquet(OUT / "futures_trading_daily.parquet", index=False)

    for r, x in out.groupby("root"):
        s = x.dropna(subset=["ret"])
        print(f"{r:3s} {s.date.min().date()} to {s.date.max().date()}  {len(s):5d} trading days  "
              f"ann.vol {s.ret.std() * np.sqrt(252):5.1%}  cumulative {np.prod(1 + s.ret) - 1:8.1%}  "
              f"flagged days {int(x.degraded.sum())}")
    print(f"weekday counts: {out.date.dt.dayofweek.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
