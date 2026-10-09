"""Align the monthly IPCA factor returns with monthly returns of CL, GC, HG, BTC futures and the equity market.

Holding window of a factor month: the factor dated at formation date d earns the return from the first trading day after d to the last
trading day of the following calendar month (verified against stock returns for all 104 months in ipca_alignment_check.csv). The
formation dates are irregular (from 2017 only June-December formations exist), so the window is NOT "until the next formation date".

Futures monthly return = compounded daily futures returns on trade dates in (d, E]; equity market and risk-free = compounded Fama-French
daily mktrf and rf over the same dates. A month needs at least 15 futures return days.

    python3 build_ipca_monthly.py  ->  ipca_monthly_panel.csv

No regression and no correlation is computed here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
SPECS = {"primary": "narr_finbert__sipca", "secondary": "narr_nolbert20__sipca"}

fut = pd.read_parquet(DATA / "commodities" / "futures_trading_daily.parquet").dropna(subset=["ret"])
ff = pd.read_parquet(DATA / "crsp" / "ff_daily.parquet").set_index("date")
td = pd.DatetimeIndex(sorted(pd.read_parquet(DATA / "crsp" / "daily.parquet", columns=["date"]).date.unique()))

base = None
for role, spec in SPECS.items():
    f = pd.read_csv(DATA / "results" / "ipca" / "M" / spec / "factors.csv", parse_dates=["date", "hold_start"])
    fc = [c for c in f.columns if c.startswith("f") and c[1:].isdigit()]
    f = f.rename(columns={c: f"{role[:3]}_{c}" for c in fc}).rename(columns={"estimate": f"estimate_{role[:3]}"})
    keep = ["date", "hold_start", "split"] + [f"{role[:3]}_{c}" for c in fc] + [f"estimate_{role[:3]}"]
    base = f[keep] if base is None else base.merge(f[keep].drop(columns=["hold_start", "split"]), on="date", validate="1:1")
assert (base.split.value_counts().to_dict() == {"train": 69, "test": 21, "validation": 14})

base["hold_end"] = [td[td <= (d.to_period("M") + 1).end_time.normalize()].max() for d in base.date]
rows = []
for r in base.itertuples():
    out = {"date": r.date}
    for root in ["CL", "GC", "HG", "BTC"]:
        x = fut[(fut.root == root) & (fut.date > r.date) & (fut.date <= r.hold_end)]
        out[root] = (1 + x.ret).prod() - 1 if len(x) >= 15 else np.nan
        out[f"{root}_days"] = len(x)
        out[f"{root}_flag"] = int(x.date.isin(pd.read_csv(DATA / "commodities" / "degraded_days.csv", parse_dates=["date"]).date).sum())
    w = ff[(ff.index > r.date) & (ff.index <= r.hold_end)]
    out["mkt"] = (1 + w.mktrf).prod() - 1
    out["rf"] = (1 + w.rf).prod() - 1
    out["mkt_days"] = len(w)
    rows.append(out)
panel = base.merge(pd.DataFrame(rows), on="date", validate="1:1")
panel.to_csv(HERE / "ipca_monthly_panel.csv", index=False)
print(f"{len(panel)} months, {panel.date.min().date()} to {panel.date.max().date()}")
print("months with a return, by split:")
print(panel.groupby("split")[["CL", "GC", "HG", "BTC", "mkt"]].count().loc[["train", "validation", "test"]])
print("window length in trading days (futures): min", panel.CL_days.min(), "max", panel.CL_days.max(), "| equity", panel.mkt_days.min(), panel.mkt_days.max())
print("BTC first month:", panel.loc[panel.BTC.notna(), "date"].min().date(), "| months containing a Databento degraded day:", int((panel.CL_flag > 0).sum()))
