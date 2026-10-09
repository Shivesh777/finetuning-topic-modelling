"""Build the weekly CL / GC panel defined in pilot_prereg.md. No regressions here and no look at outcomes beyond
constructing them.

    python3 build_weekly_panel.py  ->  weekly_panel.parquet (this folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
COM = DATA / "commodities"

NARRATIVES = {                       # name -> (model folder, topic columns)
    "N_oil": ("nolbert", ["t28", "t30"]),
    "N_oil_t55": ("nolbert", ["t28", "t30", "t55"]),   # sensitivity only
    "F_oil": ("finbert", ["t06"]),
    "F_gold": ("finbert", ["t10"]),
    "F_fed": ("finbert", ["t01"]),
}
MIN_DAY_WEIGHT, MIN_WEEK_ARTICLES, MIN_VALID_DAYS = 20, 300, 3
DEV = ("2015-01-01", "2019-12-31")


def daily_shocks(trade_dates, name):
    folder, cols = NARRATIVES[name]
    d = pd.read_parquet(DATA / "topics" / folder / "daily_sums.parquet").set_index("date")
    allc = [c for c in d.columns if c.startswith("t") and c[1:].isdigit()]
    # calendar day -> next trading day (weekends and holidays carry forward)
    pos = trade_dates.searchsorted(d.index)
    keep = pos < len(trade_dates)
    g = d[keep].groupby(trade_dates[pos[keep]])
    num, den = g[cols].sum().sum(axis=1), g[allc].sum().sum(axis=1)
    theta = (num / den).where(den >= MIN_DAY_WEIGHT).reindex(trade_dates)
    base = pd.concat([theta.shift(k) for k in range(1, 6)], axis=1)
    base = base.mean(axis=1).where(base.notna().sum(axis=1) >= 3)
    return (theta - base), den.reindex(trade_dates).fillna(0)


def build_root(root):
    f = pd.read_parquet(COM / "futures_trading_daily.parquet")
    f = f[f.root == root].dropna(subset=["ret"]).sort_values("date").set_index("date")
    bad = pd.read_csv(COM / "degraded_days.csv", parse_dates=["date"]).date
    bad = set(bad) | {x + pd.Timedelta(days=1) for x in bad if x.dayofweek == 6}
    td = f.index
    iso = td.isocalendar()
    wk = pd.Series(iso.year.values * 100 + iso.week.values, index=td)

    ff = pd.read_parquet(DATA / "crsp" / "ff_daily.parquet").set_index("date").mktrf
    fiso = ff.index.isocalendar()
    mkt = (1 + ff).groupby(fiso.year.values * 100 + fiso.week.values).prod() - 1

    daily_vol = f.ret
    out = pd.DataFrame({
        "L": td.to_series().groupby(wk).last(),
        "n_days": f.ret.groupby(wk).size(),
        "r": (1 + f.ret).groupby(wk).prod() - 1,
        "degraded_in_week": pd.Series([d in bad for d in td], index=td).groupby(wk).any(),
    })
    out["mkt"] = mkt.reindex(out.index)
    keys = list(out.index)
    vol_prev = []
    for i, k in enumerate(keys):
        past = daily_vol[wk.isin(keys[max(0, i - 4):i])]
        vol_prev.append(past.std() * np.sqrt(5) if len(past) >= 10 else np.nan)
    out["vol4_prev"] = vol_prev                       # to the end of week w-1
    out["vol4_cur"] = pd.Series(vol_prev, index=out.index).shift(-1)   # to the end of week w
    out["r_lag"] = out.r.shift(1)

    for name in NARRATIVES:
        z, w = daily_shocks(td, name)
        valid = z.notna()
        last = pd.Series(td, index=td).groupby(wk).transform("last") == pd.Series(td, index=td)
        out[f"Z_{name}"] = z.groupby(wk).sum(min_count=1).where(valid.groupby(wk).sum() >= MIN_VALID_DAYS)
        z_ex = z.where(~last)
        out[f"S_{name}"] = z_ex.groupby(wk).sum(min_count=1).where(z_ex.notna().groupby(wk).sum() >= MIN_VALID_DAYS)
        out["articles"] = w.groupby(wk).sum()
    out = out[(out.n_days >= MIN_VALID_DAYS) & (out.articles >= MIN_WEEK_ARTICLES)].copy()
    out["root"] = root
    out["L"] = pd.to_datetime(out.L)
    out["window"] = pd.cut(out.L, pd.to_datetime(["2014-12-31", "2019-12-31", "2021-12-31", "2024-12-31"]),
                           labels=["dev", "val", "test"])
    # standardise shocks and controls with development-sample moments only
    dev = out[out.window == "dev"]
    for c in [c for c in out if c.startswith(("Z_", "S_"))] + ["mkt", "r", "r_lag", "vol4_prev", "vol4_cur"]:
        out[c + "_std"] = out[c] / dev[c].std()          # scale only; development-sample standard deviation
    return out.reset_index(drop=True)


if __name__ == "__main__":
    panel = pd.concat([build_root(r) for r in ["CL", "GC"]], ignore_index=True)
    panel.to_parquet(HERE / "weekly_panel.parquet", index=False)
    print(panel.groupby(["root", "window"], observed=True).agg(weeks=("L", "size"), first=("L", "min"), last=("L", "max"),
                                                              articles_per_week=("articles", "median"),
                                                              flagged_weeks=("degraded_in_week", "sum")))
    print("weeks dropped for thin news / too few days:")
    for r in ["CL", "GC"]:
        n = (pd.read_parquet(COM / "futures_trading_daily.parquet").query("root == @r").dropna(subset=["ret"]).date
             .dt.isocalendar().drop_duplicates(["year", "week"]).shape[0])
        print(f"  {r}: {n - (panel.root == r).sum()} of {n}")
    print(panel[[c for c in panel if c.startswith(("Z_", "S_")) and not c.endswith("_std")]].describe().round(3).T[["count", "mean", "std"]])
