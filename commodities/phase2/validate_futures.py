"""Phase 2 / task 3: validate the continuous futures return construction before any pilot uses it.

Checks, none of which look at news:
  1. Gold: weekly and daily agreement with GLD (CRSP), and cumulative return against GLD minus the risk-free rate.
  2. Crude: agreement with DBO (a futures-based oil fund) and the oil equities XLE and XOP.
  3. Rolls: returns on days the held contract changes, and how much total variance and cumulative return sit there.
  4. Databento's degraded days: what changes if they are dropped.
  5. Basic hygiene: duplicates, zero returns, weekday coverage.

    python3 validate_futures.py   ->  futures_validation.md, futures_validation.csv (this folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
COM = DATA / "commodities"
lines, rows = [], []


def say(s=""):
    print(s)
    lines.append(s)


fut = pd.read_parquet(COM / "futures_trading_daily.parquet")
etf = pd.read_parquet(COM / "etf_daily.parquet")
ff = pd.read_parquet(DATA / "crsp" / "ff_daily.parquet").set_index("date")
bad = pd.read_csv(COM / "degraded_days.csv", parse_dates=["date"])

R = fut.pivot(index="date", columns="root", values="ret")
E = etf.pivot(index="date", columns="ticker", values="ret")
RF = ff.rf.reindex(E.index)


def weekly(x, how="comp"):
    """Friday-ending weeks; compounding of daily returns."""
    return (1 + x).resample("W-FRI").prod(min_count=1) - 1


def ann_cum(r):
    r = r.dropna()
    return np.prod(1 + r) - 1, r.mean() * 252


say("# Futures return validation\n")

# ---- 1 gold vs GLD
say("## 1. Gold (GC) against GLD from CRSP\n")
both = pd.concat([R.GC.rename("fut"), E.GLD.rename("etf"), RF.rename("rf")], axis=1).dropna()
both["etf_ex"] = both.etf - both.rf
d_corr = both[["fut", "etf_ex"]].corr().iloc[0, 1]
w = pd.concat([weekly(both.fut), weekly(both.etf_ex)], axis=1).dropna()
w.columns = ["fut", "etf_ex"]
slope = np.polyfit(w.etf_ex, w.fut, 1)[0]
w_corr = w.corr().iloc[0, 1]
cf, ce = np.prod(1 + both.fut) - 1, np.prod(1 + both.etf_ex) - 1
yrs = len(both) / 252
say(f"- common days: {len(both)}  ({both.index.min().date()} to {both.index.max().date()})")
say(f"- daily correlation futures vs GLD excess return: {d_corr:.3f}  (the futures bar closes about 3 hours after GLD, so daily is not exact)")
say(f"- weekly correlation: {w_corr:.3f}; weekly slope of futures on GLD excess return: {slope:.3f}")
say(f"- cumulative: futures {cf:+.1%}, GLD minus risk-free {ce:+.1%}; annualised log gap {(np.log1p(cf) - np.log1p(ce)) / yrs:+.2%} per year")
say("  (GLD's fee is 0.40% a year, and a futures roll earns or pays the carry, so a gap of a few tenths of a percent a year is expected;")
say("   a gap of several percent a year would point to a construction error)")
rows.append(dict(check="GC vs GLD weekly corr", value=w_corr))
rows.append(dict(check="GC vs GLD cumulative log gap per year", value=(np.log1p(cf) - np.log1p(ce)) / yrs))
# gap by sub-period
for a, b in [("2015", "2019"), ("2020", "2021"), ("2022", "2024")]:
    s = both.loc[a:b]
    g = (np.log1p(np.prod(1 + s.fut) - 1) - np.log1p(np.prod(1 + s.etf_ex) - 1)) / (len(s) / 252)
    say(f"- {a}-{b}: gap {g:+.2%} per year, weekly corr {pd.concat([weekly(s.fut), weekly(s.etf_ex)], axis=1).corr().iloc[0, 1]:.3f}")

# ---- 2 crude
say("\n## 2. Crude (CL) against oil funds in CRSP\n")
for tk in ["DBO", "XLE", "XOP"]:
    x = pd.concat([R.CL.rename("fut"), E[tk].rename("etf")], axis=1).dropna()
    wk = pd.concat([weekly(x.fut), weekly(x.etf)], axis=1).dropna()
    say(f"- {tk}: daily corr {x.corr().iloc[0, 1]:.3f}, weekly corr {wk.corr().iloc[0, 1]:.3f}")
    rows.append(dict(check=f"CL vs {tk} weekly corr", value=wk.corr().iloc[0, 1]))
say("  DBO holds futures but with an optimised roll, so it should track CL closely but not exactly; the equity funds track it loosely.")

# ---- 3 rolls
say("\n## 3. Roll days (held contract changes)\n")
say("| root | roll days | mean ret on roll days | std roll / non-roll | share of total variance on roll days | cumulative return with roll-day returns set to 0 |")
say("|---|---|---|---|---|---|")
for r, g in fut.groupby("root"):
    g = g.dropna(subset=["ret"]).sort_values("date")
    roll = g.held_expiry.ne(g.held_expiry.shift()) & g.held_expiry.shift().notna()
    var_share = (g.ret[roll] ** 2).sum() / (g.ret ** 2).sum()
    no_roll = np.prod(1 + g.ret.where(~roll, 0)) - 1
    say(f"| {r} | {int(roll.sum())} | {g.ret[roll].mean():+.4%} | {g.ret[roll].std() / g.ret[~roll].std():.2f} | {var_share:.1%} | {no_roll:+.1%} (all days: {np.prod(1 + g.ret) - 1:+.1%}) |")
    rows.append(dict(check=f"{r} roll-day variance share", value=var_share))
say("\nExpectation under a clean construction: roll days look like other days (std ratio near 1, variance share near the roll-day share of days, about 2-5%).")

# ---- 4 degraded
say("\n## 4. Databento degraded days\n")
bd = set(bad.date) | {x + pd.Timedelta(days=1) for x in bad.date if x.dayofweek == 6}
say(f"- degraded UTC days: {len(bad)}  ({', '.join(str(x.date()) for x in sorted(bad.date))})")
for r in ["CL", "GC", "HG", "BTC"]:
    g = fut[fut.root == r].dropna(subset=["ret"]).set_index("date").ret
    flag = g.index.isin(bd)
    gw, gw2 = weekly(g), weekly(g.where(~flag, 0))
    both_w = pd.concat([gw, gw2], axis=1).dropna()
    say(f"- {r}: {int(flag.sum())} flagged trading days; mean |ret| flagged {g[flag].abs().mean():.3%} vs others {g[~flag].abs().mean():.3%}; "
        f"weekly corr with/without {both_w.corr().iloc[0, 1]:.4f}; weeks changed by more than 0.5pp: {(both_w.iloc[:, 0] - both_w.iloc[:, 1]).abs().gt(0.005).sum()}; "
        f"cumulative {np.prod(1 + g) - 1:+.1%} vs {np.prod(1 + g.where(~flag, 0)) - 1:+.1%}")

# ---- 5 hygiene
say("\n## 5. Hygiene\n")
for r, g in fut.groupby("root"):
    x = g.dropna(subset=["ret"])
    say(f"- {r}: duplicate dates {int(g.date.duplicated().sum())}, exact-zero returns {(x.ret == 0).mean():.2%}, "
        f"weekdays covered {sorted(x.date.dt.dayofweek.unique().tolist())}, days with |ret| > 10%: {int((x.ret.abs() > 0.10).sum())}")

(HERE / "futures_validation.md").write_text("\n".join(lines))
pd.DataFrame(rows).to_csv(HERE / "futures_validation.csv", index=False)
