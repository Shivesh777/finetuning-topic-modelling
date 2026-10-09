"""Independent check of the CL futures return construction against the EIA WTI Cushing spot price (via FRED).

Spot and rolled futures returns are not expected to match: the difference is the roll yield (basis change), large around
contract expiries and in April 2020. The questions are whether they move together week to week, whether discrepancies
concentrate where they should (rolls, 2020), and whether the gap accumulates in the direction the curve shape implies.

    python3 -I clean_fred.py external/fred_wti/DCOILWTICO.csv external/wti_spot_clean.csv   (already run)
    python3 validate_wti.py   ->  wti_validation.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COM = HERE.parent.parent / "data" / "commodities"
L = []


def say(s=""):
    print(s)
    L.append(s)


spot = pd.read_csv(HERE / "external" / "wti_spot_clean.csv", parse_dates=["date"]).set_index("date").price
fut = pd.read_parquet(COM / "futures_trading_daily.parquet").query("root == 'CL'").dropna(subset=["ret"]).set_index("date")

iso = lambda idx: idx.isocalendar().year.values * 100 + idx.isocalendar().week.values
sw = spot.groupby(iso(spot.index)).agg(["last", "size"]).rename(columns={"last": "p", "size": "n"})
sw["end"] = spot.index.to_series().groupby(iso(spot.index)).last().values
sw["min_p"] = spot.groupby(iso(spot.index)).min()
sw["r_spot"] = (sw.p / sw.p.shift(1) - 1).where((sw.p > 0) & (sw.p.shift(1) > 0))
fw = pd.DataFrame({"end_f": fut.index.to_series().groupby(iso(fut.index)).last(),
                   "r_fut": (1 + fut.ret).groupby(iso(fut.index)).prod() - 1,
                   "rolled": fut.held_expiry.groupby(iso(fut.index)).apply(lambda s: s.nunique() > 1),
                   "carry": fut.carry.groupby(iso(fut.index)).mean()})
w = sw.join(fw, how="inner").dropna(subset=["r_spot", "r_fut"])
w["end"] = pd.to_datetime(w.end)
w["diff"] = w.r_fut - w.r_spot
w["rolled_prev"] = w.rolled.shift(1).fillna(False)
w = w[(w.end >= "2015-01-09") & (w.end <= "2024-12-31")]

say("# CL futures against WTI Cushing spot (EIA, via FRED)\n")
say(f"- spot file: 2,526 usable daily values ({spot.index.min().date()} to {spot.index.max().date()}), 117 missing (holidays); "
    f"{len(w)} Friday-ending weeks with both returns defined\n")


def stats(x, label):
    c = x[["r_spot", "r_fut"]].corr().iloc[0, 1]
    slope = np.polyfit(x.r_spot, x.r_fut, 1)[0]
    say(f"| {label} | {len(x)} | {c:.3f} | {slope:.2f} | {x['diff'].std() * 100:.2f} | {x['diff'].abs().max() * 100:.1f} |")


say("## 1. Weekly agreement\n")
say("| sample | weeks | correlation | slope of futures on spot | std of (futures - spot), pp | largest |gap|, pp |")
say("|---|---|---|---|---|---|")
stats(w, "all weeks")
stats(w[~((w.end >= "2020-03-01") & (w.end <= "2020-07-31"))], "excluding Mar-Jul 2020")
for a, b, n in [("2015", "2019", "development 2015-2019"), ("2020", "2021", "validation 2020-2021"), ("2022", "2024", "test 2022-2024")]:
    stats(w[(w.end >= f"{a}-01-01") & (w.end <= f"{b}-12-31")], n)

say("\n## 2. Largest discrepancies (futures minus spot, weekly)\n")
say("| week ending | spot return | futures return | gap, pp | rolled that week | minimum spot price in week |")
say("|---|---|---|---|---|---|")
for r in w.reindex(w["diff"].abs().sort_values(ascending=False).index).head(10).itertuples():
    say(f"| {r.end.date()} | {r.r_spot:+.1%} | {r.r_fut:+.1%} | {r.diff * 100:+.1f} | {bool(r.rolled)} | {r.min_p:.2f} |")

say("\n## 3. April 2020\n")
say("| date | spot price | futures daily return | held contract (months since year 0) |")
say("|---|---|---|---|")
for dt in pd.bdate_range("2020-04-15", "2020-04-24"):
    sp = spot.get(dt, np.nan)
    fr = fut.ret.get(dt, np.nan)
    ex = fut.held_expiry.get(dt, np.nan)
    say(f"| {dt.date()} | {sp:.2f} | {fr:+.1%} | {int(ex) if ex == ex else ''} |")
say("\nThe spot price turns negative on 2020-04-20 (the May contract expiry) and the spot series then jumps to the June contract's level. "
    "The futures series had already rolled from May (held 24245) to June (24246) on 2020-04-16, so it never held the contract that settled "
    "below zero, and its returns on 20-21 April are the June contract's.\n")

say("## 4. Rolls\n")
x = w.copy()
for lab, m in [("week in which the held contract changes", x.rolled), ("week after a roll", x.rolled_prev), ("other weeks", ~x.rolled & ~x.rolled_prev)]:
    s = x[m]
    say(f"- {lab}: {len(s)} weeks, std of gap {s['diff'].std() * 100:.2f} pp, mean gap {s['diff'].mean() * 100:+.2f} pp, mean |gap| {s['diff'].abs().mean() * 100:.2f} pp")

say("\n## 5. Does the cumulative gap follow the curve shape?\n")
say("Futures return is below spot return when the curve is in contango (the held contract converges down toward spot) and above it in backwardation. "
    "So the annual sum of weekly gaps should rise with the average slope of the curve (the `carry` column: positive means backwardation).\n")
y = w.assign(year=w.end.dt.year).groupby("year").agg(gap=("diff", "sum"), carry=("carry", "mean"), weeks=("diff", "size"))
say("| year | sum of weekly gaps, pp | mean annualised curve slope (positive = backwardation) |")
say("|---|---|---|")
for r in y.itertuples():
    say(f"| {r.Index} | {r.gap * 100:+.1f} | {r.carry:+.1%} |")
sp = y.gap.corr(y.carry, method="spearman")
say(f"\nSpearman correlation across the {len(y)} years: {sp:.2f}; Pearson {y.gap.corr(y.carry):.2f}.")
say("\n(No test statistic is attached to the ten-year comparison: ten points are too few for a formal test, so it is a direction check only.)")
(HERE / "wti_validation.md").write_text("\n".join(L))
