"""Daily commodity futures from Databento (CME Globex), rolled into continuous return series.

Every outright contract of each root is downloaded as daily bars. On each day the
series holds the contract with the most volume, never rolling back to an earlier
expiry, and the choice made at the close of day t-1 earns day t's return on that
same contract, so roll gaps never show up as returns. Carry is the annualised log
slope between the held contract and the next later expiry.

    python3 pull_futures.py            # prints the cost only
    python3 pull_futures.py --yes      # downloads and builds the series

Needs DATABENTO_API_KEY in the environment. Writes to data/commodities/.
"""
import argparse
import os
import re
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

DATASET = "GLBX.MDP3"
ROOTS = {
    "CL": "WTI crude", "BZ": "Brent crude", "NG": "Natural gas", "RB": "Gasoline",
    "HO": "Heating oil", "GC": "Gold", "SI": "Silver", "HG": "Copper",
    "PL": "Platinum", "PA": "Palladium", "ZC": "Corn", "ZW": "Chicago wheat",
    "KE": "KC wheat", "ZS": "Soybeans", "ZM": "Soybean meal", "ZL": "Soybean oil",
    "LE": "Live cattle", "HE": "Lean hogs", "GF": "Feeder cattle",
    "BTC": "Bitcoin", "ETH": "Ether",  # CME crypto futures: Bitcoin from Dec 2017, Ether from Feb 2021
}
MONTHS = "FGHJKMNQUVXZ"

HERE = Path(__file__).resolve().parent
DATA = next(p for p in (HERE.parent, HERE.parent / "data") if (p / "crsp").exists())


def load_env(path=DATA / ".env"):
    """Put KEY=value lines of the git-ignored .env into os.environ (never printed)."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_env()
OUT = DATA / "commodities"


def contract_key(symbol, root, trade_year):
    """(year, month) of an outright like CLZ5 or CLZ25; None for spreads and others."""
    m = re.fullmatch(rf"{root}([{MONTHS}])(\d{{1,2}})", symbol)
    if not m:
        return None
    month = MONTHS.index(m.group(1)) + 1
    digits = m.group(2)
    if len(digits) == 2:
        return 2000 + int(digits), month
    # one-digit year: the first year ending in that digit not before trade_year - 1
    year = trade_year - 1 + (int(digits) - (trade_year - 1)) % 10
    return year, month


def roll(bars, root):
    bars = bars.copy()
    keys = [contract_key(s, root, d.year) for s, d in zip(bars.symbol, bars.date)]
    bars["key"] = keys
    bars = bars[bars.key.notna() & (bars.close > 0)]
    bars["expiry"] = [y * 12 + m for y, m in bars.key]
    px = bars.pivot_table(index="date", columns="expiry", values="close", aggfunc="last")
    vol = bars.pivot_table(index="date", columns="expiry", values="volume", aggfunc="sum").fillna(0)

    rows, held = [], None
    for i, date in enumerate(px.index):
        if held is not None:
            prev = px.index[i - 1]
            p0, p1 = px.at[prev, held], px.at[date, held] if held in px.columns else np.nan
            ret = p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan
        else:
            ret = np.nan
        # choose the contract to hold from today's close onward
        v = vol.loc[date]
        v = v[v > 0]
        if held is not None:
            v = v[v.index >= held]
        if len(v):
            held = v.idxmax()
        later = px.loc[date].dropna()
        later = later[later.index > held] if held is not None else later.iloc[:0]
        if held is not None and len(later) and np.isfinite(px.at[date, held]):
            nxt = later.index[0]
            carry = (np.log(px.at[date, held]) - np.log(later.iloc[0])) * 12 / (nxt - held)
        else:
            carry = np.nan
        rows.append((date, ret, held, carry, vol.loc[date].sum()))
    out = pd.DataFrame(rows, columns=["date", "ret", "held_expiry", "carry", "volume"])
    out["root"] = root
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--roots", nargs="*", default=list(ROOTS))
    ap.add_argument("--yes", action="store_true", help="download (otherwise only price the request)")
    args = ap.parse_args()

    client = db.Historical()
    request = dict(dataset=DATASET, symbols=[f"{r}.FUT" for r in args.roots], stype_in="parent",
                   schema="ohlcv-1d", start=args.start, end=args.end)
    cost = client.metadata.get_cost(**request)
    print(f"Cost of {len(args.roots)} roots, {args.start} to {args.end}: ${cost:.2f}")
    if not args.yes:
        print("Run again with --yes to download.")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    raw = client.timeseries.get_range(**request).to_df().reset_index()
    raw["date"] = raw.ts_event.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    raw = raw[["date", "symbol", "instrument_id", "open", "high", "low", "close", "volume"]]
    raw.to_parquet(OUT / "futures_bars.parquet", index=False)

    series = []
    for root in args.roots:
        bars = raw[raw.symbol.str.startswith(root)]
        s = roll(bars, root)
        series.append(s)
        r = s.ret.dropna()
        print(f"{root:3s} {ROOTS[root]:15s} days {len(r):5d}  ann. vol {r.std() * np.sqrt(252):6.1%}"
              f"  rolls {s.held_expiry.diff().ne(0).sum() - 1:4d}")
    pd.concat(series).to_parquet(OUT / "futures_daily.parquet", index=False)
    print(f"Wrote {OUT / 'futures_daily.parquet'}")


if __name__ == "__main__":
    main()
