"""Assemble the return data in data/crsp/ from the files of the earlier project.

    python scripts/00_prepare_crsp.py [--legacy-root ..]

CRSP is licensed (WRDS): nothing in data/crsp/ may be committed or shared
outside the team. If you have WRDS access, scripts/00_pull_crsp.py builds the
same files directly and without the limitation described below.

Inputs (read-only), relative to --legacy-root:
  replication-kevin/paper_data/CRSP Data/clean/crsp_daily_<year>.parquet
      every common stock, daily, 1984 to 2016
  narrative-consolidated/data/crsp_daily_2014_2024.parquet
      the 1,000 largest stocks of each calendar year (chosen on that year's
      June market value), daily, with tickers
  narrative-consolidated/data/crsp_monthly_2014_2024.parquet
      every common stock, monthly returns
  narrative-consolidated/data/external/F-F_Research_Data_Factors_daily_CSV.zip

Outputs:
  universe.parquet   formed (30 June of year Y), permno, me, rank: the 1,000
                     largest stocks by market value at the end of June (from
                     the second source from 2014 on, from the full files
                     before). A list is used from the first trading day of July
                     to the next June, so membership is always known when a
                     portfolio is formed.
  daily.parquet      date, permno, ret, me for every stock of any list, 2011 on
  monthly.parquet    permno, date, ret (2014 on), used to fill the daily returns
                     the second source does not have
  tickers.parquet    permno, ticker, last_date (the stock's latest ticker)
  ff_daily.parquet   date, mktrf, smb, hml, rf in decimals

Limitation of the second source: a stock in the list of June Y that drops out
of the list of June Y+1 has no daily data from January to June of Y+1 (about
110 stocks a year from 2017 on). ftm.data.returns fills those days with the
monthly return spread evenly over the month and marks them as not observed.
"""
from pathlib import Path
import argparse
import io
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config

FIRST_YEAR = 2011


def june_list(daily: pd.DataFrame, year: int, size: int) -> pd.DataFrame:
    """The `size` largest stocks by their last market value in June of `year`."""
    june = daily[(daily["date"].dt.year == year) & (daily["date"].dt.month == 6)]
    me = june.sort_values("date").groupby("permno")["me"].last().dropna()
    top = me.sort_values(ascending=False).head(size)
    return pd.DataFrame({"formed": pd.Timestamp(year, 6, 30), "permno": top.index,
                         "me": top.to_numpy(), "rank": np.arange(1, len(top) + 1)})


def read_full(folder: Path, year: int) -> pd.DataFrame:
    d = pd.read_parquet(folder / f"crsp_daily_{year}.parquet",
                        columns=["permno", "date", "ret", "price_abs", "shrout"])
    d["me"] = d["price_abs"] * d["shrout"]
    return d[["date", "permno", "ret", "me"]]


def read_factors(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = [ln for ln in text.splitlines() if ln[:8].strip().isdigit() and len(ln.split(",")) >= 5]
    f = pd.read_csv(io.StringIO("\n".join(lines)), header=None,
                    names=["date", "mktrf", "smb", "hml", "rf"])
    f["date"] = pd.to_datetime(f["date"].astype(str), format="%Y%m%d")
    f[["mktrf", "smb", "hml", "rf"]] = f[["mktrf", "smb", "hml", "rf"]].astype(float) / 100.0
    return f[f["date"] >= f"{FIRST_YEAR}-01-01"].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-root", default=str(ROOT.parent))
    a = ap.parse_args()
    root = Path(a.legacy_root)
    full_dir = root / "replication-kevin" / "paper_data" / "CRSP Data" / "clean"
    old = root / "narrative-consolidated" / "data"
    size = config.UNIVERSE_SIZE

    listed = pd.read_parquet(old / "crsp_daily_2014_2024.parquet")
    listed["permno"] = listed["permno"].astype("int32")
    last_full = max(int(p.stem.split("_")[-1]) for p in full_dir.glob("crsp_daily_*.parquet"))

    # Lists: from the second source wherever it exists (its share-code filter matches
    # the monthly file), from the full files before.
    first_listed = int(listed["date"].dt.year.min())
    lists, parts = [], []
    full = {y: read_full(full_dir, y) for y in range(FIRST_YEAR, last_full + 1)}
    for year in range(FIRST_YEAR, int(listed["date"].dt.year.max()) + 1):
        lists.append(june_list(full[year] if year < first_listed else listed, year, size))
    universe = pd.concat(lists, ignore_index=True)
    for year in range(first_listed, last_full + 1):       # agreement where both exist
        both = set(june_list(full[year], year, size).permno) & set(lists[year - FIRST_YEAR].permno)
        print(f"list of June {year}: {len(both)} of {size} stocks also top-{size} in the full files")

    members = set(universe["permno"])
    for year in range(FIRST_YEAR, last_full + 1):
        parts.append(full[year][full[year]["permno"].isin(members)])
    parts.append(listed.loc[listed["date"].dt.year > last_full, ["date", "permno", "ret", "me"]])
    daily = pd.concat(parts, ignore_index=True).sort_values(["date", "permno"], ignore_index=True)
    daily["permno"] = daily["permno"].astype("int32")
    daily["ret"] = daily["ret"].astype("float32")

    tickers = (listed.dropna(subset=["ticker"]).sort_values("date").groupby("permno")
               .agg(ticker=("ticker", "last"), last_date=("date", "last")).reset_index())
    monthly = pd.read_parquet(old / "crsp_monthly_2014_2024.parquet")
    monthly = monthly[monthly["permno"].isin(members)].reset_index(drop=True)
    factors = read_factors(old / "external" / "F-F_Research_Data_Factors_daily_CSV.zip")

    config.CRSP.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(config.CRSP / "universe.parquet", index=False)
    daily.to_parquet(config.CRSP / "daily.parquet", index=False, compression="zstd")
    monthly.to_parquet(config.CRSP / "monthly.parquet", index=False)
    tickers.to_parquet(config.CRSP / "tickers.parquet", index=False)
    factors.to_parquet(config.CRSP / "ff_daily.parquet", index=False)
    print(f"universe: {universe.formed.dt.year.min()}-{universe.formed.dt.year.max()}, "
          f"{universe.permno.nunique():,} stocks; daily: {len(daily):,} rows "
          f"{daily.date.min().date()} to {daily.date.max().date()}; monthly: {len(monthly):,} rows; "
          f"tickers: {len(tickers):,}; factors to {factors.date.max().date()}")


if __name__ == "__main__":
    main()
