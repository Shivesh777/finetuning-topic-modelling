"""Daily returns of commodity ETFs and commodity-equity ETFs from CRSP on WRDS.

These are the tickers FNSPID files commodity news under, so they link the news
(data/news/links.parquet) to tradable returns. Futures (pull_futures.py) are the
cleaner return series; the ETFs show what an investor could actually hold.

    python3 pull_etfs.py --wrds-user YOUR_WRDS_USERNAME

Uses the password saved in ~/.pgpass. Writes data/commodities/etf_daily.parquet.
"""
import argparse
import os
from pathlib import Path

import wrds

TICKERS = ["USO", "BNO", "UNG", "UGA", "DBO", "GLD", "IAU", "SLV", "PPLT", "PALL", "CPER",
           "DBA", "CORN", "WEAT", "SOYB", "DBC", "XLE", "XOP", "OIH", "GDX"]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrds-user", required=True)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()

    conn = wrds.Connection(wrds_username=args.wrds_user)
    names = conn.raw_sql(
        """select distinct permno, ticker, comnam, namedt, nameenddt
           from crsp.stocknames
           where ticker in %(t)s and shrcd = 73
             and nameenddt >= %(s)s and namedt <= %(e)s""",
        params={"t": tuple(TICKERS), "s": args.start, "e": args.end},
        date_cols=["namedt", "nameenddt"])
    print(names.sort_values("ticker").to_string(index=False))

    daily = conn.raw_sql(
        """select permno, date, ret, prc, vol, shrout
           from crsp.dsf
           where permno in %(p)s and date between %(s)s and %(e)s""",
        params={"p": tuple(int(p) for p in names.permno.unique()), "s": args.start, "e": args.end},
        date_cols=["date"])
    conn.close()

    # attach the ticker that was valid on each date
    daily = daily.merge(names[["permno", "ticker", "namedt", "nameenddt"]], on="permno")
    daily = daily[(daily.date >= daily.namedt) & (daily.date <= daily.nameenddt)]
    daily = daily.drop(columns=["namedt", "nameenddt"]).sort_values(["ticker", "date"])

    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT / "etf_daily.parquet", index=False)
    print(daily.groupby("ticker").date.agg(["min", "max", "count"]).to_string())
    missing = sorted(set(TICKERS) - set(daily.ticker))
    if missing:
        print("Not found in CRSP:", missing)


if __name__ == "__main__":
    main()
