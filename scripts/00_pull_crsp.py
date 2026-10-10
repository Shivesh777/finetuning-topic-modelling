"""Build data/crsp/ directly from WRDS (needs a WRDS account with CRSP access).

    pip install wrds
    python scripts/00_pull_crsp.py [--first-year 2011] [--last-year 2024] [--out data/crsp_wrds]

Writes the five files that ftm.data.returns.Market reads (see data/README.md):
universe, daily, monthly, tickers and ff_daily. Compared with the files in the
team's data folder, which scripts/00_prepare_crsp.py assembled from two older
extracts, this pull has daily returns for every stock of every list in every
month, so nothing has to be filled from monthly returns. Afterwards set
DAILY_RETURNS_COMPLETE_UNTIL = None in ftm/config.py: the panels then use every
month instead of leaving out January to June from 2017 on.

NOT RUN for the results in FINDINGS.md: the environment that produced them had
no WRDS login. The queries follow the CRSP schema (crsp.dsf, crsp.msf,
crsp.msenames); check the row counts printed at the end against REFERENCE.md
before relying on it. The universe will differ slightly from ours, because the
older extract applied a share-code filter that we could not recover.

CRSP data are licensed: never commit data/crsp/ or anything derived from it at
the stock level (data/panels/, data/predictions/).
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
import requests

from ftm import config

FRENCH = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
# Ordinary common shares (share codes 10 and 11) listed on NYSE, AMEX or Nasdaq.
COMMON = "n.shrcd in (10, 11) and n.exchcd in (1, 2, 3)"


def june_lists(db, first: int, last: int, size: int) -> pd.DataFrame:
    """The `size` largest common stocks by market value at the end of June of each year."""
    june = db.raw_sql(f"""
        select m.permno, m.date, abs(m.prc) * m.shrout as me
        from crsp.msf m join crsp.msenames n
          on m.permno = n.permno and n.namedt <= m.date and m.date <= n.nameendt
        where extract(month from m.date) = 6 and {COMMON}
          and m.date between '{first - 1}-06-01' and '{last}-06-30'""", date_cols=["date"])
    june = june.dropna(subset=["me"]).sort_values(["date", "me"], ascending=[True, False])
    june["rank"] = june.groupby(june["date"].dt.year).cumcount() + 1
    june = june[june["rank"] <= size]
    june["formed"] = pd.to_datetime(june["date"].dt.year.astype(str) + "-06-30")
    return june[["formed", "permno", "me", "rank"]].astype({"permno": np.int64})


def factors(first: int) -> pd.DataFrame:
    archive = zipfile.ZipFile(io.BytesIO(requests.get(FRENCH, timeout=120).content))
    text = archive.read(archive.namelist()[0]).decode("latin-1")
    lines = [ln for ln in text.splitlines() if ln[:8].strip().isdigit() and len(ln.split(",")) >= 5]
    f = pd.read_csv(io.StringIO("\n".join(lines)), header=None,
                    names=["date", "mktrf", "smb", "hml", "rf"])
    f["date"] = pd.to_datetime(f["date"].astype(str), format="%Y%m%d")
    f[["mktrf", "smb", "hml", "rf"]] = f[["mktrf", "smb", "hml", "rf"]].astype(float) / 100.0
    return f[f["date"] >= f"{first}-01-01"].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-year", type=int, default=2011,
                    help="one year before the news starts: features need a year of returns")
    ap.add_argument("--last-year", type=int, default=int(config.NEWS_END[:4]))
    ap.add_argument("--out", type=Path, default=config.CRSP,
                    help="folder for the five files (default data/crsp/, which this overwrites)")
    a = ap.parse_args()
    import wrds

    db = wrds.Connection()
    universe = june_lists(db, a.first_year, a.last_year, config.UNIVERSE_SIZE)
    members = ", ".join(str(p) for p in sorted(universe["permno"].unique()))
    window = f"between '{a.first_year}-01-01' and '{a.last_year}-12-31'"
    daily = db.raw_sql(f"""
        select permno, date, ret, abs(prc) * shrout as me from crsp.dsf
        where date {window} and permno in ({members})""", date_cols=["date"])
    monthly = db.raw_sql(f"""
        select permno, date, ret from crsp.msf
        where date {window} and permno in ({members})""", date_cols=["date"])
    names = db.raw_sql(f"""
        select permno, ticker, nameendt as last_date from crsp.msenames n
        where permno in ({members}) and ticker is not null""", date_cols=["last_date"])
    names = names.sort_values("last_date").drop_duplicates("permno", keep="last")
    daily = daily.dropna(subset=["ret"]).astype({"permno": np.int64, "ret": np.float32})
    monthly = monthly.dropna(subset=["ret"]).astype({"permno": np.int64})

    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tables = {"universe": universe, "daily": daily.sort_values(["date", "permno"]),
              "monthly": monthly.sort_values(["permno", "date"]),
              "tickers": names.astype({"permno": np.int64}), "ff_daily": factors(a.first_year)}
    for name, table in tables.items():
        table.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(table):,} rows")
    print(f"{universe['permno'].nunique():,} stocks in {universe['formed'].nunique()} June lists; "
          f"daily returns {daily['date'].min().date()} to {daily['date'].max().date()}")


if __name__ == "__main__":
    main()
