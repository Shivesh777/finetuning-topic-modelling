"""Daily returns as matrices, the investable universe and holding-period returns.

`Market.load` reads data/crsp/ (see scripts/00_prepare_crsp.py for the file
layout) into dense (days x stocks) arrays. Three conventions matter for
honesty and are enforced here rather than in the experiments:

Universe.  The list of the 1,000 largest stocks formed at the end of June of
    year Y is used for portfolios formed from the last trading day of that June
    to the day before the last trading day of the next June. Membership is
    therefore known when a portfolio is formed.
Filled returns.  Where the daily source has no data for a stock that is still
    listed (see 00_prepare_crsp.py), its monthly return is spread evenly over
    the trading days of the month. Such days have `observed == False`: they
    enter realised portfolio returns, never a feature.
Holding periods.  A portfolio formed at the close of day f earns the returns
    of days f+1 .. f', where f' is the next formation day. A stock that stops
    trading inside the period earns zero from then on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

ALIVE_WINDOW = 5        # a stock is investable if it has a return in the last 5 trading days


@dataclass
class Market:
    dates: pd.DatetimeIndex          # trading days (D,)
    permnos: np.ndarray              # stock identifiers (N,), sorted
    ret: np.ndarray                  # (D, N) float32 daily returns, NaN = no return
    observed: np.ndarray             # (D, N) bool, True = a daily return from the source
    me: np.ndarray                   # (D, N) float32 market value, carried forward
    factors: pd.DataFrame            # mktrf, smb, hml, rf by trading day (decimals)
    lists: dict                      # year of formation -> bool (N,) membership
    tickers: pd.Series               # ticker -> column of the stock

    @property
    def rf(self) -> np.ndarray:
        return self.factors["rf"].to_numpy()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, folder: str | Path = None, start: str | None = None) -> "Market":
        folder = Path(folder) if folder is not None else config.CRSP
        daily = pd.read_parquet(folder / "daily.parquet")
        factors = pd.read_parquet(folder / "ff_daily.parquet").set_index("date")
        if start is not None:
            daily = daily[daily["date"] >= start]
        dates = pd.DatetimeIndex(np.sort(daily["date"].unique())).intersection(factors.index)
        permnos = np.sort(daily["permno"].unique())
        d = dates.get_indexer(daily["date"])
        p = np.searchsorted(permnos, daily["permno"].to_numpy())
        ok = d >= 0
        shape = (len(dates), len(permnos))
        ret = np.full(shape, np.nan, dtype=np.float32)
        me = np.full(shape, np.nan, dtype=np.float32)
        ret[d[ok], p[ok]] = daily["ret"].to_numpy()[ok]
        me[d[ok], p[ok]] = daily["me"].to_numpy()[ok]
        observed = ~np.isnan(ret)
        monthly_path = folder / "monthly.parquet"
        if monthly_path.exists():
            _fill_from_monthly(ret, observed, dates, permnos, pd.read_parquet(monthly_path))
        me = np.array(pd.DataFrame(me).ffill(limit=260), dtype=np.float32)

        universe = pd.read_parquet(folder / "universe.parquet")
        lists = {}
        for year, part in universe.groupby(universe["formed"].dt.year):
            lists[int(year)] = np.isin(permnos, part["permno"].to_numpy())
        tickers = pd.Series(dtype=np.int64)
        if (folder / "tickers.parquet").exists():
            t = pd.read_parquet(folder / "tickers.parquet").sort_values("last_date")
            t = t[t["permno"].isin(permnos)].drop_duplicates("ticker", keep="last")
            tickers = pd.Series(np.searchsorted(permnos, t["permno"].to_numpy()),
                                index=t["ticker"].to_numpy())
        return cls(dates, permnos, ret, observed, me, factors.reindex(dates), lists, tickers)

    # ----------------------------------------------------------------- universe
    def list_year(self, day: int) -> int:
        """Formation year of the list in force for a portfolio formed on day index `day`."""
        date = self.dates[day]
        last_june_day = (date.month == 6
                         and (day + 1 == len(self.dates) or self.dates[day + 1].month != 6))
        return date.year if (date.month >= 7 or last_june_day) else date.year - 1

    def complete(self, day: int) -> bool:
        """Whether the daily file holds every stock of the list in force on this day
        (see config.DAILY_RETURNS_COMPLETE_UNTIL)."""
        last, year = config.DAILY_RETURNS_COMPLETE_UNTIL, self.dates[day].year
        return last is None or year <= last or self.list_year(day) == year

    def eligible(self, day: int) -> np.ndarray:
        """Stocks that can be held from the close of `day`: on the list and still trading."""
        year = self.list_year(day)
        if year not in self.lists:
            return np.zeros(len(self.permnos), dtype=bool)
        recent = ~np.isnan(self.ret[max(0, day - ALIVE_WINDOW + 1):day + 1])
        return self.lists[year] & recent.any(axis=0)

    # ------------------------------------------------------------------ periods
    def formation_days(self, freq: str) -> np.ndarray:
        """Day indices on which portfolios are formed: the last trading day of each
        month ("M"), of each week ("W"), or of every second week ("2W")."""
        dates = self.dates
        if freq == "M":
            key = dates.year * 12 + dates.month
        elif freq in ("W", "2W"):
            week = (dates - pd.Timestamp("2000-01-03")).days // 7     # weeks start on Monday
            key = week // 2 if freq == "2W" else week
        else:
            raise ValueError(f"unknown frequency {freq!r}")
        key = np.asarray(key)
        # The final day closes a period only if the next business day starts a new one.
        after = dates[-1] + pd.offsets.BDay(1)
        if freq == "M":
            closed = after.month != dates[-1].month
        else:
            closed = (after - dates[-1]).days > 1 and (
                freq == "W" or ((after - pd.Timestamp("2000-01-03")).days // 7) % 2 == 0)
        return np.flatnonzero(np.r_[key[1:] != key[:-1], closed])

    def holding_returns(self, forms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(returns (P-1, N), risk-free (P-1,)) of the periods between consecutive formation
        days. NaN for a stock with no return at all during the period."""
        filled = np.where(np.isnan(self.ret), 0.0, self.ret).astype(np.float64)
        growth = np.vstack([np.zeros((1, filled.shape[1])), np.cumsum(np.log1p(filled), axis=0)])
        count = np.vstack([np.zeros((1, filled.shape[1])),
                           np.cumsum(~np.isnan(self.ret), axis=0)])
        rf_growth = np.r_[0.0, np.cumsum(np.log1p(self.rf))]
        a, b = forms[:-1] + 1, forms[1:] + 1          # days a .. b-1 = f+1 .. f'
        out = np.expm1(growth[b] - growth[a])
        out[(count[b] - count[a]) == 0] = np.nan
        return out, np.expm1(rf_growth[b] - rf_growth[a])


def _fill_from_monthly(ret, observed, dates, permnos, monthly) -> None:
    """Spread a monthly return evenly over the month's trading days for every stock-month
    that has a monthly return and not a single daily one (in place)."""
    monthly = monthly.dropna(subset=["ret"])
    monthly = monthly[monthly["permno"].isin(permnos) & (monthly["ret"] > -1.0)]
    month_of_day = np.asarray(dates.year * 12 + dates.month)
    months, first = np.unique(month_of_day, return_index=True)
    bounds = np.r_[first, len(dates)]
    key = np.asarray(monthly["date"].dt.year * 12 + monthly["date"].dt.month)
    m = np.searchsorted(months, key)
    known = m < len(months)
    known[known] = months[m[known]] == key[known]
    p = np.searchsorted(permnos, monthly["permno"].to_numpy())
    r = monthly["ret"].to_numpy(dtype=np.float64)
    for mi, pi, ri in zip(m[known], p[known], r[known]):
        lo, hi = bounds[mi], bounds[mi + 1]
        if not observed[lo:hi, pi].any():
            ret[lo:hi, pi] = (1.0 + ri) ** (1.0 / (hi - lo)) - 1.0
