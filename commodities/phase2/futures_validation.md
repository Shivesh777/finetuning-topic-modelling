# Futures return validation

## 1. Gold (GC) against GLD from CRSP

- common days: 2516  (2015-01-02 to 2024-12-31)
- daily correlation futures vs GLD excess return: 0.966  (the futures bar closes about 3 hours after GLD, so daily is not exact)
- weekly correlation: 0.992; weekly slope of futures on GLD excess return: 1.001
- cumulative: futures +66.2%, GLD minus risk-free +79.9%; annualised log gap -0.79% per year
  (GLD's fee is 0.40% a year, and a futures roll earns or pays the carry, so a gap of a few tenths of a percent a year is expected;
   a gap of several percent a year would point to a construction error)
- 2015-2019: gap -0.26% per year, weekly corr 0.992
- 2020-2021: gap -1.92% per year, weekly corr 0.988
- 2022-2024: gap -0.93% per year, weekly corr 0.995

## 2. Crude (CL) against oil funds in CRSP

- DBO: daily corr 0.837, weekly corr 0.871
- XLE: daily corr 0.555, weekly corr 0.608
- XOP: daily corr 0.573, weekly corr 0.607
  DBO holds futures but with an optimised roll, so it should track CL closely but not exactly; the equity funds track it loosely.

## 3. Roll days (held contract changes)

| root | roll days | mean ret on roll days | std roll / non-roll | share of total variance on roll days | cumulative return with roll-day returns set to 0 |
|---|---|---|---|---|---|
| BTC | 84 | +0.2719% | 0.65 | 2.0% | +137.1% (all days: +187.9%) |
| CL | 120 | -0.0764% | 0.83 | 3.2% | -26.1% (all days: -34.7%) |
| GC | 50 | -0.1149% | 0.77 | 1.1% | +79.5% (all days: +69.3%) |
| HG | 50 | -0.0588% | 0.96 | 1.8% | +20.2% (all days: +16.2%) |

Expectation under a clean construction: roll days look like other days (std ratio near 1, variance share near the roll-day share of days, about 2-5%).

## 4. Databento degraded days

- degraded UTC days: 14  (2017-11-13, 2018-10-21, 2019-01-15, 2019-02-22, 2019-03-13, 2019-03-26, 2020-02-27, 2020-02-28, 2020-05-05, 2020-06-30, 2020-07-01, 2021-12-05, 2022-01-02, 2024-09-18)
- CL: 14 flagged trading days; mean |ret| flagged 2.905% vs others 1.808%; weekly corr with/without 0.9850; weeks changed by more than 0.5pp: 8; cumulative -34.7% vs -47.6%
- GC: 14 flagged trading days; mean |ret| flagged 0.579% vs others 0.650%; weekly corr with/without 0.9961; weeks changed by more than 0.5pp: 4; cumulative +69.3% vs +80.0%
- HG: 14 flagged trading days; mean |ret| flagged 0.701% vs others 1.023%; weekly corr with/without 0.9987; weeks changed by more than 0.5pp: 7; cumulative +16.2% vs +9.4%
- BTC: 13 flagged trading days; mean |ret| flagged 1.341% vs others 2.986%; weekly corr with/without 0.9993; weeks changed by more than 0.5pp: 8; cumulative +187.9% vs +196.5%

## 5. Hygiene

- BTC: duplicate dates 0, exact-zero returns 1.26%, weekdays covered [0, 1, 2, 3, 4], days with |ret| > 10%: 81
- CL: duplicate dates 0, exact-zero returns 0.46%, weekdays covered [0, 1, 2, 3, 4], days with |ret| > 10%: 20
- GC: duplicate dates 0, exact-zero returns 0.46%, weekdays covered [0, 1, 2, 3, 4], days with |ret| > 10%: 0
- HG: duplicate dates 0, exact-zero returns 0.66%, weekdays covered [0, 1, 2, 3, 4], days with |ret| > 10%: 0