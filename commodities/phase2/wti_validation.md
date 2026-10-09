# CL futures against WTI Cushing spot (EIA, via FRED)

- spot file: 2,526 usable daily values (2014-12-01 to 2025-01-15), 117 missing (holidays); 521 Friday-ending weeks with both returns defined

## 1. Weekly agreement

| sample | weeks | correlation | slope of futures on spot | std of (futures - spot), pp | largest |gap|, pp |
|---|---|---|---|---|---|
| all weeks | 521 | 0.914 | 0.77 | 2.82 | 50.4 |
| excluding Mar-Jul 2020 | 499 | 0.963 | 0.95 | 1.33 | 12.8 |
| development 2015-2019 | 260 | 0.961 | 0.95 | 1.28 | 8.0 |
| validation 2020-2021 | 105 | 0.882 | 0.65 | 5.86 | 50.4 |
| test 2022-2024 | 156 | 0.987 | 0.97 | 0.85 | 3.7 |

## 2. Largest discrepancies (futures minus spot, weekly)

| week ending | spot return | futures return | gap, pp | rolled that week | minimum spot price in week |
|---|---|---|---|---|---|
| 2020-04-03 | +83.2% | +32.8% | -50.4 | False | 14.10 |
| 2020-04-24 | -12.7% | -31.7% | -19.0 | False | -36.98 |
| 2020-03-27 | -20.5% | -7.6% | +12.9 | False | 15.48 |
| 2021-11-24 | +2.9% | -9.9% | -12.8 | False | 76.74 |
| 2021-12-03 | -15.2% | -2.9% | +12.4 | False | 65.44 |
| 2020-03-20 | -38.6% | -29.6% | +9.0 | True | 19.48 |
| 2020-05-01 | +23.3% | +14.6% | -8.7 | False | 12.17 |
| 2018-11-21 | -3.7% | -11.6% | -8.0 | False | 53.39 |
| 2018-11-30 | -6.7% | +0.7% | +7.3 | False | 50.06 |
| 2019-11-29 | +0.8% | -4.3% | -5.1 | False | 57.79 |

## 3. April 2020

| date | spot price | futures daily return | held contract (months since year 0) |
|---|---|---|---|
| 2020-04-15 | 19.96 | -2.7% | 24245 |
| 2020-04-16 | 19.82 | -1.8% | 24246 |
| 2020-04-17 | 18.31 | -5.2% | 24246 |
| 2020-04-20 | -36.98 | -15.4% | 24246 |
| 2020-04-21 | 8.91 | -34.1% | 24246 |
| 2020-04-22 | 13.64 | +3.4% | 24246 |
| 2020-04-23 | 15.06 | +19.3% | 24246 |
| 2020-04-24 | 15.99 | -0.6% | 24246 |

The spot price turns negative on 2020-04-20 (the May contract expiry) and the spot series then jumps to the June contract's level. The futures series had already rolled from May (held 24245) to June (24246) on 2020-04-16, so it never held the contract that settled below zero, and its returns on 20-21 April are the June contract's.

## 4. Rolls

- week in which the held contract changes: 101 weeks, std of gap 1.18 pp, mean gap +0.13 pp, mean |gap| 0.61 pp
- week after a roll: 101 weeks, std of gap 2.98 pp, mean gap -0.48 pp, mean |gap| 1.36 pp
- other weeks: 319 weeks, std of gap 3.12 pp, mean gap -0.19 pp, mean |gap| 0.85 pp

## 5. Does the cumulative gap follow the curve shape?

Futures return is below spot return when the curve is in contango (the held contract converges down toward spot) and above it in backwardation. So the annual sum of weekly gaps should rise with the average slope of the curve (the `carry` column: positive means backwardation).

| year | sum of weekly gaps, pp | mean annualised curve slope (positive = backwardation) |
|---|---|---|
| 2015 | -22.2 | -22.8% |
| 2016 | -28.2 | -27.1% |
| 2017 | -9.0 | -7.7% |
| 2018 | +3.5 | +4.5% |
| 2019 | -2.4 | -1.7% |
| 2020 | -63.2 | -35.4% |
| 2021 | +2.6 | +5.8% |
| 2022 | +13.7 | +16.6% |
| 2023 | +0.1 | +2.5% |
| 2024 | +8.2 | +8.7% |

Spearman correlation across the 10 years: 0.99; Pearson 0.95.

(No test statistic is attached to the ten-year comparison: ten points are too few for a formal test, so it is a direction check only.)
## 6. Note on the 2021-11-24 / 2021-12-03 discrepancies

The EIA spot series has no value for Friday 2021-11-26 (the Omicron sell-off day, when the futures series fell 11.9%). The spot week ending 2021-11-24 therefore misses that move and the next spot week absorbs it, giving offsetting gaps of -12.8 and +12.4 percentage points. This is a spot-calendar artefact, not a futures error.

## 7. Verdict

Outside March-July 2020 the weekly correlation is 0.96 (0.99 in the 2022-2024 test window), the annual gap between futures and spot follows the shape of the futures curve (direction check, ten years), the April 2020 collapse is handled by having rolled to the June contract before the May contract went negative, and the largest discrepancies are explained by the 2020 dislocations and the spot calendar. Nothing here indicates a construction error. Spot is not a tradable reference, so this validates behaviour, not levels.
