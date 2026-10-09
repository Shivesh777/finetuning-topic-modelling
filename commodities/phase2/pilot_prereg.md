# CL / GC pilot: specification fixed before any regression is run

Written before `build_weekly_panel.py` and `pilot.py` were run. `pilot.py` prints the SHA-256 of this file
with its results. Anything not listed here is not part of the pilot. If the specification needs to change
after results are seen, the change is recorded in a "Deviations" section at the bottom, with the reason.

## Question
Do narrative attention shocks from the frozen text models (a) co-move with, and (b) predict, weekly returns of
WTI crude (CL) and gold (GC) futures, beyond what price-based controls explain?
Contemporaneous exposure and forecasting are separate analyses and are reported separately.

## Data
- Returns: `data/commodities/futures_trading_daily.parquet` (Databento CME, volume-based roll, Sunday evening
  merged into Monday, price returns on one contract at a time, so roll gaps are never returns; a futures price
  return is already an excess return).
- Text: `topics/nolbert/daily_sums.parquet` (60 topics) and `topics/finbert/daily_sums.parquet` (20 classes).
  No new topic definitions, no retraining. Topic models were fit on news through 2019-12-31.
- Controls: Fama-French `mktrf` from `crsp/ff_daily.parquet`.

## Narratives (chosen from topics.csv terms and headlines in `topic_labels.md`, before looking at returns)
| Name | Definition | Used for |
|---|---|---|
| `N_oil` | attention share of NoLBERT-60 t28 + t30 | CL, primary |
| `F_oil` | attention share of FinBERT-20 class "Energy / Oil" (index 6) | CL, primary |
| `F_gold` | attention share of FinBERT-20 class "Gold / Metals / Materials" (index 10) | GC, primary |
| `F_fed` | attention share of FinBERT-20 class "Fed / Central Banks" (index 1) | GC, secondary |
Cross-narrative controls (expected to be near zero, not hypotheses): `F_gold` on CL, `F_oil` on GC.
Not used: NoLBERT-20 (no macro topic), growth, inflation and China (no supported topic).

## Shocks
- Daily attention share = sum of article weights on the narrative's topic(s) over the day divided by the sum of
  all topic weights that day. Articles on Saturdays, Sundays and futures holidays are added to the next trading day.
- Daily shock z_t = attention_t minus the mean attention of the previous 5 trading days. Trading days with a total
  article weight below 20 have no shock.
- Weekly: weeks end on the last futures trading day of the ISO week, L_w. A week needs at least 3 valid daily shocks and
  at least 300 articles in total; otherwise it is dropped (this removes January 2024, when the news source is nearly empty).
- Contemporaneous shock Z_w = sum of z_t over trading days in (L_{w-1}, L_w].
- Forecasting signal S_w = sum of z_t over those same trading days **except L_w** itself. The target is the return over
  (L_w, L_{w+1}], so the signal uses only news dated at least one trading day before the start of the return window.
- Both are divided by their standard deviation in the 2015-2019 development sample.

## Returns and controls
- Weekly return r_w = compounded daily futures returns over the trading days in (L_{w-1}, L_w].
- Contemporaneous controls: market excess return over the same days, r_{w-1}, 4-week realised volatility (to w-1).
- Forecasting controls: r_w, 4-week realised volatility (to w). Benchmarks: expanding historical mean, and the controls-only model.

## Windows (by L_w for contemporaneous, by the target week's end for forecasting)
| Window | Dates | Role |
|---|---|---|
| Development | 2015-01 to 2019-12 | fit |
| Validation | 2020-01 to 2021-12 | report; coefficients fitted on development are applied unchanged |
| Test | 2022-01 to 2024-12 | coefficients fitted on development + validation, applied unchanged; run once |
Caveat recorded in advance: if the NoLBERT encoder was adapted on text through 2020-08-31, validation overlaps it; and the
FinBERT classifier's training data are undocumented. The test window is unaffected by the first.

## Models
1. Contemporaneous: r_w = a + b Z_w + c'controls + e. OLS, Newey-West standard errors with 4 weekly lags. Report b (per 1
   development-standard-deviation shock, in percentage points of weekly return), 95% CI, adjusted R-squared, and the
   increase in adjusted R-squared over the controls-only model.
2. Forecasting: r_{w+1} = a + b S_w + c'controls + e, same inference. Report out-of-sample R-squared against the expanding
   historical mean and against the controls-only model (frozen coefficients), directional accuracy, and a 95% interval for the
   difference in squared error from a moving-block bootstrap (block 4 weeks, 5,000 draws, seed 0) on the test weeks.
3. Timing check: the same contemporaneous regression with the **next** week's shock Z_{w+1} in place of Z_w; its coefficient
   should be near zero.

## Multiple testing
Primary tests: CL on `N_oil` and `F_oil`, GC on `F_gold`, each for models 1 and 2 (six tests). Test-window p-values are
reported raw and Holm-adjusted across these six. Everything else (F_fed, cross-narrative controls, timing check,
sensitivities) is labelled secondary or diagnostic.

## Sensitivities (reported, never used to choose a specification)
(a) drop weeks containing a Databento degraded day; (b) add t55 to `N_oil`.

## Not done in this pilot
Daily-frequency regressions, trading rules, any spec added after seeing test results, Bitcoin, Ether, other commodities.

## Implementation notes (added before the first run; they do not depend on returns)
- Daily shock baseline: the mean of the available attention values among the previous 5 trading days, requiring at least 3.
- A week's market return is the compounded `mktrf` over weekdays of the same ISO week; the futures week is the ISO week of
  its trade dates. Holiday weeks follow each calendar's own trading days.
- Forecasting rows are indexed by the signal week w; the target is the return of week w+1, and the window label (development /
  validation / test) is that of the target week.
- Volatility controls are the standard deviation of daily returns over the previous 4 weeks times sqrt(5).
- Standardisation of shocks and controls uses development-sample (2015-2019) standard deviations only.
