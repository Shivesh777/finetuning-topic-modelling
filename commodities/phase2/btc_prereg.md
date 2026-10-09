# Bitcoin (CME BTC futures) pilot: specification fixed before any Bitcoin regression is run

Exploratory extension of `pilot_prereg.md` (CL / GC pilot; its results and hash are unchanged). Everything not stated here is
identical to that document: weekly construction, shock definition, timing, controls, models, HAC lags (4), bootstrap
(block 4, 5,000 draws, seed 0), windows for validation (2020-2021) and test (2022-2024). `pilot_btc.py` prints the SHA-256
of this file with its results. Deviations, if any, are listed at the bottom.

## Data
- Returns: `futures_trading_daily.parquet`, root BTC (CME Bitcoin futures, Databento, 2017-12-18 to 2024-12-31),
  built exactly as for CL and GC. The contract trades almost around the clock Sunday evening to Friday evening; the weekly
  cutoff is the same Friday bar (00:00 UTC Saturday). Weekend moves in spot Bitcoin appear in Monday's bar.
- Text: the same frozen FinBERT-20 classes; no Bitcoin-specific topic is created and none exists in the saved models.
- Development window: 2018-01 to 2019-12 (about 100 weeks, the first trading week of December 2017 is dropped). This is short,
  so every Bitcoin result is exploratory.

## Narratives (chosen from `topic_labels.md` before looking at any Bitcoin return)
| Name | Definition | Role |
|---|---|---|
| `F_fed` | FinBERT-20 "Fed / Central Banks" (index 1) | primary, the only macro class with a stable meaning |
| `F_macro` | FinBERT-20 "Macro" (index 14) | secondary; its label does not match its content (see `topic_labels.md`) |
| `F_curr` | FinBERT-20 "Currencies" (index 8) | secondary; very small class |
| `F_oil` | FinBERT-20 "Energy / Oil" (index 6) | cross-narrative control, expected to be near zero |
No other narrative is tried. If a secondary or control narrative looks good after the fact, it stays labelled secondary.

## Tests
Primary: `F_fed` contemporaneous and `F_fed` forecasting (2 tests). Holm adjustment is reported (a) within Bitcoin (2 tests)
and (b) across all eight primary tests of the cross-asset study (CL x 4, GC x 2, BTC x 2). Confidence intervals are reported
unadjusted and Bonferroni-adjusted for the eight-test family (95% overall).

## Forecast target and timing
Identical to the CL/GC pilot: the signal uses shocks dated through the trading day before the last day of the signal week; the
target is the next week's return.

## Not done
Any narrative not listed, any Bitcoin-specific keyword set, spot Bitcoin, Ether, trading rules, any change after seeing results.
