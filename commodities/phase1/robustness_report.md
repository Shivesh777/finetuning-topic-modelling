# Phase 1: robustness and leakage audit of the saved equity forecasts

Everything below is computed from the saved forecasts in `data/predictions/`. The training scripts are not in the folder, so statements about how the forecasts were produced are either checked against the saved data (marked **verified**) or taken from the team's description of the schedule (marked **assumed**).

## 1. Bottom line

- The specification chosen on the validation split does not carry its validation performance into the test split at any frequency. Validation Sharpe ratios of 1.80 (monthly), 2.12 (biweekly), 1.50 (weekly) become test Sharpe ratios of -0.17, 0.40, 0.07 after 10 bp per side.
- Across all 375 real specifications, the median test Sharpe ratio is between -0.02 and 0.22 before costs and negative after costs at every frequency; most specifications lose money net of 10 bp (only 24-33% have a positive Sharpe).
- Costs matter a great deal: the chosen strategies turn over 2.8, 2.1, 0.8 times the book per period (monthly, biweekly, weekly), where each leg sums to 1 so that replacing the entire book would be 4.0.
- For the chosen model, no 95% interval for the Sharpe ratio excludes zero, and no deflated Sharpe ratio exceeds 0.75 (the best case, the best-in-hindsight monthly specification with only 6 effective trials assumed).
- The saved shuffled-topic placebo is a single shuffle run through 15 models. It is not a permutation test, and at monthly frequency it contains specifications that look as good as real ones.
- These are findings about the saved forecasts and a 21 / 42 / 81-period test window. They do not show that the text signal is worthless; they show that this evidence cannot distinguish it from luck.

## 2. What was run

| item | setting |
|---|---|
| experiments | 26 feature sets x 15 models = 390 per frequency (375 real, 15 on the shuffled-topic placebo `narr_lda_placebo`) |
| splits | validation 2020-2021 and test 2022-2024 as saved; test forecasts come from a refit on training + validation (assumed, per the team's schedule) |
| decile strategy | long top 10% of forecasts, short bottom 10%, equal weighted, rebuilt every period |
| buffered strategy | enter at the top / bottom 10%, keep a name until it leaves the top / bottom 20% |
| costs | 0, 5, 10, 20 bp per side on turnover (sum of absolute weight changes; closing the book at a gap is charged) |
| selection | buffer strategy, Sharpe at 10 bp on the validation split, among the real specifications only; fixed before looking at test |
| bootstrap | circular moving blocks of 3 / 4 / 8 periods (M / 2W / W), 10000 draws, seed 0 |
| permutation null | 1000 draws, each giving every stock another stock's full forecast history (one relabelling per draw); keeps persistence, turnover and cost; removes the link between forecasts and that stock's returns |
| gaps | the panel skips Feb-Jun holding months (section 6); the book is closed and rebuilt across any gap longer than 1.6x the median spacing |
| Sharpe | annualised with sqrt(12), sqrt(26), sqrt(52) on the periods available; the periods are not contiguous |

## 3. The model chosen on validation

| frequency | chosen specification | validation Sharpe | test Sharpe 0 / 5 / 10 / 20 bp | turnover / period | max drawdown | 95% bootstrap interval, Sharpe | same, minus price-only model of the same type | test periods |
|---|---|---|---|---|---|---|---|---|
| monthly | `all_finbert__xgb` | 1.80 | 0.06 / -0.06 / -0.17 / -0.41 | 2.76 | 13.4% | [-2.50, 1.23] | [-2.93, 1.38] | 21 |
| biweekly | `all_finbert__xgb` | 2.12 | 0.76 / 0.58 / 0.40 / 0.04 | 2.11 | 14.3% | [-1.23, 2.02] | [-1.43, 2.65] | 42 |
| weekly | `text_nolbert__ipca_rk` | 1.50 | 0.39 / 0.23 / 0.07 / -0.25 | 0.78 | 10.3% | [-1.35, 1.34] | [-1.56, 1.77] | 81 |

The chosen specification differs by frequency, and the weekly choice changed when I fixed the gap handling; a result that depends on that is fragile.

## 4. All specifications

Test split, buffered strategy, real specifications only.

| frequency | median Sharpe at 0 / 5 / 10 / 20 bp | share with positive Sharpe at 0 bp / 10 bp | median turnover / period | price-only models (median / max at 10 bp) | placebo range at 10 bp | rank correlation, validation vs test Sharpe across specs |
|---|---|---|---|---|---|---|
| monthly | 0.01 / -0.13 / -0.24 / -0.47 | 51% / 33% | 2.02 | -0.13 / 0.39 | [-0.83, 1.53] | -0.04 |
| biweekly | -0.02 / -0.22 / -0.36 / -0.63 | 49% / 24% | 1.28 | -0.16 / 0.14 | [-1.42, 0.73] | 0.22 |
| weekly | 0.22 / -0.07 / -0.29 / -0.63 | 66% / 29% | 0.95 | 0.01 / 0.21 | [-0.80, 0.42] | 0.29 |

The validation-to-test rank correlation is the direct measure of whether selecting on validation helps: it is near zero at monthly frequency and modest at the others. Decile and buffered strategies give the same picture (`robustness_summary.csv` has both).

## 5. Multiple testing, placebo, permutation

**Trials.** 375 real specifications per frequency, but their returns are highly correlated. The participation ratio of the eigenvalues of their return correlations gives 6.3 (monthly), 8.4 (biweekly), 7.8 (weekly) effective independent trials. The deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) is computed with N = effective trials, N = 26 (one per feature set) and N = 375, using the cross-sectional variance of the 375 per-period Sharpe ratios. Which N is right is a judgement; all three are shown.

| frequency | best specification in test (hindsight) | its Sharpe at 10 bp | deflated, N_eff / 26 / 375 | chosen model: deflated at N=1 (probability true Sharpe > 0) | permutation p, gross / net of 10 bp | permutation null 95th percentile (net) |
|---|---|---|---|---|---|---|
| monthly | `narr_lda__rf` | 1.42 | 0.75 / 0.57 / 0.29 | 0.41 | 0.493 / 0.241 | 0.59 |
| biweekly | `text_lda__xgb` | 1.14 | 0.61 / 0.43 / 0.16 | 0.69 | 0.165 / 0.019 | 0.14 |
| weekly | `all_nolbert__xgb` | 1.07 | 0.43 / 0.21 / 0.03 | 0.53 | 0.313 / 0.106 | 0.33 |

Reading: a deflated Sharpe near 1 would mean strong evidence; here the best-in-hindsight specification is at 0.03-0.75 depending on N, and the chosen model at N = 1 is at 0.4-0.7, which is weak evidence of any positive Sharpe even without deflation. The permutation p-values test the chosen model alone against random relabelling; they do not account for having chosen it from 375. The biweekly net p-value is much smaller than the gross one, which suggests the relabelled null pays more in costs than the chosen model; I have not resolved that, so I rely on the gross values.

**What the placebo is.** `narr_lda_placebo` is one set of shuffled topic features run through the same 15 models. It is a single draw, so its 15 results are highly dependent and it does not give a distribution of what shuffled features can achieve. Its best monthly specification reaches a Sharpe comparable to real ones. I therefore do not describe it as a permutation test. The permutation null above is a separate construction.

**Independent observations.** The test split has 21 monthly, 42 biweekly and 81 weekly periods, about 1.6-1.75 years of returns in each case (non-contiguous, within 2022-2024). The usual approximation for the standard error of an annualised Sharpe ratio is 1 / sqrt(years of data), about 0.8 here; the block-bootstrap intervals are wider (half-widths 1.2-1.9) because the samples are short and the returns skewed.

## 6. Feb-Jun holding months

From 2017 on the panel contains only formation dates in June-December, i.e. holding months July-January; February-June holding months are absent from training, validation and test. The team's schedule describes this ("January and July-December from 2017 on"), so it is by design, not a data error. What I cannot establish is why: if those months were removed after seeing results, that is selection; if they were removed for computing or data reasons, it is harmless. Consequences I can confirm: the test sample is 21 months out of 36; positions cannot be carried across the gap (handled); annualised statistics treat the available months as exchangeable. Not quantified: whether results differ in the omitted months.

## 7. Leakage audit

| item | status | evidence and limits |
|---|---|---|
| Topic models fit only through December 2019 | verified from saved metadata; code not rerun | All six `quality.json` files give `fit_until` 2019-12-31 and 140,209 fitting articles. Separate caveat: the team reports the NoLBERT encoder window as Jan 2015 - Aug 2020 (per the user), which would overlap the validation split for text-derived features; its adaptation objective is undocumented, and the encoder checkpoint and script are not in the folder. |
| Rolling covariance features are lagged | verified on a sample | I rebuilt NoLBERT-20 narrative shocks from `daily_sums.parquet` and CRSP returns and compared them with the saved panel features (5 formation dates x 4 topics, monthly panel). Rank correlation with a window ending at the formation date is 0.92, ending the day before 0.93, ending 21 days later 0.72, 63 days later 0.41. The features therefore use information up to the formation date, not beyond; a one-day overlap cannot be ruled out. The exact kernel is not known. |
| Feature scaling | verified by construction | Every feature block is rank-normalised across stocks within each date (per-date mean 0, standard deviation 0.578, the value for a uniform variable on [-1, 1]), so no global statistic from the test period enters. |
| Hyperparameters and model selection use training data only | assumed | Per the team's schedule: settings fitted on 69 training months, chosen on 14 validation months, refitted once on 83 months, applied unchanged to the 21 test months. Cannot be checked without the scripts. |
| Saved forecasts free of test contamination | partly verified | All 2,340 saved arrays (780 per frequency) match the panel's validation and test row counts exactly. Whether training touched test rows cannot be verified here. |
| Realised returns are forward returns | verified | Saved `ret` equals the compounded CRSP daily return from the next business day to the next formation date (rank correlation 1.0 on consecutive months). `rev_5` and `rev_21` correlate 0.86-1.0 with past returns and about 0 with the next week. |
| Firm news-count feature | verified on a sample | Outside 2024, rank correlation with the trailing 30-day article count is 0.81-0.93 and with the following 30 days 0.35-0.54 (persistence of news flow, not look-ahead). In 2024 it drops to 0.38-0.45 because the news source changes (below). |
| Timestamps precede the trading cutoff | cannot be verified | Articles carry a calendar date but no time of day, so news dated on a formation day may have been published after the close. The lag evidence above suggests the pipeline leaves a day, but this is not proven. |
| Stock universe | mostly verified, one issue | 98.7% of monthly rows (99.9% weekly) are in the top-1,000 list formed at the most recent June 30. About 1.1% of monthly rows (0.1% weekly) are stocks that appear only in the following June's list, a small look-ahead in universe membership; its effect is not quantified. Delisting returns are not documented. |
| News coverage changes in 2024 | issue, not quantified | January 2024 has 22 articles; from February 2024 about 40% fewer tickers per month have news than in 2023. The test split includes mid-2024, so text features are thinner there. |
| Selection on test | controlled | The chosen specification per frequency was fixed on the validation split. The best-in-test specification is shown only as hindsight. |

## 8. Confirmatory versus exploratory

Exploratory: everything in this report. The selection rule was fixed before reading test results, but the strategies, cost levels, blocks and trial-counting choices were made by me after the earlier first pass over the same forecasts, which also used test results for a first look. The permutation and deflated-Sharpe calculations rest on stated assumptions (effective trials, a persistence-preserving null) that a reviewer could dispute. No result here should be presented as a confirmatory test.

## 9. Corrections to what I reported earlier

- My first backtest used gross returns and listed best-in-hindsight Sharpe ratios (1.7-1.9); those are not what a researcher could have selected in advance.
- My first permutation null redrew random forecasts every period, which made turnover and costs far higher than for any persistent forecast; it was an unfair comparison and is replaced.
- My first run carried positions across the missing months; that is fixed. All numbers here come from the corrected run.

## 10. Reproduce

```bash
cd research/phase1
python3 robustness.py   # about 2 minutes; writes robustness_summary.csv, findings.json, the two figures, config.json
python3 make_report.py
```
Seed 0; settings in `config.json`. Figures: `placebo_comparison.png`, `performance_after_costs.png`.