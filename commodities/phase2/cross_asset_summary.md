# Do news-attention shocks from frozen equity-news topic models move with, or predict, commodity and Bitcoin futures returns?

**Status.** Exploratory cross-asset external-validity test. It uses topic-level attention shocks from the team's frozen text models; it is **not** a replication of the Sparse IPCA factor analysis, because the factor returns and the loading matrix are not available. Specifications were written before the regressions ran (`pilot_prereg.md`, SHA-256 `1d51f6a83ddc26a3...` for CL and GC; `btc_prereg.md`, `fd7cbb8ee2126d1b...` for Bitcoin) and each regression was run once. Confidence intervals are labelled unadjusted or Bonferroni-adjusted throughout.

## 1. Hypotheses (two-sided; the shocks measure attention, not tone, so no sign was specified)

| id | asset | narrative | question | status |
|---|---|---|---|---|
| H1a/H1b | CL | oil (two definitions) | Do oil-attention shocks co-move with same-week CL returns beyond market, lagged return and volatility? | primary |
| H2a/H2b | CL | oil (two definitions) | Do shocks dated through the day before the last trading day predict next week's CL return beyond price controls? | primary |
| H3 | GC | gold | Do gold-attention shocks co-move with same-week GC returns? | primary |
| H4 | GC | gold | Do gold-attention shocks predict next week's GC return? | primary |
| H5 | BTC | Fed | Do Fed-attention shocks co-move with same-week BTC returns? | primary, exploratory |
| H6 | BTC | Fed | Do Fed-attention shocks predict next week's BTC return? | primary, exploratory |

The eight primary tests form one family for multiple-testing adjustment. Windows: development 2015-2019 (Bitcoin 2018-2019), validation 2020-2021, test 2022-2024. Coefficients are percentage points of weekly return per one development-sample standard deviation of the shock; HAC (Newey-West, 4 lags) standard errors; 149 test weeks for the contemporaneous tests and 147 for the forecasting tests.

## 2. Primary tests in the test window

| asset | narrative | analysis | coef | 95% CI, unadjusted | 95% CI, Bonferroni (8 tests) | raw p | Holm p (8 tests) | minimum detectable effect, 80% power: unadjusted / Bonferroni |
|---|---|---|---|---|---|---|---|---|
| CL | oil (NoLBERT-60 t28+t30) | contemporaneous | 1.30 | [0.11, 2.50] | [-0.37, 2.98] | 0.033 | 0.229 | 1.71 / 2.18 |
| CL | oil (FinBERT Energy/Oil) | contemporaneous | 0.78 | [-0.14, 1.69] | [-0.50, 2.05] | 0.095 | 0.460 | 1.30 / 1.67 |
| CL | oil (NoLBERT-60 t28+t30) | forecasting | -1.29 | [-2.31, -0.27] | [-2.71, 0.13] | 0.013 | 0.103 | 1.46 / 1.86 |
| CL | oil (FinBERT Energy/Oil) | forecasting | -0.42 | [-1.04, 0.20] | [-1.29, 0.45] | 0.186 | 0.558 | 0.89 / 1.14 |
| GC | gold (FinBERT Gold/Metals) | contemporaneous | -0.17 | [-0.36, 0.02] | [-0.43, 0.09] | 0.077 | 0.460 | 0.27 / 0.34 |
| GC | gold (FinBERT Gold/Metals) | forecasting | -0.01 | [-0.43, 0.41] | [-0.60, 0.58] | 0.954 | 0.954 | 0.60 / 0.77 |
| BTC | Fed (FinBERT Fed/Central Banks) | contemporaneous | -1.16 | [-2.49, 0.16] | [-3.01, 0.68] | 0.085 | 0.460 | 1.89 / 2.41 |
| BTC | Fed (FinBERT Fed/Central Banks) | forecasting | 1.27 | [-0.96, 3.50] | [-1.84, 4.38] | 0.264 | 0.558 | 3.18 / 4.06 |

An unadjusted interval can exclude zero while the Holm-adjusted p-value does not: the interval makes no allowance for the other seven tests, the adjusted p-value does. That is the case for the CL contemporaneous result using the NoLBERT definition (unadjusted [0.11, 2.50], raw p 0.033, Holm p 0.229; its Bonferroni interval is [-0.37, 2.98]). (In the earlier CL/GC-only report the family had six tests and this Holm p was 0.163; adding the two Bitcoin tests enlarges the family.) The last column is the smallest true effect each test could detect with 80% power. Every estimate in the table is smaller than its minimum detectable effect, so these tests cannot rule out modest effects.

## 3. Development, validation and test compared (own-window estimates, 95% CI unadjusted)

| asset | narrative | analysis | development | validation | test | same sign in all three |
|---|---|---|---|---|---|---|
| CL | oil (NoLBERT-60 t28+t30) | contemporaneous | 0.60 [0.01, 1.18] (n=258) | 0.40 [-0.70, 1.50] (n=105) | 1.30 [0.11, 2.50] (n=149) | yes |
| CL | oil (FinBERT Energy/Oil) | contemporaneous | 0.99 [0.39, 1.60] (n=258) | 0.11 [-1.62, 1.84] (n=105) | 0.78 [-0.14, 1.69] (n=149) | yes |
| CL | oil (NoLBERT-60 t28+t30) | forecasting | 0.13 [-0.40, 0.65] (n=258) | -1.27 [-3.85, 1.30] (n=105) | -1.29 [-2.31, -0.27] (n=147) | no |
| CL | oil (FinBERT Energy/Oil) | forecasting | -0.00 [-0.48, 0.48] (n=258) | -1.87 [-5.19, 1.44] (n=105) | -0.42 [-1.04, 0.20] (n=147) | yes |
| GC | gold (FinBERT Gold/Metals) | contemporaneous | 0.02 [-0.19, 0.22] (n=258) | -0.02 [-0.38, 0.34] (n=105) | -0.17 [-0.36, 0.02] (n=149) | no |
| GC | gold (FinBERT Gold/Metals) | forecasting | 0.06 [-0.13, 0.26] (n=258) | 0.21 [-0.13, 0.55] (n=105) | -0.01 [-0.43, 0.41] (n=147) | no |
| BTC | Fed (FinBERT Fed/Central Banks) | contemporaneous | -0.38 [-2.61, 1.85] (n=104) | -1.42 [-4.27, 1.42] (n=105) | -1.16 [-2.49, 0.16] (n=149) | yes |
| BTC | Fed (FinBERT Fed/Central Banks) | forecasting | 0.16 [-1.78, 2.10] (n=103) | -0.73 [-2.71, 1.26] (n=105) | 1.27 [-0.96, 3.50] (n=147) | no |

## 4. Contemporaneous versus predictive: out-of-sample usefulness (frozen coefficients, test window)

Coefficients are fitted on development + validation and applied unchanged to the test weeks. Gain is the out-of-sample R-squared relative to the price-controls-only model, with a block-bootstrap 95% interval (unadjusted).

| asset | narrative | analysis | gain vs controls-only [95% CI] | R2 vs expanding mean (full / controls-only) | hit rate (full / controls-only) |
|---|---|---|---|---|---|
| CL | oil (NoLBERT-60 t28+t30) | contemporaneous | 0.019 [-0.006, 0.045] | n/a | n/a |
| CL | oil (FinBERT Energy/Oil) | contemporaneous | 0.014 [-0.035, 0.069] | n/a | n/a |
| CL | oil (NoLBERT-60 t28+t30) | forecasting | 0.015 [0.002, 0.030] | -0.058 / -0.074 | 0.50 / 0.50 |
| CL | oil (FinBERT Energy/Oil) | forecasting | 0.009 [-0.013, 0.034] | -0.064 / -0.074 | 0.54 / 0.50 |
| GC | gold (FinBERT Gold/Metals) | contemporaneous | -0.001 [-0.003, 0.000] | n/a | n/a |
| GC | gold (FinBERT Gold/Metals) | forecasting | -0.004 [-0.029, 0.020] | -0.004 / 0.000 | 0.47 / 0.46 |
| BTC | Fed (FinBERT Fed/Central Banks) | contemporaneous | 0.007 [-0.009, 0.023] | n/a | n/a |
| BTC | Fed (FinBERT Fed/Central Banks) | forecasting | -0.002 [-0.004, 0.001] | -0.015 / -0.014 | 0.46 / 0.48 |

## 5. Secondary narratives and controls (test window, own-window, unadjusted; for context only, not tests of the hypotheses)

| asset | narrative | role | analysis | coef | raw p | development coef | validation coef |
|---|---|---|---|---|---|---|---|
| CL | gold (FinBERT Gold/Metals) | cross-narrative control | contemporaneous | -0.54 | 0.063 | -0.03 | 0.26 |
| CL | gold (FinBERT Gold/Metals) | cross-narrative control | forecasting | -0.06 | 0.850 | 0.28 | 0.64 |
| GC | Fed (FinBERT Fed/Central Banks) | secondary | contemporaneous | -0.15 | 0.518 | -0.01 | 0.04 |
| GC | oil (FinBERT Energy/Oil) | cross-narrative control | contemporaneous | 0.24 | 0.076 | 0.13 | -0.07 |
| GC | Fed (FinBERT Fed/Central Banks) | secondary | forecasting | 0.33 | 0.125 | -0.07 | 0.05 |
| GC | oil (FinBERT Energy/Oil) | cross-narrative control | forecasting | -0.02 | 0.912 | -0.13 | -0.27 |
| BTC | Macro (FinBERT) | secondary | contemporaneous | 0.17 | 0.845 | -2.80 | -0.63 |
| BTC | Currencies (FinBERT) | secondary | contemporaneous | 2.19 | 0.270 | 1.12 | 1.59 |
| BTC | oil (FinBERT Energy/Oil) | cross-narrative control | contemporaneous | 0.47 | 0.453 | -0.30 | -1.40 |
| BTC | Macro (FinBERT) | secondary | forecasting | 0.91 | 0.261 | 0.76 | -1.08 |
| BTC | Currencies (FinBERT) | secondary | forecasting | 0.98 | 0.381 | -0.39 | -1.25 |
| BTC | oil (FinBERT Energy/Oil) | cross-narrative control | forecasting | -0.39 | 0.614 | -0.32 | 1.50 |

Secondary results show why choosing a narrative after the fact is unsafe: Bitcoin's Macro class has a development coefficient of -2.80 (p < 0.001) that is -0.63 in validation and +0.17 in test; the Currencies class is significant in validation and not in test. Neither was promoted to a primary result.

## 6. Robustness

- **Databento degraded days** (14 days): dropping the weeks that contain them changes raw p-values. For CL with the NoLBERT definition the contemporaneous raw p moves from 0.033 to 0.060, and the forecasting raw p from 0.013 to 0.003, which would pass the eight-test Holm adjustment (adjusted 0.020). That sensitivity was specified in advance as a check and not as a selection device, and this coefficient (-1.5) has the opposite sign of its development estimate (+0.13), is not stable across windows, and weakens to p = 0.50 when t55 is added to the oil definition. It is reported, not relied on. No other primary test passes adjustment in this sensitivity.
- **Adding NoLBERT topic t55 to the oil definition** moves the CL forecasting coefficient from -1.29 (p 0.013) to -0.32 (p 0.50) and the contemporaneous one from 1.31 to 0.94 (p 0.09): the oil result depends on the topic definition.
- **Cross-narrative controls** are as significant as some primary results: CL on the gold narrative has raw p 0.063 and GC on the oil narrative 0.076 in the test window. Effects with p-values of that size occur by chance in this setup and should not be read as evidence.
- **Timing checks** (regress this week's return on next week's shock): near zero for CL and the Bitcoin Fed shock; not zero for gold (+0.39, p 0.001) and for the Bitcoin Currencies class. The post-hoc diagnostic (`posthoc_gold.md`) finds that weekly shocks are mechanically negatively autocorrelated, and that gold attention follows gold returns in 2022-2024 only (+0.10 sd per percentage point), not in 2015-2021. It is exploratory and leaves the primary gold results unchanged.
- **Futures construction** was checked against GLD (weekly correlation 0.992, slope 1.001) and, for crude, against EIA WTI spot (weekly correlation 0.96 outside March-July 2020, 0.99 in the test window; the annual futures-minus-spot gap follows the shape of the curve; April 2020 handled by having rolled to the June contract before the May contract went negative). Details: `futures_validation.md`, `wti_validation.md`.

## 7. What can and cannot be concluded

- **Supported (weakly):** oil attention and CL returns move together in the same week with the same sign in all three windows and under both definitions, with a small gain in explained variance. It does not survive the multiple-testing adjustment, depends on the topic definition, and cannot be interpreted causally: news coverage and prices may both respond to the same supply and demand developments, or news may follow price.
- **Not supported:** forecasting. No narrative-augmented model beats the expanding historical mean in the test window (out-of-sample R-squared against the mean: CL -5.8% and -6.4%, GC -0.4%, Bitcoin -1.5%). The CL NoLBERT oil narrative does improve on the price-controls-only model by 1.5% (interval [0.2%, 3.0%]), but both are worse than the mean, and its coefficient has the opposite sign in development.
- **Not supported:** a gold or Bitcoin contemporaneous relation. Bitcoin's Fed shock has a negative contemporaneous coefficient of similar size in validation and test (-1.4, -1.2), but with intervals that include zero.
- **No trading claim is made.** No strategy was built or tested on these signals.

## 8. Limitations

- Power: 104-149 weeks; the minimum detectable effects in section 2 are large relative to the estimates.
- Narratives: only oil has a well-supported topic (NoLBERT-60 t28 and t30; FinBERT Energy/Oil). The Fed class is 0.4% of articles and noisy; the Macro class does not match its label; NoLBERT-20 has no macro topic; inflation, growth and China have none. Bitcoin has no crypto topic.
- Look-ahead in the text models: topics were fit through 2019-12-31, but the NoLBERT encoder was adapted on text through 2020-08-31 (reported by the user), overlapping the validation window, and the FinBERT classifier's training data are undocumented.
- News data have calendar dates but no times; the forecasting signal skips the last trading day of the week to avoid overlap with the return window. News coverage per stock falls about 40% from 2024 and January 2024 is nearly empty (those weeks are dropped).
- Topic definitions were chosen from topic terms and headlines before returns were examined; the choice of t28 and t30 (and exclusion of t55) rests on those terms.
- Shocks measure attention and have a mechanical negative autocorrelation (-0.3 to -0.5 at one week).
- Bitcoin's development window is about 100 weeks; its results are exploratory.
- The sparse IPCA factors, the loading matrix and the original code were not available, so nothing here says whether the equity-derived factors transfer across assets.

## 9. Files

`pilot_prereg.md`, `btc_prereg.md` (specifications); `pilot.py`, `pilot_btc.py`, `build_weekly_panel.py` (code); `pilot_results.md/.csv`, `btc_results.md/.csv`, `cross_asset_primary.csv` (results); `posthoc_gold.md` (post-hoc); `futures_validation.md`, `wti_validation.md` (data checks); `cross_asset_coefficients.png`.