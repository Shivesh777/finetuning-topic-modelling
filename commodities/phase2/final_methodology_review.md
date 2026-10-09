# Final methodological and reproducibility review

Scope: a review of what was already run. No new model, specification, asset or strategy was tried. The only computation added is a calibration check of the already-frozen Clark-West statistic (`ipca_cw_calibration.py`), which reproduces the frozen gains, statistics and p-values exactly.

## 1. Is the Clark-West test appropriate for the contemporaneous explanatory comparison?

**Short answer: usable as an approximate descriptive test, with caveats, and it should not be the only inference reported.**

- Clark and West (2007) designed the statistic for comparing the forecasts of nested models, adjusting the mean squared error difference for the estimation noise that the extra regressors add under the null. That adjustment depends on the nesting and on parameters being estimated on earlier data, not on whether the regressors are lagged, so its logic carries over to our fixed-scheme comparison in which the factor model is fitted on earlier months and applied to the test months with the realised same-month factors. What does not carry over is the interpretation: the result is a statement about explaining returns, not forecasting them.
- Its normal approximation is justified when the test sample is small relative to the estimation sample. Here the test sample is 21 months and the fit samples are 59 (crude, gold, copper) or 28 months (Bitcoin and the common-month analysis), so that condition is not comfortably met, and the months are not contiguous.
- Check: under the null that the factors add nothing (wild-bootstrap returns built from the fitted benchmark, observed market and factor values held fixed, 5,000 draws, seed 0, the full fit-and-test procedure repeated), the analytic one-sided p-value at a nominal 5% rejects 4.2% to 8.4% of the time across the eight asset and sample combinations (nominal 2.5%: 1.5% to 4.5%). It is not badly miscalibrated, but it is somewhat liberal in several cases and conservative for crude.

Primary held-out family (four assets, primary model, asset-specific fit sample), three ways of computing the inference:

| asset | gain | analytic Clark-West p (pre-specified) | Holm (4) | calibrated Clark-West p | Holm (4) | calibrated p of the gain itself (post hoc) | Holm (4) |
|---|---|---|---|---|---|---|---|
| WTI crude | 0.309 | 0.026 | 0.103 | 0.017 | 0.066 | 0.006 | 0.022 |
| Gold | -0.093 | 0.509 | 0.509 | 0.558 | 0.558 | 0.656 | 0.656 |
| Copper | 0.127 | 0.036 | 0.108 | 0.045 | 0.134 | 0.044 | 0.133 |
| Bitcoin | 0.039 | 0.095 | 0.189 | 0.130 | 0.259 | 0.168 | 0.336 |

Same, common-month fit sample (28 months for every asset):

| asset | gain | analytic p | Holm (4) | calibrated Clark-West p | Holm (4) | calibrated p of the gain | Holm (4) |
|---|---|---|---|---|---|---|---|
| WTI crude | 0.234 | 0.055 | 0.221 | 0.062 | 0.249 | 0.041 | 0.162 |
| Gold | -0.050 | 0.278 | 0.278 | 0.382 | 0.382 | 0.438 | 0.438 |
| Copper | 0.118 | 0.061 | 0.221 | 0.085 | 0.256 | 0.076 | 0.227 |
| Bitcoin | 0.039 | 0.095 | 0.221 | 0.123 | 0.256 | 0.164 | 0.329 |

**Material finding.** The conclusion that no pre-specified held-out test survives correction holds for the pre-specified analytic Clark-West statistic (smallest Holm p 0.103) and for the simulation-calibrated Clark-West statistic (0.066). A post hoc test that uses the gain itself as the statistic puts the Holm-adjusted p-value for crude oil at 0.022. The crude-oil result is therefore **borderline, and the verdict depends on the choice of statistic and reference distribution**; it cannot be called clearly null or clearly established. The paper reports all three and treats the pre-specified one as primary. Other facts that keep us from calling it robust: 21 test months; the interval for the gain reaches zero; the common-month version is weaker (Holm 0.162 even for the gain-based version); the NoLBERT factor does not reproduce it; and the FinBERT representation is not point-in-time.

## 2. The pre-registered joint coefficient tests are reported separately from the held-out squared-error comparison

The joint tests below are the original specification's four primary tests (primary model, M2, factor coefficients jointly zero, estimated **on the 21 test months themselves**). After review they are **descriptive**: coefficients are re-estimated inside the test sample, so they are not held-out evidence. The held-out squared-error comparison of section 1 is the confirmatory evidence. The two answer different questions and must not be combined or interchanged.

| asset | n (test months) | joint Wald p, HC3 | joint p, calendar-aware Newey-West | Holm (4), HC3 |
|---|---|---|---|---|
| WTI crude | 21 | 0.443 | 0.209 | 1.000 |
| Gold | 21 | 0.550 | 0.344 | 1.000 |
| Copper | 21 | 0.719 | 0.306 | 1.000 |
| Bitcoin | 21 | 0.234 | 0.031 | 0.938 |

None is significant. The calendar-aware Newey-West p-value for Bitcoin (0.031) is small unadjusted but the test is descriptive, it uses 21 observations with five parameters, and it does not survive the Holm adjustment across four assets (HC3-based).

## 3. Leakage and information-restriction audit of the IPCA factors and their inputs

Status: **verified** against saved files, **documented** in the team's code snapshot (`~/Downloads/nolbert-ftm-runnable`, which predates the `_s63` option, so it documents intent, not the exact run), **assumed**, or **issue**.

| item | status | detail |
|---|---|---|
| Factor estimation schedule | documented and consistent with saved metadata | Every saved `summary.json` states: grid fitted on training periods, chosen on validation MSE, refitted once on training + validation, applied unchanged to the test periods. The code documents that tuning never sees test data and that validation forecasts come from a training-only fit. Saved `factors.csv` marks train and validation factor months in-sample and test months out-of-sample; the team's own reproduction of the saved test forecasts agrees to 1e-9 (29 of 50 specifications independently reproduced here). |
| Factor returns computed with frozen loadings | verified (consistency) | Factors recomputed from `gamma.csv` and the panel match the saved ones (correlation above 0.999 in four specifications checked). This shows consistency, not that test returns were unused in fitting; that rests on the documented schedule. |
| Feature scaling | verified | All characteristics are rank-normalised across stocks within each date: no statistic from later dates enters. |
| Characteristic windows | verified on a sample | Covariance exposures use a window ending at the formation date (checked against a rebuild from daily attention and returns; rank correlation 0.92, falling to 0.72 and 0.41 when shifted 21 and 63 days forward). |
| Topic models, vocabulary | documented, verified from metadata | Fitted on training-period articles only (through 2019-12-31); vocabulary built on training-period text. |
| NoLBERT representation | point-in-time pretraining; adaptation window unresolved | Pretrained only on 1976-1995 text. Adapted by headline-lead contrastive matching on training-period news, no returns, per the code documentation. Our project notes give an adaptation window of January 2015 to August 2020, which conflicts with training-period only (ends 2019); not reconciled. |
| FinBERT representation | issue: not point-in-time | Class probabilities from a classifier published in December 2022 and fine-tuned on finance tweets released in September 2022 (the team's own label: not point-in-time). The primary IPCA model and two of the three validation-selected equity specifications use it. |
| Shock window | issue: configuration mismatch | `_s63` is a 63-day attention-shock window chosen on validation for monthly rebalancing. All monthly analyses here, including the saved monthly stock forecasts, use the default 5-day window. The 63-day factor files exist but were not analysed. |
| News sources and sample | documented | FNSPID (Nasdaq articles through 2023) and FinMultiTime (article list; 2024 text); 1,255,289 unique articles from 2012 to 2024; one day with a data dump (2023-12-16) removed; 22 articles in January 2024; about 4,500 articles of 2024 have headlines only. The 2024 part of the test window uses a different source. |
| Calendar alignment | verified | Holding window = following calendar month; matches compounded stock returns for all 104 months; windows of consecutive months do not overlap. |
| Stock universe | verified, one issue | 98.7% of monthly rows are in the list formed at the latest June 30; about 1.1% belong only to the next June's list (small look-ahead in membership, not quantified). |
| In-sample versus out-of-sample factor months | design note | Coefficients of asset returns on factors are fitted on months whose factors are in-sample with respect to the loadings, and applied to test months whose factors are out-of-sample; this works against finding a held-out gain. |
| Months without a futures return | documented | Futures data start in 2015 (Bitcoin December 2017); 24 early training months have no commodity return; no extension was downloaded. |
| Intraday ordering of news | cannot be verified | Articles carry a date but no time of day; the forecasting design skips the last trading day of the week. |

## 4. Consistency of numbers, intervals, sample sizes and corrections with the frozen results

- Every number in the paper is inserted by `research/paper/build_paper.py` from the result files (`findings.json`, `robustness_summary.csv`, `cross_asset_primary.csv`, `pilot_results.csv`, `btc_results.csv`, `ipca_oos.csv`, `ipca_results.csv`, `ipca_coefficients.csv`, `ipca_cw_calibration.csv`); the build fails if a placeholder is left unfilled.
- Multiple-testing families are stated where used: eight primary weekly topic-shock tests (Holm); four primary held-out IPCA tests (Holm); 12 primary IPCA coefficients (Bonferroni intervals, descriptive).
- The Holm p-value of the CL contemporaneous topic-shock result is 0.229 with eight tests (earlier 0.163 with six); the paper uses 0.229.

## 5. Phase 1 and Phase 2 compared fairly

They differ in universe, frequency, target, evaluation and inference (a Sharpe ratio of a traded portfolio versus a squared-error gain of an explanatory regression), and in whether the information precedes the return. The paper compares what kind of claim each supports and does not rank them.

## 6. Issues that need correction or disclosure before submission

1. Report the crude-oil result as borderline with all three inference versions (section 1), not as simply non-significant.
2. Disclose that FinBERT is not point-in-time and that the only model showing the crude effect uses it.
3. Disclose the 5-day versus 63-day mismatch; do not describe our monthly models as the team's validated monthly configuration.
4. Resolve or disclose the conflicting encoder adaptation windows.
5. Keep the descriptive joint tests and the held-out comparison in separate tables.
6. Disclose the 2024 news-source change and the partly headline-only 2024 text.

---

# Addendum (after the post hoc robustness round)

Specification of the robustness round: `ipca_robustness_prereg.md` (SHA-256 `ca3697bb...`); results: `ipca_robustness_results.md`; reconstruction: `reconstruction_test.md`. No original result was changed. Everything in this addendum is post hoc: the 21-month holdout had already been examined.

## A1. 63-day reconstruction checks

- The `_s63` feature blocks are not saved, so the project's feature construction was re-implemented from saved inputs (daily attention sums, CRSP daily returns, Fama-French market). It reproduces the saved 5-day feature blocks with a minimum per-date rank correlation of 1.0000 on the 99.0% of rows with complete exposures (the rest are filled by random donor rows in the project's pipeline and cannot be reproduced exactly).
- With the window set to 63 days, the reconstruction reproduces the saved 63-day factor returns with correlation at least 0.9979 (test months at least 0.9934) for all three eligible specifications. This verifies the 63-day factors through their returns, not the unsaved feature blocks themselves.
- Information restriction, verified in code: shocks are attention minus a strictly past trailing mean (`rolling(window).mean().shift(1)`), and exposures for a portfolio formed at the close of day f use news-return pairs up to day f - 1 (`NEWS_LAG_DAYS = 1`), because article dates carry no time of day.
- The meaning of `_s63` (a 63-day window chosen on validation for monthly rebalancing) comes from a project member's message; the code snapshot predates the option.

## A2. Post hoc robustness findings

| specification | crude gain [95% CI] | Holm (4) | Holm (12) analytic / calibrated |
|---|---|---|---|
| FinBERT-20, 5-day, K=3 (original primary) | 0.31 [-0.02, 0.53] | 0.103 | n/a |
| FinBERT-20, 63-day, K=3 | 0.25 [0.04, 0.46] | 0.018 | 0.054 / 0.043 |
| NoLBERT-20, 63-day, K=3 | -0.16 [-0.50, 0.23] | 1.000 | 1.000 / 1.000 |
| LDA-60, 63-day, K=3 | 0.10 [-0.15, 0.35] | 0.155 | 0.386 / 0.432 |

- 2024 exclusion (original primary, frozen coefficients, 14 test months): crude gain 0.33 [-0.06, 0.59], Holm 0.147 (analytic), 0.114 (calibrated). The crude result does not depend on the affected months; power is lower.
- NoLBERT at 63 days shows a Bitcoin gain of 0.31 [0.01, 0.48], Holm (4) 0.050, Holm (12) 0.136; its loadings have no crypto content and it is not interpreted.
- Gains for copper, gold and the equity market are small or negative in all three 63-day specifications.

## A3. FinBERT versus NoLBERT

At K = 3 and the 63-day window the two 20-topic representations are directly comparable, which removes the K confound of the original 5-day comparison (K = 3 versus K = 1). The crude-oil link appears with FinBERT in both windows and not with NoLBERT in either; LDA shows a small gain that is not significant after adjustment. The result is therefore **representation-dependent**. Whether that reflects better class definitions in FinBERT or information the classifier acquired after much of the sample cannot be distinguished with these data. Under FinBERT-63, as under FinBERT-5, the factor most associated with crude oil is not the one that loads on the Energy/Oil class.

## A4. Remaining timing limitations

| item | status | consequence |
|---|---|---|
| FinBERT classifier | not point-in-time (published December 2022; tweets of September 2022) | every FinBERT-based result, including the primary model, is not a clean historical test |
| NoLBERT encoder adaptation | **resolved from project documentation**: pairs drawn up to 2020-09-01 (the training end of an earlier pipeline), so January-August 2020 validation text entered the adaptation; text only, no returns | validation-period numbers that use NoLBERT features are affected; the 2022-2024 test period is not. The training log itself is not in the local files, so this rests on the project's summary. Clean fix: re-adapt with the 2019-12-31 cutoff (not done) |
| Vocabulary basis | the 10,000-term basis comes from the whole corpus for every representation (fitting, centroids and scaling are training-only) | small look-ahead in LDA inputs and topic descriptions; not in NoLBERT clustering. LDA is therefore historically more compatible, not strictly point-in-time |
| Feb-Jun holding months | documented: from 2017 the daily return file holds only each year's June-list stocks, so earlier formation days are left out | removes a look-ahead; the test sample is 21 of 36 months and speaks only to holding months January and July-December |
| 63-day stock forecasts | not rebuilt | Phase 1 monthly results remain on the 5-day window |

## A5. Status of the issues listed in section 6

1. Crude described as borderline with all inference versions: done; now also described as representation-dependent.
2. FinBERT not point-in-time: disclosed in abstract, introduction, data, results table and limitations.
3. 5-day versus 63-day: disclosed; post hoc 63-day factor analysis done; stock forecasts not rebuilt.
4. Encoder window: resolved from project documentation (see A4).
5. Joint tests separate from the held-out comparison: done (separate table, labelled descriptive).
6. 2024 source change: disclosed; descriptive exclusion check done.