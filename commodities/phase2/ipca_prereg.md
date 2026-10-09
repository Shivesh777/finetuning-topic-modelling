# IPCA factor exposures of CL, GC, HG, BTC: specification fixed before any regression is run

Status: **not yet executed.** This file is hashed when the regressions run. It is a new, complementary experiment; the topic-shock pilot
(`pilot_prereg.md`, `btc_prereg.md`) and all of its results are unchanged. No model search, no trading rule.

## Question
Do the systematic factors the team extracted from equity returns and news-derived stock characteristics (Sparse IPCA) capture common
risk in crude oil, gold, copper and Bitcoin futures? This is an **explanatory** test: the factor returns are estimated from same-month
stock returns, so a relationship with same-month futures returns is not a forecasting result.

## Model selection (rule fixed before looking at any commodity or Bitcoin return)
Rule for the **primary** model: among the IPCA specifications in `data/results/ipca/M/` choose the one that is (1) narrative-only, i.e. its
feature block begins with `narr_` (no price, no firm-news, no tone features); (2) K = 3 factors, the number of factors the original paper uses
(K is the team's choice from its grid by validation mean squared error, not mine); (3) not an `_s63` variant, because the meaning of `_s63`
is not documented anywhere in the folder (verified by search; see below). The rule uses no return, Sharpe or commodity information.

| specification | K | penalty | kept / total features | _s63 | meets rule |
|---|---|---|---|---|---|
| `narr_finbert__sipca` | 3 | 0.1 | 20/20 | no | **yes** |
| `narr_finbert_s63__sipca` | 3 | 0.03 | 20/20 | yes | no |
| `narr_lda_all_s63__sipca` | 3 | 0.3 | 60/60 | yes | no |
| `narr_minilm_s63__sipca` | 3 | 0.1 | 60/60 | yes | no |
| `narr_nolbert20_s63__sipca` | 3 | 0.03 | 20/20 | yes | no |
| `narr_lda__sipca` | 1 | 0.3 | 45/60 | no | no |
| `narr_lda_all__sipca` | 1 | 0.1 | 57/60 | no | no |
| `narr_lda_s63__sipca` | 1 | 0.3 | 57/60 | yes | no |
| `narr_minilm__sipca` | 1 | 0.3 | 50/60 | no | no |
| `narr_nolbert20__sipca` | 1 | 0.3 | 14/20 | no | no |
| `narr_nolbert__sipca` | 1 | 0.03 | 58/60 | no | no |
| `narr_nolbert_s63__sipca` | 1 | 0.3 | 50/60 | yes | no |

Exactly one specification meets the rule, so there is no discretion: **primary = `narr_finbert__sipca`** (FinBERT-20 classes; K = 3).
Note that it is not NoLBERT-based. **Secondary = `narr_nolbert20__sipca`** (K = 1, 14 of 20 topics kept), as instructed. Nothing else is analysed.

`_s63`: appears only as a suffix of specification names and of their feature-block names (`narr_nolbert20_s63`, ...). The 17 `_s63` specifications
use feature blocks that do not exist in `data/panels/`, different K and penalty from their base counterparts, and identical dates; no file in
the folder documents the meaning. Candidates such as a seed, a 63-day window, or a different characteristic construction cannot be tested
from what is saved. They are excluded until the team confirms what they are.

## Data and alignment
- Factors: `factors.csv` of the two specifications (104 monthly factor returns: 69 train, 14 validation, 21 test; train and validation are in-sample for
  the loadings, test is out-of-sample with the loadings frozen; the `estimate` column marks this).
- A factor dated at formation date d is the return over the first trading day after d to the last trading day of the **following calendar month**
  (not to the next panel date: from 2017 the panel has gaps of several months). This was verified against compounded CRSP stock returns for all
  104 months (`ipca_alignment_check.csv`; median absolute difference 1e-7).
- Futures: monthly compounded daily returns of CL, GC, HG, BTC on trade dates in (d, month-end], at least 15 days per month
  (`ipca_monthly_panel.csv`, from `build_ipca_monthly.py`). Equity market: Fama-French `mktrf` compounded over the same dates.
- Months available (futures data start in 2015; Bitcoin in December 2017): CL, GC, HG: 45 train + 14 validation + 21 test = 80;
  BTC: 14 + 14 + 21 = 49; equity market: 104. The 24 training months of 2013-2014 have no futures data (an extension costs about $1.28 and would only add descriptive months).

## Regressions (months are separated by gaps, so no time-series dependence structure is assumed beyond neighbouring months)
For each asset a (excess monthly return; futures price returns are already excess returns), with f the factor vector (3 or 1) and m the equity market excess return:
- M0 (benchmark): r = a + c m + e
- M1 (factors only): r = a + b'f + e
- M2 (benchmark + factors): r = a + c m + b'f + e

Equity market row: m = a + b'f + e (no separate benchmark).

Reported for each asset and model: coefficients b, 95% confidence intervals, adjusted R-squared of M0, M1, M2, the incremental adjusted R-squared of M2 over M0,
and the joint test b = 0. Standard errors: heteroskedasticity-robust HC3 (primary); Newey-West with 1 lag on the time-ordered sample as a sensitivity.
Factor coefficients are also given standardised (b x sd(f) / sd(r)) so that assets are comparable.

## Samples and what each means
1. **All available months: descriptive.** Includes months the loadings were fitted on; not out-of-sample.
2. **Train / validation / test separately: descriptive.** Same models in each block.
3. **Held-out evaluation, test months only (21 months for every asset, including BTC):**
   (a) own-window estimates of M2 on the 21 test months, with HC3 intervals and the joint Wald test; (b) frozen out-of-sample explanatory gain: fit M0 and M2 on
   all earlier months with a return (80 minus 21 = 59 for CL/GC/HG; 28 for BTC), apply to the 21 test months with the contemporaneous realised factors, report
   1 - SSE(M2)/SSE(M0) with a 95% interval from a paired bootstrap of the 21 months (5,000 draws, seed 0). This is **contemporaneous explanation**, not forecasting.
4. **Primary confirmatory statistics:** the primary model's joint Wald p-values in (3a) for CL, GC, HG and BTC (four tests), reported raw and Holm-adjusted; confidence
   intervals for the 12 primary-model coefficients unadjusted and Bonferroni-adjusted (12 tests). Everything else (secondary model, all-months, blocks) is descriptive.
5. The minimum detectable incremental R-squared at 80% power (noncentral F, 21 observations; 80 and 49 for the larger samples) is reported with every result.

## Comparison table (equities, CL, GC, HG, BTC)
One row per asset: months used, factor betas with intervals, adjusted R-squared of M0/M1/M2, incremental R-squared, held-out gain, **explanatory evidence**;
and a separate column of **forecasting evidence**: for equities the Phase 1 out-of-sample backtest of the same two specifications (test Sharpe after 0 and 10 bp, rank among
the 375 real specifications, from `robustness_summary.csv`, no new search); for CL, GC, BTC the weekly topic-shock forecasting results already reported (a different object, labelled as such);
HG: not tested.

## Limitations stated in advance
- The factors are estimated from same-period stock returns; a relationship with same-period commodity returns shows common exposure, not predictability or a priced risk premium.
- Held-out sample: 21 months; with three factors plus a benchmark, only large effects are detectable.
- All-months results mix in-sample months and cannot be called out-of-sample.
- HG's return series was checked for hygiene only, with no independent reference. Bitcoin has no Bitcoin-specific narrative and a short history.
- The primary model uses FinBERT-20 classes, whose training data are undocumented; NoLBERT's encoder window reported as Jan 2015 - Aug 2020 overlaps the validation block.
- Months are non-contiguous; HC3 does not model dependence between neighbouring months.

## Not done
Any specification other than the two named; the `_s63` variants; other K; price, firm-news or tone blocks; weekly or biweekly factors (not saved); lagged factors or forecasting
regressions; trading rules; any change after seeing results (changes would be listed below).

## Deviations
Logged before the first run, in response to review of the specification above (the version approved had SHA-256 `615fab7cec18384be6a5a8fc3f4cd6736705a2fd05807a5e4c9b9567311ed810`):

1. **The primary confirmatory held-out statistic is now the frozen-coefficient out-of-sample test**, not the own-window Wald test. Coefficients are fitted only on training + validation months; the 21 test months are never used to fit anything that is called out-of-sample. The held-out evidence is the out-of-sample explanatory gain of M2 over M0 (with a paired bootstrap interval) and a one-sided Clark-West (2007) test of equal squared error for these nested models, Holm-adjusted across the four assets (CL, GC, HG, BTC). Regressions re-estimated inside the 21 test months (including their Wald tests and the 12 coefficient intervals) are labelled **descriptive**.
2. **Calendar-aware Newey-West.** The sensitivity standard error lets two observations be dependent only if their formation months are adjacent calendar months (weight 0.5, Bartlett with a one-month bandwidth); observations further apart in calendar time are treated as independent, and rows that are adjacent in the file but months apart are not linked. HC3 remains the primary standard error.
3. **Identical observations** are used for M0, M1 and M2 within each asset and sample. A **common-month** analysis is added: the months in which CL, GC, HG and BTC all have returns (the Bitcoin months: 14 train, 14 validation, 21 test), with the frozen fit for that analysis using the 28 pre-test common months.
4. Also added: an automatic check that holding windows of consecutive factor months do not overlap, a statement of return conventions, and the minimum detectable incremental R-squared with every result (80% power, alpha 0.05 and 0.0125).

Nothing else changes: the two specifications, the alignment, the models M0 to M2, HC3, and the sample definitions are as above.
