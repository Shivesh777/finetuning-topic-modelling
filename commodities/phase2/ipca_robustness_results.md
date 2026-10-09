# Post hoc robustness of the IPCA cross-asset result (analyses A and B)

Specification: `ipca_robustness_prereg.md` (SHA-256 `ca3697bb1c9e7c7929547ee01ca9fe940366b0bf03aa3f9c19bc808e53fac8f1`). **Post hoc:** the 21 test months were already examined in the original analysis, so none of this is independent confirmation. The original 5-day FinBERT K=3 result remains primary; nothing here replaces it. All factor files cover the same 104 months, so eligible observations are identical across specifications (59 fit months for crude, gold and copper, 28 for Bitcoin and for common-month fits, 83 for the equity market, 21 test months; 14 in analysis B). Intervals are unadjusted 95% bootstrap intervals; Holm families are stated.

## A. Held-out gain over the equity market: original (5-day) and 63-day specifications side by side

Gain = 1 - SSE(factors + market) / SSE(market only) in the test months, fitted on training + validation months only. p: one-sided Clark-West, analytic and calibrated (simulated null). Holm (4): across the four assets within a specification. Holm (12): across all 12 asset-model tests of the three 63-day specifications (conservative post hoc family).


**asset-specific fit sample**

| specification | asset | n fit / test | gain [95% CI] | analytic p | Holm (4) | calibrated p | Holm (4) cal. | Holm (12) analytic | Holm (12) cal. |
|---|---|---|---|---|---|---|---|---|---|
| FinBERT-20, 5-day, K=3 (original primary) | WTI crude | 59 / 21 | 0.31 [-0.02, 0.53] | 0.026 | 0.103 | 0.017 | 0.066 |  |  |
| FinBERT-20, 5-day, K=3 (original primary) | Gold | 59 / 21 | -0.09 [-0.42, 0.16] | 0.509 | 0.509 | 0.558 | 0.558 |  |  |
| FinBERT-20, 5-day, K=3 (original primary) | Copper | 59 / 21 | 0.13 [-0.06, 0.32] | 0.036 | 0.108 | 0.045 | 0.134 |  |  |
| FinBERT-20, 5-day, K=3 (original primary) | Bitcoin | 28 / 21 | 0.04 [-0.19, 0.21] | 0.095 | 0.189 | 0.130 | 0.259 |  |  |
| FinBERT-20, 5-day, K=3 (original primary) | Equity market | 83 / 21 | -0.09 [-0.32, 0.10] | 0.646 |  |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | WTI crude | 59 / 21 | -0.27 [-0.68, 0.02] | 0.820 | 1.000 |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | Gold | 59 / 21 | -0.08 [-0.22, 0.01] | 0.880 | 1.000 |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | Copper | 59 / 21 | 0.12 [-0.05, 0.29] | 0.061 | 0.245 |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | Bitcoin | 28 / 21 | -0.06 [-0.54, 0.36] | 0.275 | 0.824 |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | Equity market | 83 / 21 | 0.05 [-0.10, 0.18] | 0.151 |  |  |  |  |  |
| NoLBERT-20, 63-day, K=3 (post hoc) | WTI crude | 59 / 21 | -0.16 [-0.50, 0.23] | 0.545 | 1.000 | 0.581 | 1.000 | 1.000 | 1.000 |
| NoLBERT-20, 63-day, K=3 (post hoc) | Gold | 59 / 21 | -0.07 [-0.26, 0.04] | 0.766 | 1.000 | 0.734 | 1.000 | 1.000 | 1.000 |
| NoLBERT-20, 63-day, K=3 (post hoc) | Copper | 59 / 21 | -0.07 [-0.35, 0.19] | 0.473 | 1.000 | 0.438 | 1.000 | 1.000 | 1.000 |
| NoLBERT-20, 63-day, K=3 (post hoc) | Bitcoin | 28 / 21 | 0.31 [0.01, 0.48] | 0.012 | 0.050 | 0.012 | 0.047 | 0.136 | 0.130 |
| NoLBERT-20, 63-day, K=3 (post hoc) | Equity market | 83 / 21 | -0.11 [-0.31, 0.08] | 0.745 |  |  |  |  |  |
| LDA-60, 63-day, K=3 (post hoc) | WTI crude | 59 / 21 | 0.10 [-0.15, 0.35] | 0.039 | 0.155 | 0.043 | 0.173 | 0.386 | 0.432 |
| LDA-60, 63-day, K=3 (post hoc) | Gold | 59 / 21 | 0.02 [-0.10, 0.15] | 0.239 | 0.615 | 0.245 | 0.703 | 1.000 | 1.000 |
| LDA-60, 63-day, K=3 (post hoc) | Copper | 59 / 21 | 0.03 [-0.16, 0.24] | 0.205 | 0.615 | 0.234 | 0.703 | 1.000 | 1.000 |
| LDA-60, 63-day, K=3 (post hoc) | Bitcoin | 28 / 21 | -0.14 [-0.68, 0.21] | 0.291 | 0.615 | 0.328 | 0.703 | 1.000 | 1.000 |
| LDA-60, 63-day, K=3 (post hoc) | Equity market | 83 / 21 | 0.00 [-0.24, 0.24] | 0.303 |  |  |  |  |  |
| FinBERT-20, 63-day, K=3 (post hoc) | WTI crude | 59 / 21 | 0.25 [0.04, 0.46] | 0.004 | 0.018 | 0.004 | 0.014 | 0.054 | 0.043 |
| FinBERT-20, 63-day, K=3 (post hoc) | Gold | 59 / 21 | -0.03 [-0.18, 0.09] | 0.566 | 0.566 | 0.588 | 0.588 | 1.000 | 1.000 |
| FinBERT-20, 63-day, K=3 (post hoc) | Copper | 59 / 21 | 0.08 [-0.16, 0.31] | 0.125 | 0.250 | 0.156 | 0.313 | 1.000 | 1.000 |
| FinBERT-20, 63-day, K=3 (post hoc) | Bitcoin | 28 / 21 | 0.09 [-0.09, 0.25] | 0.045 | 0.134 | 0.077 | 0.231 | 0.403 | 0.693 |
| FinBERT-20, 63-day, K=3 (post hoc) | Equity market | 83 / 21 | -0.15 [-0.44, 0.08] | 0.679 |  |  |  |  |  |

**common-month fit sample**

| specification | asset | n fit / test | gain [95% CI] | analytic p | Holm (4) | calibrated p | Holm (4) cal. | Holm (12) analytic | Holm (12) cal. |
|---|---|---|---|---|---|---|---|---|---|
| FinBERT-20, 5-day, K=3 (original primary) | WTI crude | 28 / 21 | 0.23 [-0.23, 0.53] | 0.055 | 0.221 | 0.062 | 0.249 |  |  |
| FinBERT-20, 5-day, K=3 (original primary) | Gold | 28 / 21 | -0.05 [-0.42, 0.18] | 0.278 | 0.278 | 0.382 | 0.382 |  |  |
| FinBERT-20, 5-day, K=3 (original primary) | Copper | 28 / 21 | 0.12 [-0.05, 0.30] | 0.061 | 0.221 | 0.085 | 0.256 |  |  |
| FinBERT-20, 5-day, K=3 (original primary) | Bitcoin | 28 / 21 | 0.04 [-0.22, 0.21] | 0.095 | 0.221 | 0.123 | 0.256 |  |  |
| FinBERT-20, 5-day, K=3 (original primary) | Equity market | 28 / 21 | -0.08 [-0.36, 0.17] | 0.454 |  |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | WTI crude | 28 / 21 | -0.19 [-0.35, -0.04] | 0.953 | 1.000 |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | Gold | 28 / 21 | -0.51 [-1.10, -0.10] | 0.950 | 1.000 |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | Copper | 28 / 21 | 0.06 [-0.09, 0.20] | 0.148 | 0.592 |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | Bitcoin | 28 / 21 | -0.06 [-0.54, 0.36] | 0.275 | 0.824 |  |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | Equity market | 28 / 21 | -0.01 [-0.04, 0.01] | 0.874 |  |  |  |  |  |
| NoLBERT-20, 63-day, K=3 (post hoc) | WTI crude | 28 / 21 | 0.01 [-0.20, 0.25] | 0.254 | 0.507 | 0.296 | 0.592 | 1.000 | 1.000 |
| NoLBERT-20, 63-day, K=3 (post hoc) | Gold | 28 / 21 | 0.05 [-0.39, 0.37] | 0.127 | 0.380 | 0.171 | 0.512 | 1.000 | 1.000 |
| NoLBERT-20, 63-day, K=3 (post hoc) | Copper | 28 / 21 | -0.04 [-0.17, 0.12] | 0.555 | 0.555 | 0.531 | 0.592 | 1.000 | 1.000 |
| NoLBERT-20, 63-day, K=3 (post hoc) | Bitcoin | 28 / 21 | 0.31 [0.01, 0.48] | 0.012 | 0.050 | 0.012 | 0.047 | 0.149 | 0.142 |
| NoLBERT-20, 63-day, K=3 (post hoc) | Equity market | 28 / 21 | -0.47 [-1.01, -0.16] | 0.986 |  |  |  |  |  |
| LDA-60, 63-day, K=3 (post hoc) | WTI crude | 28 / 21 | 0.02 [-0.21, 0.28] | 0.191 | 0.763 | 0.296 | 1.000 | 1.000 | 1.000 |
| LDA-60, 63-day, K=3 (post hoc) | Gold | 28 / 21 | -0.09 [-0.80, 0.36] | 0.232 | 0.763 | 0.275 | 1.000 | 1.000 | 1.000 |
| LDA-60, 63-day, K=3 (post hoc) | Copper | 28 / 21 | 0.02 [-0.14, 0.22] | 0.231 | 0.763 | 0.297 | 1.000 | 1.000 | 1.000 |
| LDA-60, 63-day, K=3 (post hoc) | Bitcoin | 28 / 21 | -0.14 [-0.68, 0.21] | 0.291 | 0.763 | 0.328 | 1.000 | 1.000 | 1.000 |
| LDA-60, 63-day, K=3 (post hoc) | Equity market | 28 / 21 | -0.02 [-0.37, 0.29] | 0.299 |  |  |  |  |  |
| FinBERT-20, 63-day, K=3 (post hoc) | WTI crude | 28 / 21 | 0.18 [-0.31, 0.54] | 0.018 | 0.073 | 0.028 | 0.113 | 0.201 | 0.310 |
| FinBERT-20, 63-day, K=3 (post hoc) | Gold | 28 / 21 | -0.14 [-0.49, 0.12] | 0.578 | 0.578 | 0.613 | 0.613 | 1.000 | 1.000 |
| FinBERT-20, 63-day, K=3 (post hoc) | Copper | 28 / 21 | 0.07 [-0.13, 0.27] | 0.178 | 0.355 | 0.244 | 0.488 | 1.000 | 1.000 |
| FinBERT-20, 63-day, K=3 (post hoc) | Bitcoin | 28 / 21 | 0.09 [-0.09, 0.25] | 0.045 | 0.134 | 0.077 | 0.231 | 0.447 | 0.770 |
| FinBERT-20, 63-day, K=3 (post hoc) | Equity market | 28 / 21 | -0.24 [-0.75, 0.10] | 0.566 |  |  |  |  |  |

## A (descriptive). Fit on all available months, and exposures of crude oil

Increase in adjusted R2 over the market benchmark and the joint test that the factor coefficients are zero (HC3). Descriptive: these months include those the loadings were fitted on.

| specification | asset | n | increment adj R2 | joint p | standardised betas of the factors |
|---|---|---|---|---|---|
| FinBERT-20, 5-day, K=3 | WTI crude | 80 | 0.20 | 0.000 | -0.20, -0.09, +0.46 |
| FinBERT-20, 5-day, K=3 | Gold | 80 | 0.01 | 0.176 | +0.06, +0.21, -0.08 |
| FinBERT-20, 5-day, K=3 | Copper | 80 | 0.04 | 0.060 | -0.20, +0.12, +0.15 |
| FinBERT-20, 5-day, K=3 | Bitcoin | 49 | 0.03 | 0.116 | +0.22, +0.01, -0.23 |
| FinBERT-20, 5-day, K=3 | Equity market | 104 | 0.06 | 0.102 | -0.08, -0.20, +0.20 |
| NoLBERT-20, 5-day, K=1 | WTI crude | 80 | 0.05 | 0.056 | +0.25 |
| NoLBERT-20, 5-day, K=1 | Gold | 80 | -0.01 | 0.813 | -0.03 |
| NoLBERT-20, 5-day, K=1 | Copper | 80 | 0.04 | 0.059 | +0.22 |
| NoLBERT-20, 5-day, K=1 | Bitcoin | 49 | 0.01 | 0.211 | +0.17 |
| NoLBERT-20, 5-day, K=1 | Equity market | 104 | 0.03 | 0.057 | +0.21 |
| NoLBERT-20, 63-day, K=3 | WTI crude | 80 | 0.10 | 0.022 | +0.07, +0.35, +0.12 |
| NoLBERT-20, 63-day, K=3 | Gold | 80 | -0.02 | 0.580 | +0.02, -0.13, +0.08 |
| NoLBERT-20, 63-day, K=3 | Copper | 80 | 0.01 | 0.467 | -0.09, +0.11, +0.15 |
| NoLBERT-20, 63-day, K=3 | Bitcoin | 49 | 0.21 | 0.012 | +0.11, +0.19, +0.46 |
| NoLBERT-20, 63-day, K=3 | Equity market | 104 | 0.05 | 0.035 | -0.11, +0.24, -0.06 |
| LDA-60, 63-day, K=3 | WTI crude | 80 | 0.12 | 0.007 | -0.33, -0.21, -0.15 |
| LDA-60, 63-day, K=3 | Gold | 80 | -0.00 | 0.562 | +0.10, +0.16, -0.06 |
| LDA-60, 63-day, K=3 | Copper | 80 | 0.01 | 0.314 | -0.16, -0.12, -0.08 |
| LDA-60, 63-day, K=3 | Bitcoin | 49 | 0.05 | 0.083 | -0.05, +0.26, -0.11 |
| LDA-60, 63-day, K=3 | Equity market | 104 | 0.10 | 0.007 | -0.25, -0.09, -0.24 |
| FinBERT-20, 63-day, K=3 | WTI crude | 80 | 0.13 | 0.001 | +0.06, -0.13, -0.40 |
| FinBERT-20, 63-day, K=3 | Gold | 80 | -0.02 | 0.610 | +0.05, -0.01, +0.14 |
| FinBERT-20, 63-day, K=3 | Copper | 80 | 0.02 | 0.291 | -0.03, -0.20, -0.09 |
| FinBERT-20, 63-day, K=3 | Bitcoin | 49 | 0.01 | 0.245 | +0.14, +0.19, +0.17 |
| FinBERT-20, 63-day, K=3 | Equity market | 104 | 0.08 | 0.021 | +0.12, -0.09, -0.28 |

**Largest loadings of each 63-day factor** (latent factors: signs and rotations are arbitrary)

- NoLBERT-20, 63-day, K=3: f1: t06: investors, buy, dividend (+0.55); t02: analyst blog, blog, analyst (+0.49); t19: analyst blog, blog, analyst (-0.33) | f2: t02: analyst blog, blog, analyst (+0.48); t07: market, dow, markets (+0.41); t16: market, news, dow (-0.40) | f3: t02: analyst blog, blog, analyst (-0.49); t15: conference, remarks, prepared remarks (+0.40); t11: average, moving average, moving (+0.37)
- LDA-60, 63-day, K=3: f1: t14: moving average, moving, day moving (-0.52); t05: surprise, earnings surprise, let things (-0.30); t40: offering, notes, ipo (-0.29) | f2: t16: dow, dow jones, jones (-0.28); t09: energy, oil, gas (+0.27); t51: rating, ibd, width float (-0.26) | f3: t11: earnings report, report, reacted late (+0.32); t25: options, contract, particular contract (-0.29); t48: featured, highlights, free report (-0.28)
- FinBERT-20, 63-day, K=3: f1: Energy | Oil (+0.43); General News | Opinion (-0.41); Company | Product News (+0.36) | f2: Currencies (+0.36); General News | Opinion (+0.36); IPO (-0.36) | f3: Analyst Update (+0.50); Financials (-0.39); General News | Opinion (-0.37)

## B. 2024 sensitivity (descriptive): original specifications, frozen coefficients, 14 test months

Test months with formation on or after 2023-12-01 are dropped (7 months); fit months and coefficients are unchanged. With 14 months every interval is wider and power lower; a null here is not evidence of no effect.

| specification | asset | gain, 21 months | gain, 14 months [95% CI] | analytic p | Holm (4) | calibrated p | Holm (4) cal. |
|---|---|---|---|---|---|---|---|
| FinBERT-20, 5-day, K=3 (original primary) | WTI crude | 0.31 | 0.33 [-0.06, 0.59] | 0.037 | 0.147 | 0.028 | 0.114 |
| FinBERT-20, 5-day, K=3 (original primary) | Gold | -0.09 | -0.05 [-0.42, 0.23] | 0.420 | 0.504 | 0.460 | 0.656 |
| FinBERT-20, 5-day, K=3 (original primary) | Copper | 0.13 | 0.14 [-0.08, 0.43] | 0.056 | 0.169 | 0.074 | 0.222 |
| FinBERT-20, 5-day, K=3 (original primary) | Bitcoin | 0.04 | -0.01 [-0.32, 0.22] | 0.252 | 0.504 | 0.328 | 0.656 |
| FinBERT-20, 5-day, K=3 (original primary) | Equity market | -0.09 | -0.09 [-0.33, 0.12] | 0.630 |  |  |  |
| NoLBERT-20, 5-day, K=1 (original secondary) | WTI crude | -0.27 | -0.32 [-0.77, -0.03] | 0.923 | 1.000 | 0.925 | 1.000 |
| NoLBERT-20, 5-day, K=1 (original secondary) | Gold | -0.08 | -0.09 [-0.27, 0.00] | 0.903 | 1.000 | 0.905 | 1.000 |
| NoLBERT-20, 5-day, K=1 (original secondary) | Copper | 0.12 | 0.10 [-0.10, 0.39] | 0.140 | 0.420 | 0.184 | 0.553 |
| NoLBERT-20, 5-day, K=1 (original secondary) | Bitcoin | -0.06 | 0.21 [-0.12, 0.57] | 0.093 | 0.370 | 0.039 | 0.158 |
| NoLBERT-20, 5-day, K=1 (original secondary) | Equity market | 0.05 | 0.08 [-0.07, 0.21] | 0.089 |  |  |  |