# IPCA factor exposures of crude oil, gold, copper, Bitcoin and the equity market

Specification: `ipca_prereg.md` (SHA-256 `b13b9f418a75487d0e61538b886d8729a55c376cd74417b13e2c6faec95729ff`, including its logged deviations). Primary model `narr_finbert__sipca` (K = 3), secondary `narr_nolbert20__sipca` (K = 1). All results are **contemporaneous explanation**: the factors are estimated from same-month stock returns. Each regression was run once.

## Conventions and checks

- Factor month: return over (formation date, last trading day of the following calendar month]. Holding windows of consecutive factor months: 0 overlap with the next formation date (0 expected). Months with at least one Databento degraded day: 4 of 104.
- Futures returns are price returns on a single contract at a time; the contract is changed (volume rule) on a day's close and the roll gap is never a return, so no roll return is included. A futures price return needs no capital and is already an excess return; no collateral return is added. The equity market is Fama-French `mktrf` (excess of the risk-free rate) over the same dates.
- Standard errors: HC3 with Student-t critical values (primary). The Newey-West sensitivity links only observations in adjacent calendar months. Intervals shown are 95%, unadjusted unless marked.
- M0, M1, M2 use identical observations within each asset and sample. 'Common-month' repeats the analysis on the months in which CL, GC, HG and BTC all have returns (the Bitcoin months).

## 1. Held-out evaluation: coefficients fitted on training + validation months only, applied to the 21 test months

Gain = 1 - SSE(M2)/SSE(M0) in the test months (for the equity market row, M0 is a constant). Interval: paired bootstrap of the 21 months (95%, unadjusted). Clark-West: one-sided test that M2 has lower squared error than M0, Holm-adjusted over CL, GC, HG, BTC. Contemporaneous factors are used, so this is explanation, not forecasting.


**primary (narr_finbert__sipca), asset-specific fit sample**

| asset | fit months | gain over benchmark [95% CI] | MSPE ratio M2/M0 | Clark-West t | one-sided p | Holm p (4 assets) |
|---|---|---|---|---|---|---|
| WTI crude (CL) | 59 | 0.309 [-0.022, 0.526] | 0.691 | 2.07 | 0.026 | 0.103 |
| Gold (GC) | 59 | -0.093 [-0.420, 0.159] | 1.093 | -0.02 | 0.509 | 0.509 |
| Copper (HG) | 59 | 0.127 [-0.063, 0.324] | 0.873 | 1.90 | 0.036 | 0.108 |
| Bitcoin (BTC) | 28 | 0.039 [-0.192, 0.207] | 0.961 | 1.36 | 0.095 | 0.189 |
| Equity market | 83 | -0.093 [-0.317, 0.105] | 1.093 | -0.38 | 0.646 | n/a |

**primary (narr_finbert__sipca), common-month fit sample**

| asset | fit months | gain over benchmark [95% CI] | MSPE ratio M2/M0 | Clark-West t | one-sided p | Holm p (4 assets) |
|---|---|---|---|---|---|---|
| WTI crude (CL) | 28 | 0.234 [-0.231, 0.533] | 0.766 | 1.67 | 0.055 | 0.221 |
| Gold (GC) | 28 | -0.050 [-0.418, 0.182] | 1.050 | 0.60 | 0.278 | 0.278 |
| Copper (HG) | 28 | 0.118 [-0.054, 0.302] | 0.882 | 1.62 | 0.061 | 0.221 |
| Bitcoin (BTC) | 28 | 0.039 [-0.218, 0.212] | 0.961 | 1.36 | 0.095 | 0.221 |
| Equity market | 28 | -0.083 [-0.362, 0.167] | 1.083 | 0.12 | 0.454 | n/a |

**secondary (narr_nolbert20__sipca), asset-specific fit sample**

| asset | fit months | gain over benchmark [95% CI] | MSPE ratio M2/M0 | Clark-West t | one-sided p | Holm p (4 assets) |
|---|---|---|---|---|---|---|
| WTI crude (CL) | 59 | -0.274 [-0.680, 0.020] | 1.274 | -0.94 | 0.820 | 1.000 |
| Gold (GC) | 59 | -0.081 [-0.224, 0.015] | 1.081 | -1.21 | 0.880 | 1.000 |
| Copper (HG) | 59 | 0.122 [-0.045, 0.294] | 0.878 | 1.61 | 0.061 | 0.245 |
| Bitcoin (BTC) | 28 | -0.057 [-0.541, 0.361] | 1.057 | 0.61 | 0.275 | 0.824 |
| Equity market | 83 | 0.048 [-0.101, 0.180] | 0.952 | 1.06 | 0.151 | n/a |

**secondary (narr_nolbert20__sipca), common-month fit sample**

| asset | fit months | gain over benchmark [95% CI] | MSPE ratio M2/M0 | Clark-West t | one-sided p | Holm p (4 assets) |
|---|---|---|---|---|---|---|
| WTI crude (CL) | 28 | -0.194 [-0.355, -0.041] | 1.194 | -1.76 | 0.953 | 1.000 |
| Gold (GC) | 28 | -0.505 [-1.095, -0.104] | 1.505 | -1.72 | 0.950 | 1.000 |
| Copper (HG) | 28 | 0.058 [-0.087, 0.197] | 0.942 | 1.07 | 0.148 | 0.592 |
| Bitcoin (BTC) | 28 | -0.057 [-0.540, 0.355] | 1.057 | 0.61 | 0.275 | 0.824 |
| Equity market | 28 | -0.015 [-0.039, 0.010] | 1.015 | -1.18 | 0.874 | n/a |

## 2. Descriptive fit by period (primary model; identical observations for M0, M1, M2)

Adjusted R-squared of the benchmark (equity market), factors only, and benchmark + factors; joint test that the factor coefficients are zero. 'all' includes months the loadings were fitted on and is **not** out-of-sample. Minimum detectable incremental R-squared (80% power, alpha 0.05 / 0.0125) shows what each sample could detect.


**primary (narr_finbert__sipca)**

| asset | sample | n | adj R2 M0 | M1 | M2 | increment (M2-M0) | joint p HC3 | joint p NW-calendar | detectable R2 increment |
|---|---|---|---|---|---|---|---|---|---|
| WTI crude (CL) | all | 80 | 0.089 | 0.267 | 0.291 | 0.202 | 0.000 | 0.000 | 0.113 / 0.150 |
| Gold (GC) | all | 80 | -0.011 | 0.013 | 0.001 | 0.011 | 0.176 | 0.114 | 0.125 / 0.166 |
| Copper (HG) | all | 80 | 0.158 | 0.068 | 0.201 | 0.043 | 0.060 | 0.072 | 0.104 / 0.138 |
| Bitcoin (BTC) | all | 49 | 0.104 | 0.015 | 0.129 | 0.025 | 0.116 | 0.058 | 0.172 / 0.224 |
| Equity market | all | 104 | 0.000 | 0.056 | 0.056 | 0.056 | 0.102 | 0.097 | 0.098 / 0.131 |
| WTI crude (CL) | train | 45 | 0.168 | 0.258 | 0.280 | 0.113 | 0.086 | 0.028 | 0.172 / 0.223 |
| Gold (GC) | train | 45 | 0.067 | -0.023 | 0.040 | -0.026 | 0.765 | 0.679 | 0.192 / 0.250 |
| Copper (HG) | train | 45 | 0.072 | 0.091 | 0.108 | 0.036 | 0.097 | 0.107 | 0.191 / 0.249 |
| Bitcoin (BTC) | train | 14 | -0.075 | -0.183 | -0.135 | -0.059 | 0.790 | 0.503 | 0.544 / 0.662 |
| Equity market | train | 69 | 0.000 | 0.121 | 0.121 | 0.121 | 0.028 | 0.022 | 0.144 / 0.190 |
| WTI crude (CL) | validation | 14 | 0.353 | 0.510 | 0.554 | 0.201 | 0.317 | 0.013 | 0.327 / 0.398 |
| Gold (GC) | validation | 14 | -0.048 | -0.000 | 0.096 | 0.144 | 0.551 | 0.184 | 0.530 / 0.645 |
| Copper (HG) | validation | 14 | 0.577 | 0.263 | 0.606 | 0.029 | 0.755 | 0.485 | 0.214 / 0.260 |
| Bitcoin (BTC) | validation | 14 | 0.202 | 0.061 | 0.462 | 0.260 | 0.072 | 0.014 | 0.404 / 0.491 |
| Equity market | validation | 14 | 0.000 | 0.159 | 0.159 | 0.159 | 0.369 | 0.050 | 0.537 / 0.652 |
| WTI crude (CL) | test | 21 | -0.031 | 0.203 | 0.168 | 0.200 | 0.443 | 0.209 | 0.391 / 0.490 |
| Gold (GC) | test | 21 | -0.005 | -0.061 | -0.104 | -0.099 | 0.550 | 0.344 | 0.381 / 0.478 |
| Copper (HG) | test | 21 | 0.113 | -0.152 | 0.070 | -0.044 | 0.719 | 0.306 | 0.336 / 0.422 |
| Bitcoin (BTC) | test | 21 | 0.178 | 0.075 | 0.189 | 0.011 | 0.234 | 0.031 | 0.312 / 0.391 |
| Equity market | test | 21 | 0.000 | 0.121 | 0.121 | 0.121 | 0.296 | 0.146 | 0.396 / 0.495 |

**secondary (narr_nolbert20__sipca)**

| asset | sample | n | adj R2 M0 | M1 | M2 | increment (M2-M0) | joint p HC3 | joint p NW-calendar | detectable R2 increment |
|---|---|---|---|---|---|---|---|---|---|
| WTI crude (CL) | all | 80 | 0.089 | 0.081 | 0.137 | 0.048 | 0.056 | 0.042 | 0.082 / 0.114 |
| Gold (GC) | all | 80 | -0.011 | -0.011 | -0.023 | -0.012 | 0.813 | 0.812 | 0.091 / 0.127 |
| Copper (HG) | all | 80 | 0.158 | 0.077 | 0.195 | 0.036 | 0.059 | 0.051 | 0.076 / 0.105 |
| Bitcoin (BTC) | all | 49 | 0.104 | 0.026 | 0.115 | 0.011 | 0.211 | 0.187 | 0.126 / 0.172 |
| Equity market | all | 104 | 0.000 | 0.034 | 0.034 | 0.034 | 0.057 | 0.043 | 0.071 / 0.100 |
| WTI crude (CL) | train | 45 | 0.168 | 0.258 | 0.370 | 0.202 | 0.002 | 0.001 | 0.126 / 0.172 |
| Gold (GC) | train | 45 | 0.067 | -0.017 | 0.045 | -0.021 | 0.855 | 0.855 | 0.141 / 0.192 |
| Copper (HG) | train | 45 | 0.072 | 0.024 | 0.079 | 0.007 | 0.322 | 0.293 | 0.140 / 0.191 |
| Bitcoin (BTC) | train | 14 | -0.075 | -0.055 | -0.151 | -0.076 | 0.420 | 0.215 | 0.401 / 0.518 |
| Equity market | train | 69 | 0.000 | 0.009 | 0.009 | 0.009 | 0.200 | 0.195 | 0.105 / 0.145 |
| WTI crude (CL) | validation | 14 | 0.353 | 0.129 | 0.343 | -0.009 | 0.390 | 0.163 | 0.241 / 0.312 |
| Gold (GC) | validation | 14 | -0.048 | 0.258 | 0.424 | 0.472 | 0.058 | 0.007 | 0.390 / 0.504 |
| Copper (HG) | validation | 14 | 0.577 | 0.146 | 0.568 | -0.009 | 0.216 | 0.155 | 0.158 / 0.204 |
| Bitcoin (BTC) | validation | 14 | 0.202 | -0.056 | 0.133 | -0.069 | 0.854 | 0.810 | 0.297 / 0.384 |
| Equity market | validation | 14 | 0.000 | 0.093 | 0.093 | 0.093 | 0.501 | 0.088 | 0.400 / 0.514 |
| WTI crude (CL) | test | 21 | -0.031 | 0.061 | 0.015 | 0.046 | 0.174 | 0.113 | 0.289 / 0.382 |
| Gold (GC) | test | 21 | -0.005 | 0.051 | 0.022 | 0.028 | 0.130 | 0.038 | 0.282 / 0.372 |
| Copper (HG) | test | 21 | 0.113 | 0.150 | 0.208 | 0.095 | 0.098 | 0.040 | 0.248 / 0.328 |
| Bitcoin (BTC) | test | 21 | 0.178 | 0.044 | 0.179 | 0.001 | 0.473 | 0.386 | 0.230 / 0.304 |
| Equity market | test | 21 | 0.000 | -0.002 | -0.002 | -0.002 | 0.425 | 0.339 | 0.293 / 0.387 |

## 3. Common-month comparison across assets (months with returns for all four futures; primary model)

| asset | sample | n | adj R2 M0 | M1 | M2 | increment | joint p HC3 |
|---|---|---|---|---|---|---|---|
| WTI crude (CL) | all | 49 | 0.084 | 0.284 | 0.303 | 0.219 | 0.010 |
| Gold (GC) | all | 49 | -0.015 | 0.028 | 0.044 | 0.060 | 0.041 |
| Copper (HG) | all | 49 | 0.265 | 0.018 | 0.283 | 0.018 | 0.330 |
| Bitcoin (BTC) | all | 49 | 0.104 | 0.015 | 0.129 | 0.025 | 0.116 |
| Equity market | all | 49 | 0.000 | 0.070 | 0.070 | 0.070 | 0.215 |
| WTI crude (CL) | train | 14 | 0.322 | 0.206 | 0.261 | -0.061 | 0.834 |
| Gold (GC) | train | 14 | 0.042 | 0.087 | 0.090 | 0.048 | 0.340 |
| Copper (HG) | train | 14 | 0.150 | -0.013 | 0.169 | 0.019 | 0.371 |
| Bitcoin (BTC) | train | 14 | -0.075 | -0.183 | -0.135 | -0.059 | 0.790 |
| Equity market | train | 14 | 0.000 | 0.296 | 0.296 | 0.296 | 0.056 |
| WTI crude (CL) | validation | 14 | 0.353 | 0.510 | 0.554 | 0.201 | 0.317 |
| Gold (GC) | validation | 14 | -0.048 | -0.000 | 0.096 | 0.144 | 0.551 |
| Copper (HG) | validation | 14 | 0.577 | 0.263 | 0.606 | 0.029 | 0.755 |
| Bitcoin (BTC) | validation | 14 | 0.202 | 0.061 | 0.462 | 0.260 | 0.072 |
| Equity market | validation | 14 | 0.000 | 0.159 | 0.159 | 0.159 | 0.369 |
| WTI crude (CL) | test | 21 | -0.031 | 0.203 | 0.168 | 0.200 | 0.443 |
| Gold (GC) | test | 21 | -0.005 | -0.061 | -0.104 | -0.099 | 0.550 |
| Copper (HG) | test | 21 | 0.113 | -0.152 | 0.070 | -0.044 | 0.719 |
| Bitcoin (BTC) | test | 21 | 0.178 | 0.075 | 0.189 | 0.011 | 0.234 |
| Equity market | test | 21 | 0.000 | 0.121 | 0.121 | 0.121 | 0.296 |

## 4. Factor coefficients (primary model, M2), descriptive

Coefficient: monthly return of the asset per unit of factor return (both in decimals); standardised: coefficient x sd(factor) / sd(asset). Test-window rows are re-estimated on the 21 test months and are descriptive, not out-of-sample; they carry both the unadjusted 95% interval and a Bonferroni interval over the 12 primary coefficients.

| asset | sample | n | factor | coef | 95% CI (unadjusted) | 95% CI (Bonferroni 12) | standardised | p |
|---|---|---|---|---|---|---|---|---|
| WTI crude (CL) | all | 80 | f1 | -1.059 | [-2.414, 0.296] | [-3.070, 0.951] | -0.198 | 0.124 |
| WTI crude (CL) | all | 80 | f2 | -0.514 | [-1.534, 0.505] | [-2.027, 0.998] | -0.095 | 0.318 |
| WTI crude (CL) | all | 80 | f3 | 2.595 | [1.355, 3.834] | [0.756, 4.434] | 0.461 | 0.000 |
| Gold (GC) | all | 80 | f1 | 0.142 | [-0.490, 0.775] | [-0.796, 1.081] | 0.063 | 0.655 |
| Gold (GC) | all | 80 | f2 | 0.489 | [-0.051, 1.029] | [-0.313, 1.290] | 0.213 | 0.075 |
| Gold (GC) | all | 80 | f3 | -0.188 | [-0.763, 0.386] | [-1.041, 0.664] | -0.079 | 0.515 |
| Copper (HG) | all | 80 | f1 | -0.627 | [-1.258, 0.004] | [-1.563, 0.310] | -0.198 | 0.052 |
| Copper (HG) | all | 80 | f2 | 0.381 | [-0.283, 1.044] | [-0.604, 1.365] | 0.118 | 0.257 |
| Copper (HG) | all | 80 | f3 | 0.501 | [-0.302, 1.303] | [-0.691, 1.692] | 0.150 | 0.218 |
| Bitcoin (BTC) | all | 49 | f1 | 2.799 | [-0.533, 6.131] | [-2.198, 7.797] | 0.220 | 0.098 |
| Bitcoin (BTC) | all | 49 | f2 | 0.131 | [-3.551, 3.813] | [-5.391, 5.654] | 0.014 | 0.943 |
| Bitcoin (BTC) | all | 49 | f3 | -2.386 | [-5.600, 0.829] | [-7.207, 2.436] | -0.233 | 0.142 |
| Equity market | all | 104 | f1 | -0.185 | [-0.664, 0.295] | [-0.894, 0.524] | -0.077 | 0.446 |
| Equity market | all | 104 | f2 | -0.499 | [-1.092, 0.095] | [-1.376, 0.379] | -0.199 | 0.099 |
| Equity market | all | 104 | f3 | 0.531 | [-0.147, 1.208] | [-0.471, 1.532] | 0.205 | 0.124 |
| WTI crude (CL) | test | 21 | f1 | -3.474 | [-8.245, 1.298] | [-10.989, 4.041] | -0.533 | 0.142 |
| WTI crude (CL) | test | 21 | f2 | 0.838 | [-1.641, 3.317] | [-3.066, 4.742] | 0.201 | 0.484 |
| WTI crude (CL) | test | 21 | f3 | 2.883 | [-1.009, 6.776] | [-3.247, 9.013] | 0.614 | 0.136 |
| Gold (GC) | test | 21 | f1 | 0.917 | [-0.407, 2.241] | [-1.168, 3.002] | 0.304 | 0.161 |
| Gold (GC) | test | 21 | f2 | 0.143 | [-0.979, 1.266] | [-1.625, 1.912] | 0.074 | 0.790 |
| Gold (GC) | test | 21 | f3 | -0.423 | [-1.553, 0.707] | [-2.203, 1.356] | -0.195 | 0.439 |
| Copper (HG) | test | 21 | f1 | -0.417 | [-2.081, 1.246] | [-3.037, 2.202] | -0.103 | 0.602 |
| Copper (HG) | test | 21 | f2 | 0.840 | [-1.018, 2.699] | [-2.086, 3.767] | 0.322 | 0.352 |
| Copper (HG) | test | 21 | f3 | 0.368 | [-1.540, 2.276] | [-2.637, 3.373] | 0.125 | 0.688 |
| Bitcoin (BTC) | test | 21 | f1 | 3.298 | [-1.428, 8.025] | [-4.145, 10.742] | 0.253 | 0.158 |
| Bitcoin (BTC) | test | 21 | f2 | 1.511 | [-5.036, 8.058] | [-8.799, 11.822] | 0.181 | 0.631 |
| Bitcoin (BTC) | test | 21 | f3 | -3.701 | [-7.924, 0.521] | [-10.351, 2.949] | -0.394 | 0.082 |
| Equity market | test | 21 | f1 | 1.542 | [-1.375, 4.459] | [-3.030, 6.114] | 0.356 | 0.280 |
| Equity market | test | 21 | f2 | -1.032 | [-3.133, 1.070] | [-4.325, 2.262] | -0.372 | 0.315 |
| Equity market | test | 21 | f3 | -0.883 | [-2.992, 1.226] | [-4.188, 2.422] | -0.283 | 0.389 |

Secondary model (K = 1), same layout:

| asset | sample | n | coef | 95% CI (unadjusted) | standardised | p |
|---|---|---|---|---|---|---|
| WTI crude (CL) | all | 80 | 1.139 | [-0.032, 2.309] | 0.248 | 0.056 |
| Gold (GC) | all | 80 | -0.061 | [-0.577, 0.455] | -0.032 | 0.813 |
| Copper (HG) | all | 80 | 0.598 | [-0.023, 1.218] | 0.220 | 0.059 |
| Bitcoin (BTC) | all | 49 | 1.886 | [-1.107, 4.880] | 0.173 | 0.211 |
| Equity market | all | 104 | 0.441 | [-0.014, 0.895] | 0.208 | 0.057 |
| WTI crude (CL) | test | 21 | -1.152 | [-2.862, 0.558] | -0.312 | 0.174 |
| Gold (GC) | test | 21 | 0.480 | [-0.155, 1.115] | 0.281 | 0.130 |
| Copper (HG) | test | 21 | 0.851 | [-0.175, 1.876] | 0.369 | 0.098 |
| Bitcoin (BTC) | test | 21 | 1.551 | [-2.890, 5.992] | 0.210 | 0.473 |
| Equity market | test | 21 | 0.537 | [-0.842, 1.916] | 0.219 | 0.425 |

## 5. Comparison table: explanatory evidence versus forecasting evidence

| asset | explanatory evidence (primary model, test months, held-out gain over the equity market) | descriptive fit, all months (increment in adj R2, joint p) | forecasting evidence (a different object, see note) |
|---|---|---|---|
| WTI crude (CL) | gain 0.309 [-0.022, 0.526], one-sided p 0.026, Holm 0.103 | 0.202, p 0.000 (n=80) | weekly topic shocks, oil: R2 vs expanding mean -5.8% / -6.4% (test); no forecasting gain |
| Gold (GC) | gain -0.093 [-0.420, 0.159], one-sided p 0.509, Holm 0.509 | 0.011, p 0.176 (n=80) | weekly topic shock, gold: R2 vs mean -0.4%; hit rate 47% |
| Copper (HG) | gain 0.127 [-0.063, 0.324], one-sided p 0.036, Holm 0.108 | 0.043, p 0.060 (n=80) | not tested |
| Bitcoin (BTC) | gain 0.039 [-0.192, 0.207], one-sided p 0.095, Holm 0.189 | 0.025, p 0.116 (n=49) | weekly topic shock, Fed: R2 vs mean -1.5%; hit rate 46% |
| Equity market | gain -0.093 [-0.317, 0.105], one-sided p 0.646 | 0.056, p 0.102 (n=104) | stock-level backtest of the same models (monthly, buffered, test): primary Sharpe 0.11 gross / -0.04 after 10 bp (validation 0.21); secondary -0.25 / -0.37 (validation -0.80) |

Note on the forecasting column: for equities it is the out-of-sample stock-level backtest of the same two specifications (Phase 1, no new search). For CL, GC and BTC it is the earlier weekly topic-shock forecasting result, not an IPCA-factor forecast. IPCA factor forecasting of commodities was not tested.
