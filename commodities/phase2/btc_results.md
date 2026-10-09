# Bitcoin pilot (exploratory)

Specification: `btc_prereg.md` (SHA-256 `fd7cbb8ee2126d1bcc60d8777c94af7fc6ab41617f80f03386524ed878c14ed3`). Same construction as the CL/GC pilot. Coefficients are percentage points of weekly return per 1 development-sample standard deviation of the shock; HAC errors, 4 lags. Weeks: development 104, validation 105, test 149.


## Contemporaneous: own-window regressions

| narrative | role | window | n | coef | 95% CI (unadjusted) | p | adj R2 | adj R2 controls only |
|---|---|---|---|---|---|---|---|---|
| F_fed | primary | dev | 104 | -0.380 | [-2.613, 1.853] | 0.739 | -0.024 | -0.015 |
| F_fed | primary | val | 105 | -1.423 | [-4.271, 1.425] | 0.327 | 0.076 | 0.077 |
| F_fed | primary | test | 149 | -1.162 | [-2.486, 0.161] | 0.085 | 0.080 | 0.076 |
| F_macro | secondary | dev | 104 | -2.797 | [-4.110, -1.483] | 0.000 | 0.056 | -0.015 |
| F_macro | secondary | val | 105 | -0.630 | [-2.910, 1.649] | 0.588 | 0.070 | 0.077 |
| F_macro | secondary | test | 149 | 0.167 | [-1.511, 1.846] | 0.845 | 0.070 | 0.076 |
| F_curr | secondary | dev | 104 | 1.125 | [-0.644, 2.894] | 0.213 | -0.012 | -0.015 |
| F_curr | secondary | val | 105 | 1.586 | [0.411, 2.761] | 0.008 | 0.090 | 0.077 |
| F_curr | secondary | test | 149 | 2.190 | [-1.698, 6.079] | 0.270 | 0.090 | 0.076 |
| F_oil | cross-narrative control | dev | 104 | -0.300 | [-2.266, 1.666] | 0.765 | -0.024 | -0.015 |
| F_oil | cross-narrative control | val | 105 | -1.399 | [-2.929, 0.131] | 0.073 | 0.083 | 0.077 |
| F_oil | cross-narrative control | test | 149 | 0.475 | [-0.766, 1.716] | 0.453 | 0.074 | 0.076 |

## Contemporaneous: frozen-coefficient out-of-sample

| narrative | role | stage | n | OOS R2 vs controls-only [95% block-bootstrap CI] |
|---|---|---|---|---|
| F_fed | primary | val | 105 | 0.003 [-0.007, 0.013] |
| F_fed | primary | test | 149 | 0.007 [-0.009, 0.023] |
| F_macro | secondary | val | 105 | -0.041 [-0.137, 0.051] |
| F_macro | secondary | test | 149 | -0.039 [-0.114, 0.022] |
| F_curr | secondary | val | 105 | 0.032 [0.005, 0.068] |
| F_curr | secondary | test | 149 | 0.021 [-0.034, 0.073] |
| F_oil | cross-narrative control | val | 105 | 0.006 [-0.003, 0.017] |
| F_oil | cross-narrative control | test | 149 | -0.029 [-0.066, 0.012] |

## Forecasting: own-window regressions

| narrative | role | window | n | coef | 95% CI (unadjusted) | p | adj R2 | adj R2 controls only |
|---|---|---|---|---|---|---|---|---|
| F_fed | primary | dev | 103 | 0.161 | [-1.777, 2.098] | 0.871 | -0.014 | -0.004 |
| F_fed | primary | val | 105 | -0.727 | [-2.713, 1.259] | 0.473 | -0.004 | 0.004 |
| F_fed | primary | test | 147 | 1.268 | [-0.959, 3.496] | 0.264 | -0.005 | -0.010 |
| F_macro | secondary | dev | 103 | 0.761 | [-0.830, 2.353] | 0.348 | -0.008 | -0.004 |
| F_macro | secondary | val | 105 | -1.080 | [-3.229, 1.069] | 0.325 | 0.002 | 0.004 |
| F_macro | secondary | test | 147 | 0.914 | [-0.679, 2.507] | 0.261 | -0.008 | -0.010 |
| F_curr | secondary | dev | 103 | -0.386 | [-2.056, 1.283] | 0.650 | -0.013 | -0.004 |
| F_curr | secondary | val | 105 | -1.254 | [-2.922, 0.415] | 0.141 | 0.007 | 0.004 |
| F_curr | secondary | test | 147 | 0.980 | [-1.211, 3.170] | 0.381 | -0.013 | -0.010 |
| F_oil | cross-narrative control | dev | 103 | -0.324 | [-2.246, 1.599] | 0.741 | -0.013 | -0.004 |
| F_oil | cross-narrative control | val | 105 | 1.498 | [-0.467, 3.463] | 0.135 | 0.009 | 0.004 |
| F_oil | cross-narrative control | test | 147 | -0.392 | [-1.915, 1.132] | 0.614 | -0.015 | -0.010 |

## Forecasting: frozen-coefficient out-of-sample

| narrative | role | stage | n | OOS R2 vs controls-only [95% block-bootstrap CI] | OOS R2 vs expanding mean | hit rate (full / controls) |
|---|---|---|---|---|---|---|
| F_fed | primary | val | 105 | -0.001 [-0.004, 0.002] | -0.049 | 0.50 / 0.50 |
| F_fed | primary | test | 147 | -0.002 [-0.004, 0.001] | -0.015 | 0.46 / 0.48 |
| F_macro | secondary | val | 105 | -0.009 [-0.034, 0.015] | -0.057 | 0.49 / 0.50 |
| F_macro | secondary | test | 147 | 0.000 [-0.000, 0.001] | -0.013 | 0.49 / 0.48 |
| F_curr | secondary | val | 105 | 0.009 [-0.003, 0.024] | -0.039 | 0.50 / 0.50 |
| F_curr | secondary | test | 147 | -0.010 [-0.028, 0.006] | -0.023 | 0.53 / 0.48 |
| F_oil | cross-narrative control | val | 105 | -0.009 [-0.018, -0.000] | -0.057 | 0.50 / 0.50 |
| F_oil | cross-narrative control | test | 147 | -0.015 [-0.042, 0.011] | -0.028 | 0.52 / 0.48 |

## Timing check (test window)

| narrative | role | coef | 95% CI | p |
|---|---|---|---|---|
| F_fed | primary | -0.329 | [-2.200, 1.542] | 0.730 |
| F_macro | secondary | -0.294 | [-1.614, 1.027] | 0.663 |
| F_curr | secondary | -4.696 | [-7.370, -2.022] | 0.001 |
| F_oil | cross-narrative control | -0.140 | [-1.234, 0.955] | 0.803 |

## Sensitivity: drop weeks with Databento degraded days (test window)

| kind | coef | p | OOS R2 vs controls-only |
|---|---|---|---|
| contemporaneous | -1.223 | 0.073 | 0.008 |
| forecasting | 1.103 | 0.342 | -0.001 |

## Primary tests, test window

| asset | narrative | kind | coef | 95% CI unadjusted | 95% CI Bonferroni (8 tests) | raw p | Holm p (BTC only, 2) | Holm p (all 8) |
|---|---|---|---|---|---|---|---|---|
| BTC | F_fed | contemporaneous | -1.162 | [-2.486, 0.161] | [-3.008, 0.684] | 0.085 | 0.170 | 0.460 |
| BTC | F_fed | forecasting | 1.268 | [-0.959, 3.496] | [-1.840, 4.376] | 0.264 | 0.264 | 0.558 |