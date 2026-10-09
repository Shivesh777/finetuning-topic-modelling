# CL / GC pilot results

Specification: `pilot_prereg.md` (SHA-256 `1d51f6a83ddc26a3d9891d5962ac26492b028a82b1d7fd910f325705a676935f`). HAC standard errors, 4 weekly lags. Coefficients are percentage points of weekly return per 1 development-sample standard deviation of the shock. Each regression was run once.


## Contemporaneous: own-window regressions (descriptive exposures)

| asset | narrative | role | window | n | coef | 95% CI | p | adj R2 | adj R2 controls only |
|---|---|---|---|---|---|---|---|---|---|
| CL | N_oil | primary | dev | 258 | 0.596 | [0.011, 1.181] | 0.046 | 0.128 | 0.114 |
| CL | N_oil | primary | val | 105 | 0.396 | [-0.704, 1.495] | 0.480 | 0.127 | 0.133 |
| CL | N_oil | primary | test | 149 | 1.305 | [0.108, 2.502] | 0.033 | 0.044 | 0.013 |
| CL | F_oil | primary | dev | 258 | 0.991 | [0.387, 1.595] | 0.001 | 0.156 | 0.114 |
| CL | F_oil | primary | val | 105 | 0.112 | [-1.618, 1.841] | 0.899 | 0.124 | 0.133 |
| CL | F_oil | primary | test | 149 | 0.777 | [-0.136, 1.689] | 0.095 | 0.029 | 0.013 |
| CL | F_gold | cross-narrative control | dev | 258 | -0.035 | [-0.557, 0.487] | 0.896 | 0.111 | 0.114 |
| CL | F_gold | cross-narrative control | val | 105 | 0.261 | [-0.703, 1.224] | 0.596 | 0.126 | 0.133 |
| CL | F_gold | cross-narrative control | test | 149 | -0.545 | [-1.119, 0.030] | 0.063 | 0.021 | 0.013 |
| GC | F_gold | primary | dev | 258 | 0.018 | [-0.186, 0.222] | 0.865 | 0.024 | 0.027 |
| GC | F_gold | primary | val | 105 | -0.021 | [-0.385, 0.343] | 0.910 | 0.160 | 0.168 |
| GC | F_gold | primary | test | 149 | -0.170 | [-0.359, 0.018] | 0.077 | 0.006 | 0.004 |
| GC | F_fed | secondary | dev | 258 | -0.010 | [-0.183, 0.164] | 0.911 | 0.023 | 0.027 |
| GC | F_fed | secondary | val | 105 | 0.039 | [-0.493, 0.570] | 0.887 | 0.160 | 0.168 |
| GC | F_fed | secondary | test | 149 | -0.150 | [-0.606, 0.305] | 0.518 | 0.000 | 0.004 |
| GC | F_oil | cross-narrative control | dev | 258 | 0.131 | [-0.080, 0.342] | 0.223 | 0.029 | 0.027 |
| GC | F_oil | cross-narrative control | val | 105 | -0.066 | [-0.415, 0.283] | 0.712 | 0.161 | 0.168 |
| GC | F_oil | cross-narrative control | test | 149 | 0.241 | [-0.025, 0.508] | 0.076 | 0.012 | 0.004 |

## Contemporaneous: frozen-coefficient out-of-sample

| asset | narrative | role | stage | n | OOS R2 vs controls-only (95% block-bootstrap CI) | |
|---|---|---|---|---|---|
| CL | N_oil | primary | val | 105 | 0.006 [-0.037, 0.035] |
| CL | N_oil | primary | test | 149 | 0.019 [-0.006, 0.045] |
| CL | F_oil | primary | val | 105 | -0.004 [-0.081, 0.060] |
| CL | F_oil | primary | test | 149 | 0.014 [-0.035, 0.069] |
| CL | F_gold | cross-narrative control | val | 105 | -0.001 [-0.002, 0.001] |
| CL | F_gold | cross-narrative control | test | 149 | -0.004 [-0.007, -0.000] |
| GC | F_gold | primary | val | 105 | 0.000 [-0.003, 0.004] |
| GC | F_gold | primary | test | 149 | -0.001 [-0.003, 0.000] |
| GC | F_fed | secondary | val | 105 | -0.000 [-0.001, 0.001] |
| GC | F_fed | secondary | test | 149 | -0.000 [-0.001, 0.001] |
| GC | F_oil | cross-narrative control | val | 105 | -0.016 [-0.031, 0.007] |
| GC | F_oil | cross-narrative control | test | 149 | 0.004 [-0.001, 0.009] |

## Forecasting: own-window regressions (descriptive exposures)

| asset | narrative | role | window | n | coef | 95% CI | p | adj R2 | adj R2 controls only |
|---|---|---|---|---|---|---|---|---|---|
| CL | N_oil | primary | dev | 258 | 0.127 | [-0.396, 0.649] | 0.635 | -0.001 | 0.003 |
| CL | N_oil | primary | val | 105 | -1.275 | [-3.848, 1.299] | 0.332 | 0.061 | 0.049 |
| CL | N_oil | primary | test | 147 | -1.293 | [-2.311, -0.274] | 0.013 | 0.035 | 0.007 |
| CL | F_oil | primary | dev | 258 | -0.001 | [-0.479, 0.476] | 0.995 | -0.001 | 0.003 |
| CL | F_oil | primary | val | 105 | -1.871 | [-5.187, 1.444] | 0.269 | 0.075 | 0.049 |
| CL | F_oil | primary | test | 147 | -0.420 | [-1.043, 0.202] | 0.186 | 0.007 | 0.007 |
| CL | F_gold | cross-narrative control | dev | 258 | 0.283 | [-0.172, 0.737] | 0.223 | 0.003 | 0.003 |
| CL | F_gold | cross-narrative control | val | 105 | 0.640 | [-0.212, 1.491] | 0.141 | 0.046 | 0.049 |
| CL | F_gold | cross-narrative control | test | 147 | -0.058 | [-0.660, 0.544] | 0.850 | 0.001 | 0.007 |
| GC | F_gold | primary | dev | 258 | 0.065 | [-0.127, 0.257] | 0.508 | 0.001 | 0.003 |
| GC | F_gold | primary | val | 105 | 0.213 | [-0.126, 0.552] | 0.219 | 0.015 | 0.015 |
| GC | F_gold | primary | test | 147 | -0.012 | [-0.434, 0.410] | 0.954 | -0.021 | -0.014 |
| GC | F_fed | secondary | dev | 258 | -0.070 | [-0.240, 0.100] | 0.421 | 0.001 | 0.003 |
| GC | F_fed | secondary | val | 105 | 0.049 | [-0.431, 0.528] | 0.843 | 0.006 | 0.015 |
| GC | F_fed | secondary | test | 147 | 0.328 | [-0.091, 0.748] | 0.125 | -0.005 | -0.014 |
| GC | F_oil | cross-narrative control | dev | 258 | -0.134 | [-0.364, 0.096] | 0.253 | 0.005 | 0.003 |
| GC | F_oil | cross-narrative control | val | 105 | -0.273 | [-0.692, 0.146] | 0.202 | 0.015 | 0.015 |
| GC | F_oil | cross-narrative control | test | 147 | -0.020 | [-0.381, 0.340] | 0.912 | -0.021 | -0.014 |

## Forecasting: frozen-coefficient out-of-sample

| asset | narrative | role | stage | n | OOS R2 vs controls-only (95% block-bootstrap CI) | | OOS R2 vs expanding mean | controls-only R2 vs mean | hit rate (full / controls) |
|---|---|---|---|---|---|---|---|---|
| CL | N_oil | primary | val | 105 | -0.005 [-0.011, 0.004] | 0.047 | 0.052 | 0.55 / 0.61 |
| CL | N_oil | primary | test | 147 | 0.015 [0.002, 0.030] | -0.058 | -0.074 | 0.50 / 0.50 |
| CL | F_oil | primary | val | 105 | 0.000 [-0.000, 0.000] | 0.052 | 0.052 | 0.61 / 0.61 |
| CL | F_oil | primary | test | 147 | 0.009 [-0.013, 0.034] | -0.064 | -0.074 | 0.54 / 0.50 |
| CL | F_gold | cross-narrative control | val | 105 | 0.004 [-0.005, 0.019] | 0.056 | 0.052 | 0.60 / 0.61 |
| CL | F_gold | cross-narrative control | test | 147 | -0.008 [-0.032, 0.012] | -0.082 | -0.074 | 0.52 / 0.50 |
| GC | F_gold | primary | val | 105 | 0.004 [-0.007, 0.015] | -0.033 | -0.037 | 0.55 / 0.54 |
| GC | F_gold | primary | test | 147 | -0.004 [-0.029, 0.020] | -0.004 | 0.000 | 0.47 / 0.46 |
| GC | F_fed | secondary | val | 105 | -0.000 [-0.012, 0.007] | -0.037 | -0.037 | 0.49 / 0.54 |
| GC | F_fed | secondary | test | 147 | -0.005 [-0.011, 0.001] | -0.005 | 0.000 | 0.46 / 0.46 |
| GC | F_oil | cross-narrative control | val | 105 | -0.002 [-0.018, 0.016] | -0.039 | -0.037 | 0.52 / 0.54 |
| GC | F_oil | cross-narrative control | test | 147 | -0.003 [-0.022, 0.019] | -0.002 | 0.000 | 0.45 / 0.46 |

## Timing check (contemporaneous regression with next week's shock, test window)

| asset | narrative | role | coef | 95% CI | p |
|---|---|---|---|---|---|
| CL | N_oil | primary | 0.115 | [-1.098, 1.328] | 0.852 |
| CL | F_oil | primary | 0.021 | [-1.391, 1.434] | 0.976 |
| CL | F_gold | cross-narrative control | 0.228 | [-0.289, 0.744] | 0.387 |
| GC | F_gold | primary | 0.385 | [0.151, 0.619] | 0.001 |
| GC | F_fed | secondary | 0.115 | [-0.300, 0.530] | 0.588 |
| GC | F_oil | cross-narrative control | -0.089 | [-0.433, 0.256] | 0.613 |

## Primary tests, test window, with Holm adjustment across the six

| asset | narrative | kind | coef | 95% CI | raw p | Holm p |
|---|---|---|---|---|---|---|
| CL | N_oil | contemporaneous | 1.305 | [0.108, 2.502] | 0.033 | 0.163 |
| CL | F_oil | contemporaneous | 0.777 | [-0.136, 1.689] | 0.095 | 0.307 |
| CL | N_oil | forecasting | -1.293 | [-2.311, -0.274] | 0.013 | 0.077 |
| CL | F_oil | forecasting | -0.420 | [-1.043, 0.202] | 0.186 | 0.372 |
| GC | F_gold | contemporaneous | -0.170 | [-0.359, 0.018] | 0.077 | 0.307 |
| GC | F_gold | forecasting | -0.012 | [-0.434, 0.410] | 0.954 | 0.954 |

## Sensitivities, test window

| asset | narrative | kind | variant | coef | p | OOS R2 vs controls-only |
|---|---|---|---|---|---|---|
| CL | N_oil_t55 | contemporaneous | t55 | 0.941 | 0.092 | 0.012 |
| CL | N_oil | contemporaneous | drop_flagged | 1.241 | 0.060 | 0.014 |
| CL | F_oil | contemporaneous | drop_flagged | 0.713 | 0.138 | 0.009 |
| CL | N_oil_t55 | forecasting | t55 | -0.316 | 0.504 | 0.004 |
| CL | N_oil | forecasting | drop_flagged | -1.517 | 0.003 | 0.017 |
| CL | F_oil | forecasting | drop_flagged | -0.496 | 0.117 | 0.010 |
| GC | F_gold | contemporaneous | drop_flagged | -0.171 | 0.077 | -0.002 |
| GC | F_gold | forecasting | drop_flagged | 0.004 | 0.985 | -0.004 |