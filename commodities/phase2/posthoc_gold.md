# Post-hoc diagnostic: does news attention follow price? (gold, with oil for comparison)

**Status: exploratory and post hoc.** Written after seeing that the gold timing check failed in the test window of the pre-specified pilot. It does not change, replace or re-test the pilot's primary results, which stand as reported. Shocks and returns are those of `weekly_panel.parquet`; shocks are in development-sample standard deviations; returns in percent. HAC (Newey-West, 4 lags) errors. All three windows are shown.

## 1. Cross-correlation of weekly return with the contemporaneous-shock series at different leads and lags

corr(return in week w, shock of week w+k). k > 0 means the shock comes after the return (attention following price); k < 0 means the shock precedes the return. Approximate 95% band for zero: +/- 1.96 / sqrt(n).


**GC / F_gold**

| window | n | k=-2 | k=-1 | k=0 | k=+1 | k=+2 | band |
|---|---|---|---|---|---|---|---|
| development 2015-2019 | 260 | -0.01 | +0.02 | +0.02 | +0.03 | +0.05 | +/-0.12 |
| validation 2020-2021 | 105 | -0.08 | +0.09 | -0.02 | +0.02 | +0.05 | +/-0.19 |
| test 2022-2024 | 149 | -0.13 | +0.03 | -0.10 | +0.22 | -0.08 | +/-0.16 |

**GC / F_fed**

| window | n | k=-2 | k=-1 | k=0 | k=+1 | k=+2 | band |
|---|---|---|---|---|---|---|---|
| development 2015-2019 | 260 | -0.04 | -0.03 | -0.02 | -0.06 | +0.08 | +/-0.12 |
| validation 2020-2021 | 105 | -0.02 | +0.02 | +0.03 | -0.17 | +0.06 | +/-0.19 |
| test 2022-2024 | 149 | +0.04 | +0.09 | -0.08 | +0.05 | -0.02 | +/-0.16 |

**CL / N_oil**

| window | n | k=-2 | k=-1 | k=0 | k=+1 | k=+2 | band |
|---|---|---|---|---|---|---|---|
| development 2015-2019 | 260 | -0.08 | +0.06 | +0.10 | -0.14 | +0.03 | +/-0.12 |
| validation 2020-2021 | 105 | +0.04 | -0.13 | +0.04 | +0.08 | -0.04 | +/-0.19 |
| test 2022-2024 | 149 | +0.14 | -0.20 | +0.19 | +0.04 | -0.10 | +/-0.16 |

**CL / F_oil**

| window | n | k=-2 | k=-1 | k=0 | k=+1 | k=+2 | band |
|---|---|---|---|---|---|---|---|
| development 2015-2019 | 260 | +0.00 | +0.04 | +0.20 | -0.24 | -0.03 | +/-0.12 |
| validation 2020-2021 | 105 | +0.03 | -0.16 | -0.00 | +0.09 | -0.06 | +/-0.19 |
| test 2022-2024 | 149 | +0.16 | -0.08 | +0.15 | +0.03 | -0.12 | +/-0.16 |

## 2. Autocorrelation of the shock series itself (is part of the pattern mechanical?)

The weekly shock sums daily deviations from a 5-day trailing mean, so adjacent weeks share baseline days and are mechanically correlated.

| series | window | corr(Z_w, Z_w+1) | corr(Z_w, Z_w+2) |
|---|---|---|---|
| GC / F_gold | development 2015-2019 | -0.39 | -0.06 |
| GC / F_gold | validation 2020-2021 | -0.42 | -0.26 |
| GC / F_gold | test 2022-2024 | -0.53 | +0.16 |
| GC / F_fed | development 2015-2019 | -0.51 | +0.04 |
| GC / F_fed | validation 2020-2021 | -0.34 | -0.18 |
| GC / F_fed | test 2022-2024 | -0.36 | -0.09 |
| CL / N_oil | development 2015-2019 | -0.46 | +0.03 |
| CL / N_oil | validation 2020-2021 | -0.36 | -0.07 |
| CL / N_oil | test 2022-2024 | -0.27 | -0.10 |
| CL / F_oil | development 2015-2019 | -0.34 | -0.12 |
| CL / F_oil | validation 2020-2021 | -0.48 | +0.08 |
| CL / F_oil | test 2022-2024 | -0.41 | +0.07 |

## 3. Does next week's attention respond to this week's return, given the shock's own persistence?

Z(w+1) = a + b1 Z(w) + b2 return(w) + b3 |return(w)| + e. b2 is the response to the signed return, b3 to its size (attention may rise after any large move).

| series | window | n | b1 (own persistence) | b2 (signed return, per 1 pp) [95% CI] | b3 (|return|, per 1 pp) [95% CI] |
|---|---|---|---|---|---|
| GC / F_gold | development 2015-2019 | 260 | -0.39 | +0.019 [-0.048, +0.086] | +0.023 [-0.068, +0.115] |
| GC / F_gold | validation 2020-2021 | 105 | -0.43 | +0.004 [-0.106, +0.115] | +0.056 [-0.062, +0.174] |
| GC / F_gold | test 2022-2024 | 147 | -0.51 | +0.096 [+0.027, +0.166] | +0.002 [-0.119, +0.122] |
| GC / F_fed | development 2015-2019 | 260 | -0.51 | -0.040 [-0.091, +0.010] | +0.001 [-0.080, +0.081] |
| GC / F_fed | validation 2020-2021 | 105 | -0.33 | -0.049 [-0.093, -0.005] | +0.010 [-0.050, +0.070] |
| GC / F_fed | test 2022-2024 | 147 | -0.36 | +0.007 [-0.050, +0.065] | +0.017 [-0.064, +0.097] |
| CL / N_oil | development 2015-2019 | 260 | -0.44 | -0.022 [-0.043, -0.001] | -0.023 [-0.062, +0.017] |
| CL / N_oil | validation 2020-2021 | 105 | -0.37 | +0.012 [-0.009, +0.033] | +0.001 [-0.024, +0.026] |
| CL / N_oil | test 2022-2024 | 147 | -0.28 | +0.011 [-0.012, +0.034] | +0.019 [-0.023, +0.061] |
| CL / F_oil | development 2015-2019 | 260 | -0.30 | -0.041 [-0.065, -0.016] | -0.013 [-0.051, +0.025] |
| CL / F_oil | validation 2020-2021 | 105 | -0.50 | +0.010 [-0.006, +0.026] | +0.006 [-0.014, +0.026] |
| CL / F_oil | test 2022-2024 | 147 | -0.42 | +0.013 [-0.022, +0.048] | +0.040 [-0.019, +0.100] |

## 4. The pilot's timing check, repeated while controlling for the contemporaneous shock

return(w) = a + c1 Z(w) + c2 Z(w+1) + price controls. If c2 is only the mechanical overlap between adjacent shocks, it should shrink once Z(w) is in the model.

| series | window | n | c1 on Z(w) [95% CI] | c2 on Z(w+1) [95% CI] | c2 in the pilot's timing check (Z(w+1) alone) |
|---|---|---|---|---|---|
| GC / F_gold | development 2015-2019 | 258 | +0.036 [-0.167, +0.240] | +0.048 [-0.171, +0.266] | +0.034 |
| GC / F_gold | validation 2020-2021 | 105 | -0.005 [-0.371, +0.362] | +0.039 [-0.469, +0.548] | +0.041 |
| GC / F_gold | test 2022-2024 | 147 | +0.051 [-0.223, +0.325] | +0.411 [+0.094, +0.729] | +0.385 |
| GC / F_fed | development 2015-2019 | 258 | -0.082 [-0.278, +0.113] | -0.144 [-0.338, +0.050] | -0.102 |
| GC / F_fed | validation 2020-2021 | 105 | -0.126 [-0.755, +0.502] | -0.483 [-1.000, +0.033] | -0.441 |
| GC / F_fed | test 2022-2024 | 147 | -0.127 [-0.616, +0.363] | +0.070 [-0.378, +0.518] | +0.115 |
| CL / N_oil | development 2015-2019 | 258 | +0.312 [-0.280, +0.903] | -0.605 [-1.169, -0.042] | -0.748 |
| CL / N_oil | validation 2020-2021 | 105 | +0.778 [-0.882, +2.439] | +1.038 [-0.970, +3.046] | +0.757 |
| CL / N_oil | test 2022-2024 | 147 | +1.451 [+0.126, +2.775] | +0.510 [-0.790, +1.810] | +0.115 |
| CL / F_oil | development 2015-2019 | 258 | +0.715 [+0.136, +1.294] | -0.756 [-1.296, -0.215] | -1.004 |
| CL / F_oil | validation 2020-2021 | 105 | +0.570 [-1.547, +2.687] | +0.936 [-1.058, +2.929] | +0.669 |
| CL / F_oil | test 2022-2024 | 147 | +0.959 [+0.008, +1.910] | +0.426 [-1.115, +1.968] | +0.021 |

## 5. How to read this

Written by hand after reading sections 1-4 (numbers quoted from them).

- **Part of the pattern is mechanical.** Weekly shocks are negatively autocorrelated at lag 1 in every series and window (-0.27 to -0.53) because adjacent weeks share baseline days. A lead-lag check should therefore hold Z(w) fixed; the pilot's timing check did not.
- **Gold.** The link between this week's return and next week's gold shock is absent in 2015-2021 (cross-correlation at k=+1: +0.03 and +0.02) and appears only in 2022-2024 (+0.22, band +/-0.16). With Z(w) in the model the coefficient on Z(w+1) stays at +0.41 [+0.09, +0.73] in the test window (+0.39 without it), so mechanical overlap does not explain it. Attention responds to the signed return only in the test window (+0.10 sd per percentage point, [+0.03, +0.17]); development and validation show about zero, and the response to the size of the return is about zero throughout.
- **Oil.** No consistent response of attention to returns: about +0.01 and not significant in validation and test. The negative lead correlations in development are what a positive contemporaneous link combined with the negative shock autocorrelation would produce.
- **Reading.** Consistent with gold attention following gold rallies in 2022-2024, but not established: one window of three, among dozens of unadjusted post-hoc estimates. The pilot's gold results are unchanged (no contemporaneous relation, no forecasting power). If this is pursued, the pre-specified check should include Z(w).