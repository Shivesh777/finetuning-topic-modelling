# What the IPCA factors explain in oil, gold, copper and Bitcoin

Based on `ipca_results.md` (specification `ipca_prereg.md`, SHA-256 `b13b9f41...`). Primary model `narr_finbert__sipca` (K = 3), secondary `narr_nolbert20__sipca` (K = 1).
Everything here is **contemporaneous explanation**: the factors are estimated from same-month stock returns, so a link to same-month futures returns shows common exposure,
not predictability and not a priced risk premium. Nothing here is a tradable signal.

## Do the factors explain anything beyond the equity market?

| asset | held-out gain over the equity market (21 test months, coefficients fitted on earlier months) | Holm p (4 assets) | descriptive increment in adj R2, all months |
|---|---|---|---|
| WTI crude | 0.31 [-0.02, 0.53] | 0.103 | 0.20 (p < 0.001, n = 80) |
| Copper | 0.13 [-0.06, 0.32] | 0.108 | 0.04 (p = 0.060) |
| Bitcoin | 0.04 [-0.19, 0.21] | 0.189 | 0.03 (p = 0.116, n = 49) |
| Gold | -0.09 [-0.42, 0.16] | 0.509 | 0.01 (p = 0.176) |
| Equity market itself | -0.09 [-0.32, 0.11] | n/a | 0.06 (p = 0.102, n = 104) |

(The Holm p-values are the one-sided Clark-West tests; the intervals are unadjusted bootstrap intervals.)

## What survives, and what does not
- **Crude oil is the one result that points the same way across the evidence.** With the primary factors, the squared error of explaining crude falls by about 31% in the held-out months, and the all-months fit improves a lot (increment 0.20), driven by the third factor (standardised coefficient +0.46; its Bonferroni-adjusted interval over the 12 coefficients, [0.76, 4.43] in raw units, excludes zero, but that is a descriptive all-months estimate that includes months the loadings were fitted on). But in the 21 held-out months the bootstrap interval reaches zero and the Holm-adjusted p-value is 0.103. The effect does not clear the 0.05 bar after correction, and the common-month version is weaker (gain 0.23, Holm 0.22).
- **Copper** shows a small gain with both models (0.13 primary, 0.12 secondary), not significant after correction (0.108, 0.245).
- **Gold and Bitcoin: no evidence.** Held-out gains are zero or negative; Bitcoin's descriptive increments are positive but small and non-significant.
- **The secondary NoLBERT K = 1 factor does not reproduce the crude result** (held-out gain -0.27 for crude, -0.08 for gold). Representation and factor count differ together, so they cannot be separated here; the crude result depends on the FinBERT K = 3 factors.
- **The equity market itself is not explained** (adj R2 0.06; negative held-out gain), as expected: the factors are built from returns relative to the cross-sectional mean.
- **Power:** with 21 months, the smallest incremental R2 detectable with 80% power is 0.31 to 0.39 (alpha 0.05). Only large effects can be detected, so the small positive results for copper and Bitcoin neither confirm nor rule out modest relations.

## What the factors are, economically
Loadings of the primary model (`gamma.csv`, FinBERT classes):
- f1: Currencies (+0.41), Energy/Oil (+0.38), Gold/Metals (-0.36), M&A (-0.34): a commodity-and-currency narrative.
- f2: Treasuries/Corporate Debt (+0.63), General News (-0.43), Gold/Metals (-0.26): rates and credit.
- f3: Analyst Update (-0.53), Stock Movement (+0.43), Earnings (-0.35), Markets (+0.32): earnings and market-movement chatter.

The factor linked to crude oil, f3, is **not** an energy factor, and the factor that loads on Energy/Oil, f1, has an insignificant crude coefficient (-1.06, p 0.12). So the data do not support reading this as "oil news explains oil prices"; f3 looks more like a general risk or market-activity narrative, and this interpretation would need further work.

## Explanatory versus forecasting evidence
- Equities: the stock-level backtest of the same two specifications (monthly, buffered strategy, test months) gives a Sharpe of 0.11 gross and -0.04 after 10 bp for the primary model (validation 0.21), and -0.25 / -0.37 for the secondary (validation -0.80). No forecasting evidence.
- CL, GC, BTC: the weekly topic-shock forecasts do not beat the expanding mean. IPCA-factor forecasting of commodities was not tested.

## Limitations
21 held-out months; factors estimated from same-period stock returns; loadings fitted on train + validation, so all-months results are descriptive; copper has no independent return check; Bitcoin has 28 months for fitting;
FinBERT classes' training data are undocumented; months are non-contiguous and HC3 does not model dependence between neighbours; `_s63` variants excluded pending the team's answer.
