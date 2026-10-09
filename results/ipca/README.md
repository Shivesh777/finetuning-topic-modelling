# IPCA loadings and factor returns

Written by `scripts/09b_ipca_factors.py`: for every IPCA experiment, the model that forecast the
test period, refitted exactly as `scripts/06_run_models.py` fitted it (checked against the stored
test forecasts). One folder per experiment: `<freq>/<feature set>__<model>/`.

| File | Content |
|---|---|
| `gamma.csv` | Gamma: one row per feature (with its block), one column per factor; `kept` = whether the group penalty of sparse IPCA kept the feature (a dropped feature has a row of zeros) |
| `factors.csv` | Factor return of every holding period: in sample (`estimate = in_sample`) for the training and validation periods, out of sample with Gamma fixed for the test periods |
| `summary.json` | Feature blocks, attention-shock window, chosen setting, periods, lambda, implied coefficients, normalisation, schedule |

## Names

`<model>`: `sipca` (sparse IPCA), `ipca` (plain IPCA); `_rk` = fitted on ranked returns.

`<feature set>`:

| Name | Blocks |
|---|---|
| `price` | the eight price features |
| `narr_<m>` | narrative exposures to the attention shocks of topic model `<m>` |
| `firm_<m>` | the firm's own news under topic model `<m>` |
| `text_<m>` | `narr_<m>` and `firm_<m>` |
| `all_<m>` | `price`, `narr_<m>` and `firm_<m>` |
| `..._s63` | **the same set with the narrative block built on 63-day attention shocks instead of 5-day ones.** The shock of a topic is its attention minus its mean over the previous w trading days; the main run used w = 5, and the validation period chose w = 63 for monthly rebalancing (w = 1, suffix `_s1`, for weekly; biweekly kept 5). Price and firm-news blocks, periods, grid, seed and protocol are the same as in the set without the suffix; the number of factors and the penalty are chosen on validation for each set separately, so they can differ (REFERENCE.md, section 3.3; FINDINGS.md, sections 10.1 and 10.3) |
| `tone_fb`, `tone_lm`, `emb_fb`, `news_fb`, `px_tone`, `px_news`, `px_all` | the directional news blocks of FINDINGS.md section 10 (REFERENCE.md, section 3.3) |

`<m>` is a topic model: `lda`, `lda_all`, `minilm`, `finbert`, `nolbert`, `nolbert20`.
