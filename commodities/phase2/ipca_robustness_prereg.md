# Post hoc robustness of the IPCA cross-asset result: specification for review (nothing below has been run)

**Status.** Post hoc robustness. The 21-month holdout was examined in the original analysis (`ipca_prereg.md`, results in `ipca_results.md`), so these checks are **not independent confirmation**.
The original primary result (5-day shock window, FinBERT-20, K = 3) and the original secondary (5-day, NoLBERT-20, K = 1) **stay primary and secondary** and are never replaced. No factor count, penalty, threshold, asset or
sample is chosen from returns. No model search, no trading strategy.

## 1. Verified facts that the specification rests on

| question | finding | source |
|---|---|---|
| What is `_s63`? | A 63-day attention-shock window, chosen on validation for monthly rebalancing (weekly uses 1 day, `_s1`; biweekly 5 days, no suffix). | Shivesh, message of 8 Oct 2026 |
| Do 63-day features use only news available before each formation date? | Yes by construction: shocks are `theta - theta.rolling(window).mean().shift(1)`, a strictly past trailing mean for any window; exposures use a trailing window ending at the formation date. The `_s63` feature blocks themselves are not saved, so this is verified in the code snapshot, not on the saved s63 data (the snapshot predates the `_s63` option). | `ftm/features/narrative.py` |
| NoLBERT encoder dates | **Resolved.** Pretrained on 1976-1995 text only. Adapted with an InfoNCE headline-lead objective (no returns) on pairs drawn **up to 2020-09-01**, the training end of an earlier pipeline, not this repo's 2019-12-31. Eight months of the validation period (January-August 2020) therefore entered the adaptation, "accepted knowingly; it affects validation-period numbers, not the test period". The team's documented remedy is re-adaptation with `--train-end 2019-12-31`; not done. | `NOLBERT_RESULTS.md` (pair cutoff, config table), `HANDOFF.md` item 5 |
| Vocabulary | The 10,000-term vocabulary basis comes from the **whole corpus** for every representation (centroids, scaler and fitting are training-only). Affects LDA inputs and topic descriptions, not the NoLBERT clustering. | `NOLBERT_RESULTS.md`; `HANDOFF.md` item 4 |
| FinBERT classes | Classifier published December 2022, fine-tuned on tweets released September 2022: not point-in-time. | `REFERENCE.md` |

### Point-in-time status of each representation for the **test period (2022-2024)**
| representation | pretraining / training data | compatible with the test period? | caveats |
|---|---|---|---|
| NoLBERT-20 | 1976-1995 text; adapted on our news to 2020-09-01 | **yes** | eight validation months seen by the encoder; whole-corpus vocabulary basis |
| LDA (60 topics) | fit on 140,209 training articles (through 2019-12-31) | **yes** | whole-corpus vocabulary basis |
| MiniLM | off-the-shelf encoder released 2021 | test period yes, training period no | not used here |
| FinBERT-20 | classifier December 2022; tweets September 2022 | **no** | the original primary |

## 2. Eligible model inventory (existing saved specifications only; nothing constructed or tuned)

Rule fixed before running: narrative-only (`narr_`), K = 3, 63-day window (`_s63`), and a representation that is either (i) compatible with the test period, or (ii) the original primary's representation, kept as a same-window comparator.

| specification | K | penalty | kept / total | role |
|---|---|---|---|---|
| `narr_nolbert20_s63__sipca` | 3 | 0.03 | 20/20 | robustness, point-in-time-compatible |
| `narr_lda_all_s63__sipca` | 3 | 0.30 | 60/60 | robustness, point-in-time-compatible (strictly training-fit) |
| `narr_finbert_s63__sipca` | 3 | 0.03 | 20/20 | comparator: same representation as the original primary, 63-day window |

Not eligible (and why): `narr_minilm_s63` (K = 3, but an off-the-shelf encoder that is neither point-in-time for the training period nor the original primary); `narr_lda_s63` and `narr_nolbert_s63` (K = 1); every `text_`, `all_`, `firm_`, `price_` and `tone_` block (not narrative-only); the 5-day base K = 3 FinBERT is the original primary and is not re-run.

At K = 3 and a 63-day window, **NoLBERT-20 and FinBERT-20 are directly comparable** (same K, same window, same 20 topics), which removes the K confound of the original NoLBERT-versus-FinBERT contrast. The LDA model adds a point-in-time representation with 60 topics.
All three factor files cover the same 104 months as the panel and mark test months out-of-sample; each passes the team's own forecast-reproduction check (about 1e-10). They could not be independently rebuilt here because their feature blocks are not saved.

## 3. Analysis A: 63-day robustness (post hoc)

Identical to the original analysis: same asset returns, same equity-market benchmark, same months and alignment (`ipca_monthly_panel.csv`), same estimation (benchmark and factor model fitted on training + validation months only, applied unchanged to the 21 test months), same statistics.

| item | setting |
|---|---|
| assets | CL, GC, HG, BTC, equity market (as before) |
| fit months | 59 for CL, GC, HG; 28 for BTC and for the common-month version; 83 for the equity market |
| test months | 21 |
| outputs per specification and asset | held-out gain with paired-bootstrap 95% interval; one-sided analytic Clark-West p; simulation-calibrated Clark-West p (wild bootstrap, 5,000 draws, seed 0); common-month versions; descriptive all-months increment in adjusted R2 and joint test (HC3) labelled descriptive |
| multiple testing | Holm across the four assets within each specification; and, as the conservative post hoc family, Holm across all 12 (3 specifications x 4 assets). Original primary inference is not re-adjusted. |
| factor-loading interpretation | top loadings per factor reported, with the usual caution about latent factors |

Reporting rules fixed in advance: the table shows original (5-day) and 63-day results **side by side** for every specification, improvements and deteriorations alike. A significant 63-day result does not replace the non-significant original one; it is labelled post hoc with its own family. A null 63-day result is reported with the same prominence. The paper's sentence on the crude-oil result is updated to say whether it persists with a historically valid representation and the 63-day window, and, either way, that the holdout was not untouched.

## 4. Analysis B: 2024 data-quality sensitivity (descriptive)

Applied to the **original** primary (FinBERT-20, 5-day, K = 3) and original secondary (NoLBERT-20, 5-day, K = 1), with their frozen coefficients unchanged.
Fixed exclusion rule, chosen from the documented data problem and not from returns: drop test months whose formation date is on or after 2023-12-01. Reason: the source has no articles from 2023-12-17 until 2024-01-08 (one dump day removed), 22 articles in January 2024, and about 4,500 articles of 2024 have headlines only.

| item | setting |
|---|---|
| excluded test months | 7: formation 2023-12-29, 2024-06-28, 2024-07-31, 2024-08-30, 2024-09-30, 2024-10-31, 2024-11-29 |
| remaining test months | **14** (formation 2021-12-31 to 2023-11-30); fit months unchanged |
| outputs | held-out gain with bootstrap interval, analytic and calibrated Clark-West p, Holm over four assets, side by side with the full 21-month result |
| power | with 14 months the minimum detectable effect is larger still; every interval will be wider. The paper says so and does not read a null as evidence of no effect. |

## 5. Documentation updates that do not depend on any result (made now)
Encoder adaptation dates resolved with their source; whole-corpus vocabulary basis disclosed; the 5-day versus 63-day mismatch and the 2024 source change retained as limitations.

## 6. Not done
New factor counts, penalties, thresholds or assets; any 63-day equity-forecast rebuild (needs the team's newer code); re-adapting the encoder with the 2019-12-31 cutoff (about 150 GPU-minutes in the team's log; the documented clean fix); removing headline-only 2024 articles (article-level assignments not saved); any additional model search or trading rule.

## 7. Deviations
None yet.
