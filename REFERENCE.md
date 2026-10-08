# Reference

How every number in this repository is produced: data sources, sample, each
step of the pipeline with its parameters, the models, the portfolio rules and
the statistics. The values below are the ones behind `FINDINGS.md`. They are
set in three places: dates, sample sizes, windows and costs in `ftm/config.py`;
the hyperparameters of a model in the `defaults` and `grid()` of its class in
`ftm/models/`; the constants of the report (the rule behind one-number
summaries, 5,000 bootstrap resamples, the two halves of the test period) at
the top of `ftm/evaluation/report.py`.

## 1. Data sources

| Data | Source | Licence | Used for |
|---|---|---|---|
| News 2009 to 2023 | FNSPID, Dong, Fan and Peng (2024), `Zihan1004/FNSPID` on Hugging Face: `nasdaq_exteral_data.csv` (23 GB), Nasdaq.com articles with bodies | CC BY-NC 4.0 | **The article text** of 2012 to 2023; audit |
| News 2009 to 2025 | FinMultiTime, `Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting`: `text/sp500_news.zip` (3.6 GB, 4,694 per-ticker files) | MIT | **The article list**, and the text of 2024 |
| Returns | CRSP daily and monthly stock files (WRDS) | Licensed, not redistributable | Universe, returns, price features |
| Factors | Kenneth French's data library, daily three factors and risk-free rate | Free | Excess returns, alphas, market-neutral exposures |

FinMultiTime lists the FNSPID Nasdaq articles for 2009 to 2023 (the article
addresses agree to within rounding) and adds articles from a later scrape for
2024 and 2025. We build the article table from it, so that one list of
articles covers every year. Both datasets file an article once per ticker
page; we merge rows by address and then by (date, headline).

**FinMultiTime's text cannot be used as it is.** The files of 93 tickers
(alphabetically A to AFG, among them AAPL, ABBV, ABT, ADBE) carry the right
date, headline and address but the body of an unrelated article, often one
written years later. For 7.3% of the articles of 2012 to 2023 the text differs
from FNSPID's (83,049 articles; `results/data_audit/text_repair.csv`), in most
cases because of such a body. FNSPID has the right
bodies, so every article's text is taken from FNSPID where FNSPID has the
address. For 2024, which FNSPID does not cover, the text comes from an intact
ticker file where the article has one (2,755 articles) and is otherwise left
empty, so that the article is represented by its headline (4,513 articles).
Both extractions are therefore needed to rebuild `data/news/` from scratch.

What we did not use: FNSPID's second file (headlines of other publishers,
without bodies); the news after 2024, because returns end in December 2024.

## 2. Sample

| | |
|---|---|
| News | 1 January 2012 to 31 December 2024; articles whose body has at least 400 characters (as recorded by FinMultiTime) |
| Unique articles | 1,255,289 (2,490,572 article-ticker pairs) |
| Left out | 16 December 2023 (a dump of tens of thousands of rows on one day). The source then has no articles until 8 January 2024 and 22 in January 2024 |
| Text unit | Headline plus the first 500 characters of the body |
| Universe | The 1,000 largest common stocks by market value at the end of June, used from the last trading day of that June until the next list |
| Returns | 3 January 2011 to 31 December 2024 (2011 and 2012 only feed trailing windows) |
| Stocks | 2,020 that are in at least one June list, 1,826 with a ticker |

**Splits** (a holding period belongs to the split of the day its return starts):

| Split | Dates | Monthly | Biweekly | Weekly | Role |
|---|---|---|---|---|---|
| Training | 2013 to 2019 | 69 | 143 | 290 | Fit models; fit topic models and vocabulary (text of 2012 to 2019) |
| Validation | 2020 to 2021 | 14 | 27 | 53 | Choose hyperparameters; choose between experiments |
| Test | 2022 to 2024 | 21 | 42 | 81 | Reported with the settings chosen on validation; also split into 2022-23 and 2024 |

The counts are periods. **From 2017 only part of each year is used**: our
return file has daily returns for every stock up to 2016, and from 2017 only
for the stocks of each calendar year's June list. Between January and the day
the list changes, the stocks that have data are exactly those that will still
be among the 1,000 largest in June, so any feature observed in those months is
observed for a sample selected on the future. Formation days in that part of
the year are left out (`DAILY_RETURNS_COMPLETE_UNTIL = 2016`,
`Market.complete`). What remains from 2017 on is the formation days from the
end of June to the end of December, that is the holding periods July to
January. A complete daily file (`scripts/00_pull_crsp.py`) restores the full
years: 84, 24 and 36 months.

Panel sizes: 102,457 stock-months, 208,678 stock-fortnights, 417,373
stock-weeks; about 987 stocks per period (a listed stock must have a return
in the last five trading days).

## 3. Pipeline

Run the scripts in order. Times are for a 4-core laptop CPU.

| Step | Script | Output | Time |
|---|---|---|---|
| 0 | `00_prepare_crsp.py` or `00_pull_crsp.py` | `data/crsp/` | 1 min |
| 1 | `01_download_news.py` | Extraction shards in `~/.cache/ftm` | 15 min, 3.6 GB download |
| 2 | `02_build_articles.py` | `data/news/`, `results/data_audit/news_coverage.csv` | 2 min |
| 3 | `03_embed.py` | `data/topics/embeddings/` | 45 min on 2 cores (200 articles a second) |
| 3b | `03b_classify_topics.py` (optional) | FinBERT class probabilities | 50 min on 2 cores |
| 3c | `03c_finetune_encoder.py` (optional, GPU) | An adapted encoder in `data/topics/encoders/` | 150 min on one RTX 3060 (teammate's run) |
| 4 | `04_fit_topics.py` | `data/topics/<model>/` | LDA 10 min, applying it to every article 5 min, embedding clusters 3 min, classifier weights 1 min (as recorded in each `quality.json`) |
| 5 | `05_build_panel.py` | `data/panels/<freq>/` | 2 min for all blocks and frequencies |
| 6 | `06_run_models.py` | `data/predictions/`, `data/models/`, `results/tuning/`, `results/importance/` | about 70 min for all 1,092 experiments (26 feature sets, 14 models) |
| 6b | `06b_ensemble.py` | `data/predictions/<freq>/<features>__ens.*` | under 1 min |
| 7 | `07_evaluate.py` | `results/series/` | 2 min |
| 8 | `08_report.py` | `results/tables/` | under 1 min |
| 9 | `09_explain.py` | `results/explain/` | under 1 min |
| 10 | `10_audit.py` | `results/data_audit/firm_coverage.csv`, `stale_rows.csv`, `donor_rows.csv` | under 1 min |
| 11 | `11_missing_data_leak.py` | `results/data_audit/missing_data_leak.csv` | 1 min |
| 12 | `12_tuning_noise.py` | `results/data_audit/tuning_noise.csv` | 3 min for XGBoost on two feature sets; EBMs take 1 min a seed |

### 3.1 Articles (`ftm/data/`)

- `finmultitime.py` reads the archive ticker file by ticker file and keeps, per
  row: date, ticker, headline, hashes of the address and of (date, headline),
  body length and the first 500 characters of the body. Rows of 2024 and 2025
  have no headline in the source; it is rebuilt from the address.
- `fnspid.py` does the same for the 23 GB FNSPID file by HTTP range requests,
  without storing the file (the source of the text of 2012 to 2023).
- `news.py` merges rows into unique articles and writes `articles/` (one file
  per quarter) and `links.parquet`.
- Article dates carry no usable time of day (98.9% are stamped 00:00:00), so
  **news of day d is never used before the close of day d+1**
  (`NEWS_LAG_DAYS = 1`).

### 3.2 Topic models (`ftm/topics/`)

All models see the same articles and are fitted on training-period text only.

| Setting | Value |
|---|---|
| Attention sample | At most 150 random articles per calendar day (529,657 articles) |
| Fit sample | At most 60 of those per day, dated up to 31 December 2019 (140,209 articles) |
| Vocabulary | 10,000 unigrams and bigrams of 3+ letters, in at least 20 and at most 40% of the fit articles, English stop words plus 38 source-specific words removed |
| Topics | 60 |
| Seed | 0 (stability uses seed 1 as the second fit) |

- **`lda`**: latent Dirichlet allocation by collapsed Gibbs sampling (tomotopy,
  400 iterations, package defaults for the priors). Article-topic proportions
  of all articles are then inferred with the topic-word matrix held fixed
  (scikit-learn's variational E-step). `lda_all` applies the same topics to
  every article instead of the 150-a-day sample.
- **`minilm`**: the "BERT" topic model of the earlier project. Sentence
  embeddings from `sentence-transformers/paraphrase-MiniLM-L3-v2` (3 layers,
  384 dimensions), clustered by spherical k-means (3 starts, at most 100
  iterations). Article-topic weights are a softmax of the cosine similarity to
  each centroid at temperature 0.07. Unlike the earlier project it embeds one
  passage per article (headline and lead, first 64 tokens) instead of up to
  three, and its clusters are fitted on training-period articles only. The
  encoder itself was published in 2021: its knowledge of language predates the
  test period but not the training period.
- **`finbert`**: class probabilities of
  `nickmuchi/finbert-tone-finetuned-finance-topic-classification` (20 fixed
  subjects), run on headlines (first 32 tokens) from its author's ONNX export
  (`nickmuchi/optimum-finbert-tone-finetuned-finance-topic-classification`).
  Published in December 2022 and fine-tuned on a set of finance tweets
  released in September 2022: **not point-in-time**.
- **`nolbert`**: NoLBERT (Kakhbod and Li 2025, `alikLab/NoLBERT`, a DeBERTa-v3
  model of 109M parameters pretrained only on text from 1976 to 1995), adapted
  into a sentence encoder by our teammate (run of 8 October 2026,
  `results/encoders/nolbert_ft/train_log.json`): InfoNCE over two chunks of the
  same article with in-batch negatives, temperature 0.05, every parameter
  trained, learning rate 2e-5, batches of 32 pairs, 128 tokens, bf16, 150
  minutes on one RTX 3060, 6,205 steps; held-out in-batch top-1 retrieval rose
  from 0.08 to 0.89. Pairs were drawn from articles dated before 1 September
  2020, so **eight months of our validation-period text entered the encoder**
  (not the test period). A probe of pooling choices on 11,988 documents
  (`nolbert_probe.json`) chose the mean of the last layer without whitening
  (NPMI 0.343 against 0.268 to 0.282 for the alternatives). Articles (headline
  and lead, 64 tokens) are embedded with that pooling and clustered exactly as
  `minilm` is: 60 topics, spherical k-means, softmax weights at temperature
  0.07. `scripts/03c_finetune_encoder.py` now carries the same recipe, with the
  pair cutoff at the end of our training period by default.
- **`nolbert20`**: the same NoLBERT article vectors clustered into 20 topics
  instead of 60 (added in this repository, `04_fit_topics.py embedding
  --embeddings nolbert_ft_d150 --name nolbert20 --topics 20`). With three times
  as many articles per topic and day, its attention shocks are more reliable
  (split-half 0.47 to 2019 and 0.50 after, against 0.28 and 0.34 for 60 topics).
  Twenty is the number of FinBERT classes; it was fixed before any forecast was
  made with it.

Daily attention of a topic is the mean of the day's article weights. Quality
measures (`quality.py`): NPMI coherence of each topic's ten top terms, on
training text and on later text; diversity of top terms; stability across two
seeds (share of articles that stay in their matched topic, adjusted Rand
index); split-half reliability of the daily attention shock. Top terms are
chosen by class-based TF-IDF over each topic's articles for every model, so
that models are compared on one rule; for LDA the customary most-probable
words are reported as well.

### 3.3 Features (`ftm/features/`, `ftm/panel.py`)

One row per (formation day, investable stock). Every feature is turned into a
cross-sectional rank scaled to [-1, 1] among the stocks of that day. All
windows end at the formation day (prices) or one trading day before it (news).

The blocks are defined in one place, `ftm/features/blocks.py`: a block is a
function from a formation day to the raw features of every stock, NaN where
they cannot be computed.

**Missing values.** A stock with a missing value anywhere in a block receives
the complete feature row of another stock of the same day, drawn at random
(hot-deck imputation, `MISSING_FEATURES = "donor"`), before ranks are taken. In
our data a missing value can depend on the future (FINDINGS.md, section 2), so
a model must not be able to see it. The stock stays in the universe and in the
portfolios with its true return. The draws are seeded per block, so a panel is
reproducible. `results/data_audit/donor_rows.csv` counts the rows concerned:
2% to 7% for price features (from 2017, stocks new to the June list, whose
12-month momentum cannot be computed), 1% for narrative exposures, 39% to 51%
for firm-level news. The column `stale` of a panel's index marks stocks without
a daily return in the last 21 trading days; it is a diagnostic and never a
feature.

| Block | Features | Definition |
|---|---|---|
| `price` | 8 | `beta`, `ivol` (market model, last 252 days, at least 120 returns); `vol_21` and `max_21` (at least 15 returns); `mom_12_1` (days t-251 to t-21, at least 150 returns); `rev_21` (at least 15), `rev_5` (at least 3); `size` (log market value) |
| `narr_<m>` | one per topic | Narrative exposure: exponentially weighted covariance (252 days, half-life 60, at least 120 returns) of the stock's daily return with the topic's attention shock. The shock is attention minus its mean over the previous 5 trading days, made orthogonal to the market return inside the window. News of non-trading days counts on the next trading day; a day with fewer than 20 articles has no reading |
| `firm_<m>` | 1 + topics | The firm's own news over 21 trading days: log(1 + articles) and the share of them in each topic. Only articles filed under at most 3 tickers count. A firm with no article in the previous 126 trading days (`FIRM_COVERAGE_WINDOW`) is not covered by the source: its block is missing |
| `narr_<m>_placebo` | one per topic | Narrative exposure to the attention shocks displaced by 126 trading days (the shock of day d-126 is paired with the return of day d) |

Feature sets: `price`; `narr_<m>`; `firm_<m>`; `text_<m>` (both text blocks);
`all_<m>` (price and text).

Targets: `ret` (return of the holding period), `y` (minus the risk-free
return), `y_rel` (minus the cross-sectional mean of `y`). **Models are fitted
on `y_rel`.** A stock that stops trading during a holding period earns zero
from then on.

### 3.4 Models (`ftm/models/`)

Protocol (`scripts/06_run_models.py`): every setting of a model's grid is
fitted on the training period and scored on the validation period; the first
setting, in order from most to least regularised, whose validation mean squared
error is within 0.001% of the best is kept (`TUNE_TOLERANCE = 1e-5`, that is
0.001 percentage points of R2; a first run used 1e-4, which is the size of the
validation R2s themselves and so nearly always returned the first setting). The model
is then refitted on training plus validation and forecasts the test period
(`fit_for_test`; the EBM is the exception, see below). There is no
re-estimation inside the test period.

| Name | Model | Settings tried |
|---|---|---|
| `ols` | Least squares | none |
| `ridge` | Ridge | 25 penalties from 100 to 0.0001 times the mean feature variance |
| `lasso` | Lasso | 40 penalties from the smallest that sets every slope to zero down to a thousandth of it |
| `ipca` | Instrumented PCA (Kelly, Pruitt and Su, 2019). Forecast: features times loadings times the mean factor return | 1, 2, 3, 5 factors |
| `sipca` | Sparse IPCA, the estimator of the earlier project: the loadings of a feature are penalised as a group, so whole features drop out; factors are ridge cross-sections | 1 or 3 factors; penalty 0.3, 0.1 or 0.03 of the value that switches every feature off |
| `xgb` | XGBoost: histogram trees, learning rate 0.05, at most 600 rounds stopped early on validation (patience 50), half of the rows and of the columns per tree, at least 50 observations per leaf, L2 penalty 1, 64 histogram bins; refitted with the number of rounds that validation kept | depth 1, 2, 3, 5 |
| `rf` | Random forest: 100 trees, each on at most 30,000 sampled rows, square root of the number of features per split, at least 50 observations per leaf | depth 3, 6, 10 |
| `ebm` | Explainable boosting machine (interpret 0.7.8), the model of Bell, Kakhbod, Lettau and Nazemi (2024): additive step functions of each feature plus optional pairs. 64 bins (16 per feature in a pair), learning rate 0.02, trees of two leaves with at least 20 observations each, 50 smoothing rounds, at most 2,000 rounds per stage stopped early on validation (patience 50), 4 bags that each leave out a random 20% of the training rows and hold at most 100,000 | 0 or 10 pairwise interactions |

The linear models are solved on second moments (`linear.py`), so a whole
regularisation path costs milliseconds. EBM bags stop early on our
chronological validation rows, never on a random split of the training data.
The elastic net is implemented but not run: with returns as the target its
penalty is tiny next to the variance of the features, its ridge part does
nothing, and it reproduced the lasso forecasts to six decimals. The earlier
project chose the sparse IPCA penalty by the Sharpe ratio of a factor portfolio
on held-out months and traded that portfolio; here `sipca` is tuned and traded
like every other model.

XGBoost and the EBM stop boosting where the validation error is lowest, so
their validation scores are flattered relative to the other models. The
selection rule of section 3.6 inherits this.

**The EBM is not refitted.** Its two stages stop at different rounds (a few
for the main effects of a noisy target, many more for the pairs), and the
library takes one maximum number of rounds for both. A refit on training plus
validation rows with a fixed number of rounds would not be the model that
validation chose. The model fitted on the training rows and stopped on the
validation rows is therefore the one that forecasts the test period
(`Ebm.refit = False`). It has seen the validation returns only through the
number of boosting steps.

Every experiment is run once, with seed 0. For the tree models that is one
draw: `scripts/12_tuning_noise.py` repeats an experiment with other seeds,
tuning included, and records how far the result moves (FINDINGS.md, section
7.1).

**Variants** (added with the NoLBERT run; FINDINGS.md, section 9):

| Name | What it is |
|---|---|
| `<model>_rk` | The model fitted on the cross-sectional rank of `y_rel`, scaled to [-1, 1] like the features (ties averaged), instead of `y_rel` itself, with the same grid and protocol. For `ols`, `ridge`, `lasso`, `ipca`, `sipca` and `xgb`; not for the forest and the EBM, which would add half an hour per run. Its forecasts are scores: R2 against returns means nothing for them, rank IC and portfolios do. `ftm.models.tuning.target_of` |
| `ens` | The combination of the eight base models: in every period each model's forecast is turned into ranks scaled to [-1, 1] and the eight are averaged with equal weights. Nothing is estimated or chosen. `scripts/06b_ensemble.py` |

The eight base models (`experiments.BASE_MODELS`) make up every group of the
landscape, contrast, halves, placebo and text-against-price tables, so these
tables are comparable with the 432-experiment run. The variants are models of
their own in the per-experiment tables and in the search, and
`improvements.csv` compares each with what it varies.

### 3.5 Portfolios (`ftm/portfolio/`)

Formed at each formation day's close from that day's forecasts, held to the
next formation day. Stocks tied in forecast are ordered by a fixed arbitrary
number per stock, drawn once per backtest from `SEED`: a forecast that does
not change does not trade, and one rule's result does not depend on which
other rules are evaluated with it. A forecast that is the same for every stock
leaves the long-short rules flat and the long-only rules in the universe.

| Name | Rule |
|---|---|
| `ls_ew`, `ls_vw` | Long the top decile, short the bottom decile; equal or market-value weights within each leg |
| `rank` | Every stock, weight proportional to its forecast rank minus the mean rank |
| `mvo` | Mean-variance weights for the forecasts with a 5-factor covariance of the last 252 daily returns; dollar neutral; at most 1% per stock |
| `long_ew`, `long_vw` | The top decile only, compared with the equal- or value-weighted universe |
| `ls_buf` | `ls_ew` with a buffer: a stock enters the long (short) leg in the top (bottom) decile and stays until it leaves the top (bottom) fifth (`BUFFER_QUANTILES = 5`), equal weights; the buy/hold spread of Novy-Marx and Velikov (2016). With nothing held it is `ls_ew`. Its legs vary in size |

Long-short rules have longs of +1 and shorts of -1. Turnover is half the sum of
absolute weight changes after last period's weights have drifted with returns;
the cost is 10 basis points per dollar traded (`COST_BPS`), including the
initial purchase. Each period (validation, test) starts from cash.

### 3.6 Statistics (`ftm/evaluation/`, `scripts/08_report.py`)

| Measure | Definition |
|---|---|
| R2 out of sample | 1 - sum((y - forecast)^2) / sum(y^2) on `y_rel`: against a forecast of zero, not against the sample mean. In the tables the two sums are sums over periods of per-period means, which is the same thing up to the small variation in the number of stocks |
| Rank IC | Spearman correlation of forecast and `y_rel` across stocks, per period; reported as mean, Newey-West t statistic, and mean / standard deviation times sqrt(periods per year). A period in which the forecast is the same for every stock counts as zero if the experiment forecasts in other periods; an experiment that never forecasts has no IC |
| Small samples | Newey-West t statistics are judged by the fixed-b critical values of Kiefer and Vogelsang (2005, Bartlett kernel), column `t_critical`: 2.39, 2.25 and 2.11 for the 21, 42 and 81 test periods. In simulations with these sample sizes the normal 1.96 rejects a true null 6% to 20% of the time |
| Clark-West | t statistic of the per-period mean of 2 y forecast (one-sided test that the model beats the zero forecast, which it nests) |
| Diebold-Mariano | t statistic of the difference in per-period mean squared error; positive favours the model |
| Sharpe ratio | Annualised mean over volatility of the per-period excess return; standard error of Mertens (2002); 95% interval from a stationary bootstrap (5,000 resamples) |
| Alpha | Intercept on the three Fama-French factors compounded over the holding period, Newey-West t statistic |
| Sharpe difference | Paired stationary bootstrap of two return series |
| Reality check, SPA | White (2000) and Hansen (2005) across the 375 experiments on real features of a frequency (all models, variants included), on returns scaled to unit variance, so that the statistic is the best Sharpe ratio |
| Expected best with no skill | The mean, over the bootstrap resamples of the reality check, of the best Sharpe ratio when every experiment's mean return is set to zero. It keeps the co-movement of the experiments, which the formula of Bailey and Lopez de Prado (2014) for independent trials does not |
| False discovery rate | Benjamini-Hochberg adjustment of the one-sided p-values (positive mean, fixed-b) of the searched experiments |
| Long-only portfolios | Also measured against the universe with the same weighting: active return, tracking error, information ratio, alpha and beta. In the multiple-testing table they are measured this way only, since against zero they would pass for earning the market's return |
| Groups of experiments | `landscape.csv`: by feature set, kind of feature, topic model (its text-only sets) and model. The experiments of a group forecast the same returns, so the t statistic is that of the group's average IC per period, and the Sharpe ratio that of the equal-weighted average of the group's long-short portfolios |
| Placebo | `placebo.csv`: IC of the narrative exposures minus IC of the placebo exposures, per period, averaged over the models with a forecast from both; Newey-West t statistic |
| Improvements | `improvements.csv`: rank target against return target (per model and feature set), ensemble against the average of its eight members (per feature set), `ls_buf` against `ls_ew` (per experiment). Mean of the per-period difference in IC and in net long-short return, averaged over the pairs, with Newey-West t statistics; median Sharpe ratios of both sides, median paired difference, share of pairs improved, turnover of both |
| Group against group | `contrasts.csv`: average IC of one group minus that of another, per period (topic model against topic model on text-only sets; trees and EBM against linear models; price and text against price) |
| Halves of the test period | `halves.csv`: a group's average IC in 2022-23 minus that in 2024, divided by the root of the sum of the two squared Newey-West standard errors |

Newey-West lags follow floor(4 (n/100)^(2/9)); bootstrap blocks have mean
length n^(1/3). Periods per year: 12, 26, 52.

**Selection rule.** For each frequency the experiment with the highest mean
rank IC in the validation period is "the" experiment; its test result is the
headline. The candidates are the experiments of the eight base models on real
features: 200 per frequency (25 feature sets) since the NoLBERT topic models
were added, 136 before. The placebo is a control and cannot be selected.
`selection.csv` applies the same rule within each family (price, text, price
and text) and within each topic model, and to two further pools: `any_variant`
(every experiment, variants included: 375 per frequency) and `ensemble` (the
25 ensembles alone). Every other test result is shown for completeness and must
be read with the multiple-testing rows, which count all 375 experiments of a
frequency.

## 4. Outputs

| File | Content |
|---|---|
| `results/data_audit/news_coverage.csv`, `text_repair.csv` | Articles, tickers and articles per weekday, by year; articles whose text was replaced |
| `results/data_audit/firm_coverage.csv`, `stale_rows.csv`, `donor_rows.csv` | Share of the universe with firm-level news; stocks without recent returns; rows with donor features |
| `results/data_audit/missing_data_leak.csv`, `tuning_noise.csv` | What the missing-data leak is worth; one experiment under ten seeds |
| `results/tuning/<freq>/<features>__<model>.{csv,json,log}` | Validation score of every setting; the chosen one; run log |
| `results/importance/<freq>/<features>__<model>.csv` | Coefficients, mean absolute contributions or gain shares of the final model |
| `results/series/<freq>_{forecast,returns}.parquet`, `<freq>_factors.csv` | Per-period series behind every table |
| `results/tables/forecast.csv`, `strategies.csv` | Every experiment: forecast accuracy; every portfolio rule |
| `results/tables/selection.csv`, `frequency.csv` | The experiments chosen on validation, and their portfolios |
| `results/tables/landscape.csv`, `contrasts.csv`, `halves.csv` | Groups of experiments; group against group; 2022-23 against 2024 |
| `results/tables/improvements.csv` | Rank targets, ensembles and the buffered portfolio against what they vary |
| `results/encoders/nolbert_ft/` | The run record of the NoLBERT adaptation (configuration, training log, pooling probe); the weights (434 MB) are not in git |
| `results/tables/placebo.csv`, `text_vs_price.csv` | Exposures against the placebo; text against price |
| `results/tables/multiple_testing.csv`, `tuning.csv`, `topic_models.csv` | The best of all experiments, given the search; what tuning chose; topic quality |
| `results/explain/` | Importance, shape functions and ablations of the chosen EBM |

## 5. Differences from the previous project

| | Previous | Here |
|---|---|---|
| Years | 2015 to 2023; 61 training and 41 test months, no validation set | 2013 to 2024; 69 training, 14 validation and 21 test months (84, 24 and 36 with a complete return file) |
| Universe | 1,000 largest by June value of the same calendar year, used from January | The June list is used from the end of June |
| Topic models | Fitted on the text of the whole sample | Fitted on text up to 2019, applied forward |
| News timing | Same-day | Lagged one trading day |
| Articles per day | 120 | 150 (and every article, as a check) |
| Model | Sparse IPCA with tuning on the out-of-sample months | Eight models under one validation protocol, six of them also fitted on ranked returns, and their ensemble |
| Portfolio | Tangency portfolio of the factors, in-sample scaling | Sorted, ranked, mean-variance and long-only portfolios with costs |
| Missing data | Visible to the model | Incomplete months dropped; remaining gaps hidden from the model (sections 2 and 3.3) |
| Articles | FNSPID | FinMultiTime's list, which adds 2024, with the text of FNSPID (section 1) |
| Inference | Sharpe ratio | Standard errors, bootstrap intervals, multiple-testing corrections |

## 6. Known limitations

- Our CRSP extract has no daily returns from January to June for stocks that
  dropped out of the next June list (about 110 a year from 2017), which is why
  those months are not used from 2017 on (section 2). One trace remains: a
  holding period that starts in December and ends in January uses, for those
  stocks, the January return of the monthly file spread evenly over the
  month's days. `00_pull_crsp.py` removes the problem for anyone with WRDS
  access.
- The June lists from 2014 on are those of the earlier project's file. We
  could check them against complete data only for 2014 to 2016 (962 to 971 of
  1,000 stocks agree).
- The news source holds only tickers that were listed when it was collected,
  only the latest articles of the most covered tickers, and no articles for
  some large companies. Between 43% (2013) and 72% (2023) of the universe has
  an article in the previous year; 39% to 51% of the stock-periods get donor
  features in the firm-news block. The covered firms are themselves a selected
  sample (they survived to the collection date), which the donor rule cannot
  undo: firm-level results are indicative at best.
- From 2017, a stock new to the June list has daily returns in our file only
  from January of that year, so its 12-month momentum is missing and it gets
  donor price features until the momentum window holds 150 returns, from
  about September (about 5% of stock-periods).
- The test period has been looked at three times (FINDINGS.md, section 1).
  The changes made after each run answered defects of the data and of the
  method, not results, but they were made with a test period in view.
- The audits of scripts 10 to 12 were not repeated after the last changes (EBM
  protocol, tie rule); see FINDINGS.md, section 1.
- One seed per experiment. For tree models on many features the seed moves a
  result as much as sampling error does (FINDINGS.md, section 7.1).
- The MiniLM encoder was published in 2021 and the FinBERT classifier at the
  end of 2022: neither is point-in-time. NoLBERT's pretraining is (1976 to
  1995), but its adaptation used articles up to 1 September 2020, eight months
  into our validation period; and it was adapted on our corpus while MiniLM
  was not, so the two encoders differ in two ways at once.
- Tickers are matched to stocks through each stock's latest ticker.
- The 2024 news comes from a different scrape: 22 articles before February,
  fewer tickers (1,769 against 4,347 in 2023), headlines rebuilt from the
  article addresses, 24 weekdays with fewer than 20 articles.
- Further limits of the evaluation (one small validation period for three
  jobs, tuning on squared error and selection on rank IC, no adjustment of the
  IC for beta, flat costs) are listed in FINDINGS.md, section 9.
- Portfolios are formed at the closing prices that end the price signals, so
  short-term reversal includes bid-ask bounce.
- The topic models are not refitted after 2019, so subjects that appeared
  later (the pandemic) have no topic of their own.
- CRSP returns exclude delisting returns.
