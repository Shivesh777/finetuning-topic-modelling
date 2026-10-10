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
| Universe | The 1,000 largest ordinary common shares (CRSP share codes 10 and 11, NYSE, AMEX and Nasdaq) by market value at the end of June, used from the last trading day of that June until the next list |
| Returns | 3 January 2011 to 31 December 2024 (2011 and 2012 only feed trailing windows) |
| Stocks | 2,062 that are in at least one June list (2010 to 2024), all with a ticker: from the earlier project's file for the 1,822 it covers, from the CRSP name history for the others |

**Splits** (a holding period belongs to the split of the day its return starts):

| Split | Dates | Monthly | Biweekly | Weekly | Role |
|---|---|---|---|---|---|
| Training | 2013 to 2019 | 84 | 182 | 365 | Fit models; fit topic models and vocabulary (text of 2012 to 2019) |
| Validation | 2020 to 2021 | 24 | 52 | 104 | Choose hyperparameters and the shock window; choose between experiments |
| Test | 2022 to 2024 | 36 | 78 | 156 | Reported with the settings chosen on validation; also split into 2022-23 and 2024 |

The counts are periods. **Every month of every year is used.** `data/crsp/`
comes from WRDS (`scripts/00_pull_crsp.py`): daily returns of every member of
every June list on every day, so `DAILY_RETURNS_COMPLETE_UNTIL = None`
(`ftm/config.py`; the environment variable `FTM_DAILY_COMPLETE_UNTIL=2016`
restores the behaviour for the earlier project's file, which from 2017 held
daily returns only for the stocks of each calendar year's June list and forced
the runs up to 8 October to drop January to June: 69, 14 and 21 months). On
the 4.13 million stock-days both files hold, returns and market values agree
exactly (FINDINGS.md, section 2.4). The earlier file is kept, outside git, in
`data/crsp_legacy/`.

Panel sizes: 141,761 stock-months, 306,746 stock-fortnights, 614,525
stock-weeks; 984 stocks per period (a listed stock must have a return in the
last five trading days).

## 3. Pipeline

Run the scripts in order. Times are for a 4-core laptop CPU.

| Step | Script | Output | Time |
|---|---|---|---|
| 0 | `00_pull_crsp.py --out data/crsp` (WRDS login, Duo; the final run) or `00_prepare_crsp.py` (the earlier project's files) | `data/crsp/` | 3 min |
| 1 | `01_download_news.py` | Extraction shards in `~/.cache/ftm` | 15 min, 3.6 GB download |
| 2 | `02_build_articles.py` | `data/news/`, `results/data_audit/news_coverage.csv` | 2 min |
| 3 | `03_embed.py` | `data/topics/embeddings/` | 45 min on 2 cores (200 articles a second) |
| 3b | `03b_classify_topics.py` (optional) | FinBERT class probabilities | 50 min on 2 cores |
| 3c | `03c_finetune_encoder.py` (optional, GPU) | An adapted encoder in `data/topics/encoders/` | 150 min on one RTX 3060 (teammate's run) |
| 3d | `03d_sentiment.py` | `data/sentiment/`: FinBERT tone and headline vectors, Loughran-McDonald counts | 90 min on 2 cores for 769,338 headlines (140 a second) |
| 4 | `04_fit_topics.py` | `data/topics/<model>/` | LDA 10 min, applying it to every article 5 min, embedding clusters 3 min, classifier weights 1 min (as recorded in each `quality.json`) |
| 5 | `05_build_panel.py` | `data/panels/<freq>/` | 2 min for all blocks and frequencies; `--tone` and `--shock-window W` add the blocks of section 3.3 |
| 5b | `05b_tune_shocks.py` | `results/tuning/shock_window.csv`; then `05_build_panel.py --shock-window 63` (the chosen window, every frequency) | 3 min |
| 6 | `06_run_models.py` | `data/predictions/`, `data/models/`, `results/tuning/`, `results/importance/` | about 2.5 hours on four cores for the 1,692 experiments of the final run (per frequency 26 sets x 14 models and 25 sets x 8 base models); the weekly EBMs on 212 features need about 3 GB of memory |
| 6b | `06b_ensemble.py` | `data/predictions/<freq>/<features>__ens.*` | under 1 min |
| 6c | `06c_signals.py` | Forecasts that fit nothing: single tone features, `combo`, `best10` | under 1 min |
| 6d | `06d_grand_ensemble.py` | Combinations across topic models: `grand_s5`, `grand_s63`, `grand_text_s5`, `grand_text_s63` (model `ens`) | under 1 min |
| 7 | `07_evaluate.py` | `results/series/` | 12 min |
| 8 | `08_report.py` | `results/tables/` | 6 min; `strategies.csv` needs one run per frequency (`--freq M --tables strategies`, ...), which keeps the other frequencies' rows |
| 9 | `09_explain.py` | `results/explain/` | under 1 min |
| 9b | `09b_ipca_factors.py` | `results/ipca/<freq>/<features>__<model>/`: Gamma, factor returns, settings | 2 min for the 50 monthly sparse IPCA experiments |
| 10 | `10_audit.py` | `results/data_audit/firm_coverage.csv`, `stale_rows.csv`, `donor_rows.csv` | under 1 min |
| 11 | `11_missing_data_leak.py` | `results/data_audit/missing_data_leak.csv` (computed on the earlier file with `FTM_DAILY_COMPLETE_UNTIL=2016`; with a complete file every period is complete and the audit has nothing to show) | 1 min |
| 12 | `12_tuning_noise.py` | `results/data_audit/tuning_noise.csv` | 2 min for XGBoost on two weekly feature sets and five seeds; weekly EBMs 80 s a seed |
| 13 | `13_sharpe_tests.py` | `results/tables/sharpe_tests.csv`: Sharpe ratios of the validated picks, the combinations across topic models and the price ensemble against zero (Newey-West t, probabilistic and deflated Sharpe) and against benchmarks (Ledoit-Wolf HAC and paired bootstrap) | under 10 s |

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

**Article tone** (`ftm/topics/sentiment.py`, `scripts/03d_sentiment.py`; not a
topic model). Scored for the 769,338 articles that firm-level features can use
(filed under at most 3 tickers, one of them a stock of our universe):

- **FinBERT tone**: `ProsusAI/finbert` (Araci 2019; revision `4556d13`), BERT
  further trained on Reuters financial news of 2008 to 2010 and fine-tuned on
  the Financial PhraseBank (Malo et al. 2014), sentences of company news that
  finance professionals labelled positive, negative or neutral. It saw no text
  after 2010 and no return: **point-in-time** for our sample. Applied to the
  headline (32 tokens), a single sentence like its training data. Tone =
  P(positive) - P(negative). It runs with ONNX Runtime with 8-bit weights
  except the second dense layer of each feed-forward block (which does not
  survive rounding): on 512 of our headlines this keeps a correlation of 0.99
  with the tone of the full model and the same label for 97%, at twice its
  speed (8 bits everywhere: 0.90 and 88%). Of the scored headlines, 21% are
  labelled positive, 14% negative and 64% neutral.
- **FinBERT headline vectors**: the mean of FinBERT's last hidden layer over the
  headline's tokens (768 dimensions), reduced to 64 principal components
  (`TONE_PCA_DIM`) fitted on the 333,973 articles of the training period, which
  keep 78% of the variance; the inputs of the embedding regression (section 3.3).
- **Loughran-McDonald counts**: words, positive words and negative words of the
  headline and lead, from the Loughran and McDonald (2011) dictionary as
  shipped with `pysentiment2` (2014 release: 354 positive, 2,355 negative
  words). Its tone correlates 0.23 with FinBERT's across articles.

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
0.3% to 1.2% for price features (stocks with too short a history), under 1%
for narrative exposures, 39% to 50% for firm-level news and tone. The column `stale` of a panel's index marks stocks without
a daily return in the last 21 trading days; it is a diagnostic and never a
feature.

| Block | Features | Definition |
|---|---|---|
| `price` | 8 | `beta`, `ivol` (market model, last 252 days, at least 120 returns); `vol_21` and `max_21` (at least 15 returns); `mom_12_1` (days t-251 to t-21, at least 150 returns); `rev_21` (at least 15), `rev_5` (at least 3); `size` (log market value) |
| `narr_<m>` | one per topic | Narrative exposure: exponentially weighted covariance (252 days, half-life 60, at least 120 returns) of the stock's daily return with the topic's attention shock. The shock is attention minus its mean over the previous 5 trading days, made orthogonal to the market return inside the window. News of non-trading days counts on the next trading day; a day with fewer than 20 articles has no reading |
| `firm_<m>` | 1 + topics | The firm's own news over 21 trading days: log(1 + articles) and the share of them in each topic. Only articles filed under at most 3 tickers count. A firm with no article in the previous 126 trading days (`FIRM_COVERAGE_WINDOW`) is not covered by the source: its block is missing |
| `narr_<m>_placebo` | one per topic | Narrative exposure to the attention shocks displaced by 126 trading days (the shock of day d-126 is paired with the return of day d) |
| `narr_<m>_s<w>` | one per topic | `narr_<m>` with the shock taken over the previous w trading days instead of 5: w = 63 at every frequency, the window chosen on validation by `05b_tune_shocks.py` (FINDINGS.md, section 5; on the earlier file the rule chose 1 day weekly and kept 5 biweekly) |
| `tone_fb` | 8 | FinBERT tone of the firm's headlines, averaged over its articles of the last 1, 5, 21 and 252 trading days (0 without news); surprise = 21-day minus 252-day tone; log(1 + articles) of the last 5 and 21 days; abnormal news = log(1 + 21-day count) - log(1 + 252-day count x 21/252) |
| `tone_lm` | 5 | Loughran-McDonald tone, (positive - negative words) / (positive + negative words) summed over the firm's articles of the last 1, 5, 21 and 252 days, and its surprise |
| `emb_fb` | 64 | Mean FinBERT headline vector (principal components) of the firm's articles of the last holding period: 5, 10 or 21 trading days (weekly, biweekly, monthly; zero without news): the embedding regression of Chen, Kelly and Xiu (2022), with our models as the forecaster |
| `npx` | 6 | 5- and 21-day returns of stocks with news in that window and of stocks without (each zero in the other column), and the 5- and 21-day returns times the FinBERT tone of the same window |

The four tone blocks (`ftm/features/tone.py`) use the coverage rule and timing of
`firm_<m>`: articles filed under at most 3 tickers, dated up to the day before
formation; a firm without an article in the previous 126 trading days is
missing (donor features).

Feature sets: `price`; `narr_<m>`; `firm_<m>`; `text_<m>` (both text blocks);
`all_<m>` (price and text); `narr_`, `text_` and `all_<m>_s<w>` (the same with the
tuned shock window, at every frequency); `tone_fb`, `tone_lm`, `emb_fb`;
`news_fb` (the three together); `px_tone` (price, both tones and `npx`);
`px_news` (`px_tone` and `emb_fb`); `px_all` (`px_news` with the NoLBERT
narrative block, tuned window where there is one, and firm-news block).

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
| `ebm` | Explainable boosting machine (interpret 0.7.8), the model of Bell, Kakhbod, Lettau and Nazemi (2024): additive step functions of each feature plus optional pairs. 64 bins (16 per feature in a pair), learning rate 0.02, trees of two leaves with at least 20 observations each, 50 smoothing rounds (every term once per round, in a shuffled order) and after them the library's greedy steps (the terms that gain most are boosted; greedy ratio 10, the default), at most 2,000 rounds per stage stopped early on validation (patience 50), 4 bags that each leave out a random 20% of the training rows and hold at most 100,000 | 0 or 10 pairwise interactions |

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

**IPCA loadings, factors and their normalisation.** The pipeline keeps only an
IPCA model's forecasts (and its implied coefficients Gamma lambda in
`results/importance/`). `scripts/09b_ipca_factors.py` refits the model that
forecast the test period, checks that it reproduces the stored test forecasts,
and writes `results/ipca/<freq>/<features>__<model>/`: `gamma.csv` (one row per
feature, one column per factor; for `sipca` whether the feature is kept),
`factors.csv` (the factor return of every holding period) and `summary.json`
(setting, periods, lambda, implied coefficients). Normalisation, as in Kelly,
Pruitt and Su (2019): Gamma'Gamma = I, factors orthogonal over the fit sample
(the sum of f_t f_t' diagonal) in decreasing order of second moment, each with
a positive mean. `ipca` is estimated under it. The objective of `sipca` (fit,
plus the sum of squared factors, plus the group penalty on the rows of Gamma)
does not change under an orthogonal rotation of the factors, so its Gamma and
factors are put into the same form after the fit; forecasts and kept features
do not change. `sipca` works with sums over stocks rather than means and
estimates each period's factors by a ridge cross-section,
f_t = (Gamma' S_t Gamma + I)^-1 Gamma' q_t, as the earlier project did.
Schedule (every model): the grid is fitted on the training periods and scored
on the validation periods; the chosen setting is refitted once on training plus
validation and applied unchanged to the test periods. The factors of the
training and validation periods are therefore in-sample estimates; those of the
test periods are out of sample: the same cross-section with Gamma fixed, a
portfolio whose weights are known at formation.

Every experiment is run once, with seed 0. For the tree models that is one
draw: `scripts/12_tuning_noise.py` repeats an experiment with other seeds,
tuning included, and records how far the result moves (FINDINGS.md, section
8.1).

**Variants** (added with the NoLBERT run; FINDINGS.md, sections 6 and 8):

| Name | What it is |
|---|---|
| `<model>_rk` | The model fitted on the cross-sectional rank of `y_rel`, scaled to [-1, 1] like the features (ties averaged), instead of `y_rel` itself, with the same grid and protocol. For `ols`, `ridge`, `lasso`, `ipca`, `sipca` and `xgb`; not for the forest and the EBM, which would add half an hour per run. Its forecasts are scores: R2 against returns means nothing for them, rank IC and portfolios do. `ftm.models.tuning.target_of` |
| `ens` | The combination of the eight base models: in every period each model's forecast is turned into ranks scaled to [-1, 1] and the eight are averaged with equal weights. Nothing is estimated or chosen. `scripts/06b_ensemble.py` |
| `grand_s<w>__ens`, `grand_text_s<w>__ens` | The combination across topic models: the within-period ranks of the six `ens` forecasts on `all_<m>` (or `text_<m>`) with shock window w, averaged with equal weights; w = 5 and 63. Kind `grand` (family "all" or "text"); not a candidate of the main run's pools, counted in the search. Added in the final run after its test results were seen (FINDINGS.md, section 6). `scripts/06d_grand_ensemble.py` |

**Forecasts that fit nothing** (`scripts/06c_signals.py`; FINDINGS.md, section 6):

| Name | What it is |
|---|---|
| `sig_*` | One feature as the forecast, with the sign that the literature predicts: FinBERT tone over 1, 5, 21 days and its surprise (`sig_fb1`, `sig_fb5`, `sig_fb21`, `sig_fbsurp`); the same for Loughran-McDonald (`sig_lm*`); abnormal news (`sig_abn`); minus the 5-day return of stocks without news (`sig_quietrev`) and plus that of stocks with news (`sig_newsdrift`, Chan 2003). Stored under `tone_fb`, `tone_lm` or `px_tone` |
| `combo` | Equal-weighted average of the within-period ranks of the price ensemble (`price`, `ens`) and of the tone signal with the best validation IC (`results/tuning/<freq>/signals.csv`). Stored as `px_tone__combo`. The primary test of the news round, traded with `ls_buf` |
| `best10` | Equal-weighted average of the ranks of the ten base-model experiments with the highest validation IC, any feature set (`results/tuning/<freq>/pool.csv`). Stored as `px_pool__best10` |

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
| `ls_buf_vw` | The legs of `ls_buf`, weighted by market value within each leg (added in the final run) |
| `tilt_vw` | Every stock of the universe at its market weight times (1 + s), s = the forecast rank scaled to [-1, 1] (highest forecast at twice, lowest at zero times its market weight): the value-weighted universe tilted towards the forecast, "enhanced indexing". Judged against `univ_vw` by information ratio and alpha |

Long-short rules have longs of +1 and shorts of -1. Turnover is half the sum of
absolute weight changes after last period's weights have drifted with returns;
the cost is 10 basis points per dollar traded (`COST_BPS`), including the
initial purchase. Each period (validation, test) starts from cash.

### 3.6 Statistics (`ftm/evaluation/`, `scripts/08_report.py`)

| Measure | Definition |
|---|---|
| R2 out of sample | 1 - sum((y - forecast)^2) / sum(y^2) on `y_rel`: against a forecast of zero, not against the sample mean. In the tables the two sums are sums over periods of per-period means, which is the same thing up to the small variation in the number of stocks |
| Rank IC | Spearman correlation of forecast and `y_rel` across stocks, per period; reported as mean, Newey-West t statistic, and mean / standard deviation times sqrt(periods per year). A period in which the forecast is the same for every stock counts as zero if the experiment forecasts in other periods; an experiment that never forecasts has no IC |
| Small samples | Newey-West t statistics are judged by the fixed-b critical values of Kiefer and Vogelsang (2005, Bartlett kernel), column `t_critical`: 2.29, 2.11 and 2.06 for the 36, 78 and 156 test periods. In simulations with these sample sizes the normal 1.96 rejects a true null 6% to 20% of the time |
| Clark-West | t statistic of the per-period mean of 2 y forecast (one-sided test that the model beats the zero forecast, which it nests) |
| Diebold-Mariano | t statistic of the difference in per-period mean squared error; positive favours the model |
| Sharpe ratio | Annualised mean over volatility of the per-period excess return; standard error of Mertens (2002); 95% interval from a stationary bootstrap (5,000 resamples) |
| Alpha | Intercept on the three Fama-French factors compounded over the holding period, Newey-West t statistic |
| Sharpe difference | Paired stationary bootstrap of two return series; and the test of Ledoit and Wolf (2008): delta method on the first two moments of both series with a Bartlett HAC covariance, fixed-b p-value (`inference.sharpe_difference_hac`, `sharpe_tests.csv`) |
| Deflated Sharpe ratio | Probability that the true Sharpe ratio exceeds the best expected from N unskilled trials with the observed dispersion of Sharpe ratios (Bailey and Lopez de Prado 2014); N = the experiments of the frequency that hold a position under the rule (`sharpe_tests.csv`) |
| Reality check, SPA | White (2000) and Hansen (2005) across the experiments on real features of a frequency (all models, variants included), on returns scaled to unit variance, so that the statistic is the best Sharpe ratio. Pool `main`: the 375 experiments of the main design; pool `all`: every experiment (617 a frequency in the final run) |
| Expected best with no skill | The mean, over the bootstrap resamples of the reality check, of the best Sharpe ratio when every experiment's mean return is set to zero. It keeps the co-movement of the experiments, which the formula of Bailey and Lopez de Prado (2014) for independent trials does not |
| False discovery rate | Benjamini-Hochberg adjustment of the one-sided p-values (positive mean, fixed-b) of the searched experiments |
| Long-only portfolios | Also measured against the universe with the same weighting: active return, tracking error, information ratio, alpha and beta. In the multiple-testing table they are measured this way only, since against zero they would pass for earning the market's return |
| Groups of experiments | `landscape.csv`: by feature set, kind of feature, topic model (its text-only sets) and model. The experiments of a group forecast the same returns, so the t statistic is that of the group's average IC per period, and the Sharpe ratio that of the equal-weighted average of the group's long-short portfolios |
| Placebo | `placebo.csv`: IC of the narrative exposures minus IC of the placebo exposures, per period, averaged over the models with a forecast from both; Newey-West t statistic |
| Improvements | `improvements.csv`: rank target against return target (per model and feature set), ensemble against the average of its eight members (per feature set), `ls_buf` against `ls_ew` (per experiment), the tuned shock window against the window of 5 days (per model and feature set, `narr_`, `text_` and `all_<m>_s<w>` against the same set without the suffix). Mean of the per-period difference in IC and in net long-short return, averaged over the pairs, with Newey-West t statistics; median Sharpe ratios of both sides, median paired difference, share of pairs improved, turnover of both |
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
and text) and within each topic model, and to further pools: `any_variant`
(every experiment, variants included), `ensemble` (the ensembles alone),
`news` (the base models on the directional news feature sets), `shock` (the
base models on the sets with a tuned shock window), `signal` (the
single-feature signals) and `any_all` (the base models on every feature set).
The pools of the main run (`any`, the families, the topic models,
`any_variant`, `ensemble`) hold its experiments only. `combo` and `best10` are
made of validation choices, so their validation IC is partly that of the
choice: they are not candidates of any pool and are reported on their own. Every other test result is shown for
completeness and must be read with the multiple-testing rows, which count every
experiment of a frequency. The counts of the search are in FINDINGS.md,
section 11.

## 4. Outputs

| File | Content |
|---|---|
| `results/data_audit/news_coverage.csv`, `text_repair.csv` | Articles, tickers and articles per weekday, by year; articles whose text was replaced |
| `results/data_audit/firm_coverage.csv`, `stale_rows.csv`, `donor_rows.csv` | Share of the universe with firm-level news; stocks without recent returns; rows with donor features |
| `results/data_audit/missing_data_leak.csv`, `tuning_noise.csv` | What the missing-data leak is worth; one experiment under ten seeds |
| `results/tuning/<freq>/<features>__<model>.{csv,json,log}` | Validation score of every setting; the chosen one; run log |
| `results/tuning/shock_window.csv` | Validation rank IC of the narrative exposures for every shock window, topic model and linear model |
| `results/tuning/<freq>/signals.csv`, `pool.csv` | Validation IC of every single-feature signal (and which one `combo` uses); of every candidate of `best10` |
| `data/sentiment/` (not in git but `finbert_pca.npz`) | Article ids, FinBERT probabilities and vectors, their principal components, Loughran-McDonald counts |
| `results/importance/<freq>/<features>__<model>.csv` | Coefficients, mean absolute contributions or gain shares of the final model |
| `results/series/<freq>_{forecast,returns}.parquet`, `<freq>_factors.csv` | Per-period series behind every table |
| `results/tables/forecast.csv`, `strategies.csv` | Every experiment: forecast accuracy; every portfolio rule |
| `results/tables/selection.csv`, `frequency.csv` | The experiments chosen on validation, and their portfolios |
| `results/tables/landscape.csv`, `contrasts.csv`, `halves.csv` | Groups of experiments; group against group; 2022-23 against 2024 |
| `results/tables/improvements.csv` | Rank targets, ensembles and the buffered portfolio against what they vary |
| `results/encoders/nolbert_ft/` | The run record of the NoLBERT adaptation (configuration, training log, pooling probe); the weights (434 MB) are not in git |
| `results/tables/placebo.csv`, `text_vs_price.csv` | Exposures against the placebo; text against price |
| `results/tables/multiple_testing.csv`, `tuning.csv`, `topic_models.csv` | The best of all experiments, given the search; what tuning chose; topic quality |
| `results/tables/sharpe_tests.csv` | Tests of single Sharpe ratios and of Sharpe differences (section 3.6) |
| `results/explain/` | Importance, shape functions and ablations of the chosen EBM |
| `results/ipca/<freq>/<features>__<model>/` | Gamma (`gamma.csv`), factor returns in and out of sample (`factors.csv`), setting and schedule (`summary.json`) of IPCA models; all 50 monthly sparse IPCA experiments are there |

## 5. Differences from the previous project

| | Previous | Here |
|---|---|---|
| Years | 2015 to 2023; 61 training and 41 test months, no validation set | 2013 to 2024; 84 training, 24 validation and 36 test months (69, 14 and 21 in the runs on the earlier file) |
| Universe | 1,000 largest by June value of the same calendar year, used from January | The June list is used from the end of June |
| Topic models | Fitted on the text of the whole sample | Fitted on text up to 2019, applied forward |
| News timing | Same-day | Lagged one trading day |
| Articles per day | 120 | 150 (and every article, as a check) |
| Model | Sparse IPCA with tuning on the out-of-sample months | Eight models under one validation protocol, six of them also fitted on ranked returns, and their ensemble |
| Portfolio | Tangency portfolio of the factors, in-sample scaling | Sorted, ranked, mean-variance and long-only portfolios with costs |
| Missing data | Visible to the model | A complete return file; remaining gaps hidden from the model (sections 2 and 3.3) |
| Articles | FNSPID | FinMultiTime's list, which adds 2024, with the text of FNSPID (section 1) |
| Inference | Sharpe ratio | Standard errors, bootstrap intervals, multiple-testing corrections |

## 6. Known limitations

- The universe is the CRSP June lists of ordinary common shares; the earlier
  project's lists differ in 0% to 3% of each year's stocks. Tickers are taken
  from the earlier file where it has the stock.
- The news source holds only tickers that were listed when it was collected,
  only the latest articles of the most covered tickers, and no articles for
  some large companies. Between 43% (2013) and 72% (2023) of the universe has
  an article in the previous year; 39% to 51% of the stock-periods get donor
  features in the firm-news block. The covered firms are themselves a selected
  sample (they survived to the collection date), which the donor rule cannot
  undo: firm-level results are indicative at best.
- The test period has been looked at in every run (four times). The changes
  made after each run answered defects of the data and of the method, not
  results, but they were made with a test period in view; the combinations
  across topic models were added after the final run's test results were seen.
- The audits of scripts 10 and 12 were rerun on the final data; script 11
  describes the earlier file.
- One seed per experiment. For tree models on many features the seed moves a
  result as much as sampling error does (FINDINGS.md, section 8.1).
- The MiniLM encoder was published in 2021 and the FinBERT classifier at the
  end of 2022: neither is point-in-time. NoLBERT's pretraining is (1976 to
  1995), but its adaptation used articles up to 1 September 2020, eight months
  into our validation period; and it was adapted on our corpus while MiniLM
  was not, so the two encoders differ in two ways at once.
- Tickers are matched to stocks through each stock's latest ticker.
- FinBERT reads headlines only (the lead would make it four times slower on a
  CPU), with 8-bit weights in most layers. The Loughran-McDonald counts use the
  2014 dictionary. Neither tone is adjusted for the source (Zacks, Motley Fool
  and RTTNews write very different headlines).
- The 2024 news comes from a different scrape: 22 articles before February,
  fewer tickers (1,769 against 4,347 in 2023), headlines rebuilt from the
  article addresses, 24 weekdays with fewer than 20 articles.
- Further limits of the evaluation (one validation period for three jobs,
  tuning on squared error and selection on rank IC, no adjustment of the IC
  for beta, flat costs) are listed in FINDINGS.md, section 13.
- Portfolios are formed at the closing prices that end the price signals, so
  short-term reversal includes bid-ask bounce.
- The topic models are not refitted after 2019, so subjects that appeared
  later (the pandemic) have no topic of their own.
- CRSP returns exclude delisting returns.
