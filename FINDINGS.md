# Findings

Results of the final run of 10 October 2026, on a complete CRSP return file pulled
from WRDS (section 2.4). Per rebalancing frequency (monthly, biweekly, weekly):

- 26 feature sets of the main design (price; for each of six topic models the
  narrative exposures, the firm's own news, both, and both with price; an LDA
  placebo) under 14 models (the eight of the design, six of them also fitted
  on ranked returns): 364 experiments;
- 25 further feature sets under the eight base models: the tone of the news,
  FinBERT headline embeddings and news-times-price features (7 sets), and the
  narrative, text and price-and-text sets of the six topic models with the
  attention-shock window chosen on validation (18 sets): 200 experiments;
- the ensemble of the eight base models on each of the 51 sets, four
  combinations across topic models (section 6), and 13 forecasts that fit
  nothing (11 single-feature signals and two combinations).

That is 1,692 fitted experiments, 1,896 forecasts in all, each traded under nine
portfolio rules. Training 2013 to 2019, validation 2020 to 2021, test 2022 to
2024, now with every month of every year: **36 test months, 78 fortnights and
156 weeks** (21, 42 and 81 in the run of 8 October, which had to drop January
to June of every year from 2017). Every number below comes from a file
under `results/` (section 14 says which).

**How to read the numbers.**

- *Rank IC*: the Spearman correlation between forecast and realised return
  across the stocks of one period, averaged over periods. A period in which a
  model gives every stock the same forecast counts as zero.
- *t statistics* use Newey-West standard errors and are compared with
  small-sample critical values (Kiefer and Vogelsang 2005): 2.29 for the 36
  test months, 2.11 for the 78 fortnights, 2.06 for the 156 weeks; 2.34, 2.19
  and 2.10 in the validation period.
- *Sharpe ratios* are annualised. *Net* means after 10 basis points per unit
  traded. *Turnover* is half the sum of absolute weight changes per period.
- *Decile long-short, equal weights* (`ls_ew`) is the portfolio behind every
  Sharpe ratio that does not name another rule. *Buffered* (`ls_buf`): a stock
  enters a leg in the top (bottom) decile and stays until it leaves the top
  (bottom) fifth. *Cap-weighted* (`ls_vw`, `ls_buf_vw`): the same legs weighted
  by market value.
- *Base models*: OLS, ridge, lasso, IPCA, sparse IPCA, XGBoost, random forest,
  EBM.
- *The tuned window*: attention shocks taken against the mean of the previous
  63 trading days instead of 5, the window that the validation rule of section
  5 chose at every frequency. Feature sets with it carry the suffix `_s63`.

## 1. The short version

1. **Choosing one experiment on the validation period does not work.** At each
   frequency the experiment with the highest validation rank IC among the 200
   base experiments of the main design is evaluated on the test period:

   | Rebalancing | Chosen on validation | Validation rank IC (t) | Test rank IC (t) | Net Sharpe | buffered | 95% interval, net |
   |---|---|---|---|---|---|---|
   | Monthly | Price and LDA text, EBM | 0.070 (2.1) | 0.015 (0.5) | -0.23 | -0.37 | -1.58 to 1.32 |
   | Biweekly | FinBERT text, EBM | 0.040 (2.4) | -0.005 (-0.4) | -0.27 | -0.13 | -1.28 to 0.72 |
   | Weekly | Price and LDA text, EBM | 0.065 (3.1) | -0.013 (-0.9) | -1.08 | -0.98 | -2.38 to 0.14 |

   The EBM wins the validation period at every frequency and loses the test
   period. The chosen price-and-text EBMs lean on pairwise terms of price and
   firm-news features, and deleting those terms raises their test Sharpe ratio
   by 0.23 to 0.75 (section 10).

2. **One choice made on the validation period carries over: a longer
   attention-shock window.** Our pre-set rule (section 5) chose 63 days at
   every frequency. Against the same models and topic models with the 5-day
   window of the main design, in the test period:

   | | Pairs | Rank IC, 5 days | 63 days | Difference (t) | Net Sharpe improved |
   |---|---|---|---|---|---|
   | Monthly | 46 | 0.011 | 0.023 | +0.013 (1.8) | 91% of pairs |
   | Biweekly | 46 | 0.004 | 0.012 | +0.009 (2.0) | 80% |
   | Weekly | 48 | 0.006 | 0.011 | +0.005 (1.7) | 85% |

   The narrative exposures with the 63-day window have a positive test IC in
   43, 41 and 40 of 48 experiments, and in both 2022-23 and 2024; with the
   5-day window it was positive in 2022-23 and negative in 2024.

3. **Combining every reader gives the best and steadiest results.** The
   combination across topic models averages, with equal weights, the
   eight-model ensembles of the six topic models: nothing in it is estimated or
   chosen. On price and text with the 63-day window (`grand_s63`) and on the
   text alone (`grand_text_s63`), test period:

   | | Test rank IC (t) | Net Sharpe | buffered | Return a year, net | Volatility | Maximum drawdown |
   |---|---|---|---|---|---|---|
   | Monthly, price and text | 0.039 (2.9) | 1.12 | 1.19 | 11.6% | 10.4% | -7% |
   | Biweekly, price and text | 0.024 (2.1) | 0.69 | 0.77 | 10.9% | 15.8% | -14% |
   | Weekly, price and text | 0.020 (2.0) | 0.30 | 0.39 | 4.7% | 15.8% | -14% |
   | Monthly, text alone | 0.025 (1.9) | 0.86 | 0.84 | 9.4% | 10.9% | -8% |
   | Biweekly, text alone | 0.016 (1.5) | 0.63 | 0.86 | 8.0% | 12.7% | -11% |
   | Weekly, text alone | 0.009 (1.2) | 0.24 | 0.50 | 3.0% | 12.8% | -15% |

   The monthly IC is beyond its critical value and the biweekly one at it. The
   same combinations with the 5-day window are weaker everywhere, and on the
   text alone negative at every frequency (-0.010, -0.009, -0.008). This
   combination was added after the test results of this run had shown that the
   63-day window and the eight-model ensemble each help (section 6).

4. **Significance is within reach monthly, and nothing survives the count of
   what we tried.** The monthly price-and-text combination has a net Sharpe
   ratio of 1.12 with t = 2.36 (p = 0.045 under fixed-b critical values),
   buffered 1.19 (p = 0.040). Against the price features alone, or after
   deflating for the 575 experiments of the frequency, it is not significant
   (Ledoit-Wolf p = 0.46 against the price ensemble; deflated Sharpe ratio
   0.04 for the 587 experiments of the frequency). The one difference that is significant is the shock window on the
   text alone, monthly: 0.86 against -0.45 (Ledoit-Wolf p = 0.028; section 11).

5. **Past prices remain the most reliable input.** The eight price features
   alone have a positive average test IC at every frequency (0.022, t = 2.4,
   monthly; 0.012 and 0.016 at the higher frequencies), and their ensemble a
   monthly IC of 0.034 (t = 2.4). Adding text with the 5-day window lowers the
   IC in most pairs (in 33, 38 and 43 of 48); with the 63-day window it raises
   it in 33 of 48 pairs monthly and 26 biweekly, and lowers it in 36 of 48
   weekly.

6. **Trading less helps; weighting by market value does not.** The buffer
   raises the median net Sharpe ratio of the 48 experiment pairs of each
   frequency and improves it in 65%, 89% and 73% of them, with turnover about
   a third lower; the return gain (0.4%, 1.1%, 1.4% a year) is not
   significant on its own. Weighting the decile legs by market value lowers the
   median net Sharpe ratio by 0.12, 0.21 and 0.09 and improves it in only 37%
   to 43% of experiments; the cap-weighted buffer recovers part of that.

7. **The tone of the news does not forecast these returns.** Tone and
   embedding sets alone have test ICs between -0.005 and 0.009, and the
   price-and-news sets do about as well as price alone. Of the single-feature
   tone signals, FinBERT's 21-day tone and its surprise reach t = 2.0 and 2.3
   biweekly and nothing elsewhere.

8. **The test can now rule out less.** With three years of test data the
   standard error of an annual Sharpe ratio is about 0.58 at every frequency
   (0.75 to 0.80 before). A true Sharpe ratio of about 1.7 (1.8 monthly) is
   detected four times out of five; a single experiment's true IC of about
   0.04 (monthly), 0.034 (biweekly) or 0.028 (weekly).

9. **Two picks of an earlier run do not hold up on the complete data.**
   Price and every news block with an EBM (weekly) had a net Sharpe ratio of
   0.61 on the 81 weeks of the old file; on the 156 weeks it has 0.02
   (buffered 0.32), with a test IC of 0.018 (t = 2.4). NoLBERT-20 exposures
   with 63-day shocks and IPCA (monthly) had 0.68 on 21 months; on 36 it has
   -0.43. What does hold up is the direction behind them, the longer shock
   window, now chosen on validation for every frequency and every topic model.

10. **The data still decide more than the models do.** The return file (now
    complete), the article text of one source (wrong for 7% of articles), the
    survivorship of the news source and the 2024 regime change are described
    in section 2; the first was worth a spurious Sharpe ratio of 2 in our
    first run.

## 2. Data

### 2.1 More years of news

The same source goes back further than 2015 and can be extended by one year.

| Year | Unique articles | Tickers | Articles per weekday (median) | Weekdays with fewer than 20 articles |
|---|---|---|---|---|
| 2012 | 22,606 | 2,116 | 83 | 9 |
| 2013 | 38,489 | 2,458 | 148 | 10 |
| 2014 | 55,364 | 2,691 | 206 | 6 |
| 2015 | 67,249 | 2,882 | 246 | 3 |
| 2016 | 80,220 | 3,048 | 292 | 2 |
| 2017 | 98,649 | 3,141 | 367 | 1 |
| 2018 | 102,114 | 3,466 | 376 | 0 |
| 2019 | 76,806 | 3,567 | 252 | 3 |
| 2020 | 87,837 | 3,727 | 301 | 0 |
| 2021 | 119,912 | 3,914 | 419 | 0 |
| 2022 | 169,895 | 3,937 | 596 | 0 |
| 2023 | 219,720 | 4,347 | 789 | 10 |
| 2024 | 116,428 | 1,769 | 412 | 24 |

- **Before 2012** the source is too thin: 850 article rows in 2009, 10,278 in
  2010, 15,704 in 2011.
- **2012 to 2014** are usable and add three years. 2012 serves as the run-in
  year for the 252-day windows.
- **2024** exists in FinMultiTime, from a second scrape. It is a different
  regime, and we report 2022-23 and 2024 separately (section 4.7):
  - after 15 December 2023 there are no articles until 8 January 2024, and
    only 22 in January 2024;
  - it covers fewer than half as many tickers;
  - its rows have no headline: we rebuild one from the article address, so
    every 2024 headline is in lower case.
- **Tagging drifts in the second half of 2023.** An article is filed under 1.4
  to 1.8 tickers on average in every month until June 2023, 2.4 in July, 3.6
  to 4.9 from
  August to November and 24 in December 2023; in 2024 it is 1.3 to 1.4.
  Firm-level features use only articles filed under at most three tickers,
  which keeps 94% to 98% of the articles of every year except 2023 (87%; 29%
  in December 2023).
- **16 December 2023** is a dump of tens of thousands of rows (47,890 in
  FNSPID). It is dropped.
- Article dates have no time of day (98.9% are stamped midnight), so news is
  used with a lag of one trading day.

### 2.2 One source has the wrong text for about 7% of the articles

FinMultiTime lists the same Nasdaq.com articles as FNSPID and adds 2024, so we
build the article list from it. Its text cannot be used as it is: the files of
93 tickers (alphabetically A to AFG, among them AAPL, ABBV, ABT and ADBE) carry
the right date, headline and address but the body of an unrelated article.

| Year | Articles | Text differs from FNSPID's | Share |
|---|---|---|---|
| 2012 | 22,606 | 1,524 | 6.7% |
| 2013 | 38,489 | 2,507 | 6.5% |
| 2014 | 55,364 | 3,600 | 6.5% |
| 2015 | 67,249 | 4,401 | 6.5% |
| 2016 | 80,220 | 4,442 | 5.5% |
| 2017 | 98,649 | 5,863 | 5.9% |
| 2018 | 102,114 | 5,499 | 5.4% |
| 2019 | 76,806 | 5,152 | 6.7% |
| 2020 | 87,837 | 6,484 | 7.4% |
| 2021 | 119,912 | 6,513 | 5.4% |
| 2022 | 169,895 | 13,936 | 8.2% |
| 2023 | 219,720 | 23,128 | 10.5% |

In all, 83,049 of the 1,138,861 articles of 2012 to 2023 (7.3%), and 32,988
of the 541,497 dated up to 2019 (6.1%). The count is of articles whose first
150 characters differ from FNSPID's for the same address. It can include
differences other than a wrong body (character encoding, for instance), so the
share with another article's text may be somewhat below 7%. The wrong bodies
are often years younger than the article they are attached to, so
text "of the training period" contained later text: the vocabulary of our
first topic models, built on articles dated up to 2019, included covid,
coronavirus and chatgpt.

FNSPID has the right bodies. Every article's text is now taken from FNSPID
where FNSPID lists the article (all but 423 of the articles of 2012 to 2023).
FNSPID does not cover 2024: there, an article from a damaged ticker file takes
its text from an intact ticker file if it is also filed under one (2,755
articles) and is otherwise represented by its headline alone (4,513 of the
116,428 articles of 2024). After the repair the training vocabulary contains
none of the three words. The earlier project read FNSPID directly and is not
affected.

### 2.3 The news source is not point-in-time

The articles were collected from each ticker's page at the end of the sample.
Three consequences:

- **Survivorship.** The source holds the tickers that were listed when it was
  collected: 93% of its 4,672 tickers have an article in 2023 or 2024. No
  article is filed under the ticker of any of eleven large companies that were
  acquired between 2016 and 2023 (EMC, Yahoo, Monsanto, Time Warner, Aetna,
  Celgene, Allergan, Tiffany, Twitter, Xilinx, Activision).
- **Truncation.** A heavily covered ticker has only its most recent articles,
  at most 9,435. Apple's begin on 3 June 2022, Microsoft's on 26 April 2022,
  Tesla's in May 2022, Nvidia's in August 2021, Alphabet's (GOOG) in February
  2019.
- **Gaps.** Some of the largest companies have no articles with a body
  although they were listed throughout: JPMorgan, Johnson & Johnson, Procter &
  Gamble, Mastercard, Home Depot, Bank of America, Netflix, Cisco, Pfizer,
  Verizon and Meta.

Share of the universe with at least one article filed under its ticker in the
previous 252 trading days (`results/data_audit/firm_coverage.csv`):

| End of | Stocks | By market value | Of the 50 largest |
|---|---|---|---|
| 2013 | 44% | 39% | 34% |
| 2015 | 49% | 42% | 36% |
| 2017 | 56% | 47% | 44% |
| 2019 | 62% | 53% | 54% |
| 2021 | 61% | 48% | 56% |
| 2022 | 65% | 64% | 60% |
| 2023 | 72% | 71% | 62% |
| 2024 | 46% | 28% | 34% |

For firm-level features this matters twice. About half of the universe has no
features. And "no news" is partly information about the future: it marks, among
others, the companies that later disappeared. We therefore treat a firm without
any article in the previous 126 trading days as not covered and give it the
features of a randomly drawn covered firm (section 2.4). That is the case for
50% of the stock-months of the training period, 39% of the validation period
and 40% of the test period (`donor_rows.csv`): for those rows the firm-news
block is noise by construction, which pulls every firm-news result towards
zero. The covered firms are themselves a sample that survived to the
collection date, which no imputation can undo.

For market-wide attention it means that the early years are measured on the
articles of the less covered half of the market.

### 2.4 The return file

**The final run uses a complete file.** `scripts/00_pull_crsp.py` builds
`data/crsp/` from WRDS (CRSP daily and monthly stock files and their name
history; the Fama-French daily factors): the 1,000 largest ordinary common
shares (share codes 10 and 11, NYSE, AMEX and Nasdaq) by market value at the
end of each June from 2010 to 2024, 2,062 stocks in all, with daily returns of
every list member on every day from 2011 to 2024 (5.15 million stock-days).
Against the extract of the earlier project, which the runs up to 8 October
used:

- on the 4.13 million stock-days that both files hold, returns and market
  values agree exactly;
- 97% to 100% of each June list agree (the earlier extract applied a
  share-code filter we could not recover);
- the new file has 1.02 million stock-days that the old one lacked, almost all
  of them the January-to-June days of stocks that left the next June list.

Tickers, through which news is matched to stocks, are kept from the old file
for the 1,822 stocks it covers (they were checked against the news source in
the earlier audits) and taken from the CRSP name history for the 240 others.
With the complete file `DAILY_RETURNS_COMPLETE_UNTIL = None` (`ftm/config.py`):
every formation day is used, the panels grow to 141,761 stock-months, 306,746
stock-fortnights and 614,525 stock-weeks (984 stocks per period), and no stock
of any period lacks a recent return (`results/data_audit/stale_rows.csv`).

**Why it mattered.** From 2017 the old extract held daily returns only for
stocks on the June list of the same calendar year. Between January and June a
stock about to leave the list therefore had no recent returns, and it had
none because of what happened next: such stocks went on to underperform the
cross-section by 3% to 5% a month. With the usual convention (a missing rank
set to the median) a tree model found them, and our first result, a weekly
net Sharpe ratio of 1.56 from the price features, was this leak (section 2.4
and `scripts/11_missing_data_leak.py` of the archived run). The runs up to 8
October removed it by dropping those months and hiding missing values; the
complete file removes it at the root and restores the dropped months: 84, 24
and 36 months instead of 69, 14 and 21.

**What we still do.** A stock with a missing value in a feature block receives
the complete feature row of another stock of the same day, drawn at random
(a "donor"), so that no model can see that a value is missing. With the
complete file this concerns 0.3% to 1.2% of stock-periods in the price block
(stocks with too short a history), about 0.5% in the narrative block and 39%
to 50% in the firm-news and tone blocks (section 2.3).

### 2.5 Other look-ahead in the earlier pipeline

As we read its code (we did not re-run it):

| Problem | Earlier project | Here |
|---|---|---|
| Universe | The 1,000 largest by June market value, applied from January of the same year | The June list is used from the end of June |
| Topic models and vocabulary | Fitted on all text, 2015 to 2023 | Fitted on text up to 2019 |
| Tuning | On the evaluation months; no validation set | On 2020 to 2021 only |
| News timing | The day's news with the day's portfolio | Lagged one trading day |
| Missing data | Visible to the model | Hidden from the model; a complete return file (section 2.4) |

The first row is the mechanism of section 2.4 in a stronger form: the
earlier universe contained, from January, only stocks known to be large in the
following June.

## 3. Topic models

The models are fitted on 140,209 articles dated up to 2019 and label the same
529,657 articles (at most 150 a day); "LDA, every article" applies the LDA
topics to all 1,255,289. The topic models are therefore in-sample for the
training period of the return models and out of sample after 2019.

| Model | Topics | Coherence (NPMI), training text | later text | Diversity of top words | Stability across seeds | Largest topic | Effective number of topics | Reliability of attention shocks, to 2019 | after 2019 |
|---|---|---|---|---|---|---|---|---|---|
| LDA | 60 | 0.58 | 0.37 | 0.84 | 58% | 12.7% | 40.1 | 0.30 | 0.35 |
| MiniLM clusters | 60 | 0.49 | 0.34 | 0.83 | 67% | 4.3% | 53.6 | 0.25 | 0.32 |
| FinBERT classes | 20 | 0.30 | 0.21 | 0.83 | fixed | 21.4% | 11.6 | 0.32 | 0.38 |
| NoLBERT clusters | 60 | 0.48 | 0.31 | 0.76 | 69% | 4.0% | 54.0 | 0.28 | 0.34 |
| NoLBERT, 20 clusters | 20 | 0.33 | 0.21 | 0.79 | 80% | 7.0% | 19.1 | 0.47 | 0.50 |
| LDA, every article | 60 | 0.58 | 0.37 | 0.84 | 58% | 12.7% | 40.1 | 0.33 | 0.46 |

- **Coherence depends on how top words are picked.** Above, every model's ten
  top words come from one rule (class-based TF-IDF over the topic's articles).
  With the customary rule for LDA (its most probable words) LDA scores 0.35
  and 0.24. A comparison that uses one rule for LDA and the other for
  embedding clusters shows the clusters ahead by 0.14; on equal terms LDA is
  ahead by 0.09. The earlier project compared them the first way, which is
  why its BERT topics looked more coherent.
- **Embedding clusters are more even and more stable.** The largest MiniLM
  topic holds 4.3% of articles against LDA's 12.7%, and 67% of articles keep
  their topic under another seed against 58%. NoLBERT's clusters are the most
  stable and the most even of the 60-topic models (69% kept, adjusted Rand
  index 0.64 against 0.61 for MiniLM and 0.47 for LDA; largest topic 4.0%),
  with the least diverse top words (0.76).
- **Fewer topics, more reliable shocks.** With 20 clusters instead of 60, each
  NoLBERT topic averages three times as many articles a day, and the
  split-half reliability of its daily shocks rises from 0.28 to 0.47 before
  2020 and from 0.34 to 0.50 after; the topics become broader and less
  coherent (0.33 against 0.48).
- **Coherence falls on later text** for every model (by 37% for LDA, 32% for
  MiniLM, 29% for FinBERT, 36% for NoLBERT): the topics are not refitted after
  2019.
- **Much of the corpus is templated.** Several topics of every embedding and
  LDA model are automatic notices: ex-dividend reminders, moving-average
  alerts, changes in ETF shares outstanding, option-chain summaries. NoLBERT's
  topic 3 (ETF commentary) has a later coherence of -0.21 and a stability of
  0.02, and was left in, as the junk topics of the other models were.
- **FinBERT's twenty classes are broad** (the largest takes 21% of articles;
  they are worth about 12 equally sized topics). The classifier was published
  in December 2022, fine-tuned on a set of finance tweets released that
  September, so it is not point-in-time. It ran on headlines only.
- **`minilm` is the earlier project's "BERT" topic model**: the same encoder
  (published in 2021, so not point-in-time either), spherical k-means and soft
  weights. Three things differ: one passage per article (headline and lead)
  instead of up to three, clusters fitted on training-period text only, and
  150 instead of 120 articles a day. `nolbert` is built the same way from the
  adapted NoLBERT encoder (section 9).

**Reliability of attention shocks.** Each day's articles are split at random
into two halves and the attention shock of a topic is computed from each half.
The correlation of the two, stepped up to the full sample, says what share of
the shock's variance is signal. For the LDA topics on every article (the 150
column is another random draw than the sample of the table above, hence 0.31
against 0.30):

| Articles per day | 50 | 100 | 150 | 300 | all |
|---|---|---|---|---|---|
| To 2019 | 0.22 | 0.28 | 0.31 | 0.33 | 0.33 |
| After 2019 | 0.21 | 0.29 | 0.35 | 0.42 | 0.46 |

At 150 articles a day two thirds of a typical topic's daily shock is sampling
noise, so narrative exposures are noisy rankings of stocks. More articles help
only where they exist: before 2020 there are too few a day for it to matter.
Where more articles or fewer topics raise reliability, the forecasts do not
follow (sections 4.3 and 9.2).

## 4. Do the features forecast returns?

### 4.1 The landscape

Test-period rank IC of the eight base models, averaged by kind of feature set
(`landscape.csv`, "kind"; t of the group's average IC per period; positive:
experiments with a positive IC; Sharpe: median net Sharpe ratio of the
group's long-short portfolios):

| Kind | Monthly IC (t) | positive | Sharpe | Biweekly IC (t) | positive | Sharpe | Weekly IC (t) | positive | Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| Price | 0.022 (2.4) | 6 of 8 | 0.41 | 0.012 (1.7) | 8 of 8 | -0.07 | 0.016 (1.5) | 7 of 8 | 0.09 |
| Narrative, 5 days | -0.005 (-0.9) | 19 of 48 | 0.03 | -0.004 (-0.7) | 15 of 48 | -0.20 | -0.003 (-0.7) | 19 of 48 | -0.46 |
| Narrative, 63 days | 0.018 (2.1) | 43 of 48 | 0.84 | 0.010 (1.5) | 41 of 48 | 0.39 | 0.006 (1.3) | 40 of 48 | 0.03 |
| Firm news | -0.002 (-0.4) | 15 of 48 | -0.82 | 0.001 (0.2) | 24 of 48 | -1.30 | -0.000 (-0.1) | 20 of 48 | -1.53 |
| Text, 5 days | -0.005 (-0.8) | 17 of 48 | 0.00 | -0.005 (-0.8) | 13 of 48 | -0.20 | -0.004 (-1.0) | 13 of 48 | -0.57 |
| Text, 63 days | 0.015 (1.9) | 40 of 48 | 0.64 | 0.009 (1.5) | 41 of 48 | 0.30 | 0.006 (1.3) | 39 of 48 | -0.06 |
| Price and text, 5 days | 0.011 (1.5) | 29 of 48 | 0.24 | 0.004 (0.7) | 23 of 48 | 0.05 | 0.006 (1.4) | 36 of 48 | -0.24 |
| Price and text, 63 days | 0.023 (3.1) | 43 of 48 | 0.63 | 0.012 (2.1) | 45 of 48 | 0.34 | 0.011 (2.1) | 42 of 48 | 0.00 |
| Tone (FinBERT, LM) | 0.006 (1.6) | 13 of 16 | 0.06 | 0.003 (1.0) | 13 of 16 | -1.00 | -0.002 (-0.7) | 3 of 16 | -1.96 |
| Headline embeddings | -0.005 (-0.6) | 1 of 8 | -1.12 | 0.009 (2.8) | 7 of 8 | -0.16 | 0.002 (1.1) | 5 of 8 | -2.40 |
| Price and news | 0.024 (3.7) | 20 of 24 | 0.54 | 0.015 (2.8) | 24 of 24 | 0.19 | 0.015 (2.1) | 22 of 24 | -0.01 |
| Placebo | -0.000 (-0.0) | 3 of 8 | -0.10 | 0.006 (0.8) | 5 of 8 | 0.16 | 0.006 (1.2) | 8 of 8 | -0.02 |

The kinds that work are the price features and everything built with the
63-day window. The firm's own news and the headline tone or embeddings add
turnover rather than return: their portfolios trade 1.1 to 1.6 times their
value a period and lose after costs.

### 4.2 Validation does not carry over

In the validation period the 5-day and 63-day text sets look alike (average
IC 0.009 and 0.021 monthly, 0.013 and 0.019 biweekly, 0.009 and 0.012 weekly)
and price alone looks poor (-0.001 monthly, 0.010 biweekly). In the test period
price is among the best groups. Ranked by validation IC, the 200 base
experiments of a frequency do not come out in the same order in the test
period, and the single best one fails at every frequency (section 1, point 1).
The validation period (2020-21) contains the pandemic crash and recovery and
the meme-stock months, and only 24 months, 52 fortnights or 104 weeks: it is
a poor guide to 2022-24 for any rule that picks one experiment out of
hundreds. Rules that pick a setting shared by many experiments (a shock
window, a portfolio buffer) carry over better.

### 4.3 Topic model against topic model

Average test IC of the text-only experiments of each topic model (24 each,
5-day window; `landscape.csv`, "topic_model"):

| Topic model | Monthly | Biweekly | Weekly |
|---|---|---|---|
| LDA | 0.001 | -0.000 | -0.003 |
| LDA, every article | -0.017 | -0.009 | -0.007 |
| MiniLM clusters | -0.007 | -0.005 | -0.003 |
| FinBERT classes | -0.002 | -0.002 | -0.001 |
| NoLBERT clusters | 0.001 | -0.000 | 0.001 |
| NoLBERT, 20 clusters | 0.001 | -0.001 | -0.001 |

None of the 45 paired differences (`contrasts.csv`) is significant; the
largest are LDA on the 150-a-day sample against LDA on every article (t = 1.8
monthly and biweekly). NoLBERT (60 or 20 clusters) has the highest IC monthly
and weekly and is within 0.0003 of the top biweekly; every difference
involving it is inside noise. With the 63-day window (narrative and text sets)
all six are positive at every frequency, and LDA and NoLBERT lead (monthly
0.025 and 0.026; biweekly 0.020 and 0.013; weekly 0.009 and 0.007), with LDA on
every article last (0.000 to 0.002).

### 4.4 Text against price

Each text-and-price experiment is paired with the same model on price alone
(`text_vs_price.csv`, 400 pairs per frequency). With the 5-day window adding
text lowers the test IC in 33 (monthly), 38 (biweekly) and 43 (weekly) of 48
price-and-text pairs. With the 63-day window it raises it in 33 of 48 monthly
and 26 biweekly, and lowers it in 36 of 48 weekly. In squared error adding text
is a significant improvement in 17 of the 400 monthly pairs, 2 biweekly and
none weekly.

### 4.5 Placebo

Exposures to LDA attention shocks of 126 trading days earlier against the
real ones (5-day window; `placebo.csv`): the test IC of the real exposures
minus that of the placebo is +0.003 (t = 0.2) monthly, -0.009 (-0.8) biweekly
and -0.007 (-0.9) weekly. The 5-day narrative exposures do no better than
misdated ones. No placebo was built for the 63-day window.

### 4.6 The price features

All eight base models and all six ranked-target variants have a positive
weekly test IC on price alone except sparse IPCA (0.014 to 0.036; -0.018 for
sparse IPCA). The ranked-target versions are the most uniform (0.020 to 0.050
monthly, 0.029 to 0.033 biweekly, 0.030 to 0.036 weekly).
Their weekly portfolios earn gross Sharpe ratios of about 1.0 to 1.2 and keep
0.35 to 0.66 after costs. Monthly, IPCA and sparse IPCA on price have test ICs
of 0.052 with a negative market beta (-0.8): part of their result is a short
tilt in a market that rose.

### 4.7 The two halves of the test period

The 2024 news comes from a second scrape (section 2.1). Average IC in 2022-23
and in 2024 (`halves.csv`):

| Kind | Monthly 2022-23 | 2024 | Biweekly 2022-23 | 2024 | Weekly 2022-23 | 2024 |
|---|---|---|---|---|---|---|
| Narrative, 5 days | 0.006 | -0.027 | 0.002 | -0.016 | 0.002 | -0.012 |
| Narrative, 63 days | 0.017 | 0.018 | 0.008 | 0.012 | 0.006 | 0.006 |
| Price | 0.013 | 0.041 | 0.010 | 0.016 | 0.012 | 0.023 |
| Price and text, 63 days | 0.018 | 0.033 | 0.009 | 0.019 | 0.009 | 0.015 |

With the 5-day window the narrative IC turns negative in 2024 at every
frequency, as it did in the earlier run; with the 63-day window it is stable
across the two regimes of the news source. No difference between the halves is
significant.

## 5. The attention-shock window

The main design takes a topic's attention shock against its mean over the
previous 5 trading days. `scripts/05b_tune_shocks.py` rebuilds the narrative
exposures with windows of 1, 5, 21 and 63 days, fits OLS, ridge and lasso on
the training period for each topic model and frequency, and records the
validation rank IC (18 fits per frequency and window). Rule, written down in
the run of 8 October and applied unchanged: the window with the highest
average validation IC, if it also beats 5 days for at least four of the six
topic models; otherwise 5 days.

| Validation rank IC (18 fits) | 1 day | 5 days | 21 days | 63 days | Topic models better than 5 days (1 / 21 / 63) | Window |
|---|---|---|---|---|---|---|
| Monthly | -0.008 | -0.000 | -0.001 | 0.014 | 2 / 3 / 5 | 63 |
| Biweekly | -0.002 | 0.014 | 0.010 | 0.015 | 0 / 2 / 4 | 63 |
| Weekly | 0.001 | 0.006 | 0.004 | 0.008 | 1 / 2 / 4 | 63 |

(On the old file the rule had chosen 63 days monthly, 1 day weekly and kept 5
days biweekly.) The window enters the narrative, text and price-and-text sets
of all six topic models. Against the same sets with 5 days (`improvements.csv`,
"shock_window"):

| | Validation IC difference (t) | Test IC difference (t) | Test net return a year (t) | Median net Sharpe, 5 / 63 days | Improved |
|---|---|---|---|---|---|
| Monthly | +0.006 (0.8) | +0.013 (1.8) | +5.8% (2.2) | 0.24 / 0.62 | 91% |
| Biweekly | +0.000 (0.0) | +0.009 (2.0) | +4.7% (2.0) | 0.05 / 0.34 | 80% |
| Weekly | -0.000 (-0.0) | +0.005 (1.7) | +3.9% (1.6) | -0.24 / 0.00 | 85% |

A 63-day shock compares a topic's attention with its level over the past
quarter: it marks the topics that are becoming a theme of the market rather
than the news of one week, and the exposures of stocks to such shifts are
estimated from many more independent movements. The improvement is larger in
the test period than in validation, so it is not an artefact of tuning.

## 6. Combining forecasts

| Combination (test) | Monthly IC (t) | Net Sharpe | Biweekly IC (t) | Net Sharpe | Weekly IC (t) | Net Sharpe |
|---|---|---|---|---|---|---|
| Ensemble of the eight models, price alone | 0.034 (2.4) | 0.66 | 0.018 (1.7) | 0.10 | 0.019 (1.5) | -0.04 |
| Across topic models, price and text, 5 days | 0.025 (1.4) | 0.39 | 0.008 (0.6) | 0.23 | 0.013 (1.4) | 0.01 |
| Across topic models, price and text, 63 days | 0.039 (2.9) | 1.12 | 0.024 (2.1) | 0.69 | 0.020 (2.0) | 0.30 |
| Across topic models, text alone, 5 days | -0.010 (-0.8) | -0.45 | -0.009 (-0.8) | -0.47 | -0.008 (-0.9) | -0.45 |
| Across topic models, text alone, 63 days | 0.025 (1.9) | 0.86 | 0.016 (1.5) | 0.63 | 0.009 (1.2) | 0.24 |
| Best ten by validation IC (`best10`) | 0.024 (1.6) | 0.33 | -0.002 (-0.1) | 0.12 | -0.004 (-0.4) | -0.60 |
| Price ensemble and best tone signal (`combo`) | 0.030 (2.4) | 0.69 | 0.012 (1.3) | 0.00 | 0.015 (1.5) | -0.72 |

- **The eight-model ensemble** of the six price-and-text sets has an IC above
  the average of its members by 0.006 (t = 1.2), 0.003 (0.8) and 0.003 (1.4),
  and a better net Sharpe ratio than its members' average in 6 of 6, 6 of 6
  and 4 of 6 sets (`improvements.csv`, "ensemble").
- **The combination across topic models** (`scripts/06d_grand_ensemble.py`)
  averages the within-period ranks of the six eight-model ensembles of one kind
  of feature set. It has no free parameter. It was added after this run's
  test results had shown the two ingredients to help, and it is reported for
  all four kinds, including the two that do badly.
- **Selection is the weak link.** Ten experiments chosen by validation IC
  (`best10`) do worse than the combination of everything with the 63-day
  window at every frequency, and the validation choice among the 51 ensembles
  (`selection.csv`, "ensemble") picks text sets with the 5-day window and has
  test Sharpe ratios of 0.04, -0.20 and -0.22.
- In validation the text-alone combination with 63 days is the best of the
  four monthly and biweekly (IC 0.035 and 0.034) and second weekly (0.023
  against 0.025 for price and text with 5 days); its validation net Sharpe
  ratios are 0.57, 1.00 and 0.64.

Detail of the price-and-text combination with 63 days and the text-alone one
(test, `strategies.csv`, feature sets `grand_s63` and `grand_text_s63`):

| | Rule | Return a year, gross | net | Volatility | Sharpe, gross | net | 95% interval | Max. drawdown | Turnover | Three-factor alpha (t) |
|---|---|---|---|---|---|---|---|---|---|---|
| Monthly, price and text | decile | 13.8% | 11.6% | 10.4% | 1.32 | 1.12 | 0.19 to 2.36 | -7% | 89% | 8.9% (2.1) |
| | buffered | 13.8% | 12.4% | 10.4% | 1.33 | 1.19 | 0.25 to 2.46 | -8% | 60% | 9.7% (2.2) |
| Biweekly, price and text | decile | 14.8% | 10.9% | 15.8% | 0.94 | 0.69 | -0.18 to 1.61 | -14% | 76% | 8.3% (1.5) |
| | buffered | 13.3% | 11.0% | 14.2% | 0.94 | 0.77 | -0.09 to 1.75 | -12% | 45% | 8.4% (1.7) |
| Weekly, price and text | decile | 16.1% | 4.7% | 15.8% | 1.02 | 0.30 | -0.78 to 1.26 | -14% | 110% | 1.8% (0.3) |
| | buffered | 13.3% | 5.7% | 14.7% | 0.91 | 0.39 | -0.67 to 1.35 | -13% | 73% | 3.3% (0.6) |
| Monthly, text alone | decile | 11.5% | 9.4% | 10.9% | 1.06 | 0.86 | -0.26 to 2.21 | -8% | 89% | 9.0% (1.4) |
| | buffered | 10.5% | 9.1% | 10.8% | 0.98 | 0.84 | -0.16 to 2.09 | -9% | 61% | 8.2% (1.5) |
| Biweekly, text alone | decile | 11.4% | 8.0% | 12.7% | 0.91 | 0.63 | -0.27 to 1.57 | -11% | 67% | 6.3% (1.5) |
| | buffered | 12.8% | 10.7% | 12.5% | 1.03 | 0.86 | 0.05 to 1.80 | -12% | 40% | 8.8% (2.1) |
| Weekly, text alone | decile | 8.0% | 3.0% | 12.8% | 0.63 | 0.24 | -0.79 to 1.33 | -15% | 48% | 2.9% (0.5) |
| | buffered | 8.1% | 5.6% | 11.2% | 0.73 | 0.50 | -0.50 to 1.59 | -10% | 24% | 5.0% (1.0) |

The market beta of these long-short portfolios is between -0.13 and 0.43 (the
weekly price-and-text one has the largest), so they are not a disguised market
bet. Weekly, costs take two thirds of the gross return; monthly, a sixth.

## 7. Portfolios

### 7.1 Rules

Median over the experiments of a frequency of the paired difference in net
Sharpe ratio, and the share of experiments improved (test, `strategies.csv`):

| | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Buffered against decile | +0.06 (61%) | +0.11 (79%) | +0.12 (78%) |
| Cap-weighted against equal-weighted decile | -0.12 (40%) | -0.21 (37%) | -0.09 (43%) |
| Cap-weighted buffered against cap-weighted decile | +0.03 (52%) | +0.09 (63%) | +0.13 (75%) |
| Cap-weighted buffered against equal-weighted buffered | -0.14 (42%) | -0.26 (33%) | -0.07 (44%) |

Median turnover per period: decile 101%, 77% and 76%; buffered 71%, 49% and
45%; cap-weighted 114%, 93% and 90%; cap-weighted buffered 79%, 58% and 53%.

- **The buffer** (Novy-Marx and Velikov 2016) is the one portfolio choice that
  helps across the board. Averaged over the 48 experiments of each frequency
  of `improvements.csv` ("buffer") it adds 0.4%, 1.1% and 1.4% a year net (t =
  1.7, 1.8 and 1.5), less than on the old file (where weekly and biweekly it
  was significant).
- **Cap weights** concentrate each leg in the largest stocks, where the
  forecasts are weakest and turnover is higher; the
  equal-weighted legs do better in most experiments. The cap-weighted
  long-short of the combination is near zero (0.03 monthly, -0.21 biweekly,
  -0.32 weekly), and of the text-alone combination positive (0.32, 0.07,
  0.13). In the search the best cap-weighted portfolios are no better than
  luck (section 11).
- **The cap-weighted tilt** (enhanced indexing) has a median net Sharpe ratio
  of 0.31 to 0.32 against 0.35 to 0.38 for the value-weighted universe, and a median
  information ratio of -0.20, -0.44 and -0.46. The tilt of the monthly
  text-alone combination has an information ratio of 0.36.

### 7.2 Monthly, biweekly or weekly?

Every comparison of this run points the same way: monthly rebalancing gives
the largest ICs and the largest net Sharpe ratios, weekly the smallest. The
signals that work (63-day attention shocks, price features) move slowly, and
weekly trading pays the cost of re-ranking noise: the weekly price-and-text
combination earns a gross Sharpe ratio of 1.02 and keeps 0.30. With the full
years the monthly test sample has 36 periods, enough for the monthly
combination's IC (t = 2.9) to pass its critical value.

## 8. Models

Test IC averaged over the 26 feature sets of the main design
(`landscape.csv`, "model"):

| Model | Monthly | Biweekly | Weekly |
|---|---|---|---|
| OLS | -0.006 | -0.003 | 0.001 |
| Ridge | -0.001 | 0.000 | 0.002 |
| Lasso | -0.009 | -0.005 | 0.001 |
| IPCA | 0.006 | -0.001 | 0.001 |
| Sparse IPCA | 0.005 | 0.003 | -0.007 |
| XGBoost | 0.003 | -0.000 | 0.003 |
| Random forest | 0.003 | 0.001 | 0.003 |
| EBM | 0.000 | 0.000 | 0.002 |
| Ensemble of the eight | 0.001 | -0.000 | 0.001 |
| XGBoost, ranked returns | 0.016 | 0.010 | 0.012 |
| Sparse IPCA, ranked returns | 0.011 | 0.010 | 0.008 |
| IPCA, ranked returns | 0.003 | 0.010 | 0.007 |

- **Ranked returns as the target help**, most at the higher frequencies: IC
  +0.003 (t = 0.5) monthly, +0.009 (2.5, beyond its critical value of 2.11)
  biweekly and +0.016 (1.8) weekly over the same model fitted on returns, with
  a better net Sharpe ratio in 59%, 79% and 81% of pairs (`improvements.csv`,
  "rank_target"). A ranked target removes the pull of the few extreme returns
  that squared error chases; returns are not winsorised.
- **Nonlinear against linear** models: +0.002, +0.001 and +0.003 in IC (t =
  1.0, 0.6, 1.2; `contrasts.csv`).
- **The EBM** has the best validation IC of the base models at every frequency
  and an average test IC of zero: the model that validation flatters most
  (it stops boosting on the validation rows) and the one it picks every time.

### 8.1 How much of a result is the seed?

`scripts/12_tuning_noise.py` reruns an experiment with ten seeds, tuning
included (`results/data_audit/tuning_noise.csv`). Test net Sharpe ratio across
the seeds:

| Experiment | Settings chosen | Test IC, lowest to highest | Net Sharpe, lowest to highest |
|---|---|---|---|
| Monthly, price, XGBoost | 3 | 0.001 to 0.016 | -0.41 to 0.71 |
| Monthly, price and LDA text, XGBoost | 4 | -0.016 to 0.028 | -1.00 to 1.47 |
| Monthly, FinBERT text, EBM | 1 | -0.011 to 0.003 | 0.19 to 0.45 |
| Biweekly, price, XGBoost | 3 | 0.005 to 0.015 | -0.65 to -0.09 |
| Biweekly, price and LDA text, XGBoost | 3 | -0.004 to 0.016 | -0.89 to 0.38 |
| Biweekly, price and FinBERT text, XGBoost | 4 | -0.013 to 0.005 | -1.17 to -0.12 |
| Weekly, price, XGBoost | 3 | 0.014 to 0.020 | -0.19 to -0.01 |
| Weekly, price and LDA text, XGBoost | 2 | -0.007 to 0.011 | -1.57 to -0.21 |
| Weekly, price and LDA text, EBM | 1 | -0.035 to -0.010 | -1.57 to -0.75 |

For a tree model on many features the seed moves the net Sharpe ratio by up to
2.5, as much as three years of sampling error. This is one more reason to
average over models and seeds rather than pick one experiment.

## 9. NoLBERT

NoLBERT (Kakhbod and Li 2025, `alikLab/NoLBERT`) is a DeBERTa-v3 language
model pretrained only on text published between 1976 and 1995. It cannot have
read anything about our sample, which no other text component here can claim:
MiniLM was published in 2021 and the FinBERT classifier in 2022, and LDA is
fitted on our own training text. One of us adapted it, embedded the
articles and fitted its topics (run of 8 October 2026); this section reports
that work and what it gives inside this pipeline.

### 9.1 Adaptation

A masked-language model is not a sentence encoder: mean-pooled NoLBERT vectors
are anisotropic and cluster poorly. The checkpoint was trained contrastively
(InfoNCE over two chunks of the same article, in-batch negatives, temperature
0.05, every parameter trained, learning rate 2e-5, batches of 32 pairs, 128
tokens, bf16) for 150 minutes on one RTX 3060: 6,205 steps, 198,560 pairs.
Held-out in-batch top-1 retrieval rose from 0.08 to 0.89
(`results/encoders/nolbert_ft/train_log.json`).

Pooling was chosen by a probe on 11,988 documents (`nolbert_probe.json`, NPMI
of 60 clusters on our vocabulary):

| Pooling of the adapted model | NPMI | Agreement with MiniLM's partition (AMI) |
|---|---|---|
| Last layer, mean, no whitening | 0.343 | 0.45 |
| Top four layers, mean | 0.268 | 0.48 |
| Top four layers, mean, all-but-the-top whitening | 0.282 | 0.45 |
| Top four layers, SIF, whitening | 0.273 | 0.44 |
| MiniLM, last layer, for reference | 0.308 | 1.00 |

On the adapted model the plain last-layer mean is best; whitening the top
four layers adds little (0.268 to 0.282), and the same probe shows it lowering
MiniLM's coherence (0.308 to 0.293). The two encoders agree on about half of
the partition.

Two caveats travel with every NoLBERT number:

- **The adaptation saw validation-period text.** Pairs were drawn from articles
  published up to 2020, which overlaps our validation period. That is text,
  not returns, and nothing from the test period; but it can flatter NoLBERT's
  validation numbers. `scripts/03c_finetune_encoder.py` now cuts the pairs at the end of the
  training period by default.
- **It differs from MiniLM in two ways at once.** NoLBERT was adapted to our
  corpus; MiniLM was used as published. A gap between them can come from the
  adaptation as much as from the pretraining period. The clean comparison is
  MiniLM adapted with the same recipe, which needs a GPU.

### 9.2 What it gives

- **Topics** (section 3): the most stable and most even partition of the
  60-topic models, the least diverse top words, and attention shocks about as
  reliable as LDA's.
- **Forecasts.** On the text-only sets with the 5-day window NoLBERT is first
  or tied for first of the six topic models at every frequency (test IC 0.001,
  -0.000, 0.001), inside noise; NoLBERT-20 is level with it. With the 63-day
  window the six topic models are close and all positive monthly.
- **Selection.** Validation picks no NoLBERT experiment as the headline at any
  frequency. The NoLBERT-20 experiment with 63-day shocks that had a net
  Sharpe ratio of 0.68 on the old file has -0.43 on the complete one.
- **In the combinations** NoLBERT is one of six equal members.

### 9.3 The same encoder in a separate replication

We also ran the adapted encoder through a separate replication of
Bybee, Kelly and Su (sparse IPCA with three factors, a mean-variance portfolio
of the factors, 41 out-of-sample months), next to LDA and two MiniLM variants.
These numbers come from that pipeline, which follows the earlier project's
design (no validation period; section 2.5 lists its look-ahead); we did not
re-run them:

| Topics | Out-of-sample R2, demeaned | predictive | Sharpe ratio in sample | out of sample |
|---|---|---|---|---|
| LDA | 5.73% | 0.83% | 0.74 | 0.43 |
| MiniLM | 7.28% | 0.14% | 0.66 | 0.51 |
| MiniLM, spherical | 6.32% | 0.31% | 0.60 | 0.20 |
| NoLBERT | 4.01% | 0.70% | 0.80 | 0.71 |
| Equal-weighted market | | | | 0.69 |

NoLBERT has the best out-of-sample Sharpe ratio there and is the only arm above
the market, with every paired difference below one standard error and every
bootstrap interval across zero.

### 9.4 Reading

The defensible claim is about the text layer: an encoder that cannot have seen
the future gives a stable, even and reproducible decomposition of the news at
no cost in forecasting power. The claim that it forecasts returns better than
the other readers is not supported. What the run supports is that the readers
are better together than apart.

## 10. Opening the glass box

For each frequency `scripts/09_explain.py` opens the EBM on price and text
with the best validation IC (price and LDA text at every frequency, the
monthly and weekly headline choices), following Bell, Kakhbod, Lettau and
Nazemi (2024): the importance of each term (mean absolute contribution over
the fitting rows), the shares of the blocks, and the test result after
deleting a block without re-estimating anything (`results/explain/`).

| | Share of importance: price | narrative | firm news | pairs | Net Sharpe, full model | without the pairs |
|---|---|---|---|---|---|---|
| Monthly | 0% | 1% | 0% | 98% | -0.23 | 0.00 |
| Biweekly | 3% | 31% | 9% | 58% | -0.52 | 0.23 |
| Weekly | 1% | 14% | 6% | 79% | -1.08 | -0.41 |

All three chose ten pairwise terms on validation, and the pairs carry most of
the model: interactions of a price feature (momentum, beta, volatility) or a
narrative exposure with the share of a firm's news in one LDA topic. Firm
news is donor noise for 40% to 50% of the stocks (section 2.3), so a pair on
it can fit validation by accident. Deleting the pairs raises the test IC from
0.015 to 0.016 monthly, from -0.012 to 0.012 biweekly and from -0.013 to 0.005
weekly. The glass box shows where the validation choice went wrong, which a
black box would not.

## 11. Statistical significance

### 11.1 Each portfolio against zero and against a benchmark

`scripts/13_sharpe_tests.py` (`results/tables/sharpe_tests.csv`) tests the
validated picks and the combinations with frozen forecasts: only the realised
test returns are random. Against zero: the Newey-West t of the mean return
(fixed-b p), the probabilistic Sharpe ratio, and the deflated Sharpe ratio for
the number of experiments of the frequency (587, 603 and 607 with a position).
Against a benchmark: the Ledoit and Wolf (2008) test of equal Sharpe ratios
(HAC, delta method) and a paired stationary bootstrap.

| Test period, decile long-short | Net Sharpe | t (p) | Deflated | Against | Difference | Ledoit-Wolf p | bootstrap p |
|---|---|---|---|---|---|---|---|
| Monthly, combination, price and text, 63 days | 1.12 | 2.36 (0.045) | 0.04 | price ensemble | +0.46 | 0.46 | 0.44 |
| | | | | same, 5 days | +0.74 | 0.15 | 0.16 |
| | | | | validated pick | +1.35 | 0.20 | 0.16 |
| Monthly, buffered | 1.19 | 2.42 (0.040) | 0.06 | price ensemble, buffered | +0.66 | 0.27 | 0.24 |
| Monthly, combination, text alone, 63 days | 0.86 | 1.60 (0.16) | 0.01 | same, 5 days | +1.31 | **0.028** | **0.026** |
| Biweekly, combination, price and text, 63 days | 0.69 | 1.41 (0.18) | 0.00 | validated pick | +0.96 | 0.10 | 0.06 |
| Biweekly, combination, text alone, 63 days, buffered | 0.86 | 1.84 (0.085) | 0.01 | same, 5 days | +1.24 | **0.048** | **0.026** |
| Weekly, combination, price and text, 63 days, buffered | 0.39 | 0.77 (0.46) | 0.00 | validated pick | +1.37 | **0.024** | **0.024** |
| Monthly, validated pick (EBM) | -0.23 | -0.33 (0.77) | 0.00 | price pick | +0.02 | 0.98 | 0.98 |

- The monthly combinations beat zero at the 5% level, before any correction
  for the search.
- The 63-day window beats the 5-day one significantly on the text-alone
  combination (monthly; biweekly buffered).
- Nothing beats the price features significantly: the news adds to price, if
  at all, less than three years can show.
- No deflated Sharpe ratio comes near 0.95: the best of 590 to 610 unskilled
  portfolios with the observed dispersion would be expected to show an annual
  Sharpe ratio of 2.3 (monthly), 2.2 (biweekly) and 2.7 (weekly). The
  combinations are not the best of the search but fixed forecasts, so the
  deflation is conservative for them.

### 11.2 The best of the search

Over every experiment of a frequency (`multiple_testing.csv`): the best net
Sharpe ratio, what the best of that many portfolios without skill that move
together as ours do would be expected to show, White's reality check and
Hansen's SPA test, and the count that passes a 10% false-discovery threshold.

| Pool | Cells (rule x frequency) | Best below the no-skill benchmark | Reality check p < 0.05 | SPA p < 0.05 | Pass the 10% FDR |
|---|---|---|---|---|---|
| Main design (375 a frequency) | 27 | 24 | 0 | 1 | 0 |
| Everything (617 a frequency) | 27 | 17 | 2 | 4 | 0 |

The cells that pass are monthly and biweekly, and their best portfolios are
all built with the 63-day window: monthly, LDA narrative exposures with
XGBoost (decile 2.15 against an expected best of 1.37, reality check p =
0.016, SPA p = 0.001; buffered 1.96, p = 0.045 and 0.009) and price and LDA
text with a random forest, cap-weighted and buffered (SPA p = 0.034);
biweekly, price and LDA text with ridge (SPA p = 0.030). In the main design
the best result beats the benchmark only once (monthly, cap-weighted buffered,
SPA p = 0.024). The median experiment has a net Sharpe ratio of -0.06,
-0.21 and -0.45 in the main design and 0.08, -0.04 and -0.25 with everything.

## 12. What changed since the run of 8 October

| | 8 October (old return file) | 10 October (complete file) |
|---|---|---|
| Test periods | 21 months, 42 fortnights, 81 weeks (July to January) | 36, 78, 156 (every month) |
| Training, validation | 69, 14 months | 84, 24 months |
| Universe | earlier project's June lists | CRSP June lists, share codes 10 and 11 (97% to 100% the same) |
| Shock window chosen on validation | 63 days monthly, 1 day weekly, 5 days biweekly | 63 days at every frequency |
| Headline (validated pick), net Sharpe | -0.05, 0.37, -0.66 | -0.23, -0.27, -1.08 |
| Buffer, net return gain a year (t) | 0.6% (2.0), 1.5% (3.5), 2.0% (5.2) | 0.4% (1.7), 1.1% (1.8), 1.4% (1.5) |
| Price and news, EBM, weekly | 0.61, buffered 0.65 | 0.02, buffered 0.32 |
| NoLBERT-20, 63 days, IPCA, monthly | 0.68, buffered 0.57 | -0.43, buffered -0.36 |
| New | | cap-weighted buffered rule (`ls_buf_vw`); combinations across topic models; Ledoit-Wolf and deflated-Sharpe tests |

The earlier numbers were not wrong for the sample they had; that sample was
the second half of each year and a year and three quarters long, and the
picks that stood out in it were among hundreds of candidates.

## 13. Limits, and what to do next

- **One test period, looked at more than once.** The test years were looked
  at in every run (three before this one). The 63-day window was chosen on
  validation by a rule written before this run; the combinations across topic
  models were added after its test results were seen.
- **Validation is short and unusual** (2020-21) and is asked to do three jobs:
  tune hyperparameters, stop boosting, and choose between experiments.
- **Firm-level news covers half the universe**, and the covered half is a
  survivor sample (section 2.3).
- **Topic models are not refitted after 2019**, so the pandemic and later
  subjects have no topic of their own. The 2024 news comes from another
  scrape.
- **NoLBERT's adaptation** used articles published up to 2020; MiniLM (2021)
  and the FinBERT classifier (2022) are not point-in-time.
- **Costs are flat** (10 basis points a side) and portfolios trade at the
  closing prices that end the price signals, so short-term reversal includes
  bid-ask bounce. Delisting returns are not included.
- **One seed per experiment** (section 8.1).
- **Next.** A placebo for the 63-day window; refitting topic models through
  time; a second validation period (2017-19 as validation for a 2020-24 test)
  for the shock window and the combinations; MiniLM adapted with NoLBERT's
  recipe, for a clean comparison of the encoders.

## 14. Where each number comes from

| Section | Files |
|---|---|
| 1, 4.1, 4.3, 8 | `results/tables/selection.csv`, `forecast.csv`, `strategies.csv`, `landscape.csv` |
| 2.1 to 2.3 | `results/data_audit/news_coverage.csv`, `text_repair.csv`, `firm_coverage.csv`, `donor_rows.csv` |
| 2.4 | `data/crsp/` against `data/crsp_legacy/` (not in git); `results/data_audit/stale_rows.csv` |
| 3 | `results/tables/topic_models.csv`, `data/topics/<model>/quality.json` |
| 4.2, 4.4, 4.5, 4.7 | `landscape.csv` (validation rows), `text_vs_price.csv`, `placebo.csv`, `halves.csv`, `contrasts.csv` |
| 5 | `results/tuning/shock_window.csv`; `improvements.csv` (shock_window) |
| 6 | `strategies.csv` and `forecast.csv` rows `grand_*__ens`, `price__ens`, `px_pool__best10`, `px_tone__combo`; `improvements.csv` (ensemble) |
| 7 | `strategies.csv`; `improvements.csv` (buffer) |
| 8.1 | `results/data_audit/tuning_noise.csv` |
| 9 | `results/encoders/nolbert_ft/`; `contrasts.csv`, `landscape.csv` |
| 10 | `results/explain/<freq>__all_lda/` |
| 11 | `results/tables/sharpe_tests.csv`, `multiple_testing.csv` |
| 12 | this file (the notes of the run of 8 October are not in the repository) |
