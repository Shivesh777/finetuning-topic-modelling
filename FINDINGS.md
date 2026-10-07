# Findings

Results of the run of 7 October 2026: 432 experiments (18 feature sets, 8
models, 3 rebalancing frequencies). Training 2013 to 2019, validation 2020 to
2021, test 2022 to 2024. From 2017 on, only the part of each year for which
our return file is complete is used (section 2.4), which leaves 21 test months
(42 fortnights, 81 weeks), about a year and three quarters of returns. Every
number below comes from a file under `results/` (section 10 says which).

**How to read the numbers.**

- *Rank IC*: the Spearman correlation between forecast and realised return
  across the stocks of one period, averaged over periods. A period in which a
  model gives every stock the same forecast counts as zero.
- *t statistics* use Newey-West standard errors. With so few periods the usual
  1.96 is too lenient, so each t is to be compared with a small-sample
  critical value (Kiefer and Vogelsang 2005): 2.39 for the 21 test months,
  2.25 for the 42 fortnights, 2.11 for the 81 weeks; 2.61, 2.29 and 2.19 in
  the validation period.
- *Sharpe ratios* are annualised. *Net* means after 10 basis points per unit
  traded. *Turnover* is half the sum of absolute weight changes per period, so
  a long-short portfolio that replaces both legs has 200%.
- *Decile long-short, equal weights* is the portfolio behind every Sharpe
  ratio that does not name another rule.

## 1. The short version

1. **The topic features show no out-of-sample forecasting power.** At each
   frequency one experiment is chosen on the validation period (the highest
   rank IC of 136 candidates) and evaluated on the test period:

   | Rebalancing | Chosen on validation | Validation rank IC (t) | Test rank IC (t) | Long-short Sharpe, gross | net of costs | 95% interval, net |
   |---|---|---|---|---|---|---|
   | Monthly | FinBERT text features, EBM | 0.063 (3.3) | -0.005 (-0.2) | 0.11 | -0.15 | -1.81 to 1.04 |
   | Biweekly | Price and FinBERT text features, XGBoost | 0.058 (3.4) | 0.017 (0.8) | 0.79 | 0.37 | -1.18 to 1.89 |
   | Weekly | Price and LDA text features, EBM | 0.048 (2.3) | 0.010 (0.7) | -0.23 | -0.66 | -2.17 to 0.63 |

   | | Test R2 | Clark-West t | Return a year, gross | net | Volatility | Maximum drawdown | Turnover per period |
   |---|---|---|---|---|---|---|---|
   | Monthly | -0.054% | 0.2 | 1.2% | -1.7% | 11.6% | -17% | 123% |
   | Biweekly | -0.078% | 1.0 | 12.5% | 5.9% | 15.9% | -17% | 127% |
   | Weekly | -0.001% | -0.1 | -4.3% | -12.5% | 18.9% | -24% | 79% |

   None differs from zero.

2. **The test can only rule out large effects.** The standard error of an
   annual Sharpe ratio is 0.75 to 0.80 at every frequency, because the clean
   test sample is 1.56 to 1.75 years long. A strategy would need a true Sharpe
   ratio of about 2.4 to be detected four times out of five, and a single
   experiment a true IC of about 0.06 (monthly), 0.04 (biweekly) or 0.03
   (weekly). The results say that nothing large is there. They cannot say
   that nothing is there.

3. **Nothing else survives the count of what we tried.** The best of the 136
   decile long-short portfolios per frequency has a net Sharpe ratio of 1.41
   (monthly), 1.63 (biweekly) and 0.77 (weekly). The best of 136 portfolios
   with no skill that move together as ours do would be expected to show
   1.36, 1.56 and 1.63. The median is negative at every frequency (-0.18,
   -0.54, -0.59). Across six portfolio rules and three frequencies no
   portfolio passes a 10% false-discovery threshold.

4. **The real news does no better than misdated news.** Stock exposures to LDA
   attention and exposures to the attention of 126 trading days earlier have
   the same test IC: the difference is 0.000 (t = 0.0) monthly, 0.000 (0.0)
   biweekly and -0.003 (-0.3) weekly. Adding text to the price features is
   never a significant improvement in rank IC or in squared error (0 of 90
   paired comparisons each), and at the weekly frequency it lowers the IC in
   all 30.

5. **The one pattern that repeats is in past prices, and it is weak.** With the
   eight price features alone, all eight models have a positive weekly IC in
   the validation period, in the test period (0.014 to 0.031) and in both
   halves of the test period. Their average test IC is 0.024 with t = 1.9,
   below the critical value of 2.11. At 10 basis points per trade it earns
   nothing (net Sharpe ratio -0.14 for the average portfolio), and part of it
   is a tilt towards high-beta stocks in a rising market (section 4.6).

6. **The narrative exposures have a positive IC in 2022-23 and a negative one
   in 2024.** This is the largest pattern in the text results (section 4.7). The 2024 news comes from
   a second scrape with a different format, and seven months cannot separate
   an effect of that change from ordinary variation.

7. **The data decide more than the models do.** Three defects of the data
   would each have produced a result that is not there:
   - *Missing returns.* Our return file has no recent daily returns for stocks
     that are about to drop out of the list of the 1,000 largest. With the
     usual "missing = median" convention, XGBoost on price features has a
     monthly test IC of 0.029 (t = 2.2) and a long-short Sharpe ratio of about
     2. Its IC in the months for which the file is complete is 0.009 (t =
     0.5); fitted on those months only, the same specification has 0.016
     (t = 0.6) and a Sharpe ratio of about zero.
   - *News that is not point-in-time.* The articles were collected ticker by
     ticker at the end of the sample. Companies that had disappeared by then
     have none, and the most covered companies have only their latest ones
     (Apple's begin in June 2022). About half of the stock-months have no
     firm-level news of their own.
   - *Wrong text.* In one of the two public sources, about 7% of the articles
     of 2012 to 2023 carry a text that differs from the other source's, in
     most cases the body of another article, often one written years later.
     Topic models fitted on that "training-period" text had learned the words
     covid and chatgpt.

8. **A single run of a flexible model is one draw from a wide distribution.**
   Rerun with ten random seeds, XGBoost on price and LDA features has a monthly
   test IC between -0.011 and 0.044 (t between -1.1 and 2.4). The seed of the
   main run was the worst of the ten.

9. **Weekly rebalancing gives more observations, not more certainty about the
   Sharpe ratio.** Four times the periods leave the standard error of the
   annual Sharpe ratio where it was, and the trading costs of the median
   decile portfolio rise from 2.9% to 8.0% a year.

10. **No topic model forecasts better than another.** In the validation period
    the FinBERT classes look best, significantly better than LDA at the monthly
    frequency (+0.032, t = 3.2) and just so at the weekly (+0.021, t = 2.19
    against a critical value of 2.19). In the test period they are last at all
    three frequencies. As topic models they do differ: LDA topics are the most
    coherent when all models are judged by one rule, the MiniLM clusters are
    the most even and the most stable, and the twenty FinBERT classes are
    coarse.

11. **The EBM looks best in validation and is among the worst in the test
    period.** Its average IC over the 18 feature sets is 0.021 to 0.024 in
    validation and -0.002, -0.008 and 0.002 in the test period. It stops
    boosting where the validation error is lowest, so its validation numbers
    flatter it. Trees and the EBM are not better than the linear models in the
    test period (difference in IC +0.006, -0.001 and 0.000, all t below 1.2).

**What this means for the earlier project.** We did not re-run the earlier
pipeline. From its code, its out-of-sample numbers came from a universe chosen
with information from later in the year (a stronger form of the first defect
above), topic models fitted on the whole sample, and tuning on the evaluation
months. They should not be quoted until they are reproduced under a protocol
like this one.

**How untouched is the test period?** Less than we would like. These are the
results of the third pass over it.

- After the first run, two reviews found that hiding missing returns from the
  model does not undo the selection of the stocks that have them, and that 7%
  of the article texts were wrong. We dropped the incomplete months, repaired
  the text, refitted the topic models, replaced the elastic net (identical to
  the lasso) by the sparse IPCA of the earlier project, and lowered the tuning
  tolerance from 1e-4 to 1e-5 because the larger value almost always returned
  the first setting.
- After the second run, three reviews found that the EBM that forecast the
  test period was not the model that validation had chosen (it was refitted
  with too few boosting rounds for its pairwise terms), that tied forecasts
  were ordered by a shared random stream, and that t statistics were judged
  by critical values too lenient for samples this short. We now use the
  validated EBM itself, break ties by one fixed order, count a flat forecast
  as an IC of zero, and use small-sample critical values and a bootstrap for
  the search.

None of these changes was chosen by looking for a better test result. The
last one made the headline worse (monthly net Sharpe ratio 0.08 before, -0.15
now; weekly -0.09 before, -0.66 now; biweekly unchanged). The conclusions of
all three passes are the same. The outputs of the earlier passes are not part
of the repository.

**Not repeated after the last fixes.** The audits of `scripts/10` to `12`
(`results/data_audit/`) were run before them. Their rank ICs for models other
than the EBM are unaffected. Their Sharpe ratios used the earlier tie rule and
are quoted here only roughly. The EBM rows of the seed experiment used the
earlier EBM protocol and are left out of this document.

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
previous 252 trading days:

| End of | Stocks | By market value | Of the 50 largest |
|---|---|---|---|
| 2013 | 43% | 38% | 34% |
| 2015 | 49% | 42% | 36% |
| 2017 | 56% | 47% | 44% |
| 2019 | 62% | 53% | 54% |
| 2021 | 61% | 48% | 56% |
| 2022 | 65% | 64% | 60% |
| 2023 | 72% | 71% | 62% |
| 2024 | 45% | 28% | 34% |

For firm-level features this matters twice. About half of the universe has no
features. And "no news" is partly information about the future: it marks, among
others, the companies that later disappeared. We therefore treat a firm without
any article in the previous 126 trading days as not covered and give it the
features of a randomly drawn covered firm (section 2.4 explains the method).
That is the case for 51% of the stock-months of the training period, 39% of the
validation period and 42% of the test period: for those rows the firm-news
block is noise by construction, which pulls every firm-news result towards
zero. We did not evaluate the covered firms on their own. The covered firms
are themselves a sample that survived to the collection date, which no
imputation can undo.

For market-wide attention it means that the early years are measured on the
articles of the less covered half of the market.

### 2.4 Missing returns that reveal the future

Our CRSP extract was built for the earlier project. From 2017 on it holds daily
returns only for stocks that are among the 1,000 largest in June of the same
calendar year. A stock on the June 2021 list that will drop off the June 2022
list therefore has no daily returns from January 2022. We recover its monthly
returns from the monthly file, so portfolio returns are right. But between
January and June its recent price features are missing, and they are missing
because of what happens next.

| Monthly panel, formation days from January to May | Stock-months without recent returns | Their mean return relative to the cross-section |
|---|---|---|
| Training (from 2017) | 6.9% | -2.9% a month |
| Validation | 10.3% | -3.3% a month |
| Test | 7.3% | -5.2% a month |

The usual convention sets a missing rank to the cross-sectional median. A tree
model can find the stocks that sit exactly at the median of several features at
once. `scripts/11_missing_data_leak.py` fits one model four ways (XGBoost on
the price features, tuned as everywhere, test period; Sharpe ratios under the
earlier tie rule):

| Rebalancing | Missing features | Periods | Rank IC (t) | in the incomplete periods | in the complete periods | Long-short Sharpe, gross | net |
|---|---|---|---|---|---|---|---|
| Monthly | median | all 36 | 0.029 (2.2) | 0.059 (3.8) | 0.009 (0.5) | 2.28 | 2.04 |
| Monthly | random donor | all 36 | 0.003 (0.2) | 0.004 (0.2) | 0.002 (0.1) | 0.45 | 0.22 |
| Monthly | median | the 21 complete | 0.016 (0.6) | | 0.016 (0.6) | 0.13 | -0.08 |
| Monthly | random donor | the 21 complete | 0.008 (0.3) | | 0.008 (0.3) | 0.22 | -0.05 |
| Biweekly | median | all 78 | 0.034 (3.0) | 0.064 (5.0) | 0.008 (0.6) | 1.88 | 1.53 |
| Biweekly | random donor | all 78 | 0.009 (0.9) | 0.009 (0.8) | 0.009 (0.7) | 0.97 | 0.58 |
| Biweekly | median | the 42 complete | 0.019 (1.7) | | 0.019 (1.7) | 0.59 | 0.08 |
| Biweekly | random donor | the 42 complete | 0.015 (1.3) | | 0.015 (1.3) | -0.32 | -1.08 |
| Weekly | median | all 156 | 0.041 (3.7) | 0.068 (4.8) | 0.016 (1.1) | 2.12 | 1.56 |
| Weekly | random donor | all 156 | 0.018 (1.7) | 0.022 (1.5) | 0.015 (1.0) | 1.24 | 0.60 |
| Weekly | median | the 81 complete | 0.021 (1.6) | | 0.021 (1.6) | 0.02 | -1.00 |
| Weekly | random donor | the 81 complete | 0.012 (1.0) | | 0.012 (1.0) | -0.33 | -1.65 |

The last row of each frequency is the main run's experiment with other donor
draws and, where validation then chose another tree depth, another model; the
main run has 0.011, 0.012 and 0.024 (section 7.1 shows how much such a model
moves).

With the median convention and every period, the model forecasts -2.4% a month
for the stocks without data (they go on to earn -5.2%), and its predictability
sits in the incomplete periods: an IC of 0.059 there against 0.009 in the
complete ones. This was our own first result, and it looked like a success.

**What we do instead.**

1. *Formation days on which the file is incomplete are left out.* From 2017
   these are the days from January to the day the June list changes. What
   remains of each year is the formation days from the end of June to the end
   of December, that is the holding periods from July to January. All results
   are therefore results for the second half of the year.
2. *A model never sees that a value is missing.* A stock with a missing value
   in a feature block receives the complete feature row of another stock of
   the same day, drawn at random (a "donor"), before ranks are taken. Its
   features then carry no information and nothing marks it. The stock stays in
   the universe and in every portfolio with its true return.

The second rule alone removes the effect on the IC (rows "random donor, all").
The Sharpe ratios of those rows are still positive, and the stocks that do
have data in those months are a sample selected on the future, so we apply the
first rule as well. The price is the sample:

| Periods used (left out) | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Training | 69 (15) | 143 (39) | 290 (75) |
| Validation | 14 (10) | 27 (25) | 53 (51) |
| Test | 21 (15) | 42 (36) | 81 (75) |

In the periods that remain, 2.4% of the stock-months of the training period,
6.8% of validation and 4.8% of test still have donor price features. From 2017
these are stocks new to the June list: the file has their daily returns only
from January of that year, so 12-month momentum cannot be computed for them.
Which stocks are new is known when the portfolio is formed, so this costs
information (about recent risers) but reveals nothing. For narrative exposures
the share is about 1%.

`scripts/11_missing_data_leak.py` reproduces the leak (median convention,
every period). A return file with daily data for every list member
(`scripts/00_pull_crsp.py`, which needs WRDS) removes the problem at the root
and restores the full years.

### 2.5 Other look-ahead in the earlier pipeline

As we read its code (we did not re-run it):

| Problem | Earlier project | Here |
|---|---|---|
| Universe | The 1,000 largest by June market value, applied from January of the same year | The June list is used from the end of June |
| Topic models and vocabulary | Fitted on all text, 2015 to 2023 | Fitted on text up to 2019 |
| Tuning | On the evaluation months; no validation set | On 2020 to 2021 only |
| News timing | The day's news with the day's portfolio | Lagged one trading day |
| Missing data | Visible to the model | Incomplete months dropped, the rest hidden (section 2.4) |

The first row is the same mechanism as section 2.4 in a stronger form: the
earlier universe contained, from January, only stocks known to be large in the
following June.

## 3. Topic models

The models are fitted on 140,209 articles dated up to 2019 and label the same
529,657 articles (at most 150 a day); "LDA, every article" applies the LDA
topics to all 1,255,289. The topic models are therefore in-sample for the
training period of the return models and out of sample after 2019.

| Model | Topics | Coherence (NPMI), training text | later text | Stability across seeds | Largest topic | Effective number of topics | Reliability of attention shocks, to 2019 | after 2019 |
|---|---|---|---|---|---|---|---|---|
| LDA | 60 | 0.58 | 0.37 | 58% | 12.7% | 40.1 | 0.30 | 0.35 |
| MiniLM clusters | 60 | 0.49 | 0.34 | 67% | 4.3% | 53.6 | 0.25 | 0.32 |
| FinBERT classes | 20 | 0.30 | 0.21 | fixed | 21.4% | 11.6 | 0.32 | 0.38 |
| LDA, every article | 60 | 0.58 | 0.37 | 58% | 12.7% | 40.1 | 0.33 | 0.46 |

- **Coherence depends on how top words are picked.** Above, every model's ten
  top words come from one rule (class-based TF-IDF over the topic's articles).
  With the customary rule for LDA (its most probable words) LDA scores 0.35
  and 0.24. A comparison that uses one rule for LDA and the other for
  embedding clusters shows the clusters ahead by 0.14; on equal terms LDA is
  ahead by 0.09. The earlier project compared them the first way, which is
  why its BERT topics looked more coherent.
- **Embedding clusters are more even and more stable**: the largest holds 4.3%
  of articles against 12.7%, and 67% of articles keep their topic under
  another seed against 58%.
- **Coherence falls on later text** for every model (by 37% for LDA, 32% for
  MiniLM, 29% for FinBERT): the topics are not refitted after 2019.
- **Much of the corpus is templated.** Several topics of the LDA and MiniLM
  models are automatic notices: ex-dividend reminders, moving-average alerts,
  changes in ETF shares outstanding, option-chain summaries.
- **FinBERT's twenty classes are broad** (the largest takes 21% of articles;
  they are worth about 12 equally sized topics). The classifier was published
  in December 2022, fine-tuned on a set of finance tweets released that
  September, so it is not point-in-time. It ran on headlines only.
- **`minilm` is the earlier project's "BERT" topic model**: the same encoder
  (published in 2021, so not point-in-time either), spherical k-means and soft
  weights. Three things differ: one passage per article (headline and lead)
  instead of up to three, clusters fitted on training-period text only, and
  150 instead of 120 articles a day.

**Reliability of attention shocks.** Each day's articles are split at random
into two halves and the attention shock of a topic is computed from each half.
The correlation of the two, stepped up to the full sample, says what share of
the shock's variance is signal. For the LDA topics on every article:

| Articles per day | 50 | 100 | 150 | 300 | all |
|---|---|---|---|---|---|
| To 2019 | 0.22 | 0.28 | 0.31 | 0.33 | 0.33 |
| After 2019 | 0.21 | 0.29 | 0.35 | 0.42 | 0.46 |

At 150 articles a day two thirds of a typical topic's daily shock is sampling
noise, so narrative exposures are noisy rankings of stocks. More articles help
only where they exist: before 2020 there are too few a day for it to matter.
And where they do raise reliability, the forecasts do not follow (section
4.3).

## 4. Do the features forecast returns?

### 4.1 The landscape

Average test-period rank IC of the experiments of each kind (over models and
topic models). In brackets the t statistic of the group's average IC per
period, which allows for the experiments of a group being forecasts of the
same returns. Critical values: 2.39, 2.25, 2.11.

| Features | Monthly | Biweekly | Weekly | Weekly: experiments with positive IC |
|---|---|---|---|---|
| Price only | 0.008 (0.4) | 0.011 (0.7) | 0.024 (1.9) | 8 of 8 |
| Narrative exposures | 0.002 (0.2) | -0.003 (-0.5) | 0.001 (0.3) | 17 of 30 |
| Firm-level news | -0.005 (-1.4) | 0.000 (-0.1) | 0.000 (0.1) | 16 of 31 |
| Both text blocks | 0.001 (0.1) | -0.002 (-0.3) | 0.001 (0.1) | 19 of 30 |
| Price and text | 0.009 (0.6) | 0.001 (0.1) | 0.011 (1.2) | 29 of 30 |
| Placebo exposures | 0.012 (0.8) | 0.008 (0.7) | 0.008 (1.2) | 6 of 7 |

No group is distinguishable from zero. The text-only groups are below the
placebo at every frequency. The standard error of one
experiment's mean IC is about 0.018 monthly, 0.012 biweekly and 0.010 weekly
(medians).

Of the 405 single experiments with a forecast, 7 have a test t statistic
beyond the critical value, 3 above it and 4 below; chance alone would give
about 20. The three above are monthly random forests (on price and MiniLM
text, on LDA exposures and on MiniLM text; IC 0.033 to 0.042).

### 4.2 Validation does not carry over

Across experiments the rank correlation between validation IC and test IC is
-0.30 (monthly), -0.28 (biweekly) and 0.06 (weekly). The ten experiments with
the highest validation IC have an average test IC of -0.005, -0.008 and 0.002.

In the validation period 13 experiments have a t statistic above the critical
value. Eleven of them are EBMs or XGBoost models, the two models that stop
boosting where the error on the validation rows is lowest. The 14 validation
months (27 fortnights, 53 weeks) do three jobs: early stopping, the choice of
hyperparameters and the choice between experiments. A high validation IC is
weak evidence, and the test period says so.

### 4.3 Topic model against topic model

Average IC of the 24 text-only experiments of each topic model (exposures, firm
news, both; eight models each), with the t statistic of the group average:

| Topic model | Validation: monthly | biweekly | weekly | Test: monthly | biweekly | weekly |
|---|---|---|---|---|---|---|
| LDA | -0.010 (-1.0) | 0.007 (1.0) | -0.010 (-2.9) | 0.005 (0.7) | 0.002 (0.4) | 0.003 (0.8) |
| LDA, every article | -0.008 (-0.8) | 0.010 (1.0) | 0.001 (0.2) | -0.003 (-0.3) | 0.000 (0.0) | 0.002 (0.4) |
| MiniLM clusters | -0.001 (-0.1) | 0.010 (1.4) | 0.005 (0.7) | 0.004 (0.7) | -0.004 (-0.6) | -0.001 (-0.1) |
| FinBERT classes | 0.022 (1.8) | 0.016 (2.2) | 0.012 (1.4) | -0.009 (-0.6) | -0.005 (-1.0) | -0.001 (-0.2) |

FinBERT against LDA, period by period (`contrasts.csv`):

| | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Validation: difference in IC (t) | +0.032 (3.2) | +0.009 (1.2) | +0.021 (2.2) |
| Test: difference in IC (t) | -0.014 (-0.9) | -0.008 (-1.7) | -0.004 (-0.7) |

The experiment that validation selects for each topic model, among all 32 that
use it, and its test result:

| Topic model | Rebalancing | Chosen on validation | Validation IC (t) | Test IC (t) | Net Sharpe |
|---|---|---|---|---|---|
| LDA | Monthly | Price and text, EBM | 0.035 (0.9) | -0.010 (-0.4) | -0.99 |
| LDA | Biweekly | Exposures, EBM | 0.043 (1.8) | -0.007 (-0.4) | -1.05 |
| LDA | Weekly | Price and text, EBM | 0.048 (2.3) | 0.010 (0.7) | -0.66 |
| LDA, every article | Monthly | Price and text, EBM | 0.028 (0.8) | 0.007 (0.3) | -0.24 |
| LDA, every article | Biweekly | Price and text, EBM | 0.026 (1.1) | -0.008 (-0.4) | -1.12 |
| LDA, every article | Weekly | Price and text, EBM | 0.038 (1.8) | 0.003 (0.2) | -0.97 |
| MiniLM clusters | Monthly | Price and text, sparse IPCA | 0.048 (1.1) | 0.012 (0.3) | -0.02 |
| MiniLM clusters | Biweekly | Text, EBM | 0.043 (1.9) | -0.022 (-1.4) | -0.94 |
| MiniLM clusters | Weekly | Price and text, EBM | 0.041 (1.5) | -0.003 (-0.2) | -0.17 |
| FinBERT classes | Monthly | Text, EBM | 0.063 (3.3) | -0.005 (-0.2) | -0.15 |
| FinBERT classes | Biweekly | Price and text, XGBoost | 0.058 (3.4) | 0.017 (0.8) | 0.37 |
| FinBERT classes | Weekly | Text, EBM | 0.038 (1.6) | -0.022 (-2.6) | -2.41 |

- **There is no order.** FinBERT features look best in validation and are last
  in the test period at all three frequencies; LDA is at the bottom in
  validation and first in the test period. Neither test-period difference is
  significant. Of the twelve selected experiments one has a positive net
  Sharpe ratio, and the one with a significant test IC has it with the wrong
  sign.
- **Every article instead of 150 a day changes nothing for the better.** The
  attention shocks become more reliable (0.46 against 0.35 after 2019) and the
  test ICs are not higher.
- **A finance-specific classifier does not help.** The FinBERT classifier is
  the only component trained on labelled finance text, and it was published
  inside the test period. Its classes are the worst of the four in the test
  period.
- **Why LDA and not BERT, when the earlier paper had it the other way?** The
  differences between topic models here are noise around zero, in both
  directions. The earlier ranking came from in-sample topic models and a
  coherence comparison on unequal terms (section 3).

### 4.4 Text against price

For each model and topic model, the experiment with price and text features is
compared with the same model on price features alone, period by period (32
pairs per frequency, 30 of them with a forecast from both).

| | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Text raises the IC | 18 of 30 | 7 of 30 | 0 of 30 |
| significantly (t above the critical value) | 0 | 0 | 0 |
| Text lowers the IC significantly | 1 | 0 | 0 |
| Average change in IC | 0.000 | -0.010 | -0.013 |
| Lower squared error (Diebold-Mariano t > 2) | 0 | 0 | 0 |
| Higher squared error (t < -2) | 2 | 1 | 2 |
| Higher net Sharpe ratio | 17 of 30 | 10 of 30 | 10 of 30 |
| significantly (bootstrap p < 0.05) | 1 | 2 | 0 |

Where prices carry something (weekly), 41 to 121 additional text features
dilute it in every single case. Four of the 90 differences in Sharpe ratio are
"significant" at 5% (three gains and one loss: monthly EBM with FinBERT
features, -1.83 against 0.37), about the four or five that chance gives.

### 4.5 Placebo

The placebo block is the narrative exposure to the attention shocks of 126
trading days earlier: the same construction with news that cannot be related
to the day's returns. The comparison is paired: the IC of the exposures to LDA
attention minus the IC of the placebo, per period, averaged over the seven
models that produce a forecast from both.

| Period | Rebalancing | Periods | IC, real exposures | IC, placebo | Difference | t |
|---|---|---|---|---|---|---|
| Validation | Monthly | 14 | -0.020 | -0.002 | -0.018 | -0.8 |
| Validation | Biweekly | 27 | 0.016 | 0.004 | +0.012 | 0.6 |
| Validation | Weekly | 53 | -0.011 | 0.000 | -0.011 | -1.1 |
| Test | Monthly | 21 | 0.012 | 0.012 | 0.000 | 0.0 |
| Test | Biweekly | 42 | 0.007 | 0.007 | 0.000 | 0.0 |
| Test | Weekly | 81 | 0.005 | 0.008 | -0.003 | -0.3 |
| 2022-23 | Monthly | 14 | 0.025 | 0.017 | +0.008 | 0.3 |
| 2022-23 | Biweekly | 28 | 0.022 | 0.007 | +0.015 | 0.8 |
| 2022-23 | Weekly | 54 | 0.010 | 0.006 | +0.004 | 0.4 |
| 2024 | Monthly | 7 | -0.013 | 0.002 | -0.015 | -0.4 |
| 2024 | Biweekly | 14 | -0.025 | 0.007 | -0.032 | -1.6 |
| 2024 | Weekly | 27 | -0.005 | 0.012 | -0.017 | -1.3 |

In no period and at no frequency do the real exposures beat the placebo by a
margin that could be told from zero. A narrative exposure is a weighted sum of
the stock's own past daily returns, with weights given by the attention series.
Whatever series supplies the weights, the feature carries some of what past
returns know; the small positive ICs of exposures are not evidence about news.

Two limits of this control: it is one displacement of one topic model, and 126
trading days are two quarters, so the misdated news keeps the calendar rhythm
of earnings seasons.

### 4.6 The weekly price effect

| Price features only | Validation IC | Test IC | Test t | 2022-23 | 2024 | Long-short Sharpe, gross | net | Turnover per week | Break-even cost (bp) |
|---|---|---|---|---|---|---|---|---|---|
| OLS | 0.020 | 0.027 | 1.6 | 0.025 | 0.031 | 0.91 | 0.05 | 122% | 10.6 |
| Ridge | 0.022 | 0.027 | 1.7 | 0.025 | 0.031 | 0.96 | 0.03 | 128% | 10.3 |
| Lasso | 0.002 | 0.021 | 1.4 | 0.020 | 0.023 | 1.05 | 0.17 | 143% | 11.9 |
| IPCA | 0.026 | 0.031 | 1.9 | 0.033 | 0.027 | 1.07 | 0.13 | 131% | 11.4 |
| Sparse IPCA | 0.004 | 0.014 | 0.9 | 0.013 | 0.015 | 0.03 | -0.20 | 49% | 1.2 |
| XGBoost | 0.023 | 0.024 | 1.8 | 0.027 | 0.017 | 0.50 | -0.27 | 119% | 6.5 |
| Random forest | 0.023 | 0.015 | 1.3 | 0.015 | 0.015 | 0.44 | -0.67 | 123% | 4.0 |
| EBM | 0.025 | 0.030 | 2.1 | 0.032 | 0.025 | 0.76 | -0.15 | 128% | 8.4 |
| Average of the eight | 0.018 | 0.024 | 1.9 | 0.024 | 0.023 | 0.93 | -0.14 | 118% | |

The last row gives the average IC, the t statistic of the average IC per week
and the Sharpe ratio of the equal-weighted average of the eight portfolios.

- The sign is the same for every model in every period, and the size is the
  same in both halves of the test period. That is more than can be said of
  any text feature.
- It is not significant: the t statistic of the average is 1.9 in the test
  period (critical value 2.11) and 1.0 in the validation period.
- It is not a clean reversal effect. OLS, ridge and IPCA put negative weights
  on the returns of the last 5 and 21 days and on idiosyncratic volatility,
  and positive weights on beta (the largest OLS coefficient) and on 12-month
  momentum. The average portfolio has a market beta of 0.19 and its
  three-factor alpha before costs has t = 1.2. The average IC is 0.037 in the
  45 weeks in which the market rose and 0.007 in the 36 in which it fell.
- It does not pay. Costs would have to stay below 1 to 12 basis points per
  trade for the eight portfolios to break even, with a turnover above 110% a
  week for all but one.
- Portfolios are formed at the closing prices that end the reversal signal,
  so bid-ask bounce is included.
- At the monthly and biweekly frequencies the price features show nothing
  (0.008 and 0.011, t below 1).

### 4.7 The two halves of the test period

We decided before the first run to report 2022-23 and 2024 separately, because
the 2024 news comes from a second scrape (section 2.1). Average IC by kind of
feature set, with the t statistic of the group average:

| Features | 2022-23: monthly | biweekly | weekly | 2024: monthly | biweekly | weekly |
|---|---|---|---|---|---|---|
| Price only | -0.014 (-0.6) | 0.005 (0.2) | 0.024 (1.4) | 0.050 (1.5) | 0.023 (0.8) | 0.023 (1.3) |
| Narrative exposures | 0.011 (1.6) | 0.005 (0.7) | 0.007 (1.3) | -0.018 (-1.1) | -0.021 (-2.3) | -0.010 (-1.2) |
| Firm-level news | -0.005 (-0.9) | -0.001 (-0.2) | 0.001 (0.7) | -0.005 (-2.0) | 0.001 (0.5) | -0.002 (-1.0) |
| Both text blocks | 0.013 (1.7) | 0.006 (0.9) | 0.007 (1.2) | -0.024 (-1.3) | -0.018 (-1.8) | -0.011 (-1.5) |
| Price and text | 0.005 (0.2) | 0.002 (0.2) | 0.015 (1.2) | 0.016 (1.6) | -0.003 (-0.3) | 0.003 (0.4) |
| Placebo exposures | 0.017 (0.8) | 0.009 (0.5) | 0.006 (0.7) | 0.002 (0.1) | 0.007 (0.5) | 0.012 (1.6) |

Periods: 14, 28 and 54 in 2022-23; 7, 14 and 27 in 2024. Critical values:
2.61, 2.39 and 2.18 in 2022-23; 3.27, 2.61 and 2.29 in 2024.

Narrative exposures have a positive IC in 2022-23 and a negative one in 2024
at all three frequencies. The test of the difference between the halves
(`halves.csv`) gives z = 1.6, 2.2 and 1.7 for the exposures and 1.9, 2.0 and
1.9 for both text blocks, against critical values of 3.27, 2.61 and 2.29 for
halves this short. By topic model it is beyond the critical value in two of
twelve cases (LDA biweekly, z = 3.5; MiniLM weekly, z = 2.7). The three chosen
experiments of section 1 show the same: an IC of 0.029, 0.025 and 0.022 in
2022-23 and of -0.072, 0.000 and -0.016 in 2024, with net Sharpe ratios of
1.13, 0.68 and -0.11 against -6.12, -0.49 and -2.23.

So there is a break in the exposures, with the same sign at all three
frequencies and in 10 of the 12 topic-model cases, and it is not individually
significant. Two readings fit it:

- *The second scrape.* In 2024 the headlines are rebuilt from addresses,
  coverage halves, tagging changes and about seven weeks of news are missing
  (section 2.1). Topic models fitted on earlier text may read this
  text differently.
- *Ordinary variation.* An exposure is a loading on past co-movement.
  Forecasts built on loadings do well or badly for months at a time, as factor
  returns do. In the validation period the same exposures were not
  distinguishable from zero (-0.003, 0.014 and 0.003; t = -0.3, 1.5 and 0.5),
  and against the placebo they are not
  significantly better in 2022-23 either (section 4.5).

Seven months cannot separate the two. What can be said is that 2022-23 alone
does not show a significant effect either: no text cell of that half reaches
its critical value.

## 5. Portfolios

Test period, over the experiments of each frequency that take a position. A
flat forecast (27 lasso experiments in all) holds nothing under the long-short
rules and the universe under the long-only rules, and is left out.

| Rule | Median net Sharpe: monthly | biweekly | weekly | Positive net Sharpe: monthly | biweekly | weekly | Weekly: turnover per period | cost per year | volatility |
|---|---|---|---|---|---|---|---|---|---|
| Decile long-short, equal weights | -0.18 | -0.46 | -0.56 | 46 of 133 | 37 of 136 | 22 of 136 | 77% | 8.0% | 12.7% |
| Decile long-short, value weights | -0.30 | -0.43 | -0.66 | 46 of 133 | 39 of 136 | 30 of 136 | 93% | 9.6% | 17.8% |
| Rank-weighted long-short | -0.23 | -0.59 | -0.60 | 35 of 133 | 31 of 136 | 22 of 136 | 53% | 5.5% | 7.8% |
| Mean-variance long-short | -0.65 | -1.07 | -1.67 | 37 of 133 | 8 of 136 | 4 of 136 | 62% | 6.5% | 3.5% |
| Long-only top decile, equal weights | 0.39 | 0.40 | 0.27 | 133 of 133 | 136 of 136 | 133 of 136 | 41% | 4.2% | 22.3% |
| Long-only top decile, value weights | 0.45 | 0.42 | 0.28 | 127 of 133 | 119 of 136 | 104 of 136 | 47% | 4.9% | 21.0% |

The universe itself, for reference (Sharpe ratio and maximum drawdown):

| | Validation: equal-weighted | value-weighted | Test: equal-weighted | value-weighted | Test drawdown: equal-weighted | value-weighted |
|---|---|---|---|---|---|---|
| Monthly | 1.49 | 1.70 | 0.47 | 0.52 | -14% | -13% |
| Biweekly | 2.03 | 2.36 | 0.61 | 0.61 | -15% | -15% |
| Weekly | 1.94 | 2.14 | 0.53 | 0.47 | -19% | -18% |

- **The median long-short portfolio loses money after costs** at every
  frequency and under every rule: a forecast without information still trades.
  Before costs the median equal-weighted decile portfolio has a Sharpe ratio
  of 0.05, 0.05 and 0.11; after costs it returns -2.5%, -5.0% and -7.2% a
  year, with a median maximum drawdown of -13%, -17% and -18%. None of the
  equal-weighted decile portfolios has a three-factor alpha with a t statistic
  above the critical value; 14, 22 and 43 have one below its negative.
- **Mean-variance weighting does worst.** It has the lowest volatility (3.5% a
  year at the weekly frequency), so the same costs weigh most on it.
- **The long-only rules earn the market's return and a little less.** Their
  Sharpe ratios are positive because the universe's is. Against the universe
  with the same weighting, the median information ratio of the equal-weighted
  top decile is -0.20, -0.48 and -0.59, and that of the value-weighted top
  decile 0.03, -0.19 and -0.30.

**Counting the search.** Net Sharpe ratios of the 136 experiments (the placebo
is a control and not part of the search). For the long-only rules the return
is measured against the universe, so the Sharpe ratio is an information ratio.
"Expected best with no skill" is the average best Sharpe ratio over 5,000
bootstrap resamples in which every portfolio's mean return is set to zero and
their co-movement is kept; "p of the best" is the share of resamples with a
best at least as high (White's reality check).

| Rule | Rebalancing | Best | which is | Median | Expected best with no skill | p of the best | SPA p | Significant at 5%, unadjusted | At a 10% false discovery rate |
|---|---|---|---|---|---|---|---|---|---|
| Decile long-short, equal weights | Monthly | 1.41 | LDA text, random forest | -0.18 | 1.36 | 0.41 | 0.14 | 3 | 0 |
| | Biweekly | 1.63 | LDA text, XGBoost | -0.54 | 1.56 | 0.39 | 0.42 | 1 | 0 |
| | Weekly | 0.77 | Price and LDA text, IPCA | -0.59 | 1.63 | 0.99 | 0.74 | 0 | 0 |
| Decile long-short, value weights | Monthly | 1.47 | MiniLM exposures, random forest | -0.31 | 1.48 | 0.47 | 0.56 | 2 | 0 |
| | Biweekly | 1.50 | LDA exposures, random forest | -0.43 | 1.65 | 0.61 | 0.62 | 3 | 0 |
| | Weekly | 1.45 | LDA firm news, random forest | -0.66 | 1.73 | 0.74 | 0.76 | 0 | 0 |
| Rank-weighted long-short | Monthly | 0.95 | LDA exposures (every article), random forest | -0.24 | 1.31 | 0.84 | 0.58 | 0 | 0 |
| | Biweekly | 1.22 | LDA text, XGBoost | -0.62 | 1.53 | 0.76 | 0.68 | 0 | 0 |
| | Weekly | 0.56 | FinBERT text, IPCA | -0.61 | 1.54 | 1.00 | 0.94 | 0 | 0 |
| Mean-variance long-short | Monthly | 1.19 | LDA exposures (every article), random forest | -0.64 | 1.50 | 0.76 | 0.69 | 0 | 0 |
| | Biweekly | 1.13 | LDA exposures (every article), IPCA | -1.07 | 1.67 | 0.93 | 0.86 | 0 | 0 |
| | Weekly | 0.51 | LDA exposures (every article), XGBoost | -1.70 | 1.77 | 1.00 | 0.98 | 0 | 0 |
| Long-only, equal weights | Monthly | 1.32 | Price and MiniLM text, ridge | -0.21 | 1.41 | 0.55 | 0.48 | 0 | 0 |
| | Biweekly | 1.09 | LDA firm news (every article), lasso | -0.51 | 1.60 | 0.90 | 0.87 | 0 | 0 |
| | Weekly | 0.81 | Price and LDA text, IPCA | -0.59 | 1.70 | 0.99 | 0.86 | 0 | 0 |
| Long-only, value weights | Monthly | 2.08 | Price and LDA text, ridge | 0.00 | 1.47 | 0.07 | 0.03 | 5 | 0 |
| | Biweekly | 1.45 | FinBERT text, sparse IPCA | -0.25 | 1.67 | 0.67 | 0.42 | 3 | 0 |
| | Weekly | 2.05 | LDA firm news, random forest | -0.32 | 1.80 | 0.26 | 0.38 | 1 | 0 |

(Medians here are over the searched experiments; the first table of this
section also includes the placebo experiments that take a position: 7, 8 and
7.)

In 14 of the 18 rows the best portfolio is below what the best of as many
unskilled portfolios would show, and in none is it significantly above. The
lowest p-value of the eighteen rows is 0.07 (reality check) or 0.03
(SPA) for the monthly value-weighted long-only rule, one of eighteen rows; no
portfolio passes the false-discovery threshold. The control makes the same
point from the other side: under the monthly rank-weighted rule a placebo
portfolio (1.46) beats all 136 real ones, and under the monthly decile rule
the best placebo portfolio (1.38) would be second.

**The chosen experiments under every rule**, test period, net Sharpe ratio
(for the long-only rules that of the portfolio itself, not against the
universe):

| | Long-short EW | Long-short VW | Rank | Mean-variance | Long-only EW | Long-only VW | Max drawdown, long-short EW |
|---|---|---|---|---|---|---|---|
| Monthly: FinBERT text, EBM | -0.15 | -0.28 | -0.27 | -0.75 | 0.31 | 0.20 | -17% |
| Biweekly: price and FinBERT text, XGBoost | 0.37 | -0.08 | 0.36 | -0.30 | 0.49 | 0.58 | -17% |
| Weekly: price and LDA text, EBM | -0.66 | -0.72 | -0.59 | -1.07 | 0.32 | 0.34 | -24% |

## 6. Monthly, biweekly or weekly?

| | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Periods: training / validation / test | 69 / 14 / 21 | 143 / 27 / 42 | 290 / 53 / 81 |
| Training rows | 67,791 | 140,333 | 284,639 |
| Years of test data | 1.75 | 1.62 | 1.56 |
| Standard error of a mean IC, test (median) | 0.018 | 0.012 | 0.010 |
| Standard error of an annual Sharpe ratio (median) | 0.75 | 0.79 | 0.80 |
| Median turnover per period, decile long-short | 119% | 91% | 77% |
| Median cost per year, decile long-short | 2.9% | 4.7% | 8.0% |
| Median Sharpe, decile long-short (placebo included): gross | 0.05 | 0.05 | 0.11 |
| net | -0.18 | -0.46 | -0.56 |
| Fitting time, 144 experiments (4 cores) | 8 min | 11 min | 17 min |

- **More periods do not buy certainty about the Sharpe ratio.** Its standard
  error depends on the number of years, 0.75 to 0.80 here at any frequency.
  The standard error of a mean IC falls with the number of periods, but a
  weekly IC and a monthly IC are different quantities, and what a strategy
  earns in a year is what matters.
- **More rows do not help a model that has nothing to find.** With four times
  the rows the validation period chooses no deeper trees, no more factors and
  no more pairwise terms (section 7).
- **Costs rise with the number of rebalances**, from 2.9% to 8.0% a year. They
  take about 0.2 off the median Sharpe ratio monthly and about 0.7 weekly.
- **Weekly is where the only stable pattern is**, and it is a price pattern
  that does not survive costs.
- A caution for the higher frequencies: in holding periods that reach into
  January, the stocks that will leave the list in June have only a monthly
  return in our file, which is spread evenly over the days of the month. That
  concerns one period a year.

For this project the answer is that the frequency does not matter, because
there is no signal for it to sharpen. The weekly panel is the place to look if
a point-in-time news source becomes available: it has the most periods, and
costs can be dealt with separately.

## 7. Models

Average IC of each model over the 18 feature sets (the placebo included), with
the t statistic of the average IC per period:

| Model | Validation: monthly | biweekly | weekly | Test: monthly | biweekly | weekly | Feature sets with positive test IC, of 18: M / 2W / W |
|---|---|---|---|---|---|---|---|
| OLS | -0.016 (-1.1) | 0.003 (0.6) | -0.002 (-0.3) | -0.003 (-0.6) | -0.001 (-0.1) | 0.005 (1.1) | 5 / 7 / 13 |
| Ridge | -0.015 (-1.0) | 0.009 (0.9) | 0.001 (0.2) | 0.002 (0.4) | -0.004 (-0.5) | 0.007 (1.3) | 9 / 6 / 13 |
| Lasso | 0.009 (0.6) | 0.009 (1.1) | 0.007 (1.1) | -0.003 (-0.3) | -0.004 (-0.6) | 0.004 (0.8) | 3 / 3 / 7 |
| IPCA | 0.008 (0.5) | 0.001 (0.1) | -0.002 (-0.3) | 0.003 (0.2) | 0.005 (0.7) | 0.005 (1.4) | 9 / 13 / 12 |
| Sparse IPCA | 0.014 (1.4) | 0.001 (0.0) | -0.006 (-0.7) | 0.002 (0.2) | 0.004 (0.4) | 0.001 (0.2) | 8 / 12 / 8 |
| XGBoost | 0.003 (0.4) | 0.014 (2.6) | 0.008 (1.6) | 0.006 (1.1) | 0.002 (0.7) | 0.006 (1.9) | 12 / 9 / 16 |
| Random forest | -0.006 (-0.5) | 0.010 (0.9) | 0.006 (1.0) | 0.013 (1.3) | 0.005 (0.8) | 0.007 (1.9) | 12 / 12 / 17 |
| EBM | 0.021 (1.6) | 0.024 (2.3) | 0.022 (2.2) | -0.002 (-0.3) | -0.008 (-1.1) | 0.002 (0.3) | 8 / 5 / 9 |

(The lasso averages and its counts in the last column are over the feature
sets where its forecast is not flat: 7, 10 and 10 of 18.)

- **No model stands out in the test period.** No cell reaches its critical
  value. Boosted trees, forest and EBM together against the five linear models
  (`contrasts.csv`): +0.006 (t = 1.1), -0.001 (-0.2) and 0.000 (0.1).
- **The EBM is the clearest case of validation flattering a model.** It has
  the best validation average at every frequency and is last or second to last
  in the test period at two of three. In validation the three flexible models
  beat the linear ones significantly at the biweekly and weekly frequencies
  (+0.013, t = 4.1; +0.014, t = 3.3); in the test period the difference is
  gone. They stop boosting, or are tuned, on the rows they are then scored on.
- **The EBM forecasts the test period with the model that validation chose.**
  The library cannot refit its two stages with separate numbers of rounds, so
  the model fitted on the training period and stopped on the validation
  period is used as it is. The other models are refitted on training plus
  validation data with the chosen setting.
- **What validation chose.** The linear models and the forest are pushed to
  their most regularised settings: the lasso to the empty model in 31 of 54 cases (after
  the refit 27 are still empty), ridge to the largest of its 25 penalties in
  34, sparse IPCA to one factor in 40 and to its strongest penalty in 43, IPCA
  to one factor in 29, the random forest to depth 3 in 42. XGBoost and the EBM
  go the other way about as often as not: XGBoost takes its deepest trees
  (depth 5) in 21 of 54 cases and depth 1 in 17, and the EBM takes its ten
  pairwise terms in 31. The median gap in validation R2 between the chosen
  setting and the next best is 0.003 percentage points.
- **OLS** on 8 to 129 noisy features has a validation R2 of -0.47%, -0.13%
  and -0.06% (medians); every other model is within 0.02% of zero.
- **Sparse IPCA against IPCA.** The earlier project's estimator keeps a median
  of 68% of the features (between 27% and 100%). It does not forecast better:
  0.002, 0.004 and 0.001 against 0.003, 0.005 and 0.005. On the weekly price
  features it has half of IPCA's IC (0.014 against 0.031).
- Fitting all 432 experiments takes 36 minutes on four laptop cores: 17 for
  the EBMs, 9.5 for the forests, 3.7 for sparse IPCA, 3.2 for XGBoost, 2 for
  IPCA.
  The linear models are solved from second moments in milliseconds.

### 7.1 How much of a result is the seed?

Boosted trees draw rows and columns at random. With
`scripts/12_tuning_noise.py` seven XGBoost experiments were run again with ten
seeds each (seed 0 is the experiment of the main run), tuning included.

| Rebalancing | Experiment | Settings chosen | Validation IC | Test IC | Standard deviation over seeds | Test t | Seeds with t > 2 |
|---|---|---|---|---|---|---|---|
| Monthly | Price, XGBoost | 4 | -0.008 to 0.006 | 0.002 to 0.031 | 0.008 | 0.1 to 1.5 | 0 |
| Monthly | Price and LDA text, XGBoost | 4 | -0.034 to 0.031 | -0.011 to 0.044 | 0.019 | -1.1 to 2.4 | 4 |
| Biweekly | Price, XGBoost | 4 | 0.001 to 0.028 | 0.011 to 0.020 | 0.003 | 0.7 to 2.0 | 1 |
| Biweekly | Price and LDA text, XGBoost | 4 | -0.020 to 0.022 | -0.013 to 0.022 | 0.010 | -0.7 to 2.5 | 2 |
| Biweekly | Price and FinBERT text, XGBoost (chosen) | 3 | 0.024 to 0.058 | -0.001 to 0.017 | 0.005 | -0.1 to 0.8 | 0 |
| Weekly | Price, XGBoost | 3 | 0.019 to 0.026 | 0.015 to 0.024 | 0.003 | 1.3 to 1.8 | 0 |
| Weekly | Price and LDA text, XGBoost | 4 | -0.016 to 0.023 | -0.010 to 0.020 | 0.011 | -1.8 to 2.5 | 1 |

For comparison, the sampling error of one experiment's test IC is 0.018,
0.012 and 0.010.

- **For XGBoost on 129 features the seed matters as much as the sample.** The
  standard deviation over seeds (0.010 to 0.019) is the size of the sampling
  error. Validation picks a different tree depth from seed to seed.
- **Selection picks lucky draws.** For the biweekly headline experiment, seed
  0 has the highest validation IC of the ten and the highest test IC. With any
  other seed (validation IC 0.024 to 0.037) it would not have been selected.
- **A single result can mislead in both directions.** The main run's monthly
  XGBoost on price and LDA features (-0.011) is the worst of its ten seeds;
  four others have t > 2 (one above the critical value of 2.39). Neither is
  the truth: the average over seeds is 0.018, one standard error from zero.
- On the eight price features XGBoost is stable (standard deviation 0.003 to
  0.008).

Only the XGBoost reruns are reported here (the EBM reruns used the earlier
protocol and are left out); the forests were not rerun. Averaging a tree model
over
seeds would remove this noise at ten times the fitting time (section 9).

## 8. Opening the glass box

For each frequency `scripts/09_explain.py` takes the EBM on price and text
features with the best validation IC and reports term importance, shape
functions and ablations in the manner of the Glass Box paper: a term or block
is deleted from the fitted model and nothing is re-estimated.

| | Monthly (LDA) | Biweekly (FinBERT) | Weekly (LDA) |
|---|---|---|---|
| Boosting rounds kept for the main effects | 1 | 54 | 3 |
| Share of importance: narrative exposures | 54% | 41% | 54% |
| pairwise terms | 21% | 26% | 19% |
| firm news | 15% | 13% | 18% |
| price | 10% | 20% | 9% |
| Test IC, full model | -0.010 | -0.014 | 0.010 |
| after deleting the pairwise terms | 0.013 | -0.006 | 0.013 |
| after deleting the narrative block | -0.019 | -0.002 | 0.015 |
| after deleting the firm-news block | -0.008 | -0.016 | 0.009 |
| after deleting the price block | -0.019 | -0.020 | 0.003 |

- **The monthly and the weekly model have almost nothing in them.** Validation
  stopped the first after one boosting round and the second after three: their
  "shape functions" are a handful of small steps. What follows describes what
  the method shows, not a finding about returns.
- **Importance is not usefulness.** The models put 41% to 54% of their weight
  on narrative exposures. Deleting that block improves the biweekly and the
  weekly model.
- **The price block is the only one whose deletion lowers the IC at all three
  frequencies**, with 9% to 20% of the importance.
- **Pairwise terms hurt**: deleting them raises the IC at all three
  frequencies.

None of these differences is tested, and with ICs of this size none would be
significant. The earlier project's ablation (deleting one narrative from the
fitted IPCA) has the same form. It should be redone on this panel before it
is quoted.

## 9. Limits, and what to do next

**Limits of what was done.**

- *Short test sample, second half of the year only.* 1.56 to 1.75 years, and
  only holding periods from July to January (section 2.4).
- *The validation period is small and does three jobs* (section 4.2).
- *Three different yardsticks.* Hyperparameters are tuned on squared error,
  experiments are chosen on rank IC, and the portfolios use only the ordering
  of the forecasts.
- *The rank IC is not adjusted for beta or industry*, and returns are not
  winsorised.
- *Costs are a flat 10 basis points*, and portfolios trade at the close that
  ends the signal.
- *Firm-news features are noise for 39% to 51% of the rows* (section 2.3).
- *Text features partly re-encode prices*: a narrative exposure is a function
  of the stock's past returns.
- *Two components are not point-in-time*: the FinBERT classifier (2022) and
  the MiniLM encoder (2021).
- *One seed per experiment* (section 7.1).

**What to do next.**

- **A complete return file.** The largest improvement available: with daily
  returns for every list member, the 40 monthly periods that are left out
  come back (15 of them in the test period), the test sample grows by 70% to
  90%, and the donor rule is needed for news only. `00_pull_crsp.py` does it for
  anyone with WRDS access; it is untested here.
- **A point-in-time news source.** The firm-level results are limited by
  coverage (section 2.3), not by models.
- **NoLBERT.** Not run: on our CPU it encodes about 5 articles a second (about
  200 for MiniLM), which is more than a day for the article sample. The script
  that adapts it to training-period news (`03c_finetune_encoder.py`) is
  written and was tested end to end on a small encoder; with a GPU the two
  steps take minutes. Its untuned embeddings put all articles close together
  (mean cosine 0.7), so use `04_fit_topics.py embedding --center`, or
  fine-tune. Given section 4, expect cleaner topics rather than
  predictability. It would replace the encoder of `minilm`, which is not
  point-in-time.
- **Seed averaging.** Fit XGBoost and the forest with ten seeds and average
  the forecasts (section 7.1). It would not create a signal, but it would
  stop single experiments from looking like one.
- **Less noisy attention.** Fewer topics, or weekly instead of daily
  attention, would raise reliability (section 3). The 60 topics and the daily
  shock were kept from the earlier project and are not tuned.
- **Refitting topics through time.** Topics are fixed at 2019. An expanding
  refit would let new subjects in, at the price of features whose meaning
  changes.
- **The audits again.** Rerun `scripts/10` to `12` under the final rules, and
  add an evaluation of firm-news features on covered firms only.

## 10. Where each number comes from

| Section | File |
|---|---|
| 1, 4.2 | `results/tables/selection.csv`, `forecast.csv`, `strategies.csv`, `multiple_testing.csv` |
| 2.1 | `results/data_audit/news_coverage.csv`; tagging and headline case from `data/news/articles/` |
| 2.2 | `results/data_audit/text_repair.csv` |
| 2.3 | `results/data_audit/firm_coverage.csv`, `donor_rows.csv` |
| 2.4 | `results/data_audit/stale_rows.csv`, `missing_data_leak.csv`, `donor_rows.csv` |
| 3 | `results/tables/topic_models.csv`, `data/topics/<model>/quality.json`, `topics.csv` |
| 4.1, 4.6, 7 | `results/tables/landscape.csv`, `forecast.csv`, `strategies.csv` |
| 4.3 | `results/tables/landscape.csv`, `contrasts.csv`, `selection.csv` |
| 4.4 | `results/tables/text_vs_price.csv` |
| 4.5 | `results/tables/placebo.csv` |
| 4.7 | `results/tables/landscape.csv`, `halves.csv`, `selection.csv` |
| 5 | `results/tables/strategies.csv`, `multiple_testing.csv`, `frequency.csv` |
| 6 | the same, and `forecast.csv`, `tuning.csv`, `results/data_audit/stale_rows.csv` |
| 7 | `results/tables/landscape.csv`, `tuning.csv`, `contrasts.csv`, `results/tuning/`, `results/importance/<freq>/` |
| 7.1 | `results/data_audit/tuning_noise.csv`, `results/tables/forecast.csv` |
| 8 | `results/explain/<freq>__<features>/`, `results/tables/tuning.csv` |

Counts and medians "over experiments" are taken over the rows of these tables.
The split of the weekly price IC by the sign of the market return (section
4.6) is computed from `results/series/W_forecast.parquet` and
`W_factors.csv`. The article counts by year before 2012, the named companies
of section 2.3 and the size of the dump of 16 December 2023 come from the
audit of the raw files (`ftm/data/fnspid.py`, `ftm/data/finmultitime.py`) and
of `data/news/links.parquet`.
