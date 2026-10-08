# Findings

Results of the run of 8 October 2026. 26 feature sets (two NoLBERT topic
models among them), 15 models (the eight of the design, six of them also
fitted on ranked returns, and the ensemble of the eight) and 3 rebalancing
frequencies: 1,170 experiments, each traded under seven portfolio rules.
A second part of the run (section 10) added the tone of the news, an
embedding regression, tuned attention-shock windows, forecasts that fit
nothing and a cap-weighted tilt: 552 more experiments, 1,722 in all, under
eight rules. Sections 1 to 9 describe the main run; their tables are computed
over its experiments only and did not change.
Training 2013 to 2019, validation 2020 to 2021, test 2022 to 2024. From 2017
on, only the part of each year for which our return file is complete is used
(section 2.4), which leaves 21 test months (42 fortnights, 81 weeks), about a
year and three quarters of returns. Every number below comes from a file under
`results/` (section 13 says which).

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
- *Decile long-short, equal weights* (`ls_ew`) is the portfolio behind every
  Sharpe ratio that does not name another rule. *Buffered* (`ls_buf`) is the
  same portfolio with a buffer: a stock enters a leg in the top (bottom) decile
  and stays until it leaves the top (bottom) fifth.
- *Base models*: OLS, ridge, lasso, IPCA, sparse IPCA, XGBoost, random forest,
  EBM. The groups of sections 4 to 6 are made of them; the variants of section
  9 are compared with them separately.

## 1. The short version

1. **The topic features show no out-of-sample forecasting power.** At each
   frequency one experiment is chosen on the validation period (the highest
   rank IC of the 200 base experiments on real features) and evaluated on the
   test period:

   | Rebalancing | Chosen on validation | Validation rank IC (t) | Test rank IC (t) | Long-short Sharpe, gross | net of costs | 95% interval, net |
   |---|---|---|---|---|---|---|
   | Monthly | Price and NoLBERT text features, sparse IPCA | 0.068 (1.8) | 0.006 (0.1) | 0.05 | -0.05 | -1.18 to 1.62 |
   | Biweekly | Price and FinBERT text features, XGBoost | 0.058 (3.4) | 0.017 (0.8) | 0.79 | 0.37 | -1.18 to 1.89 |
   | Weekly | Price and LDA text features, EBM | 0.048 (2.3) | 0.010 (0.7) | -0.23 | -0.66 | -2.17 to 0.63 |

   | | Test R2 | Clark-West t | Return a year, gross | net | Volatility | Maximum drawdown | Turnover per period |
   |---|---|---|---|---|---|---|---|
   | Monthly | -0.020% | -0.0 | 1.2% | -1.3% | 25.9% | -31% | 106% |
   | Biweekly | -0.078% | 1.0 | 12.5% | 5.9% | 15.9% | -17% | 127% |
   | Weekly | -0.001% | -0.1 | -4.3% | -12.5% | 18.9% | -24% | 79% |

   None differs from zero. Biweekly and weekly the choice is the same as in the
   previous run; monthly, a NoLBERT experiment now has the highest validation IC.

2. **One improvement is real: trading less.** Decided on the validation period
   before the test period was looked at, the buffered portfolio raises the net
   return of the test period averaged over all 200 base experiments by 0.6%,
   1.5% and 2.0% a year (monthly, biweekly, weekly; t = 2.0, 3.5 and 5.2, the
   last two beyond their critical values), with a turnover a fifth to nearly
   a third lower. For the three experiments above:

   | Buffered | Sharpe, gross | net | Return a year, net | Maximum drawdown | Turnover per period |
   |---|---|---|---|---|---|
   | Monthly | 0.12 | 0.05 | 1.2% | -28% | 71% |
   | Biweekly | 0.74 | 0.41 | 6.2% | -16% | 99% |
   | Weekly | -0.04 | -0.33 | -5.6% | -17% | 46% |

   It saves costs; it does not create a signal.

3. **The other levers do little or nothing** (section 9). An ensemble of the
   eight models has about the IC of its average member and a better net Sharpe
   ratio than the average member in 60% to 72% of the feature sets. Fitting
   the models on ranked returns raised the IC slightly in validation and by
   less or not at all in the test period (differences of 0.000, -0.002 and
   +0.002). Twenty NoLBERT topics instead of sixty make the attention shocks
   more reliable (0.50 against 0.34 after 2019) and the forecasts no better
   (higher only monthly, and not significantly). The protocol we fixed in advance for this run, the validation
   choice among the 25 ensembles traded with the buffer, picks the FinBERT
   text features at all three frequencies and loses (net Sharpe -0.21, -0.34,
   -0.16).

4. **NoLBERT is the best text arm at the higher frequencies, inside noise**
   (section 8). Averaged over its text-only experiments, NoLBERT has the
   highest test IC of the six topic models at the biweekly and weekly
   frequencies (weekly 0.006, t = 1.4, 21 of 22 positive), ahead of each
   other topic model in all ten such comparisons, none significant. Monthly it
   is fifth of the six, ahead only of FinBERT, and in the seven months of 2024
   its monthly IC (-0.031) is the second lowest, after FinBERT's. Our
   teammate's numbers reproduce exactly on this machine.

5. **The test can only rule out large effects.** The standard error of an
   annual Sharpe ratio is 0.75 to 0.80 at every frequency, because the clean
   test sample is 1.56 to 1.75 years long. A strategy would need a true Sharpe
   ratio of about 2.4 to be detected four times out of five, and a single
   experiment a true IC of about 0.06 (monthly), 0.04 (biweekly) or 0.03
   (weekly). The results say that nothing large is there. They cannot say
   that nothing is there.

6. **Nothing survives the count of what we tried.** Over all 375 searched
   experiments per frequency (variants included), the best decile long-short
   portfolio has a net Sharpe ratio of 1.51 (monthly), 1.63 (biweekly) and
   0.97 (weekly); the best of 375 portfolios with no skill that move together
   as ours do would be expected to show 1.55, 1.70 and 1.85. The median is
   negative at every frequency. In 17 of the 21 combinations of rule and
   frequency the best result is below that benchmark, and no portfolio passes
   a 10% false-discovery threshold. Counting the 552 experiments of section 10
   as well (451 to 613 per frequency, eight rules), the best result is below
   the benchmark in 15 of 24 combinations and above it in 9 (7 of them
   monthly); only the monthly value-weighted long-short beats it clearly
   (best 3.08 against 1.71, reality-check p = 0.004), and still nothing passes
   the false-discovery threshold (section 10.6).

7. **The real news does no better than misdated news.** Exposures to LDA
   attention and to the attention of 126 trading days earlier have the same
   test IC: a difference of 0.000 (t = 0.0) monthly, 0.000 (0.0) biweekly and
   -0.003 (-0.3) weekly. Adding text to the price features is never a
   significant improvement in rank IC or in squared error (0 of 134 paired
   comparisons), and at the weekly frequency it lowers the IC in 42 of 44.

8. **The one pattern that repeats is in past prices, and it is weak.** With
   the eight price features alone, all eight models have a positive weekly IC
   in the validation period, in the test period (0.014 to 0.031) and in both
   halves of the test period. Their average test IC is 0.024 with t = 1.9,
   below the critical value of 2.11. Before costs the eight portfolios have
   Sharpe ratios of up to 1.07; after costs between -0.67 and 0.17, and part of
   the effect is a tilt towards high-beta stocks in a rising market (section
   4.6).

9. **The narrative exposures have a positive IC in 2022-23 and a negative one
   in 2024** (section 4.7). The 2024 news comes from a second scrape with a
   different format, and seven months cannot separate an effect of that change
   from ordinary variation.

10. **The data decide more than the models do.** Three defects of the data
    would each have produced a result that is not there (section 2):
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

11. **A single run of a flexible model is one draw from a wide distribution.**
    Rerun with ten random seeds, XGBoost on price and LDA features has a
    monthly test IC between -0.011 and 0.044 (t between -1.1 and 2.4). The
    seed of the main run was the worst of the ten.

12. **No topic model forecasts reliably better than another, and of the base
    models the EBM is the one that validation flatters most.** FinBERT's classes look best in
    validation and are last in the test period at all three frequencies. The
    EBM has the best validation IC of the base models at every frequency
    (0.022, 0.023, 0.022) and 0.002, -0.005 and 0.005 in the test period.

13. **The tone of the news did not add a forecasting signal** (section 10.2).
    FinBERT tone of 769,338 headlines (a model that saw no text after 2010) and
    Loughran-McDonald tone, over the last 1 to 252 days: the tone-only
    experiments have a test IC of 0.002, 0.000 and -0.002. The test fixed in
    advance, the price ensemble combined with the tone signal that validation
    preferred and traded with the buffer, has a positive IC at all three
    frequencies (0.011, 0.016, 0.019; none significant), about that of the
    price ensemble alone (weekly much less: 0.030), and a net Sharpe ratio of
    -0.01, -0.20 and -0.62. The
    embedding regression of Chen, Kelly and Xiu (2022) on FinBERT headline
    vectors has a positive IC biweekly and weekly (0.007, 0.005; not
    significant) that its turnover of 155% to 169% a period more than eats.

14. **Tuning the attention-shock window is the one improvement of the second
    part, and it is not significant** (sections 10.1, 10.3). Validation chose
    1-day shocks for weekly and 63-day shocks for monthly rebalancing. Over 132
    and 135 paired experiments the test IC rises from 0.006 to 0.011 (weekly)
    and from 0.003 to 0.014 (monthly), t = 1.1 and 1.2, and the net Sharpe
    ratio rises in 75% and 65% of the pairs. Two validation choices reach a net
    Sharpe ratio of 0.6 to 0.7 (section 10.4), with 95% intervals that include
    zero (-0.73 to 2.03 and -0.54 to 1.44, buffered):
    - weekly, price with every news block, EBM: 0.61, buffered 0.65; test IC
      0.025 (t = 2.3), the first validation choice whose IC clears its
      critical value. The same EBM on prices alone has a higher IC (0.030),
      and the portfolio earned everything in 2022-23;
    - monthly, 20 NoLBERT topics with 63-day shocks, IPCA: 0.68, buffered 0.57,
      positive in both halves of the test period; IC t = 0.9; chosen within a
      pool that was defined after the narrative-only test results were seen.

15. **A cap-weighted tilt shows the market, not the news** (section 10.5).
    Tilted towards the forecasts, the value-weighted universe has a net Sharpe
    ratio of 0.46 to 0.60, against 0.47 to 0.61 for the universe itself, and
    a median information ratio of -0.16 to -0.29.

**What to present.** The two pre-specified headlines (item 1 and the ensemble
protocol of item 3) with their intervals; the buffer as the one improvement
that was adopted on the validation period (where its gain was significant at
the weekly frequency) and then held up in the test period (item 2); NoLBERT as
the cleanest text layer with no forecasting gain we can demonstrate (item 4);
from the second part, the failed test of news tone (item 13), the tuned shock
window as the one consistent gain, with the two validation choices at 0.6 to
0.7 and their reservations (item 14), and the tilt as the honest answer to
"start from the index" (item 15); and the search statistics (item 6), which
say what any single good-looking number in our tables is worth. Best-of results, such as a net Sharpe ratio of
2.21 for one monthly value-weighted portfolio, are not distinguishable from
what 375 tries produce by chance (1.62 expected, p = 0.09) and should only be
shown next to that benchmark.

**What this means for the earlier project.** We did not re-run the earlier
pipeline. From its code, its out-of-sample numbers came from a universe chosen
with information from later in the year (a stronger form of the first defect
of item 10), topic models fitted on the whole sample, and tuning on the
evaluation months. They should not be quoted until they are reproduced under
a protocol like this one.

**How untouched is the test period?** Less than we would like. Sections 1 to 9
are the results of the fourth pass over it, section 10 of a fifth.

- After the first run, two reviews found that hiding missing returns from the
  model does not undo the selection of the stocks that have them, and that 7%
  of the article texts were wrong. We dropped the incomplete months, repaired
  the text, refitted the topic models, replaced the elastic net (identical to
  the lasso) by the sparse IPCA of the earlier project, and lowered the tuning
  tolerance from 1e-4 to 1e-5 because the larger value almost always returned
  the first setting.
- After the second run, three reviews found that the EBM that forecast the
  test period was not the model that validation had chosen, that tied
  forecasts were ordered by a shared random stream, and that t statistics were
  judged by critical values too lenient for samples this short. We now use the
  validated EBM itself, break ties by one fixed order, count a flat forecast
  as an IC of zero, and use small-sample critical values and a bootstrap for
  the search. That made the headline worse (monthly net Sharpe ratio 0.08 to
  -0.15, weekly -0.09 to -0.66).
- In this run NoLBERT and the variants of section 9 were added. Which of them
  to adopt was decided on the validation period and written down before the
  test period was looked at (section 9.1). The buffer was adopted and held up;
  the protocol built on the ensembles was adopted and did not.
- In the second part (section 10) the shock windows and the protocol for the
  news blocks were fixed on the validation period first (section 10.1). After
  the narrative-only test results of the windows had been seen, the windows
  were carried into the text and price-and-text sets and two selection pools
  were added; section 10 says which numbers that touches.

The conclusions of all five passes are the same. The outputs of the earlier
passes are not part of the repository.

**Not repeated after the last fixes.** The audits of `scripts/10` to `12`
(`results/data_audit/`) were run before the second run's fixes. Their rank ICs
for models other than the EBM are unaffected. Their Sharpe ratios used the
earlier tie rule and are quoted here only roughly. The EBM rows of the seed
experiment used the earlier EBM protocol and are left out of this document.

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
  adapted NoLBERT encoder (section 8).

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
follow (sections 4.3 and 9).

## 4. Do the features forecast returns?

### 4.1 The landscape

Average test-period rank IC of the base experiments of each kind (over models
and the six topic models). In brackets the t statistic of the group's average
IC per period, which allows for the experiments of a group being forecasts of
the same returns. Critical values: 2.39, 2.25, 2.11.

| Features | Monthly | Biweekly | Weekly | Weekly: experiments with positive IC |
|---|---|---|---|---|
| Price only | 0.008 (0.4) | 0.011 (0.7) | 0.024 (1.9) | 8 of 8 |
| Narrative exposures | 0.002 (0.2) | -0.001 (-0.2) | 0.003 (0.6) | 28 of 44 |
| Firm-level news | -0.002 (-0.4) | 0.000 (-0.1) | 0.001 (0.7) | 30 of 47 |
| Both text blocks | -0.001 (-0.1) | 0.000 (0.0) | 0.002 (0.5) | 31 of 44 |
| Price and text | 0.008 (0.6) | 0.002 (0.3) | 0.013 (1.6) | 43 of 44 |
| Placebo exposures | 0.012 (0.8) | 0.008 (0.7) | 0.008 (1.2) | 6 of 7 |

No group is distinguishable from zero. The text-only groups are below the
placebo at every frequency. The standard error of one experiment's mean IC is
about 0.018 monthly, 0.012 biweekly and 0.010 weekly (medians).

Of the 581 base experiments with a forecast, 15 have a test t statistic beyond
the critical value, 11 above it and 4 below; chance alone would give about 29.

### 4.2 Validation does not carry over

Across the base experiments on real features the rank correlation between
validation IC and test IC is -0.22 (monthly), -0.12 (biweekly) and 0.14
(weekly). The ten experiments with the highest validation IC have an average
test IC of -0.001, -0.010 and 0.004.

In the validation period 17 base experiments have a t statistic above the
critical value. Twelve of them are EBMs or XGBoost models, the two models that
stop boosting where the error on the validation rows is lowest. The 14
validation months (27 fortnights, 53 weeks) do three jobs: early stopping, the
choice of hyperparameters and the choice between experiments. A high
validation IC is weak evidence, and the test period says so.

### 4.3 Topic model against topic model

Average IC of the text-only base experiments of each topic model (exposures,
firm news, both; eight models each), with the t statistic of the group
average:

| Topic model | Validation: monthly | biweekly | weekly | Test: monthly | biweekly | weekly |
|---|---|---|---|---|---|---|
| LDA | -0.010 (-1.0) | 0.007 (1.0) | -0.010 (-2.9) | 0.005 (0.7) | 0.002 (0.4) | 0.003 (0.8) |
| LDA, every article | -0.008 (-0.8) | 0.010 (1.0) | 0.001 (0.2) | -0.003 (-0.3) | 0.000 (0.0) | 0.002 (0.4) |
| MiniLM clusters | -0.001 (-0.1) | 0.010 (1.4) | 0.005 (0.7) | 0.004 (0.7) | -0.004 (-0.6) | -0.001 (-0.1) |
| FinBERT classes | 0.022 (1.8) | 0.016 (2.2) | 0.012 (1.4) | -0.009 (-0.6) | -0.005 (-1.0) | -0.001 (-0.2) |
| NoLBERT clusters | 0.012 (2.6) | 0.010 (1.5) | 0.009 (2.1) | -0.003 (-0.4) | 0.003 (0.5) | 0.006 (1.4) |
| NoLBERT, 20 clusters | 0.000 (0.0) | -0.007 (-0.8) | 0.007 (1.0) | 0.006 (0.5) | 0.001 (0.1) | 0.004 (0.7) |

Topic model against topic model, period by period (`contrasts.csv`):

| | Monthly | Biweekly | Weekly |
|---|---|---|---|
| FinBERT minus LDA, validation (t) | +0.032 (3.2) | +0.009 (1.2) | +0.021 (2.2) |
| FinBERT minus LDA, test (t) | -0.014 (-0.9) | -0.008 (-1.7) | -0.004 (-0.7) |
| NoLBERT minus LDA, validation (t) | +0.022 (1.8) | +0.003 (0.3) | +0.019 (3.1) |
| NoLBERT minus LDA, test (t) | -0.008 (-0.8) | +0.001 (0.1) | +0.003 (0.7) |

The experiment that validation selects for each topic model, among all 32 base
experiments that use it, and its test result:

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
| NoLBERT clusters | Monthly | Price and text, sparse IPCA | 0.068 (1.8) | 0.006 (0.1) | -0.05 |
| NoLBERT clusters | Biweekly | Price and text, EBM | 0.029 (1.4) | 0.008 (0.5) | 0.05 |
| NoLBERT clusters | Weekly | Text, EBM | 0.028 (1.7) | 0.005 (0.5) | -0.20 |
| NoLBERT, 20 clusters | Monthly | Price and text, sparse IPCA | 0.045 (0.9) | 0.007 (0.2) | -0.29 |
| NoLBERT, 20 clusters | Biweekly | Text, EBM | 0.039 (1.7) | 0.005 (0.4) | -0.67 |
| NoLBERT, 20 clusters | Weekly | Price and text, EBM | 0.035 (2.0) | 0.025 (1.6) | 0.10 |

- **There is no reliable order.** FinBERT features look best in validation and
  are last in the test period at all three frequencies; LDA is at or next to
  the bottom in validation and near the top in the test period. NoLBERT is
  the exception to the reversal: fourth (biweekly) and second (weekly) in
  validation and first in the test period at both, never significantly.
- **More articles or fewer topics change nothing for the better.** Every
  article instead of 150 a day, or 20 topics instead of 60, makes the
  attention shocks more reliable, and the test ICs are not significantly
  higher (20 topics are higher only monthly: 0.006 against -0.003, t = 1.3).
- **A finance-specific classifier does not help.** The FinBERT classifier is
  the only component trained on labelled finance text, and it was published
  inside the test period. Its classes are the worst of the six in the test
  period.
- **Why not BERT over LDA, as in the earlier paper?** The differences between
  topic models here are noise around zero, in both directions. The earlier
  ranking came from in-sample topic models and a coherence comparison on
  unequal terms (section 3).

### 4.4 Text against price

For each model and topic model, the base experiment with price and text
features is compared with the same model on price features alone, period by
period (48 pairs per frequency, 44 or 45 of them with a forecast from both).

| | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Text raises the IC | 24 of 45 | 12 of 45 | 2 of 44 |
| significantly (t above the critical value) | 0 | 0 | 0 |
| Text lowers the IC significantly | 1 | 1 | 0 |
| Average change in IC | -0.001 | -0.009 | -0.011 |
| Lower squared error (Diebold-Mariano t > 2) | 0 | 0 | 0 |
| Higher squared error (t < -2) | 3 | 1 | 2 |
| Higher net Sharpe ratio | 27 of 45 | 17 of 45 | 18 of 44 |
| significantly (bootstrap p < 0.05) | 1 | 2 | 0 |
| significantly lower | 1 | 0 | 0 |

Where prices carry something (weekly), 41 to 121 additional text features
dilute it in 42 of 44 cases. Four of the 134 differences in Sharpe ratio are
"significant" at 5% (three gains, one loss), fewer than the six or seven that
chance gives.

### 4.5 Placebo

The placebo block is the narrative exposure to the attention shocks of 126
trading days earlier: the same construction with news that cannot be related
to the day's returns. The comparison is paired: the IC of the exposures to LDA
attention minus the IC of the placebo, per period, averaged over the seven
base models that produce a forecast from both.

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

| Price features only | Validation IC | Test IC | Test t | 2022-23 | 2024 | Long-short Sharpe, gross | net | Turnover per week | Break-even cost (bp) | Buffered: net | turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OLS | 0.020 | 0.027 | 1.6 | 0.025 | 0.031 | 0.91 | 0.05 | 122% | 10.6 | 0.09 | 89% |
| Ridge | 0.022 | 0.027 | 1.7 | 0.025 | 0.031 | 0.96 | 0.03 | 128% | 10.3 | 0.16 | 95% |
| Lasso | 0.002 | 0.021 | 1.4 | 0.020 | 0.023 | 1.05 | 0.17 | 143% | 11.9 | 0.06 | 110% |
| IPCA | 0.026 | 0.031 | 1.9 | 0.033 | 0.027 | 1.07 | 0.13 | 131% | 11.4 | 0.23 | 99% |
| Sparse IPCA | 0.004 | 0.014 | 0.9 | 0.013 | 0.015 | 0.03 | -0.20 | 49% | 1.2 | 0.12 | 26% |
| XGBoost | 0.023 | 0.024 | 1.8 | 0.027 | 0.017 | 0.50 | -0.27 | 119% | 6.5 | -0.12 | 99% |
| Random forest | 0.023 | 0.015 | 1.3 | 0.015 | 0.015 | 0.44 | -0.67 | 123% | 4.0 | -0.45 | 99% |
| EBM | 0.025 | 0.030 | 2.1 | 0.032 | 0.025 | 0.76 | -0.15 | 128% | 8.4 | 0.01 | 96% |
| Ensemble of the eight | 0.024 | 0.030 | 1.9 | 0.030 | 0.029 | 1.13 | 0.17 | 127% | 11.8 | 0.21 | 97% |
| Average of the eight | 0.018 | 0.024 | 1.9 | 0.024 | 0.023 | 0.93 | -0.14 | 118% | | | |

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
- It does not pay much. Costs would have to stay below 1 to 12 basis points
  per trade for the eight portfolios to break even, with a turnover above
  110% a week for all but one. The ensemble of the eight and the buffer
  together reach a net Sharpe ratio of 0.21; this combination was not chosen
  in advance and is one of many.
- Portfolios are formed at the closing prices that end the reversal signal,
  so bid-ask bounce is included.
- At the monthly and biweekly frequencies the price features show nothing
  (0.008 and 0.011, t below 1).

### 4.7 The two halves of the test period

We decided before the first run to report 2022-23 and 2024 separately, because
the 2024 news comes from a second scrape (section 2.1). Average IC of the base
experiments by kind of feature set, with the t statistic of the group average:

| Features | 2022-23: monthly | biweekly | weekly | 2024: monthly | biweekly | weekly |
|---|---|---|---|---|---|---|
| Price only | -0.014 (-0.6) | 0.005 (0.2) | 0.024 (1.4) | 0.050 (1.5) | 0.023 (0.8) | 0.023 (1.3) |
| Narrative exposures | 0.016 (2.3) | 0.009 (1.2) | 0.010 (1.9) | -0.027 (-1.6) | -0.021 (-2.1) | -0.011 (-1.2) |
| Firm-level news | -0.002 (-0.4) | -0.001 (-0.2) | 0.002 (0.9) | -0.001 (-0.2) | 0.001 (0.3) | 0.000 (-0.2) |
| Both text blocks | 0.015 (1.8) | 0.009 (1.3) | 0.009 (1.9) | -0.032 (-1.9) | -0.018 (-1.8) | -0.011 (-1.4) |
| Price and text | 0.008 (0.4) | 0.004 (0.3) | 0.016 (1.5) | 0.008 (0.9) | -0.001 (-0.1) | 0.005 (0.7) |
| Placebo exposures | 0.017 (0.8) | 0.009 (0.5) | 0.006 (0.7) | 0.002 (0.1) | 0.007 (0.5) | 0.012 (1.6) |

Periods: 14, 28 and 54 in 2022-23; 7, 14 and 27 in 2024. Critical values:
2.61, 2.39 and 2.18 in 2022-23; 3.27, 2.61 and 2.29 in 2024.

Narrative exposures have a positive IC in 2022-23 and a negative one in 2024
at all three frequencies. The test of the difference between the halves
(`halves.csv`) gives z = 2.4, 2.4 and 2.0 for the exposures and 2.5, 2.3 and
2.2 for both text blocks, against critical values of 3.27, 2.61 and 2.29 for
halves this short. By topic model the 2022-23 IC is the higher one in 16 of 18
cases and the difference is beyond the critical value in three (LDA biweekly,
z = 3.5; NoLBERT monthly, 3.5; MiniLM weekly, 2.7). The three experiments of
section 1 do not all follow the pattern: an IC of -0.014, 0.025 and 0.022 in
2022-23 and of 0.047, 0.000 and -0.016 in 2024, with net Sharpe ratios of
-0.25, 0.68 and -0.11 against 0.90, -0.49 and -2.23.

So there is a break in the exposures, with the same sign at all three
frequencies and in most topic models, and it is not individually significant
for the groups. Two readings fit it:

- *The second scrape.* In 2024 the headlines are rebuilt from addresses,
  coverage halves, tagging changes and about seven weeks of news are missing
  (section 2.1). Topic models fitted on earlier text may read this text
  differently.
- *Ordinary variation.* An exposure is a loading on past co-movement.
  Forecasts built on loadings do well or badly for months at a time, as factor
  returns do. In the validation period the same exposures were not
  distinguishable from zero, and against the placebo they are not
  significantly better in 2022-23 either (section 4.5).

Seven months cannot separate the two.

## 5. Portfolios

Test period, over the base experiments of each frequency that take a position
(placebo included). A flat forecast (43 lasso experiments in all) holds nothing
under the long-short rules and the universe under the long-only rules, and is
left out.

| Rule | Median net Sharpe: monthly | biweekly | weekly | Positive net Sharpe: monthly | biweekly | weekly | Weekly: turnover per period | cost per year | volatility |
|---|---|---|---|---|---|---|---|---|---|
| Decile long-short, equal weights | -0.21 | -0.36 | -0.48 | 68 of 192 | 59 of 195 | 44 of 194 | 79% | 8.2% | 12.1% |
| The same, buffered | -0.12 | -0.24 | -0.27 | 79 of 192 | 68 of 195 | 56 of 194 | 49% | 5.1% | 11.3% |
| Decile long-short, value weights | -0.26 | -0.46 | -0.56 | 75 of 192 | 54 of 195 | 52 of 194 | 93% | 9.6% | 17.8% |
| Rank-weighted long-short | -0.23 | -0.44 | -0.52 | 54 of 192 | 53 of 195 | 42 of 194 | 53% | 5.5% | 7.6% |
| Mean-variance long-short | -0.61 | -0.95 | -1.63 | 43 of 192 | 25 of 195 | 4 of 194 | 63% | 6.6% | 3.4% |
| Long-only top decile, equal weights | 0.38 | 0.43 | 0.30 | 192 of 192 | 195 of 195 | 191 of 194 | 41% | 4.3% | 22.3% |
| Long-only top decile, value weights | 0.48 | 0.43 | 0.32 | 183 of 192 | 173 of 195 | 154 of 194 | 47% | 4.9% | 21.0% |

The universe itself, for reference (Sharpe ratio and maximum drawdown):

| | Validation: equal-weighted | value-weighted | Test: equal-weighted | value-weighted | Test drawdown: equal-weighted | value-weighted |
|---|---|---|---|---|---|---|
| Monthly | 1.49 | 1.70 | 0.47 | 0.52 | -14% | -13% |
| Biweekly | 2.03 | 2.36 | 0.61 | 0.61 | -15% | -15% |
| Weekly | 1.94 | 2.14 | 0.53 | 0.47 | -19% | -18% |

- **The median long-short portfolio loses money after costs** at every
  frequency and under every rule: a forecast without information still trades.
  Before costs the median equal-weighted decile portfolio has a Sharpe ratio
  of 0.03, 0.09 and 0.28; after costs it returns -2.8%, -4.6% and -5.1% a
  year, with a median maximum drawdown of -14%, -16% and -17%. None of the
  equal-weighted decile portfolios has a three-factor alpha with a t statistic
  above the critical value except one biweekly; 19, 26 and 52 have one below
  its negative.
- **The buffer is the best long-short rule after costs** at every frequency
  (section 9).
- **Mean-variance weighting does worst.** Its costs are lower than the decile
  rule's but its volatility is far lower (3.4% a year at the weekly
  frequency), so costs weigh most against its risk.
- **The long-only rules earn about the market's return.** Their
  Sharpe ratios are positive because the universe's is. Against the universe
  with the same weighting, the median information ratio of the equal-weighted
  top decile is -0.22, -0.45 and -0.57, and that of the value-weighted top
  decile 0.07, -0.17 and -0.22.

**Counting the search.** Net Sharpe ratios of the 375 experiments of each
frequency, variants included (the placebo is a control and not part of the
search). For the long-only rules the return is measured against the universe,
so the Sharpe ratio is an information ratio. "Expected best with no skill" is
the average best Sharpe ratio over 5,000 bootstrap resamples in which every
portfolio's mean return is set to zero and their co-movement is kept; "p of
the best" is the share of resamples with a best at least as high (White's
reality check).

| Rule | Rebalancing | Best | which is | Median | Expected best with no skill | p of the best | SPA p | Significant at 5%, unadjusted | At a 10% false discovery rate |
|---|---|---|---|---|---|---|---|---|---|
| Decile long-short, equal weights | Monthly | 1.51 | NoLBERT-20 firm news, XGBoost on ranks | -0.25 | 1.55 | 0.49 | 0.22 | 4 | 0 |
| | Biweekly | 1.63 | LDA text, XGBoost | -0.50 | 1.70 | 0.52 | 0.56 | 2 | 0 |
| | Weekly | 0.97 | Price and NoLBERT text, XGBoost | -0.37 | 1.85 | 0.99 | 0.91 | 0 | 0 |
| Buffered decile long-short | Monthly | 1.46 | LDA exposures, random forest | -0.15 | 1.54 | 0.53 | 0.15 | 3 | 0 |
| | Biweekly | 1.52 | LDA text, XGBoost | -0.34 | 1.67 | 0.61 | 0.66 | 1 | 0 |
| | Weekly | 1.11 | Price and NoLBERT text, XGBoost | -0.23 | 1.79 | 0.97 | 0.88 | 0 | 0 |
| Decile long-short, value weights | Monthly | 2.21 | Price and NoLBERT-20 text, OLS on ranks | -0.25 | 1.62 | 0.09 | 0.08 | 8 | 0 |
| | Biweekly | 1.50 | LDA exposures, random forest | -0.47 | 1.83 | 0.77 | 0.40 | 7 | 0 |
| | Weekly | 1.45 | LDA firm news, random forest | -0.58 | 1.92 | 0.87 | 0.87 | 0 | 0 |
| Rank-weighted long-short | Monthly | 0.98 | NoLBERT-20 exposures, lasso on ranks | -0.27 | 1.49 | 0.91 | 0.74 | 0 | 0 |
| | Biweekly | 1.22 | LDA text, XGBoost | -0.53 | 1.66 | 0.86 | 0.82 | 0 | 0 |
| | Weekly | 0.70 | Price and NoLBERT text, XGBoost | -0.35 | 1.74 | 1.00 | 0.99 | 0 | 0 |
| Mean-variance long-short | Monthly | 1.19 | LDA exposures (every article), random forest | -0.60 | 1.64 | 0.87 | 0.82 | 0 | 0 |
| | Biweekly | 1.13 | NoLBERT firm news, XGBoost | -0.89 | 1.81 | 0.97 | 0.73 | 0 | 0 |
| | Weekly | 0.66 | LDA exposures (every article), sparse IPCA on ranks | -1.38 | 1.95 | 1.00 | 1.00 | 0 | 0 |
| Long-only, equal weights | Monthly | 1.73 | LDA text (every article), XGBoost on ranks | -0.28 | 1.61 | 0.35 | 0.41 | 1 | 0 |
| | Biweekly | 1.09 | LDA firm news (every article), lasso | -0.54 | 1.76 | 0.97 | 0.96 | 0 | 0 |
| | Weekly | 1.27 | NoLBERT-20 exposures, XGBoost | -0.56 | 1.90 | 0.96 | 0.97 | 0 | 0 |
| Long-only, value weights | Monthly | 2.08 | Price and LDA text, ridge | -0.02 | 1.65 | 0.15 | 0.06 | 9 | 0 |
| | Biweekly | 1.54 | NoLBERT-20 text, IPCA | -0.32 | 1.85 | 0.75 | 0.58 | 7 | 0 |
| | Weekly | 2.05 | LDA firm news, random forest | -0.32 | 1.99 | 0.40 | 0.60 | 4 | 0 |

The best portfolio is below what the best of 375 unskilled portfolios would
show in 17 of the 21 rows and nowhere significantly above it. The lowest
p-values are 0.06 (SPA) and 0.09 (reality check), for two monthly
value-weighted rules; no portfolio passes the false-discovery threshold.

**The experiments of section 1 under every rule**, test period, net Sharpe
ratio (for the long-only rules that of the portfolio itself, not against the
universe):

| | Long-short EW | Buffered | Long-short VW | Rank | Mean-variance | Long-only EW | Long-only VW | Max drawdown, long-short EW |
|---|---|---|---|---|---|---|---|---|
| Monthly: price and NoLBERT text, sparse IPCA | -0.05 | 0.05 | -0.26 | -0.13 | 0.10 | 0.37 | 0.16 | -31% |
| Biweekly: price and FinBERT text, XGBoost | 0.37 | 0.41 | -0.08 | 0.36 | -0.30 | 0.49 | 0.58 | -17% |
| Weekly: price and LDA text, EBM | -0.66 | -0.33 | -0.72 | -0.59 | -1.07 | 0.32 | 0.34 | -24% |

## 6. Monthly, biweekly or weekly?

| | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Periods: training / validation / test | 69 / 14 / 21 | 143 / 27 / 42 | 290 / 53 / 81 |
| Training rows | 67,791 | 140,333 | 284,639 |
| Years of test data | 1.75 | 1.62 | 1.56 |
| Standard error of a mean IC, test (median) | 0.018 | 0.012 | 0.010 |
| Standard error of an annual Sharpe ratio (median) | 0.75 | 0.79 | 0.80 |
| Median turnover per period, decile long-short | 119% | 89% | 79% |
| buffered | 93% | 60% | 49% |
| Median cost per year, decile long-short | 2.9% | 4.6% | 8.2% |
| buffered | 2.2% | 3.1% | 5.1% |
| Median net Sharpe, decile long-short | -0.21 | -0.36 | -0.48 |
| buffered | -0.12 | -0.24 | -0.27 |
| Median gross Sharpe, decile long-short | 0.03 | 0.09 | 0.28 |
| Fitting time, all 364 fitted experiments (4 cores) | 15 min | 21 min | 33 min |

- **More periods do not buy certainty about the Sharpe ratio.** Its standard
  error depends on the number of years, 0.75 to 0.80 here at any frequency.
  The standard error of a mean IC falls with the number of periods, but a
  weekly IC and a monthly IC are different quantities, and what a strategy
  earns in a year is what matters.
- **More rows do not help a model that has nothing to find.** With four times
  the rows the validation period still chooses mostly simple settings (section
  7).
- **Costs rise with the number of rebalances**, from 2.9% to 8.2% a year for
  the median decile portfolio. They take about 0.2 off its Sharpe ratio
  monthly and about 0.8 weekly, which is why the buffer helps most at the
  higher frequencies.
- **Weekly is where the only stable pattern is**, and it is a price pattern
  that does not survive costs.
- A caution for the higher frequencies: in holding periods that reach into
  January, the stocks that will leave the list in June have only a monthly
  return in our file, which is spread evenly over the days of the month. That
  concerns one period a year.

For this project the answer is that the frequency does not matter, because
there is no signal for it to sharpen. The weekly panel is the place to look if
a point-in-time news source becomes available: it has the most periods, and
costs can be dealt with by the buffer.

## 7. Models

Average IC of each model over the 26 feature sets (the placebo included), with
the t statistic of the average IC per period. The eight base models first,
then the variants of section 9:

| Model | Validation: monthly | biweekly | weekly | Test: monthly | biweekly | weekly | Feature sets with positive test IC: M / 2W / W | Median net Sharpe, test: M / 2W / W |
|---|---|---|---|---|---|---|---|---|
| OLS | -0.012 (-1.6) | 0.000 (-0.1) | -0.002 (-0.4) | -0.003 (-0.5) | 0.000 (-0.1) | 0.005 (1.3) | 10 / 10 / 19 | -0.23 / -0.52 / -0.50 |
| Ridge | -0.011 (-1.2) | 0.007 (0.7) | 0.003 (0.4) | 0.003 (0.4) | -0.001 (-0.2) | 0.008 (1.5) | 13 / 12 / 21 | 0.00 / -0.38 / -0.22 |
| Lasso | 0.007 (0.7) | 0.010 (1.2) | 0.007 (1.3) | -0.005 (-0.4) | -0.004 (-0.5) | 0.004 (0.9) | 3 / 3 / 9 | -0.11 / -0.90 / -0.51 |
| IPCA | 0.015 (1.2) | 0.001 (0.2) | 0.002 (0.4) | 0.002 (0.2) | 0.004 (0.8) | 0.005 (1.5) | 13 / 18 / 18 | -0.32 / -0.18 / -0.27 |
| Sparse IPCA | 0.016 (1.5) | 0.000 (0.0) | -0.002 (-0.3) | 0.000 (0.0) | 0.003 (0.4) | 0.001 (0.3) | 11 / 16 / 14 | -0.30 / -0.16 / -0.28 |
| XGBoost | 0.002 (0.3) | 0.012 (2.5) | 0.010 (1.8) | 0.004 (0.7) | 0.002 (0.7) | 0.008 (2.4) | 16 / 13 / 24 | -0.34 / -0.26 / -0.62 |
| Random forest | -0.005 (-0.5) | 0.006 (0.6) | 0.006 (0.9) | 0.013 (1.4) | 0.006 (1.2) | 0.008 (2.0) | 19 / 20 / 24 | 0.02 / -0.36 / -0.67 |
| EBM | 0.022 (2.5) | 0.023 (2.4) | 0.022 (2.2) | 0.002 (0.3) | -0.005 (-0.7) | 0.005 (0.9) | 16 / 11 / 17 | -0.35 / -0.77 / -0.90 |
| Ensemble of the eight | 0.003 (0.4) | 0.010 (1.0) | 0.007 (0.9) | 0.004 (0.6) | 0.001 (0.1) | 0.008 (1.7) | 16 / 17 / 20 | -0.21 / -0.30 / -0.29 |
| OLS on ranks | -0.014 (-1.9) | -0.002 (-0.3) | -0.002 (-0.4) | -0.004 (-0.6) | -0.001 (-0.1) | 0.008 (1.8) | 10 / 11 / 20 | -0.42 / -0.74 / -0.46 |
| Ridge on ranks | -0.006 (-0.7) | 0.006 (0.8) | 0.005 (0.7) | -0.002 (-0.2) | -0.005 (-0.6) | 0.007 (1.4) | 11 / 6 / 22 | -0.15 / -0.61 / -0.23 |
| Lasso on ranks | 0.011 (1.2) | 0.009 (2.1) | 0.011 (2.5) | 0.006 (0.5) | -0.005 (-0.6) | 0.008 (1.7) | 6 / 4 / 18 | -0.01 / -0.61 / -0.35 |
| IPCA on ranks | 0.019 (1.7) | 0.008 (1.4) | 0.006 (1.1) | -0.003 (-0.4) | -0.003 (-0.5) | 0.003 (0.6) | 12 / 13 / 16 | -0.31 / -0.73 / -0.23 |
| Sparse IPCA on ranks | 0.018 (1.5) | -0.001 (-0.1) | 0.005 (0.7) | -0.007 (-1.0) | -0.001 (-0.1) | 0.005 (0.7) | 9 / 15 / 16 | -0.39 / -0.41 / -0.35 |
| XGBoost on ranks | 0.014 (1.1) | 0.022 (2.1) | 0.016 (1.7) | 0.016 (1.3) | 0.007 (0.8) | 0.013 (2.2) | 23 / 20 / 25 | -0.01 / -0.31 / -0.24 |

(The lasso averages and counts are over the feature sets where its forecast is
not flat: 10, 13 and 12 of 26; on ranks 10, 18 and 19.)

- **No model stands out in the test period.** Two of the 45 cells pass their
  critical value, both weekly XGBoost (2.4, and 2.2 on ranks, against 2.11),
  about what chance gives, with an IC (0.008 and 0.013) below what the price
  features give alone at that frequency (0.024). Boosted trees, forest and EBM
  together against the five linear base models (`contrasts.csv`): +0.007
  (t = 1.4), +0.001 (0.2) and +0.002 (0.7).
- **The EBM is the clearest case of validation flattering a model.** It has
  the best validation average of the base models at every frequency; in the
  test period it is fourth, eighth and sixth of the eight by IC, and last or
  second to last by median net Sharpe ratio. In validation the three flexible models
  beat the linear ones significantly at the biweekly and weekly frequencies
  (+0.011, t = 3.4; +0.012, t = 2.8); in the test period the difference is
  gone. They stop boosting, or are tuned, on the rows they are then scored on.
- **The EBM forecasts the test period with the model that validation chose.**
  The library cannot refit its two stages with separate numbers of rounds, so
  the model fitted on the training period and stopped on the validation
  period is used as it is. The other models are refitted on training plus
  validation data with the chosen setting.
- **What validation chose** (78 experiments per base model). The linear models
  and the forest are mostly pushed to their most regularised settings: the lasso to
  the empty model in 48 cases (after the refit 43 are still empty), ridge to
  the largest of its 25 penalties in 51, sparse IPCA to one factor in 52 and
  to its strongest penalty in 56, IPCA to one factor in 35, the random forest
  to depth 3 in 60. XGBoost and the EBM go the other way about as often as
  not: XGBoost takes its deepest trees (depth 5) in 30 cases and depth 1 in
  22, and the EBM takes its ten pairwise terms in 47.
- **Sparse IPCA against IPCA.** The earlier project's estimator does not
  forecast better: 0.000, 0.003 and 0.001 against 0.002, 0.004 and 0.005. On
  the weekly price features it has half of IPCA's IC (0.014 against 0.031).
- Fitting all 1,092 fitted experiments takes 69 minutes on four laptop cores:
  25 for the EBMs, 13 for the forests, 11 for the two sparse IPCAs, 11 for the
  two XGBoosts, 7 for the two IPCAs and 2 for the six linear fits, which are
  solved from second moments. The 624 base experiments take 53 minutes.

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
over seeds would remove this noise at ten times the fitting time (section 12).

## 8. NoLBERT

NoLBERT (Kakhbod and Li 2025, `alikLab/NoLBERT`) is a DeBERTa-v3 language
model pretrained only on text published between 1976 and 1995. It cannot have
read anything about our sample, which no other text component here can claim:
MiniLM was published in 2021 and the FinBERT classifier in 2022, and LDA is
fitted on our own training text. Our teammate adapted it, embedded the
articles and fitted its topics (run of 8 October 2026); this section reports
that work and what it gives inside this pipeline.

### 8.1 Adaptation

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
  dated before 1 September 2020, the cutoff of the teammate's other pipeline,
  which is eight months into our validation period. That is text, not
  returns, and nothing from the test period; but it can flatter NoLBERT's
  validation numbers, and the monthly headline choice of section 1 rests on
  them. `scripts/03c_finetune_encoder.py` now cuts the pairs at the end of the
  training period by default.
- **It differs from MiniLM in two ways at once.** NoLBERT was adapted to our
  corpus; MiniLM was used as published. A gap between them can come from the
  adaptation as much as from the pretraining period. The clean comparison is
  MiniLM adapted with the same recipe, which needs a GPU.

### 8.2 What it gives

- **Topics** (section 3): the most stable and most even partition of the
  60-topic models, the least diverse top words, and attention shocks about as
  reliable as LDA's.
- **Selection.** Monthly, the validation period now chooses price and NoLBERT
  text with sparse IPCA (validation IC 0.068, t = 1.8); in the test period it
  has an IC of 0.006 (t = 0.1) and a net Sharpe ratio of -0.05. Within the
  NoLBERT feature sets validation picks the EBM biweekly (price and text) and
  weekly (text only), with test ICs of 0.008 and 0.005, neither significant.
- **Forecasts.** On the text-only feature sets NoLBERT has the highest test IC
  of the six topic models at the biweekly and weekly frequencies (0.003 and
  0.006; weekly t = 1.4, 21 of 22 positive) and is ahead of each of the other
  five in all ten paired comparisons at those frequencies, the largest t being
  1.6. In validation it beats LDA significantly at the weekly frequency (t =
  3.1 against 2.19). Monthly it is fifth of the six, ahead only of FinBERT,
  and in 2024 its monthly IC (-0.031) is the second lowest.
- **20 topics.** The same vectors in 20 clusters give far more reliable shocks
  and no better forecasts: lower in validation at every frequency, about the
  same in the test period (section 4.3).
- **Search.** Price and NoLBERT text with XGBoost is the best weekly decile
  long-short portfolio of the 375 (net Sharpe 0.97, buffered 1.11), against
  1.85 and 1.79 expected from the best of 375 without skill.

### 8.3 The same encoder in the teammate's other pipeline

The teammate also ran the adapted encoder through the separate replication of
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

### 8.4 Reading

The defensible claim is about the text layer: an encoder that cannot have seen
the future gives a stable, even and reproducible decomposition of the news at
no cost in forecasting power. The claim that it forecasts returns better is
not supported, here or in the other pipeline.

## 9. What we tried to improve the results

Four levers, each a standard remedy for one weakness found in the previous
runs: costs (the weekly price signal earns before costs and not after), noise
in single models (section 7.1), outliers in the target (returns are not
winsorised) and noise in the attention shocks (section 3). Each became a
variant run on every feature set and frequency:

| Lever | Variant | Compared with |
|---|---|---|
| Trade less | `ls_buf`: enter a leg in the top (bottom) decile, leave it below the top (bottom) fifth (Novy-Marx and Velikov 2016) | `ls_ew`, every base experiment |
| Combine models | `ens`: the average of the eight base models' forecasts, as ranks, with equal weights | the average of its eight members, per feature set |
| Robust target | `<model>_rk`: OLS, ridge, lasso, IPCA, sparse IPCA and XGBoost fitted on the cross-sectional rank of the return | the same model on the return |
| Less noisy attention | `nolbert20`: 20 NoLBERT topics instead of 60 | `nolbert` |

### 9.1 Decided on the validation period

Before any test number of this run was looked at, the validation rows of
`improvements.csv` were read and the decisions below written down:

| | Monthly | Biweekly | Weekly | Decision |
|---|---|---|---|---|
| Buffer: net return, a year (t) | +0.1% (0.3) | +0.8% (1.9) | +1.9% (3.4) | adopt, at every frequency |
| Buffer: turnover per period | 116% to 93% | 90% to 67% | 73% to 51% | |
| Ensemble: IC against its members (t) | +0.000 (0.1) | +0.003 (0.9) | +0.002 (0.6) | no measurable gain; use the 25 ensembles as the candidates of a second, smaller search |
| Rank target: IC (t) | +0.004 (0.7) | +0.003 (0.9) | +0.004 (1.5) | higher everywhere, never significant, Sharpe mixed: keep as candidates, not as the default |
| 20 against 60 NoLBERT topics, text-only IC | 0.000 against 0.012 | -0.007 against 0.010 | 0.007 against 0.009 | not adopted |

The improved protocol was therefore fixed as: choose among the 25 ensembles by
validation IC, trade the choice with the buffer.

### 9.2 The test period

Averages over all pairs, test period (`improvements.csv`, group "every"):

| | Monthly | Biweekly | Weekly |
|---|---|---|---|
| **Buffer**: net return, a year (t) | +0.6% (2.0) | +1.5% (3.5) | +2.0% (5.2) |
| median net Sharpe, without / with | -0.24 / -0.15 | -0.39 / -0.24 | -0.48 / -0.29 |
| experiments with a higher net Sharpe | 62% | 74% | 84% |
| turnover per period, without / with | 117% / 93% | 92% / 69% | 77% / 55% |
| **Ensemble**: IC against its members (t) | +0.001 (0.5) | 0.000 (-0.1) | +0.003 (1.4) |
| feature sets with a higher net Sharpe than the average member | 60% | 72% | 72% |
| **Rank target**: IC (t) | 0.000 (0.0) | -0.002 (-0.5) | +0.002 (0.7) |
| pairs with a higher net Sharpe | 46% | 43% | 39% |
| **Improved protocol** (text_finbert ensemble, buffered): test IC (t) | -0.015 (-0.6) | -0.014 (-1.1) | -0.004 (-0.4) |
| its net Sharpe ratio (95% interval) | -0.21 (-1.91 to 0.85) | -0.34 (-1.61 to 0.69) | -0.16 (-1.25 to 1.19) |

- **The buffer held up.** The gain is significant at the biweekly and weekly
  frequencies and larger where costs are higher. It is clearest for the text
  feature sets (weekly: +2.0% a year, t = 5.2) and less regular for the eight
  price experiments (+1.5%, 0.0% and +2.0% a year). It is a saving on costs:
  the median buffered portfolio still loses money, because the forecasts it
  trades on carry no information. (These averages count the flat lasso
  experiments, which hold nothing under both rules, as zero.)
- **The ensemble helps a little and is the safer default.** Its IC is that of
  an average model, without the risk of having picked the worst one, and its
  Sharpe ratio is higher than the average member's in 60% to 72% of the
  feature sets, although it trades more. On the weekly price features it has
  the highest net Sharpe ratio of the nine (0.17, just ahead of the lasso),
  the third IC (0.030, after IPCA and the EBM) and, buffered, the second
  Sharpe ratio (0.21, after IPCA's 0.23).
- **The rank target did not carry over** from validation to test.
- **The improved protocol lost**, because validation again chose the FinBERT
  text features, which do well in 2020 to 2021 and badly afterwards (section
  4.3). Reducing the search from 200 candidates to 25 does not help when the
  validation period itself points the wrong way.

### 9.3 What stays in the pipeline

All four variants stay as options: `ls_buf` is one of the eight portfolio
rules, `scripts/06b_ensemble.py` builds the ensembles, the rank-target models
are fitted by default, and `nolbert20` is one of the six topic models. The
groups of sections 4 to 6 and the original headline keep the eight base models
and the decile rule, so they remain comparable with the earlier runs.

## 10. Directional news, attention-shock windows and a cap-weighted tilt

The topic features say what a firm is in the news about, not whether the news
is good or bad, and the 5-day window of the attention shock was set without
trying another. A second part of the run (8 October) added, under the same
protocol:

| | What | Experiments per frequency |
|---|---|---|
| Tone of each article | FinBERT (`ProsusAI/finbert`: trained on news of 2008 to 2010 and the Financial PhraseBank, so point-in-time for our sample) on the headlines of the 769,338 articles filed under at most three tickers, one of them in our universe: 21% positive, 14% negative, 64% neutral. Loughran-McDonald word counts on headline and lead (correlation with FinBERT's tone 0.23) | |
| Tone features | `tone_fb`, `tone_lm`: tone of the firm's articles over the last 1, 5, 21 and 252 trading days (news up to the day before formation), the 21-day surprise against the 252-day tone, the amount of news and its abnormal level | 8 base models on `tone_fb`, `tone_lm` |
| Embedding regression | `emb_fb`: the firm's mean FinBERT headline vector (64 principal components fitted on training-period articles, 78% of the variance) since the last rebalancing, with our models in the role of the linear forecaster of Chen, Kelly and Xiu (2022) | `emb_fb`, `news_fb` (all three blocks) |
| News and past returns | `npx`: the 5- and 21-day returns of stocks with and without news in that window (Chan 2003: moves on news drift, moves without news reverse) and the returns times the tone | `px_tone`, `px_news`, `px_all` (price with the news blocks; `px_all` adds the NoLBERT text blocks) |
| Signals | One feature as the forecast, with the sign the literature predicts: FinBERT and LM tone over 1, 5 and 21 days and their surprise, abnormal news, reversal without news, drift with news | 11 |
| Combinations | `combo`: the price ensemble averaged with the tone signal that validation prefers. `best10`: the average of the ten base experiments (any feature set) with the highest validation IC | 2 |
| Shock windows | Windows of 1, 5, 21 and 63 days compared on validation (section 10.1); the chosen windows, 1 day weekly and 63 days monthly, in the narrative, text and price-and-text sets of all six topic models | 144 weekly and monthly |
| Tilt | `tilt_vw`: the value-weighted universe with each weight times (1 + s), s the forecast rank scaled to [-1, 1]; judged against the value-weighted universe | a portfolio rule, for every experiment |

With ensembles that makes 552 new experiments, 1,722 in all (628 monthly, 466
biweekly, 628 weekly). Firm-level news covers part of the universe: the share
of stocks with an article in the last five trading days rises from 19% in
2013 to 49% in 2022-23 and falls to 36% in 2024. The tables of sections 1 to 9
are computed over the experiments of the main run only, and are unchanged.

### 10.1 Decided on the validation period

**Shock window.** For each frequency, topic model and window, the narrative
exposures were rebuilt and OLS, ridge and lasso fitted and tuned as usual
(`scripts/05b_tune_shocks.py`; window 5 reproduces the validation forecasts of
the main run). Rule: the window with the highest validation IC averaged over
the 18 fits, if it also beats 5 days for at least four of the six topic models.
The rule was written down after this table had been seen; it differs from
"highest average" only at the biweekly frequency, where 1 and 5 days are
0.0002 apart.

| Validation rank IC (18 fits) | 1 day | 5 days | 21 days | 63 days | Topic models better than 5 days (1 / 21 / 63) | Window |
|---|---|---|---|---|---|---|
| Monthly | -0.012 | -0.010 | -0.018 | -0.002 | 3 / 2 / 4 | 63 |
| Biweekly | 0.007 | 0.006 | 0.000 | 0.003 | 3 / 2 / 3 | 5 (unchanged) |
| Weekly | 0.007 | -0.001 | -0.002 | 0.002 | 6 / 4 / 4 | 1 |

**News.** Fixed before any validation number: the primary test of this part is
`combo` traded with the buffer, one per frequency. Read on validation before
any test number of the news experiments:

| Validation rank IC | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Price and tone (`px_tone`) minus price, 8 models (t) | +0.001 (0.3) | +0.010 (1.2) | -0.003 (-1.6) |
| `px_all` minus price (t) | +0.009 (0.7) | +0.015 (1.1) | -0.008 (-0.6) |
| Best tone signal (t), the one `combo` uses | 21-day FinBERT tone +0.029 (2.9) | 1-day LM tone +0.009 (2.1) | 1-day LM tone +0.002 (0.4) |
| Best base experiment on the news sets ("news") | `px_all`, lasso, 0.045 | `px_news`, EBM, 0.036 | `px_all`, EBM, 0.027 |

Decisions: no tone block becomes a default (none adds to price by more than
its noise, and weekly they subtract); the signs of the signals stay those of
the literature although several have the opposite sign in validation;
`combo` and `best10` cannot be selected in any pool, since their validation
IC is partly that of a choice.

After the narrative-only test results of the tuned windows had been seen, the
windows were also used in the text and price-and-text sets, and two pools were
added to the selection table: "shock" (base models on the tuned-window sets)
and "any_all" (base models on every set). They are counted in every search
statistic below.

### 10.2 Tone does not forecast these returns

| Test period | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Tone blocks alone, average IC of 16 experiments (t) | 0.002 (0.3) | 0.000 (-0.1) | -0.002 (-1.2) |
| The eight single tone signals: test IC, lowest to highest | -0.002 to 0.006 | -0.006 to 0.003 | -0.006 to 0.008 |
| Embedding regression (`emb_fb`), average IC (t), positive of 8 | -0.009 (-2.5), 0 | 0.007 (1.6), 7 | 0.005 (1.9), 7 |
| its turnover per period, decile long-short | 155% | 167% | 169% |
| `px_tone` minus price, IC, average of 8 models (higher in) | +0.000 (5 of 8) | +0.004 (4) | -0.003 (0) |
| `px_all` minus price | +0.007 (7) | -0.004 (3) | -0.004 (2) |
| **Primary test: `combo`**, IC (t) | 0.011 (0.5) | 0.016 (1.0) | 0.019 (1.5) |
| its net Sharpe ratio, buffered (95% interval) | -0.01 (-1.21 to 1.72) | -0.20 (-1.36 to 0.93) | -0.62 (-2.26 to 0.69) |
| the price ensemble alone: IC, net Sharpe buffered | 0.010, -0.04 | 0.016, -0.18 | 0.030, 0.21 |
| `best10`: IC, net Sharpe buffered | 0.009, 0.29 | -0.002, -0.91 | 0.001, -0.32 |

- **A day after publication, the tone of a headline says nothing about the
  next week's or month's return of these stocks.** The monthly signal that
  validation liked (21-day FinBERT tone, 0.029) has 0.006 in the test period.
  This is what the literature leads one to expect for the largest stocks:
  news tone is priced within a day or two, and what predictability remains
  sits in small stocks (Tetlock, Saar-Tsechansky and Macskassy 2008; Ke, Kelly
  and Xiu 2019). Our articles carry dates without times, so we cannot trade
  on the day of the news.
- **The embedding regression** has a positive IC in 7 of 8 models at the
  biweekly and weekly frequencies (not significant) and a negative one in all
  8 monthly (t = -2.5, beyond the critical value in that direction). Its
  forecasts change with every article, and at 155% to 169% turnover per period
  the costs take more than the signal.
- **News added to prices does not raise the IC**, except in the monthly
  `px_all`, which contains the tuned narrative block (section 10.3).
- **The primary test fails.** `combo` has a positive IC at all three
  frequencies, about that of the price ensemble alone (monthly 0.011 against
  0.010, biweekly 0.016 against 0.016, weekly 0.019 against 0.030), and loses
  money after costs. `best10`, the validation choice spread over ten
  experiments, has a test IC of about zero.

### 10.3 The tuned shock windows help, without significance

`improvements.csv`, comparison "shock_window", group "every": each experiment
of the eight base models on a narrative, text or price-and-text set against
the same experiment with the 5-day window.

| Test period | Monthly (63 days) | Weekly (1 day) |
|---|---|---|
| Pairs | 135 | 132 |
| IC, 5 days and tuned (t of the difference) | 0.003 and 0.014 (1.2) | 0.006 and 0.011 (1.1) |
| Net long-short return, a year (t) | +2.6% (0.8) | +4.9% (1.4) |
| Median net Sharpe, 5 days and tuned | -0.07 and 0.13 | -0.24 and 0.11 |
| Pairs with a higher net Sharpe | 65% | 75% |
| IC difference in 2022-23 and in 2024 | +0.003 and +0.026 | +0.008 and -0.002 |
| Topic models with a higher IC (narrative, text and price-and-text sets) | 5 of 6 (not MiniLM) | 5 of 6 (not LDA) |
| Equal-weighted average of the 48 narrative-only portfolios: net Sharpe, 5 days and tuned | -0.02 and 0.69 | -0.63 and 0.31 |

This is the most consistent improvement of the second part: the gain shows in
validation (where the windows were chosen), in the test period, for five of
six topic models at each frequency, and in the Sharpe ratios of most pairs.
It is not significant (t statistics of 0.8 to 1.4 against critical values of
2.1 to 2.4),
and its timing differs: the monthly gain is in 2024, the weekly one in
2022-23.

The strongest test numbers of the whole run are here, and validation did not
point to them: on the monthly narrative exposures to the 20 NoLBERT topics with
63-day shocks, the eight models have an average test IC of 0.027 (7 positive,
t = 3.4 against 2.39) and their average portfolio a net Sharpe ratio of 1.66;
their ensemble, buffered, 1.76. Their validation ICs run from -0.05 to +0.065
(the ensemble's is -0.009). Found after the fact among some fifty monthly
feature sets, this is a lead for a later test, not a result.

### 10.4 What validation chooses now, and the Sharpe ratios of 0.6 to 0.7

| Pool (validation IC) | Monthly | Biweekly | Weekly |
|---|---|---|---|
| "any", the main run's headline | unchanged (section 1) | unchanged | unchanged |
| "any_all", every base experiment | unchanged | unchanged | `all_lda_s1`, EBM (0.058): test IC 0.002, net Sharpe -0.39 |
| "news" | `px_all`, lasso (0.045): IC -0.001, net -0.61 | `px_news`, EBM (0.036): IC 0.024 (1.3), net -0.07 | **`px_all`, EBM (0.027): IC 0.025 (2.3), net 0.61, buffered 0.65** |
| "shock" | **`narr_nolbert20_s63`, IPCA (0.065): IC 0.018 (0.9), net 0.68, buffered 0.57** | (no tuned window) | `all_lda_s1`, EBM: as "any_all" |
| "signal" | 21-day FinBERT tone (0.029): IC 0.006, net -0.82 | reversal without news (0.018): IC -0.006, net -2.43 | reversal without news (0.003): IC 0.004, net -1.96 |

- **Weekly, price and all the news, EBM** is the first experiment chosen on
  validation whose test IC clears its critical value (0.025, t = 2.31 against
  2.11; Clark-West t = 1.9). Net Sharpe 0.61, buffered 0.65 (95% interval
  -0.73 to 2.03), 9% a year with 14% volatility, maximum drawdown -13%. Three
  things limit it. The same EBM on the price features alone has a higher IC
  (0.030): the news does not add to the IC, and the Sharpe difference (0.76)
  has a bootstrap p-value of 0.27. It earns everything in 2022-23 (IC 0.037,
  buffered Sharpe 1.45) and loses in 2024 (IC 0.001, -1.08). And it is one of
  the 47 rows of the selection table (several of them the same experiment),
  of which two or so would clear a 5% critical value by chance.
- **Monthly, 20 NoLBERT topics with 63-day shocks, IPCA** (the "shock" pool)
  is positive in both halves of the test period (net 0.70 and 0.66), but its
  IC is not significant (t = 0.9) and the pool was defined after the
  narrative-only test results had been seen.
- **Biweekly** nothing beats the main run's choice (price and FinBERT text,
  XGBoost: net 0.37, buffered 0.41).

So a net Sharpe ratio of 0.6 to 0.7 is reached by validation choices at two
frequencies, with intervals that include zero and with the reservations
above. None of them can be presented as a demonstrated edge.

### 10.5 The cap-weighted tilt shows the market

| Test period | Monthly | Biweekly | Weekly |
|---|---|---|---|
| Value-weighted universe, net Sharpe | 0.52 | 0.61 | 0.47 |
| `tilt_vw`, median over the base experiments: net Sharpe | 0.50 | 0.60 | 0.46 |
| information ratio against the universe (positive in) | -0.16 (38%) | -0.27 (34%) | -0.29 (38%) |
| tracking error, turnover per period | 2.2%, 20% | 2.5%, 17% | 2.6%, 12% |
| experiments with an alpha t above 2 | 7 of 400 | 2 of 256 | 1 of 400 |

Tilting the index towards the forecasts leaves its Sharpe ratio where the
market put it: 0.46 to 0.60 total. The median information ratio is 0.07, 0.13
and 0.31 before costs (the weekly one is mostly the price reversal of section
4.6) and negative after. The tilt is a fair way to implement a signal
without shorting, if it is reported against the index; its total Sharpe ratio
is not evidence of skill.

### 10.6 The search, with everything

`multiple_testing.csv`, pool "all": 613 experiments monthly, 451 biweekly, 613
weekly, under eight rules.

- The best result is below what the best of that many portfolios without
  skill would show in 15 of the 24 combinations of rule and frequency: all 8
  biweekly, 6 of 8 weekly and 1 of 8 monthly.
- The monthly value-weighted long-short portfolios are the one place where the
  best result beats the search: 3.08 net (price, 20 NoLBERT topics with 63-day
  shocks, ridge) against 1.71 expected without skill, reality-check p = 0.004,
  SPA p = 0.003. The monthly tilt (best 2.24 against 1.66, SPA p = 0.017) and
  value-weighted long-only rule (2.33 against 1.74, SPA p = 0.045) point the
  same way. With 24 combinations searched, a p-value of 0.004 is about 0.1
  after a Bonferroni correction.
- No experiment passes a 10% false-discovery threshold in any combination.

### 10.7 Reading

- The directional news we can measure, headline tone and headline vectors a
  day after publication, does not forecast the returns of large stocks over
  the next week or month. The literature puts that effect in the first days
  and in small stocks; testing it needs time-stamped news and daily trading.
- The window of the attention shock matters more than anything else tried in
  this part: one day for weekly rebalancing and a quarter for monthly. The gain
  is consistent and not significant, and the best monthly numbers came from
  it.
- Net Sharpe ratios of 0.6 to 0.7 exist among validation choices; their
  intervals include zero, they are concentrated in one half of the test
  period, and the search statistics, apart from the monthly value-weighted
  portfolios, say that numbers like them are what this many trials produce.

## 11. Opening the glass box

For each frequency `scripts/09_explain.py` takes the EBM on price and text
features with the best validation IC and reports term importance, shape
functions and ablations in the manner of the Glass Box paper: a term or block
is deleted from the fitted model and nothing is re-estimated.

| | Monthly (NoLBERT) | Biweekly (FinBERT) | Weekly (LDA) |
|---|---|---|---|
| Boosting rounds kept for the main effects | 1 | 54 | 3 |
| Share of importance: narrative exposures | 14% | 41% | 54% |
| pairwise terms | 79% | 26% | 19% |
| firm news | 4.5% | 13% | 18% |
| price | 2.5% | 20% | 9% |
| Test IC, full model | 0.017 | -0.014 | 0.010 |
| after deleting the pairwise terms | 0.019 | -0.006 | 0.013 |
| after deleting the narrative block | 0.015 | -0.002 | 0.015 |
| after deleting the firm-news block | 0.017 | -0.016 | 0.009 |
| after deleting the price block | 0.016 | -0.020 | 0.003 |

- **The monthly and the weekly model have almost nothing in them.** Validation
  stopped the first after one boosting round and the second after three: their
  main effects are a handful of small steps, and the monthly model is almost
  all pairwise terms. What follows describes what the method shows, not a
  finding about returns.
- **Importance is not usefulness.** Narrative exposures carry 14% to 54% of
  the weight. Deleting them raises the IC of the biweekly and the weekly
  model.
- **The price block is the only one whose deletion lowers the IC at all three
  frequencies**, with 2.5% to 20% of the importance.
- **Pairwise terms hurt the IC**: deleting them raises it at all three
  frequencies, but lowers the net Sharpe ratio of the monthly (0.34 to 0.01)
  and biweekly (-0.33 to -0.62) models.

None of these differences is tested, and with ICs of this size none would be
significant. The earlier project's ablation (deleting one narrative from the
fitted IPCA) has the same form. It should be redone on this panel before it
is quoted.

## 12. Limits, and what to do next

**Limits of what was done.**

- *Short test sample, second half of the year only.* 1.56 to 1.75 years, and
  only holding periods from July to January (section 2.4).
- *The validation period is small and does three jobs* (section 4.2), and it
  pointed the wrong way twice (FinBERT, the EBM).
- *Three different yardsticks.* Hyperparameters are tuned on squared error,
  experiments are chosen on rank IC, and the portfolios use only the ordering
  of the forecasts.
- *The rank IC is not adjusted for beta or industry*, and returns are not
  winsorised (the rank-target models are the one exception).
- *Costs are a flat 10 basis points*, and portfolios trade at the close that
  ends the signal.
- *Firm-news features are noise for 39% to 51% of the rows* (section 2.3).
- *Text features partly re-encode prices*: a narrative exposure is a function
  of the stock's past returns.
- *Few text components are point-in-time*: the FinBERT topic classifier
  (2022) and MiniLM (2021) were published after the training period, and
  NoLBERT's adaptation used text up to September 2020 (section 8.1). The
  FinBERT tone model of section 10 (text to 2010) and the Loughran-McDonald
  dictionary (2014 release) are the exceptions.
- *News dates carry no time of day*, so every news feature waits a day, and
  the tone of section 10 is measured on headlines only.
- *One seed per experiment* (section 7.1).

**What to do next, roughly by how much it could change the conclusions.**

- **A fresh test of the tuned attention shocks.** Section 10.3 is the one lead
  of the second part: one-day shocks weekly, 63-day shocks monthly, and the
  20 NoLBERT topics among them. Fixing these choices now and testing them on
  months the project has not used (the January-to-June months that a complete
  return file brings back, or 2025) is the cleanest next test.
- **News at the speed it is priced.** Tone did not forecast a day after
  publication. Time-stamped news with daily (or open-to-close) returns, and
  the smaller stocks the literature finds it in, are what a tone test needs.

- **A complete return file.** With daily returns for every list member, the 40
  monthly periods that are left out come back (15 of them in the test period),
  the test sample grows by 70% to 95%, and the donor rule is needed for news
  only. `00_pull_crsp.py` does it for anyone with WRDS access; it is untested
  here.
- **A point-in-time news source.** The firm-level results are limited by
  coverage (section 2.3), not by models.
- **A clean NoLBERT comparison.** Re-adapt NoLBERT with pairs up to the end of
  2019 (`scripts/03c_finetune_encoder.py`, the default) and adapt MiniLM with
  the same recipe, then run both through the pipeline (GPU, about three hours
  each with the embedding).
- **A longer test.** A rolling origin (refit every year, test the next) would
  turn 2018 to 2024 into test years, at the price of a shorter first training
  period.
- **Seed averaging.** Fit XGBoost and the forest with ten seeds and average the
  forecasts (section 7.1). It would not create a signal, but it would stop
  single experiments from looking like one.
- **Refitting topics through time.** Topics are fixed at 2019. An expanding
  refit would let new subjects in, at the price of features whose meaning
  changes.
- **The audits again.** Rerun `scripts/10` to `12` under the final rules, and
  add an evaluation of firm-news features on covered firms only.

## 13. Where each number comes from

| Section | File |
|---|---|
| 1, 4.2 | `results/tables/selection.csv`, `forecast.csv`, `strategies.csv`, `multiple_testing.csv`, `improvements.csv` |
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
| 8 | `results/encoders/nolbert_ft/`, `results/tables/`; section 8.3 from the teammate's other pipeline |
| 9 | `results/tables/improvements.csv`, `selection.csv`, `strategies.csv` |
| 10 | `results/tuning/shock_window.csv`, `results/tuning/<freq>/signals.csv` and `pool.csv`; `results/tables/improvements.csv` (shock_window), `landscape.csv`, `text_vs_price.csv`, `selection.csv` (pools news, shock, signal, any_all), `strategies.csv` (tilt_vw), `multiple_testing.csv` (pool all); the validation comparisons of section 10.1 from the validation rows of `forecast.csv` and `results/series/<freq>_forecast.parquet`; the coverage of firm news computed with `ftm/features/tone.py` |
| 11 | `results/explain/<freq>__<features>/`, `results/tables/tuning.csv` |

Counts and medians "over experiments" are taken over the rows of these tables.
The split of the weekly price IC by the sign of the market return (section
4.6) is computed from `results/series/W_forecast.parquet` and
`W_factors.csv`. The article counts by year before 2012, the named companies
of section 2.3 and the size of the dump of 16 December 2023 come from the
audit of the raw files (`ftm/data/fnspid.py`, `ftm/data/finmultitime.py`) and
of `data/news/links.parquet`.
