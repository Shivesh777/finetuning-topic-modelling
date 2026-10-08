# Fine-tuning topic models for return prediction

Do the topics that financial news is about help to forecast stock returns?
This repository builds daily topic attention from 1.26 million Nasdaq.com
articles (2012 to 2024), turns it into stock-level features, and tests those
features with eight forecasting models (and variants of them) and seven
portfolio rules under a strict training / validation / test protocol. The short answer, on our data, is that
we find no evidence that they do, in a test sample that could only have shown a
large effect: `FINDINGS.md` says why and what the data would have to look like
to say more.

It is a rebuild of our earlier project (latent Dirichlet allocation or
sentence-embedding topics, IPCA, a Sharpe-optimised portfolio). The earlier
pipeline had no validation set and several ways for future information to
reach the model. This one separates training, validation and test periods, and
says where information from the future can still get in (`FINDINGS.md`,
sections 2 and 9).

- **`FINDINGS.md`**: what we found, with the numbers.
- **`REFERENCE.md`**: data sources, every step and parameter, the statistics.
- **`data/README.md`**: what is in the data folder and what is (not) in git.

## Layout

```
ftm/                 the library
  config.py            paths, dates and the parameters the stages share
  data/                news extraction, article table, returns and universe
  topics/              LDA, embedding clusters, classifier topics, quality measures
  features/            price features, narrative exposures, firm-level news
  panel.py             the stock-period panel (ranks, targets, missing data)
  models/              OLS, ridge, lasso, IPCA, sparse IPCA, XGBoost, random forest, EBM
  portfolio/           weights, risk model, backtest with costs, strategies
  evaluation/          performance metrics, statistical tests, the result tables
  experiments.py       names of feature sets and forecast files
scripts/             the pipeline, numbered in the order it runs
tests/               unit tests and an end-to-end run on an artificial market
data/                inputs and intermediate files (mostly not in git)
results/             tables, per-period series, tuning records, explanations
```

## Set-up

Python 3.10 or newer.

```bash
pip install -r requirements.txt
python -m pytest tests -q          # about a minute; needs no data
```

## Data

Three things are needed, in `data/` (see `data/README.md` for the details):

1. **Returns** (`data/crsp/`). CRSP is licensed and is **not in this
   repository**. Either unzip the team's data folder (`data.zip`, shared
   privately within the team), or, with a WRDS account, run
   `python scripts/00_pull_crsp.py` (written for this purpose but not yet run
   by us). `00_prepare_crsp.py` built our files from the folders of the earlier
   project and is of use only to someone who has those.
2. **Articles** (`data/news/`). In the team's data folder; or rebuilt from the
   public FinMultiTime and FNSPID datasets with scripts 01 and 02 (a 3.6 GB
   download and a 23 GB file that is read in pieces without being stored).
3. **Topic models** (`data/topics/<model>/`). Committed (`lda`, `lda_all`,
   `minilm`, `finbert`, `nolbert`, `nolbert20`), so steps 3 and 4 can be
   skipped unless you want to change the topic models. The adapted NoLBERT
   encoder (434 MB) and the article embeddings (about 1.2 GB) are not in git:
   they are in the teammate's bundle `nolbert-ftm-runnable` (`encoder/nolbert_ft/`
   goes to `data/topics/encoders/nolbert_ft/`, and
   `data/topics/embeddings/nolbert_ft_d150.*` to the same place here).

With the team zip: unzip it so that `data/crsp/`, `data/news/` and
`data/topics/` sit in the repository's `data/` folder.

## Replicate

```bash
python scripts/00_prepare_crsp.py          # or 00_pull_crsp.py; skip with the team zip
python scripts/01_download_news.py --source finmultitime   # skip with the team zip
python scripts/01_download_news.py --source fnspid         # (the right article text)
python scripts/02_build_articles.py        # skip with the team zip

python scripts/03_embed.py                 # sentence embeddings, 30-45 min on a CPU
python scripts/04_fit_topics.py lda        # fits new topics; add --reuse lda to apply ours
python scripts/04_fit_topics.py lda --all --reuse lda --name lda_all
python scripts/04_fit_topics.py embedding --embeddings paraphrase-MiniLM-L3-v2_d150 --name minilm
python scripts/03b_classify_topics.py      # optional: FinBERT topic classifier, 50 min
python scripts/04_fit_topics.py weights --weights finbert-topic_d150 --name finbert
python scripts/03c_finetune_encoder.py --grad-ckpt --out data/topics/encoders/nolbert_ft   # optional, GPU, 150 min
python scripts/03_embed.py --encoder data/topics/encoders/nolbert_ft    # GPU, about 1 hour
python scripts/04_fit_topics.py embedding --embeddings nolbert_ft_d150 --name nolbert
python scripts/04_fit_topics.py embedding --embeddings nolbert_ft_d150 --name nolbert20 --topics 20

python scripts/05_build_panel.py --placebo lda     # monthly, biweekly and weekly panels
python scripts/06_run_models.py            # tune, fit, forecast: 1,092 experiments, 70 min
python scripts/06b_ensemble.py             # the ensemble of the eight base models, 1 min
python scripts/07_evaluate.py              # forecast accuracy and backtests, 5 min
python scripts/08_report.py                # tables in results/tables/
python scripts/09_explain.py               # EBM importances, shapes, ablations
python scripts/10_audit.py                 # news coverage, stale rows, donor rows
python scripts/11_missing_data_leak.py     # what the missing-data leak is worth
python scripts/12_tuning_noise.py          # one experiment under ten random seeds
```

`06_run_models.py --models ols ridge lasso ipca sipca xgb rf ebm` fits only the
eight base models (624 experiments, 53 min); the six rank-target variants are
the rest. `03c_finetune_encoder.py` draws its pairs from articles up to the end
of the training period by default; the committed `nolbert` topics come from the
teammate's encoder, which used pairs up to September 2020 (FINDINGS.md, section
8.1).

Times are for four laptop CPU cores. With the team zip, start at
`05_build_panel.py`: everything from there to the tables takes about an hour
and a half, under an hour with the base models only (or start at
`07_evaluate.py` if the zip includes `predictions/`).
The results in `results/` are committed, so the tables can be read and
`08_report.py` rerun without fitting anything.

Every script prints what it does with `--help`. Scripts 03, 06 and 07 can be
stopped and restarted: they continue where they were (`--budget SECONDS` stops
them on purpose). To redo one experiment, empty its `.log` file in
`results/tuning/<freq>/` or pass `--force`.

LDA by Gibbs sampling is not bit-reproducible across machines and thread
counts. The fitted topics we used are in `data/topics/lda/model.npz`;
`--reuse lda` applies them instead of fitting new ones.

## Work on top of it

- **A new topic model.** Anything that gives each article a vector works.
  Embed with another encoder (`03_embed.py --encoder NAME`), then
  `04_fit_topics.py embedding --embeddings <tag> --name <name>`; or write
  article-topic weights to `data/topics/embeddings/<tag>.npy` (with the
  article ids in `<tag>.ids.npy` and the topic names in `<tag>.labels.json`)
  and use `04_fit_topics.py weights`. Then `05_build_panel.py --topics <name>
  --skip-price` adds its two feature blocks, and scripts 06 to 08 pick up the
  new feature sets by themselves.
- **NoLBERT** (a language model with no knowledge of our sample period):
  `03c_finetune_encoder.py` adapts it to training-period news, after which it
  is embedded like any other encoder. Both steps need a GPU; see the script.
- **A new model.** Add a class with `name`, `grid()`, `fit`, `predict` and
  `refit_params` (see `ftm/models/tuning.py`), register it in `MODELS` and add
  its name to `MODEL_ORDER` in `ftm/experiments.py`. A variant fitted on ranked
  returns comes for free: add the class to `RANK_TARGET` in
  `ftm/models/tuning.py` and `<name>_rk` to `RANK_MODELS`.
- **Another combination of models.** `scripts/06b_ensemble.py` averages the
  ranks of `experiments.BASE_MODELS`; any other list of members works the same
  way.
- **A new portfolio rule.** Add a branch to `weights_for` in
  `ftm/portfolio/strategies.py` and its name to `LONG_SHORT` or `LONG_ONLY`; a
  rule that depends on what was held before (like `ls_buf`) goes into the
  loop of `backtests`.
- **A new feature block.** A function from a formation day to an array
  (stocks x features), added to `makers` in `ftm/features/blocks.py`.
  Missing values are fine: see the note on missing data in `ftm/panel.py`
  before adding anything that depends on data availability. A feature set
  that uses the block is declared in `feature_sets` in `ftm/experiments.py`.
- **A complete return file.** With WRDS access, `00_pull_crsp.py` and
  `DAILY_RETURNS_COMPLETE_UNTIL = None` in `ftm/config.py` bring back the
  months that are left out now (15 of 36 test months). This is the most
  useful thing anyone can add.
- **Other dates, universe size, costs, windows:** `ftm/config.py`.

## Rules we kept

1. Chronological split; settings and the headline experiment are chosen on
   the validation period. (The test period has been evaluated three times in
   all, after corrections of data and method that `FINDINGS.md` lists.)
2. Nothing fitted on later data is used earlier: topic models, vocabulary and
   forecasting models see the training period only (the forecasting models
   also the validation period, for the final fit).
3. The universe of a date is a list that existed on that date; news of a day is
   used from the next day on.
4. Where our data are incomplete for reasons that depend on the future, the
   periods are left out (returns) or the model cannot see what is missing (news).
5. One experiment is the headline: the one the validation period selects.
   Everything else is reported with the number of things we tried.
6. Controls next to the main claims: exposures to misdated news next to
   exposures to the news, price features next to price and text, ten seeds
   next to one.
7. t statistics are read against small-sample critical values, and the best
   of many experiments against what the best of as many would show by chance.
8. A change meant to improve the results is adopted on validation evidence
   written down before the test period is looked at (FINDINGS.md, section
   9.1), and reported whether or not it then helps.

## Licences and credits

Code: see `LICENSE`. News: FNSPID (Dong, Fan and Peng, 2024; CC BY-NC 4.0) and
FinMultiTime (MIT); the articles belong to their publishers. Returns: CRSP via
WRDS (licensed, not redistributed). Methods: Bybee, Kelly and Su (2023) for
narrative exposures; Kelly, Pruitt and Su (2019) for IPCA; Bell, Kakhbod,
Lettau and Nazemi (2024) for the explainable boosting approach; Kakhbod and Li
(2025) for NoLBERT.
