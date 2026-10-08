# The data folder

Everything the scripts read and write that is not a result table lives here.
Nothing in this folder is needed from outside the repository once it is filled,
so zipping `data/` is enough to hand the project to a teammate.

| Folder | What it holds | Made by | In git? |
|---|---|---|---|
| `crsp/` | Daily and monthly returns, June lists of the 1,000 largest stocks, tickers, Fama-French daily factors | `00_prepare_crsp.py` or `00_pull_crsp.py` | **No (licensed)** |
| `news/articles/` | One row per unique article: date, headline, first 500 characters of the body. One file per quarter | `01_download_news.py`, `02_build_articles.py` | No (third-party text) |
| `news/links.parquet` | The tickers each article is filed under | `02_build_articles.py` | No |
| `topics/embeddings/` | Per-article vectors: sentence embeddings (`03_embed.py`) or class probabilities (`03b_classify_topics.py`), with the article ids | scripts 03 | No (hundreds of MB) |
| `topics/encoders/` | Adapted encoders, such as NoLBERT (`03c_finetune_encoder.py`; the teammate's is `nolbert_ft`, 434 MB) | script 03c | No |
| `topics/<model>/` | A fitted topic model and what the panels need from it (see below): `lda`, `lda_all`, `minilm`, `finbert`, `nolbert`, `nolbert20` | `04_fit_topics.py` | Yes |
| `panels/<freq>/` | Stock-period panels: `index.parquet` and one `.npy` per feature block | `05_build_panel.py` | **No (derived from CRSP)** |
| `predictions/<freq>/` | Forecasts of every experiment for the validation and test rows | `06_run_models.py` | **No (derived from CRSP)** |
| `models/<freq>/` | Fitted EBMs, for the explanations | `06_run_models.py` | **No** |

`<freq>` is `M` (monthly), `2W` (biweekly) or `W` (weekly).

## Why some of it is not in git

- **CRSP** comes from WRDS under a licence that does not allow redistribution.
  `data/crsp/` and every stock-level file derived from it (`panels/`,
  `predictions/`, `models/`) are ignored by git whatever their size. Share
  them only with teammates who are covered by the same licence. The result
  tables in `results/` hold portfolio-level numbers only and are committed.
- **Article text** comes from two public research datasets (see REFERENCE.md);
  the articles themselves belong to their publishers. We keep excerpts out of
  the repository; `scripts/01` and `02` rebuild them from the public datasets.
- **Embeddings** are large (400 MB per encoder) and are rebuilt by `scripts/03`.

To make the zip smaller, leave out `panels/`, `predictions/` and `models/`:
scripts 05, 06 and 06b rebuild them from `crsp/` and `topics/` in about 70 minutes.

## Files of a topic model (`topics/<model>/`)

| File | Content |
|---|---|
| `daily_sums.parquet` | Per calendar day: `n_articles` and the sum of the articles' weights on each topic (`t00`, `t01`, ...). Daily attention is a row divided by its sum |
| `firm_daily.parquet` | Number of articles by `date`, `ticker` and main `topic`, for articles filed under at most three tickers |
| `topics.csv` | One row per topic: share of training articles, coherence in training and later text, stability across seeds, top terms |
| `quality.json` | Summary measures, including the reliability of the daily attention shocks |
| `model.npz` | What is needed to label new text: topic-word matrix or centroids, and the vocabulary |

## Files of `crsp/`

| File | Columns |
|---|---|
| `universe.parquet` | `formed` (30 June of year Y), `permno`, `me`, `rank` (1 to 1,000) |
| `daily.parquet` | `date`, `permno`, `ret` (decimal), `me` (market value) |
| `monthly.parquet` | `permno`, `date` (month end), `ret` |
| `tickers.parquet` | `permno`, `ticker`, `last_date` |
| `ff_daily.parquet` | `date`, `mktrf`, `smb`, `hml`, `rf` (decimals) |

Raw downloads (several GB) never enter this folder: they go to `~/.cache/ftm`,
or to the folder named by the environment variable `FTM_CACHE`.
