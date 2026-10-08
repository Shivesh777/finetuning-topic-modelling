"""Tone of the firm-level articles: FinBERT on headlines, Loughran-McDonald word counts,
and principal components of FinBERT's headline vectors.

    python scripts/03d_sentiment.py [--tickers FILE] [--threads N] [--budget SECONDS]

Scores the articles that firm-level features can use: filed under at most
config.FIRM_MAX_TICKERS tickers, one of which belongs to a stock of our universe
(the tickers of data/crsp/tickers.parquet, or of FILE, a csv or parquet file with a
`ticker` column). See ftm/topics/sentiment.py for the two measures. Writes
data/sentiment/:

    ids.npy             article ids of the scored articles (ascending)
    lm.npy              (n, 3) int32: words, LM positive and LM negative words (headline + lead)
    finbert_probs.npy   (n, 3) float16: P(positive), P(negative), P(neutral) of the headline
    finbert_emb.npy     (n, 768) float16: FinBERT's mean last hidden state over the headline
    finbert_pca.npz     mean and components of a PCA of those vectors, fitted on all the
                        articles of the training period and on nothing later
    finbert_pca.npy     (n, config.TONE_PCA_DIM) float16: the vectors on those components

FinBERT needs about three hours of two laptop cores for 760,000 headlines (ONNX Runtime,
8-bit weights); the run resumes when restarted, and --budget stops it on purpose.
"""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config
from ftm.data import news
from ftm.topics import sentiment


def universe_tickers(path: str | None) -> set:
    if path is None:
        path = config.CRSP / "tickers.parquet"
    path = Path(path)
    table = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    return set(table["ticker"].dropna().astype(str))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="csv or parquet file with a `ticker` column")
    ap.add_argument("--threads", type=int, help="ONNX Runtime threads (default: all cores)")
    ap.add_argument("--budget", type=float, help="stop FinBERT after this many seconds")
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()
    out = config.SENTIMENT
    out.mkdir(parents=True, exist_ok=True)

    index = news.load_articles(columns=["article_id", "date", "n_tickers"])
    links = pd.read_parquet(config.NEWS / "links.parquet")
    linked = np.zeros(int(index["article_id"].max()) + 1, dtype=bool)
    linked[links.loc[links["ticker"].isin(universe_tickers(a.tickers)), "article_id"]] = True
    keep = linked & np.isin(np.arange(len(linked)), index.loc[
        index["n_tickers"] <= config.FIRM_MAX_TICKERS, "article_id"])
    ids = np.flatnonzero(keep)
    if (out / "ids.npy").exists() and not np.array_equal(np.load(out / "ids.npy"), ids):
        raise SystemExit(f"{out} holds scores of other articles; move it away to start again")
    np.save(out / "ids.npy", ids)
    print(f"{len(ids):,} firm articles of {len(index):,}", flush=True)
    done = int((out / "finbert.done").read_text()) if (out / "finbert.done").exists() else 0
    if done < len(ids) or not (out / "lm.npy").exists():
        articles = news.load_articles(columns=["article_id", "title", "lead"], keep=keep)
        if not (out / "lm.npy").exists():
            t0 = time.time()
            np.save(out / "lm.npy", sentiment.lm_counts(news.text_of(articles).tolist()))
            print(f"Loughran-McDonald counts ({time.time() - t0:.0f} s)", flush=True)
        titles = articles["title"].fillna("").tolist()
        del articles
        sentiment.finbert(titles, out, threads=a.threads, budget=a.budget,
                          batch_size=a.batch_size)
        del titles
        done = int((out / "finbert.done").read_text())
    if done < len(ids):
        print(f"FinBERT: {done:,} of {len(ids):,} done; run again to continue")
        return
    if not (out / "finbert_pca.npy").exists():
        emb = np.load(out / "finbert_emb.npy", mmap_mode="r")
        dates = index.set_index("article_id").loc[ids, "date"].to_numpy()
        train = np.flatnonzero(dates <= np.datetime64(config.SPLITS["train"][1]))
        mean, components = sentiment.pca_fit(emb, train, config.TONE_PCA_DIM)
        np.savez(out / "finbert_pca.npz", mean=mean, components=components)
        np.save(out / "finbert_pca.npy", sentiment.pca_apply(emb, mean, components))
        print(f"PCA of {len(train):,} training-period vectors -> {config.TONE_PCA_DIM} components")


if __name__ == "__main__":
    main()
