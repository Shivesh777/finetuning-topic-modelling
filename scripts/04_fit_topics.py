"""Fit a topic model on training-period text and apply it, unchanged, to every period.

    python scripts/04_fit_topics.py lda [--name lda] [--backend auto|tomotopy|sklearn]
    python scripts/04_fit_topics.py embedding --embeddings TAG [--name NAME]
    python scripts/04_fit_topics.py weights --weights TAG --name NAME
    python scripts/04_fit_topics.py lda --all --reuse lda --name lda_all

lda        latent Dirichlet allocation on word counts (the benchmark)
embedding  spherical k-means on the sentence embeddings of scripts/03_embed.py
weights    article-topic weights that already exist: the class probabilities of
           scripts/03b_classify_topics.py (nothing is fitted)

The topics and the vocabulary are estimated on a per-day sample of articles
dated up to the end of the training period (config.SPLITS); later articles are
only labelled. LDA labels the per-day attention sample (every article with
--all); the other models label the articles they have vectors for. With
--reuse NAME the topics of data/topics/NAME are applied instead of fitted, and
--per-day N keeps at most N labelled articles a day.

Writes data/topics/<name>/:
    daily_sums.parquet   per calendar day, the number of labelled articles and the
                         sum of their topic weights (attention = sums / their total)
    firm_daily.parquet   articles by (date, ticker, main topic); articles filed under
                         more than config.FIRM_MAX_TICKERS tickers are left out
    topics.csv           per topic: top terms, size, coherence, stability
    quality.json         summary of the quality measures (ftm/topics/quality.py)
    model.npz            what is needed to label new text
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config
from ftm.data import news
from ftm.topics import attention, cluster, lda, quality, text

TOP_TERMS = 10
RELIABILITY_SIZES = (50, 100, 150, 300)        # articles per day


def samples(dates: pd.DatetimeIndex, labelled: np.ndarray):
    """(fit, later): per-day samples of labelled articles in and after the training period.

    Both are drawn from the standard attention sample, so that every model that
    labels that sample is fitted and scored on the same articles."""
    pool = attention.sample_per_day(dates, config.ATTENTION_ARTICLES_PER_DAY, config.SEED)
    base = np.flatnonzero(pool & labelled)
    pick = attention.sample_per_day(dates[base], config.FIT_ARTICLES_PER_DAY, config.SEED + 1)
    is_train = np.asarray(dates[base] <= config.SPLITS["train"][1])
    return base[pick & is_train], base[pick & ~is_train]


def reliability(weights, dates, train_end) -> dict:
    """Mean split-half reliability of the topics' attention shocks, by articles per day."""
    out = {}
    per_day = pd.Series(1, index=dates).groupby(level=0).sum()
    typical = per_day[per_day.index.dayofweek < 5].median()
    sizes = [s for s in RELIABILITY_SIZES if s < typical] + [None]
    for label, span in (("training", dict(end=train_end)),
                        ("later", dict(start=train_end + pd.Timedelta(days=1)))):
        for size in sizes:
            table = quality.shock_reliability(weights, dates, size, config.SHOCK_WINDOW,
                                              config.SEED, **span)
            out[f"{label}_{size or 'all'}_per_day"] = {
                "mean": round(float(table["reliability"].mean()), 4),
                "median": round(float(table["reliability"].median()), 4),
                "lowest_topic": round(float(table["reliability"].min()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["lda", "embedding", "weights"])
    ap.add_argument("--name", help="folder of data/topics/ (default: the model, or the encoder)")
    ap.add_argument("--embeddings", "--weights", dest="embeddings",
                    help="tag of data/topics/embeddings/<tag>.npy")
    ap.add_argument("--all", action="store_true", help="lda: label every article")
    ap.add_argument("--per-day", type=int, help="label at most this many articles a day")
    ap.add_argument("--center", action="store_true",
                    help="embedding: subtract the mean training embedding first (for encoders "
                         "that were not trained on sentence pairs, whose vectors all point "
                         "one way)")
    ap.add_argument("--reuse", help="apply the fitted topics of this folder of data/topics/")
    ap.add_argument("--topics", type=int, default=config.N_TOPICS)
    ap.add_argument("--backend", default="auto", choices=["auto", "tomotopy", "sklearn"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.model != "lda" and not a.embeddings:
        ap.error(f"{a.model} needs the tag of the article vectors")
    name = a.name or ("lda" if a.model == "lda" else a.embeddings.split("_")[0])
    out = config.TOPICS / name
    k, t0 = a.topics, time.time()
    old = np.load(config.TOPICS / a.reuse / "model.npz", allow_pickle=False) if a.reuse else None
    if old is not None:
        k = len(old["topic_word"] if a.model == "lda" else old["centroids"])

    index = news.load_articles(columns=["article_id", "date", "n_tickers"])
    n = len(index)
    assert (index["article_id"].to_numpy() == np.arange(n)).all()
    dates = pd.DatetimeIndex(index["date"])
    train_end = pd.Timestamp(config.SPLITS["train"][1])

    # --- which articles the model labels, and the samples it is fitted and scored on -----
    labelled = np.zeros(n, dtype=bool)
    if a.model == "lda":
        labelled[:] = True if a.all else attention.sample_per_day(
            dates, config.ATTENTION_ARTICLES_PER_DAY, config.SEED)
    else:
        store = config.TOPICS / "embeddings"
        ids = np.load(store / f"{a.embeddings}.ids.npy")
        assert (np.diff(ids) > 0).all(), "article ids of the vectors must be sorted"
        labelled[ids] = True
    if a.per_day:
        labelled &= attention.sample_per_day(dates, a.per_day, config.SEED)
    rows = np.flatnonzero(labelled)
    if a.model != "lda":                     # the vectors of the labelled articles, in row order
        vectors = np.load(store / f"{a.embeddings}.npy", mmap_mode="r")
        if len(rows) < len(ids):
            vectors = vectors[np.searchsorted(ids, rows)]
        if a.model == "weights":
            k = vectors.shape[1]
    fit, later = samples(dates, labelled)
    print(f"{name}: {len(rows):,} of {n:,} articles labelled; topics from {len(fit):,} training "
          f"articles; {len(later):,} later articles for out-of-sample coherence", flush=True)

    # --- word counts (LDA needs them for every labelled article) ---------------------------
    need = labelled.copy() if a.model == "lda" else np.zeros(n, dtype=bool)
    need[fit] = need[later] = True
    position = np.cumsum(need) - 1                       # article id -> row of `counts`
    cleaned = text.clean(news.text_of(
        news.load_articles(columns=["article_id", "title", "lead"], keep=need)))
    vectoriser = (text.with_vocabulary(old["vocabulary"]) if old is not None
                  else text.fit_vocabulary(cleaned.iloc[position[fit]]))
    counts = text.count(vectoriser, cleaned)
    del cleaned
    counts_fit, counts_later = counts[position[fit]], counts[position[later]]
    print(f"vocabulary of {counts.shape[1]:,} terms ({time.time() - t0:.0f} s)", flush=True)

    # --- topics, article weights and a second fit with another seed -------------------------
    model = {"vocabulary": np.asarray(text.terms_of(vectoriser), dtype=str)}
    labels_other = None                  # main topics under a second seed (not when reusing)
    if a.model == "lda":
        topic_word = (old["topic_word"] if old is not None else
                      lda.fit_topic_word(counts_fit, k, config.SEED, a.backend, n_jobs=a.jobs))
        print(f"topics fitted ({time.time() - t0:.0f} s)", flush=True)
        weights = lda.infer(counts[position[rows]], topic_word, n_jobs=a.jobs)
        if old is None:
            other = lda.fit_topic_word(counts_fit, k, config.SEED + 1, a.backend, n_jobs=a.jobs)
            labels_other = lda.infer(counts_fit, other, n_jobs=a.jobs).argmax(axis=1)
        model["topic_word"] = topic_word.astype(np.float32)
    elif a.model == "embedding":
        centre = (old["centre"] if old is not None and "centre" in old
                  else np.zeros(vectors.shape[1], np.float32))
        if a.center and old is None:
            centre = np.asarray(vectors[np.searchsorted(rows, fit)], np.float32).mean(axis=0)
        if centre.any():
            vectors = cluster.centred(vectors, centre)
        x_fit = np.asarray(vectors[np.searchsorted(rows, fit)], dtype=np.float32)
        centroids = (old["centroids"] if old is not None else
                     cluster.spherical_kmeans(x_fit, k, config.SEED)[0])
        print(f"topics fitted ({time.time() - t0:.0f} s)", flush=True)
        weights = cluster.soft_weights(vectors, centroids, config.TOPIC_TEMPERATURE)
        if old is None:
            other = cluster.spherical_kmeans(x_fit, k, config.SEED + 1, n_init=1)[0]
            labels_other = cluster.assign(x_fit, other)
        model.update(centroids=centroids, centre=centre, encoder=a.embeddings,
                     temperature=config.TOPIC_TEMPERATURE)
    else:                                    # class probabilities: nothing to fit
        weights = np.asarray(vectors, dtype=np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        model.update(classifier=a.embeddings, classes=np.asarray(json.loads(
            (store / f"{a.embeddings}.labels.json").read_text()), dtype=str))
    labels = np.full(n, -1, dtype=np.int16)
    labels[rows] = weights.argmax(axis=1)

    # --- attention and firm-level tables -------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "model.npz", **model)
    sums = attention.daily_sums(weights, dates[rows])
    attention.save_daily_sums(sums, out / "daily_sums.parquet")
    links = pd.read_parquet(config.NEWS / "links.parquet")
    article = links["article_id"].to_numpy()
    focused = (labels[article] >= 0) & (index["n_tickers"].to_numpy()[article]
                                        <= config.FIRM_MAX_TICKERS)
    firm = attention.firm_daily(labels, dates, links[focused])
    firm.to_parquet(out / "firm_daily.parquet", index=False)

    # --- quality ---------------------------------------------------------------------------------
    # Top terms by class-based TF-IDF of each topic's training articles: one rule for every model.
    top = quality.top_terms(quality.ctfidf(counts_fit, labels[fit], k), TOP_TERMS)
    terms = model["vocabulary"]
    share = np.bincount(labels[fit], minlength=k) / len(fit)
    if labels_other is not None:
        kept, rand = quality.label_agreement(labels[fit], labels_other, k)
    elif a.model == "weights":             # a fixed classifier has no seed to vary
        kept, rand = np.full(k, np.nan), None
    else:                                  # stability belongs to the fitted topics: carry it over
        kept = pd.read_csv(config.TOPICS / a.reuse / "topics.csv")["stability"].to_numpy()
        rand = json.loads((config.TOPICS / a.reuse / "quality.json").read_text())[
            "stability_adjusted_rand"]
    table = pd.DataFrame({
        "topic": model["classes"] if a.model == "weights" else np.arange(k),
        "share_of_training_articles": share.round(4),
        "npmi_training": quality.npmi(top, counts_fit).round(4),
        "npmi_later": quality.npmi(top, counts_later).round(4),
        "stability": kept.round(4),
        "top_terms": [", ".join(terms[row]) for row in top]})
    summary = {
        "name": name, "model": a.model, "topics": int(k), "articles_labelled": int(len(rows)),
        "articles_fit": int(len(fit)), "articles_later": int(len(later)),
        "fit_until": str(train_end.date()), "vocabulary": int(len(terms)),
        "npmi_training": round(float(table["npmi_training"].mean()), 4),
        "npmi_later": round(float(table["npmi_later"].mean()), 4),
        "diversity_top10": round(quality.diversity(top), 4),
        "stability_share_kept": None if rand is None else round(float((kept * share).sum()), 4),
        "stability_adjusted_rand": None if rand is None else round(rand, 4),
        "largest_topic_share": round(float(share.max()), 4),
        "effective_topics": round(float(np.exp(-(share[share > 0] * np.log(share[share > 0])).sum())), 1),
        "firm_articles": int(firm["n"].sum()),
    }
    if a.model == "lda":
        # The customary LDA listing: the most probable words of each topic.
        likely = quality.top_terms(topic_word, TOP_TERMS)
        table["top_terms_by_probability"] = [", ".join(terms[row]) for row in likely]
        summary["npmi_training_by_probability"] = round(
            float(quality.npmi(likely, counts_fit).mean()), 4)
        summary["npmi_later_by_probability"] = round(
            float(quality.npmi(likely, counts_later).mean()), 4)
    summary["shock_reliability"] = reliability(weights, dates[rows], train_end)
    summary["seconds"] = round(time.time() - t0)
    table.to_csv(out / "topics.csv", index=False)
    (out / "quality.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(table.sort_values("share_of_training_articles", ascending=False)
          .head(12)[["topic", "share_of_training_articles", "npmi_training", "npmi_later",
                     "stability", "top_terms"]].to_string(index=False))


if __name__ == "__main__":
    main()
