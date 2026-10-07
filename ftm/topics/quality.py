"""Quality measures of a topic model.

coherence    NPMI of each topic's top terms, from document co-occurrence in a
             reference set of articles (Lau, Newman and Baldwin, 2014); a term
             pair that never co-occurs scores -1
diversity    share of distinct terms among all topics' top terms
stability    two fits that differ only in the random seed label the same articles;
             topics are matched one to one and a topic's stability is the share of
             its articles that stay in its match (the same measure for every kind
             of model, unlike a similarity of topic vectors)
reliability  how much of the daily attention SHOCK of a topic is signal: the
             day's articles are split at random into two halves, the shock
             series is computed from each half, and their correlation r is
             stepped up to the full day by Spearman-Brown, 2r / (1 + r)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from . import attention as att


def ctfidf(counts: sparse.csr_matrix, labels: np.ndarray, k: int) -> np.ndarray:
    """Class-based TF-IDF (k, n_terms): term frequency within a topic's articles
    times log(1 + average topic size / term frequency) (Grootendorst, 2022)."""
    onehot = sparse.csr_matrix((np.ones(len(labels), dtype=np.float64),
                                (labels, np.arange(len(labels)))), shape=(k, len(labels)))
    tf = np.asarray((onehot @ counts).todense())
    total = tf.sum(axis=0)
    idf = np.log1p(tf.sum() / k / np.maximum(total, 1.0))
    return tf / np.maximum(tf.sum(axis=1, keepdims=True), 1.0) * idf


def top_terms(scores: np.ndarray, n: int = 10) -> np.ndarray:
    """Indices (k, n) of each topic's highest-scoring terms."""
    return np.argsort(-scores, axis=1)[:, :n]


def npmi(top: np.ndarray, counts: sparse.csr_matrix) -> np.ndarray:
    """Mean pairwise NPMI of each topic's top terms on the reference articles."""
    n_docs = counts.shape[0]
    presence = (counts > 0).astype(np.float32).tocsc()
    out = np.empty(len(top))
    upper = np.triu_indices(top.shape[1], 1)
    for j, terms in enumerate(top):
        sub = presence[:, terms].toarray()
        joint = (sub.T @ sub) / n_docs
        p = np.diag(joint)
        pij = joint[upper]
        pipj = np.outer(p, p)[upper]
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.log(pij / pipj) / -np.log(pij)
        out[j] = np.where(pij > 0, score, -1.0).mean()
    return out


def diversity(top: np.ndarray) -> float:
    return len(np.unique(top)) / top.size


def label_agreement(a: np.ndarray, b: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """How far two fits group the same articles together.

    a, b: main topic of every article under fit A and under fit B. Topics are matched
    one to one so that as many articles as possible keep their (matched) topic.
    Returns the share of each A-topic's articles found in its match, and the adjusted
    Rand index of the two partitions (0 = chance, 1 = identical)."""
    table = np.zeros((k, k))
    np.add.at(table, (a, b), 1.0)
    rows, cols = linear_sum_assignment(-table)
    kept = np.zeros(k)
    kept[rows] = table[rows, cols] / np.maximum(table.sum(axis=1)[rows], 1.0)
    return kept, float(adjusted_rand_score(a, b))


def shocks(sums: pd.DataFrame, window: int = 5, weekdays_only: bool = True) -> pd.DataFrame:
    """Attention minus its mean over the previous `window` days."""
    theta = att.attention(sums)
    if weekdays_only:
        theta = theta[theta.index.dayofweek < 5]
    return theta - theta.rolling(window).mean().shift(1)


def shock_reliability(weights: np.ndarray, dates, per_day: int | None = None, window: int = 5,
                      seed: int = 0, start=None, end=None) -> pd.DataFrame:
    """Split-half reliability of each topic's daily attention shock.

    per_day: first cut every day to this many random articles (None = all).
    Returns one row per topic: r_half (correlation of the two half-sample shock
    series over days) and reliability (Spearman-Brown, for the whole day)."""
    dates = pd.DatetimeIndex(dates)
    keep = np.ones(len(dates), dtype=bool)
    if start is not None:
        keep &= dates >= start
    if end is not None:
        keep &= dates <= end
    if per_day is not None:
        keep &= att.sample_per_day(dates, per_day, seed)
    half = att.random_halves(dates, seed + 1)
    za = shocks(att.daily_sums(weights, dates, keep & half), window)
    zb = shocks(att.daily_sums(weights, dates, keep & ~half), window)
    za, zb = za.align(zb, join="inner", axis=0)
    ok = za.notna().all(axis=1) & zb.notna().all(axis=1)
    a, b = za[ok].to_numpy(), zb[ok].to_numpy()
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    r = (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return pd.DataFrame({"r_half": r, "reliability": 2 * r / (1 + r), "days": int(ok.sum())},
                        index=za.columns)
