"""Tokenisation and vocabulary shared by all topic models.

The vocabulary (unigrams and bigrams) is built on training-period articles
only and then applied unchanged to later text, so that no word enters because
of what was written during validation or test. LDA uses the counts directly;
the embedding models use them only to name topics and to score coherence.
"""
from __future__ import annotations

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from .. import config

# Words that say nothing about the subject of a Nasdaq.com article.
EXTRA_STOP = {
    "said", "says", "mr", "ms", "inc", "corp", "co", "ltd", "llc", "company", "companies",
    "share", "shares", "stock", "stocks", "percent", "pct", "quarter", "year", "week", "month",
    "reuters", "nasdaq", "nyse", "views", "opinions", "author", "herein", "com", "www", "http",
    "https", "zacks", "motley", "fool", "benzinga", "rttnews", "symbol",
}
STOPWORDS = sorted(ENGLISH_STOP_WORDS | EXTRA_STOP)


def clean(texts: pd.Series) -> pd.Series:
    """Lower-case letters only, addresses removed."""
    texts = texts.fillna("").str.replace(r"https?://\S+", " ", regex=True)
    texts = texts.str.replace(r"[^A-Za-z\s]", " ", regex=True).str.lower()
    return texts.str.replace(r"\s+", " ", regex=True).str.strip()


def _vectoriser(**settings) -> CountVectorizer:
    return CountVectorizer(ngram_range=(1, 2), stop_words=STOPWORDS, token_pattern=r"[a-z]{3,}",
                           dtype="int32", **settings)


def fit_vocabulary(texts: pd.Series, max_features: int = config.VOCAB_SIZE,
                   min_df: int = config.VOCAB_MIN_DF,
                   max_df: float = config.VOCAB_MAX_DF) -> CountVectorizer:
    """Vectoriser fitted on (cleaned) training texts."""
    return _vectoriser(max_features=max_features, min_df=min_df, max_df=max_df).fit(texts)


def with_vocabulary(terms) -> CountVectorizer:
    """Vectoriser for a vocabulary stored with a fitted topic model."""
    return _vectoriser(vocabulary=list(terms))


def terms_of(vectoriser: CountVectorizer):
    return vectoriser.get_feature_names_out()


def count(vectoriser: CountVectorizer, texts: pd.Series) -> sparse.csr_matrix:
    """Article-by-term counts of (cleaned) texts under a fitted vocabulary."""
    return vectoriser.transform(texts).tocsr()
