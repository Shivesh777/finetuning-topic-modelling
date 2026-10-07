"""Latent Dirichlet allocation, the benchmark topic model.

The topics are estimated on a sample of training-period articles. Two
estimators are available: collapsed Gibbs sampling (tomotopy, the estimator of
the original paper, used when the package is installed) and online
variational Bayes (scikit-learn). Whatever the estimator, article-topic
proportions of ALL articles are then inferred with the topic-word matrix held
fixed, by scikit-learn's variational E-step, which is deterministic and fast.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation


def fit_topic_word(counts: sparse.csr_matrix, n_topics: int, seed: int = 0,
                   backend: str = "auto", iterations: int = 400, passes: int = 8,
                   n_jobs: int = 2) -> np.ndarray:
    """Topic-word probabilities (n_topics, n_terms), rows summing to one."""
    if backend == "auto":
        try:
            import tomotopy  # noqa: F401
            backend = "tomotopy"
        except ImportError:
            backend = "sklearn"
    if backend == "tomotopy":
        return _fit_gibbs(counts, n_topics, seed, iterations, n_jobs)
    model = LatentDirichletAllocation(
        n_components=n_topics, learning_method="online", batch_size=4096, max_iter=passes,
        learning_offset=50.0, evaluate_every=-1, n_jobs=n_jobs, random_state=seed)
    model.fit(counts)
    return model.components_ / model.components_.sum(axis=1, keepdims=True)


def _fit_gibbs(counts, n_topics, seed, iterations, n_jobs) -> np.ndarray:
    import tomotopy as tp

    model = tp.LDAModel(k=n_topics, seed=seed, tw=tp.TermWeight.ONE)
    terms = [str(j) for j in range(counts.shape[1])]
    counts = counts.tocsr()
    for i in range(counts.shape[0]):
        lo, hi = counts.indptr[i], counts.indptr[i + 1]
        words = np.repeat(counts.indices[lo:hi], counts.data[lo:hi])
        if len(words):
            model.add_doc([terms[j] for j in words])
    model.train(iterations, workers=n_jobs)
    phi = np.full((n_topics, counts.shape[1]), 1e-12)
    vocab = np.array([int(w) for w in model.used_vocabs])
    for k in range(n_topics):
        phi[k, vocab] = model.get_topic_word_dist(k)
    return phi / phi.sum(axis=1, keepdims=True)


def infer(counts: sparse.csr_matrix, topic_word: np.ndarray, batch: int = 100_000,
          n_jobs: int = 2) -> np.ndarray:
    """Article-topic proportions (n_articles, n_topics) float32 for a fixed topic-word matrix."""
    n_topics = topic_word.shape[0]
    model = LatentDirichletAllocation(n_components=n_topics, n_jobs=n_jobs, random_state=0)
    # The E-step needs only the (pseudo-)counts of the topic-word matrix.
    model.components_ = topic_word * 1e6
    model.exp_dirichlet_component_ = np.exp(
        _dirichlet_expectation(model.components_))
    model.doc_topic_prior_ = 1.0 / n_topics
    model.n_features_in_ = topic_word.shape[1]
    out = np.empty((counts.shape[0], n_topics), dtype=np.float32)
    for lo in range(0, counts.shape[0], batch):
        out[lo:lo + batch] = model.transform(counts[lo:lo + batch])
    return out


def _dirichlet_expectation(alpha: np.ndarray) -> np.ndarray:
    from scipy.special import psi

    return psi(alpha) - psi(alpha.sum(axis=1, keepdims=True))
