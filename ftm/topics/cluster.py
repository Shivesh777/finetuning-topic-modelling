"""Spherical k-means on unit-length sentence embeddings, and soft topic weights.

Articles are assigned to the centroid with the highest cosine similarity; a
centroid is the normalised mean of its members. Centroids are estimated on
training-period embeddings and then held fixed.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import kmeans_plusplus


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def centred(embeddings: np.ndarray, centre: np.ndarray, batch: int = 200_000) -> np.ndarray:
    """Unit-length rows of embeddings - centre, as float32 (a copy)."""
    out = np.empty(embeddings.shape, dtype=np.float32)
    for lo in range(0, len(embeddings), batch):
        out[lo:lo + batch] = _unit(np.asarray(embeddings[lo:lo + batch], dtype=np.float32) - centre)
    return out


def assign(embeddings: np.ndarray, centroids: np.ndarray, batch: int = 200_000) -> np.ndarray:
    """Index of the most similar centroid for every row."""
    labels = np.empty(len(embeddings), dtype=np.int32)
    for lo in range(0, len(embeddings), batch):
        block = np.asarray(embeddings[lo:lo + batch], dtype=np.float32)
        labels[lo:lo + batch] = (block @ centroids.T).argmax(axis=1)
    return labels


def spherical_kmeans(embeddings: np.ndarray, k: int, seed: int = 0, n_init: int = 3,
                     max_iter: int = 100, tol: float = 1e-5):
    """(centroids (k, d), mean cosine to the assigned centroid) of the best of n_init runs."""
    x = _unit(np.asarray(embeddings, dtype=np.float32))
    rng = np.random.default_rng(seed)
    best = (None, -np.inf)
    for _ in range(n_init):
        state = int(rng.integers(2**31 - 1))
        sample = x[rng.choice(len(x), size=min(len(x), 50_000), replace=False)]
        centroids = _unit(kmeans_plusplus(sample, k, random_state=state)[0].astype(np.float32))
        previous = -np.inf
        for _ in range(max_iter):
            sims = x @ centroids.T
            labels = sims.argmax(axis=1)
            score = float(sims[np.arange(len(x)), labels].mean())
            sums = np.zeros_like(centroids)
            np.add.at(sums, labels, x)
            empty = np.flatnonzero(np.bincount(labels, minlength=k) == 0)
            if len(empty):                       # reseed on the worst-fitting articles
                worst = np.argsort(sims.max(axis=1))[:len(empty)]
                sums[empty] = x[worst]
            centroids = _unit(sums)
            if score - previous < tol:
                break
            previous = score
        if score > best[1]:
            best = (centroids, score)
    return best


def soft_weights(embeddings: np.ndarray, centroids: np.ndarray, temperature: float,
                 batch: int = 200_000) -> np.ndarray:
    """Article-topic weights (n, k) float32: softmax of cosine similarity / temperature."""
    out = np.empty((len(embeddings), len(centroids)), dtype=np.float32)
    for lo in range(0, len(embeddings), batch):
        z = np.asarray(embeddings[lo:lo + batch], dtype=np.float32) @ centroids.T / temperature
        z -= z.max(axis=1, keepdims=True)
        np.exp(z, out=z)
        out[lo:lo + batch] = z / z.sum(axis=1, keepdims=True)
    return out
