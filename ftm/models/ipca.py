"""Instrumented principal components (Kelly, Pruitt and Su, 2019) as a return forecaster.

The model of the original project, kept as a baseline:

    r_{i,t+1} = z_{i,t}' Gamma f_{t+1} + e_{i,t+1}

with K latent factors f and loadings that are linear in the stock's features z.
Gamma and the factors are estimated by alternating least squares on per-period
moments (Z_t' Z_t and the managed-portfolio returns Z_t' r_{t+1}), so an
iteration costs the same whatever the number of stocks. The forecast is

    E[r_{i,t+1}] = z_{i,t}' Gamma lambda,     lambda = mean of f over the sample

that is, a linear model whose coefficient vector is restricted to the span of
K factors.

Ipca        the plain estimator; K is the only hyperparameter.
SparseIpca  the estimator of the original project: the K loadings of a feature
            are penalised as a group, so that whole features drop out of the
            model, and the factors are ridge cross-sectional regressions,

                min  sum_t |r_t - Z_t Gamma f_t|^2 + sum_t |f_t|^2
                     + penalty * sum_features |Gamma_feature|

            The original chose the penalty by the Sharpe ratio of a factor
            portfolio on held-out months; here it is chosen like every other
            hyperparameter, by the validation error of the forecasts.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Ipca", "SparseIpca", "period_moments"]


def period_moments(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """(S (T, p, p), q (T, p)): per-period means of z z' and of z r. Rows sorted by group."""
    groups = np.asarray(groups)
    bounds = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True])
    S = np.empty((len(bounds) - 1, X.shape[1], X.shape[1]))
    q = np.empty((len(bounds) - 1, X.shape[1]))
    for t, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        z = np.asarray(X[a:b], dtype=np.float64)
        S[t] = z.T @ z / (b - a)
        q[t] = z.T @ np.asarray(y[a:b], dtype=np.float64) / (b - a)
    return S, q


def _factors(gamma, S, q, ridge):
    """f_t = (Gamma' S_t Gamma)^-1 Gamma' q_t for every period."""
    a = gamma.T @ S @ gamma + ridge * np.eye(gamma.shape[1])
    return np.linalg.solve(a, (q @ gamma)[..., None])[..., 0]


def _loadings(f, S, q, ridge):
    """Gamma that minimises the squared error given the factors."""
    t, p = q.shape
    k = f.shape[1]
    ff = np.einsum("tk,tl->tkl", f, f).reshape(t, k * k)
    h = (S.reshape(t, p * p).T @ ff).reshape(p, p, k, k).transpose(0, 2, 1, 3).reshape(p * k, p * k)
    h[np.diag_indices_from(h)] += ridge * np.trace(h) / (p * k)
    return np.linalg.solve(h, (q.T @ f).reshape(p * k)).reshape(p, k)


def _normalise(gamma, f):
    """The identification of Kelly, Pruitt and Su: Gamma' Gamma = I, orthogonal factors in
    order of decreasing variance, each with a positive mean."""
    r = np.linalg.cholesky(gamma.T @ gamma).T
    u = np.linalg.svd(r @ (f.T @ f) @ r.T)[0]
    gamma, f = gamma @ np.linalg.solve(r, u), f @ r.T @ u
    sign = np.where(f.mean(axis=0) < 0, -1.0, 1.0)
    return gamma * sign, f * sign


class Ipca:
    name = "ipca"
    needs_groups = True              # fit() must be told which rows form a period

    def __init__(self, n_factors: int = 3, ridge: float = 1e-8, max_iter: int = 500,
                 tol: float = 1e-6, **_):
        self.n_factors, self.ridge, self.max_iter, self.tol = n_factors, ridge, max_iter, tol

    @staticmethod
    def grid() -> list[dict]:
        return [{"n_factors": k} for k in (1, 2, 3, 5)]

    def fit(self, X, y, X_val=None, y_val=None, groups=None):
        if groups is None:
            raise ValueError("Ipca.fit needs groups: the period of every row")
        S, q = period_moments(X, y, groups)
        k = min(self.n_factors, q.shape[1], q.shape[0])
        gamma = np.linalg.svd(q, full_matrices=False)[2][:k].T         # PCA of managed portfolios
        f = _factors(gamma, S, q, self.ridge)
        for self.n_iter_ in range(1, self.max_iter + 1):
            new, f = _normalise(_loadings(f, S, q, self.ridge), f)
            f = _factors(new, S, q, self.ridge)
            change, gamma = np.abs(new - gamma).max(), new
            if change < self.tol:
                break
        self.gamma_, self.factors_ = gamma, f
        self.lambda_ = f.mean(axis=0)
        self.coef_ = gamma @ self.lambda_
        self.intercept_ = 0.0
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.coef_

    def refit_params(self) -> dict:
        return {"n_factors": self.n_factors}


class SparseIpca(Ipca):
    name = "sipca"

    def __init__(self, n_factors: int = 3, penalty: float = 0.1, max_iter: int = 100,
                 tol: float = 1e-6, **_):
        """penalty: as a share of the smallest penalty that would switch every feature off
        at the unpenalised fit, so that a value keeps its meaning on another sample."""
        self.n_factors, self.penalty, self.max_iter, self.tol = n_factors, penalty, max_iter, tol

    @staticmethod
    def grid() -> list[dict]:
        return [{"n_factors": k, "penalty": share} for k in (1, 3) for share in (0.3, 0.1, 0.03)]

    @staticmethod
    def _factors(gamma, S, q):
        """Ridge cross-sections: f_t = (Gamma' S_t Gamma + I)^-1 Gamma' q_t."""
        a = gamma.T @ S @ gamma + np.eye(gamma.shape[1])
        return np.linalg.solve(a, (q @ gamma)[..., None])[..., 0]

    @staticmethod
    def _quadratic(f, S, q):
        """(H, c): the squared error is const - 2 c'g + g'H g in g = vec(Gamma)."""
        t, p = q.shape
        k = f.shape[1]
        ff = np.einsum("tk,tl->tkl", f, f).reshape(t, k * k)
        h = (S.reshape(t, p * p).T @ ff).reshape(p, p, k, k).transpose(0, 2, 1, 3)
        return h.reshape(p * k, p * k), q.T @ f

    @staticmethod
    def _loadings(gamma, h, c, weight, steps: int = 200):
        """Minimise g'H g - 2 c'g + weight * sum of row norms: exactly without a penalty,
        otherwise by accelerated proximal gradient (FISTA) starting from gamma."""
        p, k = gamma.shape
        lipschitz = 2.0 * np.linalg.eigvalsh(h)[-1]
        if not lipschitz > 0:
            return np.zeros_like(gamma)
        if weight == 0.0:
            ridge = 1e-10 * lipschitz * np.eye(p * k)
            return np.linalg.solve(h + ridge, c.reshape(-1)).reshape(p, k)
        x, z, momentum = gamma.copy(), gamma.copy(), 1.0
        for _ in range(steps):
            step = z - (2.0 / lipschitz) * ((h @ z.reshape(-1)).reshape(p, k) - c)
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            new = step * np.maximum(0.0, 1.0 - (weight / lipschitz) / np.maximum(norm, 1e-300))
            nxt = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
            z = new + (momentum - 1.0) / nxt * (new - x)
            x, momentum = new, nxt
        return x

    def fit(self, X, y, X_val=None, y_val=None, groups=None):
        if groups is None:
            raise ValueError("SparseIpca.fit needs groups: the period of every row")
        S, q = period_moments(X, y, groups)
        counts = np.diff(np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True]))
        S, q = S * counts[:, None, None], q * counts[:, None]          # sums, as in the original
        k = min(self.n_factors, q.shape[1], q.shape[0])
        gamma = np.linalg.svd(q, full_matrices=False)[2][:k].T
        for _ in range(15):                                             # unpenalised start
            h, c = self._quadratic(self._factors(gamma, S, q), S, q)
            gamma = self._loadings(gamma, h, c, 0.0)
        # The penalty at which the strongest feature would be switched off at this fit.
        h, c = self._quadratic(self._factors(gamma, S, q), S, q)
        p = gamma.shape[0]
        rest = c - (h @ gamma.reshape(-1)).reshape(p, k)
        own = np.einsum("pkpl,pl->pk", h.reshape(p, k, p, k), gamma)
        weight = self.penalty * 2.0 * np.linalg.norm(rest + own, axis=1).max()
        previous = np.inf
        for self.n_iter_ in range(1, self.max_iter + 1):
            f = self._factors(gamma, S, q)
            h, c = self._quadratic(f, S, q)
            gamma = self._loadings(gamma, h, c, weight)
            g = gamma.reshape(-1)
            objective = (g @ h @ g - 2.0 * (c.reshape(-1) @ g)
                         + weight * np.linalg.norm(gamma, axis=1).sum() + (f * f).sum())
            if np.isfinite(previous) and abs(previous - objective) <= self.tol * abs(previous):
                break
            previous = objective
        self.gamma_, self.factors_ = gamma, self._factors(gamma, S, q)
        self.active_ = np.linalg.norm(gamma, axis=1) > 0
        self.lambda_ = self.factors_.mean(axis=0)
        self.coef_ = gamma @ self.lambda_
        self.intercept_ = 0.0
        return self

    def refit_params(self) -> dict:
        return {"n_factors": self.n_factors, "penalty": self.penalty}
