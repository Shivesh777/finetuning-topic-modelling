"""Explainable boosting machine (EBM): a glass-box additive model of returns.

The model of Lou, Caruana, Gehrke and Hooker, used for bond returns in Bell, Kakhbod,
Lettau and Nazemi ("Glass Box Machine Learning and Corporate Bond Returns"):

    y = b + sum_i f_i(x_i) + sum_{i<j} f_ij(x_i, x_j)

Each f is a step function on quantile bins of its feature(s), learned by boosting one term
at a time with very small trees, so the forecast is an exact sum of readable terms. Ebm
wraps interpret.glassbox.ExplainableBoostingRegressor (interpret-core 0.7.8) and keeps its
boosting defaults for everything that is not a hyperparameter below.

Conventions
-----------
* X is a float array (n, p) without NaN, y a float array (n,); predict returns float64.
  All features are declared continuous. (The library's automatic type detection converts
  every value to a string, which costs about ten times the rest of a fit.)
* Terms are ordered main effects first, in column order, then the pairs. Term names are
  the library's: "feature_0003" for column 3 and "feature_0003 & feature_0007" for a pair.
* Outer bags. The model is the average of outer_bags boosted models. The library would
  give each a random split of the rows into training and early-stopping samples; here
  every bag stops early on OUR validation rows instead (the chronological validation
  set). With identical training rows the bags would differ only through the library's
  random numbers (their shape functions correlate above 0.999), so each bag leaves out a
  random share bag_exclude of the training rows, drawn with numpy from seed. On a large
  panel the share rises so that a bag is boosted on at most max_rows training rows: the
  cost of a fit then stops growing with the panel, while the bags together still see
  most of it.
* What validation data may influence: the number of boosting steps of each bag, and
  nothing else. Each bag keeps the model with the lowest validation MSE. Pairs are ranked
  on training rows only and the intercept is recentred so that the residuals of the
  training rows average zero (the library centres on all rows it is given). The bin
  edges are quantiles of the stacked features of both samples: they use validation
  features but no validation returns.
* Rounds. A round is one boosting step per term. The library reports the steps it RAN
  (best_iteration_, which includes the early_stopping_rounds of patience after the best
  model: about twice the best round when returns are mostly noise), so the best step of
  each bag is recorded with a callback instead. best_rounds_ shows the best round of
  every stage and bag; rounds_ is the median over bags of the best main-effect round.
* No refit. The two stages stop at different rounds (a few for the main effects of a
  noisy target, many more for the pairs), and the library takes one max_rounds for
  both. A model refitted on training plus validation rows with a fixed number of rounds
  would therefore not be the model that validation chose. Ebm.refit is False: the
  validated model, fitted on the training rows and stopped on the validation rows, is
  the model that forecasts afterwards (ftm.models.tuning.fit_for_test).
* Without validation data (or with early_stopping_rounds=0) no bag has early-stopping
  rows and exactly max_rounds rounds are run on the training rows of each bag.
* Parallelism. The outer bags run as n_jobs threads of this process (joblib's threading
  backend; the native code releases the GIL), which the callback needs and which gives
  the same numbers as the library's default worker processes. When a callback is set the
  library also starts a small manager process by fork for the duration of the fit;
  Python 3.12 and 3.13 emit a DeprecationWarning about fork in a multi-threaded process.
* Memory: a fit with validation data copies X and X_val into one stacked array.
* seed fixes the bags and the library's random numbers: a fit is reproducible.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from copy import deepcopy
from typing import Sequence

import joblib
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from numpy.typing import ArrayLike


class _BestStep:
    """Boosting callback keeping, per outer bag and stage, the step of the lowest loss.

    The library calls it after every boosting step with the number of steps taken in the
    stage and the validation loss; a stage (main effects, then pairs) begins at step 1.
    Returning False lets boosting continue.
    """

    def __init__(self, n_bags: int) -> None:
        self.stages: list[list[tuple[float, int]]] = [[] for _ in range(n_bags)]

    def __call__(self, bag: int, step: int, progress: bool, loss: float) -> bool:
        if progress:
            stages = self.stages[bag]
            if step == 1:
                stages.append((loss, step))
            elif loss < stages[-1][0]:
                stages[-1] = (loss, step)
        return False


def _bags(n_train: int, n_val: int, n_bags: int, exclude: float, seed: int) -> np.ndarray:
    """Outer-bag roles (n_train + n_val, n_bags) int8: +1 train, -1 early stopping, 0 out.

    Rows are ordered training first. Every validation row is -1 in every bag; in each bag
    int(exclude * n_train) training rows drawn without replacement are 0 (none with a
    single bag, where leaving rows out would only lose data).
    """
    bags = np.ones((n_train + n_val, n_bags), dtype=np.int8, order="F")
    bags[n_train:] = -1
    n_out = int(exclude * n_train) if n_bags > 1 else 0
    rng = np.random.default_rng(seed)
    for bag in range(n_bags):
        bags[rng.choice(n_train, size=n_out, replace=False), bag] = 0
    return bags


def _threads() -> AbstractContextManager:
    """joblib context in which the library runs its outer bags as threads of this process."""
    context = getattr(joblib, "parallel_config", joblib.parallel_backend)
    return context("threading")


class Ebm:
    """Explainable boosting regressor with early stopping on a given validation set.

    interactions is the number of pairs added after the main effects (0 = purely additive
    in the features), chosen by the library's FAST ranking; max_rounds is the maximum
    number of boosting rounds of each stage and smoothing_rounds the number of initial
    rounds with random splits, also of each stage.
    """

    name = "ebm"
    refit = False                       # see "No refit" in the module docstring
    defaults = {"interactions": 0, "max_bins": 64, "max_interaction_bins": 16,
                "learning_rate": 0.02, "max_rounds": 2000, "early_stopping_rounds": 50,
                "outer_bags": 4, "inner_bags": 0, "min_samples_leaf": 20, "max_leaves": 2,
                "smoothing_rounds": 50, "bag_exclude": 0.2, "max_rows": 100_000, "n_jobs": 4,
                "seed": 0}

    def __init__(self, **params) -> None:
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise TypeError(f"Ebm has no hyperparameter {unknown}")
        self.params = {**self.defaults, **params}

    @staticmethod
    def grid() -> list[dict]:
        """Main effects only, then main effects and ten pairs."""
        return [{"interactions": 0}, {"interactions": 10}]

    def fit(self, X: np.ndarray, y: ArrayLike, X_val: np.ndarray | None = None,
            y_val: ArrayLike | None = None) -> Ebm:
        """Boost on (X, y); with validation data each bag stops at its best validation MSE.

        Sets model_ (the fitted library estimator), rounds_ and best_rounds_ (array
        (stages, outer_bags) of best rounds, None without early stopping).
        """
        p = self.params
        y = np.asarray(y, dtype=np.float64)
        n, n_features = X.shape
        patience = int(p["early_stopping_rounds"] or 0)          # None, like 0, switches it off
        stop_early = X_val is not None and patience > 0
        if stop_early:
            y_val = np.asarray(y_val, dtype=np.float64)
            features, target = np.concatenate([X, X_val]), np.concatenate([y, y_val])
        else:
            features, target = X, y
        exclude = max(p["bag_exclude"], 1.0 - p["max_rows"] / n)
        bags = _bags(n, len(target) - n, int(p["outer_bags"]), exclude, p["seed"])
        recorder = _BestStep(int(p["outer_bags"])) if stop_early else None
        model = ExplainableBoostingRegressor(
            feature_types=["continuous"] * n_features, max_bins=int(p["max_bins"]),
            max_interaction_bins=int(p["max_interaction_bins"]),
            interactions=p["interactions"], outer_bags=int(p["outer_bags"]),
            inner_bags=int(p["inner_bags"]), learning_rate=float(p["learning_rate"]),
            smoothing_rounds=int(p["smoothing_rounds"]),
            interaction_smoothing_rounds=int(p["smoothing_rounds"]),
            max_rounds=int(p["max_rounds"]),
            early_stopping_rounds=patience if stop_early else 0,
            callback=recorder, min_samples_leaf=int(p["min_samples_leaf"]),
            max_leaves=int(p["max_leaves"]), n_jobs=p["n_jobs"], random_state=p["seed"])
        with _threads():
            model.fit(features, target, bags=bags)
        model.callback = None
        self.model_ = model
        if stop_early:
            # The library sets the intercept so that residuals average zero over ALL rows it
            # was given, validation rows included: sum_train = -sum_val, so this shift makes
            # the training residuals average zero instead.
            shift = -float(np.sum(y_val - model.predict(X_val))) / n
            model.intercept_ += shift
            model.bagged_intercept_ = model.bagged_intercept_ + shift
            steps = np.array([[step for _, step in stages] for stages in recorder.stages],
                             dtype=np.float64).T
            terms = np.array([n_features, len(model.term_features_) - n_features])
            self.best_rounds_ = steps / terms[: len(steps), None]
            self.rounds_ = max(1, int(round(float(np.median(self.best_rounds_[0])))))
        else:
            self.best_rounds_ = None
            self.rounds_ = int(p["max_rounds"])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """intercept_ plus the sum of the term scores, as float64 (n,)."""
        return np.asarray(self.model_.predict(X), dtype=np.float64)

    def refit_params(self) -> dict:
        """The hyperparameters as they are: the model is fitted the same way again."""
        return dict(self.params)

    @property
    def intercept_(self) -> float:
        """The constant b of the model."""
        return float(self.model_.intercept_)

    def term_names(self) -> list[str]:
        """Names of the terms: main effects in column order, then pairs."""
        return list(self.model_.term_names_)

    def term_features(self) -> list[tuple[int, ...]]:
        """Column indices of the feature(s) of each term, in the order of term_names."""
        return [tuple(int(i) for i in features) for features in self.model_.term_features_]

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Score of every term on every row, (n, n_terms).

        intercept_ + contributions(X).sum(axis=1) equals predict(X).
        """
        return np.asarray(self.model_.eval_terms(X), dtype=np.float64)

    def importances(self, X: np.ndarray) -> np.ndarray:
        """Mean absolute contribution of each term over the rows of X, (n_terms,).

        S(i) = mean_n |f_i(x_n)|, the importance measure of the Glass Box paper, in the
        units of y.
        """
        return np.abs(self.contributions(X)).mean(axis=0)

    def shape(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Shape function of main-effect term i as a step function: (edges, scores).

        scores (k,) holds the score of each bin of non-missing values and edges (k + 1,)
        its bounds: edges[1:-1] are the cut points and edges[0], edges[-1] the smallest and
        largest value of the feature in the fitting data. The term equals scores[j] on
        edges[j] <= x < edges[j + 1], and values beyond the outer edges take the first or
        last score, so matplotlib's stairs(scores, edges) draws it.
        """
        features = self.model_.term_features_[i]
        if len(features) != 1:
            raise ValueError(f"term {i} ({self.model_.term_names_[i]}) is not a main effect")
        low, high = self.model_.feature_bounds_[features[0]]
        edges = np.concatenate([[low], self.model_.bins_[features[0]][0], [high]])
        return edges, np.array(self.model_.term_scores_[i][1:-1], dtype=np.float64)

    def without_terms(self, names: str | int | Sequence[str | int]) -> Ebm:
        """Deep copy of the model with the named terms (names or indices) deleted.

        This is the paper's ablation: the terms are removed from the fitted sum and
        nothing is re-estimated, so predictions change by minus their contributions.
        """
        names = [names] if isinstance(names, (str, int)) else list(names)
        out = deepcopy(self)
        out.model_.remove_terms(names)
        return out
