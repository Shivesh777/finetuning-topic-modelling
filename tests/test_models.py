"""Tests of ftm.models (linear, trees, ebm, tuning) on synthetic data.

Every number is checked against an independent calculation: scikit-learn or xgboost fitted
directly on the raw data, an explicit formula or loop, a run of the EBM library itself, or
a simulation with a fixed seed whose true function is known.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn import linear_model as sk
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score

from ftm.models import ebm as ebm_module
from ftm.models import tuning
from ftm.models.ebm import Ebm
from ftm.models.ipca import Ipca
from ftm.models.linear import ElasticNet, Gram, Lasso, Ols, Ridge, simplest_within
from ftm.models.trees import Forest, Xgb
from ftm.models.tuning import (MODELS, chosen_setting, fit_final, fit_for_test, get_model, mse,
                               r2_zero, tune)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def linear_sample(seed: int, n: int, p: int, noise: float = 1.0,
                  draw: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Correlated float32 features in (-0.8, 1.2) and y = 0.3 + X b + noise with sparse b.

    seed fixes the slopes and the correlation of the features, (seed, draw) the rows: two
    draws of one seed are a training and a validation sample of the same model.
    """
    gen = rng(seed)
    mix = np.eye(p) + 0.4 * gen.normal(size=(p, p))
    b = np.where(np.arange(p) % 3 == 0, gen.normal(size=p), 0.0)
    gen = np.random.default_rng([seed, draw])
    X = (np.tanh(gen.normal(size=(n, p)) @ mix) + 0.2).astype(np.float32)
    return X, 0.3 + X.astype(np.float64) @ b + noise * gen.normal(size=n)


def direct_moments(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Sxx, Sxy and syy (divided by n) from np.cov of the raw data."""
    cov = np.cov(np.column_stack([X.astype(np.float64), y]), rowvar=False, bias=True)
    return cov[:-1, :-1], cov[:-1, -1], float(cov[-1, -1])


# --------------------------------------------------------------------------- Gram


def test_gram_matches_direct_moments_and_adds_up():
    X, y = linear_sample(0, 3001, 7)
    Xd = X.astype(np.float64)
    gram = Gram.from_data(X, y, chunk=500)                    # the last chunk has one row
    assert gram.n == 3001
    assert gram.XtX.dtype == gram.Xty.dtype == gram.sum_x.dtype == np.float64
    np.testing.assert_allclose(gram.sum_x, Xd.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(gram.XtX, Xd.T @ Xd, rtol=1e-12)
    np.testing.assert_allclose(gram.Xty, Xd.T @ y, rtol=1e-11, atol=1e-10)
    assert gram.sum_y == pytest.approx(y.sum(), rel=1e-12)
    assert gram.yty == pytest.approx(float(y @ y), rel=1e-12)

    Sxx, Sxy, syy, x_mean, y_mean = gram.centred()
    cov_xx, cov_xy, var_y = direct_moments(X, y)
    np.testing.assert_allclose(Sxx, cov_xx, atol=1e-12)
    np.testing.assert_allclose(Sxy, cov_xy, atol=1e-12)
    assert syy == pytest.approx(var_y, rel=1e-12)
    np.testing.assert_allclose(x_mean, Xd.mean(axis=0), rtol=1e-12)
    assert y_mean == pytest.approx(y.mean(), rel=1e-12)

    both = Gram.from_data(X[:1000], y[:1000]) + Gram.from_data(X[1000:], y[1000:])
    assert both.n == gram.n
    for name in ("sum_x", "XtX", "Xty"):
        np.testing.assert_allclose(getattr(both, name), getattr(gram, name), rtol=1e-11,
                                   atol=1e-10)
    assert both.yty == pytest.approx(gram.yty, rel=1e-12)


def test_gram_rejects_bad_input():
    X, y = linear_sample(1, 50, 3)
    with pytest.raises(ValueError, match="rows"):
        Gram.from_data(X, y[:-1])
    broken = X.copy()
    broken[7, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        Gram.from_data(broken, y)
    with pytest.raises(ValueError, match="NaN"):
        Gram.from_data(X, np.where(np.arange(50) == 3, np.inf, y))
    with pytest.raises(ValueError, match="features"):
        Gram.from_data(X, y) + Gram.from_data(X[:, :2], y)


def test_gram_mse_equals_direct_mse():
    X, y = linear_sample(2, 800, 6)
    Xd = X.astype(np.float64)
    gram = Gram.from_data(X, y, chunk=300)
    gen = rng(3)
    coefs, intercepts = gen.normal(size=(5, 6)), gen.normal(size=5)   # arbitrary, not fitted
    direct = np.array([np.mean((y - a - Xd @ b) ** 2) for b, a in zip(coefs, intercepts)])
    np.testing.assert_allclose(gram.mse(coefs, intercepts), direct, rtol=1e-12)
    single = gram.mse(coefs[2], intercepts[2])
    assert isinstance(single, float) and single == pytest.approx(direct[2], rel=1e-12)
    np.testing.assert_allclose(gram.r2_zero(coefs, intercepts),
                               1 - direct * len(y) / np.sum(y ** 2), rtol=1e-11)
    assert gram.mse(np.zeros(6), 0.0) == pytest.approx(np.mean(y ** 2), rel=1e-12)

    exact = Gram.from_data(X, 2.0 - Xd @ coefs[0])                    # a perfect linear fit
    assert 0.0 <= exact.mse(-coefs[0], 2.0) < 1e-12


# --------------------------------------------------------------------------- linear models


def test_ols_equals_sklearn_and_survives_collinear_columns():
    X, y = linear_sample(4, 1500, 9)
    Xd = X.astype(np.float64)
    ols = Ols().fit(X, y)
    reference = sk.LinearRegression().fit(Xd, y)
    np.testing.assert_allclose(ols.coef_, reference.coef_, atol=1e-9)
    assert ols.intercept_ == pytest.approx(reference.intercept_, abs=1e-9)
    assert Ols.grid() == [{}] and ols.refit_params() == {}

    # A duplicated column and a constant column: the Gram is singular.
    Xc = np.column_stack([X, X[:, 0], np.full(len(X), 0.5, dtype=np.float32)])
    collinear = Ols().fit(Xc, y)
    Xcd = Xc.astype(np.float64)
    minimum_norm = np.linalg.pinv(Xcd - Xcd.mean(axis=0)) @ (y - y.mean())
    np.testing.assert_allclose(collinear.coef_, minimum_norm, atol=1e-7)
    assert collinear.coef_[0] == pytest.approx(collinear.coef_[9], abs=1e-7)   # shared equally
    assert abs(collinear.coef_[10]) < 1e-7
    np.testing.assert_allclose(collinear.predict(Xc), ols.predict(X), atol=1e-7)


def test_ols_drops_directions_that_are_collinear_up_to_float32_rounding():
    """x3 = float32(x1 + x2) is collinear only up to rounding: the fit must not chase it.

    The eigenvalue of that direction is at machine precision; with numpy's default cutoff
    lstsq keeps it in several of these 40 samples and returns slopes of the order of 1e5.
    """
    kept_by_default = 0
    for seed in range(40):
        gen = rng(2000 + seed)
        A = gen.uniform(-1, 1, size=(3000, 2)).astype(np.float32)
        X = np.column_stack([A, A.sum(axis=1, dtype=np.float32)])
        y = A.astype(np.float64) @ gen.normal(size=2) + gen.normal(size=3000)
        model, without = Ols().fit(X, y), Ols().fit(A, y)
        assert np.abs(model.coef_).max() < 5.0
        np.testing.assert_allclose(model.predict(X), without.predict(A), atol=1e-6)
        Sxx, Sxy, *_ = Gram.from_data(X, y).centred()
        kept_by_default += np.abs(np.linalg.lstsq(Sxx, Sxy, rcond=None)[0]).max() > 100
    assert kept_by_default > 0                                # the cutoff is not idle


@pytest.mark.parametrize("alpha", [1e-4, 0.03, 2.0])
def test_ridge_equals_sklearn_with_alpha_times_n(alpha):
    X, y = linear_sample(5, 2000, 10)
    Xd = X.astype(np.float64)
    ridge = Ridge(alpha=alpha).fit(X, y)
    reference = sk.Ridge(alpha=len(y) * alpha, solver="cholesky").fit(Xd, y)
    np.testing.assert_allclose(ridge.coef_, reference.coef_, atol=1e-9)
    assert ridge.intercept_ == pytest.approx(reference.intercept_, abs=1e-9)
    # The closed form on the raw moments.
    Sxx, Sxy, _ = direct_moments(X, y)
    np.testing.assert_allclose(ridge.coef_, np.linalg.solve(Sxx + alpha * np.eye(10), Sxy),
                               atol=1e-10)
    prediction = ridge.predict(X)
    assert prediction.dtype == np.float64 and prediction.shape == (2000,)
    np.testing.assert_allclose(prediction, ridge.intercept_ + Xd @ ridge.coef_, atol=1e-12)


@pytest.mark.parametrize("share", [0.5, 0.1, 0.01])
def test_lasso_and_elastic_net_equal_sklearn_on_raw_data(share):
    X, y = linear_sample(6, 2500, 12)
    Xd = X.astype(np.float64)
    _, Sxy, _ = direct_moments(X, y)
    alpha = share * np.abs(Sxy).max()

    lasso = Lasso(alpha=alpha, tol=1e-12).fit(X, y)
    reference = sk.Lasso(alpha=alpha, tol=1e-13, max_iter=100_000).fit(Xd, y)
    np.testing.assert_allclose(lasso.coef_, reference.coef_, atol=1e-7)
    assert lasso.intercept_ == pytest.approx(reference.intercept_, abs=1e-7)
    assert np.array_equal(lasso.coef_ == 0, reference.coef_ == 0)

    for l1_ratio in (0.5, 0.2):
        net = ElasticNet(alpha=2 * alpha, l1_ratio=l1_ratio, tol=1e-12).fit(X, y)
        reference = sk.ElasticNet(alpha=2 * alpha, l1_ratio=l1_ratio, tol=1e-13,
                                  max_iter=100_000).fit(Xd, y)
        np.testing.assert_allclose(net.coef_, reference.coef_, atol=1e-7)
        assert net.intercept_ == pytest.approx(reference.intercept_, abs=1e-7)

    # The default tolerance is tight enough for the 1e-5 agreement asked of the module.
    np.testing.assert_allclose(Lasso(alpha=alpha).fit(X, y).coef_, reference_lasso(Xd, y, alpha),
                               atol=1e-5)


def reference_lasso(Xd: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return sk.Lasso(alpha=alpha, tol=1e-13, max_iter=100_000).fit(Xd, y).coef_


def test_penalties_are_per_observation():
    """Stacking the sample on itself must not change a per-observation penalised fit."""
    X, y = linear_sample(7, 600, 8)
    X2, y2 = np.vstack([X, X]), np.concatenate([y, y])
    gram = Gram.from_data(X, y)
    for model in (Ridge(alpha=0.05), Lasso(alpha=0.02, tol=1e-12),
                  ElasticNet(alpha=0.03, tol=1e-12), Ols()):
        once = type(model)(**model.params).fit(X, y)
        twice = type(model)(**model.params).fit(X2, y2)
        from_grams = type(model)(**model.params).fit_gram(gram + gram)
        np.testing.assert_allclose(twice.coef_, once.coef_, atol=1e-9)
        np.testing.assert_allclose(from_grams.coef_, once.coef_, atol=1e-9)
        assert from_grams.intercept_ == pytest.approx(once.intercept_, abs=1e-9)


def test_linear_models_ignore_validation_data_and_reject_unknown_parameters():
    X, y = linear_sample(8, 500, 5)
    Xv, yv = linear_sample(8, 200, 5, draw=1)
    for cls in (Ols, Ridge, Lasso, ElasticNet):
        plain, with_validation = cls().fit(X, y), cls().fit(X, y, Xv, 100 * yv)
        assert np.array_equal(plain.coef_, with_validation.coef_)
        assert plain.intercept_ == with_validation.intercept_
        assert plain.refit_params() == plain.params == cls.defaults
        with pytest.raises(TypeError, match="no hyperparameter"):
            cls(depth=3)
    assert Ridge.grid() == Lasso.grid() == ElasticNet.grid() == []


def test_alpha_paths_follow_the_documented_grids():
    X, y = linear_sample(10, 900, 14)
    Xv, yv = linear_sample(10, 400, 14, draw=1)
    Sxx, Sxy, _ = direct_moments(X, y)

    _, ridge, _ = Ridge.tune_path(X, y, Xv, yv)
    assert len(ridge) == 25 and list(ridge.columns) == [
        "alpha", "n_nonzero", "val_mse", "val_r2", "train_r2", "seconds", "chosen"]
    np.testing.assert_allclose(ridge["alpha"], np.trace(Sxx) / 14 * np.logspace(2, -4, 25),
                               rtol=1e-10)
    assert (ridge["n_nonzero"] == 14).all()
    assert ridge["train_r2"].is_monotonic_increasing        # less penalty, better in-sample fit

    lasso_best, lasso, _ = Lasso.tune_path(X, y, Xv, yv)
    alpha_max = np.abs(Sxy).max()
    assert len(lasso) == 40 and "l1_ratio" not in lasso.columns
    np.testing.assert_allclose(lasso["alpha"], alpha_max * np.logspace(0, -3, 40), rtol=1e-10)
    assert lasso["n_nonzero"].iloc[0] == 0 and lasso["n_nonzero"].iloc[1] >= 1
    # alpha_max is the smallest alpha at which scikit-learn's lasso on the raw data is null.
    Xd = X.astype(np.float64)
    assert not reference_lasso(Xd, y, 1.001 * alpha_max).any()
    assert reference_lasso(Xd, y, 0.999 * alpha_max).any()
    # The null model forecasts the training mean.
    assert lasso["val_mse"].iloc[0] == pytest.approx(np.mean((yv - y.mean()) ** 2), rel=1e-10)

    best, net, _ = ElasticNet.tune_path(X, y, Xv, yv)
    assert len(net) == 40 and (net["l1_ratio"] == 0.5).all() and best["l1_ratio"] == 0.5
    np.testing.assert_allclose(net["alpha"], 2 * alpha_max * np.logspace(0, -3, 40), rtol=1e-10)
    assert net["n_nonzero"].iloc[0] == 0 and net["n_nonzero"].iloc[1] >= 1
    for table in (ridge, lasso, net):
        assert table["chosen"].sum() == 1 and (table["seconds"] >= 0).all()

    # tune_gram is the same tuning on statistics that the caller already holds.
    gram_best, gram_table = Lasso.tune_gram(Gram.from_data(X, y), Gram.from_data(Xv, yv))
    assert gram_best == lasso_best
    pd.testing.assert_frame_equal(gram_table.drop(columns="seconds"),
                                  lasso.drop(columns="seconds"))


@pytest.mark.parametrize("cls", [Ridge, Lasso, ElasticNet])
def test_tune_path_picks_the_alpha_with_the_lowest_direct_validation_mse(cls):
    X, y = linear_sample(12, 260, 30, noise=2.0)              # few rows: overfitting matters
    Xv, yv = linear_sample(12, 3000, 30, noise=2.0, draw=1)
    Xd, Xvd = X.astype(np.float64), Xv.astype(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        best, table, val_pred = tune(cls, X, y, Xv, yv)       # delegates to cls.tune_path

    def reference(alpha: float):
        if cls is Ridge:
            return sk.Ridge(alpha=len(y) * alpha, solver="cholesky").fit(Xd, y)
        l1_ratio = 1.0 if cls is Lasso else 0.5
        return sk.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-13,
                             max_iter=200_000).fit(Xd, y)

    models = [reference(alpha) for alpha in table["alpha"]]
    direct = np.array([mean_squared_error(yv, model.predict(Xvd)) for model in models])
    np.testing.assert_allclose(table["val_mse"], direct, rtol=1e-6)
    np.testing.assert_allclose(table["val_r2"], 1 - direct * len(yv) / np.sum(yv ** 2),
                               atol=1e-6)
    train_r2 = [1 - np.sum((y - model.predict(Xd)) ** 2) / np.sum(y ** 2) for model in models]
    np.testing.assert_allclose(table["train_r2"], train_r2, atol=1e-6)
    if cls is not Ridge:
        # The first alpha is the smallest that sets every slope to zero: a reference that
        # computes X'y in another order may leave a slope of 1e-17 there.
        assert list(table["n_nonzero"]) == [int(np.sum(np.abs(model.coef_) > 1e-12))
                                            for model in models]

    chosen = int(np.argmin(direct))
    assert 0 < chosen < len(table) - 1                        # an interior optimum
    assert list(np.flatnonzero(table["chosen"])) == [chosen]
    assert best["alpha"] == pytest.approx(table["alpha"].iloc[chosen], rel=1e-12)

    # val_pred: the forecasts of the validation rows at the chosen alpha, from training data.
    assert val_pred.dtype == np.float64 and val_pred.shape == yv.shape
    np.testing.assert_allclose(val_pred, models[chosen].predict(Xvd), atol=1e-5)
    assert mse(yv, val_pred) == pytest.approx(table["val_mse"].iloc[chosen], rel=1e-10)
    np.testing.assert_allclose(val_pred, fit_final(cls, best, X, y).predict(Xv), atol=1e-6)

    # best_params are ready for fit_final: the fit on all rows is scikit-learn's at that alpha.
    X_all, y_all = np.vstack([X, Xv]), np.concatenate([y, yv])
    final = fit_final(cls, best, X_all, y_all)
    if cls is Ridge:
        expected = sk.Ridge(alpha=len(y_all) * best["alpha"], solver="cholesky")
    else:
        expected = sk.ElasticNet(alpha=best["alpha"], l1_ratio=1.0 if cls is Lasso else 0.5,
                                 tol=1e-13, max_iter=200_000)
    expected.fit(X_all.astype(np.float64), y_all)
    np.testing.assert_allclose(final.coef_, expected.coef_, atol=1e-5)
    assert final.intercept_ == pytest.approx(expected.intercept_, abs=1e-5)


def test_path_tolerance_keeps_the_most_regularised_setting_that_is_close():
    X, y = linear_sample(15, 300, 20, noise=2.0)
    Xv, yv = linear_sample(15, 2000, 20, noise=2.0, draw=1)
    _, exact, exact_pred = tune(Lasso, X, y, Xv, yv)
    best, table, val_pred = tune(Lasso, X, y, Xv, yv, tolerance=0.02)
    np.testing.assert_allclose(table["val_mse"], exact["val_mse"], rtol=1e-12)
    lowest = table["val_mse"].min()
    first_close = next(i for i, value in enumerate(table["val_mse"]) if value <= 1.02 * lowest)
    assert list(np.flatnonzero(table["chosen"])) == [first_close]
    assert first_close < int(np.flatnonzero(exact["chosen"])[0])
    assert best["alpha"] == pytest.approx(table["alpha"].iloc[first_close])
    # The forecasts returned are those of the setting chosen, not of the lowest MSE.
    assert mse(yv, val_pred) == pytest.approx(table["val_mse"].iloc[first_close], rel=1e-10)
    assert mse(yv, exact_pred) == pytest.approx(lowest, rel=1e-10)
    assert table["n_nonzero"].iloc[first_close] < exact.loc[exact["chosen"], "n_nonzero"].iloc[0]


# --------------------------------------------------------------------------- trees


def tree_sample(seed: int, n: int, p: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """y = 1.5 x0 + 2 x1 x2 + noise on uniform features: columns 3+ are irrelevant."""
    gen = rng(seed)
    X = gen.uniform(-1, 1, size=(n, p)).astype(np.float32)
    return X, 1.5 * X[:, 0] + 2.0 * X[:, 1] * X[:, 2] + gen.normal(size=n)


FAST_XGB = {"learning_rate": 0.3, "n_estimators": 300, "early_stopping_rounds": 10,
            "min_child_weight": 20, "n_jobs": 2}


def test_xgb_stops_early_on_the_validation_set_and_refits_to_the_same_model():
    X, y = tree_sample(20, 6000)
    Xv, yv = tree_sample(21, 2000)
    model = Xgb(max_depth=2, **FAST_XGB).fit(X, y, Xv, yv)
    booster = model.model_.get_booster()

    # The round kept is the minimum of the validation RMSE curve that xgboost recorded.
    curve = np.array(model.model_.evals_result()["validation_0"]["rmse"])
    assert model.rounds_ == int(np.argmin(curve)) + 1 == model.model_.best_iteration + 1
    assert model.rounds_ < FAST_XGB["n_estimators"]
    assert booster.num_boosted_rounds() == model.rounds_ + FAST_XGB["early_stopping_rounds"]
    # predict uses the trees up to that round only.
    prediction = model.predict(Xv)
    assert prediction.dtype == np.float64 and prediction.shape == (2000,)
    assert np.sqrt(mse(yv, prediction)) == pytest.approx(curve.min(), rel=1e-5)
    assert np.sqrt(mse(yv, booster.inplace_predict(Xv))) > curve.min()     # all trees: worse

    refit = model.refit_params()
    assert refit["n_estimators"] == model.rounds_ and refit["early_stopping_rounds"] is None
    assert {k: v for k, v in refit.items() if k not in ("n_estimators", "early_stopping_rounds")
            } == {k: v for k, v in model.params.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}
    again = fit_final(Xgb, refit, X, y)
    assert again.model_.get_booster().num_boosted_rounds() == model.rounds_ == again.rounds_
    np.testing.assert_allclose(again.predict(Xv), prediction, atol=1e-6)
    assert again.refit_params() == refit


def test_xgb_without_validation_trains_exactly_n_estimators():
    X, y = tree_sample(22, 1500)
    Xv, yv = tree_sample(23, 500)
    plain = Xgb(n_estimators=17, n_jobs=2).fit(X, y)
    assert plain.rounds_ == 17 and plain.model_.get_booster().num_boosted_rounds() == 17
    # Validation data without early stopping change nothing.
    ignored = Xgb(n_estimators=17, early_stopping_rounds=None, n_jobs=2).fit(X, y, Xv, yv)
    assert ignored.rounds_ == 17
    np.testing.assert_array_equal(ignored.predict(Xv), plain.predict(Xv))
    assert Xgb().params["seed"] == 0 and Xgb().params["n_jobs"] == 4
    with pytest.raises(TypeError, match="no hyperparameter"):
        Xgb(depth=3)


def test_xgb_feature_importance_is_the_share_of_total_gain():
    X, y = tree_sample(24, 6000)
    Xv, yv = tree_sample(25, 2000)
    model = Xgb(max_depth=3, **FAST_XGB).fit(X, y, Xv, yv)
    importance = model.feature_importance()
    assert importance.shape == (6,) and importance.sum() == pytest.approx(1.0)

    trees = model.model_.get_booster().trees_to_dataframe()
    splits = trees[(trees["Feature"] != "Leaf") & (trees["Tree"] < model.rounds_)]
    gain = splits.groupby("Feature")["Gain"].sum()
    expected = np.array([gain.get(f"f{j}", 0.0) for j in range(6)])
    np.testing.assert_allclose(importance, expected / expected.sum(), rtol=1e-5)
    assert importance[:3].sum() > 0.9 and importance[:3].min() > 0.15

    stump = Xgb(n_estimators=3, min_child_weight=1e9, n_jobs=2).fit(X, y)     # no split allowed
    assert np.array_equal(stump.feature_importance(), np.zeros(6))


def test_forest_caps_the_rows_of_each_tree_and_matches_sklearn():
    X, y = tree_sample(26, 1200, p=5)
    Xv, yv = tree_sample(27, 800, p=5)
    settings = {"n_estimators": 20, "max_depth": 4, "max_features": 0.6, "min_samples_leaf": 5,
                "n_jobs": 2, "seed": 3}
    capped = Forest(max_rows=500, **settings).fit(X, y, Xv, yv)
    assert capped.model_.max_samples == pytest.approx(500 / 1200)
    # The root of every tree holds the bootstrap sample: 500 draws of the 1200 rows.
    assert {tree.tree_.weighted_n_node_samples[0] for tree in capped.model_.estimators_} == {500}
    assert max(tree.get_depth() for tree in capped.model_.estimators_) <= 4
    assert min(tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1].min()
               for tree in capped.model_.estimators_) >= 5

    reference = RandomForestRegressor(n_estimators=20, max_depth=4, max_features=0.6,
                                      min_samples_leaf=5, max_samples=500, random_state=3)
    prediction = capped.predict(Xv)
    assert prediction.dtype == np.float64
    np.testing.assert_allclose(prediction, reference.fit(X, y).predict(Xv), atol=1e-12)
    assert r2_zero(yv, prediction) > 0.3

    whole = Forest(**settings).fit(X, y)                       # n below the default cap
    assert whole.model_.max_samples == 1.0
    assert {tree.tree_.weighted_n_node_samples[0] for tree in whole.model_.estimators_} == {1200}
    assert whole.refit_params() == whole.params
    assert Forest().params["max_rows"] == 30_000 and Forest().params["n_estimators"] == 100
    assert Forest().params["min_samples_leaf"] == 50


def test_grids_run_from_the_most_regularised_setting():
    assert Xgb.grid() == [{"max_depth": 1}, {"max_depth": 2}, {"max_depth": 3}, {"max_depth": 5}]
    assert Forest.grid() == [{"max_depth": 3}, {"max_depth": 6}, {"max_depth": 10}]
    assert Ebm.grid() == [{"interactions": 0}, {"interactions": 10}]


# --------------------------------------------------------------------------- EBM


def glass_function(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    return np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 2 * X[:, 2] * X[:, 3]


@pytest.fixture(scope="module")
def glass():
    """20,000 rows (15,000 training, 5,000 validation) and 8 features, plus a blind sample."""
    gen = rng(30)
    X = gen.uniform(-1, 1, size=(25_000, 8)).astype(np.float32)
    y = glass_function(X) + 0.5 * gen.normal(size=len(X))
    return {"X": X[:15_000], "y": y[:15_000], "Xv": X[15_000:20_000], "yv": y[15_000:20_000],
            "Xb": X[20_000:], "yb": y[20_000:]}


@pytest.fixture(scope="module")
def ebm_main(glass):
    """Default settings but for a faster learning rate (fewer rounds, to keep the file quick)."""
    return Ebm(interactions=0, learning_rate=0.05).fit(glass["X"], glass["y"], glass["Xv"],
                                                       glass["yv"])


@pytest.fixture(scope="module")
def ebm_pairs(glass):
    return Ebm(interactions=10, learning_rate=0.05).fit(glass["X"], glass["y"], glass["Xv"],
                                                        glass["yv"])


def test_ebm_with_pairs_beats_main_effects_which_beat_ridge(glass, ebm_main, ebm_pairs):
    best, _, ridge_val = tune(Ridge, glass["X"], glass["y"], glass["Xv"], glass["yv"])
    ridge = fit_final(Ridge, best, glass["X"], glass["y"])
    np.testing.assert_allclose(ridge_val, ridge.predict(glass["Xv"]), atol=1e-9)
    for X, y in ((glass["Xv"], glass["yv"]), (glass["Xb"], glass["yb"])):
        r2_ridge = r2_zero(y, ridge.predict(X))
        r2_main = r2_zero(y, ebm_main.predict(X))
        r2_pairs = r2_zero(y, ebm_pairs.predict(X))
        assert r2_pairs > r2_main + 0.2 and r2_main > r2_ridge + 0.1
        # With the pair x2 * x3 the model is close to the true function.
        assert r2_pairs > r2_zero(y, glass_function(X)) - 0.02
    assert (2, 3) in ebm_pairs.term_features()
    assert ebm_main.params["outer_bags"] == 4 and ebm_main.params["seed"] == 0


def test_ebm_contributions_sum_to_predictions(glass, ebm_main, ebm_pairs):
    for model, n_terms in ((ebm_main, 8), (ebm_pairs, 18)):
        contributions = model.contributions(glass["Xb"])
        prediction = model.predict(glass["Xb"])
        assert contributions.shape == (5000, n_terms) and prediction.dtype == np.float64
        np.testing.assert_allclose(model.intercept_ + contributions.sum(axis=1), prediction,
                                   atol=1e-10)
        np.testing.assert_allclose(model.importances(glass["Xb"]),
                                   np.abs(contributions).mean(axis=0), atol=1e-14)
        names, features = model.term_names(), model.term_features()
        assert names[:8] == [f"feature_{j:04d}" for j in range(8)]
        assert features[:8] == [(j,) for j in range(8)]
        assert all(len(pair) == 2 and names[8 + k] == " & ".join(f"feature_{j:04d}" for j in pair)
                   for k, pair in enumerate(features[8:]))
    # The three largest terms are the three true components.
    importances = pd.Series(ebm_pairs.importances(glass["Xb"]), index=ebm_pairs.term_names())
    assert set(importances.nlargest(3).index) == {"feature_0000", "feature_0001",
                                                  "feature_0002 & feature_0003"}


def test_ebm_shape_is_the_step_function_behind_a_main_effect(glass, ebm_pairs):
    contributions = ebm_pairs.contributions(glass["Xb"])
    stacked = np.vstack([glass["X"], glass["Xv"]])
    for term, truth in ((0, lambda x: np.sin(3 * x)), (1, lambda x: x ** 2)):
        edges, scores = ebm_pairs.shape(term)
        assert len(edges) == len(scores) + 1 == 63            # max_bins 64 = 62 bins + 2 special
        assert (np.diff(edges) > 0).all()
        assert edges[0] == stacked[:, term].min() and edges[-1] == stacked[:, term].max()
        x = glass["Xb"][:, term].astype(np.float64)
        steps = scores[np.searchsorted(edges[1:-1], x, side="right")]
        np.testing.assert_allclose(steps, contributions[:, term], atol=1e-12)
        # The shape is the true component up to its level, which sits in the intercept: a
        # bin holds about 200 rows per bag with noise 0.5, so its score is within 0.1 or so.
        target = truth((edges[:-1] + edges[1:]) / 2)
        error = (scores - scores.mean()) - (target - target.mean())
        assert np.sqrt(np.mean(error ** 2)) < 0.06 and np.abs(error).max() < 0.2
        assert np.corrcoef(scores, target)[0, 1] > 0.99
    with pytest.raises(ValueError, match="not a main effect"):
        ebm_pairs.shape(8)


def test_ebm_without_terms_removes_exactly_those_contributions(glass, ebm_pairs):
    X = glass["Xb"]
    names = ebm_pairs.term_names()
    contributions = ebm_pairs.contributions(X)
    before = ebm_pairs.predict(X)

    k = names.index("feature_0002 & feature_0003")
    ablated = ebm_pairs.without_terms(names[k])
    np.testing.assert_allclose(ablated.predict(X) - before, -contributions[:, k], atol=1e-10)
    assert ablated.term_names() == names[:k] + names[k + 1:]
    np.testing.assert_allclose(ablated.contributions(X), np.delete(contributions, k, axis=1),
                               atol=1e-12)
    assert ablated.intercept_ == ebm_pairs.intercept_          # nothing is re-estimated

    several = ebm_pairs.without_terms([names[0], 1, names[k]])   # names and an index
    np.testing.assert_allclose(several.predict(X) - before,
                               -contributions[:, [0, 1, k]].sum(axis=1), atol=1e-10)
    assert len(several.term_names()) == len(names) - 3

    # The original model is untouched.
    assert ebm_pairs.term_names() == names
    np.testing.assert_array_equal(ebm_pairs.predict(X), before)


def test_ebm_stops_early_on_our_validation_rows_only(glass, ebm_main, ebm_pairs):
    for model, n_stages in ((ebm_main, 1), (ebm_pairs, 2)):
        library = model.model_
        executed = library.best_iteration_ / np.array([8, 10])[:n_stages, None]   # rounds run
        assert model.best_rounds_.shape == (n_stages, 4)
        assert (model.best_rounds_ > 0).all() and (model.best_rounds_ <= executed).all()
        assert (executed < model.params["max_rounds"]).all()  # every bag stopped early
        assert model.rounds_ == round(float(np.median(model.best_rounds_[0])))
        assert model.rounds_ < model.params["max_rounds"]
        # Every bag trained on 80 percent of the training rows and on no validation row.
        np.testing.assert_array_equal(library.bag_weights_, np.full(4, 12_000.0))
        # The intercept is centred on the training rows, not on the stacked sample.
        assert abs(np.mean(glass["y"] - model.predict(glass["X"]))) < 1e-10
        assert abs(np.mean(glass["yv"] - model.predict(glass["Xv"]))) > 1e-6
    np.testing.assert_array_equal(ebm_pairs.best_rounds_[0], ebm_main.best_rounds_[0])


def test_ebm_bags_keep_validation_rows_for_early_stopping():
    bags = ebm_module._bags(1000, 300, 4, 0.2, seed=5)
    assert bags.shape == (1300, 4) and bags.dtype == np.int8
    assert (bags[1000:] == -1).all()
    assert set(np.unique(bags[:1000])) == {0, 1}
    assert ((bags[:1000] == 0).sum(axis=0) == 200).all()
    assert len({bags[:1000, b].tobytes() for b in range(4)}) == 4          # the bags differ
    np.testing.assert_array_equal(bags, ebm_module._bags(1000, 300, 4, 0.2, seed=5))
    assert not np.array_equal(bags, ebm_module._bags(1000, 300, 4, 0.2, seed=6))
    assert (ebm_module._bags(1000, 0, 1, 0.2, seed=5) == 1).all()            # one bag: all rows
    assert (ebm_module._bags(1000, 0, 3, 0.0, seed=5) == 1).all()


FAST_EBM = {"learning_rate": 0.2, "max_rounds": 400, "early_stopping_rounds": 10,
            "smoothing_rounds": 5, "n_jobs": 2}


def library_ebm(n_features: int, outer_bags: int, interactions: int, **changes):
    """The library's regressor configured as Ebm configures it for FAST_EBM, seed 0."""
    settings = dict(
        feature_types=["continuous"] * n_features, max_bins=64, max_interaction_bins=16,
        interactions=interactions, outer_bags=outer_bags, inner_bags=0, learning_rate=0.2,
        smoothing_rounds=5, interaction_smoothing_rounds=5, max_rounds=400,
        early_stopping_rounds=10, min_samples_leaf=20, max_leaves=2, n_jobs=1, random_state=0)
    return ExplainableBoostingRegressor(**{**settings, **changes})


def test_ebm_best_rounds_are_the_minimum_of_the_validation_loss_path():
    """Rerun the library with a callback that stores every loss and take the argmin."""
    gen = rng(31)
    X = gen.uniform(-1, 1, size=(5000, 3)).astype(np.float32)
    y = np.sin(3 * X[:, 0]) + X[:, 1] * X[:, 2] + gen.normal(size=5000)
    model = Ebm(interactions=2, outer_bags=2, **FAST_EBM).fit(X[:3500], y[:3500], X[3500:],
                                                               y[3500:])
    path: dict[int, list[tuple[int, float]]] = {0: [], 1: []}

    def record(bag: int, step: int, progress: bool, loss: float) -> bool:
        if progress:
            path[bag].append((step, loss))
        return False

    bags = ebm_module._bags(3500, 1500, 2, 0.2, seed=0)
    library_ebm(3, 2, 2, callback=record).fit(X, y, bags=bags)
    for bag in (0, 1):
        steps, losses = np.array(path[bag]).T
        start = int(np.flatnonzero(steps == 1)[1])            # the pair stage restarts at 1
        best_main = steps[:start][np.argmin(losses[:start])]
        best_pair = steps[start:][np.argmin(losses[start:])]
        assert model.best_rounds_[0, bag] == pytest.approx(best_main / 3)
        assert model.best_rounds_[1, bag] == pytest.approx(best_pair / 2)
        assert best_main < steps[:start].max()                # the library ran past the best


def test_ebm_validation_rows_only_choose_the_number_of_steps():
    """The early-stopped model is the model boosted for the kept steps without those rows."""
    gen = rng(32)
    X = gen.uniform(-1, 1, size=(6000, 1)).astype(np.float32)
    y = np.sin(3 * X[:, 0]) + 2.0 * gen.normal(size=6000)
    model = Ebm(outer_bags=1, **FAST_EBM).fit(X[:4000], y[:4000], X[4000:], y[4000:])
    rounds = model.best_rounds_[0, 0]                         # one term: a step is a round
    assert rounds == model.rounds_ and 1 < rounds < 400

    bags = np.ones((6000, 1), dtype=np.int8)
    bags[4000:] = 0                                           # validation rows left out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = library_ebm(1, 1, 0, max_rounds=int(rounds), early_stopping_rounds=0)
        direct.fit(X, y, bags=bags)
    np.testing.assert_allclose(model.model_.term_scores_[0], direct.term_scores_[0], atol=1e-12)
    # The library centres its intercept on all 6000 rows, Ebm on the 4000 training rows.
    residual = y[:4000] - direct.predict(X[:4000])
    assert model.intercept_ == pytest.approx(direct.intercept_ + residual.mean(), abs=1e-10)


def test_ebm_is_not_refitted_the_validated_model_forecasts(glass, ebm_pairs):
    assert Ebm.refit is False and ebm_pairs.refit_params() == ebm_pairs.params
    # The two stages stop at different rounds, which one max_rounds could not reproduce.
    main_rounds, pair_rounds = np.median(ebm_pairs.best_rounds_, axis=1)
    assert abs(main_rounds - pair_rounds) > 5

    X = np.vstack([glass["X"], glass["Xv"]])
    y = np.concatenate([glass["y"], glass["yv"]])
    train = np.arange(len(y)) < len(glass["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        again = fit_for_test(Ebm, ebm_pairs.refit_params(), X, y, train, ~train)
    np.testing.assert_array_equal(again.best_rounds_, ebm_pairs.best_rounds_)
    np.testing.assert_array_equal(again.predict(glass["Xb"]), ebm_pairs.predict(glass["Xb"]))
    np.testing.assert_array_equal(again.model_.bag_weights_, np.full(4, 12_000.0))   # no validation row


def test_fit_for_test_refits_the_other_models_on_training_plus_validation():
    X, y = linear_sample(34, 1500, 6, noise=1.0)
    train = np.arange(1500) < 1000
    ridge = fit_for_test(Ridge, {"alpha": 0.1}, X, y, train, ~train)
    np.testing.assert_array_equal(ridge.coef_, fit_final(Ridge, {"alpha": 0.1}, X, y).coef_)
    assert not np.allclose(ridge.coef_, fit_final(Ridge, {"alpha": 0.1}, X[train], y[train]).coef_)

    tuned = Xgb(**FAST_XGB).fit(X[train], y[train], X[~train], y[~train])
    final = fit_for_test(Xgb, tuned.refit_params(), X, y, train, ~train)
    assert final.model_.get_booster().num_boosted_rounds() == tuned.rounds_
    np.testing.assert_array_equal(final.predict(X), fit_final(Xgb, tuned.refit_params(), X, y).predict(X))

    periods = np.repeat(np.arange(30), 50)
    ipca = fit_for_test(Ipca, {"n_factors": 2}, X, y, train, ~train, groups=periods)
    np.testing.assert_allclose(ipca.predict(X), fit_final(Ipca, {"n_factors": 2}, X, y, periods).predict(X))


def test_ebm_is_reproducible_and_ignores_validation_without_early_stopping():
    gen = rng(33)
    X = gen.uniform(-1, 1, size=(3000, 3)).astype(np.float32)
    y = X[:, 0] ** 2 + gen.normal(size=3000)
    settings = {"max_rounds": 30, "outer_bags": 2, "learning_rate": 0.2, "n_jobs": 2}
    first = Ebm(**settings).fit(X[:2000], y[:2000])
    np.testing.assert_array_equal(first.model_.best_iteration_, np.full((1, 2), 90))
    assert first.rounds_ == 30 and first.best_rounds_ is None
    again = Ebm(**settings).fit(X[:2000], y[:2000])
    np.testing.assert_array_equal(again.predict(X), first.predict(X))
    serial = Ebm(**{**settings, "n_jobs": 1}).fit(X[:2000], y[:2000])       # no threads
    np.testing.assert_array_equal(serial.predict(X), first.predict(X))
    other = Ebm(seed=1, **settings).fit(X[:2000], y[:2000])
    assert not np.array_equal(other.predict(X), first.predict(X))
    for off in (0, None):                                     # both switch early stopping off
        ignored = Ebm(early_stopping_rounds=off, **settings).fit(X[:2000], y[:2000], X[2000:],
                                                                 100 * y[2000:])
        np.testing.assert_array_equal(ignored.predict(X), first.predict(X))
        assert ignored.rounds_ == 30 and ignored.best_rounds_ is None
    with pytest.raises(TypeError, match="no hyperparameter"):
        Ebm(depth=3)


# --------------------------------------------------------------------------- tuning


def test_mse_and_r2_zero_formulas():
    gen = rng(40)
    y, pred = gen.normal(size=500), gen.normal(size=500)
    assert mse(y, pred) == pytest.approx(mean_squared_error(y, pred), rel=1e-12)
    assert r2_zero(y, pred) == pytest.approx(1 - np.sum((y - pred) ** 2) / np.sum(y ** 2))
    centred = y - y.mean()                                    # then zero is the mean forecast
    assert r2_zero(centred, pred) == pytest.approx(r2_score(centred, pred), rel=1e-10)
    assert r2_zero(y, np.zeros(500)) == 0.0 and r2_zero(y, y) == 1.0
    assert mse(y.astype(np.float32), pred.astype(np.float32)) == pytest.approx(mse(y, pred),
                                                                               rel=1e-5)
    assert np.isnan(r2_zero(np.zeros(4), np.ones(4)))


def test_simplest_within_tolerance_rule():
    losses = [1.10, 1.04, 1.00, 1.02, 1.30]
    assert simplest_within(losses) == 2
    assert simplest_within(losses, 0.03) == 2
    assert simplest_within(losses, 0.05) == 1
    assert simplest_within(losses, 0.11) == 0
    assert simplest_within([2.0, 1.0, 1.0, 3.0]) == 1          # ties: the first, simpler one
    assert simplest_within([np.nan, 5.0, 4.0], 0.5) == 1       # NaN is never chosen
    with pytest.raises(ValueError):
        simplest_within(losses, -0.1)
    with pytest.raises(ValueError):
        simplest_within([np.nan, np.nan])


class Toy:
    """Forecasts a constant: against y_val = 0 its validation MSE is level squared."""

    name = "toy"
    levels = (1.10, 1.04, 1.00, 1.02, 1.30)                    # validation MSEs, in grid order

    def __init__(self, level: float = 0.0) -> None:
        self.params = {"level": level}

    @staticmethod
    def grid() -> list[dict]:
        return [{"level": float(np.sqrt(value))} for value in Toy.levels]

    def fit(self, X, y, X_val=None, y_val=None) -> Toy:
        self.saw_validation = X_val is not None and y_val is not None
        self.rounds_ = int(round(100 * self.params["level"] ** 2))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.params["level"])

    def refit_params(self) -> dict:
        return {**self.params, "rounds": self.rounds_, "validated": self.saw_validation}


@pytest.mark.parametrize("tolerance, expected", [(0.0, 2), (0.03, 2), (0.05, 1), (0.11, 0)])
def test_tune_tolerance_rule_on_a_toy_model(tolerance, expected):
    X, y = np.zeros((50, 2), dtype=np.float32), np.full(50, 2.0)
    X_val, y_val = np.zeros((20, 2), dtype=np.float32), np.zeros(20)
    best, table, val_pred = tune(Toy, X, y, X_val, y_val, tolerance=tolerance)

    assert list(table.columns) == ["level", "rounds", "val_mse", "val_r2", "train_r2",
                                   "seconds", "chosen"]
    np.testing.assert_allclose(table["val_mse"], Toy.levels, rtol=1e-12)
    assert table["val_r2"].isna().all()                        # y_val is zero: R2 undefined
    levels = np.sqrt(Toy.levels)
    np.testing.assert_allclose(table["train_r2"], 1 - (2.0 - levels) ** 2 / 4.0, rtol=1e-12)
    assert list(table["rounds"]) == [110, 104, 100, 102, 130]
    assert (table["seconds"] >= 0).all()
    assert list(table["chosen"]) == [i == expected for i in range(5)]
    # best_params: the fitted model's refit_params merged over the grid setting.
    assert best == {"level": pytest.approx(levels[expected]),
                    "rounds": [110, 104, 100, 102, 130][expected], "validated": True}
    # val_pred: the validation forecasts of the setting chosen (the toy forecasts its level).
    assert val_pred.dtype == np.float64
    np.testing.assert_allclose(val_pred, np.full(20, levels[expected]), rtol=1e-12)


def test_tune_accepts_a_grid_and_delegates_to_tune_path():
    X, y = np.zeros((10, 1)), np.ones(10)
    X_val, y_val = np.zeros((5, 1)), np.zeros(5)
    best, table, val_pred = tune(Toy, X, y, X_val, y_val, grid=[{"level": 3.0}, {"level": 2.0}])
    assert list(table["val_mse"]) == [9.0, 4.0] and best["level"] == 2.0
    np.testing.assert_array_equal(val_pred, np.full(5, 2.0))
    with pytest.raises(ValueError, match="no settings"):
        tune(Toy, X, y, X_val, y_val, grid=[])

    class WithPath(Toy):
        @classmethod
        def tune_path(cls, X_train, y_train, X_val, y_val, tolerance):
            return ({"level": -1.0, "tolerance": tolerance}, pd.DataFrame({"chosen": [True]}),
                    np.full(len(X_val), -1.0))

    best, table, val_pred = tune(WithPath, X, y, X_val, y_val, tolerance=0.25)
    assert best == {"level": -1.0, "tolerance": 0.25} and len(table) == 1
    np.testing.assert_array_equal(val_pred, np.full(5, -1.0))
    best, table, val_pred = tune(WithPath, X, y, X_val, y_val, grid=[{"level": 3.0}])
    assert best["level"] == 3.0 and list(table["val_mse"]) == [9.0]     # explicit grid: no path
    np.testing.assert_array_equal(val_pred, np.full(5, 3.0))


class GroupedToy(Toy):
    """A toy whose fit must be told the period of every row, as IPCA's must."""

    needs_groups = True

    def fit(self, X, y, X_val=None, y_val=None, groups=None) -> GroupedToy:
        self.groups = groups
        return super().fit(X, y, X_val, y_val)

    def refit_params(self) -> dict:
        return {**super().refit_params(), "periods": len(np.unique(self.groups))}


def test_groups_reach_only_the_models_that_need_them():
    X, y = np.zeros((12, 1)), np.ones(12)
    X_val, y_val = np.zeros((4, 1)), np.zeros(4)
    groups = np.repeat(np.arange(3), 4)                        # three periods of four rows

    best, table, _ = tune(GroupedToy, X, y, X_val, y_val, groups=groups)
    assert best["periods"] == 3 and len(table) == 5
    final = fit_final(GroupedToy, {"level": 1.0}, X, y, groups=groups)
    assert final.groups is groups and final.saw_validation is False
    assert fit_final(GroupedToy, {"level": 1.0}, X, y).groups is None

    # Toy.fit has no groups argument: passing groups to tune or fit_final must not reach it.
    best, _, _ = tune(Toy, X, y, X_val, y_val, groups=groups)
    assert best["level"] == pytest.approx(1.0)
    assert fit_final(Toy, {"level": 1.0}, X, y, groups=groups).saw_validation is False


def test_ipca_is_tuned_and_refitted_through_the_registry():
    """IPCA needs the period of every row; tune and fit_final hand it over as groups."""
    gen = rng(43)
    periods, stocks, p = 50, 80, 6
    X = gen.uniform(-1, 1, size=(periods * stocks, p)).astype(np.float32)
    groups = np.repeat(np.arange(periods), stocks)
    loadings = gen.normal(size=(p, 2))
    factors = np.array([0.5, -0.3]) + 0.5 * gen.normal(size=(periods, 2))
    y = np.einsum("ij,ij->i", X.astype(np.float64) @ loadings, factors[groups])
    y += gen.normal(size=len(y))
    train = groups < 35
    X_train, y_train, X_val, y_val = X[train], y[train], X[~train], y[~train]

    cls = get_model("ipca")
    assert cls is Ipca and cls.needs_groups
    best, table, val_pred = tune(cls, X_train, y_train, X_val, y_val, groups=groups[train])
    assert list(table["n_factors"]) == [g["n_factors"] for g in Ipca.grid()]
    assert table["chosen"].sum() == 1 and best == {"n_factors": table.loc[table["chosen"],
                                                                        "n_factors"].iloc[0]}
    chosen = int(np.flatnonzero(table["chosen"])[0])
    assert chosen == int(np.argmin(table["val_mse"]))
    assert mse(y_val, val_pred) == pytest.approx(table["val_mse"].iloc[chosen], rel=1e-12)
    assert r2_zero(y_val, val_pred) == pytest.approx(table["val_r2"].iloc[chosen], rel=1e-10)
    assert table["val_r2"].iloc[chosen] > 0.02                 # the mean factor is forecastable

    again = fit_final(cls, best, X_train, y_train, groups=groups[train])
    np.testing.assert_allclose(again.predict(X_val), val_pred, atol=1e-10)
    with pytest.raises(ValueError, match="groups"):
        tune(cls, X_train, y_train, X_val, y_val)              # IPCA cannot be fitted without


def test_registry_lists_every_model_with_the_common_interface():
    from ftm.models.ipca import SparseIpca

    names = ("ols", "ridge", "lasso", "enet", "xgb", "rf", "ebm", "ipca", "sipca")
    assert set(MODELS) == set(names)
    assert [get_model(name) for name in names] == [Ols, Ridge, Lasso, ElasticNet, Xgb, Forest,
                                                   Ebm, Ipca, SparseIpca]
    with pytest.raises(KeyError, match="available"):
        get_model("svm")
    for name, cls in MODELS.items():
        assert cls.name == name and isinstance(cls.grid(), list)
        model = cls()
        assert all(callable(getattr(model, method)) for method in ("fit", "predict",
                                                                    "refit_params"))
        if not issubclass(cls, Ipca):                          # ipca.py keeps plain attributes
            assert model.params == cls.defaults
    assert [name for name, cls in MODELS.items() if getattr(cls, "needs_groups", False)
            ] == ["ipca", "sipca"]
    assert all({"seed", "n_jobs"} <= set(MODELS[name].defaults) for name in ("xgb", "rf", "ebm"))
    assert tuning.Model.__name__ == "Model"


def test_tune_then_fit_final_for_the_tree_models():
    X, y = tree_sample(41, 4000, p=4)
    Xv, yv = tree_sample(42, 1500, p=4)

    grid = [{"max_depth": depth, **FAST_XGB} for depth in (1, 2)]
    best, table, val_pred = tune(Xgb, X, y, Xv, yv, grid=grid)
    assert list(table["max_depth"]) == [1, 2] and "rounds" in table.columns
    assert list(table["chosen"]) == [False, True]              # x1 * x2 needs depth 2
    assert best["max_depth"] == 2 and best["early_stopping_rounds"] is None
    assert best["n_estimators"] == table["rounds"].iloc[1] < FAST_XGB["n_estimators"]
    # Fitted again on the training data, best_params give back the validated model.
    again = fit_final(Xgb, best, X, y)
    assert mse(yv, again.predict(Xv)) == pytest.approx(table["val_mse"].iloc[1], rel=1e-6)
    assert r2_zero(y, again.predict(X)) == pytest.approx(table["train_r2"].iloc[1], rel=1e-6)
    assert val_pred.dtype == np.float64 and val_pred.shape == yv.shape
    np.testing.assert_allclose(val_pred, again.predict(Xv), atol=1e-6)

    small = {"n_estimators": 20, "min_samples_leaf": 5, "n_jobs": 2}
    best, table, val_pred = tune(Forest, X, y, Xv, yv, grid=[{"max_depth": 1, **small},
                                                             {"max_depth": 6, **small}])
    assert "rounds" not in table.columns and best["max_depth"] == 6
    again = fit_final(Forest, best, X, y)
    assert mse(yv, again.predict(Xv)) == pytest.approx(table["val_mse"].iloc[1], rel=1e-12)
    np.testing.assert_allclose(val_pred, again.predict(Xv), atol=1e-12)

    fast = {"outer_bags": 2, **FAST_EBM}
    best, table, val_pred = tune(Ebm, X, y, Xv, yv, grid=[{"interactions": 0, **fast},
                                                          {"interactions": 3, **fast}])
    assert list(table["chosen"]) == [False, True] and best["interactions"] == 3
    # The EBM is not refitted: its setting comes back unchanged, early stopping still on.
    assert best == {**Ebm.defaults, "interactions": 3, **fast}
    assert table["val_r2"].iloc[1] > table["val_r2"].iloc[0] + 0.1
    assert mse(yv, val_pred) == pytest.approx(table["val_mse"].iloc[1], rel=1e-12)
    train = np.arange(len(y) + len(yv)) < len(y)
    final = fit_for_test(Ebm, best, np.vstack([X, Xv]), np.concatenate([y, yv]), train, ~train)
    assert (1, 2) in final.term_features()
    np.testing.assert_array_equal(final.predict(Xv), val_pred)        # the validated model itself
    assert chosen_setting(table) == "interactions=3, outer_bags=2, learning_rate=0.2, " \
                                    "max_rounds=400, early_stopping_rounds=10, " \
                                    "smoothing_rounds=5, n_jobs=2"


def test_chosen_setting_names_numbers_and_other_values():
    table = pd.DataFrame({"max_features": ["sqrt", "log2"], "depth": [3, 6], "bootstrap": [True, False],
                          "share": [0.25, 2_500_000.0], "val_mse": [1.0, 0.9],
                          "chosen": [False, True]})
    assert chosen_setting(table) == "max_features=log2, depth=6, bootstrap=False, share=2.5e+06"
    assert chosen_setting(pd.DataFrame({"val_mse": [1.0], "chosen": [True]})) == ""


# ---------------------------------------------------------------------------------------------
# sparse IPCA (the estimator of the original project)
# ---------------------------------------------------------------------------------------------
def _factor_panel(seed=0, periods=80, stocks=400, features=24, relevant=5, factors=2):
    rng = np.random.default_rng(seed)
    gamma = np.zeros((features, factors))
    gamma[:relevant] = np.linalg.qr(rng.normal(size=(relevant, factors)))[0]
    premium = np.array([0.02, 0.01])[:factors]
    X = rng.uniform(-1, 1, (periods * stocks, features)).astype(np.float32)
    groups = np.repeat(np.arange(periods), stocks)
    f = premium + rng.normal(0, 0.04, (periods, factors))
    y = np.einsum("nk,nk->n", X @ gamma, f[groups]) + rng.normal(0, 0.1, len(X))
    return X, y, groups, X @ (gamma @ premium)


def test_sparse_ipca_keeps_the_features_that_matter():
    from ftm.models.ipca import SparseIpca

    X, y, groups, truth = _factor_panel()
    model = SparseIpca(n_factors=2, penalty=0.3).fit(X, y, groups=groups)
    assert model.active_[:5].all() and model.active_[5:].sum() <= 2
    assert (model.coef_[~model.active_] == 0).all()
    assert np.corrcoef(model.predict(X), truth)[0, 1] > 0.9
    loose = SparseIpca(n_factors=2, penalty=1e-4).fit(X, y, groups=groups)
    plain = Ipca(n_factors=2).fit(X, y, groups=groups)
    assert np.corrcoef(loose.predict(X), plain.predict(X))[0, 1] > 0.999
    assert loose.active_.all() and loose.n_iter_ > 1


def test_sparse_ipca_with_a_prohibitive_penalty_forecasts_nothing():
    from ftm.models.ipca import SparseIpca

    X, y, groups, _ = _factor_panel(seed=1)
    model = SparseIpca(n_factors=2, penalty=1.5).fit(X, y, groups=groups)
    assert not model.active_.any() and np.ptp(model.predict(X)) == 0.0
    assert get_model("sipca") is SparseIpca and SparseIpca.grid()[0]["penalty"] >= SparseIpca.grid()[-1]["penalty"]
