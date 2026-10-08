"""The scripts 05 to 12, end to end, on an artificial market with a planted signal.

Stocks differ in volatility and expected returns rise with it, so the price
features (which include volatility) must forecast returns while the artificial
news, which is pure noise, must not be needed. Runs in about a minute.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ftm import experiments
from ftm.topics import attention

from synthetic import make_market, make_news, make_sentiment

ROOT = Path(__file__).resolve().parents[1]
MODELS = list(experiments.MODEL_ORDER)          # what 06 fits
BASE = list(experiments.BASE_MODELS)            # the models of the group tables
FEATURES = ["price", "all_toy", "narr_toy", "narr_toy_placebo"]
NEW = ["px_tone", "emb_fb", "narr_toy_s1"]       # tone blocks and a tuned shock window: base models
SIGNALS = list(experiments.SIGNALS) + [experiments.COMBO, experiments.POOL]


def run(script, *args, env):
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    data, results = tmp_path_factory.mktemp("data"), tmp_path_factory.mktemp("results")
    truth = make_market(data / "crsp", n_stocks=120, list_size=80, end="2024-12-31",
                        full_until=2016, drift=0.1)
    sums, firm = make_news(truth["days"], truth["permnos"])
    (data / "topics" / "toy").mkdir(parents=True)
    attention.save_daily_sums(sums, data / "topics" / "toy" / "daily_sums.parquet")
    firm.to_parquet(data / "topics" / "toy" / "firm_daily.parquet", index=False)
    make_sentiment(data, truth["days"], truth["permnos"])
    env = {**os.environ, "FTM_DATA": str(data), "FTM_RESULTS": str(results)}
    run("05_build_panel.py", "--freq", "M", "--placebo", "toy", env=env)
    run("05_build_panel.py", "--freq", "M", "--tone", env=env)
    run("05_build_panel.py", "--freq", "M", "--shock-window", "1", env=env)
    run("05b_tune_shocks.py", "--freq", "M", "--windows", "1", "5", "--models", "ridge", env=env)
    run("06_run_models.py", "--freq", "M", "--features", *FEATURES, env=env)
    run("06_run_models.py", "--freq", "M", "--features", *NEW, "--models", *BASE, env=env)
    run("06b_ensemble.py", "--freq", "M", env=env)
    run("06c_signals.py", "--freq", "M", env=env)
    run("07_evaluate.py", "--freq", "M", env=env)
    run("08_report.py", "--freq", "M", env=env)
    run("09_explain.py", "--freq", "M", env=env)
    run("09b_ipca_factors.py", "--freq", "M", "--models", "sipca", "ipca", "--features", "price",
        "all_toy", env=env)
    run("10_audit.py", "--topics", "toy", env=env)
    run("11_missing_data_leak.py", "--freq", "M", env=env)
    run("12_tuning_noise.py", "--freq", "M", "--features", "price", "--seeds", "1", env=env)
    run("12_tuning_noise.py", "--freq", "M", "--features", "price", "--seeds", "0", "1", env=env)
    return data, results, env


def test_every_experiment_has_forecasts(workspace):
    data, results, _ = workspace
    index = pd.read_parquet(data / "panels" / "M" / "index.parquet")
    assert set(index["split"]) == {"train", "validation", "test"}
    files = sorted(p.name for p in (data / "predictions" / "M").glob("*.test.npy"))
    signals = [f"{features}__{name}.test.npy" for name, (block, _, _) in experiments.SIGNALS.items()
               for features in [{"tone_fb": "tone_fb", "tone_lm": "tone_lm", "npx": "px_tone"}[block]]]
    assert files == sorted([f"{f}__{m}.test.npy" for f in FEATURES
                            for m in MODELS + [experiments.ENSEMBLE]]
                           + [f"{f}__{m}.test.npy" for f in NEW for m in BASE + [experiments.ENSEMBLE]]
                           + signals + ["px_tone__combo.test.npy", "px_pool__best10.test.npy"])
    logs = list((results / "tuning" / "M").glob("*.log"))
    assert len(logs) == len(FEATURES) * len(MODELS) + len(NEW) * len(BASE)
    assert all("finished" in p.read_text() for p in logs)
    chosen = pd.read_csv(results / "tuning" / "M" / "signals.csv")
    assert chosen["combined"].sum() == 1 and chosen.loc[chosen["combined"], "signal"].str.match(
        r"sig_(fb|lm)").all()
    pool = pd.read_csv(results / "tuning" / "M" / "pool.csv")
    assert pool["chosen"].sum() == experiments.POOL_SIZE
    assert len(pool) == (len(FEATURES) - 1 + len(NEW)) * len(BASE)
    shocks = pd.read_csv(results / "tuning" / "shock_window.csv")
    assert set(shocks["window"]) == {1, 5} and set(shocks["topics"]) == {"toy"}
    stored = pd.read_csv(results / "tables" / "forecast.csv").query(
        "features == 'narr_toy' and model == 'ridge' and split == 'validation'")["ic_mean"].iloc[0]
    assert shocks.query("window == 5")["ic_mean"].iloc[0] == pytest.approx(stored, abs=1e-6)
    tuning = pd.read_csv(results / "tuning" / "M" / "price__xgb.csv")
    assert tuning["chosen"].sum() == 1 and len(tuning) == 4


def test_planted_signal_is_found(workspace):
    _, results, _ = workspace
    forecast = pd.read_csv(results / "tables" / "forecast.csv")
    row = forecast[(forecast["features"] == "price") & (forecast["model"] == "ridge")
                   & (forecast["split"] == "test")].iloc[0]
    assert row["ic_mean"] > 0.03 and row["ic_t"] > 2 and row["r2_oos_pct"] > 0
    strategies = pd.read_csv(results / "tables" / "strategies.csv")
    spread = strategies[(strategies["features"] == "price") & (strategies["model"] == "ridge")
                        & (strategies["split"] == "test") & (strategies["strategy"] == "ls_ew")].iloc[0]
    assert spread["sharpe_gross"] > spread["sharpe_net"] > 0
    assert spread["sharpe_ci_low"] < spread["sharpe_net"] < spread["sharpe_ci_high"]
    assert 0 < spread["turnover"] <= 2 and spread["max_drawdown"] <= 0
    assert set(strategies["split"]) == {"validation", "test", "test_2022_23", "test_2024"}
    assert "ls_buf" in set(strategies["strategy"])
    ranked = strategies[(strategies["features"] == "price") & (strategies["model"] == "ridge_rk")
                        & (strategies["split"] == "test") & (strategies["strategy"] == "ls_ew")].iloc[0]
    assert ranked["sharpe_net"] > 0                     # the rank target finds the planted signal


def test_tables_are_complete(workspace):
    _, results, _ = workspace
    tables = {p.stem: pd.read_csv(p) for p in (results / "tables").glob("*.csv")}
    assert {"forecast", "strategies", "selection", "text_vs_price", "multiple_testing",
            "frequency", "landscape", "contrasts", "halves", "placebo", "tuning",
            "improvements"} <= set(tables)
    tuning = tables["tuning"].set_index(["features", "model"])
    assert len(tuning) == len(FEATURES) * len(MODELS) + len(NEW) * len(BASE)
    assert tuning.loc[("price", "ols"), "terms_used"] == 8 and tuning.loc[("price", "xgb"), "rounds"] >= 1
    assert (tuning["place_in_grid"] <= tuning["settings_tried"]).all()
    # the placebo is a control: it is evaluated but can never be the selected experiment
    assert set(tables["selection"]["family"]) == {"price", "all", "text", "any", "topic_toy",
                                                  "any_variant", "ensemble", "news", "signal",
                                                  "shock", "any_all"}
    selection = tables["selection"].set_index("family")
    assert selection.loc["ensemble", "model"] == experiments.ENSEMBLE
    assert selection.loc["any", "model"] in BASE
    assert selection.loc["news", "features"] in ("px_tone", "emb_fb") and selection.loc["news", "model"] in BASE
    assert selection.loc["signal", "model"] in experiments.SIGNALS
    assert not set(tables["selection"]["model"]) & {experiments.COMBO, experiments.POOL}
    assert "narr_toy_placebo" not in set(tables["selection"]["features"])
    groups = tables["landscape"][tables["landscape"]["split"] == "test"].set_index(["by", "group"])
    assert groups.loc[("kind", "price"), "experiments"] == len(BASE)
    assert groups.loc[("kind", "price"), "ic_t"] > 2
    assert groups.loc[("topic_model", "toy"), "experiments"] == len(BASE)        # narr_toy only
    assert groups.loc[("model", "ens"), "experiments"] == len(FEATURES)          # main run only
    assert set(groups.loc["kind"].index) == {"price", "all", "narr", "placebo", "px", "emb", "narr_s"}
    placebo = tables["placebo"].set_index("split").loc["test"]
    assert placebo["features"] == "narr_toy" and abs(placebo["difference_t"]) < 4    # noise against noise
    pairs = tables["text_vs_price"]
    assert len(pairs) == (3 + len(NEW)) * len(BASE) and set(pairs["family"]) == {"all", "text", "control"}
    testing = tables["multiple_testing"].set_index(["pool", "strategy"])         # without the placebo
    assert testing.loc[("main", "ls_ew"), "experiments"] == (len(FEATURES) - 1) * (len(MODELS) + 1)
    testing = testing.loc[("all", "ls_ew")]
    assert testing["experiments"] == ((len(FEATURES) - 1) * (len(MODELS) + 1)
                                      + len(NEW) * (len(BASE) + 1) + len(SIGNALS))
    assert 0 <= testing["spa_p"] <= 1
    assert "tilt_vw" in set(tables["multiple_testing"]["strategy"])
    assert testing["with_position"] <= testing["experiments"]
    assert testing["best_features"] != "narr_toy_placebo"
    assert testing["expected_best_sharpe_if_no_skill"] > 0
    contrasts = tables["contrasts"].set_index(["by", "split"])
    assert contrasts.loc[("kind", "test"), "a"] == "all" and contrasts.loc[("kind", "test"), "b"] == "price"
    assert contrasts.loc[("model", "test"), "periods"] == tables["forecast"]["periods"].max()
    halves = tables["halves"].set_index(["by", "group"])
    assert halves.loc[("kind", "price"), "periods_2022_23"] + halves.loc[("kind", "price"), "periods_2024"] \
        == contrasts.loc[("kind", "test"), "periods"]
    assert (tables["forecast"]["t_critical"] > 1.96).all()
    better = tables["improvements"].set_index(["split", "comparison", "group"])
    assert set(better.index.get_level_values("comparison")) == {"rank_target", "ensemble", "buffer",
                                                                "shock_window"}
    assert better.loc[("test", "shock_window", "every"), "pairs"] <= len(BASE)
    forecasts = tables["forecast"].query("split == 'test' and features != 'narr_toy_placebo'")
    forecasts = forecasts.dropna(subset=["ic_mean"]).set_index(["features", "model"]).index
    both = sum((f, m) in forecasts and (f, m[:-3]) in forecasts      # pairs where both sides forecast
               for f in FEATURES[:3] for m in experiments.RANK_MODELS)
    assert better.loc[("test", "rank_target", "every"), "pairs"] == both >= 12
    assert better.loc[("test", "ensemble", "every"), "pairs"] == 3              # main run only
    buffer = better.loc[("test", "buffer", "every")]
    assert buffer["turnover_new"] < buffer["turnover_base"] and np.isnan(buffer["ic_diff"])
    long_only = tables["strategies"][tables["strategies"]["strategy"] == "long_ew"]
    differs = long_only["tracking_error"] > 0             # a flat forecast holds the universe
    assert differs.any() and (long_only["information_ratio"].notna() == differs).all()
    tilt = tables["strategies"].query("strategy == 'tilt_vw' and split == 'test'")
    assert tilt["information_ratio"].notna().any() and (tilt["tracking_error"] < 0.2).all()


def test_audits(workspace):
    _, results, _ = workspace
    audit = results / "data_audit"
    donors = pd.read_csv(audit / "donor_rows.csv").set_index(["block", "split"])["share_donor"]
    assert set(donors.index.get_level_values("block")) == {"price", "narr_toy", "narr_toy_placebo",
                                                           "firm_toy", "narr_toy_s1", "tone_fb",
                                                           "tone_lm", "emb_fb", "npx"}
    assert donors.between(0, 1).all() and donors["price"].max() < 0.5
    stale = pd.read_csv(audit / "stale_rows.csv")
    assert set(stale["return_file"]) == {"complete", "incomplete"}
    leak = pd.read_csv(audit / "missing_data_leak.csv")
    assert list(leak["missing_features"]) == ["median", "median", "donor", "donor"]
    noise = pd.read_csv(audit / "tuning_noise.csv")
    assert list(noise["seed"]) == [0, 1] and noise["test_ic_mean"].notna().all()
    stored = pd.read_csv(results / "tables" / "forecast.csv")
    stored = stored[(stored["features"] == "price") & (stored["model"] == "xgb") & (stored["split"] == "test")]
    assert noise["test_ic_mean"].iloc[0] == pytest.approx(stored["ic_mean"].iloc[0], rel=1e-4)


def test_second_run_does_nothing(workspace):
    _, _, env = workspace
    out = run("06_run_models.py", "--freq", "M", "--features", *FEATURES, env=env)
    assert "fitted" not in out and "0 unfinished" in out
    assert "0 experiments evaluated" in run("07_evaluate.py", "--freq", "M", env=env)


def test_explanations(workspace):
    _, results, _ = workspace
    folder = results / "explain" / "M__all_toy"
    importance = pd.read_csv(folder / "importance.csv")
    assert np.isclose(importance["share"].sum(), 1) and set(importance["block"]) >= {"price"}
    blocks = pd.read_csv(folder / "blocks.csv").set_index("block")["share"]
    assert blocks["price"] > 0.3                         # the planted signal is a price feature
    ablation = pd.read_csv(folder / "ablation.csv").set_index("deleted")
    assert ablation.loc["block: price", "change_in_ic_mean"] < -0.02
    shapes = pd.read_csv(folder / "shapes.csv")
    assert (shapes["to"] >= shapes["from"]).all() and shapes["term"].nunique() >= 5


def test_ipca_loadings_and_factors(workspace):
    data, results, _ = workspace
    index = pd.read_parquet(data / "panels" / "M" / "index.parquet")
    for features, model in [("price", "sipca"), ("price", "ipca"), ("all_toy", "sipca")]:
        folder = results / "ipca" / "M" / f"{features}__{model}"
        summary = __import__("json").loads((folder / "summary.json").read_text())
        gamma = pd.read_csv(folder / "gamma.csv")
        factors = pd.read_csv(folder / "factors.csv", parse_dates=["date", "hold_start"])
        cols = [f"f{j + 1}" for j in range(summary["factors"])]
        assert summary["test_forecasts_reproduced_max_abs_error"] < 1e-6
        g = gamma[cols].to_numpy()
        assert np.allclose(g.T @ g, np.eye(len(cols)), atol=1e-8)          # Gamma'Gamma = I
        fit = factors[factors["estimate"] == "in_sample"]
        assert len(fit) == index.loc[index["split"] != "test", "date"].nunique()
        assert (factors["estimate"] == "out_of_sample").sum() == index.loc[
            index["split"] == "test", "date"].nunique()
        assert (factors.loc[factors["estimate"] == "out_of_sample", "split"] == "test").all()
        f = fit[cols].to_numpy()
        assert (f.mean(axis=0) > 0).all()                                    # positive means
        c = np.cov(f, rowvar=False, bias=True) + np.outer(f.mean(0), f.mean(0))
        assert np.allclose(c - np.diag(np.diag(c)), 0, atol=1e-10 * np.abs(c).max())
        importance = pd.read_csv(results / "importance" / "M" / f"{features}__{model}.csv")
        implied = pd.Series(summary["implied_coefficients"])
        assert np.allclose(implied[importance["feature"]].to_numpy(), importance["coef"], rtol=1e-6)
        if model == "sipca":
            assert gamma["kept"].sum() == summary["features_kept"]
            assert (gamma.loc[~gamma["kept"], cols] == 0).all().all()
