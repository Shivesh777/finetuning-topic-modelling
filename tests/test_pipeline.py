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

from synthetic import make_market, make_news

ROOT = Path(__file__).resolve().parents[1]
MODELS = list(experiments.MODEL_ORDER)
FEATURES = ["price", "all_toy", "narr_toy", "narr_toy_placebo"]


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
    env = {**os.environ, "FTM_DATA": str(data), "FTM_RESULTS": str(results)}
    run("05_build_panel.py", "--freq", "M", "--placebo", "toy", env=env)
    run("06_run_models.py", "--freq", "M", "--features", *FEATURES, env=env)
    run("07_evaluate.py", "--freq", "M", env=env)
    run("08_report.py", "--freq", "M", env=env)
    run("09_explain.py", "--freq", "M", env=env)
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
    assert files == sorted(f"{f}__{m}.test.npy" for f in FEATURES for m in MODELS)
    logs = list((results / "tuning" / "M").glob("*.log"))
    assert len(logs) == len(files) and all("finished" in p.read_text() for p in logs)
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


def test_tables_are_complete(workspace):
    _, results, _ = workspace
    tables = {p.stem: pd.read_csv(p) for p in (results / "tables").glob("*.csv")}
    assert {"forecast", "strategies", "selection", "text_vs_price", "multiple_testing",
            "frequency", "landscape", "contrasts", "halves", "placebo", "tuning"} <= set(tables)
    tuning = tables["tuning"].set_index(["features", "model"])
    assert len(tuning) == len(FEATURES) * len(MODELS)
    assert tuning.loc[("price", "ols"), "terms_used"] == 8 and tuning.loc[("price", "xgb"), "rounds"] >= 1
    assert (tuning["place_in_grid"] <= tuning["settings_tried"]).all()
    # the placebo is a control: it is evaluated but can never be the selected experiment
    assert set(tables["selection"]["family"]) == {"price", "all", "text", "any", "topic_toy"}
    assert "narr_toy_placebo" not in set(tables["selection"]["features"])
    groups = tables["landscape"][tables["landscape"]["split"] == "test"].set_index(["by", "group"])
    assert groups.loc[("kind", "price"), "experiments"] == len(MODELS)
    assert groups.loc[("kind", "price"), "ic_t"] > 2
    assert groups.loc[("topic_model", "toy"), "experiments"] == len(MODELS)      # narr_toy only
    assert set(groups.loc["kind"].index) == {"price", "all", "narr", "placebo"}
    placebo = tables["placebo"].set_index("split").loc["test"]
    assert placebo["features"] == "narr_toy" and abs(placebo["difference_t"]) < 4    # noise against noise
    pairs = tables["text_vs_price"]
    assert len(pairs) == 3 * len(MODELS) and set(pairs["family"]) == {"all", "text", "control"}
    testing = tables["multiple_testing"].set_index("strategy").loc["ls_ew"]         # without the placebo
    assert testing["experiments"] == (len(FEATURES) - 1) * len(MODELS) and 0 <= testing["spa_p"] <= 1
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
    long_only = tables["strategies"][tables["strategies"]["strategy"] == "long_ew"]
    differs = long_only["tracking_error"] > 0             # a flat forecast holds the universe
    assert differs.any() and (long_only["information_ratio"].notna() == differs).all()


def test_audits(workspace):
    _, results, _ = workspace
    audit = results / "data_audit"
    donors = pd.read_csv(audit / "donor_rows.csv").set_index(["block", "split"])["share_donor"]
    assert set(donors.index.get_level_values("block")) == {"price", "narr_toy", "narr_toy_placebo",
                                                           "firm_toy"}
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
