"""The result tables that scripts/08_report.py writes, computed from per-period series.

Conventions
-----------
* series has one row per (freq, split, features, model, date): the rank IC of the period,
  the mean squared error of the forecast (mse) and of a zero forecast (mse_zero) and the
  Clark-West term (cw). returns has one row per (freq, split, features, model, strategy,
  date): excess (the net return of the period in excess of the risk-free rate where the
  portfolio is fully invested), cost and turnover. Both come from scripts/07_evaluate.py
  through load().
* split is validation, test, or one of the two halves of the test period, test_2022_23 and
  test_2024, by the day on which the holding period starts.
* Flat forecasts. A period in which an experiment's forecast is the same for every stock
  has no rank IC and its long-short portfolios hold nothing. If the experiment forecasts
  in other periods, such a period counts as an IC of zero. An experiment that never
  forecasts in validation or test (a lasso with every coefficient at zero) has no IC at
  all: it is counted as flat and left out of averages.
* t statistics use Newey-West standard errors. With 7 to 81 periods the normal 1.96 is too
  lenient; the column t_critical gives the 5 percent two-sided critical value for the
  number of periods of the row (inference.critical_value).
* The experiments of a frequency forecast the same returns and are far from independent.
  Statements about groups of experiments are therefore tested on the group's average per
  period, one series, never by counting experiments.
* Models. The eight models of the design (experiments.BASE_MODELS) make up every group of
  the landscape, contrasts, halves, placebo and text-against-price tables, so that those
  tables do not change when variants are added. The variants, models fitted on ranked
  returns ("<model>_rk") and the combination of the eight ("ens"), appear as models of
  their own in the landscape by model, in forecast, strategies and multiple_testing, and
  are compared with what they vary in improvements_table.
"""
from __future__ import annotations

from itertools import combinations
from math import sqrt
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from .. import experiments
from ..models.tuning import chosen_setting
from ..portfolio.strategies import LONG_ONLY, LONG_SHORT
from . import inference, metrics

SUBPERIODS = {"test_2022_23": ("2022-01-01", "2023-12-31"), "test_2024": ("2024-01-01", "2024-12-31")}
BENCHMARK_OF = {"long_ew": "univ_ew", "long_vw": "univ_vw", "tilt_vw": "univ_vw"}
NEWS_KINDS = ("tone", "emb", "news", "px")   # feature sets of the directional news blocks
SHOCK_KINDS = ("narr_s", "text_s", "all_s")  # feature sets with the tuned attention-shock window
# Kinds of feature set of the main run (FINDINGS.md, sections 1 to 9). The tables that
# section quotes are computed over these kinds only, so that the extension of section 10
# adds rows to them instead of changing them.
MAIN_KINDS = ("price", "narr", "firm", "text", "all", "placebo")
KEYS = ["freq", "split", "features", "model"]
HEADLINE = "ls_ew"                   # the strategy used when one number per experiment is needed
TEXT_ONLY = ("narr", "firm", "text")  # kinds of feature set without price features
NONLINEAR = ("xgb", "rf", "ebm")
LINEAR = ("ols", "ridge", "lasso", "ipca", "sipca")
BUFFERED = "ls_buf"                  # the rule compared with HEADLINE in improvements_table
N_BOOT = 5000


# --------------------------------------------------------------------------------- input
def _with_subperiods(table: pd.DataFrame) -> pd.DataFrame:
    """Adds copies of the test rows labelled by the half of the test period they fall in."""
    parts = [table]
    test = table[table["split"] == "test"]
    for name, (start, end) in SUBPERIODS.items():
        inside = (test["hold_start"] >= start) & (test["hold_start"] <= end)
        parts.append(test[inside].assign(split=name))
    return pd.concat(parts, ignore_index=True)


def _flat_periods_as_zero(series: pd.DataFrame) -> pd.DataFrame:
    """Marks periods without a rank IC (column flat) and counts them as an IC of zero for
    experiments that have an IC in another period of validation or test."""
    flat = series["ic"].isna()
    forecasts = (~flat).groupby([series["features"], series["model"]]).transform("any")
    series = series.assign(flat=flat)
    series.loc[flat & forecasts, "ic"] = 0.0
    return series


def load(series_dir: Path, freqs) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(series, returns, factors by frequency) of the frequencies that have been evaluated."""
    series, returns, factors = [], [], {}
    for freq in freqs:
        if not (series_dir / f"{freq}_returns.parquet").exists():
            continue
        forecast = _flat_periods_as_zero(pd.read_parquet(series_dir / f"{freq}_forecast.parquet"))
        series.append(_with_subperiods(forecast).assign(freq=freq))
        returns.append(_with_subperiods(pd.read_parquet(series_dir / f"{freq}_returns.parquet"))
                       .assign(freq=freq))
        factors[freq] = pd.read_csv(series_dir / f"{freq}_factors.csv",
                                    parse_dates=["date"]).set_index("date")
    if not series:
        raise SystemExit(f"no evaluated series in {series_dir}: run scripts/07_evaluate.py first")
    return pd.concat(series, ignore_index=True), pd.concat(returns, ignore_index=True), factors


def family(features: str) -> str:
    """price, text (no price features), all (price and text, the px_ sets included) or
    control (the placebo)."""
    if features.startswith("grand_text"):
        return "text"
    return {"price": "price", "all": "all", "all_s": "all", "px": "all", "grand": "all",
            "placebo": "control"}.get(experiments.kind(features), "text")


def _labelled(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(kind=table["features"].map(experiments.kind),
                        topic_model=table["features"].map(experiments.topic_model))


def _base(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of the eight models of the design (and the benchmark portfolios)."""
    return table[table["model"].isin(experiments.BASE_MODELS) | (table["features"] == "benchmark")]


def _main(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of the feature sets of the main run (MAIN_KINDS) and the benchmark."""
    return table[table["features"].map(experiments.kind).isin(MAIN_KINDS)
                 | (table["features"] == "benchmark")]


def _group_ic(part: pd.DataFrame) -> pd.Series:
    """Average rank IC per period over the experiments of `part`, in date order."""
    return part.groupby("date")["ic"].mean().sort_index()


# --------------------------------------------------------------------- one experiment each
def forecast_table(series: pd.DataFrame, ppy_of: dict) -> pd.DataFrame:
    """Per experiment and split: out-of-sample R2, rank IC, Clark-West and Diebold-Mariano
    tests against a zero forecast."""
    rows = []
    for key, g in series.groupby(KEYS, sort=False):
        g = g.sort_values("date")
        ic = metrics.ic_summary(g["ic"])
        cw = inference.mean_test(g["cw"])
        dm = inference.diebold_mariano(g["mse"], g["mse_zero"])
        rows.append({**dict(zip(KEYS, key)), "periods": len(g), "flat_periods": int(g["flat"].sum()),
                     "r2_oos_pct": 100 * (1 - g["mse"].sum() / g["mse_zero"].sum()),
                     "ic_mean": ic["mean"], "ic_t": ic["t"],
                     "ic_ir_annual": ic["icir"] * sqrt(ppy_of[key[0]]),
                     "ic_share_positive": ic["share_positive"], "clark_west_t": cw["t"],
                     "clark_west_p_one_sided": inference.p_value(cw["t"], len(g), two_sided=False),
                     "diebold_mariano_t": -dm["stat"],
                     "t_critical": inference.critical_value(len(g))})
    return pd.DataFrame(rows)


def strategy_table(returns: pd.DataFrame, factors_of: dict, ppy_of: dict) -> pd.DataFrame:
    """Per experiment, split and strategy: return, risk, Sharpe ratio gross and net of costs,
    drawdown, turnover, break-even cost and three-factor alpha; for the long-only rules
    also the result against the universe with the same weighting."""
    bench = returns[returns["features"] == "benchmark"]
    bench = {k: g.set_index("date")["excess"] for k, g in bench.groupby(["freq", "split", "strategy"])}
    rows = []
    for key, g in returns.groupby(KEYS + ["strategy"], sort=False):
        g = g.sort_values("date")
        freq, split, strategy = key[0], key[1], key[4]
        ppy = ppy_of[freq]
        r = g["excess"].to_numpy()
        gross = r + g["cost"].to_numpy()
        turnover = g["turnover"].mean()
        row = {**dict(zip(KEYS + ["strategy"], key)), **metrics.performance(r, ppy),
               "mean_gross": gross.mean() * ppy, "sharpe_gross": metrics.sharpe(gross, ppy),
               "turnover": turnover, "cost_annual": g["cost"].mean() * ppy,
               "breakeven_cost_bps": 1e4 * gross.mean() / (2 * turnover) if turnover > 0 else np.nan,
               "n_long": g["n_long"].mean(), "n_short": g["n_short"].mean(),
               "t_critical": inference.critical_value(len(g))}
        if split == "test":
            row["sharpe_ci_low"], row["sharpe_ci_high"] = inference.sharpe_ci(r, ppy, n_boot=N_BOOT)
        f = factors_of[freq].reindex(g["date"]).to_numpy()
        fit = inference.ols_hac(r, np.column_stack([np.ones(len(r)), f]))
        row.update(alpha_ff3=fit["coef"][0] * ppy, alpha_ff3_t=fit["t"][0], beta_mkt=fit["coef"][1])
        if strategy in LONG_ONLY:
            b = bench[(freq, split, BENCHMARK_OF[strategy])].reindex(g["date"]).to_numpy()
            rel = metrics.performance(r, ppy, benchmark=b)
            row.update({k: rel[k] for k in ("active_mean", "tracking_error", "information_ratio",
                                            "alpha", "t_alpha", "beta")})
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"n": "periods", "mean": "mean_net", "sharpe": "sharpe_net",
                                              "sharpe_se": "sharpe_net_se", "alpha": "alpha_universe",
                                              "t_alpha": "alpha_universe_t", "beta": "beta_universe"})


def selection_table(forecast: pd.DataFrame, strategy: pd.DataFrame) -> pd.DataFrame:
    """Per frequency: the experiment with the highest validation IC, and its test result.

    Main run (feature sets of MAIN_KINDS): among the experiments of the eight base models,
    over all of them ("any", the headline), within each family (price, text, all) and
    within each topic model (topic_<name>); "any_variant", every experiment of the main
    run including the rank-target models and the ensembles; "ensemble", its ensembles.
    Extension (FINDINGS.md, section 10): "news", the base models on the directional news
    feature sets (NEWS_KINDS); "shock", the base models on the sets with the tuned
    attention-shock window (SHOCK_KINDS); "signal", the single-feature signals
    (experiments.SIGNALS); "any_all", the base models on every feature set. The placebo is
    a control and cannot be selected, and neither can the two combinations of
    scripts/06c_signals.py, which are made of validation choices (their validation IC is
    partly that of the choice)."""
    candidates = forecast[forecast["split"] == "validation"].dropna(subset=["ic_mean"])
    candidates = candidates.assign(family=candidates["features"].map(family),
                                   kind=candidates["features"].map(experiments.kind))
    # combo and best10 are built from validation choices: their validation IC is not a
    # forecast of their test IC, so they are not candidates (they are reported on their own)
    candidates = candidates[(candidates["family"] != "control")
                            & ~candidates["model"].isin([experiments.COMBO, experiments.POOL])]
    base = candidates[candidates["model"].isin(experiments.BASE_MODELS)]
    every = candidates[candidates["kind"].isin(MAIN_KINDS)]
    val = base[base["kind"].isin(MAIN_KINDS)]
    by_topic = val.assign(family=val["features"].map(experiments.topic_model)).dropna(subset=["family"])
    by_topic["family"] = "topic_" + by_topic["family"]
    ensembles = every[every["model"] == experiments.ENSEMBLE]
    best = lambda t, name: t.loc[t.groupby("freq")["ic_mean"].idxmax()].assign(family=name)
    picks = [val.loc[val.groupby(["freq", "family"])["ic_mean"].idxmax()],
             best(val, "any"),
             by_topic.loc[by_topic.groupby(["freq", "family"])["ic_mean"].idxmax()],
             best(every, "any_variant"), best(ensembles, "ensemble"),
             best(base[base["kind"].isin(NEWS_KINDS)], "news"),
             best(base[base["kind"].isin(SHOCK_KINDS)], "shock"),
             best(candidates[candidates["model"].isin(list(experiments.SIGNALS))], "signal"),
             best(base, "any_all")]
    picks = pd.concat(picks)[["freq", "family", "features", "model", "ic_mean", "ic_t", "r2_oos_pct"]]
    picks = picks.rename(columns={c: f"validation_{c}" for c in ("ic_mean", "ic_t", "r2_oos_pct")})
    head = strategy[strategy["strategy"] == HEADLINE]
    for split in ("validation", "test", *SUBPERIODS):
        f = forecast[forecast["split"] == split][["freq", "features", "model", "ic_mean", "ic_t",
                                                  "r2_oos_pct", "clark_west_t", "t_critical"]]
        s = head[head["split"] == split][["freq", "features", "model", "mean_gross", "mean_net",
                                          "sharpe_gross", "sharpe_net", "sharpe_net_se", "t_mean",
                                          "max_drawdown", "turnover"]]
        both = f.merge(s, on=["freq", "features", "model"])
        if split == "validation":
            both = both.drop(columns=["ic_mean", "ic_t", "r2_oos_pct"])
        both = both.rename(columns={c: f"{split}_{c}" for c in both.columns[3:]})
        picks = picks.merge(both, on=["freq", "features", "model"], how="left")
    return picks.sort_values(["freq", "family"], ignore_index=True)


def frequency_table(forecast, strategy, selection) -> pd.DataFrame:
    """The experiments of selection_table under every strategy: test-period accuracy,
    return, Sharpe ratio, costs and drawdown."""
    rows = []
    for pick in selection.itertuples(index=False):
        chosen = ((forecast["freq"] == pick.freq) & (forecast["features"] == pick.features)
                  & (forecast["model"] == pick.model) & (forecast["split"] == "test"))
        f = forecast[chosen].iloc[0]
        s = strategy[(strategy["freq"] == pick.freq) & (strategy["features"] == pick.features)
                     & (strategy["model"] == pick.model) & (strategy["split"] == "test")]
        for strat in s.itertuples(index=False):
            rows.append({"freq": pick.freq, "family": pick.family, "features": pick.features,
                         "model": pick.model, "strategy": strat.strategy, "periods": strat.periods,
                         "ic_mean": f["ic_mean"], "ic_t": f["ic_t"], "ic_ir_annual": f["ic_ir_annual"],
                         "r2_oos_pct": f["r2_oos_pct"], "sharpe_gross": strat.sharpe_gross,
                         "sharpe_net": strat.sharpe_net, "sharpe_net_se": strat.sharpe_net_se,
                         "mean_gross": strat.mean_gross, "mean_net": strat.mean_net,
                         "vol": strat.vol, "turnover_per_period": strat.turnover,
                         "cost_annual": strat.cost_annual,
                         "breakeven_cost_bps": strat.breakeven_cost_bps,
                         "max_drawdown": strat.max_drawdown})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- comparisons
def paired(series: pd.DataFrame, returns: pd.DataFrame, ppy_of: dict) -> pd.DataFrame:
    """Test-period comparison of every experiment that uses text (or the placebo) with the
    price-only experiment of the same model and frequency, period by period."""
    rows = []
    series, returns = _base(series), _base(returns)
    test_f = series[series["split"] == "test"]
    test_r = returns[(returns["split"] == "test") & (returns["strategy"] == HEADLINE)]
    f_by = {k: g.set_index("date").sort_index() for k, g in test_f.groupby(["freq", "features", "model"])}
    r_by = {k: g.set_index("date")["excess"].sort_index()
            for k, g in test_r.groupby(["freq", "features", "model"])}
    for (freq, features, model), f in f_by.items():
        base = f_by.get((freq, "price", model))
        if features == "price" or base is None:
            continue
        base = base.reindex(f.index)
        d_ic = inference.mean_test(f["ic"] - base["ic"])
        dm = inference.diebold_mariano(f["mse"], base["mse"])
        r, r_price = r_by[(freq, features, model)], r_by[(freq, "price", model)]
        sr = inference.sharpe_difference(r, r_price, ppy_of[freq], n_boot=N_BOOT)
        rows.append({"freq": freq, "features": features, "model": model, "family": family(features),
                     "ic": f["ic"].mean(), "ic_price": base["ic"].mean(), "ic_diff_t": d_ic["t"],
                     "t_critical": d_ic["critical"],
                     "r2_oos_pct": 100 * (1 - f["mse"].sum() / f["mse_zero"].sum()),
                     "r2_oos_pct_price": 100 * (1 - base["mse"].sum() / base["mse_zero"].sum()),
                     "diebold_mariano_t": -dm["stat"],
                     "sharpe_net": metrics.sharpe(r, ppy_of[freq]),
                     "sharpe_net_price": metrics.sharpe(r_price, ppy_of[freq]),
                     "sharpe_diff": sr["diff"], "sharpe_diff_p": sr["p"]})
    return pd.DataFrame(rows)


def multiple_testing(returns: pd.DataFrame, ppy_of: dict) -> pd.DataFrame:
    """Per frequency and strategy: the best test-period result among all experiments, and
    what the search for it is worth. The placebo is a control and not part of the search.
    pool "main" searches the experiments of the main run (MAIN_KINDS), pool "all" every
    experiment, the extension of FINDINGS.md section 10 included.

    A long-only portfolio is measured against the universe with the same weighting, so its
    "Sharpe ratio" here is an information ratio: against zero it would pass for earning the
    market's return. Experiments with a flat forecast hold no position (or the universe)
    and cannot be the best; they are counted in `experiments`, not in `with_position`.

    expected_best_sharpe_if_no_skill and best_sharpe_p come from White's reality check on
    returns scaled to unit variance: the periods are resampled with every experiment's
    mean set to zero, and the best Sharpe ratio of each resample is recorded. The mean of
    those is what the best of these experiments would show without skill, given how they
    move together; the p-value is the share of resamples at least as good as the observed
    best. spa_p is Hansen's test, which is not weakened by very poor experiments. The two
    counts use one-sided tests of a positive mean: at 5 percent each, and after the
    Benjamini-Hochberg adjustment at a false discovery rate of 10 percent.
    """
    rows = []
    test = returns[returns["split"] == "test"]
    bench = {k: g.set_index("date")["excess"] for k, g in
             test[test["features"] == "benchmark"].groupby(["freq", "strategy"])}
    real = (test["features"] != "benchmark") & (test["features"].map(family) != "control")
    main = test["features"].map(experiments.kind).isin(MAIN_KINDS)
    groups = [("main", g) for g in test[real & main].groupby(["freq", "strategy"])]
    groups += [("all", g) for g in test[real].groupby(["freq", "strategy"])]
    for pool, ((freq, strategy), g) in groups:
        wide = g.pivot(index="date", columns=["features", "model"], values="excess").sort_index()
        if wide.isna().any().any():
            raise ValueError(f"{freq} {strategy}: experiments cover different periods")
        if strategy in LONG_ONLY:
            universe = bench[(freq, BENCHMARK_OF[strategy])].reindex(wide.index)
            if universe.isna().any():
                raise ValueError(f"{freq} {strategy}: the benchmark lacks periods")
            wide = wide.sub(universe, axis=0)
            wide = wide.mask(wide.abs() <= 1e-12, 0.0)       # a flat forecast holds the universe
        held = wide.loc[:, wide.std(ddof=1) > 0]
        if held.shape[1] == 0:
            continue
        scaled = held / held.std(ddof=1)
        per_period = scaled.mean()
        best = per_period.idxmax()
        annual = sqrt(ppy_of[freq])
        check = inference.reality_check(scaled.to_numpy(), n_boot=N_BOOT)
        t = np.array([inference.mean_test(held[c].to_numpy())["t"] for c in held])
        p = np.array([inference.p_value(value, len(held), two_sided=False) for value in t])
        rows.append({"pool": pool, "freq": freq, "strategy": strategy, "experiments": wide.shape[1],
                     "with_position": held.shape[1], "periods": len(held),
                     "best_features": best[0], "best_model": best[1],
                     "best_sharpe_net": per_period[best] * annual,
                     "median_sharpe_net": per_period.median() * annual,
                     "share_positive_sharpe": float((per_period > 0).mean()),
                     "expected_best_sharpe_if_no_skill": check["expected_stat"] / sqrt(len(held)) * annual,
                     "best_sharpe_p": check["p_reality_check"], "spa_p": check["p_spa"],
                     "significant_5pct_unadjusted": int((p < 0.05).sum()),
                     "significant_10pct_fdr": int((inference.benjamini_hochberg(p) < 0.10).sum())})
    return pd.DataFrame(rows)


def landscape(series: pd.DataFrame, returns: pd.DataFrame, factors_of: dict,
              ppy_of: dict) -> pd.DataFrame:
    """Groups of experiments, per frequency and split: by feature set (over the models), by
    kind of feature set, by topic model (over its text-only feature sets) and by model.

    ic_mean is the group's average rank IC and ic_t the Newey-West t statistic of its
    average IC per period. The Sharpe ratios, the market beta and the t statistic of the
    three-factor alpha (before costs) belong to the equal-weighted average of the group's
    long-short portfolios; averaging lowers the volatility but not the costs, so read
    sharpe_net next to sharpe_net_median, the median over the single portfolios.
    Experiments with a flat forecast are counted and left out of everything else. The
    groups by feature set, kind and topic model hold the eight base models; the grouping
    by model has every model, variants included, on the feature sets of the main run.
    """
    series = _labelled(series)
    spreads = _labelled(returns[(returns["strategy"] == HEADLINE) & (returns["features"] != "benchmark")])
    spreads = spreads[spreads.groupby(KEYS)["turnover"].transform("sum") > 0]
    rows = []
    for by in ("features", "kind", "topic_model", "model"):
        base = (lambda t: _main(t)) if by == "model" else _base
        pick = ((lambda t: base(t)[base(t)["kind"].isin(TEXT_ONLY)]) if by == "topic_model"
                else base)
        portfolios = dict(tuple(pick(spreads).groupby(["freq", "split", by])))
        for (freq, split, group), g in pick(series).groupby(["freq", "split", by]):
            ppy = ppy_of[freq]
            each = g.groupby(["features", "model"])["ic"].mean()
            pooled = inference.mean_test(_group_ic(g))
            r = portfolios.get((freq, split, group), spreads.iloc[:0])
            average = r.groupby("date")[["excess", "cost"]].mean().sort_index()
            gross = (average["excess"] + average["cost"]).to_numpy()
            net = metrics.performance(average["excess"], ppy)
            row = {"freq": freq, "split": split, "by": by, "group": group,
                   "experiments": len(each), "flat_forecasts": int(each.isna().sum()),
                   "periods": g["date"].nunique(), "ic_mean": each.mean(),
                   "ic_min": each.min(), "ic_max": each.max(),
                   "ic_positive": int((each > 0).sum()), "ic_t": pooled["t"],
                   "t_critical": pooled["critical"],
                   "sharpe_gross": metrics.sharpe(gross, ppy), "sharpe_net": net["sharpe"],
                   "sharpe_net_se": net["sharpe_se"],
                   "sharpe_net_median": r.groupby(["features", "model"])["excess"]
                                         .apply(metrics.sharpe, ppy).median(),
                   "turnover": r["turnover"].mean(), "beta_mkt": np.nan, "alpha_ff3_gross_t": np.nan}
            if len(gross) > 4:
                f = factors_of[freq].reindex(average.index).to_numpy()
                fit = inference.ols_hac(gross, np.column_stack([np.ones(len(gross)), f]))
                row.update(beta_mkt=fit["coef"][1], alpha_ff3_gross_t=fit["t"][0])
            rows.append(row)
    return pd.DataFrame(rows)


def _difference(a: pd.Series, b: pd.Series) -> dict:
    """Mean of each of two per-period series over their common periods, and the test of the
    mean difference."""
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().sort_index()
    test = inference.mean_test(both["a"] - both["b"])
    return {"periods": len(both), "ic_a": both["a"].mean(), "ic_b": both["b"].mean(),
            "difference": test["mean"], "difference_t": test["t"], "t_critical": test["critical"]}


def contrasts(series: pd.DataFrame) -> pd.DataFrame:
    """Group against group, per frequency and split: the average rank IC of group a minus
    that of group b in each period, and the Newey-West t statistic of the mean difference.

        topic_model   every pair of topic models, on their text-only feature sets
        model         boosted trees, forest and EBM ("nonlinear") against the five linear
                      models, over the feature sets of the main run but the placebo
        kind          feature sets with price and text ("all") against price alone
    """
    series = _labelled(_base(_main(series)))
    series = series[series["kind"] != "placebo"].copy()
    series["class"] = np.where(series["model"].isin(NONLINEAR), "nonlinear", "linear")
    text_only = series[series["kind"].isin(TEXT_ONLY)]
    topic_models = sorted(text_only["topic_model"].dropna().unique())
    pairs = [("topic_model", text_only, "topic_model", a, b) for a, b in combinations(topic_models, 2)]
    pairs += [("model", series, "class", "nonlinear", "linear"), ("kind", series, "kind", "all", "price")]
    rows = []
    for by, table, column, a, b in pairs:
        for (freq, split), g in table.groupby(["freq", "split"]):
            part_a, part_b = g[g[column] == a], g[g[column] == b]
            if part_a["ic"].notna().any() and part_b["ic"].notna().any():
                rows.append({"freq": freq, "split": split, "by": by, "a": a, "b": b,
                             **_difference(_group_ic(part_a), _group_ic(part_b))})
    return pd.DataFrame(rows)


def halves(series: pd.DataFrame) -> pd.DataFrame:
    """The two halves of the test period against each other, per frequency and group (kind
    of feature set; topic model, on its text-only sets): the group's average rank IC in
    each half, the difference, and z = difference / sqrt(se_1^2 + se_2^2) with the
    Newey-West standard errors of the two means. z is to be read like a t statistic of
    the shorter half (t_critical)."""
    series = _labelled(_base(series))
    rows = []
    for by in ("kind", "topic_model"):
        table = series[series["kind"].isin(TEXT_ONLY)] if by == "topic_model" else series
        for (freq, group), g in table.groupby(["freq", by]):
            tests = [inference.mean_test(_group_ic(g[g["split"] == split])) for split in SUBPERIODS]
            if any(np.isnan(test["mean"]) for test in tests):
                continue
            first, second = tests
            se = sqrt(first["se"] ** 2 + second["se"] ** 2)
            rows.append({"freq": freq, "by": by, "group": group, "periods_2022_23": first["n"],
                         "periods_2024": second["n"], "ic_2022_23": first["mean"],
                         "ic_2024": second["mean"], "difference": first["mean"] - second["mean"],
                         "z": (first["mean"] - second["mean"]) / se if se > 0 else np.nan,
                         "t_critical": max(first["critical"], second["critical"])})
    return pd.DataFrame(rows)


def placebo_table(series: pd.DataFrame) -> pd.DataFrame:
    """Narrative exposures against their placebo, per frequency and split: the rank IC of
    each, averaged over the models that forecast from both, and the Newey-West t statistic
    of the difference per period. If the content of the news matters, it is positive."""
    rows = []
    series = _base(series)
    for control in sorted(f for f in series["features"].unique() if experiments.kind(f) == "placebo"):
        real = control[:-len("_placebo")]
        both = series[series["features"].isin([real, control])]
        for (freq, split), g in both.groupby(["freq", "split"]):
            paired_ic = g.pivot_table(index=["date", "model"], columns="features", values="ic").dropna()
            if not {real, control} <= set(paired_ic.columns) or paired_ic.empty:
                continue
            per_period = paired_ic.groupby("date").mean()
            out = _difference(per_period[real], per_period[control])
            rows.append({"freq": freq, "split": split, "features": real, "placebo": control,
                         "models": paired_ic.index.get_level_values("model").nunique(),
                         "periods": out["periods"], "ic": out["ic_a"], "ic_placebo": out["ic_b"],
                         "difference": out["difference"], "difference_t": out["difference_t"],
                         "t_critical": out["t_critical"]})
    return pd.DataFrame(rows)


def improvements_table(series: pd.DataFrame, returns: pd.DataFrame, ppy_of: dict) -> pd.DataFrame:
    """What each variant changes against what it varies, per frequency, split and group.

        rank_target   <model>_rk against <model>, for every model with a rank-target variant
        ensemble      the combination of the eight base models (ens) against the average of
                      the eight, feature set by feature set
        buffer        the decile long-short portfolio with a buffer (ls_buf) against the one
                      without (ls_ew), for every experiment of the base models
        shock_window  feature sets with the attention-shock window chosen on validation
                      (narr_, text_ and all_<m>_s<w>) against the same sets with the
                      window of the main run, model by model

    Pairs are formed within a feature set; group is the family of the feature set (price,
    text, all) or "every". Real feature sets only: the placebo is left out. The first three
    comparisons use the feature sets of the main run (MAIN_KINDS).

    ic_base, ic_new: average rank IC of the two sides (for the ensemble, ic_base averages
    the members); ic_diff and ic_diff_t: the mean over periods of the per-period difference
    averaged over the pairs, with its Newey-West t statistic (not for the buffer, which
    keeps the forecasts). return_diff_annual and return_diff_t: the same for the net
    return of the decile long-short portfolio (for the buffer, ls_buf minus ls_ew).
    Sharpe ratios are net and annualised: the median over the pairs of each side, the
    median of the paired differences and the share of pairs in which the variant is
    higher. turnover_*: average turnover per period. A side whose forecast is flat in every
    period has no Sharpe ratio; such pairs are left out of the Sharpe columns.
    """
    real = lambda t: t[(t["features"] != "benchmark") & (t["features"].map(family) != "control")]
    series, returns = real(series), real(returns)
    rules = (HEADLINE, BUFFERED)
    returns = returns[returns["strategy"].isin(rules)]
    wide = lambda t, column: t.pivot(index="date", columns=["features", "model"],
                                     values=column).sort_index()
    rows = []
    for (freq, split), f in series.groupby(["freq", "split"]):
        ppy = ppy_of[freq]
        ic = wide(f, "ic")
        r = returns[(returns["freq"] == freq) & (returns["split"] == split)]
        net = {rule: wide(g, "excess") for rule, g in r.groupby("strategy")}
        turnover = {rule: g.groupby(["features", "model"])["turnover"].mean()
                    for rule, g in r.groupby("strategy")}
        sharpe = {rule: table.apply(lambda x: metrics.sharpe(x, ppy)) for rule, table in net.items()}
        if HEADLINE not in net:
            continue
        features = sorted({k for k, _ in ic.columns})
        pairs = []                          # (comparison, features, new side, base sides, rule)
        for feat in features:
            if experiments.kind(feat) in SHOCK_KINDS:
                for model in experiments.BASE_MODELS:
                    pairs.append(("shock_window", feat, (feat, model),
                                  [(re.sub(r"_s\d+$", "", feat), model)], None))
            if experiments.kind(feat) not in MAIN_KINDS:
                continue
            for model in experiments.RANK_MODELS:
                pairs.append(("rank_target", feat, (feat, model), [(feat, model[:-len("_rk")])],
                              None))
            pairs.append(("ensemble", feat, (feat, experiments.ENSEMBLE),
                          [(feat, m) for m in experiments.BASE_MODELS], None))
            if BUFFERED in net:
                for model in experiments.BASE_MODELS:
                    pairs.append(("buffer", feat, (feat, model), [(feat, model)], BUFFERED))
        found = []
        for comparison, feat, new, base, rule in pairs:
            base = [b for b in base if b in ic.columns]
            if new not in ic.columns or not base:
                continue
            new_rule = rule or HEADLINE
            if new not in net[new_rule].columns or not all(b in net[HEADLINE].columns for b in base):
                continue
            ic_new = ic[new] if rule is None else pd.Series(np.nan, index=ic.index)
            ic_old = ic[base].mean(axis=1) if rule is None else pd.Series(np.nan, index=ic.index)
            if rule is None and (ic_new.isna().all() or ic_old.isna().all()):
                continue
            found.append({"comparison": comparison, "group": family(feat),
                          "ic_new": ic_new, "ic_base": ic_old,
                          "ret_new": net[new_rule][new], "ret_base": net[HEADLINE][base].mean(axis=1),
                          "sr_new": sharpe[new_rule][new],
                          "sr_base": float(np.nanmean([sharpe[HEADLINE][b] for b in base]))
                          if any(np.isfinite(sharpe[HEADLINE][b]) for b in base) else np.nan,
                          "to_new": turnover[new_rule][new],
                          "to_base": float(np.mean([turnover[HEADLINE][b] for b in base]))})
        if not found:
            continue
        found = pd.DataFrame(found)
        for (comparison, group), g in pd.concat([found, found.assign(group="every")]).groupby(
                ["comparison", "group"], sort=False):
            ic_diff = pd.concat([n - b for n, b in zip(g["ic_new"], g["ic_base"])], axis=1).mean(axis=1)
            ret_diff = pd.concat([n - b for n, b in zip(g["ret_new"], g["ret_base"])], axis=1).mean(axis=1)
            ic_test, ret_test = inference.mean_test(ic_diff), inference.mean_test(ret_diff)
            sr = g[["sr_new", "sr_base"]].dropna()
            rows.append({"freq": freq, "split": split, "comparison": comparison, "group": group,
                         "pairs": len(g), "periods": len(ret_diff),
                         "ic_base": float(pd.concat(list(g["ic_base"])).mean()),
                         "ic_new": float(pd.concat(list(g["ic_new"])).mean()),
                         "ic_diff": ic_test["mean"], "ic_diff_t": ic_test["t"],
                         "return_diff_annual": ret_test["mean"] * ppy, "return_diff_t": ret_test["t"],
                         "t_critical": ret_test["critical"],
                         "sharpe_base_median": sr["sr_base"].median(),
                         "sharpe_new_median": sr["sr_new"].median(),
                         "sharpe_diff_median": (sr["sr_new"] - sr["sr_base"]).median(),
                         "sharpe_improved_share": float((sr["sr_new"] > sr["sr_base"]).mean())
                         if len(sr) else np.nan,
                         "turnover_base": g["to_base"].mean(), "turnover_new": g["to_new"].mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ from other folders
def tuning_table(results: Path, freqs) -> pd.DataFrame:
    """What the validation period chose in each finished experiment, from results/tuning/
    and results/importance/: the setting and its place in the grid (1 = the most
    regularised), the boosting rounds kept, validation and training R2, how many features
    (or EBM terms) carry a weight in the model that forecasts the test period, and the
    fitting time."""
    rows = []
    for freq in freqs:
        for path in sorted((results / "tuning" / freq).glob("*.json")):
            meta = json.loads(path.read_text())
            weights = results / "importance" / freq / f"{path.stem}.csv"
            if "seconds_final" not in meta or not weights.exists():       # final fit still to come
                continue
            grid = pd.read_csv(path.with_suffix(".csv"))
            place = int(np.flatnonzero(grid["chosen"])[0])
            weights = pd.read_csv(weights)["importance"]
            rows.append({"freq": freq, "features": meta["features"], "model": meta["model"],
                         "n_features": meta["n_features"], "settings_tried": meta["settings_tried"],
                         "setting": chosen_setting(grid), "place_in_grid": place + 1,
                         "rounds": grid.iloc[place].get("rounds", np.nan),
                         "validation_r2_pct": 100 * meta["validation_r2"],
                         "train_r2_pct": 100 * meta["train_r2"], "terms": len(weights),
                         "terms_used": int((weights != 0).sum()),
                         "seconds": meta["seconds_tune"] + meta["seconds_final"]})
    return pd.DataFrame(rows)


def topic_table(topics: Path) -> pd.DataFrame:
    """Quality measures of the topic models, one row each (data/topics/*/quality.json)."""
    rows = []
    for path in sorted(topics.glob("*/quality.json")):
        q = json.loads(path.read_text())
        shock = q.pop("shock_reliability", {})
        rows.append({**q, **{f"shock_reliability_{k}": v["mean"] for k, v in shock.items()}})
    return pd.DataFrame(rows)
