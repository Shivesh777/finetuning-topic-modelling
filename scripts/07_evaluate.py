"""Score the forecasts and run the portfolio backtests of every finished experiment.

    python scripts/07_evaluate.py [--freq M 2W W] [--budget SECONDS] [--force]

For the validation and the test period of each experiment of scripts/06_run_models.py:

  forecast accuracy   out-of-sample R2 against a zero forecast, the rank information
                      coefficient of every period, the mean squared error of every
                      period next to that of the zero forecast
  portfolios          the strategies of ftm/portfolio/strategies.py, with turnover and
                      costs, each starting from cash at the start of the period

Outputs (per-period series; scripts/08_report.py turns them into tables with inference)
    results/series/<freq>_forecast.parquet   split, features, model, date, ic, mse, mse_zero, cw
    results/series/<freq>_returns.parquet    split, features, model, strategy, date, gross, net,
                                             excess, turnover, cost, ...
    results/series/<freq>_factors.csv        market, size and value factor returns per period
    results/series/<freq>_done.csv           the experiments covered (and their time stamps)

Experiments already covered are skipped unless their forecasts changed.
"""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config, experiments
from ftm.data.returns import Market
from ftm.evaluation import metrics
from ftm.portfolio import strategies

KEYS = ["split", "features", "model"]
BENCHMARK = ("benchmark", "-")              # features and model of the universe portfolios


def factor_returns(market: Market, freq: str) -> pd.DataFrame:
    """Compounded factor returns of every holding period, indexed by formation date."""
    forms = market.formation_days(freq)
    growth = np.vstack([np.zeros((1, 3)), np.cumsum(np.log1p(
        market.factors[["mktrf", "smb", "hml"]].to_numpy()), axis=0)])
    out = np.expm1(growth[forms[1:] + 1] - growth[forms[:-1] + 1])
    return pd.DataFrame(out, index=market.dates[forms[:-1]].rename("date"),
                        columns=["mktrf", "smb", "hml"])


def forecast_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Per formation date: rank IC, mean squared error of the forecast and of a zero
    forecast, and the Clark-West term mean(2 y pred) = MSE(0) - MSE(model) + mean(pred^2)."""
    y, pred, date = frame["y_rel"].to_numpy(), frame["pred"].to_numpy(), frame["date"]
    by = lambda values: pd.Series(values).groupby(date.to_numpy()).mean()
    out = pd.DataFrame({"ic": metrics.rank_ic(y, pred, date), "mse": by((y - pred) ** 2),
                        "mse_zero": by(y ** 2), "cw": by(2 * y * pred)})
    return out.rename_axis("date").reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--budget", type=float, help="start no new experiment after this many seconds")
    ap.add_argument("--force", action="store_true", help="evaluate everything again")
    a = ap.parse_args()
    t0 = time.time()
    out = config.RESULTS / "series"
    out.mkdir(parents=True, exist_ok=True)
    market = Market.load()
    for freq in a.freq:
        index = pd.read_parquet(config.PANELS / freq / "index.parquet")
        paths = {name: out / f"{freq}_{name}.parquet" for name in ("forecast", "returns")}
        done_path = out / f"{freq}_done.csv"
        old = {name: pd.read_parquet(p) for name, p in paths.items()
               if p.exists() and done_path.exists() and not a.force}
        done = (pd.read_csv(done_path) if old else
                pd.DataFrame(columns=["features", "model", "stamp"]))
        stamp = {(f, m): int(experiments.prediction_path(freq, f, m, "test").stat().st_mtime)
                 for f, m in experiments.finished(freq)}
        fresh = {(f, m) for f, m, s in done.itertuples(index=False) if stamp.get((f, m)) == s}
        todo = [key for key in stamp if key not in fresh]
        new = {"forecast": [], "returns": []}
        starts, covs = {}, {}

        def add(name, table, split, features, model):
            new[name].append(table.assign(hold_start=table["date"].map(starts[split]), split=split,
                                          features=features, model=model))

        for split in experiments.EVALUATED:
            rows = index[index["split"] == split]
            starts[split] = rows.drop_duplicates("date").set_index("date")["hold_start"]
            covs[split] = strategies.covariances(rows, market) if todo else None
            add("returns", strategies.backtests(rows.assign(pred=0.0), strategies.BENCHMARKS),
                split, *BENCHMARK)                     # the universe portfolios, always redone
        evaluated = []
        for features, model in todo:
            if a.budget and time.time() - t0 > a.budget:
                break
            for split in experiments.EVALUATED:
                frame = experiments.load_predictions(index, freq, features, model, split)
                add("forecast", forecast_series(frame), split, features, model)
                add("returns", strategies.backtests(frame, strategies.STRATEGIES, covs[split]),
                    split, features, model)
            evaluated.append((features, model))
        covered = [(*BENCHMARK, 0)] + [(f, m, stamp[(f, m)]) for f, m in evaluated]
        # keep what is still valid of the old tables, replace the rest
        redone = {(f, m) for f, m, _ in covered}
        for name, path in paths.items():
            parts = new[name]
            if name in old:
                key = list(zip(old[name]["features"], old[name]["model"]))
                keep = [k in fresh and k not in redone for k in key]
                parts = [old[name][keep]] + parts
            if parts:
                table = pd.concat(parts, ignore_index=True)
                first = KEYS + [c for c in ("strategy", "date") if c in table]
                table[first + [c for c in table if c not in first]].to_parquet(path, index=False)
        kept = done[[(f, m) in fresh and (f, m) not in redone
                     for f, m in zip(done["features"], done["model"])]]
        pd.concat([kept, pd.DataFrame(covered, columns=["features", "model", "stamp"])],
                  ignore_index=True).to_csv(done_path, index=False)
        factor_returns(market, freq).to_csv(out / f"{freq}_factors.csv")
        print(f"{freq}: {len(evaluated)} experiments evaluated, {len(todo) - len(evaluated)} left, "
              f"{len(fresh)} unchanged ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
