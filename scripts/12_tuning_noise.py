"""How much of one experiment's result is the luck of the draw?

    python scripts/12_tuning_noise.py [--freq M 2W W] [--features price all_lda]
                                      [--model xgb] [--seeds 0 1 2 3 4 5 6 7 8 9]

Boosted trees and forests draw rows and columns at random. Here one experiment is run
again with other seeds, each time as everywhere else: tuned on the validation period,
refitted on training plus validation, scored on the test period. Nothing but the seed
changes, so the spread of the results across seeds is the part of a result that owes
nothing to the data. Seed 0 is the experiment of scripts/06_run_models.py.

Writes results/data_audit/tuning_noise.csv: per seed the setting that validation chose, its
validation R2 and rank IC, the test rank IC and the long-short Sharpe ratio. Rows already in the file
are kept unless the same experiment and seed are run again, so a long job can be split
over several calls.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config, experiments
from ftm.evaluation import metrics
from ftm.models.tuning import chosen_setting, fit_for_test, get_model, target_of, tune
from ftm.panel import Panel
from ftm.portfolio import strategies


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--features", nargs="+", default=["price", "all_lda"])
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    a = ap.parse_args()
    cls = get_model(a.model)
    if "seed" not in getattr(cls, "defaults", {}):
        raise SystemExit(f"{a.model} has no random element: its result does not depend on a seed")
    out = config.RESULTS / "data_audit" / "tuning_noise.csv"
    rows = []
    for freq in a.freq:
        ppy = metrics.PERIODS_PER_YEAR[freq]
        # load only the blocks of the requested feature sets (all weekly blocks need > 4 GB)
        names = json.loads((config.PANELS / freq / "names.json").read_text())["names"]
        sets = experiments.feature_sets(names)
        panel = Panel.load(config.PANELS / freq, sorted({b for f in a.features for b in sets[f]}))
        index = panel.index
        train, val, test = (panel.rows(s) for s in ("train", "validation", "test"))
        groups = index["period"].to_numpy()
        y = target_of(cls, index["y_rel"].to_numpy(), groups)
        for features in a.features:
            X = panel.X(sets[features])
            for seed in a.seeds:
                grid = [{**setting, "seed": seed} for setting in cls.grid()]
                params, table, val_pred = tune(cls, X[train], y[train], X[val], y[val],
                                               config.TUNE_TOLERANCE, grid=grid,
                                               groups=groups[train])
                chosen = table[table["chosen"]].iloc[0]
                model = fit_for_test(cls, params, X, y, train, val, groups)
                frame = index[test].assign(pred=model.predict(X[test]))
                ic = metrics.ic_summary(metrics.rank_ic(frame["y_rel"], frame["pred"], frame["date"]))
                val_ic = metrics.rank_ic(index.loc[val, "y_rel"], val_pred, index.loc[val, "date"])
                spread = strategies.backtests(frame, ("ls_ew",))
                rows.append({"freq": freq, "features": features, "model": a.model, "seed": seed,
                             "setting": chosen_setting(table.drop(columns="seed")),
                             "rounds": chosen.get("rounds"),
                             "validation_r2_pct": float(100 * chosen["val_r2"]),
                             "validation_ic_mean": metrics.ic_summary(val_ic)["mean"],
                             "test_ic_mean": ic["mean"], "test_ic_t": ic["t"],
                             "test_sharpe_gross": metrics.sharpe(spread["gross"], ppy),
                             "test_sharpe_net": metrics.sharpe(spread["excess"], ppy)})
                print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rows[-1].items()), flush=True)
    table = pd.DataFrame(rows)
    keys = ["freq", "features", "model", "seed"]
    if out.exists():                              # keep the rows of earlier calls
        old = pd.read_csv(out)
        rerun = old.set_index(keys).index.isin(table.set_index(keys).index)
        table = pd.concat([old[~rerun], table], ignore_index=True)
    order = table["freq"].map({f: i for i, f in enumerate(config.FREQUENCIES)})
    table = table.iloc[np.lexsort([table["seed"], table["model"], table["features"], order])]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.5g")
    summary = table.groupby(["freq", "features", "model"], sort=False).agg(
        seeds=("seed", "size"), settings_chosen=("setting", "nunique"),
        validation_ic_min=("validation_ic_mean", "min"), validation_ic_max=("validation_ic_mean", "max"),
        ic_min=("test_ic_mean", "min"),
        ic_median=("test_ic_mean", "median"), ic_max=("test_ic_mean", "max"),
        ic_sd=("test_ic_mean", "std"), sharpe_net_min=("test_sharpe_net", "min"),
        sharpe_net_max=("test_sharpe_net", "max"))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
