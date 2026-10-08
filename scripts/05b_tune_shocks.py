"""Choose the window of the attention shocks on the validation period.

    python scripts/05b_tune_shocks.py [--freq M 2W W] [--windows 1 5 21 63]
                                      [--models ols ridge lasso] [--budget SECONDS]

An attention shock (ftm/features/narrative.py) is a topic's attention minus its mean
over the previous `window` trading days; the main run used config.SHOCK_WINDOW = 5
without trying anything else. For every frequency, topic model and window this
script computes the narrative exposures exactly as the panel does, fits each model
on the training period with its hyperparameters chosen on validation, and records
the rank IC of the validation forecasts (a period with a flat forecast counts as
zero). Nothing of the test period is computed. Window 5 reproduces the validation
forecasts of the narr_<model> experiments of scripts/06_run_models.py.

Writes results/tuning/shock_window.csv; finished rows are kept, so the script can be
stopped (--budget) and run again.
"""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config
from ftm.data.returns import Market
from ftm.evaluation import metrics
from ftm.features import narrative
from ftm.models.tuning import get_model, target_of, tune
from ftm.panel import Panel
from ftm.topics import attention

OUT = config.RESULTS / "tuning" / "shock_window.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 5, 21, 63])
    ap.add_argument("--models", nargs="+", default=["ols", "ridge", "lasso"])
    ap.add_argument("--topics", nargs="*", help="topic models (default: all)")
    ap.add_argument("--budget", type=float, help="start nothing new after this many seconds")
    a = ap.parse_args()
    t0 = time.time()
    topics = a.topics or sorted(p.parent.name for p in config.TOPICS.glob("*/daily_sums.parquet"))
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(
        columns=["freq", "topics", "window", "model"])
    seen = set(zip(done["freq"], done["topics"], done["window"], done["model"]))
    market = Market.load()
    rows = []
    for freq in a.freq:
        index = pd.read_parquet(config.PANELS / freq / "index.parquet")
        panel = Panel(freq, index)
        period = index["period"].to_numpy()
        train, val = panel.rows("train"), panel.rows("validation")
        val_dates = index.loc[val, "date"]
        for topic in topics:
            todo = [w for w in a.windows if any((freq, topic, w, m) not in seen for m in a.models)]
            if not todo:
                continue
            sums = attention.load_daily_sums(config.TOPICS / topic / "daily_sums.parquet")
            theta = narrative.trading_day_attention(sums, market.dates)
            names = [f"narr_{topic}_t{j:02d}" for j in range(theta.shape[1])]
            for window in todo:
                if a.budget and time.time() - t0 > a.budget:
                    break
                t1 = time.time()
                shocks = narrative.attention_shocks(theta, window)
                # the block's name seeds the donor draws: the same name as in the panel
                panel.add_block(f"narr_{topic}", lambda f, s=shocks: narrative.exposures(market, s, f),
                                names)
                # stored panels hold half precision (Panel.save): round the same way
                X = panel.blocks[f"narr_{topic}"].astype(np.float16).astype(np.float32)
                for model in a.models:
                    if (freq, topic, window, model) in seen:
                        continue
                    cls = get_model(model)
                    y = target_of(cls, index["y_rel"].to_numpy(), period)
                    _, _, pred = tune(cls, X[train], y[train], X[val], y[val],
                                      config.TUNE_TOLERANCE, groups=period[train])
                    ic = metrics.rank_ic(index.loc[val, "y_rel"].to_numpy(), pred,
                                         val_dates.to_numpy())
                    ic = ic.reindex(np.unique(val_dates.to_numpy())).fillna(0.0)
                    s = metrics.ic_summary(ic.to_numpy())
                    rows.append({"freq": freq, "topics": topic, "window": window, "model": model,
                                 "periods": len(ic), "ic_mean": s["mean"], "ic_t": s["t"],
                                 "seconds": round(time.time() - t1, 1)})
                print(f"{freq:>2} {topic:<10} window {window:>2}: " + ", ".join(
                    f"{r['model']} {r['ic_mean']:+.4f}" for r in rows[-len(a.models):]), flush=True)
    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT, index=False)
    table = pd.read_csv(OUT)
    print(table.pivot_table(index=["freq", "window"], columns="model", values="ic_mean").round(4))
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
