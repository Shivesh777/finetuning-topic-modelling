"""Forecasts without a fitted model: single tone features, and the combination of the
price ensemble with the best of them.

    python scripts/06c_signals.py [--freq M 2W W] [--force]

signals   experiments.SIGNALS: model name -> (block, feature, sign). The forecast is the
          feature as the panel stores it (its cross-sectional rank) times the sign, which
          comes from the literature, not from our data: good news and more attention
          before higher returns, a move without news reversing, a move with news
          drifting (Chan 2003). Stocks without news coverage hold another stock's
          features in the panel, as in every experiment, so they are placed at random.
          Stored under the feature set "tone_fb" or "tone_lm" (the block of the
          feature), or "px_tone" for the signals that use past returns.
combo     the equal-weighted average of the within-period ranks of the price ensemble
          (price, ens) and of the tone signal (sig_fb*, sig_lm*) with the highest
          validation rank IC at that frequency: price and news combined without fitting
          anything on them together. Stored as px_tone, model "combo".
best10    the equal-weighted average of the within-period ranks of the ten experiments of
          the base models (any real feature set) with the highest validation rank IC: the
          validation choice spread over ten experiments instead of staked on one. Stored
          as px_pool, model "best10" (experiments.POOL); run it after every experiment
          that should be a candidate.

Nothing here looks at the test period: the one choice, the signal of the combination,
is made on validation. Run after scripts/06_run_models.py and 06b_ensemble.py.

Outputs
    data/predictions/<freq>/<features>__<signal>.{validation,test}.npy
    results/tuning/<freq>/signals.csv      validation rank IC of every signal
    results/tuning/<freq>/pool.csv         validation rank IC of every candidate of best10
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
from ftm.panel import Panel
from ftm.models.tuning import ranks_within

FEATURES_OF = {"tone_fb": "tone_fb", "tone_lm": "tone_lm", "npx": "px_tone"}


def validation_ic(y, pred, dates) -> float:
    ic = metrics.rank_ic(y, pred, dates)
    return float(ic.reindex(np.unique(dates)).fillna(0.0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for freq in a.freq:
        names = json.loads((config.PANELS / freq / "names.json").read_text())["names"]
        blocks = sorted({b for b, _, _ in experiments.SIGNALS.values()})
        if not all(b in names for b in blocks):
            print(f"{freq}: tone blocks missing; run scripts/05_build_panel.py --tone")
            continue
        panel = Panel.load(config.PANELS / freq, blocks)
        index = panel.index
        val = panel.rows("validation")
        rows, made = [], 0
        for signal, (block, feature, sign) in experiments.SIGNALS.items():
            features = FEATURES_OF[block]
            column = sign * panel.blocks[block][:, names[block].index(feature)].astype(np.float64)
            for split in experiments.EVALUATED:
                path = experiments.prediction_path(freq, features, signal, split)
                if a.force or not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, column[panel.rows(split)].astype(np.float32))
                    made += 1
            rows.append({"signal": signal, "features": features, "block": block,
                         "feature": feature, "sign": sign,
                         "validation_ic": validation_ic(index.loc[val, "y_rel"].to_numpy(),
                                                        column[val], index.loc[val, "date"].to_numpy())})
        table = pd.DataFrame(rows)
        tone = table[table["signal"].str.match(r"sig_(fb|lm)")]
        best = tone.loc[tone["validation_ic"].idxmax()]
        table["combined"] = table["signal"] == best["signal"]
        (config.RESULTS / "tuning" / freq).mkdir(parents=True, exist_ok=True)
        table.to_csv(config.RESULTS / "tuning" / freq / "signals.csv", index=False)
        combo = {s: experiments.prediction_path(freq, "px_tone", experiments.COMBO, s)
                 for s in experiments.EVALUATED}
        inputs = [experiments.prediction_path(freq, f, m, s) for s in experiments.EVALUATED
                  for f, m in (("price", experiments.ENSEMBLE), (best["features"], best["signal"]))]
        newest = max(p.stat().st_mtime for p in inputs)
        for split in experiments.EVALUATED:
            if not a.force and combo[split].exists() and combo[split].stat().st_mtime >= newest:
                continue
            period = index.loc[index["split"] == split, "date"].to_numpy()
            price = np.load(experiments.prediction_path(freq, "price", experiments.ENSEMBLE, split))
            news = np.load(experiments.prediction_path(freq, best["features"], best["signal"], split))
            ranks = [np.nan_to_num(ranks_within(f.astype(np.float64), period), nan=0.0)
                     for f in (price, news)]
            np.save(combo[split], np.mean(ranks, axis=0).astype(np.float32))
        print(f"{freq}: {made} signal forecasts written; combination with {best['signal']} "
              f"(validation IC {best['validation_ic']:+.4f})", flush=True)
        pool(freq, index, a.force)


def pool(freq: str, index: pd.DataFrame, force: bool) -> None:
    """Writes px_pool__best10 from the base-model experiments that exist now."""
    candidates = [(f, m) for f, m in experiments.finished(freq)
                  if m in experiments.BASE_MODELS and experiments.kind(f) != "placebo"]
    paths = [experiments.prediction_path(freq, f, m, "validation") for f, m in candidates]
    out = {s: experiments.prediction_path(freq, "px_pool", experiments.POOL, s)
           for s in experiments.EVALUATED}
    newest = max(p.stat().st_mtime for p in paths)
    if not force and all(p.exists() and p.stat().st_mtime >= newest for p in out.values()):
        return
    val = index[index["split"] == "validation"]
    table = pd.DataFrame([{"features": f, "model": m, "validation_ic": validation_ic(
        val["y_rel"].to_numpy(), np.load(p).astype(np.float64), val["date"].to_numpy())}
        for (f, m), p in zip(candidates, paths)])
    table = table.sort_values("validation_ic", ascending=False, ignore_index=True)
    table["chosen"] = table.index < experiments.POOL_SIZE
    table.to_csv(config.RESULTS / "tuning" / freq / "pool.csv", index=False)
    for split in experiments.EVALUATED:
        period = index.loc[index["split"] == split, "date"].to_numpy()
        ranks = [np.nan_to_num(ranks_within(np.load(experiments.prediction_path(
            freq, f, m, split)).astype(np.float64), period), nan=0.0)
            for f, m in table.loc[table["chosen"], ["features", "model"]].itertuples(index=False)]
        np.save(out[split], np.mean(ranks, axis=0).astype(np.float32))
    print(f"{freq}: best10 of {len(table)} experiments, validation IC "
          f"{table.loc[table['chosen'], 'validation_ic'].min():+.4f} to "
          f"{table['validation_ic'].max():+.4f}", flush=True)


if __name__ == "__main__":
    main()
