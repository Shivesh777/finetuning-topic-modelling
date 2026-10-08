"""Forecast combination: the equal-weighted average of the base models' forecasts.

    python scripts/06b_ensemble.py [--freq M 2W W] [--force]

For every frequency and feature set whose eight base models (experiments.BASE_MODELS)
have all been run, the forecasts of each model are turned into cross-sectional ranks
scaled to [-1, 1] in every period, so that no model weighs more because its forecasts are
larger, and averaged with equal weights. A model whose forecast is flat in a period adds
zero. Nothing is estimated and nothing is chosen: no weights, no tuning, no selection of
members. The combination is stored as the forecasts of a model named "ens"
(experiments.ENSEMBLE) for the validation and the test rows, and scripts 07 and 08
evaluate it like any other experiment.

Combining forecasts is the oldest way to reduce the noise of single models (Bates and
Granger 1969; for returns Rapach, Strauss and Zhou 2010). Here it also removes the
choice of one model per feature set, which the validation period makes badly (FINDINGS.md,
section 4.2).

Outputs
    data/predictions/<freq>/<features>__ens.{validation,test}.npy
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config, experiments
from ftm.models.tuning import ranks_within


def combine(forecasts: list[np.ndarray], period: np.ndarray) -> np.ndarray:
    """Average of the within-period ranks of several forecasts of the same rows, float32."""
    ranks = [np.nan_to_num(ranks_within(f, period), nan=0.0) for f in forecasts]
    return np.mean(ranks, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--force", action="store_true", help="combine again even if up to date")
    a = ap.parse_args()
    for freq in a.freq:
        index = pd.read_parquet(config.PANELS / freq / "index.parquet", columns=["date", "split"])
        stems = {f"{f}__{m}" for f, m in experiments.finished(freq)}
        feature_sets = sorted({stem.split("__")[0] for stem in stems})
        made, missing = 0, []
        for features in feature_sets:
            members = [experiments.prediction_path(freq, features, m, "test")
                       for m in experiments.BASE_MODELS]
            if not all(p.exists() for p in members):
                missing.append(features)
                continue
            out = {s: experiments.prediction_path(freq, features, experiments.ENSEMBLE, s)
                   for s in experiments.EVALUATED}
            newest = max(p.stat().st_mtime for m in experiments.BASE_MODELS
                         for p in (experiments.prediction_path(freq, features, m, s)
                                   for s in experiments.EVALUATED))
            if not a.force and all(p.exists() and p.stat().st_mtime >= newest for p in out.values()):
                continue
            for split in experiments.EVALUATED:
                period = index.loc[index["split"] == split, "date"].to_numpy()
                forecasts = [np.load(experiments.prediction_path(freq, features, m, split))
                             .astype(np.float64) for m in experiments.BASE_MODELS]
                np.save(out[split], combine(forecasts, period))
            made += 1
        print(f"{freq}: {made} ensembles written"
              + (f"; members missing for {', '.join(missing)}" if missing else ""), flush=True)


if __name__ == "__main__":
    main()
