"""Forecast combination across topic models: one forecast that uses every reader at once.

    python scripts/06d_grand_ensemble.py [--freq M 2W W] [--windows 5 63]

For each frequency the ensembles of scripts/06b_ensemble.py on the price-and-text sets of
the six topic models (all_<model> for the 5-day attention shocks of the main run,
all_<model>_s<w> for a tuned window) are turned into within-period ranks and averaged with
equal weights; the same for their text-only sets (text_<model>, text_<model>_s<w>), which
says what the news gives without the price features. Nothing is estimated and nothing is chosen: every base model and every
topic model enters with the same weight, so the combination needs no validation period
and cannot overfit one. Stored as the forecasts of the feature sets "grand_s<w>" and
"grand_text_s<w>", model "ens" (experiments.kind "grand"), and evaluated like any other
experiment.

Added in the final run (10 October), after its test results had shown that the tuned
63-day window and the eight-model ensemble each improve most experiments: a
consequence of those two validated choices, not a search, but reported as such.

Outputs
    data/predictions/<freq>/grand_s<w>__ens.{validation,test}.npy
    data/predictions/<freq>/grand_text_s<w>__ens.{validation,test}.npy
"""
from pathlib import Path
import argparse
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config, experiments

TOPICS = ("lda", "lda_all", "minilm", "finbert", "nolbert", "nolbert20")
spec = importlib.util.spec_from_file_location("ens", ROOT / "scripts" / "06b_ensemble.py")
combine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combine)
combine = combine.combine


def members(prefix: str, window: int) -> list[str]:
    return [f"{prefix}_{m}" if window == config.SHOCK_WINDOW else f"{prefix}_{m}_s{window}"
            for m in TOPICS]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--windows", type=int, nargs="+", default=[config.SHOCK_WINDOW, 63])
    a = ap.parse_args()
    for freq in a.freq:
        index = pd.read_parquet(config.PANELS / freq / "index.parquet", columns=["date", "split"])
        for prefix, name, w in [(p, n, w) for w in a.windows
                                for p, n in (("all", f"grand_s{w}"), ("text", f"grand_text_s{w}"))]:
            sets = members(prefix, w)
            paths = {f: {s: experiments.prediction_path(freq, f, experiments.ENSEMBLE, s)
                         for s in experiments.EVALUATED} for f in sets}
            missing = [f for f in sets if not all(p.exists() for p in paths[f].values())]
            if missing:
                print(f"{freq} {name}: ensembles missing for {', '.join(missing)}")
                continue
            for split in experiments.EVALUATED:
                period = index.loc[index["split"] == split, "date"].to_numpy()
                out = combine([np.load(paths[f][split]).astype(np.float64) for f in sets], period)
                np.save(experiments.prediction_path(freq, name, experiments.ENSEMBLE, split), out)
            print(f"{freq} {name}: {len(sets)} ensembles combined", flush=True)


if __name__ == "__main__":
    main()
