"""Build the stock-period panels in data/panels/<frequency>/.

    python scripts/05_build_panel.py [--freq M 2W W] [--topics lda minilm ...] [--skip-price]
                                     [--placebo lda ...]
    python scripts/05_build_panel.py --shock-window 1 --freq W [--topics ...]
    python scripts/05_build_panel.py --tone [--freq ...]

Each panel has one row per (formation day, investable stock) with the next
period's return, and feature blocks stored as cross-sectional ranks (see
ftm/features/blocks.py): price, and for each topic model narr_<model> and
firm_<model>; --placebo adds narr_<model>_placebo.

<model> is a folder of data/topics/ (scripts/04_fit_topics.py). Existing blocks are kept, so a new topic
model only adds its two blocks. Needs data/crsp/ (scripts/00_prepare_crsp.py).
"""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from ftm import config
from ftm.data.returns import Market
from ftm.features import blocks
from ftm.panel import Panel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--topics", nargs="*", default=None,
                    help="topic models to add (default: every folder of data/topics)")
    ap.add_argument("--skip-price", action="store_true")
    ap.add_argument("--placebo", nargs="*", default=[],
                    help="topic models for which to add narr_<model>_placebo as well")
    ap.add_argument("--shock-window", type=int,
                    help="add only narr_<model>_s<W> blocks: shocks over W days (05b_tune_shocks)")
    ap.add_argument("--tone", action="store_true",
                    help="add only the tone_fb, tone_lm, emb_fb and npx blocks (03d_sentiment)")
    a = ap.parse_args()
    topics = a.topics
    if topics is None:
        topics = sorted(p.parent.name for p in config.TOPICS.glob("*/daily_sums.parquet"))
    market = Market.load()
    if a.shock_window:
        makers = {}
        for model in topics:
            name, maker, names = blocks.narrative_maker(market, model, a.shock_window)
            makers[name] = (maker, names)
    elif not a.tone:
        makers = blocks.makers(market, topics, a.placebo, with_price=not a.skip_price)
    for freq in a.freq:
        t0 = time.time()
        if a.tone:
            makers = blocks.tone_makers(market, freq)
        folder = config.PANELS / freq
        panel = Panel.skeleton(market, freq)
        if (folder / "index.parquet").exists():
            old = pd.read_parquet(folder / "index.parquet", columns=["date", "permno"])
            if not old.equals(panel.index[["date", "permno"]]):
                raise SystemExit(f"{folder} holds a panel with other rows; delete it to rebuild")
        for name, (maker, names) in makers.items():
            panel.add_block(name, maker, names)
        panel.save(folder)
        index = panel.index
        print(f"{freq}: {index['date'].nunique()} periods, {len(index):,} rows, "
              f"{index.groupby('date').size().median():.0f} stocks per period, blocks "
              f"{list(panel.blocks)} ({time.time() - t0:.0f} s)", flush=True)
        print(index.groupby("split")["date"].agg(["nunique", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
