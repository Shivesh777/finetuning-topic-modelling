"""Download the news and extract the article index.

    python scripts/01_download_news.py [--source finmultitime | fnspid] [--workers 2]
                                       [--budget SECONDS]

Both sources are needed (scripts/02_build_articles.py explains why):

finmultitime (default)  one 3.6 GB archive of Nasdaq.com articles, 2009 to April 2025:
                        the list of articles and their tickers. The archive can be
                        deleted afterwards.
fnspid                  the same articles up to December 2023 with the right bodies,
                        read by range requests without storing the 23 GB file.

Everything goes to the cache folder, outside the repository: ~/.cache/ftm, or the
folder named by the environment variable FTM_CACHE. Both runs resume when restarted.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm import config
from ftm.data import finmultitime, fnspid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["finmultitime", "fnspid"], default="finmultitime")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--slices", type=int, default=256, help="fnspid: number of byte ranges")
    ap.add_argument("--budget", type=float, help="fnspid: stop starting slices after SECONDS")
    ap.add_argument("--bodies", action="store_true", help="also store truncated article bodies")
    a = ap.parse_args()
    out = config.RAW / a.source
    if a.source == "finmultitime":
        archive = finmultitime.download(out / "sp500_news.zip")
        done = finmultitime.extract(archive, out, workers=a.workers, keep_bodies=a.bodies)
        print(f"{sum(s['index_rows'] for s in done):,} rows extracted in this run -> {out}")
    else:
        fnspid.run("nasdaq", out, n_slices=a.slices, workers=a.workers, budget=a.budget,
                   keep_bodies=a.bodies)
        counts = fnspid.counts_by_year(out)
        print(counts.set_index("year")[["records", "with_body", "with_long_body"]].to_string())


if __name__ == "__main__":
    main()
