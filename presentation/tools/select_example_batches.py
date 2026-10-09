#!/usr/bin/env python3
"""Two example batches of 32 article pairs for slides 3.1 and 3.3, chosen by a fixed rule before
any model is run, so that the displayed batch is not picked on its results.

    FTM_DATA=/path/to/data python tools/select_example_batches.py [--out /path/to/data/news/example_batches.json]

The original pair cache of the adaptation run (two chunks of one article, on the teammate's
machine) is not available here, so each pair is (headline, lead) of one article from
data/news/articles, the pair the repo's own 03c_finetune_encoder.py uses by default:

  train_example    articles dated 1 Jan 2015 to 31 Aug 2020, the period the adaptation drew its pairs from
  heldout_example  articles dated 1 Sep 2020 or later, after the adaptation's pair cutoff

Rule, applied identically to both windows: lead of at least 300 characters, headline of at least
30 characters, filed under at least one ticker, one article per distinct lead; the first 32 by
article_id. The output holds article text, so it stays in the data folder, out of git.
"""
import argparse
import json
import os
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = Path(os.environ.get("FTM_DATA", REPO / "data"))
WINDOWS = {"train_example": ("2015-01-01", "2020-08-31"), "heldout_example": ("2020-09-01", "2024-12-31")}
RULE = ("date inside the window; lead >= 300 characters; headline >= 30 characters; n_tickers >= 1; "
        "one article per distinct lead; the first 32 by article_id")


def pick(con, lo, hi, n=32):
    q = f"""
        select article_id, strftime(date, '%Y-%m-%d') as date, title, lead from (
            select *, row_number() over (partition by lead order by article_id) as rn
            from read_parquet('{DATA}/news/articles/*.parquet')
            where date >= '{lo}' and date <= '{hi} 23:59:59'
              and length(lead) >= 300 and length(title) >= 30 and n_tickers >= 1)
        where rn = 1 order by article_id limit {n}"""
    return con.execute(q).fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "news" / "example_batches.json"))
    a = ap.parse_args()
    con = duckdb.connect()
    out = {"rule": RULE, "pair": "(headline, lead) of one article", "batches": {}}
    for name, (lo, hi) in WINDOWS.items():
        rows = pick(con, lo, hi)
        assert len(rows) == 32, f"{name}: {len(rows)} articles"
        out["batches"][name] = {"window": [lo, hi], "ids": [int(r[0]) for r in rows], "dates": [r[1] for r in rows],
                                "pairs": [[r[2], r[3]] for r in rows]}
        print(f"{name}: 32 articles, {rows[0][1]} to {rows[-1][1]}, ids {rows[0][0]}..{rows[-1][0]}")
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
