"""De-duplicate the extracted index into the article table, and repair its text.

    python scripts/02_build_articles.py [--source finmultitime] [--no-repair]

Writes data/news/articles/<quarter>.parquet (one row per unique article: date,
title, lead), data/news/links.parquet (article, ticker) and, in
results/data_audit/, news_coverage.csv (articles per year and per weekday) and
text_repair.csv (how many articles had the body of another article, by year).

The repair needs the FNSPID extraction as well (scripts/01_download_news.py
--source fnspid): see ftm/data/news.py. Without it about 7% of the articles
carry the text of an unrelated article.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from ftm import config
from ftm.data import news


def coverage(articles: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    days = articles.groupby("date").size().reindex(
        pd.date_range(articles["date"].min(), articles["date"].max()), fill_value=0)
    weekdays = days[days.index.dayofweek < 5]
    by_year = weekdays.groupby(weekdays.index.year)
    year = articles["date"].dt.year
    link_year = year.to_numpy()[links["article_id"].to_numpy()]
    return pd.DataFrame({
        "articles": articles.groupby(year).size(),
        "article_ticker_pairs": pd.Series(link_year).value_counts().sort_index(),
        "tickers": links.groupby(link_year)["ticker"].nunique(),
        "weekday_median": by_year.median(),
        "weekday_p10": by_year.quantile(0.10),
        "weekdays_without_news": by_year.apply(lambda s: int((s < 20).sum())),
    }).rename_axis("year")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="finmultitime")
    ap.add_argument("--no-repair", action="store_true",
                    help="keep the source's text although some of it is wrong")
    a = ap.parse_args()
    raw, reference = config.RAW / a.source, config.RAW / "fnspid"
    if not (a.no_repair or (reference / "index").exists()):
        raise SystemExit("The text repair needs the FNSPID extraction: run\n"
                         "  python scripts/01_download_news.py --source fnspid\n"
                         "or pass --no-repair (7% of the articles then have a wrong text).")
    keys = news.load_keys(raw, a.source)
    articles, links = news.deduplicate(keys)
    articles = news.attach_text(articles, raw, a.source)
    out = config.RESULTS / "data_audit"
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_repair:
        articles, report, damaged = news.repair_text(articles, keys, raw, a.source, reference)
        report.to_csv(out / "text_repair.csv")
        print(f"{len(damaged)} damaged ticker files: {' '.join(damaged)}")
        print(report.to_string())
    config.NEWS.mkdir(parents=True, exist_ok=True)
    news.save_articles(articles, config.NEWS / "articles")
    links.to_parquet(config.NEWS / "links.parquet", compression="zstd", index=False)
    table = coverage(articles, links)
    table.to_csv(out / "news_coverage.csv")
    print(f"{len(articles):,} articles, {len(links):,} article-ticker pairs")
    print(table.to_string())


if __name__ == "__main__":
    main()
