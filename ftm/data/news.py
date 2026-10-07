"""The article table: one row per unique article, and the tickers it is filed under.

The sources list an article once for every ticker page it appears on. Rows
are merged by address and then by (date, headline), so an article counts once
in topic attention while `links` keeps every (article, ticker) pair for the
firm-level features. Dates are calendar dates without a reliable time of day.

Text repair. In FinMultiTime the files of 93 tickers (alphabetically A to AFG)
have the right date, headline and address but the BODY OF ANOTHER ARTICLE,
often one written years later. FNSPID, which holds the same articles up to
2023, has the right bodies. `repair_text` therefore takes the lead of every
article from FNSPID where FNSPID has the address, and for the others (2024)
uses a ticker file that is intact or, failing that, leaves the lead empty so
that the article is represented by its headline alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import config

KEY_COLUMNS = ["date", "ticker", "url_hash", "title_hash", "body_chars"]


def _shards(raw: str | Path, source: str) -> list[Path]:
    files = sorted((Path(raw) / "index").glob(f"{source}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no index shards of {source} under {raw}")
    return files


def load_keys(raw: str | Path, source: str) -> pd.DataFrame:
    """The small columns of every index row, with the row's shard and position."""
    frames = []
    for shard, path in enumerate(_shards(raw, source)):
        frame = pq.read_table(path, columns=KEY_COLUMNS).to_pandas()
        frame["shard"] = np.int16(shard)
        frame["row"] = np.arange(len(frame), dtype=np.int32)
        frames.append(frame)
    keys = pd.concat(frames, ignore_index=True)
    keys["date"] = pd.to_datetime(keys["date"])
    return keys


def deduplicate(keys: pd.DataFrame, start: str = config.NEWS_START, end: str = config.NEWS_END,
                bad_days=config.NEWS_BAD_DAYS, min_body_chars: int = config.MIN_BODY_CHARS):
    """(articles, links) from the index keys.

    articles: article_id, date, n_tickers, shard, row (where its text is stored),
    url_hash and text_ticker (the address and ticker file of that row), sorted by
    date; links: article_id, ticker. A row is kept if its body has
    min_body_chars or more, its date lies in [start, end] and is not one of the
    source's dump days. Rows with the same address are one article, and so are
    rows with the same date and headline under different addresses."""
    keep = ((keys["body_chars"] >= min_body_chars) & (keys["date"] >= start)
            & (keys["date"] <= end) & ~keys["date"].isin(pd.to_datetime(list(bad_days))))
    rows = keys.loc[keep].sort_values(["date", "url_hash", "ticker"], kind="stable")
    by_url = rows.groupby("url_hash", sort=False)["title_hash"].transform("first")
    key = by_url.groupby(by_url, sort=False).ngroup().to_numpy()
    first = ~pd.Series(key).duplicated().to_numpy()
    order = np.argsort(rows["date"].to_numpy()[first], kind="stable")
    new_id = np.empty(first.sum(), dtype=np.int64)
    new_id[order] = np.arange(first.sum())
    links = pd.DataFrame({"article_id": new_id[key], "ticker": rows["ticker"].to_numpy()})
    links = links.drop_duplicates().sort_values(["article_id", "ticker"], ignore_index=True)
    articles = rows.loc[first, ["date", "shard", "row", "url_hash", "ticker"]].assign(
        article_id=new_id).rename(columns={"ticker": "text_ticker"})
    articles = articles.sort_values("article_id", ignore_index=True)
    articles["n_tickers"] = links.groupby("article_id").size().to_numpy().astype(np.int32)
    return articles[["article_id", "date", "n_tickers", "shard", "row", "url_hash",
                     "text_ticker"]], links


def _read_rows(files: list[Path], shard: np.ndarray, row: np.ndarray, columns: list[str]):
    """Columns of the given (shard, row) positions, as object arrays in the given order."""
    out = {c: np.empty(len(shard), dtype=object) for c in columns}
    for s in np.unique(shard):
        where = np.flatnonzero(shard == s)
        picked = pq.read_table(files[s], columns=columns).take(row[where])
        for c in columns:
            out[c][where] = picked.column(c).to_pylist()
    return out


def attach_text(articles: pd.DataFrame, raw: str | Path, source: str) -> pd.DataFrame:
    """Adds title and lead, read shard by shard for the selected rows only."""
    text = _read_rows(_shards(raw, source), articles["shard"].to_numpy(),
                      articles["row"].to_numpy(), ["title", "lead"])
    return articles.assign(title=text["title"], lead=text["lead"])


def _same(a: pd.Series, b: pd.Series, chars: int = 150) -> pd.Series:
    """Whether two texts start alike, ignoring white space."""
    squeeze = lambda t: t.fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str[:chars]
    return squeeze(a) == squeeze(b)


def reference_leads(raw: str | Path, source: str, wanted: np.ndarray) -> pd.Series:
    """Lead by address hash from another extraction, for the addresses in `wanted`."""
    wanted = np.unique(wanted)
    parts = []
    for path in _shards(raw, source):
        frame = pq.read_table(path, columns=["url_hash", "lead"]).to_pandas()
        frame = frame[np.isin(frame["url_hash"].to_numpy(), wanted)]
        parts.append(frame.drop_duplicates("url_hash"))
    leads = pd.concat(parts, ignore_index=True).drop_duplicates("url_hash")
    return leads.set_index("url_hash")["lead"]


def repair_text(articles: pd.DataFrame, keys: pd.DataFrame, raw: str | Path, source: str,
                reference_raw: str | Path, reference: str = "nasdaq", threshold: float = 0.5):
    """(articles with repaired leads, per-year report, damaged ticker files). See the module text.

    articles: output of attach_text, still with url_hash and text_ticker. reference: the
    file prefix of the FNSPID shards that hold article bodies ("nasdaq")."""
    truth = articles["url_hash"].map(reference_leads(reference_raw, reference,
                                                     articles["url_hash"].to_numpy()))
    known = truth.notna()
    wrong = known & ~_same(articles["lead"], truth)
    by_file = pd.DataFrame({"n": known, "wrong": wrong}).groupby(
        articles["text_ticker"].to_numpy()).sum()
    damaged = by_file.index[(by_file["n"] >= 20) & (by_file["wrong"] > threshold * by_file["n"])]
    lead = articles["lead"].where(~known, truth)

    # Articles the reference does not have, whose text came from a damaged file.
    orphan = (~known & articles["text_ticker"].isin(damaged)).to_numpy()
    rows = keys[keys["url_hash"].isin(articles.loc[orphan, "url_hash"])
                & ~keys["ticker"].isin(damaged)].drop_duplicates("url_hash")
    spare = pd.Series(_read_rows(_shards(raw, source), rows["shard"].to_numpy(),
                                 rows["row"].to_numpy(), ["lead"])["lead"],
                      index=rows["url_hash"].to_numpy(), dtype=object)
    other = articles.loc[orphan, "url_hash"].map(spare)
    lead[orphan] = other.fillna("").to_numpy()
    year = articles["date"].dt.year
    report = pd.DataFrame({
        "articles": year.groupby(year).size(),
        "in_reference": known.groupby(year).sum(),
        "wrong_text_replaced": wrong.groupby(year).sum(),
        "text_from_another_ticker_file": pd.Series(orphan & other.reindex(
            articles.index).notna().to_numpy()).groupby(year.to_numpy()).sum(),
        "headline_only": pd.Series(orphan & ~other.reindex(
            articles.index).notna().to_numpy()).groupby(year.to_numpy()).sum(),
    }).rename_axis("year")
    return articles.assign(lead=lead), report, list(damaged)


def save_articles(articles: pd.DataFrame, folder: str | Path) -> None:
    """Write the article table as one file per quarter (each a few MB)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    articles = articles[["article_id", "date", "title", "lead", "n_tickers"]]
    quarter = articles["date"].dt.to_period("Q").astype(str)
    for name, part in articles.groupby(quarter.to_numpy(), sort=True):
        part.to_parquet(folder / f"{name}.parquet", compression="zstd", index=False)


def load_articles(folder: str | Path = None, columns=None, keep=None) -> pd.DataFrame:
    """The article table, sorted by article_id (which is also date order).

    keep: boolean array indexed by article_id; only these articles are loaded
    (the text of a million articles does not fit comfortably in a laptop's memory)."""
    folder = Path(folder) if folder is not None else config.NEWS / "articles"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no article files in {folder}; run scripts/02_build_articles.py")
    frames = []
    for path in files:
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame if keep is None else frame[keep[frame["article_id"].to_numpy()]])
    frame = pd.concat(frames, ignore_index=True)
    if "article_id" in frame:
        frame = frame.sort_values("article_id", ignore_index=True)
    return frame


def text_of(articles: pd.DataFrame) -> pd.Series:
    """The text unit used by every topic model: headline plus the lead of the body."""
    return articles["title"].fillna("") + ". " + articles["lead"].fillna("")
