"""Nasdaq.com articles of the FinMultiTime data set (Xu et al., 2025), 2009 to April 2025.

`text/sp500_news.zip` on Hugging Face (3.6 GB) holds one JSON-lines file per
ticker with the fields Date, Article_title, Stock_symbol, Url and Article. For
2009 to 2023 it is the Nasdaq part of FNSPID (the article URLs coincide); the
rows of 2024 and 2025 come from a second scrape in April 2025, have no title
and leave a hole from 17 December 2023 to the end of January 2024.

`extract` writes shards with the layout of `ftm.data.fnspid` (index and bodies),
so the rest of the pipeline does not depend on which source was used.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import zipfile

import pandas as pd
import requests

from .fnspid import BODY_CHARS, LEAD_CHARS, MIN_BODY_CHARS, _hash, _write

URL = ("https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_"
       "TimeSeries-for-Financial-Time-Series-Forecasting/resolve/main/text/sp500_news.zip")
PARTS = 24


def download(path: str | Path, chunk: int = 1 << 24) -> Path:
    """Fetch the archive, resuming a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(requests.head(URL, allow_redirects=True, timeout=60).headers["Content-Length"])
    have = path.stat().st_size if path.exists() else 0
    if have < total:
        with requests.get(URL, headers={"Range": f"bytes={have}-"}, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "ab") as fh:
                for block in r.iter_content(chunk):
                    fh.write(block)
    if path.stat().st_size != total:
        raise IOError(f"{path} has {path.stat().st_size} of {total} bytes; run again to resume")
    return path


def title_from_url(url: pd.Series) -> pd.Series:
    """The headline as it appears in the article address (the 2024-25 rows have no title)."""
    slug = url.str.rstrip("/").str.rsplit("/", n=1).str[-1]
    return slug.str.replace(r"-\d+$", "", regex=True).str.replace("-", " ").str.strip()


def _extract_part(job) -> dict:
    archive, names, out, part, keep_bodies = job
    cols = {c: [] for c in ("date", "ticker", "title", "url", "body")}
    with zipfile.ZipFile(archive) as z:
        for name in names:
            with z.open(name) as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    cols["date"].append(str(d.get("Date") or "")[:10])
                    cols["ticker"].append(str(d.get("Stock_symbol") or "").strip().upper())
                    cols["title"].append(str(d.get("Article_title") or ""))
                    cols["url"].append(str(d.get("Url") or ""))
                    cols["body"].append(str(d.get("Article") or ""))
    f = pd.DataFrame(cols)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f[f["date"].notna() & (f["ticker"] != "")]
    untitled = f["title"].str.strip() == ""
    f.loc[untitled, "title"] = title_from_url(f.loc[untitled, "url"])
    rec = pd.DataFrame({
        "date": f["date"], "second_of_day": -1, "ticker": f["ticker"], "title": f["title"],
        "publisher": "", "url_hash": _hash(f["url"]),
        "title_hash": _hash(f["date"].dt.strftime("%Y-%m-%d") + "|" + f["title"].str.lower()),
        "body_chars": f["body"].str.len().astype("int32"),
        "lead": f["body"].str.slice(0, LEAD_CHARS), "source": "finmultitime"})
    rec["second_of_day"] = rec["second_of_day"].astype("int32")
    rec = rec[rec["body_chars"] > 0]
    name = f"finmultitime_{part:04d}of{PARTS:04d}"
    _write(rec, Path(out) / "index" / f"{name}.parquet")
    long = (rec["body_chars"] >= MIN_BODY_CHARS).to_numpy()
    if keep_bodies and long.any():
        bodies = rec.loc[long, ["date", "ticker", "title", "url_hash", "title_hash"]].assign(
            body=f.loc[rec.index[long], "body"].str.slice(0, BODY_CHARS))
        _write(bodies, Path(out) / "bodies" / f"{name}.parquet")
    status = {"source": "finmultitime", "part": part, "index_rows": int(len(rec)),
              "by_year": {str(y): {"with_body": int(n)}
                          for y, n in rec.groupby(rec["date"].dt.year).size().items()}}
    path = Path(out) / "status" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status))
    return status


def extract(archive: str | Path, out: str | Path, workers: int = 2,
            keep_bodies: bool = False) -> list[dict]:
    """Parse the archive into index (and optionally body) shards; finished parts are skipped."""
    with zipfile.ZipFile(archive) as z:
        infos = sorted((i for i in z.infolist() if i.filename.endswith(".jsonl")),
                       key=lambda i: -i.file_size)
    names = [i.filename for i in infos]
    jobs = [(str(archive), names[p::PARTS], str(out), p, keep_bodies) for p in range(PARTS)
            if not (Path(out) / "status" / f"finmultitime_{p:04d}of{PARTS:04d}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_extract_part, jobs))
