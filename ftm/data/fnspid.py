"""Streaming extraction of the FNSPID news files (Dong, Fan and Peng, 2024).

Hugging Face hosts two CSV files: `nasdaq_exteral_data.csv` (23 GB, Nasdaq.com
articles with bodies) and `All_external.csv` (5.7 GB, headlines of other
publishers). Both are read by HTTP range requests, slice by slice, without
storing the raw file. A slice owns the records that start inside its byte
range and reads past its end to complete the last one, so every record is
parsed exactly once and the output does not depend on the number of slices.

Per slice the module writes

    index/<file>_<slice>.parquet    every record with a valid date, a ticker and
                                    an article body: date, time of day, ticker,
                                    title, publisher, two hashes for
                                    de-duplication, the length of the body and
                                    its first LEAD_CHARS characters
    bodies/<file>_<slice>.parquet   records whose body has MIN_BODY_CHARS or
                                    more, body truncated to BODY_CHARS
    status/<file>_<slice>.json      record counts by year

A slice whose status file exists is skipped, so an interrupted run resumes.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import io
import json
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .. import config

BASE_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/"
FILES = {"nasdaq": "nasdaq_exteral_data.csv", "external": "All_external.csv"}
COLUMNS = ("Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article")
# A record starts on a new line with an optional numeric index and a UTC timestamp.
RECORD = re.compile(rb"\n(?:\d*(?:\.\d*)?,)?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC,")
TICKER = r"[A-Z][A-Z0-9.\-]{0,7}"
SLACK = 4 << 20                 # bytes read past a slice to finish its last record
LEAD_CHARS = config.LEAD_CHARS
BODY_CHARS = 4000
MIN_BODY_CHARS = config.MIN_BODY_CHARS
BLOCK_BYTES = 16 << 20          # records are parsed in pieces of this size


def remote_size(url: str) -> int:
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    return int(head.headers["Content-Length"])


def _get(session, url: str, start: int, end: int, tries: int = 6) -> bytes:
    """Bytes [start, end) of the remote file."""
    for attempt in range(tries):
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=180)
            r.raise_for_status()
            if len(r.content) != end - start:
                raise IOError(f"short read: {len(r.content)} of {end - start} bytes")
            return r.content
        except (requests.RequestException, IOError):
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def read_slice(session, url: str, total: int, i: int, n: int) -> bytes:
    """The complete records that start in slice i of n (without the header line)."""
    lo, hi = i * total // n, (i + 1) * total // n
    start = max(lo - 1, 0)               # one byte back: a record may start exactly at lo
    end = min(hi + SLACK, total)
    buf = _get(session, url, start, end)
    if i == 0:
        first = buf.index(b"\n") + 1     # skip the header line
    else:
        m = RECORD.search(buf)
        if m is None:
            return b""
        first = m.start() + 1
    while True:                           # first record that starts at or after hi
        m = RECORD.search(buf, max(hi - start - 1, 0))
        if m is not None or end == total:
            break
        more = min(end + SLACK, total)
        buf += _get(session, url, end, more)
        end = more
    cut = len(buf) if m is None else m.start() + 1
    return buf[first:cut] if first < cut else b""


def _read_csv(header: bytes, body: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(header + body), dtype=str, engine="c",
                       usecols=lambda c: c in COLUMNS, on_bad_lines="skip",
                       encoding="utf-8", encoding_errors="replace")


def parse_records(header: bytes, body: bytes) -> tuple[pd.DataFrame, int]:
    """Parse a block of records; returns the frame and the number of records
    dropped. A record with unbalanced quotes would swallow its neighbours, so a
    block whose row count differs from its record count is bisected until the
    offending records are isolated and dropped."""
    starts = [0] + [m.start() + 1 for m in RECORD.finditer(body)]

    def parse(a: int, b: int) -> tuple[list[pd.DataFrame], int]:
        lo = starts[a]
        hi = starts[b] if b < len(starts) else len(body)
        try:
            frame = _read_csv(header, body[lo:hi])
        except (pd.errors.ParserError, ValueError):
            frame = None
        if frame is not None and len(frame) == b - a:
            return [frame], 0
        if b - a == 1:
            return [], 1
        mid = (a + b) // 2
        left, bad_left = parse(a, mid)
        right, bad_right = parse(mid, b)
        return left + right, bad_left + bad_right

    frames, bad = parse(0, len(starts))
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS)), bad
    return pd.concat(frames, ignore_index=True), bad


def _hash(strings: pd.Series) -> np.ndarray:
    return pd.util.hash_array(strings.fillna("").to_numpy(dtype=object), categorize=False)


def transform(frame: pd.DataFrame, source: str):
    """(index, bodies, counts by year) of one parsed block."""
    for col in COLUMNS:
        if col not in frame:
            frame[col] = None
    stamp = frame["Date"].fillna("")
    frame = frame[stamp.str.slice(0, 10).str.fullmatch(r"\d{4}-\d{2}-\d{2}")]
    stamp = frame["Date"]
    date = stamp.str.slice(0, 10)
    clock = [pd.to_numeric(stamp.str.slice(a, a + 2), errors="coerce") for a in (11, 14, 17)]
    body = frame["Article"].fillna("")
    title = frame["Article_title"].fillna("")
    rec = pd.DataFrame({
        "date": pd.to_datetime(date, errors="coerce"),
        "second_of_day": (clock[0] * 3600 + clock[1] * 60 + clock[2]).fillna(-1).astype("int32"),
        "ticker": frame["Stock_symbol"].fillna("").str.strip(),
        "title": title,
        "publisher": frame["Publisher"].fillna(""),
        "url_hash": _hash(frame["Url"]),
        "title_hash": _hash(date + "|" + title),
        "body_chars": body.str.len().astype("int32"),
        "lead": body.str.slice(0, LEAD_CHARS),
        "source": source,
        "body": body.str.slice(0, BODY_CHARS),
    })
    rec = rec[rec["date"].notna()]
    has_ticker = rec["ticker"].str.fullmatch(TICKER)
    long_body = rec["body_chars"] >= MIN_BODY_CHARS
    counts = pd.DataFrame({
        "records": 1, "with_ticker": has_ticker, "with_body": has_ticker & (rec["body_chars"] > 0),
        "with_long_body": has_ticker & long_body}).groupby(rec["date"].dt.year).sum()
    rec = rec[has_ticker]
    index = rec.drop(columns="body")
    bodies = rec.loc[long_body[has_ticker],
                     ["date", "ticker", "title", "url_hash", "title_hash", "body"]]
    return index, bodies, counts


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if "date" in table.column_names:                    # calendar date, not a timestamp
        table = table.set_column(table.column_names.index("date"), "date",
                                 table.column("date").cast(pa.date32()))
    pq.write_table(table, path, compression="zstd", compression_level=3)


def _blocks(body: bytes, size: int = BLOCK_BYTES):
    """Split a run of records into consecutive pieces of about `size` bytes."""
    start = 0
    while start < len(body):
        m = RECORD.search(body, start + size) if start + size < len(body) else None
        end = len(body) if m is None else m.start() + 1
        yield body[start:end]
        start = end


def process_slice(source: str, i: int, n: int, out: str, total: int, header: bytes,
                  keep_bodies: bool = True, bodies_only: bool = True) -> dict:
    """Download, parse and store slice i of n of one file; returns its status.

    bodies_only: keep in the index only the records that have an article body
    (the other publishers' headline-only rows are counted but not stored)."""
    out = Path(out)
    name = f"{source}_{i:04d}of{n:04d}"
    status_path = out / "status" / f"{name}.json"
    if status_path.exists():
        return json.loads(status_path.read_text())
    t0 = time.time()
    with requests.Session() as session:
        raw = read_slice(session, BASE_URL + FILES[source], total, i, n)
    t1 = time.time()
    parsed = dropped = 0
    index, bodies, counts = [], [], []
    for block in _blocks(raw):                    # bounded memory whatever the row size
        frame, bad = parse_records(header, block)
        parsed, dropped = parsed + len(frame), dropped + bad
        idx, bod, cnt = transform(frame, source)
        index.append(idx[idx["body_chars"] > 0] if bodies_only else idx)
        bodies.append(bod)
        counts.append(cnt)
    index, bodies = pd.concat(index, ignore_index=True), pd.concat(bodies, ignore_index=True)
    counts = pd.concat(counts).groupby(level=0).sum()
    if len(index):
        _write(index, out / "index" / f"{name}.parquet")
    if keep_bodies and len(bodies):
        _write(bodies, out / "bodies" / f"{name}.parquet")
    status = {"source": source, "slice": i, "slices": n, "bytes": len(raw),
              "records_parsed": int(parsed), "records_dropped": int(dropped),
              "index_rows": int(len(index)), "body_rows": int(len(bodies)),
              "seconds_download": round(t1 - t0, 1), "seconds_total": round(time.time() - t0, 1),
              "by_year": {str(y): {k: int(v) for k, v in row.items()}
                          for y, row in counts.to_dict("index").items()}}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status))
    return status


def run(source: str, out: str | Path, n_slices: int = 64, workers: int = 2,
        keep_bodies: bool = True, budget: float | None = None, verbose: bool = True) -> list[dict]:
    """Process every slice of one file that is not done yet.

    budget: stop handing out new slices after this many seconds (rerun to continue)."""
    url = BASE_URL + FILES[source]
    total = remote_size(url)
    with requests.Session() as session:
        head = _get(session, url, 0, 4096)
    header = head[:head.index(b"\n") + 1]
    out = Path(out)
    todo = [i for i in range(n_slices)
            if not (out / "status" / f"{source}_{i:04d}of{n_slices:04d}.json").exists()]
    t0, done = time.time(), []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending, queue = set(), iter(todo)
        while True:
            while len(pending) < workers and not (budget and time.time() - t0 > budget):
                i = next(queue, None)
                if i is None:
                    break
                pending.add(pool.submit(process_slice, source, i, n_slices, str(out), total,
                                        header, keep_bodies))
            if not pending:
                break
            finished = next(as_completed(pending))
            pending.remove(finished)
            status = finished.result()
            done.append(status)
            if verbose:
                print(f"[{source}] slice {status['slice']:3d}/{n_slices}: "
                      f"{status['records_parsed']:,} records, {status['records_dropped']} dropped, "
                      f"{status['seconds_total']:.0f} s ({time.time() - t0:.0f} s elapsed)",
                      flush=True)
    return done


def counts_by_year(out: str | Path) -> pd.DataFrame:
    """Record counts by source and year, summed over the finished slices."""
    rows = []
    for path in sorted((Path(out) / "status").glob("*.json")):
        status = json.loads(path.read_text())
        for year, counts in status["by_year"].items():
            rows.append({"source": status["source"], "year": int(year), **counts})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["source", "year"]).sum().reset_index()
