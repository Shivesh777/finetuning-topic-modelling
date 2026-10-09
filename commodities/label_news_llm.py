"""Label commodity news with Claude Haiku 5.5: which commodity, which direction, which driver.

The model reads only the headline and lead of each article, with the date removed,
and is told to judge what the article's information implies for the price, not to
recall what happened later. It also flags articles that only recap a price move that
already happened, since those carry no new information and would mechanically
"predict" the same day's return.

    python3 label_news_llm.py plan --sample 200      # what would be sent, rough cost
    python3 label_news_llm.py submit --sample 200    # pilot batch; read the labels by hand
    python3 label_news_llm.py submit                 # full batch
    python3 label_news_llm.py collect                # fetch results, write labels and daily signal

Uses the Message Batches API (half price). Credentials come from ANTHROPIC_API_KEY or an
`ant auth login` profile. Writes to data/commodities/.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

MODEL = "claude-haiku-5-5"

HERE = Path(__file__).resolve().parent
DATA = next(p for p in (HERE.parent, HERE.parent / "data") if (p / "crsp").exists())


def load_env(path=DATA / ".env"):
    """Put KEY=value lines of the git-ignored .env into os.environ (never printed)."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_env()
OUT = DATA / "commodities"
BATCH_FILE = OUT / "llm_batch_id.txt"

FUND_TICKERS = ["USO", "BNO", "UNG", "UGA", "DBO", "GLD", "IAU", "SLV", "PPLT", "PALL", "CPER",
                "DBA", "CORN", "WEAT", "SOYB", "DBC", "XLE", "XOP", "OIH", "GDX"]
TITLE_WORDS = r"(?i)\b(oil|crude|opec|brent|wti|gold|silver|natural gas|copper|platinum|palladium|wheat|corn|soybeans?|commodit\w*)"

COMMODITIES = ["crude_oil", "natural_gas", "gasoline", "heating_oil", "gold", "silver", "copper",
               "platinum", "palladium", "corn", "wheat", "soybeans", "cattle", "hogs"]
DRIVERS = ["supply", "demand", "inventory", "monetary_policy", "geopolitics", "weather",
           "regulation", "speculative_flows", "other"]

SYSTEM = f"""You label financial news for a research study on commodity prices.

For the article you are given, decide which commodities it carries information about and, for each,
whether that information points to a higher, lower, or unchanged price for the commodity.

Rules:
- Judge only the information in the article text. Do not use anything you may know about what
  happened to prices after the article was written, and do not guess the article's date.
- A sentence that merely reports a price move that already happened ("oil rose 2% today") is not
  information about the future. Set is_price_recap to true when the article is mainly such a recap.
- List a commodity only if the article says something specific about it. Equity articles about a
  producer (for example an oil company's earnings) count only if they state something about the
  commodity itself, such as output, costs of supply, or demand.
- confidence is 1 (weak or indirect), 2 (clear), or 3 (explicit and central to the article).
- If the article has no commodity information, return an empty list and relevant false.

Commodities: {", ".join(COMMODITIES)}.
Drivers: {", ".join(DRIVERS)}."""

SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "is_price_recap": {"type": "boolean"},
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "commodity": {"type": "string", "enum": COMMODITIES},
                    "direction": {"type": "string", "enum": ["up", "down", "neutral"]},
                    "driver": {"type": "string", "enum": DRIVERS},
                    "confidence": {"type": "integer", "enum": [1, 2, 3]},
                },
                "required": ["commodity", "direction", "driver", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relevant", "is_price_recap", "signals"],
    "additionalProperties": False,
}


def select_articles(start, end, sample):
    links = pd.read_parquet(DATA / "news" / "links.parquet")
    fund_ids = set(links.loc[links.ticker.isin(FUND_TICKERS), "article_id"])
    arts = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / "news" / "articles" / "*.parquet"))))
    arts = arts[(arts.date >= start) & (arts.date <= end)]
    keep = arts.article_id.isin(fund_ids) | arts.title.str.contains(TITLE_WORDS, regex=True, na=False)
    arts = arts[keep].drop_duplicates("article_id").reset_index(drop=True)
    if sample:
        arts = arts.sample(n=min(sample, len(arts)), random_state=0)
    return arts


def request_for(row):
    text = f"Headline: {row.title}\n\nLead: {row.lead}"
    return Request(
        custom_id=f"a{row.article_id}",
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=1024,  # thinking is on by default on Haiku 5.5 and counts toward this
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": text}],
        ),
    )


def plan(args):
    arts = select_articles(args.start, args.end, args.sample)
    chars = (arts.title.str.len() + arts.lead.str.len()).sum() + len(arts) * len(SYSTEM)
    in_tok, out_tok = chars / 4, len(arts) * 250
    cost = (in_tok * 0.10 + out_tok * 0.50) / 1e6 / 2  # batch is half price
    print(f"{len(arts):,} articles, {args.start} to {args.end}")
    print(f"~{in_tok / 1e6:.1f}M input and ~{out_tok / 1e6:.1f}M output tokens -> about ${cost:.2f} with the Batch API")
    print(arts.date.dt.year.value_counts().sort_index().to_string())
    return arts


def submit(args):
    arts = plan(args)
    client = anthropic.Anthropic()
    requests = [request_for(r) for r in arts.itertuples()]
    ids = []
    for i in range(0, len(requests), 50_000):  # batch limit is 100,000 requests / 256 MB
        batch = client.messages.batches.create(requests=requests[i:i + 50_000])
        ids.append(batch.id)
        print(f"Submitted {batch.id} ({len(requests[i:i + 50_000]):,} requests)")
    OUT.mkdir(parents=True, exist_ok=True)
    BATCH_FILE.write_text("\n".join(ids))


def collect(args):
    client = anthropic.Anthropic()
    rows, failed = [], 0
    for batch_id in BATCH_FILE.read_text().split():
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            print(f"{batch_id} still {batch.processing_status}: {batch.request_counts}")
            return
        for res in client.messages.batches.results(batch_id):
            msg = res.result.message if res.result.type == "succeeded" else None
            if msg is None or msg.stop_reason != "end_turn":
                failed += 1
                continue
            out = json.loads(next(b.text for b in msg.content if b.type == "text"))
            aid = int(res.custom_id[1:])
            for s in out["signals"] or [{}]:
                rows.append({"article_id": aid, "relevant": out["relevant"],
                             "is_price_recap": out["is_price_recap"], **s})
    labels = pd.DataFrame(rows)
    print(f"{labels.article_id.nunique():,} articles labelled, {failed} failed or refused (resubmit those)")

    arts = select_articles("1900-01-01", "2100-01-01", 0)[["article_id", "date"]]
    labels = labels.merge(arts, on="article_id")
    labels.to_parquet(OUT / "llm_labels.parquet", index=False)

    # Daily signal per commodity: confidence-weighted (up - down) over articles that are not recaps.
    s = labels.dropna(subset=["commodity"])
    s = s[~s.is_price_recap]
    s["score"] = s.direction.map({"up": 1, "down": -1, "neutral": 0}) * s.confidence
    daily = s.groupby(["date", "commodity"]).agg(score=("score", "sum"), n_articles=("article_id", "nunique"))
    daily.reset_index().to_parquet(OUT / "llm_daily_signal.parquet", index=False)
    print(f"Wrote {OUT / 'llm_labels.parquet'} and llm_daily_signal.parquet")
    print(s.groupby(["commodity", "direction"]).size().unstack(fill_value=0).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "submit", "collect"])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--sample", type=int, default=0, help="label a random sample first (pilot)")
    args = ap.parse_args()
    {"plan": plan, "submit": submit, "collect": collect}[args.step](args)


if __name__ == "__main__":
    main()
