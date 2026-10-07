"""Sentence embeddings of the articles that enter daily attention.

    python scripts/03_embed.py [--encoder NAME] [--per-day 150] [--all] [--budget SECONDS]

Embeds a per-day random sample of articles (or all of them with --all) with a
Hugging Face encoder and writes data/topics/embeddings/<tag>.npy (float16,
unit rows) and <tag>.ids.npy (their article ids). The run resumes when
restarted. On a laptop CPU the default encoder does 150 to 600 articles a
second; a GPU or Apple-silicon device is 10 to 50 times faster (--device mps).
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ftm import config
from ftm.data import news
from ftm.topics import attention, embed


def tag_of(encoder: str, per_day: int | None) -> str:
    return encoder.rstrip("/").split("/")[-1] + ("_all" if per_day is None else f"_d{per_day}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default=config.ENCODER)
    ap.add_argument("--per-day", type=int, default=config.ATTENTION_ARTICLES_PER_DAY)
    ap.add_argument("--all", action="store_true", help="embed every article")
    ap.add_argument("--max-tokens", type=int, default=config.ENCODER_MAX_TOKENS)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: automatic)")
    ap.add_argument("--budget", type=float, help="stop after this many seconds")
    ap.add_argument("--revision", help="commit of the encoder's repository to pin")
    a = ap.parse_args()
    per_day = None if a.all else a.per_day
    index = news.load_articles(columns=["article_id", "date"])
    keep = (np.ones(len(index), dtype=bool) if per_day is None
            else attention.sample_per_day(index["date"], per_day, config.SEED))
    out = config.TOPICS / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    tag = tag_of(a.encoder, per_day)
    np.save(out / f"{tag}.ids.npy", index["article_id"].to_numpy()[keep])
    texts = news.text_of(news.load_articles(columns=["article_id", "title", "lead"],
                                            keep=keep)).tolist()
    print(f"{len(texts):,} of {len(index):,} articles -> {out / tag}.npy", flush=True)
    embed.encode(texts, a.encoder, a.max_tokens, out / f"{tag}.npy", batch_size=a.batch_size,
                 device=a.device, budget=a.budget, revision=a.revision)


if __name__ == "__main__":
    main()
