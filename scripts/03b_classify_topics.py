"""Optional: topic weights from the FinBERT finance-topic classifier.

    python scripts/03b_classify_topics.py [--per-day 150] [--all] [--budget SECONDS]
    python scripts/04_fit_topics.py weights --weights finbert-topic_d150 --name finbert

Runs the classifier of ftm/topics/classify.py on the HEADLINE of each article of
the per-day attention sample (the classifier was trained on tweets, which read
like headlines) and writes data/topics/embeddings/<tag>.npy: twenty class
probabilities per article, with <tag>.ids.npy and <tag>.labels.json. About 170
headlines a second on two CPU cores; the run resumes when restarted.

The classifier was published at the end of 2022: results that use it are not
point-in-time (see FINDINGS.md, section 3).
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ftm import config
from ftm.data import news
from ftm.topics import attention, classify

MAX_TOKENS = 32


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=classify.FINBERT_TOPICS, help="Hugging Face repository")
    ap.add_argument("--tag", default="finbert-topic")
    ap.add_argument("--per-day", type=int, default=config.ATTENTION_ARTICLES_PER_DAY)
    ap.add_argument("--all", action="store_true", help="classify every article")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--budget", type=float, help="stop before this many seconds have passed")
    a = ap.parse_args()
    index = news.load_articles(columns=["article_id", "date"])
    keep = (np.ones(len(index), dtype=bool) if a.all
            else attention.sample_per_day(index["date"], a.per_day, config.SEED))
    out = config.TOPICS / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag + ("_all" if a.all else f"_d{a.per_day}")
    np.save(out / f"{tag}.ids.npy", index["article_id"].to_numpy()[keep])
    titles = news.load_articles(columns=["article_id", "title"], keep=keep)["title"].fillna("")
    print(f"{len(titles):,} of {len(index):,} headlines -> {out / tag}.npy", flush=True)
    classify.classify(titles.tolist(), out / f"{tag}.npy", a.model, MAX_TOKENS,
                      threads=a.threads, budget=a.budget)


if __name__ == "__main__":
    main()
