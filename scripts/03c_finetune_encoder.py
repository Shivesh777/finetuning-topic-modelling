"""Optional (needs a GPU): adapt an encoder to the news of the training period.

    python scripts/03c_finetune_encoder.py [--encoder alikLab/NoLBERT] [--pairs 100000]
                                           [--epochs 1] [--device cuda]
    python scripts/03_embed.py --encoder data/topics/encoders/NoLBERT-news
    python scripts/04_fit_topics.py embedding --embeddings NoLBERT-news_d150 --name nolbert_ft

NoLBERT (Kakhbod and Li, 2025) is a language model trained only on text from 1976 to
1995, so it cannot have read anything about our sample. It is a masked-language model,
not a sentence encoder: the mean of its token vectors puts all articles close together
(average cosine similarity of about 0.7 between unrelated articles). This script turns it
into a sentence encoder without giving up that property: it is trained to match each
article's HEADLINE to its own LEAD against the other articles of the batch (multiple
negatives ranking loss), using only articles dated up to the end of the training period
(config.SPLITS). Settings follow the course material: AdamW, learning rate 2e-5, warm-up
over the first 6 percent of the steps, mean pooling.

On a CPU NoLBERT encodes about 5 to 10 articles a second, so both this script and the
embedding that follows need a GPU: about 10 minutes for 100,000 pairs on one.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ftm import config
from ftm.data import news
from ftm.topics import attention

NOLBERT = "alikLab/NoLBERT"
NOLBERT_REVISION = "0b0359c62b3f7031c8f0ffa9d323881e829a147b"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default=NOLBERT)
    ap.add_argument("--revision", help="repository commit (default: the pinned NoLBERT commit)")
    ap.add_argument("--out", help="folder for the adapted encoder")
    ap.add_argument("--pairs", type=int, default=100_000, help="headline-lead pairs to train on")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=config.ENCODER_MAX_TOKENS)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)

    index = news.load_articles(columns=["article_id", "date"])
    keep = attention.sample_per_day(index["date"], config.ATTENTION_ARTICLES_PER_DAY, config.SEED)
    keep &= (index["date"] <= config.SPLITS["train"][1]).to_numpy()      # training period only
    rows = np.flatnonzero(keep)
    rows = np.sort(np.random.default_rng(config.SEED).choice(rows, min(a.pairs, len(rows)),
                                                             replace=False))
    keep[:] = False
    keep[rows] = True
    articles = news.load_articles(columns=["article_id", "title", "lead"], keep=keep)
    articles = articles[(articles["title"].str.len() > 10) & (articles["lead"].str.len() > 50)]
    pairs = Dataset.from_dict({"anchor": articles["title"].tolist(),
                               "positive": articles["lead"].tolist()})
    revision = a.revision or (NOLBERT_REVISION if a.encoder == NOLBERT else None)
    model = SentenceTransformer(a.encoder, device=a.device, revision=revision)
    model.max_seq_length = a.max_tokens
    out = Path(a.out) if a.out else config.TOPICS / "encoders" / (a.encoder.split("/")[-1] + "-news")
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, learning_rate=2e-5, warmup_ratio=0.06,
        weight_decay=0.01, seed=config.SEED, save_strategy="no", report_to="none",
        logging_steps=50)
    SentenceTransformerTrainer(model=model, args=arguments, train_dataset=pairs,
                               loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(str(out))
    print(f"{len(pairs):,} pairs up to {config.SPLITS['train'][1]} -> {out}")


if __name__ == "__main__":
    main()
