"""Optional (needs a GPU): turn NoLBERT into a sentence encoder for our news, contrastively.

    python scripts/03c_finetune_encoder.py [--budget-min 150] [--max-pairs 240000]
                                           [--train-end 2019-12-31] [--grad-ckpt]
    python scripts/03_embed.py --encoder data/topics/encoders/NoLBERT-news
    python scripts/04_fit_topics.py embedding --embeddings NoLBERT-news_d150 --name nolbert

NoLBERT (Kakhbod and Li 2025, `alikLab/NoLBERT`) is a DeBERTa-v3 model trained only on
text from 1976 to 1995, so it cannot have read anything about our sample. It is a
masked-language model, not a sentence encoder: mean-pooled, its vectors are anisotropic
(unrelated articles have a cosine similarity of about 0.5) and cluster poorly. This
script adapts it with the recipe of the run behind the committed topics (FINDINGS.md,
section 9):

* InfoNCE with in-batch negatives over positive pairs, temperature 0.05, mean pooling of
  the last layer, L2 normalisation, symmetric loss;
* every parameter trained (or only the top --unfreeze-top blocks), AdamW with learning
  rate 2e-5 (3e-5 for top blocks only), weight decay 0.01, betas (0.9, 0.98), 5% linear
  warm-up then cosine decay, gradient clipping at 1, bf16 autocast (fp16 overflows in
  DeBERTa's disentangled attention), optional gradient checkpointing;
* length-bucketed batches of 32 pairs, a wall-clock budget instead of epochs,
  checkpoints every 15 minutes that are loadable (save_pretrained) and resumable
  (--resume), and on a CUDA out-of-memory error the batch is halved and the pass
  restarted;
* in-batch top-1 retrieval on 2,048 held-out pairs before and after (chance 1/32).

Pairs come only from articles dated up to --train-end, by default the end of the
training period (config.SPLITS): the encoder then knows nothing of validation or test
text. (The run behind the committed topics drew its pairs from articles published up to
2020; see REFERENCE.md.) One pair per article, so a positive is never an in-batch
negative. The default pair is (headline, lead) from data/news/. With --chunks and
--doc-meta the pairs are two random chunks of the same article's full text, as in that
run (chunks.parquet with doc_index and chunk, doc_meta.parquet with date by row number).

Measured on one RTX 3060: 150 minutes and 6,205 steps lift held-out top-1 retrieval from
0.08 to 0.89; embedding 1.26 million articles afterwards takes about 2.5 hours. On a CPU
neither is practical. Do not pass --center to 04_fit_topics.py for the adapted encoder:
the adaptation already removes the anisotropy that centring corrects.
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

from ftm import config
from ftm.data import news

NOLBERT = "alikLab/NoLBERT"
NOLBERT_REVISION = "0b0359c62b3f7031c8f0ffa9d323881e829a147b"
VALIDATION_PAIRS = 2048


def log(message: str, start=[time.time()]) -> None:
    print(f"[{(time.time() - start[0]) / 60:6.1f} min] {message}", flush=True)


def pair_texts(train_end: str, max_pairs: int, chunks: str | None, doc_meta: str | None,
               seed: int) -> tuple[list[str], list[str]]:
    """Two texts of the same article for up to max_pairs articles dated up to train_end."""
    rng = np.random.default_rng(seed)
    if chunks:
        meta = pd.read_parquet(doc_meta, columns=["date"])
        keep = np.flatnonzero(pd.to_datetime(meta["date"]).to_numpy() <= np.datetime64(train_end))
        frame = pd.read_parquet(chunks, columns=["doc_index", "chunk"])
        frame = frame[frame["doc_index"].isin(keep)].drop_duplicates("chunk")   # boilerplate
        a, b = [], []
        for _, rows in frame.groupby("doc_index").indices.items():
            if len(rows) >= 2:
                first, second = rng.permutation(rows)[:2]
                a.append(frame["chunk"].iat[first]); b.append(frame["chunk"].iat[second])
        order = rng.permutation(len(a))[:max_pairs]
        return [a[i] for i in order], [b[i] for i in order]
    index = news.load_articles(columns=["article_id", "date"])
    keep = (index["date"] <= train_end).to_numpy()
    rows = np.flatnonzero(keep)
    rows = np.sort(rng.choice(rows, min(len(rows), max_pairs * 2), replace=False))
    mask = np.zeros(len(index), dtype=bool)
    mask[rows] = True
    articles = news.load_articles(columns=["title", "lead"], keep=mask)
    articles = articles[(articles["title"].str.len() > 10) & (articles["lead"].str.len() > 50)]
    articles = articles.drop_duplicates("lead").sample(frac=1.0, random_state=seed).head(max_pairs)
    return articles["title"].tolist(), articles["lead"].tolist()


def tokenize(texts: list[str], tokenizer, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-padded token ids (n, max_len) uint16 and true lengths (n,)."""
    pad = tokenizer.pad_token_id or 0
    ids = np.full((len(texts), max_len), pad, dtype=np.uint16)
    lengths = np.zeros(len(texts), dtype=np.int16)
    for lo in range(0, len(texts), 2048):
        encoded = tokenizer(texts[lo:lo + 2048], truncation=True, max_length=max_len)["input_ids"]
        for j, sequence in enumerate(encoded):
            ids[lo + j, :len(sequence)] = sequence
            lengths[lo + j] = len(sequence)
    return ids, lengths


def batches(n_pairs: int, batch: int, lengths: np.ndarray, rng) -> list[np.ndarray]:
    """Pair indices in batches, sorted by length inside shuffled blocks of 50 batches."""
    order = rng.permutation(n_pairs)
    block = batch * 50
    order = np.concatenate([order[i:i + block][np.argsort(lengths[order[i:i + block]], kind="stable")]
                            for i in range(0, n_pairs, block)])
    return [order[i:i + batch] for i in range(0, n_pairs - batch + 1, batch)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default=NOLBERT)
    ap.add_argument("--revision", help="repository commit (default: the pinned NoLBERT commit)")
    ap.add_argument("--out", help="folder for the adapted encoder "
                                  "(default data/topics/encoders/<encoder>-news)")
    ap.add_argument("--train-end", default=config.SPLITS["train"][1],
                    help="pairs come from articles dated up to this day")
    ap.add_argument("--chunks", help="chunks.parquet of full-text chunks (needs --doc-meta)")
    ap.add_argument("--doc-meta", help="doc_meta.parquet with the date of every document")
    ap.add_argument("--max-pairs", type=int, default=240_000)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32, help="pairs per step (= in-batch negatives)")
    ap.add_argument("--lr", type=float, help="default 2e-5, or 3e-5 with --unfreeze-top")
    ap.add_argument("--unfreeze-top", type=int, default=0, help="0 trains every parameter")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--budget-min", type=float, default=150.0, help="wall-clock cap on training")
    ap.add_argument("--ckpt-min", type=float, default=15.0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.chunks and not a.doc_meta:
        raise SystemExit("--chunks needs --doc-meta")
    a.lr = a.lr or (3e-5 if a.unfreeze_top else 2e-5)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(config.SEED)
    cuda = a.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise SystemExit("no CUDA device: this script needs a GPU (or --device cpu, very slow)")
    revision = a.revision or (NOLBERT_REVISION if a.encoder == NOLBERT else None)
    out = Path(a.out) if a.out else config.TOPICS / "encoders" / (a.encoder.split("/")[-1] + "-news")
    out.mkdir(parents=True, exist_ok=True)

    first, second = pair_texts(a.train_end, a.max_pairs, a.chunks, a.doc_meta, config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(a.encoder, revision=revision)
    ids_a, len_a = tokenize(first, tokenizer, a.max_len)
    ids_b, len_b = tokenize(second, tokenizer, a.max_len)
    lengths = len_a.astype(np.int32) + len_b
    n_val = min(VALIDATION_PAIRS, len(first) // 20)
    log(f"{len(first):,} pairs up to {a.train_end} ({n_val} held out), "
        f"mean length {lengths.mean() / 2:.0f} tokens")

    model = AutoModel.from_pretrained(a.encoder, revision=revision).to(a.device)
    if a.unfreeze_top:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for block in model.encoder.layer[-a.unfreeze_top:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    steps = max(1, (len(first) - n_val) // a.batch)
    warmup = max(10, int(0.05 * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: s / warmup if s < warmup else
                                                 0.5 * (1 + math.cos(math.pi * (s - warmup) /
                                                                     max(1, steps - warmup))))
    state_file, step = out / "state.pt", 0
    if a.resume and state_file.exists():
        state = torch.load(state_file, map_location=a.device)
        model.load_state_dict(state["model"]); optimiser.load_state_dict(state["opt"])
        schedule.load_state_dict(state["sched"]); step = state["step"]
        log(f"resumed at step {step}")

    def embed(ids: np.ndarray) -> "torch.Tensor":
        x = torch.as_tensor(ids.astype(np.int64), device=a.device)
        mask = (x != (tokenizer.pad_token_id or 0)).long()
        hidden = model(input_ids=x, attention_mask=mask).last_hidden_state
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weight).sum(1) / weight.sum(1).clamp(min=1e-6)

    def info_nce(rows: np.ndarray):
        with torch.autocast("cuda", torch.bfloat16, enabled=cuda):
            za, zb = embed(ids_a[rows]), embed(ids_b[rows])
        za, zb = F.normalize(za.float(), dim=-1), F.normalize(zb.float(), dim=-1)
        logits = za @ zb.T / a.temperature
        target = torch.arange(len(rows), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        return loss, (logits.argmax(1) == target).float().mean()

    held_out = np.arange(n_val)

    def retrieval() -> float:
        model.eval()
        with torch.inference_mode():
            scores = [info_nce(held_out[i:i + a.batch])[1].item()
                      for i in range(0, n_val - a.batch + 1, a.batch)]
        model.train()
        return float(np.mean(scores)) if scores else float("nan")

    def save(final: bool = False) -> None:
        torch.save({"model": model.state_dict(), "opt": optimiser.state_dict(),
                    "sched": schedule.state_dict(), "step": step}, state_file)
        model.save_pretrained(out)
        if final:
            tokenizer.save_pretrained(out)

    before = retrieval()
    log(f"held-out in-batch top-1 before: {before:.4f} (chance {1 / a.batch:.4f})")
    rng = np.random.default_rng(config.SEED + step)
    train_rows = np.arange(n_val, len(first))
    history, started, saved, done = [], time.time(), time.time(), False
    model.train()
    while not done:
        restart = False
        for batch in batches(len(train_rows), a.batch, lengths[train_rows], rng):
            try:
                loss, accuracy = info_nce(train_rows[batch])
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                optimiser.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if a.batch <= 8:
                    raise SystemExit("out of memory at a batch of 8 pairs")
                a.batch //= 2
                log(f"out of memory: batch halved to {a.batch}, pass restarted")
                restart = True
                break
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step(); schedule.step(); optimiser.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                history.append({"step": step, "loss": round(loss.item(), 4),
                                "acc": round(accuracy.item(), 4), "lr": schedule.get_last_lr()[0]})
                log(f"step {step:6d}  loss {loss.item():.4f}  in-batch top-1 {accuracy.item():.3f}")
            if (time.time() - saved) / 60 >= a.ckpt_min:
                save(); saved = time.time(); log(f"checkpoint at step {step}")
            if (time.time() - started) / 60 >= a.budget_min:
                log(f"budget of {a.budget_min:g} minutes reached"); done = True
                break
        else:
            done = True                     # a full pass over the pairs
        if restart:
            continue
    after = retrieval()
    log(f"held-out in-batch top-1 after: {after:.4f} (before {before:.4f})")
    save(final=True)
    (out / "train_log.json").write_text(json.dumps(
        {"config": vars(a), "steps": step, "pairs_seen": step * a.batch,
         "val_top1_before": before, "val_top1_after": after, "history": history},
        indent=2, default=str))
    print(f"adapted encoder -> {out}")


if __name__ == "__main__":
    main()
