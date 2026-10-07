"""Topic weights from a ready-made text classifier.

The FinBERT finance-topic classifier assigns a text to one of twenty fixed
subjects ("Earnings", "Fed | Central Banks", "M&A | Investments", ...). Used as
a topic model, an article's topic weights are simply its class probabilities:
nothing is estimated on our data. Two caveats belong to every result made with
it: the classifier was published in December 2022, fine-tuned on a set of
finance tweets released that September, so it is NOT point-in-time for our
sample, and its twenty subjects were chosen by its authors, not learned from
this corpus.

The model is run with ONNX Runtime from the quantised export that its author
publishes, which needs no login and no PyTorch.
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np

FINBERT_TOPICS = "nickmuchi/optimum-finbert-tone-finetuned-finance-topic-classification"


def load(repo: str = FINBERT_TOPICS, threads: int | None = None):
    """(tokenizer, ONNX session, labels) of a sequence classifier exported to ONNX."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    import onnxruntime as ort

    tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    labels = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())["id2label"]
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(hf_hub_download(repo, "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    return tokenizer, session, [labels[str(i)] for i in range(len(labels))]


def probabilities(texts: list[str], tokenizer, session, max_tokens: int,
                  batch_size: int = 64) -> np.ndarray:
    """Class probabilities (n, classes) float32; texts are cut to max_tokens tokens."""
    tokenizer.enable_truncation(max_length=max_tokens)
    tokenizer.no_padding()
    encoded = tokenizer.encode_batch(texts)
    order = np.argsort([len(e.ids) for e in encoded], kind="stable")     # pad little
    wanted = {i.name for i in session.get_inputs()}
    out = None
    for lo in range(0, len(texts), batch_size):
        rows = order[lo:lo + batch_size]
        width = max(len(encoded[i].ids) for i in rows)
        ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for j, i in enumerate(rows):
            n = len(encoded[i].ids)
            ids[j, :n], mask[j, :n] = encoded[i].ids, 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        logits = session.run(None, {k: v for k, v in feed.items() if k in wanted})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        if out is None:
            out = np.empty((len(texts), p.shape[1]), dtype=np.float32)
        out[rows] = p / p.sum(axis=1, keepdims=True)
    return out


def classify(texts, out: str | Path, repo: str = FINBERT_TOPICS, max_tokens: int = 64,
             block: int = 2_000, threads: int | None = None, budget: float | None = None,
             verbose: bool = True) -> np.ndarray:
    """Class probabilities of every text, written to `out` (.npy, float16), resumable.

    Mirrors ftm.topics.embed.encode: `out`.done holds the number of finished rows and
    budget stops the run before that many seconds have passed. `out`.labels.json lists
    the classes."""
    out = Path(out)
    done_path = out.with_suffix(".done")
    tokenizer, session, labels = load(repo, threads)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".labels.json").write_text(json.dumps(labels))
    n = len(texts)
    if out.exists():
        array = np.lib.format.open_memmap(out, mode="r+")
        if array.shape != (n, len(labels)):
            raise ValueError(f"{out} has shape {array.shape}, expected {(n, len(labels))}")
    else:
        array = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                          shape=(n, len(labels)))
    start = int(done_path.read_text()) if done_path.exists() else 0
    t0 = time.time()
    for lo in range(start, n, block):
        per_block = (time.time() - t0) / max(1, (lo - start) // block)
        if budget and lo > start and time.time() - t0 + per_block > budget:
            break
        chunk = list(texts[lo:lo + block])
        array[lo:lo + len(chunk)] = probabilities(chunk, tokenizer, session, max_tokens)
        array.flush()
        done_path.write_text(str(lo + len(chunk)))
        if verbose and (lo // block % 10 == 9 or lo + block >= n):
            rate = (lo + len(chunk) - start) / (time.time() - t0)
            print(f"classified {lo + len(chunk):,} of {n:,} ({rate:.0f} texts/s)", flush=True)
    return array
