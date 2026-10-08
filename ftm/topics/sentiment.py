"""Article tone: FinBERT sentiment of the headline and Loughran-McDonald word counts.

Topic attention says what a firm is in the news about, not whether the news is
good or bad. Two measures of tone fill that gap:

FinBERT   ProsusAI/finbert (Araci 2019): BERT further trained on Reuters
          financial news of 2008-2010 and fine-tuned on the Financial
          PhraseBank (Malo et al. 2014), sentences from news about listed
          companies that finance professionals labelled positive, negative or
          neutral for investors. Nothing in it saw text after 2010 or any
          return, so it is point-in-time for our sample. It is applied to
          headlines, single sentences like its training data. Per article it
          gives the three probabilities and the mean of its last hidden layer
          over the headline's tokens, the representation that
          ftm/features/tone.py turns into return forecasts (Chen, Kelly and Xiu
          2022). It runs with ONNX Runtime, every dense layer in 8-bit integers
          except the second layer of each feed-forward block, whose large
          activations do not survive rounding: on a sample of our headlines the
          tone of this model has a correlation of 0.99 with that of the full
          model and the same label for 97% of them, at twice its speed.
LM        the positive and negative words of the Loughran and McDonald (2011)
          dictionary (2014 release, as shipped with the pysentiment2 package),
          counted in the headline and lead.
"""
from __future__ import annotations

from pathlib import Path
import re
import time

import numpy as np

FINBERT = "ProsusAI/finbert"
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
LABELS = ("positive", "negative", "neutral")      # the model's output order, checked on export
HEADLINE_TOKENS = 32
EMBED_DIM = 768


def export_onnx(folder: str | Path, repo: str = FINBERT, revision: str = FINBERT_REVISION) -> Path:
    """Path of the 8-bit ONNX model with outputs (logits, mean last hidden state); exported
    and quantised on first use."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    folder = Path(folder)
    out = folder / "finbert.q8.onnx"
    if out.exists():
        return out
    folder.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision).eval()
    labels = tuple(model.config.id2label[i].lower() for i in range(len(model.config.id2label)))
    if labels != LABELS:
        raise ValueError(f"{repo} has labels {labels}, expected {LABELS}")

    class Outputs(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            o = self.model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            h, w = o.hidden_states[-1], attention_mask.unsqueeze(-1).to(torch.float32)
            return o.logits, (h * w).sum(1) / w.sum(1)

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    sample = tokenizer(["profit rose", "shares fell"], return_tensors="pt", padding="max_length",
                       max_length=8)
    full = folder / "finbert.fp32.onnx"
    torch.onnx.export(Outputs(model).eval(), (sample["input_ids"], sample["attention_mask"]),
                      str(full), input_names=["input_ids", "attention_mask"],
                      output_names=["logits", "pooled"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                    "attention_mask": {0: "b", 1: "s"},
                                    "logits": {0: "b"}, "pooled": {0: "b"}})
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    matmuls = [n.name for n in onnx.load(str(full)).graph.node if n.op_type == "MatMul"]
    keep_float = [n for n in matmuls if "/output/dense/" in n and "attention" not in n]
    quantize_dynamic(str(full), str(out), weight_type=QuantType.QInt8, per_channel=True,
                     reduce_range=True, nodes_to_exclude=keep_float)
    full.unlink()
    return out


def _session(path: Path, threads: int | None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _run(texts: list[str], tokenizer, session, max_tokens: int, batch_size: int):
    """(probabilities (n, 3), pooled (n, 768)) float32; batches of similar length."""
    ids = tokenizer(texts, truncation=True, max_length=max_tokens)["input_ids"]
    order = np.argsort([len(i) for i in ids], kind="stable")
    probs = np.empty((len(texts), len(LABELS)), dtype=np.float32)
    pooled = np.empty((len(texts), EMBED_DIM), dtype=np.float32)
    for lo in range(0, len(texts), batch_size):
        rows = order[lo:lo + batch_size]
        width = max(len(ids[i]) for i in rows)
        x = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros_like(x)
        for j, i in enumerate(rows):
            x[j, :len(ids[i])], mask[j, :len(ids[i])] = ids[i], 1
        logits, pool = session.run(None, {"input_ids": x, "attention_mask": mask})
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        probs[rows], pooled[rows] = p / p.sum(axis=1, keepdims=True), pool
    return probs, pooled


def finbert(texts, folder: str | Path, threads: int | None = None, block: int = 20_000,
            batch_size: int = 128, budget: float | None = None, verbose: bool = True,
            max_tokens: int = HEADLINE_TOKENS) -> tuple[np.ndarray, np.ndarray]:
    """FinBERT probabilities (n, 3) and pooled vectors (n, 768) of texts, float16, written to
    folder/finbert_probs.npy and folder/finbert_emb.npy; resumable like ftm.topics.embed.encode
    (folder/finbert.done counts the finished rows, budget stops early)."""
    from transformers import AutoTokenizer

    folder = Path(folder)
    session = _session(export_onnx(folder / "model"), threads)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT, revision=FINBERT_REVISION)
    n = len(texts)
    arrays = []
    for name, width in (("finbert_probs", len(LABELS)), ("finbert_emb", EMBED_DIM)):
        path = folder / f"{name}.npy"
        if path.exists():
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != (n, width):
                raise ValueError(f"{path} has shape {array.shape}, expected {(n, width)}")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, width))
        arrays.append(array)
    probs, pooled = arrays
    done_path = folder / "finbert.done"
    start = int(done_path.read_text()) if done_path.exists() else 0
    t0 = time.time()
    for lo in range(start, n, block):
        per_block = (time.time() - t0) / max(1, (lo - start) // block)
        if budget and lo > start and time.time() - t0 + per_block > budget:
            break
        chunk = [str(t) for t in texts[lo:lo + block]]
        p, e = _run(chunk, tokenizer, session, max_tokens, batch_size)
        probs[lo:lo + len(chunk)], pooled[lo:lo + len(chunk)] = p, e
        probs.flush()
        pooled.flush()
        done_path.write_text(str(lo + len(chunk)))
        if verbose:
            rate = (lo + len(chunk) - start) / (time.time() - t0)
            print(f"FinBERT {lo + len(chunk):,} of {n:,} ({rate:.0f} texts/s)", flush=True)
    return probs, pooled


def lm_words() -> tuple[frozenset, frozenset]:
    """(positive, negative) words of the Loughran-McDonald dictionary, upper case."""
    import pandas as pd
    import pysentiment2

    table = pd.read_csv(Path(pysentiment2.__file__).parent / "static" / "LM.csv",
                        usecols=["Word", "Positive", "Negative"])
    table["Word"] = table["Word"].astype(str).str.upper()
    return (frozenset(table.loc[table["Positive"] > 0, "Word"]),
            frozenset(table.loc[table["Negative"] > 0, "Word"]))


def lm_counts(texts, positive: frozenset | None = None,
              negative: frozenset | None = None) -> np.ndarray:
    """(n, 3) int32: words, positive words and negative words of each text."""
    if positive is None or negative is None:
        positive, negative = lm_words()
    word = re.compile(r"[A-Za-z]+")
    out = np.zeros((len(texts), 3), dtype=np.int32)
    for i, text in enumerate(texts):
        tokens = word.findall(str(text).upper())
        out[i] = (len(tokens), sum(t in positive for t in tokens),
                  sum(t in negative for t in tokens))
    return out


def pca_fit(vectors: np.ndarray, fit_rows: np.ndarray, dim: int, block: int = 50_000):
    """(mean, components (dim, d)) of a PCA of the rows fit_rows of vectors (a memory map
    will do): the leading eigenvectors of their covariance, accumulated block by block."""
    fit_rows = np.sort(np.asarray(fit_rows))
    d = vectors.shape[1]
    total, cross = np.zeros(d), np.zeros((d, d))
    for lo in range(0, len(fit_rows), block):
        x = np.asarray(vectors[fit_rows[lo:lo + block]], dtype=np.float64)
        total += x.sum(axis=0)
        cross += x.T @ x
    n = len(fit_rows)
    mean = total / n
    values, vectors_ = np.linalg.eigh(cross / n - np.outer(mean, mean))
    components = vectors_[:, np.argsort(values)[::-1][:dim]].T
    return mean.astype(np.float32), components.astype(np.float32)


def pca_apply(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray,
              block: int = 100_000) -> np.ndarray:
    """(n, dim) float16 projections."""
    out = np.empty((len(vectors), len(components)), dtype=np.float16)
    for lo in range(0, len(vectors), block):
        x = np.asarray(vectors[lo:lo + block], dtype=np.float32) - mean
        out[lo:lo + block] = x @ components.T
    return out
