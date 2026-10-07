"""Sentence embeddings of article texts.

Any Hugging Face encoder works: a sentence-transformers model is used as it
is, and a plain masked-language model (for example NoLBERT) is wrapped with
attention-masked mean pooling. Output rows have unit length and are stored as
float16; long runs write to a .npy file block by block and resume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_encoder(name: str, max_tokens: int, device: str | None = None,
                 revision: str | None = None):
    """A sentence-transformers model truncating inputs to max_tokens.

    name: a Hugging Face repository or a local folder (a fine-tuned encoder). A model
    that is not packaged for sentence-transformers (a plain masked-language model) gets
    attention-masked mean pooling over its last layer. revision pins a repository commit."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=device, revision=revision)
    model.max_seq_length = max_tokens
    return model


def encode(texts, name: str, max_tokens: int, out: str | Path, batch_size: int = 256,
           block: int = 10_000, device: str | None = None, budget: float | None = None,
           revision: str | None = None, verbose: bool = True) -> np.ndarray:
    """Embeddings (n, d) float16 of texts, written to `out` (.npy), resumable.

    budget: stop before this many seconds have passed (rerun to continue). Returns the
    memory-mapped array; rows not yet computed are zero and `out`.done holds
    the number of finished rows."""
    import time

    out = Path(out)
    done_path = out.with_suffix(".done")
    model = load_encoder(name, max_tokens, device, revision)
    dim = model.get_sentence_embedding_dimension()
    n = len(texts)
    if out.exists():
        array = np.lib.format.open_memmap(out, mode="r+")
        if array.shape != (n, dim):
            raise ValueError(f"{out} has shape {array.shape}, expected {(n, dim)}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n, dim))
    start = int(done_path.read_text()) if done_path.exists() else 0
    t0 = time.time()
    for lo in range(start, n, block):
        per_block = (time.time() - t0) / max(1, (lo - start) // block)
        if budget and lo > start and time.time() - t0 + per_block > budget:
            break
        chunk = list(texts[lo:lo + block])
        array[lo:lo + len(chunk)] = model.encode(
            chunk, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        array.flush()
        done_path.write_text(str(lo + len(chunk)))
        if verbose and (lo // block % 5 == 4 or lo + block >= n):
            rate = (lo + len(chunk) - start) / (time.time() - t0)
            print(f"embedded {lo + len(chunk):,} of {n:,} ({rate:.0f} texts/s)", flush=True)
    return array
