#!/usr/bin/env python3
"""Cosine similarities of same-article pairs under the base and the adapted NoLBERT, on test-period
articles that neither the encoder nor any topic model has seen (2022-2024). Writes only numbers to
data/nolbert_similarity.json; the selected article text stays in the repository's data folder.

    python scripts/nolbert_similarity.py select            # pick the pairs (duckdb)
    python scripts/nolbert_similarity.py embed base 0 256  # embed in slices (CPU), repeat for ft
    python scripts/nolbert_similarity.py score             # write data/nolbert_similarity.json

Pair: (headline, lead) of one article, the default pair of 03c_finetune_encoder.py. Rule: dated
2022-01-01 to 2024-12-31, lead >= 300 characters, headline >= 30 characters, filed under a ticker,
one article per lead, the first N by article_id. Encoding reproduces the adaptation run (128
tokens, pad id 1 attended, mean of the last layer, L2 norm), as presentation/tools/export_similarity.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent / "finetuning-topic-modelling"
CACHE = REPO / "data" / "news" / "paper_similarity"          # article text: stays out of the paper folder and git
MODELS = {"base": "alikLab/NoLBERT", "ft": str(PAPER.parent / "nolbert-ftm-runnable" / "encoder" / "nolbert_ft")}
N, B, TAU = 512, 32, 0.05


def select():
    import duckdb
    q = f"""select article_id, title, lead from (
              select *, row_number() over (partition by lead order by article_id) as rn
              from read_parquet('{REPO}/data/news/articles/*.parquet')
              where date >= '2022-01-01' and date <= '2024-12-31 23:59:59'
                and length(lead) >= 300 and length(title) >= 30 and n_tickers >= 1)
            where rn = 1 order by article_id limit {N}"""
    rows = duckdb.connect().execute(q).fetchall()
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "pairs.json").write_text(json.dumps([[r[0], r[1], r[2]] for r in rows]))
    print(len(rows), "pairs")


def embed(which, lo, hi):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    pairs = json.loads((CACHE / "pairs.json").read_text())[lo:hi]
    tok = AutoTokenizer.from_pretrained(MODELS[which])
    model = AutoModel.from_pretrained(MODELS[which]).eval()
    out = []
    with torch.no_grad():
        for side in (1, 2):
            texts = [p[side] for p in pairs]
            zs = []
            for k in range(0, len(texts), 16):
                seqs = tok(texts[k:k + 16], truncation=True, max_length=128)["input_ids"]
                ids = torch.full((len(seqs), 128), tok.pad_token_id or 0, dtype=torch.long)
                for i, s in enumerate(seqs):
                    ids[i, :len(s)] = torch.tensor(s)
                mask = (ids != 0).long()
                h = model(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).float()
                zs.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy())
            out.append(np.concatenate(zs))
    np.save(CACHE / f"{which}_{lo}_{hi}.npy", np.stack(out))
    print(which, lo, hi, "done")


def load(which):
    parts = sorted(CACHE.glob(f"{which}_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    return np.concatenate([np.load(p) for p in parts], axis=1)     # (2, N, 768)


def score():
    res = {"pair": "headline and lead of one article", "window": ["2022-01-01", "2024-12-31"], "n": N,
           "batch": B, "temperature": TAU}
    for which in MODELS:
        z = load(which)
        za, zb = z[0], z[1]
        n = len(za) // B * B
        top1, diag, off = 0, [], []
        for k in range(0, n, B):
            c = za[k:k + B] @ zb[k:k + B].T
            top1 += int((c.argmax(1) == np.arange(B)).sum())
            diag += list(np.diag(c)); off += list(c[~np.eye(B, dtype=bool)])
        c0 = za[:B] @ zb[:B].T
        res[which] = {"top1": top1 / n, "pairs": n, "cos_same_mean": float(np.mean(diag)),
                      "cos_other_mean": float(np.mean(off)), "cos_same_q": np.percentile(diag, [10, 50, 90]).round(4).tolist(),
                      "cos_other_q": np.percentile(off, [10, 50, 90]).round(4).tolist(),
                      "batch0_cos": np.round(c0, 4).tolist(),
                      "hist_same": np.histogram(diag, bins=50, range=(-0.2, 1))[0].tolist(),
                      "hist_other": np.histogram(off, bins=50, range=(-0.2, 1))[0].tolist()}
        print(which, res[which]["top1"], res[which]["cos_same_mean"], res[which]["cos_other_mean"])
    (PAPER / "data").mkdir(exist_ok=True)
    (PAPER / "data" / "nolbert_similarity.json").write_text(json.dumps(res))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        select()
    elif cmd == "embed":
        embed(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        score()
