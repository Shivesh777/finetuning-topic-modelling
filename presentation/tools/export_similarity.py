#!/usr/bin/env python3
"""Slides 3.1 and 3.3: what the base NoLBERT and the adapted encoder make of two example batches of
32 article pairs. Writes deck/js/similarity.js, which git ignores like portfolio.js; without it
3.1 draws the calculation schematically and 3.3 omits the matching diagram.

    FTM_DATA=/path/to/data python tools/select_example_batches.py       # the two batches (duckdb)
    pip install torch transformers safetensors
    python tools/export_similarity.py --ft ../nolbert-ftm-runnable/encoder/nolbert_ft \
        --batches /path/to/data/news/example_batches.json

For each batch and each checkpoint (base alikLab/NoLBERT = before, the adapted encoder = after):
cosine similarities of every query chunk A with every candidate chunk B, the temperature-scaled
scores S = Za Zb^T / 0.05, row softmax (A->B) and column softmax (B->A) probabilities, the argmax
retrieval choice of every row and column, the number correct, and the symmetric InfoNCE loss of
the batch. Choices and probabilities are computed in full precision before the exported values
are rounded.

Two regimes. --regime as-trained (the default) reproduces train_nolbert.py exactly: every input
is truncated to 128 tokens and right-padded to 128 positions with the tokenizer's pad id (1), and
the attention mask is `ids != 0`, so the padding positions are attended and enter the mean, as
they did in the run that scored 7.96% -> 89.16%. --regime masked pads to the longest text in the
batch and masks the padding, the regime of the downstream embedding step (SentenceTransformers).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "deck" / "js" / "similarity.js"


def embed(model_name, texts, max_len, device, regime="as-trained"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        if regime == "as-trained":
            seqs = tok(texts, truncation=True, max_length=max_len)["input_ids"]
            ids = torch.full((len(texts), max_len), tok.pad_token_id or 0, dtype=torch.long)
            for i, seq in enumerate(seqs):
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask = (ids != 0).long()                       # train_nolbert.py:230: pad id 1 counts as a token
            hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        else:
            enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"]
        m = mask.unsqueeze(-1).to(hidden.dtype).to(hidden.device)
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(pooled, dim=-1).float().cpu()


def score(za, zb, tau):
    import torch
    n = len(za)
    cos = za @ zb.T
    s = cos / tau
    y = torch.arange(n)
    p_row = torch.softmax(s, dim=1)                 # row i: which candidate B is A_i's partner
    p_col = torch.softmax(s, dim=0)                 # column j: which query A is B_j's partner
    choice_row, choice_col = cos.argmax(1), cos.argmax(0)
    loss = 0.5 * (torch.nn.functional.cross_entropy(s, y) + torch.nn.functional.cross_entropy(s.T, y))
    return {"cos": [[round(float(v), 4) for v in r] for r in cos],
            "p_row": [[round(float(v), 5) for v in r] for r in p_row],
            "p_col": [[round(float(v), 5) for v in r] for r in p_col.T],      # p_col[j] = the column j distribution over rows
            "choice_row": [int(v) for v in choice_row], "choice_col": [int(v) for v in choice_col],
            "correct_row": int((choice_row == y).sum()), "correct_col": int((choice_col == y).sum()),
            "loss": round(float(loss), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="alikLab/NoLBERT", help="base NoLBERT, a Hugging Face id or a local path")
    ap.add_argument("--ft", required=True, help="the adapted encoder, e.g. nolbert-ftm-runnable/encoder/nolbert_ft")
    ap.add_argument("--batches", required=True, help="json written by tools/select_example_batches.py")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--regime", choices=["as-trained", "masked"], default="as-trained")
    a = ap.parse_args()
    src = json.loads(Path(a.batches).read_text())
    out = {"available": True, "n": 32, "max_len": a.max_len, "temperature": a.temperature, "base": a.base, "adapted": str(a.ft),
           "regime": a.regime, "pair": src.get("pair"), "rule": src.get("rule"), "batches": {}}
    for name, b in src["batches"].items():
        pairs = b["pairs"]
        assert len(pairs) == 32 and all(len(p) == 2 for p in pairs), f"{name}: 32 pairs of two texts"
        left, right = [p[0] for p in pairs], [p[1] for p in pairs]
        rec = {"window": b["window"], "ids": b["ids"], "dates": b.get("dates")}
        for key, model_name in [("before", a.base), ("after", a.ft)]:
            za, zb = embed(model_name, left, a.max_len, a.device, a.regime), embed(model_name, right, a.max_len, a.device, a.regime)
            rec[key] = score(za, zb, a.temperature)
            print(f"{name} {key}: correct A->B {rec[key]['correct_row']}/32, B->A {rec[key]['correct_col']}/32, "
                  f"loss {rec[key]['loss']}, p(same) of pair 1: {rec[key]['p_row'][0][0]:.3f}")
        out["batches"][name] = rec
    OUT.write_text("window.DATA.similarity=" + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
