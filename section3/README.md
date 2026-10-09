# Section 3: NoLBERT fine-tuning (standalone)

The fine-tuning section of the group deck as a self-contained app, so it can
run and evolve without touching `presentation/`.

    pip install flask
    python section3/app.py        ->  http://127.0.0.1:5056

Four slides, one click each (R replays a slide's animations):

- **3.1 Find the partner** — the training replay, held-out 8% -> 89%.
- **3.2 Supervision trick** — two chunks of one article form a labeled pair;
  the trick the NoLBERT paper uses on patent abstracts, run on news.
- **3.3 What it read** — the adaptation window against the train/validation/test
  protocol (pair cutoff 31 Dec 2019).
- **3.4 Reshape the space** — what the contrastive fine-tune does to the
  vectors: pairs pull together, clusters emerge for spherical k-means.

`/reshape.html` is the interactive companion page for 3.4.

Assets (`css/`, `js/`, `fonts/`, `js/data.js`) are copies from
`presentation/deck/` at the time this folder was built; the numbers come from
`results/encoders/nolbert_ft/train_log.json` via `presentation/tools/build_data.py`.
