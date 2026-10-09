# Group presentation (MFE 230ZB)

A browser deck of 26 slides and 18 backup slides. One click per slide: each slide's animations then play by themselves, at twice the speed they were drawn at (four times in section 4: `pace` in `deck/js/charts-a.js`; the `data-auto` times in `index.html` are the fast ones).

The title and closing slides show what the news was about, month by month: which of the 20 NoLBERT topics took more of the news than usual, with four callouts whose headline keywords were checked on the article file (`deck/js/panel.js`).

    pip install flask
    python presentation/app.py        # http://127.0.0.1:5055

Without Flask: `python -m http.server 5055 --directory presentation/deck`.

| Key | |
|---|---|
| Right, Space, Enter, a click | next slide |
| Left | previous slide, shown complete |
| R | replay this slide's animations |
| G, a number, Enter | jump to that slide |
| I | list of slides |
| F, T, B | full screen, timer, black screen |
| N | this slide's speaker notes |

- Pre-read: `/?still=1` shows every slide finished, one under another; print it to PDF from Chrome. `/?still=1&only=12` shows slide 12 alone; `/?still=1&notes=1` prints the speaker notes under their slides.
- Checks: `/?still=1&qa=1` lists overlapping text, text lying over chart shapes and text past the margins; `/?qa=1#12.3` checks the live slide 12 held at step 3; `/?selftest=1` clicks through every slide and step.

## Numbers

Every number on the slides comes from `deck/js/data.js`, which `tools/build_data.py` builds from this repo (FINDINGS.md, results/tables, results/explain, results/encoders, results/series, data/topics, as at commit b97a27a). `--check` prints the values next to the ones in FINDINGS.md. It needs pandas, numpy and duckdb.

    python presentation/tools/build_data.py --check

Backup slide B7 animates the weekly forecast ranks of individual, unnamed stocks. They come from `data/predictions` and `data/panels`, which are derived from CRSP and stay out of git, so the builder writes them to `deck/js/portfolio.js`, which is gitignored. Without that file the slide shows a short note instead of the dots; the same file carries the one-stock contributions of backup slide B5. To build it, point `FTM_DATA` at a data folder with `predictions/W` and `panels/W`.

## The NoLBERT example batches (slides 3.1 and 3.3)

Slide 3.1 colours its 32 × 32 similarity matrix and unfolds its probabilities from measured values, and slide 3.3 draws the matching decisions of the base and the adapted encoder, when `deck/js/similarity.js` exists (gitignored). Without it, 3.1 draws the calculation schematically (labelled so) and 3.3 shows the recorded loss and the held-out evaluation only.

    FTM_DATA=/path/to/data python tools/select_example_batches.py      # 32 (headline, lead) pairs from Jan 2015 and 32 from Sep 2020, by a fixed rule (duckdb)
    pip install torch transformers safetensors                            # any Python 3.10+ environment; CPU is enough
    python tools/export_similarity.py --ft ../nolbert-ftm-runnable/encoder/nolbert_ft --batches /path/to/data/news/example_batches.json

The exporter downloads the base `alikLab/NoLBERT` (about 440 MB) once, scores both batches under both checkpoints and, by default, reproduces the run's own padding regime (`--regime as-trained`: 128 positions, pad id 1 attended, as `train_nolbert.py` did). The original 2,048 held-out pairs of the adaptation are not in the repository, so the slides label the batch “New example batch · 32 pairs”.
