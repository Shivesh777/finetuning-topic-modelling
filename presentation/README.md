# Group presentation (MFE 230ZB)

A browser deck of 19 slides. One click per slide: each slide's animations then play by themselves.

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

- Pre-read: `/?still=1` shows every slide finished, one under another; print it to PDF from Chrome. `/?still=1&only=12` shows slide 12 alone.
- Checks: `/?still=1&qa=1` lists overlapping text and text past the margins; `/?selftest=1` clicks through every slide and step.

## Numbers

Every number on the slides comes from `deck/js/data.js`, which `tools/build_data.py` builds from this repo (FINDINGS.md, results/, data/topics, as at commit b97a27a). `--check` prints the values next to the ones in FINDINGS.md. It needs pandas, numpy and duckdb.

    python presentation/tools/build_data.py --check

Slide 4.3 animates the weekly forecast ranks of individual, unnamed stocks. They come from `data/predictions` and `data/panels`, which are derived from CRSP and stay out of git, so the builder writes them to `deck/js/portfolio.js`, which is gitignored. Without that file the slide shows a short note instead of the dots. To build it, point `FTM_DATA` at a data folder with `predictions/W` and `panels/W`.
