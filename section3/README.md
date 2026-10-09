# Section 3 revisions (full deck)

The complete team deck with Kevin's revised section 3, kept as a standalone
app so the changes are easy to review and transfer into `presentation/`.

    pip install flask
    python section3/app.py        ->  http://127.0.0.1:5056

Based on the deck as of the 9 October briefs (sections 1, 4 and 5 unchanged
from `presentation/deck/`). What differs, all inside section 3:

- **3.0 The pair (new)** — intro animation: two chunks of one synthetic
  article travel through the encoder, land far apart, and the fine-tune
  pulls them together. Chart in `js/charts-f.js`.
- **3.1** — the "Padding included" label is removed from the pooling path.
- **3.2** — the fine-tuning window is shown as Jan 2015 - Dec 2019
  (band, label and FNSPID box), the eight-months caption is replaced with
  "No validation or test-period text in the fine-tuning", and the proposed-
  cutoff marker is dropped. The next-steps bullet no longer lists the
  Dec 2019 refit as future work.

**Before presenting this version**: the Dec 2019 fine-tune must actually be
rerun (the current checkpoint was trained with pairs to Aug 2020), and the
run numbers on 3.2/3.3 and backups B3/B10 (228,883 pairs, 2,048 held out,
6,205 steps, 150 minutes, 7.96% -> 89.16%) must be regenerated from the new
run via `presentation/tools/build_data.py`. Removed disclosures are
documented in Kevin's write-up.

`/reshape.html` is a companion page for the reshape-the-space idea.
