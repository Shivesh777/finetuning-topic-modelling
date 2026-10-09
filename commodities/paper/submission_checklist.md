# Submission-readiness checklist (`main.tex`), final review of 8 Oct 2026

## Ready
- [x] Numbers: every result in text and tables is inserted from the saved result files by `build_paper.py` (177 values, `numbers_used.csv`); every hand-written number is recomputed or located in documentation by `verify_paper_numbers.py` (`paper_number_audit.md`, 0 failures). Unaccounted-for numerals are bibliography volumes, pages and identifiers.
- [x] Central findings agree across abstract, introduction, results and conclusion (31% held-out reduction; interval reaches zero; Holm 0.103 / 0.066 / 0.022; borderline; representation-dependent; FinBERT not point-in-time; NoLBERT and LDA do not reproduce it; explanation, not prediction; gold, Bitcoin and forecasting negative; no robust trading signal).
- [x] Original, descriptive and post hoc analyses separated (Table 2 and captions). The 63-day and 2024 checks never replace the primary result.
- [x] No representation is called point-in-time. FinBERT is "not point-in-time"; NoLBERT and LDA are "historically more compatible", with residual look-ahead stated.
- [x] Gain metric defined with formula, benchmark, frozen coefficients and bootstrap method.
- [x] Feb-Jun exclusion explained; short, non-contiguous sample and the serial-dependence limitation documented (17 of 20 successive test-month pairs are adjacent; lag-1 loss-differential correlations +0.30, -0.10, -0.34, +0.08, each within about 1.5 standard errors of zero; a block bootstrap was not run).
- [x] Static LaTeX checks: environments, braces, math, labels and references consistent; tables 1-7 and figures 1-3 in order; T1 font encoding added (fixes the literal "|" in class names); long identifiers use \url. No compiler is installed here, so nothing has been compiled.
- [x] Citations: every in-text citation has a reference entry and vice versa.

## Reference verification against primary sources (8 Oct 2026)
| entry | result |
|---|---|
| Clark and West (2007), J. Econometrics 138(1), 291-311 | matches Crossref |
| MacKinnon and White (1985), J. Econometrics 29(3), 305-325 | matches Crossref |
| Newey and West (1987), Econometrica 55(3) | matches Crossref (record gives only the first page, 703; the end page 708 is not confirmed) |
| Kelly, Pruitt and Su (2019), J. Financial Economics 134(3), 501-524 | matches Crossref (DOI 10.1016/j.jfineco.2019.05.001) |
| Bailey and Lopez de Prado (2014), J. Portfolio Management 40(5), 94-107 | matches Crossref |
| Bybee, Kelly and Su (2023), Review of Financial Studies 36(12), 4759-4787 | matches Crossref (DOI 10.1093/rfs/hhad042) |
| Dong, Fan and Peng (2024), arXiv:2402.06698 | title and authors match arXiv; the dataset covers 1999-2023 |
| Kakhbod and Li (2025), arXiv:2509.01110 | authors and "pretrained exclusively on text from 1976 to 1995" confirmed on arXiv. The arXiv title is "NoLBERT: A No Lookahead(back) Foundational Language Model"; the earlier team paper added "for empirical research". We use the arXiv title; check the current version |
| Holm (1979), Scand. J. Statistics 6(2), 65-70 | **not verified**: the original is not in Crossref (only a 1987 correction was returned) |
| nickmuchi classifier | Hugging Face API: model repository created 30 Dec 2022, last modified 12 Jul 2023; fine-tuned on `zeroshot/twitter-financial-news-topic` (repository created 7 Sep 2022); the card gives **no tweet collection dates**. The paper words the dates as repository dates |
| FinMultiTime | listed by repository id only; no authors or paper cited. Confirm whether a citation is preferred |

## Model-training claims
| claim | status |
|---|---|
| NoLBERT pretrained only on 1976-1995 text | verified (arXiv abstract) |
| NoLBERT adaptation pairs drawn up to 2020-09-01 | **unverified against the training log**; rests on `NOLBERT_RESULTS.md` and `HANDOFF.md`, which quote `results/nolbert_ft/train_log.json` (not in the local files) |
| `_s63` is a 63-day window chosen on validation for monthly rebalancing | the 63-day construction is verified by reproducing the saved factor returns (correlation at least 0.998); "chosen on validation" rests on a project member's message |
| Training schedule of the equity and IPCA models (tune on training, choose on validation, refit once, test frozen) | documented in the project code snapshot and in every IPCA `summary.json`; 29 of 50 IPCA test-forecast files independently reproduced; the equity models cannot be retrained here |
| Whole-corpus vocabulary basis | project documentation only; effect not tested |
| FinBERT classifier base model | the card names `yiyanghkust/finbert-tone` as base; not used in the paper |

## Overleaf compilation (exact steps)
1. In Overleaf choose New Project, Upload Project, and upload a zip of `research/paper/` containing `main.tex` and the `figures/` folder (three PNG files). Run `cd ~/230ZB/research/paper && zip -r overleaf.zip main.tex figures` first.
2. Menu, Compiler: pdfLaTeX; TeX Live version: current default; main document: `main.tex`.
3. Recompile twice so that table and figure references resolve.
4. Expected packages (all in standard TeX Live): fontenc, lmodern, geometry, booktabs, amsmath, graphicx, longtable, array, hyperref, caption.
5. Likely warnings that are harmless: a few Overfull/Underfull hbox messages in the wide tables. Anything labelled "Undefined control sequence" or "Misplaced alignment tab" is a real error; please send it to me.
6. Check visually: the formula for the gain (Section 4), the seven tables, the three figures, and that class names such as "Energy | Oil" print with a vertical bar.

## Manual confirmation still needed
- [ ] Compile in Overleaf and read the PDF once, including table widths (Tables 2, 5 and 6 are the widest).
- [ ] Author names and affiliations.
- [ ] Length against the course or venue limit (about 7,900 words including markup).
- [ ] Co-authors confirm the project-pipeline statements (encoder pair cutoff, `_s63`, Feb-Jun rule, vocabulary basis) and, if the training log exists, check the 1 Sep 2020 cutoff.
- [ ] Verify Holm (1979) and the end page of Newey and West (1987) against the journals.
- [ ] Decide whether to cite FinMultiTime differently.

## What could materially weaken the paper if left unresolved
1. The primary model's FinBERT classifier is not point-in-time, and it is the only representation that shows the crude-oil link. A reviewer can read the main finding as possible look-ahead; the paper says this plainly but cannot resolve it.
2. The 21-month sample is short and non-contiguous, the bootstrap ignores possible serial dependence, and the held-out interval reaches zero; the inference for crude oil is borderline and method-dependent.
3. The encoder-date and `_s63` claims rest on project documentation and a message, not on the training log or the current code.

## Interpretations deliberately left open
Why the crude-oil link appears only with FinBERT; what the third factor captures; whether a common macroeconomic cause drives attention, the factors and crude-oil returns.
