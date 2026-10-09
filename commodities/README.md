# Cross-asset extension: commodities and Bitcoin

Code, result tables and the draft paper for the extension of the narrative-factor study to crude oil, gold, copper and Bitcoin futures.
Status: empirical results frozen; the paper (`paper/main.tex`) has not been compiled yet and still needs author names and co-author confirmation of the pipeline claims (see `paper/submission_checklist.md`).

## Layout
| folder | contents |
|---|---|
| `pull_futures.py`, `build_returns.py`, `pull_etfs.py`, `label_news_llm.py` | data pulls (Databento CME futures, CRSP funds, optional Claude news labelling) |
| `phase1/` | robustness of the saved equity forecasts: costs, deflated Sharpe, bootstrap, placebo and permutation checks; `robustness_report.md` |
| `phase2/` | weekly topic-shock tests for CL, GC, BTC; IPCA factor exposures; post hoc 63-day and 2024 checks; reconstruction test of the 63-day features; futures validation; specifications (`*_prereg.md`); methodology review |
| `paper/` | `main.tex`, figures, `build_paper.py` (fills every number from the result files), number audit, submission checklist |

## Main findings (see `paper/main.tex`)
- Equity forecasts chosen on validation are not distinguishable from luck in the test period after costs.
- Weekly topic shocks move with crude-oil returns but do not forecast; no primary test survives multiple-testing correction.
- FinBERT-based IPCA factors reduce held-out squared error for crude oil by about 31% in 21 months; the interval reaches zero, the adjusted p-value is 0.103 (0.066 calibrated, 0.022 post hoc), and the link is representation-dependent: not reproduced at comparable strength by NoLBERT or LDA.
- The FinBERT classes are not point-in-time. Nothing here establishes a forecasting or trading signal.

## Not in this folder (licensed or third-party)
CRSP and Databento data and every file derived from them (futures return series, the aligned monthly and weekly panels, parquet files), article text and headline excerpts, and API keys (`.env`). Rebuild order: `pull_futures.py`, `build_returns.py`, `phase2/build_ipca_monthly.py`, `phase2/build_weekly_panel.py`, then the analysis scripts in `phase2/`, then `paper/build_paper.py`.

## Path assumptions
Scripts find the repository root two levels above `phase1/`, `phase2/` and `paper/` and expect `data/` there, with the IPCA factor files under `data/results/ipca/M/` and the futures outputs under `data/commodities/`.
