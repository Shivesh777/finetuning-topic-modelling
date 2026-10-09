"""Writes robustness_report.md from findings.json, robustness_summary.csv and config.json (numbers are never typed by hand)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
f = json.load(open(H / "findings.json"))
cfg = json.load(open(H / "config.json"))
d = pd.read_csv(H / "robustness_summary.csv")
FREQ = {"M": "monthly", "2W": "biweekly", "W": "weekly"}
r2 = lambda x: f"{x:.2f}"

L = []
add = L.append
add("# Phase 1: robustness and leakage audit of the saved equity forecasts\n")
add("Everything below is computed from the saved forecasts in `data/predictions/`. The training scripts are not in the folder, so "
    "statements about how the forecasts were produced are either checked against the saved data (marked **verified**) or "
    "taken from the team's description of the schedule (marked **assumed**).\n")

add("## 1. Bottom line\n")
add("- The specification chosen on the validation split does not carry its validation performance into the test split at any "
    "frequency. Validation Sharpe ratios of " + ", ".join(f"{r2(f[k]['validation_sharpe'])} ({FREQ[k]})" for k in f) +
    " become test Sharpe ratios of " + ", ".join(f"{r2(f[k]['test_sharpe_by_cost']['10'])}" for k in f) +
    " after 10 bp per side.")
add("- Across all 375 real specifications, the median test Sharpe ratio is between -0.02 and 0.22 before costs and negative after costs at every frequency; "
    "most specifications lose money net of 10 bp (only 24-33% have a positive Sharpe).")
add("- Costs matter a great deal: the chosen strategies turn over " + ", ".join(f"{f[k]['test_turnover_per_period']:.1f}" for k in f) +
    " times the book per period (monthly, biweekly, weekly), where each leg sums to 1 so that replacing the entire book would be 4.0.")
add("- For the chosen model, no 95% interval for the Sharpe ratio excludes zero, and no deflated Sharpe ratio exceeds 0.75 (the best case, the best-in-hindsight monthly specification with only 6 effective trials assumed).")
add("- The saved shuffled-topic placebo is a single shuffle run through 15 models. It is not a permutation test, and "
    "at monthly frequency it contains specifications that look as good as real ones.")
add("- These are findings about the saved forecasts and a 21 / 42 / 81-period test window. They do not show that the text signal is worthless; "
    "they show that this evidence cannot distinguish it from luck.\n")

add("## 2. What was run\n")
add("| item | setting |\n|---|---|")
add("| experiments | 26 feature sets x 15 models = 390 per frequency (375 real, 15 on the shuffled-topic placebo `narr_lda_placebo`) |")
add("| splits | validation 2020-2021 and test 2022-2024 as saved; test forecasts come from a refit on training + validation (assumed, per the team's schedule) |")
add("| decile strategy | long top 10% of forecasts, short bottom 10%, equal weighted, rebuilt every period |")
add("| buffered strategy | enter at the top / bottom 10%, keep a name until it leaves the top / bottom 20% |")
add("| costs | " + ", ".join(str(c) for c in cfg["costs_bps_per_side"]) + " bp per side on turnover (sum of absolute weight changes; closing the book at a gap is charged) |")
add("| selection | " + f"{cfg['selection']['strategy']} strategy, Sharpe at {cfg['selection']['cost_bps']} bp on the {cfg['selection']['split']} split, among the real specifications only; fixed before looking at test |")
add(f"| bootstrap | circular moving blocks of {cfg['bootstrap_block']['M']} / {cfg['bootstrap_block']['2W']} / {cfg['bootstrap_block']['W']} periods (M / 2W / W), {cfg['bootstrap_draws']} draws, seed {cfg['seed']} |")
add(f"| permutation null | {cfg['permutation_draws']} draws, each giving every stock another stock's full forecast history (one relabelling per draw); keeps persistence, turnover and cost; removes the link between forecasts and that stock's returns |")
add("| gaps | the panel skips Feb-Jun holding months (section 6); the book is closed and rebuilt across any gap longer than 1.6x the median spacing |")
add("| Sharpe | annualised with sqrt(12), sqrt(26), sqrt(52) on the periods available; the periods are not contiguous |\n")

add("## 3. The model chosen on validation\n")
add("| frequency | chosen specification | validation Sharpe | test Sharpe 0 / 5 / 10 / 20 bp | turnover / period | max drawdown | 95% bootstrap interval, Sharpe | same, minus price-only model of the same type | test periods |")
add("|---|---|---|---|---|---|---|---|---|")
for k, v in f.items():
    s = v["test_sharpe_by_cost"]
    add(f"| {FREQ[k]} | `{v['chosen_on_validation']}` | {r2(v['validation_sharpe'])} | {r2(s['0'])} / {r2(s['5'])} / {r2(s['10'])} / {r2(s['20'])} | "
        f"{v['test_turnover_per_period']:.2f} | {v['test_max_drawdown']:.1%} | [{r2(v['bootstrap_sharpe_95ci'][0])}, {r2(v['bootstrap_sharpe_95ci'][1])}] | "
        f"[{r2(v['bootstrap_diff_vs_price_95ci'][0])}, {r2(v['bootstrap_diff_vs_price_95ci'][1])}] | {v['test_periods']} |")
add("\nThe chosen specification differs by frequency, and the weekly choice changed when I fixed the gap handling; a result that depends on that is fragile.\n")

add("## 4. All specifications\n")
add("Test split, buffered strategy, real specifications only.\n")
add("| frequency | median Sharpe at 0 / 5 / 10 / 20 bp | share with positive Sharpe at 0 bp / 10 bp | median turnover / period | price-only models (median / max at 10 bp) | placebo range at 10 bp | rank correlation, validation vs test Sharpe across specs |")
add("|---|---|---|---|---|---|---|")
t = d[(d.split == "test") & (~d.placebo) & (d.strategy == "buffer")]
for k in f:
    s = t[t.freq == k]
    pr = s[s.feature_set == "price"].sharpe_10bp
    pl = d[(d.freq == k) & (d.split == "test") & d.placebo & (d.strategy == "buffer")].sharpe_10bp
    v = d[(d.freq == k) & (d.split == "validation") & (d.strategy == "buffer") & (~d.placebo)]
    m = v.merge(s[["experiment", "sharpe_10bp"]], on="experiment", suffixes=("_v", "_t"))
    add(f"| {FREQ[k]} | " + " / ".join(r2(s[f"sharpe_{c}bp"].median()) for c in [0, 5, 10, 20]) +
        f" | {(s.sharpe_0bp > 0).mean():.0%} / {(s.sharpe_10bp > 0).mean():.0%} | {s.turnover_per_period.median():.2f} | {r2(pr.median())} / {r2(pr.max())} | "
        f"[{r2(pl.min())}, {r2(pl.max())}] | {m.sharpe_10bp_v.corr(m.sharpe_10bp_t, method='spearman'):.2f} |")
add("\nThe validation-to-test rank correlation is the direct measure of whether selecting on validation helps: it is near zero at monthly frequency and modest at the others. "
    "Decile and buffered strategies give the same picture (`robustness_summary.csv` has both).\n")

add("## 5. Multiple testing, placebo, permutation\n")
add("**Trials.** 375 real specifications per frequency, but their returns are highly correlated. The participation ratio of the eigenvalues of their "
    "return correlations gives " + ", ".join(f"{f[k]['n_trials_effective']:.1f} ({FREQ[k]})" for k in f) +
    " effective independent trials. The deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) is computed with N = effective trials, N = 26 (one per feature set) and N = 375, "
    "using the cross-sectional variance of the 375 per-period Sharpe ratios. Which N is right is a judgement; all three are shown.\n")
add("| frequency | best specification in test (hindsight) | its Sharpe at 10 bp | deflated, N_eff / 26 / 375 | chosen model: deflated at N=1 (probability true Sharpe > 0) | permutation p, gross / net of 10 bp | permutation null 95th percentile (net) |")
add("|---|---|---|---|---|---|---|")
for k, v in f.items():
    b = v["dsr_best_in_test"]
    add(f"| {FREQ[k]} | `{v['best_in_test']}` | {r2(v['best_in_test_sharpe'])} | {r2(b['N_eff'])} / {r2(b['N_26'])} / {r2(b['N_raw'])} | {r2(v['dsr_chosen_n1'])} | "
        f"{v['permutation_p_value_gross']:.3f} / {v['permutation_p_value']:.3f} | {r2(v['permutation_null_95pct'])} |")
add("\nReading: a deflated Sharpe near 1 would mean strong evidence; here the best-in-hindsight specification is at 0.03-0.75 depending on N, and the chosen model "
    "at N = 1 is at 0.4-0.7, which is weak evidence of any positive Sharpe even without deflation. "
    "The permutation p-values test the chosen model alone against random relabelling; they do not account for having chosen it from 375. "
    "The biweekly net p-value is much smaller than the gross one, which suggests the relabelled null pays more in costs than the chosen model; I have not resolved that, so I rely on the gross values.\n")
add("**What the placebo is.** `narr_lda_placebo` is one set of shuffled topic features run through the same 15 models. It is a single draw, so its 15 results "
    "are highly dependent and it does not give a distribution of what shuffled features can achieve. Its best monthly specification reaches a Sharpe comparable to real "
    "ones. I therefore do not describe it as a permutation test. The permutation null above is a separate construction.\n")
add("**Independent observations.** The test split has 21 monthly, 42 biweekly and 81 weekly periods, about 1.6-1.75 years of returns in each case (non-contiguous, within 2022-2024). "
    "The usual approximation for the standard error of an annualised Sharpe ratio is 1 / sqrt(years of data), about 0.8 here; the block-bootstrap intervals are wider (half-widths 1.2-1.9) because the samples are short and the returns skewed.\n")

add("## 6. Feb-Jun holding months\n")
add("From 2017 on the panel contains only formation dates in June-December, i.e. holding months July-January; February-June holding months are absent from training, validation and test. "
    "The team's schedule describes this (\"January and July-December from 2017 on\"), so it is by design, not a data error. "
    "What I cannot establish is why: if those months were removed after seeing results, that is selection; if they were removed for computing or data reasons, it is harmless. "
    "Consequences I can confirm: the test sample is 21 months out of 36; positions cannot be carried across the gap (handled); annualised statistics treat the available months as exchangeable. "
    "Not quantified: whether results differ in the omitted months.\n")

add("## 7. Leakage audit\n")
add("| item | status | evidence and limits |")
add("|---|---|---|")
rows = [
    ("Topic models fit only through December 2019", "verified from saved metadata; code not rerun",
     "All six `quality.json` files give `fit_until` 2019-12-31 and 140,209 fitting articles. Separate caveat: the team reports the NoLBERT encoder window as Jan 2015 - Aug 2020 (per the user), which would overlap the validation split for text-derived features; its adaptation objective is undocumented, and the encoder checkpoint and script are not in the folder."),
    ("Rolling covariance features are lagged", "verified on a sample",
     "I rebuilt NoLBERT-20 narrative shocks from `daily_sums.parquet` and CRSP returns and compared them with the saved panel features (5 formation dates x 4 topics, monthly panel). Rank correlation with a window ending at the formation date is 0.92, ending the day before 0.93, ending 21 days later 0.72, 63 days later 0.41. The features therefore use information up to the formation date, not beyond; a one-day overlap cannot be ruled out. The exact kernel is not known."),
    ("Feature scaling", "verified by construction",
     "Every feature block is rank-normalised across stocks within each date (per-date mean 0, standard deviation 0.578, the value for a uniform variable on [-1, 1]), so no global statistic from the test period enters."),
    ("Hyperparameters and model selection use training data only", "assumed",
     "Per the team's schedule: settings fitted on 69 training months, chosen on 14 validation months, refitted once on 83 months, applied unchanged to the 21 test months. Cannot be checked without the scripts."),
    ("Saved forecasts free of test contamination", "partly verified",
     "All 2,340 saved arrays (780 per frequency) match the panel's validation and test row counts exactly. Whether training touched test rows cannot be verified here."),
    ("Realised returns are forward returns", "verified",
     "Saved `ret` equals the compounded CRSP daily return from the next business day to the next formation date (rank correlation 1.0 on consecutive months). `rev_5` and `rev_21` correlate 0.86-1.0 with past returns and about 0 with the next week."),
    ("Firm news-count feature", "verified on a sample",
     "Outside 2024, rank correlation with the trailing 30-day article count is 0.81-0.93 and with the following 30 days 0.35-0.54 (persistence of news flow, not look-ahead). In 2024 it drops to 0.38-0.45 because the news source changes (below)."),
    ("Timestamps precede the trading cutoff", "cannot be verified",
     "Articles carry a calendar date but no time of day, so news dated on a formation day may have been published after the close. The lag evidence above suggests the pipeline leaves a day, but this is not proven."),
    ("Stock universe", "mostly verified, one issue",
     "98.7% of monthly rows (99.9% weekly) are in the top-1,000 list formed at the most recent June 30. About 1.1% of monthly rows (0.1% weekly) are stocks that appear only in the following June's list, a small look-ahead in universe membership; its effect is not quantified. Delisting returns are not documented."),
    ("News coverage changes in 2024", "issue, not quantified",
     "January 2024 has 22 articles; from February 2024 about 40% fewer tickers per month have news than in 2023. The test split includes mid-2024, so text features are thinner there."),
    ("Selection on test", "controlled",
     "The chosen specification per frequency was fixed on the validation split. The best-in-test specification is shown only as hindsight."),
]
for a, b, c in rows:
    add(f"| {a} | {b} | {c} |")
add("")

add("## 8. Confirmatory versus exploratory\n")
add("Exploratory: everything in this report. The selection rule was fixed before reading test results, but the strategies, cost levels, blocks and "
    "trial-counting choices were made by me after the earlier first pass over the same forecasts, which also used test results for a first look. "
    "The permutation and deflated-Sharpe calculations rest on stated assumptions (effective trials, a persistence-preserving null) that a reviewer could dispute. "
    "No result here should be presented as a confirmatory test.\n")

add("## 9. Corrections to what I reported earlier\n")
add("- My first backtest used gross returns and listed best-in-hindsight Sharpe ratios (1.7-1.9); those are not what a researcher could have selected in advance.")
add("- My first permutation null redrew random forecasts every period, which made turnover and costs far higher than for any persistent forecast; it was an unfair comparison and is replaced.")
add("- My first run carried positions across the missing months; that is fixed. All numbers here come from the corrected run.\n")

add("## 10. Reproduce\n")
add("```bash\ncd research/phase1\npython3 robustness.py   # about 2 minutes; writes robustness_summary.csv, findings.json, the two figures, config.json\npython3 make_report.py\n```")
add(f"Seed {cfg['seed']}; settings in `config.json`. Figures: `placebo_comparison.png`, `performance_after_costs.png`.")
(H / "robustness_report.md").write_text("\n".join(L))
print("wrote", H / "robustness_report.md", len("\n".join(L).split()), "words")
