"""Writes final_methodology_review.md: the final methodological and reproducibility review. No new estimation except what is
read from ipca_cw_calibration.csv (a calibration check of the already-frozen Clark-West statistic)."""
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
cal = pd.read_csv(H / "ipca_cw_calibration.csv")
res = pd.read_csv(H / "ipca_results.csv")
oos = pd.read_csv(H / "ipca_oos.csv")
A = ["CL", "GC", "HG", "BTC"]
LBL = {"CL": "WTI crude", "GC": "Gold", "HG": "Copper", "BTC": "Bitcoin"}
f3 = lambda x: f"{x:.3f}"


def holm(p):
    o = np.argsort(p)
    out = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        out[i] = min(1.0, run)
    return out


L = ["# Final methodological and reproducibility review\n",
     "Scope: a review of what was already run. No new model, specification, asset or strategy was tried. The only computation added is a calibration check of the "
     "already-frozen Clark-West statistic (`ipca_cw_calibration.py`), which reproduces the frozen gains, statistics and p-values exactly.\n",
     "## 1. Is the Clark-West test appropriate for the contemporaneous explanatory comparison?\n",
     "**Short answer: usable as an approximate descriptive test, with caveats, and it should not be the only inference reported.**\n",
     "- Clark and West (2007) designed the statistic for comparing the forecasts of nested models, adjusting the mean squared error difference for the estimation noise that the extra regressors add under the null. "
     "That adjustment depends on the nesting and on parameters being estimated on earlier data, not on whether the regressors are lagged, so its logic carries over to our fixed-scheme comparison in which the factor model is fitted on earlier months and applied to the test months with the realised same-month factors. "
     "What does not carry over is the interpretation: the result is a statement about explaining returns, not forecasting them.",
     "- Its normal approximation is justified when the test sample is small relative to the estimation sample. Here the test sample is 21 months and the fit samples are 59 (crude, gold, copper) or 28 months (Bitcoin and the common-month analysis), so that condition is not comfortably met, and the months are not contiguous.",
     "- Check: under the null that the factors add nothing (wild-bootstrap returns built from the fitted benchmark, observed market and factor values held fixed, 5,000 draws, seed 0, the full fit-and-test procedure repeated), the analytic one-sided p-value at a nominal 5% rejects "
     f"{cal.size_at_5pct.min():.1%} to {cal.size_at_5pct.max():.1%} of the time across the eight asset and sample combinations (nominal 2.5%: {cal.size_at_2_5pct.min():.1%} to {cal.size_at_2_5pct.max():.1%}). It is not badly miscalibrated, but it is somewhat liberal in several cases and conservative for crude.\n",
     "Primary held-out family (four assets, primary model, asset-specific fit sample), three ways of computing the inference:\n",
     "| asset | gain | analytic Clark-West p (pre-specified) | Holm (4) | calibrated Clark-West p | Holm (4) | calibrated p of the gain itself (post hoc) | Holm (4) |", "|---|---|---|---|---|---|---|---|"]
g = cal[cal.scope == "asset-specific"].set_index("asset").loc[A]
gh = holm(g.p_calibrated_gain.values)
for a, h in zip(A, gh):
    r = g.loc[a]
    L.append(f"| {LBL[a]} | {r.gain_obs:.3f} | {f3(r.p_analytic)} | {f3(r.p_holm_analytic)} | {f3(r.p_calibrated_cw)} | {f3(r.p_holm_calibrated_cw)} | {f3(r.p_calibrated_gain)} | {f3(h)} |")
gc = cal[cal.scope == "common-month"].set_index("asset").loc[A]
L.append("\nSame, common-month fit sample (28 months for every asset):\n")
L.append("| asset | gain | analytic p | Holm (4) | calibrated Clark-West p | Holm (4) | calibrated p of the gain | Holm (4) |\n|---|---|---|---|---|---|---|---|")
ghc = holm(gc.p_calibrated_gain.values)
for a, h in zip(A, ghc):
    r = gc.loc[a]
    L.append(f"| {LBL[a]} | {r.gain_obs:.3f} | {f3(r.p_analytic)} | {f3(r.p_holm_analytic)} | {f3(r.p_calibrated_cw)} | {f3(r.p_holm_calibrated_cw)} | {f3(r.p_calibrated_gain)} | {f3(h)} |")
L.append("\n**Material finding.** The conclusion that no pre-specified held-out test survives correction holds for the pre-specified analytic Clark-West statistic (smallest Holm p 0.103) and for the simulation-calibrated Clark-West statistic (0.066). "
         f"A post hoc test that uses the gain itself as the statistic puts the Holm-adjusted p-value for crude oil at {gh[0]:.3f}. The crude-oil result is therefore **borderline, and the verdict depends on the choice of statistic and reference distribution**; it cannot be called clearly null or clearly established. "
         "The paper reports all three and treats the pre-specified one as primary. Other facts that keep us from calling it robust: 21 test months; the interval for the gain reaches zero; the common-month version is weaker (Holm 0.162 even for the gain-based version); the NoLBERT factor does not reproduce it; and the FinBERT representation is not point-in-time.\n")

L.append("## 2. The pre-registered joint coefficient tests are reported separately from the held-out squared-error comparison\n")
L.append("The joint tests below are the original specification's four primary tests (primary model, M2, factor coefficients jointly zero, estimated **on the 21 test months themselves**). "
         "After review they are **descriptive**: coefficients are re-estimated inside the test sample, so they are not held-out evidence. The held-out squared-error comparison of section 1 is the confirmatory evidence. The two answer different questions and must not be combined or interchanged.\n")
L.append("| asset | n (test months) | joint Wald p, HC3 | joint p, calendar-aware Newey-West | Holm (4), HC3 |\n|---|---|---|---|---|")
t = res[(res.role == "primary") & (res["sample"] == "test") & res.asset.isin(A)].set_index("asset").loc[A]
for a in A:
    L.append(f"| {LBL[a]} | {int(t.loc[a, 'n'])} | {f3(t.loc[a, 'joint_p_hc3'])} | {f3(t.loc[a, 'joint_p_nw_calendar'])} | {f3(t.loc[a, 'joint_p_hc3_holm_4_descriptive'])} |")
L.append("\nNone is significant. The calendar-aware Newey-West p-value for Bitcoin (0.031) is small unadjusted but the test is descriptive, it uses 21 observations with five parameters, and it does not survive the Holm adjustment across four assets (HC3-based).\n")

L.append("## 3. Leakage and information-restriction audit of the IPCA factors and their inputs\n")
L.append("Status: **verified** against saved files, **documented** in the team's code snapshot (`~/Downloads/nolbert-ftm-runnable`, which predates the `_s63` option, so it documents intent, not the exact run), **assumed**, or **issue**.\n")
L.append("| item | status | detail |\n|---|---|---|")
rows = [
 ("Factor estimation schedule", "documented and consistent with saved metadata", "Every saved `summary.json` states: grid fitted on training periods, chosen on validation MSE, refitted once on training + validation, applied unchanged to the test periods. The code documents that tuning never sees test data and that validation forecasts come from a training-only fit. Saved `factors.csv` marks train and validation factor months in-sample and test months out-of-sample; the team's own reproduction of the saved test forecasts agrees to 1e-9 (29 of 50 specifications independently reproduced here)."),
 ("Factor returns computed with frozen loadings", "verified (consistency)", "Factors recomputed from `gamma.csv` and the panel match the saved ones (correlation above 0.999 in four specifications checked). This shows consistency, not that test returns were unused in fitting; that rests on the documented schedule."),
 ("Feature scaling", "verified", "All characteristics are rank-normalised across stocks within each date: no statistic from later dates enters."),
 ("Characteristic windows", "verified on a sample", "Covariance exposures use a window ending at the formation date (checked against a rebuild from daily attention and returns; rank correlation 0.92, falling to 0.72 and 0.41 when shifted 21 and 63 days forward)."),
 ("Topic models, vocabulary", "documented, verified from metadata", "Fitted on training-period articles only (through 2019-12-31); vocabulary built on training-period text."),
 ("NoLBERT representation", "point-in-time pretraining; adaptation window unresolved", "Pretrained only on 1976-1995 text. Adapted by headline-lead contrastive matching on training-period news, no returns, per the code documentation. Our project notes give an adaptation window of January 2015 to August 2020, which conflicts with training-period only (ends 2019); not reconciled."),
 ("FinBERT representation", "issue: not point-in-time", "Class probabilities from a classifier published in December 2022 and fine-tuned on finance tweets released in September 2022 (the team's own label: not point-in-time). The primary IPCA model and two of the three validation-selected equity specifications use it."),
 ("Shock window", "issue: configuration mismatch", "`_s63` is a 63-day attention-shock window chosen on validation for monthly rebalancing. All monthly analyses here, including the saved monthly stock forecasts, use the default 5-day window. The 63-day factor files exist but were not analysed."),
 ("News sources and sample", "documented", "FNSPID (Nasdaq articles through 2023) and FinMultiTime (article list; 2024 text); 1,255,289 unique articles from 2012 to 2024; one day with a data dump (2023-12-16) removed; 22 articles in January 2024; about 4,500 articles of 2024 have headlines only. The 2024 part of the test window uses a different source."),
 ("Calendar alignment", "verified", "Holding window = following calendar month; matches compounded stock returns for all 104 months; windows of consecutive months do not overlap."),
 ("Stock universe", "verified, one issue", "98.7% of monthly rows are in the list formed at the latest June 30; about 1.1% belong only to the next June's list (small look-ahead in membership, not quantified)."),
 ("In-sample versus out-of-sample factor months", "design note", "Coefficients of asset returns on factors are fitted on months whose factors are in-sample with respect to the loadings, and applied to test months whose factors are out-of-sample; this works against finding a held-out gain."),
 ("Months without a futures return", "documented", "Futures data start in 2015 (Bitcoin December 2017); 24 early training months have no commodity return; no extension was downloaded."),
 ("Intraday ordering of news", "cannot be verified", "Articles carry a date but no time of day; the forecasting design skips the last trading day of the week."),
]
for a, b, c in rows:
    L.append(f"| {a} | {b} | {c} |")
L.append("\n## 4. Consistency of numbers, intervals, sample sizes and corrections with the frozen results\n")
L.append("- Every number in the paper is inserted by `research/paper/build_paper.py` from the result files (`findings.json`, `robustness_summary.csv`, `cross_asset_primary.csv`, `pilot_results.csv`, `btc_results.csv`, `ipca_oos.csv`, `ipca_results.csv`, `ipca_coefficients.csv`, `ipca_cw_calibration.csv`); the build fails if a placeholder is left unfilled.")
L.append("- Multiple-testing families are stated where used: eight primary weekly topic-shock tests (Holm); four primary held-out IPCA tests (Holm); 12 primary IPCA coefficients (Bonferroni intervals, descriptive).")
L.append("- The Holm p-value of the CL contemporaneous topic-shock result is 0.229 with eight tests (earlier 0.163 with six); the paper uses 0.229.\n")
L.append("## 5. Phase 1 and Phase 2 compared fairly\n")
L.append("They differ in universe, frequency, target, evaluation and inference (a Sharpe ratio of a traded portfolio versus a squared-error gain of an explanatory regression), and in whether the information precedes the return. The paper compares what kind of claim each supports and does not rank them.\n")
L.append("## 6. Issues that need correction or disclosure before submission\n")
L.append("1. Report the crude-oil result as borderline with all three inference versions (section 1), not as simply non-significant.")
L.append("2. Disclose that FinBERT is not point-in-time and that the only model showing the crude effect uses it.")
L.append("3. Disclose the 5-day versus 63-day mismatch; do not describe our monthly models as the team's validated monthly configuration.")
L.append("4. Resolve or disclose the conflicting encoder adaptation windows.")
L.append("5. Keep the descriptive joint tests and the held-out comparison in separate tables.")
L.append("6. Disclose the 2024 news-source change and the partly headline-only 2024 text.")

# ======================================================================================= ADDENDUM
rob = pd.read_csv(H / "ipca_robustness_results.csv")
rt = (H / "reconstruction_test.md").read_text()
def rr(spec, asset, col, an="A", scope="asset-specific"):
    d = rob[(rob.spec == spec) & (rob.asset == asset) & (rob.scope == scope)]
    d = d[d.analysis.str.startswith(an)] if an in ("A", "B") else d[d.analysis == "original"]
    return d.iloc[0][col]
s63 = [l for l in rt.splitlines() if "_s63__sipca" in l and l.startswith("|")]
p1 = [l for l in rt.splitlines() if l.startswith(("| nolbert20 |", "| finbert |", "| lda_all |"))]
FB, NB, LD = "narr_finbert_s63__sipca", "narr_nolbert20_s63__sipca", "narr_lda_all_s63__sipca"
ci = lambda spec, a, an="A": f"[{rr(spec, a, 'gain_lo', an):.2f}, {rr(spec, a, 'gain_hi', an):.2f}]"
L.append("\n---\n\n# Addendum (after the post hoc robustness round)\n")
L.append("Specification of the robustness round: `ipca_robustness_prereg.md` (SHA-256 `ca3697bb...`); results: `ipca_robustness_results.md`; reconstruction: `reconstruction_test.md`. "
         "No original result was changed. Everything in this addendum is post hoc: the 21-month holdout had already been examined.\n")
L.append("## A1. 63-day reconstruction checks\n")
L.append(f"- The `_s63` feature blocks are not saved, so the project's feature construction was re-implemented from saved inputs (daily attention sums, CRSP daily returns, Fama-French market). "
         f"It reproduces the saved 5-day feature blocks with a minimum per-date rank correlation of {min(float(l.split('|')[6]) for l in p1):.4f} on the {float(p1[0].split('|')[4].strip().rstrip('%')):.1f}% of rows with complete exposures "
         "(the rest are filled by random donor rows in the project's pipeline and cannot be reproduced exactly).")
L.append(f"- With the window set to 63 days, the reconstruction reproduces the saved 63-day factor returns with correlation at least {min(float(l.split('|')[5]) for l in s63):.4f} "
         f"(test months at least {min(float(l.split('|')[7]) for l in s63):.4f}) for all three eligible specifications. This verifies the 63-day factors through their returns, not the unsaved feature blocks themselves.")
L.append("- Information restriction, verified in code: shocks are attention minus a strictly past trailing mean (`rolling(window).mean().shift(1)`), and exposures for a portfolio formed at the close of day f use news-return pairs up to day f - 1 (`NEWS_LAG_DAYS = 1`), because article dates carry no time of day.")
L.append("- The meaning of `_s63` (a 63-day window chosen on validation for monthly rebalancing) comes from a project member's message; the code snapshot predates the option.\n")
L.append("## A2. Post hoc robustness findings\n")
L.append("| specification | crude gain [95% CI] | Holm (4) | Holm (12) analytic / calibrated |\n|---|---|---|---|")
L.append(f"| FinBERT-20, 5-day, K=3 (original primary) | {rr('narr_finbert__sipca','CL','gain','orig'):.2f} {ci('narr_finbert__sipca','CL','orig')} | {rr('narr_finbert__sipca','CL','holm4_analytic','orig'):.3f} | n/a |")
for spec, lab in [(FB, "FinBERT-20, 63-day, K=3"), (NB, "NoLBERT-20, 63-day, K=3"), (LD, "LDA-60, 63-day, K=3")]:
    L.append(f"| {lab} | {rr(spec,'CL','gain'):.2f} {ci(spec,'CL')} | {rr(spec,'CL','holm4_analytic'):.3f} | {rr(spec,'CL','holm12_analytic'):.3f} / {rr(spec,'CL','holm12_calibrated'):.3f} |")
L.append(f"\n- 2024 exclusion (original primary, frozen coefficients, 14 test months): crude gain {rr('narr_finbert__sipca','CL','gain','B'):.2f} {ci('narr_finbert__sipca','CL','B')}, Holm {rr('narr_finbert__sipca','CL','holm4_analytic','B'):.3f} (analytic), {rr('narr_finbert__sipca','CL','holm4_calibrated','B'):.3f} (calibrated). The crude result does not depend on the affected months; power is lower.")
L.append(f"- NoLBERT at 63 days shows a Bitcoin gain of {rr(NB,'BTC','gain'):.2f} {ci(NB,'BTC')}, Holm (4) {rr(NB,'BTC','holm4_analytic'):.3f}, Holm (12) {rr(NB,'BTC','holm12_analytic'):.3f}; its loadings have no crypto content and it is not interpreted.")
L.append("- Gains for copper, gold and the equity market are small or negative in all three 63-day specifications.\n")
L.append("## A3. FinBERT versus NoLBERT\n")
L.append("At K = 3 and the 63-day window the two 20-topic representations are directly comparable, which removes the K confound of the original 5-day comparison (K = 3 versus K = 1). "
         "The crude-oil link appears with FinBERT in both windows and not with NoLBERT in either; LDA shows a small gain that is not significant after adjustment. The result is therefore **representation-dependent**. "
         "Whether that reflects better class definitions in FinBERT or information the classifier acquired after much of the sample cannot be distinguished with these data. Under FinBERT-63, as under FinBERT-5, the factor most associated with crude oil is not the one that loads on the Energy/Oil class.\n")
L.append("## A4. Remaining timing limitations\n")
L.append("| item | status | consequence |\n|---|---|---|")
L.append("| FinBERT classifier | not point-in-time (published December 2022; tweets of September 2022) | every FinBERT-based result, including the primary model, is not a clean historical test |")
L.append("| NoLBERT encoder adaptation | **resolved from project documentation**: pairs drawn up to 2020-09-01 (the training end of an earlier pipeline), so January-August 2020 validation text entered the adaptation; text only, no returns | validation-period numbers that use NoLBERT features are affected; the 2022-2024 test period is not. The training log itself is not in the local files, so this rests on the project's summary. Clean fix: re-adapt with the 2019-12-31 cutoff (not done) |")
L.append("| Vocabulary basis | the 10,000-term basis comes from the whole corpus for every representation (fitting, centroids and scaling are training-only) | small look-ahead in LDA inputs and topic descriptions; not in NoLBERT clustering. LDA is therefore historically more compatible, not strictly point-in-time |")
L.append("| Feb-Jun holding months | documented: from 2017 the daily return file holds only each year's June-list stocks, so earlier formation days are left out | removes a look-ahead; the test sample is 21 of 36 months and speaks only to holding months January and July-December |")
L.append("| 63-day stock forecasts | not rebuilt | Phase 1 monthly results remain on the 5-day window |")
L.append("\n## A5. Status of the issues listed in section 6\n")
L.append("1. Crude described as borderline with all inference versions: done; now also described as representation-dependent.")
L.append("2. FinBERT not point-in-time: disclosed in abstract, introduction, data, results table and limitations.")
L.append("3. 5-day versus 63-day: disclosed; post hoc 63-day factor analysis done; stock forecasts not rebuilt.")
L.append("4. Encoder window: resolved from project documentation (see A4).")
L.append("5. Joint tests separate from the held-out comparison: done (separate table, labelled descriptive).")
L.append("6. 2024 source change: disclosed; descriptive exclusion check done.")

(H / "final_methodology_review.md").write_text("\n".join(L))
print("wrote final_methodology_review.md")
