"""Builds cross_asset_summary.md and cross_asset_coefficients.png from pilot_results.csv, btc_results.csv, cross_asset_primary.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

H = Path(__file__).resolve().parent
res = pd.concat([pd.read_csv(H / "pilot_results.csv"), pd.read_csv(H / "btc_results.csv")], ignore_index=True)
fam = pd.read_csv(H / "cross_asset_primary.csv")
sha = {k: res[res.root.isin(v)].prereg_sha256.iloc[0] for k, v in {"CL / GC": ["CL", "GC"], "BTC": ["BTC"]}.items()}
f = lambda x, d=2: "" if pd.isna(x) else f"{x:.{d}f}"
Z8 = stats.norm.ppf(1 - 0.025 / 8)
MDE_U, MDE_B = 1.96 + 0.8416, Z8 + 0.8416
NAME = {"N_oil": "oil (NoLBERT-60 t28+t30)", "F_oil": "oil (FinBERT Energy/Oil)", "F_gold": "gold (FinBERT Gold/Metals)",
        "F_fed": "Fed (FinBERT Fed/Central Banks)", "F_macro": "Macro (FinBERT)", "F_curr": "Currencies (FinBERT)"}
ASSET = {"CL": "WTI crude (CL)", "GC": "Gold (GC)", "BTC": "Bitcoin (BTC)"}
L = []
add = L.append

add("# Do news-attention shocks from frozen equity-news topic models move with, or predict, commodity and Bitcoin futures returns?\n")
add("**Status.** Exploratory cross-asset external-validity test. It uses topic-level attention shocks from the team's frozen text models; it is **not** a replication of "
    "the Sparse IPCA factor analysis, because the factor returns and the loading matrix are not available. Specifications were written before the regressions ran "
    f"(`pilot_prereg.md`, SHA-256 `{sha['CL / GC'][:16]}...` for CL and GC; `btc_prereg.md`, `{sha['BTC'][:16]}...` for Bitcoin) and each regression was run once. "
    "Confidence intervals are labelled unadjusted or Bonferroni-adjusted throughout.\n")

add("## 1. Hypotheses (two-sided; the shocks measure attention, not tone, so no sign was specified)\n")
add("| id | asset | narrative | question | status |\n|---|---|---|---|---|")
H_ = [("H1a/H1b", "CL", "oil (two definitions)", "Do oil-attention shocks co-move with same-week CL returns beyond market, lagged return and volatility?", "primary"),
      ("H2a/H2b", "CL", "oil (two definitions)", "Do shocks dated through the day before the last trading day predict next week's CL return beyond price controls?", "primary"),
      ("H3", "GC", "gold", "Do gold-attention shocks co-move with same-week GC returns?", "primary"),
      ("H4", "GC", "gold", "Do gold-attention shocks predict next week's GC return?", "primary"),
      ("H5", "BTC", "Fed", "Do Fed-attention shocks co-move with same-week BTC returns?", "primary, exploratory"),
      ("H6", "BTC", "Fed", "Do Fed-attention shocks predict next week's BTC return?", "primary, exploratory")]
for r in H_:
    add("| " + " | ".join(r) + " |")
add("\nThe eight primary tests form one family for multiple-testing adjustment. Windows: development 2015-2019 (Bitcoin 2018-2019), validation 2020-2021, test 2022-2024. "
    "Coefficients are percentage points of weekly return per one development-sample standard deviation of the shock; HAC (Newey-West, 4 lags) standard errors; "
    "149 test weeks for the contemporaneous tests and 147 for the forecasting tests.\n")

add("## 2. Primary tests in the test window\n")
add("| asset | narrative | analysis | coef | 95% CI, unadjusted | 95% CI, Bonferroni (8 tests) | raw p | Holm p (8 tests) | minimum detectable effect, 80% power: unadjusted / Bonferroni |")
add("|---|---|---|---|---|---|---|---|---|")
for r in fam.itertuples():
    add(f"| {r.root} | {NAME[r.narrative]} | {r.kind} | {f(r.coef)} | [{f(r.ci_lo)}, {f(r.ci_hi)}] | [{f(r.ci_bonf8_lo)}, {f(r.ci_bonf8_hi)}] | {f(r.p, 3)} | {f(r.p_holm_8, 3)} | {f(MDE_U * r.se)} / {f(MDE_B * r.se)} |")
c = fam[(fam.root == "CL") & (fam.narrative == "N_oil") & (fam.kind == "contemporaneous")].iloc[0]
add(f"\nAn unadjusted interval can exclude zero while the Holm-adjusted p-value does not: the interval makes no allowance for the other seven tests, the adjusted p-value does. "
    f"That is the case for the CL contemporaneous result using the NoLBERT definition (unadjusted [{f(c.ci_lo)}, {f(c.ci_hi)}], raw p {f(c.p, 3)}, Holm p {f(c.p_holm_8, 3)}; its Bonferroni interval is [{f(c.ci_bonf8_lo)}, {f(c.ci_bonf8_hi)}]). "
    "(In the earlier CL/GC-only report the family had six tests and this Holm p was 0.163; adding the two Bitcoin tests enlarges the family.) "
    f"The last column is the smallest true effect each test could detect with 80% power. Every estimate in the table is smaller than its minimum detectable effect, so these tests cannot rule out modest effects.\n")

add("## 3. Development, validation and test compared (own-window estimates, 95% CI unadjusted)\n")
add("| asset | narrative | analysis | development | validation | test | same sign in all three |")
add("|---|---|---|---|---|---|---|")
own = res[res.stage.str.contains("own-window") & res.variant.eq("main")]
for root, nar, kind in fam[["root", "narrative", "kind"]].itertuples(index=False):
    cells, signs = [], []
    for w in ["dev", "val", "test"]:
        r = own[(own.root == root) & (own.narrative == nar) & (own.kind == kind) & own.stage.str.startswith(w)].iloc[0]
        cells.append(f"{f(r.coef)} [{f(r.ci_lo)}, {f(r.ci_hi)}] (n={r.n})")
        signs.append(np.sign(r.coef))
    add(f"| {root} | {NAME[nar]} | {kind} | " + " | ".join(cells) + f" | {'yes' if len(set(signs)) == 1 else 'no'} |")

add("\n## 4. Contemporaneous versus predictive: out-of-sample usefulness (frozen coefficients, test window)\n")
add("Coefficients are fitted on development + validation and applied unchanged to the test weeks. Gain is the out-of-sample R-squared relative to the price-controls-only model, "
    "with a block-bootstrap 95% interval (unadjusted).\n")
add("| asset | narrative | analysis | gain vs controls-only [95% CI] | R2 vs expanding mean (full / controls-only) | hit rate (full / controls-only) |")
add("|---|---|---|---|---|---|")
oos = res[res.stage.str.startswith("test") & res.stage.str.contains("applied") & res.variant.eq("main") & res.role.eq("primary")]
for r in oos.itertuples():
    extra = f"{f(r.oos_r2_vs_hist_mean, 3)} / {f(r.oos_r2_controls_vs_hist_mean, 3)}" if r.kind == "forecasting" else "n/a"
    hit = f"{f(r.hit_rate_full)} / {f(r.hit_rate_controls)}" if r.kind == "forecasting" else "n/a"
    add(f"| {r.root} | {NAME[r.narrative]} | {r.kind} | {f(r.oos_r2_vs_controls, 3)} [{f(r.oos_r2_vs_controls_lo, 3)}, {f(r.oos_r2_vs_controls_hi, 3)}] | {extra} | {hit} |")
add("")

add("## 5. Secondary narratives and controls (test window, own-window, unadjusted; for context only, not tests of the hypotheses)\n")
add("| asset | narrative | role | analysis | coef | raw p | development coef | validation coef |")
add("|---|---|---|---|---|---|---|---|")
sec = own[own.role.isin(["secondary", "cross-narrative control"]) & own.stage.str.startswith("test")]
for r in sec.itertuples():
    g = lambda w: own[(own.root == r.root) & (own.narrative == r.narrative) & (own.kind == r.kind) & own.stage.str.startswith(w)].iloc[0].coef
    add(f"| {r.root} | {NAME[r.narrative]} | {r.role} | {r.kind} | {f(r.coef)} | {f(r.p, 3)} | {f(g('dev'))} | {f(g('val'))} |")
add("\nSecondary results show why choosing a narrative after the fact is unsafe: Bitcoin's Macro class has a development coefficient of -2.80 (p < 0.001) that is -0.63 in validation and +0.17 in test; "
    "the Currencies class is significant in validation and not in test. Neither was promoted to a primary result.\n")

add("## 6. Robustness\n")
d_ = res[res.variant.eq("drop_flagged") & res.stage.eq("test: own-window OLS")].set_index(["root", "narrative", "kind"])
m_ = fam.set_index(["root", "narrative", "kind"])
def holm_(p):
    o = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); out[i] = min(1.0, run)
    return out
dd = d_.loc[[tuple(x) for x in fam[["root", "narrative", "kind"]].values]].copy()
dd["holm8"] = holm_(dd.p.values)
hn = dd.loc[("CL", "N_oil", "forecasting")]
add("- **Databento degraded days** (14 days): dropping the weeks that contain them changes raw p-values. "
    f"For CL with the NoLBERT definition the contemporaneous raw p moves from {f(m_.loc[('CL','N_oil','contemporaneous'),'p'], 3)} to {f(d_.loc[('CL','N_oil','contemporaneous'),'p'], 3)}, "
    f"and the forecasting raw p from {f(m_.loc[('CL','N_oil','forecasting'),'p'], 3)} to {f(hn.p, 3)}, which would pass the eight-test Holm adjustment (adjusted {f(hn.holm8, 3)}). "
    "That sensitivity was specified in advance as a check and not as a selection device, and this coefficient (-1.5) has the opposite sign of its development estimate (+0.13), "
    "is not stable across windows, and weakens to p = 0.50 when t55 is added to the oil definition. It is reported, not relied on. No other primary test passes adjustment in this sensitivity.")
add("- **Adding NoLBERT topic t55 to the oil definition** moves the CL forecasting coefficient from -1.29 (p 0.013) to -0.32 (p 0.50) and the contemporaneous one from 1.31 to 0.94 (p 0.09): the oil result depends on the topic definition.")
cc = own[own.role.eq("cross-narrative control") & own.stage.str.startswith("test") & own.kind.eq("contemporaneous")].set_index(["root", "narrative"])
add(f"- **Cross-narrative controls** are as significant as some primary results: CL on the gold narrative has raw p {f(cc.loc[('CL','F_gold'),'p'], 3)} and GC on the oil narrative {f(cc.loc[('GC','F_oil'),'p'], 3)} in the test window. Effects with p-values of that size occur by chance in this setup and should not be read as evidence.")
add("- **Timing checks** (regress this week's return on next week's shock): near zero for CL and the Bitcoin Fed shock; not zero for gold (+0.39, p 0.001) and for the Bitcoin Currencies class. "
    "The post-hoc diagnostic (`posthoc_gold.md`) finds that weekly shocks are mechanically negatively autocorrelated, and that gold attention follows gold returns in 2022-2024 only (+0.10 sd per percentage point), not in 2015-2021. It is exploratory and leaves the primary gold results unchanged.")
add("- **Futures construction** was checked against GLD (weekly correlation 0.992, slope 1.001) and, for crude, against EIA WTI spot (weekly correlation 0.96 outside March-July 2020, 0.99 in the test window; the annual futures-minus-spot gap follows the shape of the curve; April 2020 handled by having rolled to the June contract before the May contract went negative). Details: `futures_validation.md`, `wti_validation.md`.\n")

add("## 7. What can and cannot be concluded\n")
add("- **Supported (weakly):** oil attention and CL returns move together in the same week with the same sign in all three windows and under both definitions, with a small gain in explained variance. "
    "It does not survive the multiple-testing adjustment, depends on the topic definition, and cannot be interpreted causally: news coverage and prices may both respond to the same supply and demand developments, or news may follow price.")
add("- **Not supported:** forecasting. No narrative-augmented model beats the expanding historical mean in the test window (out-of-sample R-squared against the mean: CL -5.8% and -6.4%, GC -0.4%, Bitcoin -1.5%). "
    "The CL NoLBERT oil narrative does improve on the price-controls-only model by 1.5% (interval [0.2%, 3.0%]), but both are worse than the mean, and its coefficient has the opposite sign in development.")
add("- **Not supported:** a gold or Bitcoin contemporaneous relation. Bitcoin's Fed shock has a negative contemporaneous coefficient of similar size in validation and test (-1.4, -1.2), but with intervals that include zero.")
add("- **No trading claim is made.** No strategy was built or tested on these signals.\n")

add("## 8. Limitations\n")
for s in [
    "Power: 104-149 weeks; the minimum detectable effects in section 2 are large relative to the estimates.",
    "Narratives: only oil has a well-supported topic (NoLBERT-60 t28 and t30; FinBERT Energy/Oil). The Fed class is 0.4% of articles and noisy; the Macro class does not match its label; NoLBERT-20 has no macro topic; inflation, growth and China have none. Bitcoin has no crypto topic.",
    "Look-ahead in the text models: topics were fit through 2019-12-31, but the NoLBERT encoder was adapted on text through 2020-08-31 (reported by the user), overlapping the validation window, and the FinBERT classifier's training data are undocumented.",
    "News data have calendar dates but no times; the forecasting signal skips the last trading day of the week to avoid overlap with the return window. News coverage per stock falls about 40% from 2024 and January 2024 is nearly empty (those weeks are dropped).",
    "Topic definitions were chosen from topic terms and headlines before returns were examined; the choice of t28 and t30 (and exclusion of t55) rests on those terms.",
    "Shocks measure attention and have a mechanical negative autocorrelation (-0.3 to -0.5 at one week).",
    "Bitcoin's development window is about 100 weeks; its results are exploratory.",
    "The sparse IPCA factors, the loading matrix and the original code were not available, so nothing here says whether the equity-derived factors transfer across assets."]:
    add("- " + s)

add("\n## 9. Files\n")
add("`pilot_prereg.md`, `btc_prereg.md` (specifications); `pilot.py`, `pilot_btc.py`, `build_weekly_panel.py` (code); `pilot_results.md/.csv`, `btc_results.md/.csv`, "
    "`cross_asset_primary.csv` (results); `posthoc_gold.md` (post-hoc); `futures_validation.md`, `wti_validation.md` (data checks); `cross_asset_coefficients.png`.")
(H / "cross_asset_summary.md").write_text("\n".join(L))

# figure
fig, ax = plt.subplots(figsize=(11, 5.2))
rows = list(fam[["root", "narrative", "kind"]].itertuples(index=False))
for j, (root, nar, kind) in enumerate(rows):
    for k, w in enumerate(["dev", "val", "test"]):
        r = own[(own.root == root) & (own.narrative == nar) & (own.kind == kind) & own.stage.str.startswith(w)].iloc[0]
        ax.errorbar(r.coef, j + (k - 1) * 0.24, xerr=[[r.coef - r.ci_lo], [r.ci_hi - r.coef]], fmt="o", color=f"C{k}", capsize=2, label=w if j == 0 else None)
    t = fam.iloc[j]
    ax.plot([t.ci_bonf8_lo, t.ci_bonf8_hi], [j + 0.24] * 2, color="C2", lw=5, alpha=0.25, label="test, Bonferroni (8) interval" if j == 0 else None)
ax.axvline(0, color="grey", lw=0.8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r} {n} - {k[:5]}." for r, n, k in rows])
ax.invert_yaxis()
ax.set_xlabel("percentage points of weekly return per 1 development-sd shock (95% CI, unadjusted unless marked)")
ax.set_title("Primary tests: development, validation, test")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(H / "cross_asset_coefficients.png", dpi=150)
print("wrote summary,", len(" ".join(L).split()), "words")
