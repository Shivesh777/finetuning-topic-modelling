"""POST-HOC DIAGNOSTIC (not confirmatory). Written after the CL/GC pilot showed that gold's next-week shock is related to this week's
return (timing check, test window). Question: does attention follow price, or is the pattern produced by how the shocks are built?

Reads weekly_panel.parquet and changes nothing in it. Reports all three windows, not only the test window. Oil is shown beside gold
to see whether the pattern is specific to gold. Output: posthoc_gold.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
panel = pd.read_parquet(HERE / "weekly_panel.parquet")
CASES = [("GC", "F_gold"), ("GC", "F_fed"), ("CL", "N_oil"), ("CL", "F_oil")]
WIN = {"dev": "development 2015-2019", "val": "validation 2020-2021", "test": "test 2022-2024"}
L = ["# Post-hoc diagnostic: does news attention follow price? (gold, with oil for comparison)\n",
     "**Status: exploratory and post hoc.** Written after seeing that the gold timing check failed in the test window of the pre-specified pilot. "
     "It does not change, replace or re-test the pilot's primary results, which stand as reported. Shocks and returns are those of `weekly_panel.parquet`; "
     "shocks are in development-sample standard deviations; returns in percent. HAC (Newey-West, 4 lags) errors. All three windows are shown.\n"]


def lagged(df, col, k):
    """value of `col` k weeks later (k may be negative), only if the rows are consecutive weeks"""
    out = df[col].shift(-k)
    gap = (df.L.shift(-k) - df.L).dt.days
    ok = gap.between(7 * k - 3, 7 * k + 3) if k > 0 else gap.between(7 * k - 3, 7 * k + 3)
    return out.where(ok) if k != 0 else df[col]


def prep(root):
    d = panel[panel.root == root].sort_values("L").reset_index(drop=True).copy()
    d["y"] = d.r * 100
    d["abs_y"] = d.y.abs()
    return d


L.append("## 1. Cross-correlation of weekly return with the contemporaneous-shock series at different leads and lags\n")
L.append("corr(return in week w, shock of week w+k). k > 0 means the shock comes after the return (attention following price); "
         "k < 0 means the shock precedes the return. Approximate 95% band for zero: +/- 1.96 / sqrt(n).\n")
for root, n in CASES:
    d = prep(root)
    L.append(f"\n**{root} / {n}**\n")
    L.append("| window | n | k=-2 | k=-1 | k=0 | k=+1 | k=+2 | band |\n|---|---|---|---|---|---|---|---|")
    for w in ["dev", "val", "test"]:
        s = d[d.window == w]
        cells = []
        for k in [-2, -1, 0, 1, 2]:
            z = lagged(d, f"Z_{n}_std", k).loc[s.index]
            cells.append(f"{pd.concat([s.y, z], axis=1).dropna().corr().iloc[0, 1]:+.2f}")
        L.append(f"| {WIN[w]} | {len(s)} | " + " | ".join(cells) + f" | +/-{1.96 / np.sqrt(len(s)):.2f} |")

L.append("\n## 2. Autocorrelation of the shock series itself (is part of the pattern mechanical?)\n")
L.append("The weekly shock sums daily deviations from a 5-day trailing mean, so adjacent weeks share baseline days and are mechanically correlated.\n")
L.append("| series | window | corr(Z_w, Z_w+1) | corr(Z_w, Z_w+2) |\n|---|---|---|---|")
for root, n in CASES:
    d = prep(root)
    for w in ["dev", "val", "test"]:
        s = d[d.window == w]
        a = [pd.concat([s[f"Z_{n}_std"], lagged(d, f"Z_{n}_std", k).loc[s.index]], axis=1).dropna().corr().iloc[0, 1] for k in (1, 2)]
        L.append(f"| {root} / {n} | {WIN[w]} | {a[0]:+.2f} | {a[1]:+.2f} |")


def hac(y, X):
    m = pd.concat([y, X], axis=1).dropna()
    f = sm.OLS(m.iloc[:, 0], sm.add_constant(m.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return f, len(m)


L.append("\n## 3. Does next week's attention respond to this week's return, given the shock's own persistence?\n")
L.append("Z(w+1) = a + b1 Z(w) + b2 return(w) + b3 |return(w)| + e. b2 is the response to the signed return, b3 to its size (attention may rise after any large move).\n")
L.append("| series | window | n | b1 (own persistence) | b2 (signed return, per 1 pp) [95% CI] | b3 (|return|, per 1 pp) [95% CI] |\n|---|---|---|---|---|---|")
for root, n in CASES:
    d = prep(root)
    d["Znext"] = lagged(d, f"Z_{n}_std", 1)
    d["Z0"] = d[f"Z_{n}_std"]
    for w in ["dev", "val", "test"]:
        s = d[d.window == w]
        f, k = hac(s.Znext, s[["Z0", "y", "abs_y"]])
        ci = f.conf_int()
        c = lambda v: f"{f.params[v]:+.3f} [{ci.loc[v, 0]:+.3f}, {ci.loc[v, 1]:+.3f}]"
        L.append(f"| {root} / {n} | {WIN[w]} | {k} | {f.params['Z0']:+.2f} | {c('y')} | {c('abs_y')} |")

L.append("\n## 4. The pilot's timing check, repeated while controlling for the contemporaneous shock\n")
L.append("return(w) = a + c1 Z(w) + c2 Z(w+1) + price controls. If c2 is only the mechanical overlap between adjacent shocks, it should shrink once Z(w) is in the model.\n")
L.append("| series | window | n | c1 on Z(w) [95% CI] | c2 on Z(w+1) [95% CI] | c2 in the pilot's timing check (Z(w+1) alone) |\n|---|---|---|---|---|---|")
for root, n in CASES:
    d = prep(root)
    d["Z0"] = d[f"Z_{n}_std"]
    d["Zn"] = lagged(d, f"Z_{n}_std", 1)
    for w in ["dev", "val", "test"]:
        s = d[d.window == w]
        f, k = hac(s.y, s[["Z0", "Zn", "mkt_std", "r_lag_std", "vol4_prev_std"]])
        g, _ = hac(s.y, s[["Zn", "mkt_std", "r_lag_std", "vol4_prev_std"]])
        ci = f.conf_int()
        c = lambda v: f"{f.params[v]:+.3f} [{ci.loc[v, 0]:+.3f}, {ci.loc[v, 1]:+.3f}]"
        L.append(f"| {root} / {n} | {WIN[w]} | {k} | {c('Z0')} | {c('Zn')} | {g.params['Zn']:+.3f} |")

L.append("\n## 5. How to read this\n")
L.append("Written by hand after reading sections 1-4 (numbers quoted from them).\n")
L.append("- **Part of the pattern is mechanical.** Weekly shocks are negatively autocorrelated at lag 1 in every series and window (-0.27 to -0.53) because "
         "adjacent weeks share baseline days. A lead-lag check should therefore hold Z(w) fixed; the pilot's timing check did not.")
L.append("- **Gold.** The link between this week's return and next week's gold shock is absent in 2015-2021 (cross-correlation at k=+1: +0.03 and +0.02) and appears only "
         "in 2022-2024 (+0.22, band +/-0.16). With Z(w) in the model the coefficient on Z(w+1) stays at +0.41 [+0.09, +0.73] in the test window (+0.39 without it), "
         "so mechanical overlap does not explain it. Attention responds to the signed return only in the test window (+0.10 sd per percentage point, [+0.03, +0.17]); "
         "development and validation show about zero, and the response to the size of the return is about zero throughout.")
L.append("- **Oil.** No consistent response of attention to returns: about +0.01 and not significant in validation and test. The negative lead correlations in "
         "development are what a positive contemporaneous link combined with the negative shock autocorrelation would produce.")
L.append("- **Reading.** Consistent with gold attention following gold rallies in 2022-2024, but not established: one window of three, among dozens of unadjusted post-hoc "
         "estimates. The pilot's gold results are unchanged (no contemporaneous relation, no forecasting power). If this is pursued, the pre-specified check should include Z(w).")
(HERE / "posthoc_gold.md").write_text("\n".join(L))
print("\n".join(L))
