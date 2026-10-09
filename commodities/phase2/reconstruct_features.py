"""Reproducibility test of the narrative features, in two parts. No regression of futures returns is run here.

Re-implements the team's construction (ftm/features/narrative.py, ftm/panel.py: trailing-mean attention shocks, news up to t-1,
exponentially weighted covariance exposures orthogonalised to the market, cross-sectional rank to [-1, 1]) from the saved daily
attention sums, CRSP daily returns and Fama-French market factor.

Part 1: window = 5 days. Compare with the SAVED feature block `panels/M/narr_<model>.npy` (rows with complete exposures).
Part 2: window = 63 days. The s63 feature blocks are not saved, so test the reconstruction through the SAVED s63 factor returns:
        f_t = least squares of next-period relative returns on (reconstructed features x saved loadings), compared with factors.csv.
        The same comparison is made for the 5-day model, where the saved features also exist, to show what agreement a faithful reconstruction gives.

    python3 reconstruct_features.py -> reconstruction_test.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
LAG, EXPO_WINDOW, HALFLIFE, MIN_OBS, MIN_ARTICLES = 1, 252, 60, 120, 20

idx = pd.read_parquet(DATA / "panels" / "M" / "index.parquet")
daily = pd.read_parquet(DATA / "crsp" / "daily.parquet", columns=["date", "permno", "ret"])
ff = pd.read_parquet(DATA / "crsp" / "ff_daily.parquet").set_index("date")
dates = pd.DatetimeIndex(sorted(daily.date.unique()))
R = daily.pivot(index="date", columns="permno", values="ret").reindex(dates)
perm_col = {p: i for i, p in enumerate(R.columns)}
RET, OBS = R.to_numpy(np.float64), R.notna().to_numpy()
MKT = ff.mktrf.reindex(dates).to_numpy(np.float64)
assert not np.isnan(MKT).any()


def attention(model):
    sums = pd.read_parquet(DATA / "topics" / model / "daily_sums.parquet").set_index("date")
    day = np.searchsorted(dates.values, sums.index.values, side="left")
    inside = day < len(dates)
    total = sums[inside].groupby(day[inside]).sum().reindex(range(len(dates)), fill_value=0.0)
    topics = total.drop(columns="n_articles")
    theta = topics.div(topics.sum(axis=1).where(total["n_articles"] >= MIN_ARTICLES), axis=0)
    theta.index = dates
    return theta


def shocks(theta, window):
    trailing = theta.rolling(window, min_periods=max(2, window - 2)).mean().shift(1)
    return np.array(theta - trailing, dtype=np.float64)


def exposures(z_all, f):
    end = f - LAG + 1
    lo = max(0, end - EXPO_WINDOW)
    z = z_all[lo:end]
    valid = ~np.isnan(z).any(axis=1)
    w = 0.5 ** ((end - 1 - np.arange(lo, end)) / HALFLIFE) * valid
    z = np.where(valid[:, None], z, 0.0)
    b = np.column_stack([np.ones(len(z)), MKT[lo:end]])
    bw = b * w[:, None]
    z = z - b @ np.linalg.lstsq(bw.T @ b, bw.T @ z, rcond=None)[0]
    obs = OBS[lo:end] & valid[:, None]
    r = np.where(obs, RET[lo:end], 0.0)
    wo = w[:, None] * obs
    wsum = wo.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rbar = (wo * r).sum(axis=0) / wsum
        cov = ((wo * (r - rbar)).T @ z) / wsum[:, None]
    cov[obs.sum(axis=0) < MIN_OBS] = np.nan
    return cov


def rank_transform(x):
    f = pd.DataFrame(x)
    n = f.notna().sum(axis=0).to_numpy()
    ranks = f.rank(axis=0, method="average").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0
    return s


def build(model, window):
    """(n_rows, K) rank features with NaN where the exposure is missing, in the order of panels/M/index.parquet."""
    z_all = shocks(attention(model), window)
    out = np.full((len(idx), z_all.shape[1]), np.nan)
    for dt, sub in idx.groupby("date"):
        f = dates.get_loc(dt)
        e = exposures(z_all, f)
        cols = np.array([perm_col[p] for p in sub.permno])
        out[sub.index.values] = rank_transform(e[cols])
    return out


L = ["# Reproducibility test of the narrative features\n",
     "Re-implementation of the team's construction from saved inputs (see `reconstruct_features.py`). Missing exposures are filled by random donor rows in the team's pipeline, "
     "so only rows with complete exposures can be compared exactly.\n"]

# ---------------- part 1: 5-day features against the saved block
L.append("## Part 1. 5-day window, against the saved feature block\n")
L.append("| model | topics | rows compared | share of all rows | mean per-date Spearman | minimum per-date Spearman | median absolute difference | share within 0.02 |\n|---|---|---|---|---|---|---|---|")
for model, block in [("nolbert20", "narr_nolbert20"), ("finbert", "narr_finbert"), ("lda_all", "narr_lda_all")]:
    mine = build(model, 5)
    saved = np.load(DATA / "panels" / "M" / f"{block}.npy").astype(np.float64)
    ok = ~np.isnan(mine).any(axis=1)
    rhos = []
    for dt, sub in idx[ok].groupby("date"):
        ii = sub.index.values
        rhos.append(np.nanmean([spearmanr(mine[ii, k], saved[ii, k])[0] for k in range(mine.shape[1])]))
    d = np.abs(mine[ok] - saved[ok])
    L.append(f"| {model} | {mine.shape[1]} | {int(ok.sum()):,} | {ok.mean():.1%} | {np.mean(rhos):.4f} | {np.min(rhos):.4f} | {np.median(d):.5f} | {(d < 0.02).mean():.1%} |")

# ---------------- part 2: factor-return reproduction
L.append("\n## Part 2. Saved factor returns reproduced from reconstructed features and saved loadings\n")
L.append("Factor return of month t = least-squares coefficient of next-period relative returns on (features x loadings). Comparison with the saved `factors.csv`.\n")
L.append("| specification | window | K | months | correlation with saved factors (min over factors) | slope (range) | test months: min correlation |\n|---|---|---|---|---|---|---|")
checks = [("narr_nolbert20__sipca", "nolbert20", 5), ("narr_nolbert20_s63__sipca", "nolbert20", 63),
          ("narr_finbert__sipca", "finbert", 5), ("narr_finbert_s63__sipca", "finbert", 63),
          ("narr_lda_all__sipca", "lda_all", 5), ("narr_lda_all_s63__sipca", "lda_all", 63)]
for spec, model, window in checks:
    g = pd.read_csv(DATA / "results" / "ipca" / "M" / spec / "gamma.csv")
    sv = pd.read_csv(DATA / "results" / "ipca" / "M" / spec / "factors.csv", parse_dates=["date"]).set_index("date")
    fc = [c for c in sv.columns if c.startswith("f") and c[1:].isdigit()]
    X = build(model, window)
    G = g[fc].to_numpy()
    mine = {}
    for dt, sub in idx.groupby("date"):
        ii = sub.index.values
        Xi = np.nan_to_num(X[ii], nan=0.0)             # a missing exposure enters as 0 (neutral); random donors are not reproduced
        Zi = Xi @ G
        mine[dt] = np.linalg.lstsq(Zi, sub.y_rel.to_numpy(), rcond=None)[0]
    m = pd.DataFrame(mine, index=fc).T
    cors = [m[c].corr(sv[c]) for c in fc]
    slopes = [np.polyfit(m[c], sv[c], 1)[0] for c in fc]
    test = sv.split == "test"
    tcor = [m.loc[test.values, c].corr(sv.loc[test, c]) for c in fc]
    L.append(f"| `{spec}` | {window} | {len(fc)} | {len(m)} | {min(cors):.4f} | {min(slopes):.2f} to {max(slopes):.2f} | {min(tcor):.4f} |")

L.append("\nReading: part 1 shows whether the re-implementation reproduces features that exist in the saved data; part 2 shows how well it reproduces the saved factor returns, for the 5-day models (where part 1 applies) and for the 63-day models (where it is the only check available).")
(HERE / "reconstruction_test.md").write_text("\n".join(L))
print("\n".join(L))
