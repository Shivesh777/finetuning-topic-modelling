"""Are the validated portfolios' Sharpe ratios distinguishable from zero and from the
price-only benchmark? Inference for the paper's headline table.

    python scripts/13_sharpe_tests.py

For the experiment that validation picks in each family (results/tables/selection.csv:
"any", the headline; "shock", the tuned shock windows; "news", the tone and embedding sets;
"price", the no-text benchmark) and for the long-short rules ls_ew, ls_buf and ls_vw,
on the test period, net of costs:

  sharpe_net, n          annualised Sharpe ratio and number of periods
  t_mean, p_mean         Newey-West t of the mean return, fixed-b p-value (inference.mean_test)
  psr                    probability that the true Sharpe ratio is above zero (Bailey and
                         Lopez de Prado 2012), allowing for skewness and kurtosis
  dsr, trials            deflated Sharpe ratio (Bailey and Lopez de Prado 2014): the same
                         probability against the best Sharpe ratio expected from `trials`
                         unskilled experiments, where trials = every experiment of the
                         frequency under that rule that held a position in the test period
                         (the "all" pool of multiple_testing.csv less flat forecasts) and
                         their dispersion is that of their observed test Sharpe ratios
  vs = <baseline>        diff_sharpe, and its Ledoit and Wolf (2008) HAC test (lw_t, lw_p,
                         inference.sharpe_difference_hac) and a paired stationary-bootstrap
                         p-value (boot_p, inference.sharpe_difference) over the common periods

Also, without any selection, the combinations across topic models of
scripts/06d_grand_ensemble.py (price and text: "grand_s63", "grand_s5"; text alone:
"grand_text_s63", "grand_text_s5") and the ensemble of the eight models on price alone
("price_ens").

Baselines: the price-only pick under the same rule ("price"), the headline pick under the
same rule ("headline", for the shock and news picks and the combinations), for the
combinations also the price ensemble ("price_ens"), the 63-day combinations against their
5-day versions, grand_s63 against grand_text_s63 (what the price features add), and for
the other rules the same experiment's ls_ew ("ls_ew"). The models stay frozen: only the
realised test returns are random here, so the tests ask whether 36 months, 78 fortnights
or 156 weeks can tell these Sharpe ratios apart from zero and from each other.

Writes results/tables/sharpe_tests.csv.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config
from ftm.evaluation import inference, metrics
from ftm.evaluation.report import family

FAMILIES = ("any", "shock", "news", "price")
FIXED = {"grand_s63": ("grand_s63", "ens"), "grand_s5": ("grand_s5", "ens"),
         "grand_text_s63": ("grand_text_s63", "ens"), "grand_text_s5": ("grand_text_s5", "ens"),
         "price_ens": ("price", "ens")}
RULES = ("ls_ew", "ls_buf", "ls_vw", "ls_buf_vw")
N_BOOT = 4000


def series(test: pd.DataFrame, features: str, model: str, rule: str) -> pd.Series:
    g = test[(test.features == features) & (test.model == model) & (test.strategy == rule)]
    return g.set_index("date")["excess"].sort_index()


def main():
    selection = pd.read_csv(ROOT / "results/tables/selection.csv")
    rows = []
    for freq in config.FREQUENCIES:
        ppy = metrics.PERIODS_PER_YEAR[freq]
        returns = pd.read_parquet(ROOT / f"results/series/{freq}_returns.parquet",
                                  columns=["split", "features", "model", "strategy", "date", "excess"])
        test = returns[returns.split == "test"]
        real = test[(test.features != "benchmark") & (test.features.map(family) != "control")]
        picks = selection[selection.freq == freq].set_index("family")
        for name, (features, model) in FIXED.items():
            picks.loc[name, ["features", "model"]] = [features, model]
        for rule in RULES:
            pool = real[real.strategy == rule].pivot_table(index="date", columns=["features", "model"],
                                                            values="excess")
            per_period = pool.mean() / pool.std(ddof=1)
            trials, sd_trials = int(per_period.notna().sum()), float(per_period.std(ddof=1))
            for fam in FAMILIES + tuple(FIXED):
                if fam not in picks.index:
                    continue
                pick = picks.loc[fam]
                r = series(test, pick.features, pick.model, rule)
                test_mean = inference.mean_test(r.to_numpy())
                deflated = inference.deflated_sharpe(r.to_numpy(), trials, sd_trials)
                base = {"freq": freq, "family": fam, "features": pick.features, "model": pick.model,
                        "rule": rule, "n": len(r), "sharpe_net": metrics.sharpe(r.to_numpy(), ppy),
                        "t_mean": test_mean["t"], "p_mean": test_mean["p"],
                        "psr": inference.probabilistic_sharpe(r.to_numpy()),
                        "dsr": deflated["dsr"], "trials": trials,
                        "dsr_threshold_annual": deflated["threshold"] * np.sqrt(ppy)}
                baselines = []
                if fam != "price":
                    baselines.append(("price", picks.loc["price"], rule))
                if fam in ("shock", "news", "price_ens") or fam.startswith("grand"):
                    baselines.append(("headline", picks.loc["any"], rule))
                if fam.startswith("grand"):
                    baselines.append(("price_ens", picks.loc["price_ens"], rule))
                if fam in ("grand_s63", "grand_text_s63"):
                    five = fam.replace("s63", "s5")
                    baselines.append((five, picks.loc[five], rule))
                if fam == "grand_s63":
                    baselines.append(("grand_text_s63", picks.loc["grand_text_s63"], rule))
                if rule != "ls_ew":
                    baselines.append(("ls_ew", pick, "ls_ew"))
                if not baselines:
                    rows.append(base)
                for name, other, other_rule in baselines:
                    b = series(test, other.features, other.model, other_rule)
                    common = r.index.intersection(b.index)
                    x, y = r.loc[common].to_numpy(), b.loc[common].to_numpy()
                    lw = inference.sharpe_difference_hac(x, y, ppy)
                    boot = inference.sharpe_difference(x, y, ppy, n_boot=N_BOOT, seed=0)
                    rows.append({**base, "vs": name, "vs_features": other.features, "vs_model": other.model,
                                 "vs_sharpe_net": metrics.sharpe(y, ppy),
                                 "diff_sharpe": metrics.sharpe(x, ppy) - metrics.sharpe(y, ppy),
                                 "lw_t": lw["t"], "lw_p": lw["p"], "boot_p": boot["p"],
                                 "common_periods": len(common)})
    out = pd.DataFrame(rows)
    path = ROOT / "results/tables/sharpe_tests.csv"
    out.to_csv(path, index=False, float_format="%.6g")
    print(f"{path.relative_to(ROOT)}: {len(out)} rows")


if __name__ == "__main__":
    main()
