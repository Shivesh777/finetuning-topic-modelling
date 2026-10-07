"""Open the glass box: what a fitted EBM uses, and what happens when a term is deleted.

    python scripts/09_explain.py [--freq M 2W W] [--features all_lda]

For the explainable boosting machine of one experiment per frequency (by default
the `all_...` feature set whose EBM has the highest validation rank IC), following
Bell, Kakhbod, Lettau and Nazemi (2024):

    importance.csv   S(i), the mean absolute contribution of each term over the rows
                     the model was fitted on, with the term's feature block
    blocks.csv       the same summed by block (price, narrative exposure, firm news)
    shapes.csv       the shape function of the most important main effects, as steps
    ablation.csv     test-period R2, rank IC and long-short Sharpe ratio of the full
                     model and after deleting a block or one of the top terms from it,
                     WITHOUT re-estimating anything

Written to results/explain/<freq>__<features>/.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from ftm import config
from ftm.evaluation import metrics
from ftm.experiments import feature_sets
from ftm.panel import Panel
from ftm.portfolio import strategies

TOP_TERMS = 10
SHAPES = 12


def default_features(freq: str) -> str:
    """The all_... feature set whose EBM did best in the validation period."""
    table = pd.read_csv(config.RESULTS / "tables" / "forecast.csv")
    table = table[(table["freq"] == freq) & (table["split"] == "validation")
                  & (table["model"] == "ebm") & table["features"].str.startswith("all_")]
    return table.loc[table["ic_mean"].idxmax(), "features"]


def scores(frame: pd.DataFrame, pred: np.ndarray, ppy: float) -> dict:
    frame = frame.assign(pred=pred)
    spread = strategies.backtests(frame, ("ls_ew",))
    return {"r2_oos_pct": 100 * metrics.r2_oos(frame["y_rel"], pred),
            "ic_mean": metrics.rank_ic(frame["y_rel"], pred, frame["date"]).mean(),
            "sharpe_gross": metrics.sharpe(spread["gross"], ppy),
            "sharpe_net": metrics.sharpe(spread["excess"], ppy)}


def explain(freq: str, features: str) -> None:
    names = json.loads((config.PANELS / freq / "names.json").read_text())["names"]
    blocks = feature_sets(names)[features]
    panel = Panel.load(config.PANELS / freq, blocks)
    X, index = panel.X(blocks), panel.index
    model = joblib.load(config.DATA / "models" / freq / f"{features}__ebm.joblib")
    block_of = np.array([b for b in blocks for _ in panel.names[b]])
    feature = np.array(panel.feature_names(blocks))
    terms = model.term_features()
    label = [" & ".join(feature[list(t)]) for t in terms]
    kind = np.array([block_of[t[0]].split("_")[0] if len({block_of[i] for i in t}) == 1
                     else "mixed pair" for t in terms])
    out = config.RESULTS / "explain" / f"{freq}__{features}"
    out.mkdir(parents=True, exist_ok=True)

    fitted = np.flatnonzero(panel.rows("train") | panel.rows("validation"))
    sample = np.sort(np.random.default_rng(config.SEED).choice(
        fitted, min(len(fitted), 100_000), replace=False))
    importance = pd.DataFrame({"term": label, "block": kind,
                               "importance": model.importances(X[sample])})
    importance["share"] = importance["importance"] / importance["importance"].sum()
    importance = importance.sort_values("importance", ascending=False, ignore_index=True)
    importance.to_csv(out / "importance.csv", index=False, float_format="%.6g")
    by_block = importance.groupby("block").agg(terms=("term", "size"), share=("share", "sum"))
    by_block.to_csv(out / "blocks.csv", float_format="%.4f")

    shapes = []
    for name in importance.loc[~importance["term"].str.contains(" & "), "term"].head(SHAPES):
        edges, score = model.shape(label.index(name))
        shapes.append(pd.DataFrame({"term": name, "from": edges[:-1], "to": edges[1:],
                                    "score": score}))
    pd.concat(shapes, ignore_index=True).to_csv(out / "shapes.csv", index=False,
                                                float_format="%.6g")

    test = panel.rows("test")
    frame = index[test]
    contribution = model.contributions(X[test])
    full = model.intercept_ + contribution.sum(axis=1)
    ppy = metrics.PERIODS_PER_YEAR[freq]
    rows = [{"deleted": "nothing", "terms": 0, **scores(frame, full, ppy)}]
    groups = {f"block: {b}": np.flatnonzero(kind == b) for b in np.unique(kind)}
    groups.update({f"term: {t}": np.array([label.index(t)])
                   for t in importance["term"].head(TOP_TERMS)})
    for name, members in groups.items():
        rows.append({"deleted": name, "terms": len(members),
                     **scores(frame, full - contribution[:, members].sum(axis=1), ppy)})
    ablation = pd.DataFrame(rows)
    for column in ("r2_oos_pct", "ic_mean", "sharpe_gross", "sharpe_net"):
        ablation[f"change_in_{column}"] = ablation[column] - ablation[column].iloc[0]
    ablation.to_csv(out / "ablation.csv", index=False, float_format="%.6g")
    print(f"{freq} {features}: {len(terms)} terms; importance by block "
          f"{by_block['share'].round(3).to_dict()}")
    print(ablation[["deleted", "r2_oos_pct", "ic_mean", "sharpe_net"]].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--features", help="feature set (default: chosen by validation IC)")
    a = ap.parse_args()
    for freq in a.freq:
        explain(freq, a.features or default_features(freq))


if __name__ == "__main__":
    main()
