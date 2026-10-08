"""What an experiment is: a rebalancing frequency, a set of features and a model.

Forecasts are stored per experiment under data/predictions/<frequency>/ as
<features>__<model>.<split>.npy, in the row order of the panel; the helpers here
name the feature sets and read the forecasts back next to their rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PREDICTIONS = config.DATA / "predictions"
# The eight forecasting models of the design, cheapest first. The elastic net ("enet") is
# implemented but not run: with returns as the target the penalty is so small next to the
# variance of the features that its ridge part does nothing, and it reproduces the lasso
# forecasts to six decimals.
BASE_MODELS = ("ols", "ridge", "lasso", "ipca", "sipca", "xgb", "rf", "ebm")
# The same models fitted on the cross-sectional rank of the return (ftm.models.tuning).
RANK_MODELS = ("ols_rk", "ridge_rk", "lasso_rk", "ipca_rk", "sipca_rk", "xgb_rk")
# What scripts/06_run_models.py fits by default.
MODEL_ORDER = BASE_MODELS + RANK_MODELS
# The equal-weighted combination of the base models (scripts/06b_ensemble.py): nothing is
# fitted, so it is not in MODEL_ORDER.
ENSEMBLE = "ens"
EVALUATED = ("validation", "test")


def feature_sets(blocks) -> dict[str, list[str]]:
    """Named combinations of a panel's feature blocks.

    price          the no-text benchmark
    narr_<m>       narrative exposures of topic model m
    firm_<m>       the firm's own news under topic model m
    text_<m>       both text blocks
    all_<m>        price and both text blocks
    """
    blocks = list(blocks)
    sets = {"price": ["price"]} if "price" in blocks else {}
    for block in blocks:
        if not block.startswith("narr_"):
            continue
        model = block[len("narr_"):]
        sets[f"narr_{model}"] = [block]
        if f"firm_{model}" in blocks:
            sets[f"firm_{model}"] = [f"firm_{model}"]
            sets[f"text_{model}"] = [block, f"firm_{model}"]
            if "price" in blocks:
                sets[f"all_{model}"] = ["price", block, f"firm_{model}"]
    return sets


def kind(features: str) -> str:
    """What a feature set is made of: price, narr, firm, text, all, or placebo."""
    return "placebo" if features.endswith("_placebo") else features.partition("_")[0]


def topic_model(features: str) -> str | None:
    """The topic model behind a feature set; None for the price features and for placebos."""
    return None if kind(features) in ("price", "placebo") else features.partition("_")[2]


def prediction_path(freq: str, features: str, model: str, split: str) -> Path:
    return PREDICTIONS / freq / f"{features}__{model}.{split}.npy"


def finished(freq: str) -> list[tuple[str, str]]:
    """(features, model) of every experiment of a frequency that has test forecasts."""
    stems = sorted(p.name[:-len(".test.npy")] for p in (PREDICTIONS / freq).glob("*.test.npy"))
    return [tuple(stem.split("__")) for stem in stems]


def load_predictions(index: pd.DataFrame, freq: str, features: str, model: str,
                     split: str) -> pd.DataFrame:
    """The rows of the panel index that belong to `split`, with the forecasts as `pred`."""
    part = index[index["split"] == split].copy()
    part["pred"] = np.load(prediction_path(freq, features, model, split)).astype(np.float64)
    return part
