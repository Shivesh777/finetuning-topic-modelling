"""What an experiment is: a rebalancing frequency, a set of features and a model.

Forecasts are stored per experiment under data/predictions/<frequency>/ as
<features>__<model>.<split>.npy, in the row order of the panel; the helpers here
name the feature sets and read the forecasts back next to their rows.
"""
from __future__ import annotations

from pathlib import Path
import re

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


TONE_BLOCKS = ("tone_fb", "tone_lm", "emb_fb", "npx")
# Feature sets of the directional news blocks (ftm/features/tone.py); px_all adds the
# narrative and firm-news blocks of NoLBERT (narrative with the validated shock window
# where there is one).
NEWS_SETS = {
    "tone_fb": ["tone_fb"],
    "tone_lm": ["tone_lm"],
    "emb_fb": ["emb_fb"],
    "news_fb": ["tone_fb", "tone_lm", "emb_fb"],
    "px_tone": ["price", "tone_fb", "tone_lm", "npx"],
    "px_news": ["price", "tone_fb", "tone_lm", "emb_fb", "npx"],
}
# Forecasts that are a single feature, no model fitted (scripts/06c_signals.py):
# model name -> (block, feature, sign). The signs are those the literature predicts:
# good news and more attention before higher returns, moves without news reversing.
SIGNALS = {
    "sig_fb1": ("tone_fb", "fb_tone_1", 1), "sig_fb5": ("tone_fb", "fb_tone_5", 1),
    "sig_fb21": ("tone_fb", "fb_tone_21", 1), "sig_fbsurp": ("tone_fb", "fb_surprise_21", 1),
    "sig_lm1": ("tone_lm", "lm_tone_1", 1), "sig_lm5": ("tone_lm", "lm_tone_5", 1),
    "sig_lm21": ("tone_lm", "lm_tone_21", 1), "sig_lmsurp": ("tone_lm", "lm_surprise_21", 1),
    "sig_abn": ("tone_fb", "news_abnormal_21", 1),
    "sig_quietrev": ("npx", "rev5_quiet", -1), "sig_newsdrift": ("npx", "rev5_news", 1),
}
# The combination of the price ensemble with the best tone signal (scripts/06c_signals.py).
COMBO = "combo"
# The average of the POOL_SIZE base-model experiments with the highest validation rank IC
# (scripts/06c_signals.py), stored under the feature set "px_pool".
POOL, POOL_SIZE = "best10", 10


def feature_sets(blocks) -> dict[str, list[str]]:
    """Named combinations of a panel's feature blocks.

    price          the no-text benchmark
    narr_<m>       narrative exposures of topic model m
    firm_<m>       the firm's own news under topic model m
    text_<m>       both text blocks
    all_<m>        price and both text blocks
    narr_<m>_s<w>  narrative exposures with attention shocks over w days; text_<m>_s<w> and
                   all_<m>_s<w> combine them like text_<m> and all_<m>
    NEWS_SETS      the tone blocks alone, together, and with the price features; px_all
                   all of that with the NoLBERT text blocks
    """
    blocks = list(blocks)
    sets = {"price": ["price"]} if "price" in blocks else {}
    for block in blocks:
        if not block.startswith("narr_"):
            continue
        model = block[len("narr_"):]
        sets[f"narr_{model}"] = [block]
        tuned = re.fullmatch(r"(.+)_s\d+", model)
        topic = tuned.group(1) if tuned and not model.endswith("_placebo") else model
        if f"firm_{topic}" in blocks:
            if topic == model:
                sets[f"firm_{model}"] = [f"firm_{model}"]
            sets[f"text_{model}"] = [block, f"firm_{topic}"]
            if "price" in blocks:
                sets[f"all_{model}"] = ["price", block, f"firm_{topic}"]
    if all(b in blocks for b in TONE_BLOCKS):
        sets.update({name: list(b) for name, b in NEWS_SETS.items()
                     if all(x in blocks for x in b)})
        narr = next((b for b in blocks if b.startswith("narr_nolbert_s")), "narr_nolbert")
        if "price" in blocks and narr in blocks and "firm_nolbert" in blocks:
            sets["px_all"] = NEWS_SETS["px_news"] + [narr, "firm_nolbert"]
    return sets


def kind(features: str) -> str:
    """What a feature set is made of: price, narr, firm, text, all, placebo; narr_s, text_s
    and all_s for those made with another shock window; tone, emb, news (directional news
    blocks) and px (price with them)."""
    if features.endswith("_placebo"):
        return "placebo"
    prefix = features.partition("_")[0]
    if prefix in ("narr", "text", "all") and re.search(r"_s\d+$", features):
        return f"{prefix}_s"
    return prefix


def topic_model(features: str) -> str | None:
    """The topic model behind a feature set; None for feature sets without one (price,
    placebos, tuned shock windows, directional news)."""
    return features.partition("_")[2] if kind(features) in ("narr", "firm", "text", "all") else None


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
