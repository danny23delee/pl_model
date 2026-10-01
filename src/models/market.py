"""The bookmaker closing line as a model. fit() is a no-op; predict_proba de-vigs the
benchmark odds (mkt_h/mkt_d/mkt_a, chosen and labelled in clean.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from devig import devig  # noqa: E402
from models.base import Model  # noqa: E402

ODDS_COLS = ["mkt_h", "mkt_d", "mkt_a"]


class MarketModel(Model):
    requires_odds = True

    def __init__(self, method: str = "proportional"):
        self.method = method

    def fit(self, matches: pd.DataFrame) -> "MarketModel":
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        odds = fixtures[ODDS_COLS]
        if odds.isna().any().any():
            raise ValueError("MarketModel needs closing odds on every row; restrict to the closing-odds subset")
        return devig(odds.to_numpy(float), self.method)
