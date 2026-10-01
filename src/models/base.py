"""The model contract. Every model, including the market baseline, implements it,
so the backtest loop never branches on model type."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Model:
    # True for models that read the closing-odds columns (mkt_h/mkt_d/mkt_a). The backtest
    # then passes those columns and predicts only matches that have them.
    requires_odds = False

    def fit(self, matches: pd.DataFrame) -> "Model":
        raise NotImplementedError

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """(n, 3) array of P(home), P(draw), P(away). Rows sum to 1."""
        raise NotImplementedError


class GenerativeModel(Model):
    """Models of the scoreline. predict_proba MUST be to_1x2(predict_scoreline(...))."""

    def predict_scoreline(self, fixtures: pd.DataFrame) -> np.ndarray:
        """(n, 8, 8) joint distribution over (home goals, away goals)."""
        raise NotImplementedError

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        # Defined once, here, so 1X2 can never disagree with the scoreline matrix.
        from models.reducers import to_1x2
        return to_1x2(self.predict_scoreline(fixtures))
