"""Elo ratings with a goal-difference-adjusted K, mapped to H/D/A by an ordered logit.

Ratings are updated match by match in date order (World Football Elo style):
    E  = 1 / (1 + 10 ** (-(R_home - R_away + home_advantage) / 400))
    R += K * G(goal difference) * (S - E),   S = 1 / 0.5 / 0 for win / draw / loss
    G  = 1 (margin <= 1), 1.5 (margin 2), (11 + margin) / 8 (margin >= 3)

Elo alone gives an expected score, not a draw probability. fit() therefore also replays the
training matches and fits a 3-parameter ordered logit of the result on the pre-match rating
gap (x = (R_home - R_away) / 400): P(A) = s(c1 - b x), P(H) = 1 - s(c2 - b x), c1 < c2.
Everything is estimated from matches passed to fit(), so nothing can see the future.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base import Model  # noqa: E402

_CODE = {"H": 0, "D": 1, "A": 2}


def goal_diff_multiplier(margin: int) -> float:
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


class EloModel(Model):
    def __init__(self, k: float = 20.0, initial_rating: float = 1500.0,
                 goal_diff_adjustment: bool = True, home_advantage: float = 65.0):
        self.k = k
        self.initial = initial_rating
        self.gd_adj = goal_diff_adjustment
        self.home_adv = home_advantage
        self.ratings_: dict[str, float] = {}
        self.params_ = np.array([-0.6, 0.4, 1.0])  # c1, c2, b (overwritten by fit)

    # ------------------------------------------------------------ ratings
    def _replay(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Update ratings through `matches` in date order; return the pre-match rating
        gap and outcome code for every match."""
        m = matches.sort_values(["date", "home", "away"], kind="stable")
        R: dict[str, float] = {}
        init, k, ha, adj = self.initial, self.k, self.home_adv, self.gd_adj
        gaps, ys = [], []
        for h, a, hg, ag in zip(m["home"], m["away"], m["fthg"], m["ftag"]):
            rh, ra = R.get(h, init), R.get(a, init)
            gaps.append(rh - ra)
            e = 1.0 / (1.0 + 10.0 ** (-(rh - ra + ha) / 400.0))
            s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            ys.append(0 if hg > ag else (1 if hg == ag else 2))
            delta = k * (goal_diff_multiplier(int(abs(hg - ag))) if adj else 1.0) * (s - e)
            R[h] = rh + delta
            R[a] = ra - delta
        self.ratings_ = R
        return np.array(gaps), np.array(ys, dtype=int)

    # ------------------------------------------------------------ result mapping
    @staticmethod
    def _probs(params: np.ndarray, gap: np.ndarray) -> np.ndarray:
        c1, w, b = params
        c2 = c1 + np.exp(w)
        x = gap / 400.0
        p_away = expit(c1 - b * x)
        p_home = 1.0 - expit(c2 - b * x)
        p = np.column_stack([p_home, 1.0 - p_home - p_away, p_away])
        return np.clip(p, 1e-9, None) / np.clip(p, 1e-9, None).sum(axis=1, keepdims=True)

    def fit(self, matches: pd.DataFrame) -> "EloModel":
        self.ratings_ = {}
        if matches.empty:
            return self
        gap, y = self._replay(matches)

        def nll(params):
            p = self._probs(params, gap)
            return -np.log(p[np.arange(len(y)), y]).mean()

        res = minimize(nll, self.params_, method="L-BFGS-B")
        self.params_ = res.x
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        R = self.ratings_
        gap = np.array([R.get(h, self.initial) - R.get(a, self.initial)
                        for h, a in zip(fixtures["home"], fixtures["away"])])
        return self._probs(self.params_, gap)
