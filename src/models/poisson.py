"""Independent Poisson goals model, fitted by maximum likelihood.

    home goals ~ Poisson(exp(mu + h + attack[home] - defence[away]))
    away goals ~ Poisson(exp(mu + attack[away] - defence[home]))

Identifiability: adding c to every attack (and subtracting it from mu) or to every defence
(and adding it to mu) leaves all rates unchanged, so sum(attack) = 0 and sum(defence) = 0.
Both are enforced by construction: only n-1 free values per side are optimised and the last
team is minus their sum.

Only matches inside a lookback window (`lookback_years` before the newest training match)
are used, since team strength drifts. A team absent from the window is treated as average
(attack = defence = 0), which is the promoted-club cold start.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base import GenerativeModel  # noqa: E402

MAX_GOALS = 8            # scoreline matrix covers 0..7 goals per side
PARAM_BOUND = 2.5        # keeps a winless/goalless team from running off to -infinity


def sum_zero_map(n: int) -> np.ndarray:
    """n x (n-1) matrix M with A = M @ a_free and sum(A) = 0."""
    return np.vstack([np.eye(n - 1), -np.ones((1, n - 1))])


def scoreline_matrix(lam_h: np.ndarray, lam_a: np.ndarray, max_goals: int = MAX_GOALS) -> np.ndarray:
    """(n, G, G) outer product of two Poisson pmfs, renormalised over the truncated grid."""
    g = np.arange(max_goals)
    ph = poisson.pmf(g[None, :], np.asarray(lam_h)[:, None])
    pa = poisson.pmf(g[None, :], np.asarray(lam_a)[:, None])
    S = ph[:, :, None] * pa[:, None, :]
    return S / S.sum(axis=(1, 2), keepdims=True)


class PoissonModel(GenerativeModel):
    def __init__(self, lookback_years: float | None = 3.0, max_goals: int = MAX_GOALS):
        self.lookback_years = lookback_years
        self.max_goals = max_goals
        self.mu_ = 0.0
        self.home_adv_ = 0.0
        self.attack_: dict[str, float] = {}
        self.defence_: dict[str, float] = {}

    # ------------------------------------------------------------ likelihood
    @staticmethod
    def rates(mu, h, A, D, hi, ai):
        return np.exp(mu + h + A[hi] - D[ai]), np.exp(mu + A[ai] - D[hi])

    @staticmethod
    def loglik(mu, h, A, D, hi, ai, x, y, w=None) -> float:
        """Full weighted Poisson log-likelihood (including the log x! constants)."""
        w = np.ones(len(x)) if w is None else w
        lh, la = PoissonModel.rates(mu, h, A, D, hi, ai)
        return float((w * (x * np.log(lh) - lh - gammaln(x + 1) + y * np.log(la) - la - gammaln(y + 1))).sum())

    @staticmethod
    def unpack(theta: np.ndarray, M: np.ndarray):
        n1 = M.shape[1]
        return theta[0], theta[1], M @ theta[2:2 + n1], M @ theta[2 + n1:]

    @staticmethod
    def objective(theta, M, hi, ai, x, y, w):
        """Negative mean log-likelihood (constants dropped) and its gradient."""
        n = M.shape[0]
        mu, h, A, D = PoissonModel.unpack(theta, M)
        lh, la = PoissonModel.rates(mu, h, A, D, hi, ai)
        scale = w.sum()
        nll = -(w * (x * np.log(lh) - lh + y * np.log(la) - la)).sum() / scale
        rh, ra = w * (x - lh), w * (y - la)
        gA = np.bincount(hi, rh, n) + np.bincount(ai, ra, n)
        gD = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n)
        grad = -np.concatenate([[rh.sum() + ra.sum(), rh.sum()], M.T @ gA, M.T @ gD]) / scale
        return nll, grad

    def _fit_arrays(self, hi, ai, x, y, n, w=None) -> np.ndarray:
        w = np.ones(len(x)) if w is None else np.asarray(w, float)
        M = sum_zero_map(n)
        theta0 = np.zeros(2 + 2 * (n - 1))
        theta0[0] = np.log(max((x.mean() + y.mean()) / 2, 1e-3))
        bounds = [(None, None), (None, None)] + [(-PARAM_BOUND, PARAM_BOUND)] * (2 * (n - 1))
        res = minimize(self.objective, theta0, args=(M, hi, ai, x, y, w), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
        return res.x

    # ------------------------------------------------------------ Model interface
    def _window(self, matches: pd.DataFrame) -> pd.DataFrame:
        if self.lookback_years is None or matches.empty:
            return matches
        cutoff = matches["date"].max() - pd.Timedelta(days=365.25 * self.lookback_years)
        return matches[matches["date"] >= cutoff]

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        self.attack_, self.defence_, self.mu_, self.home_adv_ = {}, {}, 0.0, 0.0
        m = self._window(matches)
        teams = sorted(set(m["home"]) | set(m["away"])) if len(m) else []
        if len(m) < 2 or len(teams) < 2:
            return self
        idx = {t: i for i, t in enumerate(teams)}
        hi = m["home"].map(idx).to_numpy()
        ai = m["away"].map(idx).to_numpy()
        x, y = m["fthg"].to_numpy(float), m["ftag"].to_numpy(float)
        theta = self._fit_arrays(hi, ai, x, y, len(teams))
        self.mu_, self.home_adv_, A, D = self.unpack(theta, sum_zero_map(len(teams)))
        self.attack_ = dict(zip(teams, A))
        self.defence_ = dict(zip(teams, D))
        return self

    def expected_goals(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        a, d = self.attack_, self.defence_
        A_h = np.array([a.get(t, 0.0) for t in fixtures["home"]])
        D_h = np.array([d.get(t, 0.0) for t in fixtures["home"]])
        A_a = np.array([a.get(t, 0.0) for t in fixtures["away"]])
        D_a = np.array([d.get(t, 0.0) for t in fixtures["away"]])
        mu = self.mu_ if self.attack_ else np.log(1.4)
        return np.exp(mu + self.home_adv_ + A_h - D_a), np.exp(mu + A_a - D_h)

    def predict_scoreline(self, fixtures: pd.DataFrame) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(fixtures)
        return scoreline_matrix(lam_h, lam_a, self.max_goals)
