"""Dixon-Coles (1997): independent Poisson goals with a low-score dependence correction and
exponential time decay, fitted by weighted maximum likelihood.

    rates:   lam_h = exp(mu + h + attack[home] - defence[away]),  lam_a = exp(mu + attack[away] - defence[home])
    tau:     (0,0): 1 - lam_h lam_a rho    (0,1): 1 + lam_h rho    (1,0): 1 + lam_a rho    (1,1): 1 - rho    else 1
    weight:  w_i = exp(-xi * days between match i and the newest training match)

Special cases, both asserted in tests/test_models.py:
    xi = 0            every match weighs the same: plain weighted-equally Poisson + tau
    use_tau = False   rho is pinned to 0: exactly the independent Poisson model
The parametrisation (sum-to-zero attack and defence) is shared with poisson.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.poisson import MAX_GOALS, PARAM_BOUND, PoissonModel, sum_zero_map  # noqa: E402

RHO_BOUND = 0.5
TAU_FLOOR = 1e-10


def tau(x, y, lam_h, lam_a, rho):
    """Vectorised Dixon-Coles correction factor for observed scores (x, y)."""
    t = np.ones(np.broadcast(x, y, lam_h, lam_a, rho).shape)
    t = np.where((x == 0) & (y == 0), 1.0 - lam_h * lam_a * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam_h * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + lam_a * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


def time_weights(dates: pd.Series, xi: float) -> np.ndarray:
    age_days = (dates.max() - dates).dt.days.to_numpy(float)
    return np.exp(-xi * age_days)


def scoreline_matrix_dc(lam_h, lam_a, rho, max_goals: int = MAX_GOALS) -> np.ndarray:
    """(n, G, G) Poisson outer product times tau on the four low-score cells, renormalised."""
    from models.poisson import scoreline_matrix
    S = scoreline_matrix(lam_h, lam_a, max_goals)
    lam_h, lam_a = np.asarray(lam_h), np.asarray(lam_a)
    S[:, 0, 0] *= 1.0 - lam_h * lam_a * rho
    S[:, 0, 1] *= 1.0 + lam_h * rho
    S[:, 1, 0] *= 1.0 + lam_a * rho
    S[:, 1, 1] *= 1.0 - rho
    S = np.clip(S, 0.0, None)
    return S / S.sum(axis=(1, 2), keepdims=True)


class DixonColesModel(PoissonModel):
    def __init__(self, xi: float = 0.0018, lookback_years: float | None = 5.0,
                 use_tau: bool = True, max_goals: int = MAX_GOALS):
        super().__init__(lookback_years, max_goals)
        self.xi = xi
        self.use_tau = use_tau
        self.rho_ = 0.0

    # ------------------------------------------------------------ likelihood
    @staticmethod
    def full_loglik(mu, h, rho, A, D, hi, ai, x, y, w=None) -> float:
        """Weighted log-likelihood including tau and the log x! constants."""
        w = np.ones(len(x)) if w is None else w
        lh, la = PoissonModel.rates(mu, h, A, D, hi, ai)
        base = PoissonModel.loglik(mu, h, A, D, hi, ai, x, y, w)
        return base + float((w * np.log(np.maximum(tau(x, y, lh, la, rho), TAU_FLOOR))).sum())

    @staticmethod
    def objective(theta, M, hi, ai, x, y, w):
        """Negative weighted mean log-likelihood (constants dropped) and gradient.
        theta = [mu, h, a_free, d_free, rho]."""
        n = M.shape[0]
        rho = theta[-1]
        mu, h, A, D = PoissonModel.unpack(theta[:-1], M)
        lh, la = PoissonModel.rates(mu, h, A, D, hi, ai)
        scale = w.sum()

        t = np.ones_like(lh)
        dth = np.zeros_like(lh)    # d tau / d log lam_h
        dta = np.zeros_like(lh)    # d tau / d log lam_a
        dtr = np.zeros_like(lh)    # d tau / d rho
        c00 = (x == 0) & (y == 0)
        c01 = (x == 0) & (y == 1)
        c10 = (x == 1) & (y == 0)
        c11 = (x == 1) & (y == 1)
        p = lh * la
        t[c00], dth[c00], dta[c00], dtr[c00] = 1 - p[c00] * rho, -p[c00] * rho, -p[c00] * rho, -p[c00]
        t[c01], dth[c01], dtr[c01] = 1 + lh[c01] * rho, lh[c01] * rho, lh[c01]
        t[c10], dta[c10], dtr[c10] = 1 + la[c10] * rho, la[c10] * rho, la[c10]
        t[c11], dtr[c11] = 1 - rho, -1.0
        t = np.maximum(t, TAU_FLOOR)

        nll = -(w * (x * np.log(lh) - lh + y * np.log(la) - la + np.log(t))).sum() / scale
        rh, ra = w * (x - lh + dth / t), w * (y - la + dta / t)
        gA = np.bincount(hi, rh, n) + np.bincount(ai, ra, n)
        gD = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n)
        g_rho = (w * dtr / t).sum()
        grad = -np.concatenate([[rh.sum() + ra.sum(), rh.sum()], M.T @ gA, M.T @ gD, [g_rho]]) / scale
        return nll, grad

    def _fit_arrays(self, hi, ai, x, y, n, w=None) -> np.ndarray:
        w = np.ones(len(x)) if w is None else np.asarray(w, float)
        M = sum_zero_map(n)
        theta0 = np.zeros(3 + 2 * (n - 1))
        theta0[0] = np.log(max((x.mean() + y.mean()) / 2, 1e-3))
        rho_b = (-RHO_BOUND, RHO_BOUND) if self.use_tau else (0.0, 0.0)
        bounds = [(None, None), (None, None)] + [(-PARAM_BOUND, PARAM_BOUND)] * (2 * (n - 1)) + [rho_b]
        res = minimize(self.objective, theta0, args=(M, hi, ai, x, y, w), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
        return res.x

    # ------------------------------------------------------------ Model interface
    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        self.attack_, self.defence_, self.mu_, self.home_adv_, self.rho_ = {}, {}, 0.0, 0.0, 0.0
        m = self._window(matches)
        teams = sorted(set(m["home"]) | set(m["away"])) if len(m) else []
        if len(m) < 2 or len(teams) < 2:
            return self
        idx = {t: i for i, t in enumerate(teams)}
        hi = m["home"].map(idx).to_numpy()
        ai = m["away"].map(idx).to_numpy()
        x, y = m["fthg"].to_numpy(float), m["ftag"].to_numpy(float)
        theta = self._fit_arrays(hi, ai, x, y, len(teams), time_weights(m["date"], self.xi))
        self.mu_, self.home_adv_, A, D = self.unpack(theta[:-1], sum_zero_map(len(teams)))
        self.rho_ = float(theta[-1])
        self.attack_ = dict(zip(teams, A))
        self.defence_ = dict(zip(teams, D))
        return self

    def predict_scoreline(self, fixtures: pd.DataFrame) -> np.ndarray:
        lam_h, lam_a = self.expected_goals(fixtures)
        return scoreline_matrix_dc(lam_h, lam_a, self.rho_, self.max_goals)
