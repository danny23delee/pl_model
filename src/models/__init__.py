"""Model registry: name -> factory(cfg) -> unfitted Model. The backtest only ever
sees this interface."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baserate import BaseRateModel  # noqa: E402
from models.dixon_coles import DixonColesModel  # noqa: E402
from models.elo import EloModel  # noqa: E402
from models.market import MarketModel  # noqa: E402
from models.poisson import PoissonModel  # noqa: E402


def _elo(cfg, k=None):
    c = cfg["models"]["elo"]
    return EloModel(k=c["k_default"] if k is None else k, initial_rating=c["initial_rating"],
                    goal_diff_adjustment=c["goal_diff_adjustment"], home_advantage=c["home_advantage"])


def _dc(cfg, xi=None, lookback=None):
    c = cfg["models"]["dixon_coles"]
    return DixonColesModel(xi=c["xi_default"] if xi is None else xi,
                           lookback_years=c["lookback_years"] if lookback is None else lookback,
                           max_goals=c["max_goals"])


REGISTRY = {
    "baserate": lambda cfg: BaseRateModel(),
    "elo": _elo,
    "poisson": lambda cfg: PoissonModel(cfg["models"]["poisson"]["lookback_default"], cfg["models"]["poisson"]["max_goals"]),
    "dixon_coles": lambda cfg: _dc(cfg),
    "market": lambda cfg: MarketModel(cfg["market"]["devig"]),
}

# Models with one tuned hyperparameter. The backtest sweeps the grid, draws the curve, and
# reports the model with the value chosen out-of-sample season by season (see backtest.py).
TUNING = {
    "elo": {
        "param": "K",
        "grid": lambda cfg: cfg["models"]["elo"]["k_grid"],
        "default": lambda cfg: cfg["models"]["elo"]["k_default"],
        "make": lambda cfg, v: _elo(cfg, v),
        "figure": "elo_k_tuning.png",
    },
    "poisson": {
        "param": "lookback_years",
        "grid": lambda cfg: cfg["models"]["poisson"]["lookback_grid"],
        "default": lambda cfg: cfg["models"]["poisson"]["lookback_default"],
        "make": lambda cfg, v: PoissonModel(v, cfg["models"]["poisson"]["max_goals"]),
        "figure": "poisson_lookback_tuning.png",
    },
    "dixon_coles": {
        "param": "xi",
        "grid": lambda cfg: cfg["models"]["dixon_coles"]["xi_grid"],
        "default": lambda cfg: cfg["models"]["dixon_coles"]["xi_default"],
        "make": lambda cfg, v: _dc(cfg, xi=v),
        "figure": "dc_xi_tuning.png",
    },
}
