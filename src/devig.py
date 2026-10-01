"""Remove the bookmaker margin from decimal 1X2 odds.

Both methods take an (n, 3) array of decimal odds (H, D, A) and return an (n, 3)
array of probabilities that sum to 1 and lie strictly inside (0, 1).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

METHODS = ("proportional", "shin")


def _check(odds) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    if o.ndim != 2 or o.shape[1] != 3:
        raise ValueError(f"odds must have shape (n, 3), got {o.shape}")
    if not np.isfinite(o).all() or (o <= 1.0).any():
        raise ValueError("decimal odds must be finite and > 1.0")
    return o


def overround(odds) -> np.ndarray:
    """Sum of implied probabilities minus 1 (the margin), per row."""
    return (1.0 / _check(odds)).sum(axis=1) - 1.0


def proportional(odds) -> np.ndarray:
    """Divide each implied probability by the booksum."""
    q = 1.0 / _check(odds)
    return q / q.sum(axis=1, keepdims=True)


def _shin_row(q: np.ndarray) -> np.ndarray:
    """Shin (1993): p_i = (sqrt(z^2 + 4(1-z) q_i^2 / S) - z) / (2(1-z)), with the
    insider-trading share z in [0, 1) chosen so that sum(p) = 1."""
    s = q.sum()
    if s <= 1.0:  # no margin (or an arbitrage book): nothing to remove
        return q / s

    def p_of(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / s) - z) / (2.0 * (1.0 - z))

    # sum(p) is sqrt(S) > 1 at z=0 and sum(q^2)/S < 1 as z -> 1, so a root exists.
    z = brentq(lambda z: p_of(z).sum() - 1.0, 0.0, 1.0 - 1e-12, xtol=1e-15, rtol=1e-14)
    p = p_of(z)
    return p / p.sum()


def shin(odds) -> np.ndarray:
    q = 1.0 / _check(odds)
    return np.vstack([_shin_row(r) for r in q])


def devig(odds, method: str = "proportional") -> np.ndarray:
    if method == "proportional":
        return proportional(odds)
    if method == "shin":
        return shin(odds)
    raise ValueError(f"unknown de-vig method {method!r}; choose from {METHODS}")


def proportional_two_way(over, under) -> np.ndarray:
    """De-vig a two-outcome market (e.g. over/under 2.5). Returns P(first outcome), shape (n,)."""
    o, u = np.asarray(over, float), np.asarray(under, float)
    if not (np.isfinite(o).all() and np.isfinite(u).all()) or (o <= 1).any() or (u <= 1).any():
        raise ValueError("decimal odds must be finite and > 1.0")
    qo, qu = 1.0 / o, 1.0 / u
    return qo / (qo + qu)
