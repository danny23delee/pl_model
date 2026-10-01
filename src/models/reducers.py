"""Pure functions of the scoreline matrix S, shape (n, G, G), S[k, i, j] = P(home i, away j).

Every goals-derived market is a reducer over the same matrix, so 1X2, totals, BTTS and
handicaps are consistent with each other by construction. S must sum to 1 per fixture
(the truncated tail is renormalised away by whoever builds S).
"""
from __future__ import annotations

import numpy as np


def _margin_total(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = S.shape[1]
    i = np.arange(g)
    return i[:, None] - i[None, :], i[:, None] + i[None, :]      # home margin, total goals


def to_1x2(S: np.ndarray) -> np.ndarray:
    """(n, 3): P(home win), P(draw), P(away win)."""
    margin, _ = _margin_total(S)
    return np.stack([(S * (margin > 0)).sum((1, 2)), (S * (margin == 0)).sum((1, 2)),
                     (S * (margin < 0)).sum((1, 2))], axis=1)


def to_over_under(S: np.ndarray, line: float) -> np.ndarray:
    """(n, 2): P(total goals > line), P(total goals < line). Half-goal lines only
    (whole and quarter lines have pushes and are not a two-way market)."""
    if (2 * line) % 2 != 1:
        raise ValueError(f"over/under line must be a half-integer such as 2.5, got {line}")
    _, total = _margin_total(S)
    over = (S * (total > line)).sum((1, 2))
    return np.stack([over, 1.0 - over], axis=1)


def to_btts(S: np.ndarray) -> np.ndarray:
    """(n, 2): P(both teams score), P(not)."""
    yes = S[:, 1:, 1:].sum((1, 2))
    return np.stack([yes, 1.0 - yes], axis=1)


def _ah_whole_or_half(S: np.ndarray, line: float) -> np.ndarray:
    margin, _ = _margin_total(S)
    adj = margin + line          # home margin after the handicap is applied to the home side
    return np.stack([(S * (adj > 0)).sum((1, 2)), (S * (adj == 0)).sum((1, 2)),
                     (S * (adj < 0)).sum((1, 2))], axis=1)


def to_asian_handicap(S: np.ndarray, line: float) -> np.ndarray:
    """(n, 3): stake fractions [home wins, push (stake returned), away wins] for a handicap
    `line` applied to the home side (-0.5: home must win; +0.25: home gets a quarter goal).
    A quarter line splits the stake across the two neighbouring lines, so its 'push'
    column can be 0.5 of the stake on some outcomes."""
    if (4 * line) % 1 != 0:
        raise ValueError(f"handicap line must be a multiple of 0.25, got {line}")
    if (2 * line) % 1 == 0:
        return _ah_whole_or_half(S, line)
    return 0.5 * (_ah_whole_or_half(S, line - 0.25) + _ah_whole_or_half(S, line + 0.25))
