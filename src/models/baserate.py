"""Base-rate model: predicts the historical H/D/A frequencies for every fixture.
The floor every other model has to clear."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.base import Model  # noqa: E402


class BaseRateModel(Model):
    def __init__(self):
        self.p_ = np.full(3, 1 / 3)

    def fit(self, matches: pd.DataFrame) -> "BaseRateModel":
        if len(matches):
            counts = matches["ftr"].value_counts()
            self.p_ = np.array([counts.get(k, 0) for k in "HDA"], float)
            self.p_ /= self.p_.sum()
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p_, (len(fixtures), 1))
