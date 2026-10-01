"""Feature construction under the leakage contract:
no feature may be computed from any row dated on or after `as_of`."""
from __future__ import annotations

import numpy as np
import pandas as pd

FORM_WINDOW = 5


def build_features(matches: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per team that has played before `as_of`, describing that team's state
    as of `as_of`: games played, rolling form over its last FORM_WINDOW matches, and
    rest since its last match. The input is filtered to date < as_of before anything
    else happens, and rows are sorted internally so input order is irrelevant.
    """
    as_of = pd.Timestamp(as_of)
    past = matches.loc[matches["date"] < as_of, ["date", "home", "away", "fthg", "ftag"]]
    cols = ["team", "games_played", "ppg_last5", "gf_last5", "ga_last5", "days_since_last"]
    if past.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64" if c != "team" else "object") for c in cols})

    home = pd.DataFrame({"date": past["date"], "team": past["home"], "gf": past["fthg"], "ga": past["ftag"]})
    away = pd.DataFrame({"date": past["date"], "team": past["away"], "gf": past["ftag"], "ga": past["fthg"]})
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.where(long["gf"] > long["ga"], 3.0, np.where(long["gf"] == long["ga"], 1.0, 0.0))
    long = long.sort_values(["team", "date", "gf", "ga"], kind="mergesort")

    g = long.groupby("team", sort=True)
    last = g.tail(FORM_WINDOW).groupby("team", sort=True)
    out = pd.DataFrame({
        "games_played": g.size().astype("int64"),
        "ppg_last5": last["pts"].mean(),
        "gf_last5": last["gf"].mean(),
        "ga_last5": last["ga"].mean(),
        "days_since_last": (as_of - g["date"].max()).dt.days.astype("int64"),
    })
    return out.rename_axis("team").reset_index()[cols]
