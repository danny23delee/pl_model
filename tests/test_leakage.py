"""Leakage tests: the most important file in the repo.

Every model in REGISTRY is subject to the mutation and fit-date tests below, so new
models (Elo, Poisson, Dixon-Coles) are covered the moment they are registered.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtest as bt  # noqa: E402
import clean  # noqa: E402
from features import build_features  # noqa: E402
from ingest import load_config  # noqa: E402
from models import REGISTRY  # noqa: E402
from models.base import Model  # noqa: E402

CFG = load_config()


def synthetic_league(n_seasons=6, n_teams=8, seed=0) -> pd.DataFrame:
    """Double round-robin, one round per week, random scores, fake closing odds."""
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(n_teams)]
    strength = rng.normal(0, 0.3, n_teams)
    rows = []
    for s in range(n_seasons):
        start = pd.Timestamp(2000 + s, 8, 10)
        rnd = 0
        for leg in range(2):
            arr = list(range(n_teams))
            for r in range(n_teams - 1):
                for i in range(n_teams // 2):
                    a, b = arr[i], arr[n_teams - 1 - i]
                    h, aw = (a, b) if (r + leg) % 2 == 0 else (b, a)
                    date = start + pd.Timedelta(days=7 * rnd + int(rng.integers(0, 2)))
                    hg = rng.poisson(np.exp(0.25 + strength[h] - strength[aw]))
                    ag = rng.poisson(np.exp(strength[aw] - strength[h]))
                    rows.append((2000 + s, date, teams[h], teams[aw], hg, ag))
                arr = [arr[0]] + [arr[-1]] + arr[1:-1]
                rnd += 1
    df = pd.DataFrame(rows, columns=["season", "date", "home", "away", "fthg", "ftag"])
    df["ftr"] = np.where(df.fthg > df.ftag, "H", np.where(df.fthg < df.ftag, "A", "D"))
    df["season_label"] = df["season"].astype(str)
    df["kickoff"] = None
    df = clean.add_matchweek(df)
    df["mkt_h"], df["mkt_d"], df["mkt_a"] = 2.4, 3.4, 3.2
    df["mkt_source"] = pd.array(["PSC"] * len(df), dtype="string")
    return df


@pytest.fixture(scope="module")
def league():
    return synthetic_league()


# ---------------------------------------------------------------- features
@pytest.mark.parametrize("k", [1, 40, 100, 150])
def test_features_identical_when_future_rows_deleted(league, k):
    dates = np.sort(league["date"].unique())
    T = pd.Timestamp(dates[k])
    full = build_features(league, T)
    truncated = build_features(league[league["date"] < T].copy(), T)
    pd.testing.assert_frame_equal(full, truncated, check_exact=True)
    assert len(full) > 0


def test_features_ignore_mutated_future_rows_and_row_order(league):
    T = pd.Timestamp(np.sort(league["date"].unique())[60])
    base = build_features(league, T)
    mutated = league.copy()
    fut = mutated["date"] >= T
    mutated.loc[fut, ["fthg", "ftag"]] = 99
    mutated = mutated.sample(frac=1.0, random_state=1)  # also scramble row order
    pd.testing.assert_frame_equal(base, build_features(mutated, T), check_exact=True)


def test_features_exclude_the_as_of_date_itself(league):
    """A row dated exactly `as_of` must be excluded ('on or after')."""
    T = pd.Timestamp(np.sort(league["date"].unique())[30])
    assert (league["date"] == T).any()
    before = build_features(league[league["date"] < T], T)
    at = build_features(league[league["date"] <= T], T)
    pd.testing.assert_frame_equal(before, at, check_exact=True)


def test_features_hand_computed():
    m = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
        "home": ["a", "b", "a"], "away": ["b", "a", "c"], "fthg": [2, 0, 1], "ftag": [0, 0, 1],
    })
    f = build_features(m, pd.Timestamp("2020-01-20")).set_index("team")
    assert f.loc["a", "games_played"] == 3
    assert f.loc["a", "ppg_last5"] == pytest.approx((3 + 1 + 1) / 3)
    assert f.loc["a", "gf_last5"] == pytest.approx(1.0) and f.loc["a", "ga_last5"] == pytest.approx(1 / 3)
    assert f.loc["a", "days_since_last"] == 5
    assert list(f.index) == ["a", "b", "c"]
    # nothing before the first date -> empty frame with the right columns
    assert build_features(m, pd.Timestamp("2020-01-01")).empty


def test_features_real_data_no_leak():
    m = clean.load_clean()
    for T in ["2005-01-15", "2014-03-01", "2023-11-04"]:
        T = pd.Timestamp(T)
        pd.testing.assert_frame_equal(build_features(m, T), build_features(m[m["date"] < T].copy(), T),
                                      check_exact=True)


# ---------------------------------------------------------------- backtest
class RecordingModel(Model):
    """Records what the backtest hands it."""
    calls = []

    def fit(self, matches):
        self.train_max = matches["date"].max() if len(matches) else pd.NaT
        self.n_train = len(matches)
        return self

    def predict_proba(self, fixtures):
        RecordingModel.calls.append((self.train_max, self.n_train, fixtures["date"].min(), list(fixtures.columns)))
        return np.tile([0.45, 0.27, 0.28], (len(fixtures), 1))


def test_backtest_never_fits_on_rows_at_or_after_prediction_date(league):
    RecordingModel.calls = []
    preds = bt.walk_forward(league, RecordingModel, burn_in_seasons=2)
    assert RecordingModel.calls
    for train_max, n_train, first_fixture, _ in RecordingModel.calls:
        assert n_train > 0 and train_max < first_fixture


def test_backtest_hands_over_no_result_columns(league):
    RecordingModel.calls = []
    bt.walk_forward(league, RecordingModel, burn_in_seasons=2)
    forbidden = {"fthg", "ftag", "ftr", "hthg", "htag", "htr", "h_shots", "a_shots", "mkt_h", "mkt_d", "mkt_a"}
    for *_, cols in RecordingModel.calls:
        assert not forbidden & set(cols), forbidden & set(cols)


def test_a_model_that_peeks_at_the_result_is_caught(league):
    class Cheat(Model):
        def fit(self, matches):
            return self

        def predict_proba(self, fixtures):
            p = np.full((len(fixtures), 3), 0.01)
            p[np.arange(len(p)), fixtures["ftr"].map({"H": 0, "D": 1, "A": 2}).to_numpy()] = 0.98
            return p

    with pytest.raises(KeyError):
        bt.walk_forward(league, Cheat, burn_in_seasons=2)


def test_burn_in_seasons_are_trained_on_but_not_scored(league):
    preds = bt.walk_forward(league, RecordingModel, burn_in_seasons=3)
    assert preds["season"].min() == league["season"].min() + 3
    assert len(preds) == (league["season"] >= league["season"].min() + 3).sum()


def test_each_match_predicted_exactly_once(league):
    preds = bt.walk_forward(league, lambda: REGISTRY["baserate"](CFG), burn_in_seasons=2)
    assert not preds.duplicated(["date", "home", "away"]).any()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_mutating_results_from_T_onwards_leaves_earlier_predictions_unchanged(league, name):
    """Predictions for every block starting before T must be bit-identical when all
    results dated >= T are scrambled. (Covers Elo state and team-strength parameters.)"""
    dates = np.sort(league["date"].unique())
    T = pd.Timestamp(dates[len(dates) * 3 // 4])
    factory = lambda: REGISTRY[name](CFG)  # noqa: E731
    base = bt.walk_forward(league, factory, burn_in_seasons=2)

    mutated = league.copy()
    fut = mutated["date"] >= T
    rng = np.random.default_rng(7)
    mutated.loc[fut, "fthg"] = rng.integers(0, 6, fut.sum())
    mutated.loc[fut, "ftag"] = rng.integers(0, 6, fut.sum())
    mutated["ftr"] = np.where(mutated.fthg > mutated.ftag, "H", np.where(mutated.fthg < mutated.ftag, "A", "D"))
    alt = bt.walk_forward(mutated, factory, burn_in_seasons=2)

    # blocks are keyed by their first date; the mutated frame keeps identical fixtures
    block_start = league.groupby(["season", "matchweek"])["date"].transform("min")
    early = league.loc[block_start < T, ["date", "home", "away"]]
    cols = ["date", "home", "away"] + bt.PRED_COLS
    a = base.merge(early, on=["date", "home", "away"])[cols].sort_values(cols[:3]).reset_index(drop=True)
    b = alt.merge(early, on=["date", "home", "away"])[cols].sort_values(cols[:3]).reset_index(drop=True)
    assert len(a) > 20
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    # and the mutation really did change something later (the test has teeth)
    if name != "market":  # the market ignores results entirely
        after = base["date"] >= T + pd.Timedelta(days=60)
        assert after.any()
        assert not np.array_equal(base.loc[after, bt.PRED_COLS].to_numpy(),
                                  alt.loc[alt["date"] >= T + pd.Timedelta(days=60), bt.PRED_COLS].to_numpy())


# ---------------------------------------------------------------- real data
def test_baserate_scores_like_a_baseline_and_market_beats_it():
    """A baseline that scored suspiciously well would indicate a leak. The base rate
    should sit clearly behind the closing line on the same matches."""
    m = clean.load_clean()
    br = bt.walk_forward(m, lambda: REGISTRY["baserate"](CFG), CFG["backtest"]["burn_in_seasons"])
    mk = bt.walk_forward(m, lambda: REGISTRY["market"](CFG), CFG["backtest"]["burn_in_seasons"])
    assert br["season"].min() == m["season"].min() + CFG["backtest"]["burn_in_seasons"]
    same = br.merge(mk[["date", "home", "away"]], on=["date", "home", "away"])
    assert len(same) == len(mk)
    r_br = bt.score_subsets(same, "baserate", CFG)
    r_mk = bt.score_subsets(mk, "market", CFG)
    rps_br = next(r["rps"] for r in r_br if r["subset"] == "all_scored")
    rps_mk = next(r["rps"] for r in r_mk if r["subset"] == "all_scored")
    assert 0.22 < rps_br < 0.245
    assert rps_mk < rps_br - 0.02
