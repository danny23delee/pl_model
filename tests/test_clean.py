"""M1 data-cleaning tests.

Unit tests run on tests/fixtures/mini_E0.csv. Integration tests run on the real
cached files and fail (not skip) if `make data` has not been run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import clean  # noqa: E402
from ingest import current_season_start, load_config, raw_path  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "mini_E0.csv"
CFG = load_config()


# ---------------------------------------------------------------- unit
def test_fixture_parsing():
    tm = clean.load_team_map()
    df = clean.clean_season(FIXTURE, 2000, tm)
    # blank row dropped, unplayed fixture dropped, wider-than-header row kept
    assert len(df) == 3
    assert list(df["home"]) == ["man_united", "chelsea", "liverpool"]
    # 2-digit and 4-digit years both parse
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2000-08-19", "2000-08-20", "2000-08-27"]
    # BOM did not corrupt the first column name; non-ASCII referee byte stripped
    assert "Div" not in df.columns and df["date"].notna().all()
    assert df["referee"].iloc[0] == "A Wiley"
    assert df["referee"].iloc[1].endswith("Rennie") and df["referee"].iloc[1].isascii()
    # odds <= 1.0 are invalid -> NaN
    assert np.isnan(df["psch"].iloc[2])
    assert df["psch"].iloc[0] == 1.5


def test_unmapped_team_is_hard_failure():
    with pytest.raises(KeyError, match="unmapped"):
        clean.map_teams(pd.Series(["Arsenal", "Nowhere FC"]), clean.load_team_map())


def test_duplicate_alias_is_hard_failure(tmp_path):
    p = tmp_path / "t.json"
    p.write_text('{"teams": {"a": {"name": "A", "aliases": ["X"]}, "b": {"name": "B", "aliases": ["X"]}}}')
    with pytest.raises(ValueError):
        clean.load_team_map(p)


def test_matchweek_counts_games_played():
    df = pd.DataFrame({
        "season": 2000,
        "date": pd.to_datetime(["2000-08-19", "2000-08-19", "2000-08-26", "2000-08-26", "2000-09-05"]),
        "home": ["a", "c", "a", "b", "b"], "away": ["b", "d", "c", "d", "c"],
    })
    out = clean.add_matchweek(df)
    assert list(out["matchweek"]) == [1, 1, 2, 2, 3]


def test_market_selection_never_mixes():
    cols = {c: np.nan for c in ["psch", "pscd", "psca", "avgch", "avgcd", "avgca"]}
    df = pd.DataFrame([dict(cols) for _ in range(3)])
    df.loc[0, ["psch", "pscd", "psca"]] = [2.0, 3.0, 4.0]
    df.loc[0, ["avgch", "avgcd", "avgca"]] = [2.1, 3.1, 4.1]
    df.loc[1, ["avgch", "avgcd", "avgca"]] = [2.2, 3.2, 4.2]
    df.loc[2, ["psch", "pscd"]] = [2.0, 3.0]  # incomplete primary, no fallback -> no market
    out = clean.add_market(df, CFG)
    assert out["mkt_source"].iloc[:2].tolist() == ["PSC", "AvgC"]
    assert pd.isna(out["mkt_source"].iloc[2])
    assert out.loc[0, ["mkt_h", "mkt_d", "mkt_a"]].tolist() == [2.0, 3.0, 4.0]
    assert out.loc[1, ["mkt_h", "mkt_d", "mkt_a"]].tolist() == [2.2, 3.2, 4.2]
    assert out.loc[2, ["mkt_h", "mkt_d", "mkt_a"]].isna().all()


# ---------------------------------------------------------------- integration
@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    missing = [y for y in range(CFG["data"]["first_season"], current_season_start() + 1)
               if not raw_path(CFG, y).exists()]
    assert not missing, f"raw files missing for seasons {missing}: run `make data`"
    return clean.build_clean(CFG)


def test_every_team_name_maps():
    """Read raw team columns directly: any unmapped name is a hard failure."""
    tm = clean.load_team_map()
    for y in range(CFG["data"]["first_season"], current_season_start() + 1):
        raw = clean.read_raw_season(raw_path(CFG, y))
        names = pd.concat([raw["HomeTeam"], raw["AwayTeam"]]).dropna().str.strip()
        assert set(names) <= set(tm), f"{y}: {sorted(set(names) - set(tm))}"


def test_completed_seasons_have_380(matches):
    counts = matches.groupby("season").size()
    cur = current_season_start()
    for s, n in counts.items():
        if s < cur:
            assert n == 380, f"season {s}: {n} matches"
        else:
            assert 0 < n <= 380


def test_current_season_equals_played_to_date(matches):
    cur = current_season_start()
    raw = clean.read_raw_season(raw_path(CFG, cur))
    played = raw[pd.to_numeric(raw["FTHG"], errors="coerce").notna()
                 & pd.to_numeric(raw["FTAG"], errors="coerce").notna()]
    assert (matches["season"] == cur).sum() == len(played)
    assert matches.loc[matches["season"] == cur, "date"].max() <= pd.Timestamp.today().normalize()


def test_no_duplicate_keys(matches):
    assert not matches.duplicated(["date", "home", "away"]).any()
    assert not matches.duplicated(["season", "home", "away"]).any()  # each fixture once per season


def test_no_nulls_in_required(matches):
    assert matches[clean.REQUIRED].isna().sum().sum() == 0
    assert (matches["home"] != matches["away"]).all()


def test_result_agrees_with_goals(matches):
    res = np.where(matches.fthg > matches.ftag, "H", np.where(matches.fthg < matches.ftag, "A", "D"))
    assert (res == matches["ftr"]).all()


def test_dates_fall_inside_season(matches):
    start = pd.to_datetime(matches["season"].astype(str) + "-07-01")
    end = pd.to_datetime((matches["season"] + 1).astype(str) + "-07-31")
    assert ((matches["date"] >= start) & (matches["date"] <= end)).all()


def test_each_team_plays_38_in_completed_seasons(matches):
    m = matches[matches["season"] < current_season_start()]
    long = pd.concat([m[["season", "home"]].rename(columns={"home": "t"}),
                      m[["season", "away"]].rename(columns={"away": "t"})])
    assert (long.groupby(["season", "t"]).size() == 38).all()
    assert (long.groupby("season")["t"].nunique() == 20).all()


def test_market_source_recorded_and_odds_valid(matches):
    has = matches["mkt_source"].notna()
    assert set(matches.loc[has, "mkt_source"]) <= {CFG["market"]["primary"], CFG["market"]["fallback"]}
    assert matches.loc[has, ["mkt_h", "mkt_d", "mkt_a"]].gt(1.0).all().all()
    assert matches.loc[~has, ["mkt_h", "mkt_d", "mkt_a"]].isna().all().all()


def test_first_closing_odds_season_matches_config(matches):
    first = matches.loc[matches["psch"].notna(), "season"].min()
    assert first == CFG["backtest"]["min_closing_odds_season"]
    assert matches.loc[matches["season"] < first, "mkt_source"].isna().all()
