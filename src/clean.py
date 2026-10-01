"""Normalise the raw football-data.co.uk season files into one match table."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import ROOT, current_season_start, load_config, raw_path  # noqa: E402

# raw column -> clean column
CORE = {
    "Date": "date", "Time": "kickoff", "HomeTeam": "home_raw", "AwayTeam": "away_raw",
    "FTHG": "fthg", "FTAG": "ftag", "FTR": "ftr", "HTHG": "hthg", "HTAG": "htag", "HTR": "htr",
    "Referee": "referee",
    "HS": "h_shots", "AS": "a_shots", "HST": "h_sot", "AST": "a_sot",
    "HF": "h_fouls", "AF": "a_fouls", "HC": "h_corners", "AC": "a_corners",
    "HY": "h_yellow", "AY": "a_yellow", "HR": "h_red", "AR": "a_red",
}
INT_COLS = ["fthg", "ftag", "hthg", "htag", "h_shots", "a_shots", "h_sot", "a_sot", "h_fouls",
            "a_fouls", "h_corners", "a_corners", "h_yellow", "a_yellow", "h_red", "a_red"]

# 1X2 books kept: clean prefix -> raw prefix. Opening = H/D/A, closing = CH/CD/CA.
BOOKS = {"ps": "PS", "avg": "Avg", "max": "Max", "b365": "B365", "bfe": "BFE"}
# Closing over/under 2.5 (needed in M5): clean prefix -> raw prefix.
OU_BOOKS = {"psc": "PC", "avgc": "AvgC", "maxc": "MaxC", "b365c": "B365C"}


def odds_column_map() -> dict[str, str]:
    m = {}
    for cp, rp in BOOKS.items():
        for o, r in (("h", "H"), ("d", "D"), ("a", "A")):
            m[f"{rp}{r}"] = f"{cp}{o}"
            m[f"{rp}C{r}"] = f"{cp}c{o}"
    for cp, rp in OU_BOOKS.items():
        m[f"{rp}>2.5"] = f"{cp}_o25"
        m[f"{rp}<2.5"] = f"{cp}_u25"
    return m


ODDS_MAP = odds_column_map()
ODDS_COLS = sorted(ODDS_MAP.values())
REQUIRED = ["season", "date", "home", "away", "fthg", "ftag", "ftr"]


# ---------------------------------------------------------------- teams
def load_team_map(path: Path | None = None) -> dict[str, str]:
    """raw name -> canonical id. Exact match only. Duplicate aliases are an error."""
    path = path or ROOT / load_config()["data"]["teams_path"]
    with open(path, encoding="utf-8") as f:
        teams = json.load(f)["teams"]
    out: dict[str, str] = {}
    for tid, rec in teams.items():
        for alias in rec["aliases"]:
            if alias in out:
                raise ValueError(f"alias {alias!r} maps to both {out[alias]!r} and {tid!r}")
            out[alias] = tid
    return out


def map_teams(raw: pd.Series, team_map: dict[str, str]) -> pd.Series:
    unmapped = sorted(set(raw.unique()) - set(team_map))
    if unmapped:
        raise KeyError(f"unmapped team names (add to config/teams.json): {unmapped}")
    return raw.map(team_map)


# ---------------------------------------------------------------- parsing
def read_raw_season(path: Path) -> pd.DataFrame:
    """Parse one raw CSV into an all-string frame keyed by the header.

    football-data files are messy: a UTF-8 BOM on some, cp1252 bytes in referee
    names, blank trailing rows, and (2003/04, 2004/05) rows that are wider than
    the header. Extra trailing fields beyond the header are dropped; short rows
    are padded. The named columns are unaffected because they always come first.
    """
    text = path.read_bytes().decode("cp1252", errors="replace")
    text = text.lstrip("\ufeff").removeprefix("\u00ef\u00bb\u00bf")  # BOM, or its cp1252 rendering
    rows = list(csv.reader(text.splitlines()))
    header = [h.strip() for h in rows[0]]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in rows[1:] if any(c.strip() for c in r)]
    df = pd.DataFrame(body, columns=header)
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    return df.replace({"": np.nan})


def parse_dates(date: pd.Series) -> pd.Series:
    """dd/mm/yy and dd/mm/yyyy (both occur, sometimes within one file)."""
    d = date.astype(str).str.strip()
    four = d.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}")
    out = pd.to_datetime(d.where(four), format="%d/%m/%Y", errors="coerce")
    return out.fillna(pd.to_datetime(d.where(~four), format="%d/%m/%y", errors="coerce"))


def clean_season(path: Path, start_year: int, team_map: dict[str, str]) -> pd.DataFrame:
    raw = read_raw_season(path)
    df = pd.DataFrame(index=raw.index)
    for rc, cc in {**CORE, **ODDS_MAP}.items():
        df[cc] = raw[rc] if rc in raw.columns else np.nan
    df = df.dropna(subset=["home_raw", "away_raw", "date"], how="any")

    df["date"] = parse_dates(df["date"])
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ODDS_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        df[c] = v.where(v > 1.0)  # decimal odds must exceed 1
    # unplayed fixtures (no full-time score) are not matches yet
    df = df[df["fthg"].notna() & df["ftag"].notna()].copy()

    df["home"] = map_teams(df["home_raw"].str.strip(), team_map)
    df["away"] = map_teams(df["away_raw"].str.strip(), team_map)
    df["referee"] = df["referee"].astype("string").str.replace(r"[^A-Za-z .'\-]", "", regex=True).str.strip()
    df["season"] = start_year
    df["season_label"] = f"{start_year}/{(start_year + 1) % 100:02d}"
    return df.drop(columns=["home_raw", "away_raw"])


def add_matchweek(df: pd.DataFrame) -> pd.DataFrame:
    """matchweek = the larger of the two clubs' games-played count (1-indexed) within
    the season, in date order. Robust to postponements without needing a fixture list."""
    df = df.sort_values(["season", "date", "home", "away"], kind="stable").reset_index(drop=True)
    long = pd.concat([
        df[["season", "date"]].assign(team=df["home"], idx=df.index),
        df[["season", "date"]].assign(team=df["away"], idx=df.index),
    ]).sort_values(["season", "team", "date", "idx"], kind="stable")
    long["gp"] = long.groupby(["season", "team"]).cumcount() + 1
    mw = long.groupby("idx")["gp"].max()
    df["matchweek"] = mw.reindex(df.index).astype(int).to_numpy()
    return df


def add_market(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Benchmark odds: primary closing book where all three prices exist, else the
    fallback. mkt_source records which; the two are never averaged or mixed per row."""
    def cols(name: str) -> list[str]:  # 'PSC' -> psch/pscd/psca; 'AvgC' -> avgch/avgcd/avgca
        p = name[:-1].lower()
        return [f"{p}ch", f"{p}cd", f"{p}ca"]

    prim, fall = cfg["market"]["primary"], cfg["market"]["fallback"]
    src = pd.Series(pd.NA, index=df.index, dtype="string", name="mkt_source")
    mkt = pd.DataFrame(np.nan, index=df.index, columns=["mkt_h", "mkt_d", "mkt_a"])
    for name in (fall, prim):  # fallback first so the primary overwrites it
        c = cols(name)
        ok = df[c].notna().all(axis=1)
        mkt.loc[ok, :] = df.loc[ok, c].to_numpy()
        src[ok] = name
    return pd.concat([df, mkt, src], axis=1)


def build_clean(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    team_map = load_team_map(ROOT / cfg["data"]["teams_path"])
    frames = []
    for y in range(cfg["data"]["first_season"], current_season_start() + 1):
        p = raw_path(cfg, y)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run `make data` first")
        frames.append(clean_season(p, y, team_map))
    df = add_market(add_matchweek(pd.concat(frames, ignore_index=True)), cfg)
    first = ["season", "season_label", "matchweek", "date", "kickoff", "home", "away",
             "fthg", "ftag", "ftr", "hthg", "htag", "htr", "referee"]
    return df[first + [c for c in df.columns if c not in first]]


def write_clean(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = build_clean(cfg)
    out = ROOT / cfg["data"]["clean_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} matches, {df['season'].nunique()} seasons)")
    return df


def load_clean(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(ROOT / cfg["data"]["clean_path"])


if __name__ == "__main__":
    write_clean()
