# Data notes

Source: football-data.co.uk, Premier League (`E0`), one CSV per season at
`https://www.football-data.co.uk/mmz4281/<yyzz>/E0.csv` (e.g. `2526`). Fetched by
`src/ingest.py`, cached in `data/raw/` (gitignored), normalised by `src/clean.py`
into `data/clean/matches.parquet`. All figures below were measured on the files
downloaded on 2026-09-30, not assumed.

## Coverage

27 files, 2000/01 to 2026/27. **9,930 matches** after cleaning. Every completed season
has exactly 380 matches (20 teams x 38 games). 2026/27 is in progress: 50 matches played,
last on 2026-09-20.

| Column group | Coverage |
|---|---|
| Result, teams, date | all seasons, all rows |
| Half-time score, shots, shots on target, fouls, corners, yellows, reds, referee | all seasons, all rows (verified non-null) |
| Kick-off time (`Time`) | 2019/20 onwards only |
| Bet365 opening 1X2 (`B365H/D/A`) | 2002/03 onwards |
| Pinnacle opening 1X2 (`PSH/D/A`) | 2012/13 onwards (2025/26: 210 of 380) |
| **Pinnacle closing 1X2 (`PSCH/D/A`)** | **2012/13 onwards. First season with closing odds = 2012/13 (380/380).** 2025/26: 210 of 380 |
| Market average and max, opening and closing (`Avg*`, `Max*`) | 2019/20 onwards (380/380 each season) |
| Bet365 closing (`B365C*`) | 2019/20 onwards |
| Betfair Exchange closing (`BFEC*`) | 2024/25 onwards (2025/26: 358 of 380) |
| Closing over/under 2.5 (Pinnacle `PC>2.5`, `PC<2.5`) | 2019/20 onwards; 2022/23: 379, 2023/24: 373, 2024/25: 377, 2025/26: 210 |
| Closing over/under 2.5 (market average `AvgC>2.5`) | 2019/20 onwards, complete |

`min_closing_odds_season` in `config.yaml` is therefore **2012**. `test_clean.py` asserts that the
config value equals the first season with non-null `PSCH` in the data.

Market benchmark (`mkt_*` columns): Pinnacle closing where all three prices exist, else
market-average closing, else null. The chosen book is recorded per match in `mkt_source`.

| Seasons | `mkt_source` |
|---|---|
| 2000/01 to 2011/12 | none (no closing odds) |
| 2012/13 to 2024/25 | `PSC` for all matches |
| 2025/26 | `PSC` for 210, `AvgC` for 170 |
| 2026/27 | `AvgC` for all 50 |

Bookmaker columns that are kept: for each of Pinnacle, market average, market max,
Bet365 and Betfair Exchange, the opening and closing H/D/A prices. Every other
bookmaker (William Hill, Ladbrokes, Interwetten, ...), Asian handicap columns and the
old `Bb*` Betbrain aggregates are dropped: they are not opening/closing-labelled
consistently and v1 does not use them.

## Quirks found

1. **Pinnacle closing prices stop mid-2025/26.** `PSCH/D/A` is empty for every match from
   2026-01-17 onwards (the last populated match is 2026-01-08), and absent in 2026/27. The
   market benchmark falls back to `AvgC` for those matches. In M2/M7 the market comparison
   must use a fixed subset (see README limitations) and report which book was used.
2. **2003/04 and 2004/05 files have ragged rows.** The header names 52 columns, but data
   rows in the second half of each file carry 57 to 72 fields. A default `pandas.read_csv`
   silently drops those rows (335 of 380 matches survive, dropping the ones from about
   round 30 onwards). `read_raw_season` uses the `csv` module, keeps the first
   *header-length* fields and discards the extras, so all 380 rows are recovered. Only
   bookmaker columns past the header are lost; the match, score and stat columns come
   first and are intact.
3. **Date format is mixed:** `dd/mm/yy` in most seasons, `dd/mm/yyyy` in others (2002/03,
   2015/16, 2017/18 onwards), so dates are parsed per value.
4. **Encoding:** some files carry a UTF-8 BOM (turns the first header into `ï»¿Div` if read
   as latin-1), and 2004/05 has stray non-ASCII bytes in `Referee`. Files are decoded as
   cp1252 and the BOM stripped; referee names are reduced to letters, spaces, `.`, `'`, `-`.
5. **Blank trailing rows and `Unnamed:` columns** appear in a few files; both are dropped.
6. **Unplayed fixtures:** rows without a full-time score are dropped. This is why the
   current season equals matches played to date.
7. **Invalid odds:** values of 1.0 or below are set to null (a decimal price must exceed 1).
8. **2026/27 header has 114 columns and includes `HxG`/`AxG`.** xG is out of scope for this
   project and is not ingested.
9. **Team names:** 46 distinct raw names across all seasons, all stable across seasons
   (football-data uses one spelling per club, e.g. `Man United`, `Nott'm Forest`). Mapped
   with exact matching in `config/teams.json` to snake_case ids. An unmapped name raises.
   Clubs come and go (Bradford, Blackpool and Luton appear in one season each), but ids
   are never reused, so promoted/relegated clubs need no special handling in the table.
   Cold-start for promoted clubs is a modelling problem (M4+), not a data one.
10. **`matchweek` is derived, not supplied.** The source has no round number. It is the
    larger of the two clubs' games-played counts within the season, ordered by date, so it
    stays sensible around postponements and the 2019/20 COVID restart. Every completed
    season runs 1 to 38.
