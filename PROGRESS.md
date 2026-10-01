# Progress

## M1 Ingest + clean: done (2026-09-30)

**Built:** `src/ingest.py` (download and cache 27 seasons), `src/clean.py` (one normalised match
table, team mapping, derived matchweek, benchmark-market columns with `mkt_source`),
`config/teams.json` (46 clubs), `config.yaml`, `Makefile`, `DATA_NOTES.md`, `tests/test_clean.py`.

**Verified:** 15 tests pass. All completed seasons have 380 matches and every club plays 38.
No duplicate keys, no nulls in required columns, FTR agrees with goals on all 9,930 rows.
First closing-odds season is 2012/13, confirmed empirically and pinned in `config.yaml`.

**Open issues / decisions:**
- Pinnacle closing odds disappear from 2026-01-17 (170 matches in 2025/26, all of 2026/27).
  The AvgC fallback is recorded per match. M2 must compare models on one fixed subset.
- 2003/04 and 2004/05 raw files are ragged; a plain `read_csv` loses 45 matches per season.
  The parser handles it (DATA_NOTES quirk 2).
- Python 3.10 is the only interpreter on this machine; the brief asks for 3.11+. Code uses
  nothing 3.11-specific. `make` is not installed on this Windows box, so the Makefile is
  untested here; the equivalent commands (`python src/ingest.py`, `python src/clean.py`,
  `python -m pytest tests`) were run directly.
- Dependencies added beyond the §1 list: `pyarrow` (parquet) and `pyyaml` (config.yaml),
  both implied by the brief's file formats. `requests` is not used; ingest uses `urllib`.
- One test of mine was wrong on first run (`test_matchweek_counts_games_played` had a fixture
  where a team played twice on one day). I fixed the fixture, not the code.

## M2 De-vig + evaluation module: done (2026-09-30)

**Built:** `src/devig.py` (proportional and Shin), `src/evaluate.py` (RPS, log loss, Brier,
reliability table, ECE, reliability plot, metrics.csv upsert with config hash, market report),
`src/models/base.py` (Model / GenerativeModel contract), `src/models/market.py` (market as a
model), `tests/test_evaluate.py`.

**Verified:** 43 tests pass (28 new). Market over the closing-odds subset (n = 5,370, 2012/13 to
2026/27): RPS 0.1937, log loss 0.9562, Brier 0.5661. Two `make market` runs produce a
byte-identical `reports/metrics.csv`. `reports/figures/calibration_market.png` exists.

**Notes and open issues:**
- Two subsets are recorded: `closing_any` (Pinnacle, else AvgC; 5,150 + 220 matches) and
  `closing_psc` (Pinnacle only, 5,150). Models must be compared with the market on the same one.
- Brier is the multiclass sum over three outcomes (range 0 to 2), not the mean; it equals the sum of
  sklearn's binary Brier scores per class.
- Log loss clips probabilities at 1e-15, as scikit-learn does.
- Burn-in seasons (2000 to 2002) sit before the closing-odds era, so they never touch the market subset.
- The isotonic recalibration test is deferred to M7 as planned.

## M3 Walk-forward harness + base rates: done (2026-09-30)

**Built:** `src/backtest.py` (expanding-window loop, one fit per matchweek block, model-agnostic),
`src/features.py` (`build_features(matches, as_of)`: per-team games played, last-5 form, rest days),
`src/models/baserate.py`, `src/models/__init__.py` (registry: baserate, market),
`tests/test_leakage.py`. `make backtest MODEL=baserate` writes metrics rows.

**Verified:** 59 tests pass (16 new). Base rate scores RPS 0.2287 over all 8,790 scored matches and
0.2322 on the 5,370-match closing subset (market: 0.1937). Backtesting the market through the loop
reproduces the direct market numbers from M2 to the last digit. The mutation test was checked to
have teeth: a deliberately leaky model (fits on the full data) fails the same comparison.

**Design decisions / notes:**
- A block is one (season, matchweek); the training cut-off is the block's first date, strictly
  before. A postponed match keeps its matchweek and is predicted with the block's original cut-off,
  which is conservative, never leaky.
- `requires_odds` on `Model` lets the loop pass odds columns and restrict predictions to matches with
  closing odds, without ever checking model type. For the market, `all_scored` therefore equals
  `closing_any`.
- Compare models on the same subset: `all_scored` (post burn-in, no odds needed), `closing_any`,
  `closing_psc`.
- Predictions are stored in `data/predictions/` (gitignored).
- The Elo-mutation test from the brief is covered generically: it runs over every model in the
  registry, so Elo is tested the moment it is registered.

## M4 Elo: done (2026-09-30)

**Built:** `src/models/elo.py` (rating updates with goal-difference multiplier; ordered-logit map from
rating gap to H/D/A, fitted in `fit()`), `walk_forward_tuned` in `src/backtest.py` (grid sweep with
per-season out-of-sample choice), `plot_tuning_curve` in `src/evaluate.py`, `TUNING` in
`src/models/__init__.py`, `tests/test_models.py`.

**Verified:** 77 tests pass (18 new). Elo RPS 0.1982 vs base rate 0.2287 (all scored, n = 8,790);
0.2002 vs market 0.1937 on the closing subset (n = 5,370; paired difference 0.0065, SE 0.0008).
`reports/figures/elo_k_tuning.png`, `reports/tuning_elo.csv` (curve) and `reports/tuning_elo_chosen.csv`
(K per season) exist. Best fixed K over the whole walk-forward is 20 (RPS 0.19804); the honest
season-by-season choice scores 0.19825. The result is within a few thousandths of the market floor and
well short of it, so there is nothing to investigate as a leak. The mutation test ran against Elo
through the registry and a direct test checks ratings are unchanged by later results.

**Notes:**
- The K curve is a sweep over the whole walk-forward, so its minimum is not itself an unbiased
  estimate. The reported `elo` row is the nested, season-by-season choice, which is.
- Home advantage in the rating update is a fixed 65 points (`config.yaml`); the ordered logit's
  cutpoints absorb whatever is left. Not tuned.
- No between-season regression of ratings. Left out to keep the model to the brief.
- Runtime: `make backtest MODEL=elo` takes roughly 6 minutes (6 K values x ~900 refits).
- A test of mine used a grid that did not contain the default K; fixed the test.

## M5 Independent Poisson + scoreline matrix + reducers: done (2026-09-30)

**Built:** `src/models/reducers.py` (`to_1x2`, `to_over_under`, `to_btts`, `to_asian_handicap`),
`src/models/poisson.py` (analytic-gradient MLE, sum-to-zero identifiability by construction, 8x8 matrix
renormalised after truncation), `GenerativeModel.predict_proba = to_1x2(predict_scoreline)` in
`models/base.py`, over/under closing-odds comparison (`evaluate.report_ou25`, `devig.proportional_two_way`,
`reports/metrics_ou25.csv`), Poisson lookback tuning (`reports/figures/poisson_lookback_tuning.png`,
`reports/tuning_poisson*.csv`). The backtest attaches `p_over25` to any generative model's predictions.

**Verified:** 93 tests pass (16 new). Reducers agree with brute-force sums over the matrix, including
whole, half and quarter Asian lines. `predict_proba` equals `to_1x2(predict_scoreline)` exactly. Gradient
checked numerically, fit checked to be a likelihood maximum, true parameters recovered on 60 simulated
seasons, ordering recovered. Poisson RPS 0.2004 (all scored, n = 8,790), 0.2019 on the closing subset
(market 0.1937). Over/under 2.5 on 2,710 matches: log loss Poisson 0.6921, historical rate 0.6916, market
0.6734.

**Notes and open issues:**
- Poisson lookback: 2 years wins the curve (RPS 0.2003; 3 years 0.2005); one year and eight years are
  clearly worse. Chosen season by season out of sample, as for Elo.
- Poisson over/under has no edge over the base rate; the market's over/under line is far better. Its
  probabilities are too spread out (calibration bins in the README). Expected to improve with time decay in M6.
- The `penaltyblog` cross-check the brief asks for is an M6 test (Dixon-Coles likelihood); it is not
  installed and I will ask before adding it.
- Unseen clubs (promoted, absent from the window) get average attack/defence; no promoted-club prior.
- `config_hash` in `metrics.csv` changes whenever `config.yaml` changes (it hashes the whole file), so rows
  written before this milestone carry older hashes. A clean `make all` rewrites every row consistently; I
  will do that at M7.
- The Poisson backtest is slow (6 lookbacks x ~900 refits); run it in the background.

## M6 Dixon-Coles: done (2026-09-30)

**Built:** `src/models/dixon_coles.py` (tau correction, exponential time decay, analytic gradient incl. rho,
sum-to-zero parametrisation shared with Poisson), registry and tuning entries, xi sweep
(`reports/figures/dc_xi_tuning.png`, `reports/tuning_dixon_coles.csv`, `..._chosen.csv`), post-hoc lookback
sensitivity (`reports/dc_lookback_sensitivity.csv`), penaltyblog cross-checks in `tests/test_models.py`.
`penaltyblog` 1.12.3 installed with the user's approval; it is imported only by the test file (guarded, so the
suite still runs without it) and never by `src/`.

**Verified:** 110 tests pass (17 new). Against penaltyblog: time weights identical, our log-likelihood
evaluated at penaltyblog's fitted parameters equals the value it reports (agreement to about 1e-12), scoreline
grid and 1X2 and over/under 2.5 agree to 1e-12 at identical parameters, and RPS matches its `rps_average`.
Our optimum is fractionally higher than penaltyblog's (-1064.0949 against -1064.0982 on one check window), i.e.
it stops slightly short of the maximum. xi = 0 gives equal weights; tau off reproduces Poisson to 1e-5 in the
parameters. Gradient checked numerically.

**Results:** RPS 0.1989 (all scored, n = 8,790), 0.2000 on the closing subset (market 0.1937). Paired: better
than Poisson by 0.0014 (SE 0.0003), worse than Elo by 0.0007 (SE 0.0005, a tie), 0.0062 behind the market
(SE 0.0008). The xi curve has a flat minimum at 0.0018 to 0.0025; nested out-of-sample choice is 0.0018 in 21 of 24 seasons (the first uses the default)
and 0.0025 in 3.

**Notes and open issues:**
- Lookback is fixed at 5 years by config, not tuned jointly with xi (a 2-D sweep was not worth the runtime);
  the after-the-fact check shows 3/5/8 years differ by under 0.0003 RPS.
- xi is per day. The brief's grid values were used unchanged.
- Nothing suspiciously good: DC does not beat the market or Elo materially, so no leak suspected. The
  mutation test also runs against Dixon-Coles through the registry.
- The Dixon-Coles backtest is slow (30+ minutes: 6 xi values plus 3 lookback runs); run in the background.
- Dependency note: `pip install penaltyblog` did not change numpy, pandas or scipy versions.

## M7 Calibration + write-up: done (2026-09-30)

**Built:** isotonic recalibration (pool-adjacent-violators via `scipy.optimize.isotonic_regression`, fitted
per outcome on earlier seasons' out-of-sample predictions only, renormalised), `report_models` in
`src/evaluate.py` (model-vs-market table, ECE, calibration curves for all five models, RPS by season,
gap-to-market, model-vs-market scatters), process-parallel tuning sweeps in `src/backtest.py`, final
Makefile (`make all`: data, market, backtests, report, test), README rewritten.

**Verified:** 118 tests pass (8 new, in `tests/test_evaluate.py`): isotonic fit equals scikit-learn's and a
hand case, recalibration never uses the season it predicts or later (mutation test with teeth), it helps a
deliberately overconfident synthetic model and not a calibrated one, paired gap hand-computed, common-subset
helper. A complete rebuild from an empty `data/` and `reports/` (raw files re-downloaded) followed by a second
complete rebuild gave 25 report files with identical md5 hashes, PNGs included. All rows carry one config hash
(2f9bdfc7ecc7). The parallel tuning gives the same numbers as the serial path (checked on Elo).

**Results:** Dixon-Coles 0.1999, Elo 0.2002, Poisson 0.2019, base rate 0.2322 against the market 0.1937
(n = 5,370). Market better than every model in 14 of the 15 seasons 2012/13 to 2026/27, the exception being
the 50-match 2026/27 partial season. Isotonic recalibration made all five models slightly worse out of sample
(+0.0002 to +0.0018 RPS); reported as such in the README. The README opens with the headline table, has eight
figures and a limitations section naming the promoted-team cold start, no schedule congestion, no squad
information, and the closing-odds subset.

**Open issues:**
- `make` is not installed on the development machine, so `make all` itself has not been run; the same
  commands in the same order were (Makefile recipes are simple, but a typo would only show up under `make`).
- Only tested on Python 3.10 (brief asks for 3.11+).
- Editing `config.yaml` changes the config hash and therefore every metrics row; rerun `make all` after any
  change to it (none pending).
- Scope left untouched as the brief requires: no P&L, no other leagues, no xG, no corners or cards.

## Pre-publication fixes (items 1-4): done (2026-10-01)

**Item 1, CI and portability.** Read the Makefile and scripts for Windows-isms: none (no backslash paths, no
cmd-style commands, `if __name__` guard present around the multiprocessing entry point). Changes: parallel worker
count is now `PL_WORKERS` or `os.cpu_count()` (`PL_WORKERS=1` runs serially); `requirements.txt` pinned to the
tested versions and now lists `scikit-learn`, which the tests import but was missing; `.github/workflows/ci.yml`
added (Ubuntu, Python 3.11, pip cache, `make all`, informational `git diff --stat -- reports/`, `reports/` uploaded
as an artifact, `workflow_dispatch` plus push to `main` ignoring markdown, concurrency cancel-in-progress, 300
minute timeout). A test I wrote in M6 hard-coded the scored-match count (8790) and would have failed on any fresh run
because the current season keeps growing; it now compares with the live data (a wrong test, fixed here as the brief
requires). README gains the snapshot line (2026-09-20, the most recent match in the data). The Python 3.10 / no-`make`
caveat stays until CI is green.

**Item 2, ordering and subsets.** New `reports/paired_model_comparisons.csv` (via the existing `paired_gap` helper,
now also called from `report_models`). Confirmed from stored predictions: Dixon-Coles minus Elo is +0.0007
(SE 0.0005) over all 8,790 scored matches, matching the README, and -0.0003 (SE 0.0007, 95% interval -0.0016 to
+0.0011) on the 5,370 closing-odds matches, so the sign flips and the two are indistinguishable; sentence added under
the headline table. The Dixon-Coles vs Poisson gain (0.0014, SE 0.0003) was computed ad hoc in M6 and not stored; it
is now in the CSV (all 8,790: -0.0014; closing 5,370: -0.0020). Subset audit: every RPS figure in the model ladder is
on the 8,790 scored matches except where now labelled otherwise (the base-rate, Elo, Poisson, xi-curve, K-curve and
0.030 figures were all verified against `metrics.csv` and `tuning_*.csv`). Three sentences were wrong or unlabelled
and were rewritten, not the numbers: the recalibration result mixed subsets (the four models are on 8,030 matches from
2005/06, the market on 4,610 from 2014/15; now stated); "swings between roughly 0.18 and 0.22" (the per-season table
reaches 0.225); and two interpretive claims with no computed backing ("less sharp than the market", "noisier ... which is
what estimating 20 clubs' strengths looks like") were softened to what the tables show.

**Item 3, closing-price reasoning.** The profit bullet is rewritten: no model beats the closing line on any proper scoring
rule so there is no basis for a profit claim; reasons a closing-odds simulation would mislead are that the last quote cannot
be known in advance, the closing price absorbs later information, and the price may not be available at size. It says "no
edge found here". No other absolute wording about market efficiency was found in the README.

**Item 4, season count.** From `reports/rps_by_season.csv`: the market beat the best of the other four models in every one
of the 14 complete seasons (2012/13 to 2025/26), smallest margin 0.0007 (2019/20). The best model per season is Dixon-Coles
nine times, Elo four, Poisson once. 2026/27 has 50 matches and is called too small.

**Publishing.** `git init` was run in this session (the folder was not a repository); the baseline commit is the v1 build.
`CLAUDE.md` and `CLAUDE_FIXES.md` are working instructions for the assistant, mention a private odds-collection tool and
the owner's job search, and are excluded locally (`.git/info/exclude`), not published. `data/predictions/` was missing from
`.gitignore` and is now ignored. CI result: see below (not recorded in this session; `gh` is not installed here).
