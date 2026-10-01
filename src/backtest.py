"""Walk-forward backtest. Model-agnostic: it only ever calls fit / predict_proba.

For each matchweek block (season, matchweek), in order of the block's first date, a fresh
model is fitted on every match dated strictly before that first date and then predicts the
block's fixtures. Fixtures are handed over WITHOUT result columns. Seasons before the
burn-in cut-off are used for training only and never predicted or scored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
from clean import load_clean  # noqa: E402
from ingest import ROOT, load_config  # noqa: E402
from models import REGISTRY, TUNING, _dc  # noqa: E402
from models.base import GenerativeModel, Model  # noqa: E402
from models.reducers import to_over_under  # noqa: E402

# Everything a fixture list legitimately knows before kick-off.
FIXTURE_COLS = ["season", "season_label", "matchweek", "date", "kickoff", "home", "away"]
ODDS_COLS = ["mkt_h", "mkt_d", "mkt_a", "mkt_source"]
PRED_COLS = ["p_home", "p_draw", "p_away"]


def make_fixtures(block: pd.DataFrame, requires_odds: bool) -> pd.DataFrame:
    return block[FIXTURE_COLS + (ODDS_COLS if requires_odds else [])].copy()


def walk_forward(matches: pd.DataFrame, factory: Callable[[], Model], burn_in_seasons: int = 3) -> pd.DataFrame:
    """Run the expanding-window backtest and return one prediction row per scored match."""
    matches = matches.sort_values(["date", "home", "away"], kind="stable").reset_index(drop=True)
    first_scored = int(matches["season"].min()) + burn_in_seasons
    requires_odds = bool(getattr(factory(), "requires_odds", False))

    blocks = []
    for (season, mw), g in matches.groupby(["season", "matchweek"], sort=False):
        if season >= first_scored:
            blocks.append((g["date"].min(), season, mw, g))
    blocks.sort(key=lambda b: (b[0], b[1], b[2]))

    out = []
    for start, season, mw, g in blocks:
        target = g[g["mkt_source"].notna()] if requires_odds else g
        if target.empty:
            continue
        train = matches[matches["date"] < start]
        assert train.empty or train["date"].max() < start, "training data reaches the prediction date"
        model = factory().fit(train)
        p = np.asarray(model.predict_proba(make_fixtures(target, requires_odds)), dtype=float)
        _check_probs(p, len(target), season, mw)
        res = target[["season", "matchweek", "date", "home", "away", "ftr", "mkt_source"]].copy()
        res[PRED_COLS] = p
        if isinstance(model, GenerativeModel):   # goals-derived markets ride along for free
            res["p_over25"] = to_over_under(model.predict_scoreline(make_fixtures(target, requires_odds)), 2.5)[:, 0]
        out.append(res)
    if not out:
        raise ValueError("no matches to predict: check burn-in against the data range")
    return pd.concat(out, ignore_index=True)


def _check_probs(p: np.ndarray, n: int, season, mw) -> None:
    if p.shape != (n, 3) or not np.isfinite(p).all():
        raise ValueError(f"bad prediction shape/values in season {season} week {mw}: {p.shape}")
    if np.abs(p.sum(axis=1) - 1).max() > 1e-9 or (p <= 0).any() or (p >= 1).any():
        raise ValueError(f"probabilities not in (0,1) or not summing to 1 in season {season} week {mw}")


def score_subsets(preds: pd.DataFrame, model: str, cfg: dict) -> list[dict]:
    """Metrics rows for a model on each named subset it has predictions for."""
    h = ev.config_hash(cfg)
    masks = {
        "all_scored": np.ones(len(preds), bool),
        "closing_any": preds["mkt_source"].notna().to_numpy(bool),
        "closing_psc": (preds["mkt_source"] == "PSC").fillna(False).to_numpy(bool),
    }
    rows = []
    for name, m in masks.items():
        sub = preds[m]
        if sub.empty:
            continue
        rows.append({"model": model, "subset": name,
                     **ev.score(sub[PRED_COLS].to_numpy(), ev.encode_outcome(sub["ftr"])), "config_hash": h})
    return rows


def _parallel(fn, args: list) -> list:
    """Map fn over args in worker processes, preserving order. Each job is a pure function of
    the config files and data on disk, so the output does not depend on scheduling."""
    from concurrent.futures import ProcessPoolExecutor
    import os
    with ProcessPoolExecutor(max_workers=max(1, min(len(args), os.cpu_count() or 1))) as pool:
        return list(pool.map(fn, args))


def _grid_job(arg) -> pd.DataFrame:
    model_name, v = arg
    cfg = load_config()
    return walk_forward(load_clean(cfg), lambda: TUNING[model_name]["make"](cfg, v), cfg["backtest"]["burn_in_seasons"])


def _lookback_job(arg) -> pd.DataFrame:
    xi, lb = arg
    cfg = load_config()
    return walk_forward(load_clean(cfg), lambda: _dc(cfg, xi=xi, lookback=lb), cfg["backtest"]["burn_in_seasons"])


def walk_forward_tuned(matches: pd.DataFrame, make: Callable[[float], Model], grid: list, default: float,
                       burn_in_seasons: int = 3, runs: dict | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Tune one hyperparameter without peeking.

    Every grid value gets a full walk-forward run. The reported model then uses, for each
    scored season s, the value with the lowest RPS over the out-of-sample predictions of
    seasons < s only (those results are all known before s kicks off). The first scored
    season has no such evidence and uses `default`.

    `runs` may carry precomputed {value: predictions} (e.g. from parallel workers); the
    results are identical to the serial path because each run is independent.

    Returns (final predictions, {value: predictions}, chosen value per season).
    """
    if runs is None:
        runs = {v: walk_forward(matches, lambda v=v: make(v), burn_in_seasons) for v in grid}
    ref = runs[grid[0]]
    for v in grid[1:]:
        assert runs[v][["date", "home", "away"]].equals(ref[["date", "home", "away"]])
    chosen, parts = [], []
    for s in sorted(ref["season"].unique()):
        prior = ref["season"] < s
        if prior.any():
            y = ev.encode_outcome(ref.loc[prior, "ftr"])
            v_star = min(grid, key=lambda v: ev.rps(runs[v].loc[prior, PRED_COLS].to_numpy(), y))
        else:
            v_star = default
        chosen.append({"season": int(s), "chosen": v_star})
        rows = runs[v_star].loc[runs[v_star]["season"] == s]
        parts.append(rows)
    return pd.concat(parts).sort_index(), runs, pd.DataFrame(chosen)


def run(model_name: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    if model_name not in REGISTRY:
        raise SystemExit(f"unknown model {model_name!r}; choose from {sorted(REGISTRY)}")
    matches = load_clean(cfg)
    burn = cfg["backtest"]["burn_in_seasons"]
    if model_name in TUNING:
        t = TUNING[model_name]
        grid = list(t["grid"](cfg))
        pre = _parallel(_grid_job, [(model_name, v) for v in grid])
        preds, runs, chosen = walk_forward_tuned(matches, lambda v: t["make"](cfg, v), grid, t["default"](cfg), burn,
                                                 runs=dict(zip(grid, pre)))
        rows = []
        for v, r in runs.items():
            y = ev.encode_outcome(r["ftr"])
            rows.append({t["param"]: v, **ev.score(r[PRED_COLS].to_numpy(), y)})
        curve = pd.DataFrame(rows)
        rep = ROOT / "reports"
        curve.to_csv(rep / f"tuning_{model_name}.csv", index=False, float_format="%.6f")
        chosen.to_csv(rep / f"tuning_{model_name}_chosen.csv", index=False)
        if model_name == "dixon_coles":
            _dc_lookback_sensitivity(matches, cfg, curve, burn)
        base = ev.rps(*_baseline(matches, cfg))
        ev.plot_tuning_curve(curve[t["param"]], curve["rps"], t["param"],
                             f"{model_name}: out-of-sample RPS by {t['param']} (walk-forward, "
                             f"n = {int(curve['n'].iloc[0]):,} matches)",
                             rep / "figures" / t["figure"], baseline=base, baseline_label="base rate")
    else:
        preds = walk_forward(matches, lambda: REGISTRY[model_name](cfg), burn)
    out = ROOT / "data" / "predictions" / f"{model_name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    metrics = ev.write_metrics(score_subsets(preds, model_name, cfg))
    if "p_over25" in preds.columns:
        ev.report_ou25(cfg)
    return metrics


def _dc_lookback_sensitivity(matches: pd.DataFrame, cfg: dict, curve: pd.DataFrame, burn: int) -> None:
    """Post-hoc check, not used for the headline numbers: at the best full-sample xi, how much
    does the length of the training window matter?"""
    xi = float(curve.loc[curve["rps"].idxmin(), "xi"])
    lbs = cfg["models"]["dixon_coles"]["lookback_sensitivity"]
    rows = []
    for lb, p in zip(lbs, _parallel(_lookback_job, [(xi, lb) for lb in lbs])):
        rows.append({"lookback_years": lb, "xi": xi, **ev.score(p[PRED_COLS].to_numpy(), ev.encode_outcome(p["ftr"]))})
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "dc_lookback_sensitivity.csv", index=False, float_format="%.6f")


def _baseline(matches: pd.DataFrame, cfg: dict):
    b = walk_forward(matches, lambda: REGISTRY["baserate"](cfg), cfg["backtest"]["burn_in_seasons"])
    return b[PRED_COLS].to_numpy(), ev.encode_outcome(b["ftr"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    print(run(args.model).to_string(index=False))
