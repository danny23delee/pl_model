"""Single source of truth for scoring. No metric is computed anywhere else.

Probabilities are (n, 3) arrays ordered (home, draw, away); outcomes are ints
0 = home win, 1 = draw, 2 = away win. Accuracy is deliberately not implemented.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import ROOT, load_config  # noqa: E402

OUTCOMES = ["home", "draw", "away"]
LOG_EPS = 1e-15
METRIC_COLS = ["model", "subset", "n", "rps", "log_loss", "brier", "config_hash"]


# ---------------------------------------------------------------- inputs
def encode_outcome(ftr) -> np.ndarray:
    """'H'/'D'/'A' -> 0/1/2."""
    m = {"H": 0, "D": 1, "A": 2}
    return np.array([m[x] for x in ftr], dtype=int)


def _prep(probs, outcomes) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"probs must have shape (n, 3), got {p.shape}")
    if len(y) != len(p):
        raise ValueError("probs and outcomes differ in length")
    if np.isnan(p).any():
        raise ValueError("probs contain NaN")
    if ((y < 0) | (y > 2)).any():
        raise ValueError("outcomes must be 0, 1 or 2")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to 1")
    return p, y


def _onehot(y: np.ndarray) -> np.ndarray:
    return np.eye(3)[y]


# ---------------------------------------------------------------- metrics
def rps_per_match(probs, outcomes) -> np.ndarray:
    """Ranked probability score for 3 ordered outcomes:
    (1 / (r - 1)) * sum_{k=1}^{r-1} (cumP_k - cumO_k)^2, r = 3. Range [0, 1]."""
    p, y = _prep(probs, outcomes)
    cum_diff = np.cumsum(p, axis=1)[:, :2] - np.cumsum(_onehot(y), axis=1)[:, :2]
    return (cum_diff ** 2).sum(axis=1) / 2.0


def rps(probs, outcomes) -> float:
    return float(rps_per_match(probs, outcomes).mean())


def log_loss_per_match(probs, outcomes) -> np.ndarray:
    p, y = _prep(probs, outcomes)
    return -np.log(np.clip(p[np.arange(len(y)), y], LOG_EPS, 1.0))


def log_loss(probs, outcomes) -> float:
    return float(log_loss_per_match(probs, outcomes).mean())


def brier_per_match(probs, outcomes) -> np.ndarray:
    """Multiclass Brier: squared error summed over the 3 outcomes. Range [0, 2]."""
    p, y = _prep(probs, outcomes)
    return ((p - _onehot(y)) ** 2).sum(axis=1)


def brier(probs, outcomes) -> float:
    return float(brier_per_match(probs, outcomes).mean())


def score(probs, outcomes) -> dict:
    return {"n": len(outcomes), "rps": rps(probs, outcomes),
            "log_loss": log_loss(probs, outcomes), "brier": brier(probs, outcomes)}


def binary_log_loss(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(-(y * np.log(np.clip(p, LOG_EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, LOG_EPS, 1))).mean())


def binary_brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(((p - y) ** 2).mean())


# ---------------------------------------------------------------- calibration
def reliability_table(probs, outcomes, n_bins: int = 10) -> pd.DataFrame:
    """Per outcome and probability bin: observations, mean predicted, observed frequency.
    A bin with no observations has NaN for mean_pred and obs_freq, never 0."""
    p, y = _prep(probs, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for k, name in enumerate(OUTCOMES):
        b = np.clip(np.digitize(p[:, k], edges[1:-1], right=False), 0, n_bins - 1)
        hit = (y == k).astype(float)
        for i in range(n_bins):
            m = b == i
            n = int(m.sum())
            rows.append({"outcome": name, "bin": i, "lo": edges[i], "hi": edges[i + 1], "n": n,
                         "mean_pred": p[m, k].mean() if n else np.nan,
                         "obs_freq": hit[m].mean() if n else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> dict:
    """Per outcome: sum over non-empty bins of (n_bin / N) * |obs_freq - mean_pred|."""
    out = {}
    for name, g in table.groupby("outcome", sort=False):
        g = g[g["n"] > 0]
        out[name] = float((g["n"] * (g["obs_freq"] - g["mean_pred"]).abs()).sum() / g["n"].sum())
    return out


def plot_reliability(probs, outcomes, title: str, path: Path, n_bins: int = 10) -> pd.DataFrame:
    """One reliability chart per outcome (small multiples), observations per bin annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = reliability_table(probs, outcomes, n_bins)
    ece = expected_calibration_error(tab)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="white")
    for ax, name in zip(axes, OUTCOMES):
        g = tab[(tab["outcome"] == name) & (tab["n"] > 0)]
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect")
        ax.plot(g["mean_pred"], g["obs_freq"], color="#1f5fa8", lw=1.5)
        thick, thin = g[g["n"] >= 30], g[g["n"] < 30]
        ax.plot(thick["mean_pred"], thick["obs_freq"], "o", color="#1f5fa8", label="observed (30+ matches)")
        ax.plot(thin["mean_pred"], thin["obs_freq"], "o", mfc="white", mec="#1f5fa8", label="fewer than 30 matches")
        for _, r in g.iterrows():
            ax.annotate(f"{int(r['n'])}", (r["mean_pred"], r["obs_freq"]), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7, color="#555555")
        ax.set_title(f"{name} (ECE {ece[name]:.3f})", fontsize=10)
        ax.set_xlabel("mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor("white")
        ax.grid(color="#e6e6e6", lw=0.6)
    axes[0].set_ylabel("observed frequency")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(f"{title}  |  n = {len(outcomes):,}; numbers under points are observations per bin", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return tab


def plot_tuning_curve(x, y, xlabel: str, title: str, path: Path, baseline: float | None = None,
                      baseline_label: str = "baseline") -> None:
    """Metric (RPS, lower is better) against a hyperparameter; the minimum is marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = np.asarray(x, float), np.asarray(y, float)
    fig, ax = plt.subplots(figsize=(6.5, 4.4), facecolor="white")
    ax.plot(x, y, marker="o", color="#1f5fa8", lw=1.6, label="out-of-sample RPS")
    best = int(np.argmin(y))
    ax.scatter([x[best]], [y[best]], s=110, facecolors="none", edgecolors="#c0392b", lw=1.8,
               label=f"minimum at {xlabel} = {x[best]:g}")
    if baseline is not None:
        if baseline - y.max() < 3 * (y.max() - y.min()):
            ax.axhline(baseline, color="#999999", ls="--", lw=1, label=f"{baseline_label} ({baseline:.4f})")
        else:  # would flatten the curve; say so instead of drawing it
            ax.plot([], [], " ", label=f"{baseline_label}: {baseline:.4f} (off scale)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RPS (lower is better)")
    ax.set_title(title, fontsize=9)
    ax.set_facecolor("white")
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- reporting
def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def write_metrics(rows: list[dict], path: Path | None = None) -> pd.DataFrame:
    """Upsert rows keyed by (model, subset) into reports/metrics.csv, sorted for determinism."""
    path = path or ROOT / "reports" / "metrics.csv"
    new = pd.DataFrame(rows)[METRIC_COLS]
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(new["model"], new["subset"]))
        old = old[[k not in keys for k in zip(old["model"], old["subset"])]]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["subset", "model"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6f")
    return new


def market_subsets(matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Named evaluation subsets. Models are always compared on the same subset."""
    has = matches[matches["mkt_source"].notna()]
    return {"closing_any": has, "closing_psc": has[has["mkt_source"] == "PSC"]}


def report_market(cfg: dict | None = None) -> pd.DataFrame:
    """M2 deliverable: the market's RPS / log loss / Brier and calibration curve."""
    from clean import load_clean
    from models.market import MarketModel

    cfg = cfg or load_config()
    matches = load_clean(cfg)
    h = config_hash(cfg)
    rows = []
    for subset, df in market_subsets(matches).items():
        y = encode_outcome(df["ftr"])
        for method in ("proportional", "shin"):
            p = MarketModel(method).fit(df).predict_proba(df)
            rows.append({"model": f"market_{method}", "subset": subset, **score(p, y), "config_hash": h})
    table = write_metrics(rows)

    df = market_subsets(matches)["closing_any"]
    method = cfg["market"]["devig"]
    p = MarketModel(method).predict_proba(df)
    plot_reliability(p, encode_outcome(df["ftr"]),
                     f"Market closing line ({method} de-vig), calibration",
                     ROOT / "reports" / "figures" / "calibration_market.png", cfg["evaluate"]["calibration_bins"])
    return table


def report_ou25(cfg: dict | None = None) -> pd.DataFrame | None:
    """Over/under 2.5 goals: every generative model's derived P(over 2.5) against the de-vigged
    closing over/under line, on the matches where both exist. Written to reports/metrics_ou25.csv.
    The market price is Pinnacle closing where both sides exist, else market-average closing."""
    from clean import load_clean
    from devig import proportional_two_way

    cfg = cfg or load_config()
    files = sorted((ROOT / "data" / "predictions").glob("*.parquet"))
    preds = {f.stem: pd.read_parquet(f) for f in files}
    preds = {k: v for k, v in preds.items() if "p_over25" in v.columns}
    if not preds:
        return None
    m = load_clean(cfg).sort_values(["date", "home", "away"], kind="stable").reset_index(drop=True)
    m["over"] = ((m["fthg"] + m["ftag"]) > 2.5).astype(float)
    ps = m["psc_o25"].notna() & m["psc_u25"].notna()
    av = m["avgc_o25"].notna() & m["avgc_u25"].notna()
    m["ou_src"] = np.where(ps, "PSC", np.where(av, "AvgC", ""))
    o = np.where(ps, m["psc_o25"], m["avgc_o25"])
    u = np.where(ps, m["psc_u25"], m["avgc_u25"])
    has = (m["ou_src"] != "").to_numpy()
    m["p_mkt"] = np.nan
    m.loc[has, "p_mkt"] = proportional_two_way(o[has], u[has])
    # expanding historical over-rate using strictly earlier dates only
    day = m.groupby("date")["over"].agg(["sum", "count"])
    cum = day.cumsum().shift(1)
    m["p_base"] = m["date"].map(cum["sum"] / cum["count"])

    h = config_hash(cfg)
    common = None
    for name, p in preds.items():
        k = p[["date", "home", "away"]].merge(m.loc[has, ["date", "home", "away"]], on=["date", "home", "away"])
        common = k if common is None else common.merge(k, on=["date", "home", "away"])
    sub = m.merge(common, on=["date", "home", "away"])
    sub = sub[sub["p_base"].notna()]
    rows = []
    def add(name, p):
        rows.append({"model": name, "subset": "ou25_closing", "n": len(sub),
                     "log_loss": binary_log_loss(p, sub["over"]), "brier": binary_brier(p, sub["over"]),
                     "config_hash": h})
    add("market_ou25", sub["p_mkt"])
    add("base_rate_ou25", sub["p_base"])
    for name, p in preds.items():
        add(name, sub[["date", "home", "away"]].merge(p, on=["date", "home", "away"])["p_over25"])
    out = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    out.to_csv(ROOT / "reports" / "metrics_ou25.csv", index=False, float_format="%.6f")
    return out


# ---------------------------------------------------------------- M7: recalibration and model-vs-market
MODEL_ORDER = ["market", "dixon_coles", "elo", "poisson", "baserate"]
MODEL_LABEL = {"market": "Market (closing)", "dixon_coles": "Dixon-Coles", "elo": "Elo",
               "poisson": "Poisson", "baserate": "Base rate"}
# fixed categorical order, validated with the dataviz palette checker (adjacent-pair CVD and
# normal-vision separation pass; three slots sit under 3:1 contrast on white, so every chart
# also carries direct labels, a legend and distinct markers, and the numbers ship as CSV)
MODEL_COLOR = {"market": "#2a78d6", "dixon_coles": "#eb6834", "elo": "#1baf7a",
               "poisson": "#eda100", "baserate": "#e87ba4"}
MODEL_MARKER = {"market": "o", "dixon_coles": "s", "elo": "^", "poisson": "D", "baserate": "v"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
PRED_COLS = ["p_home", "p_draw", "p_away"]
KEY = ["date", "home", "away"]


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def load_predictions() -> dict[str, pd.DataFrame]:
    out = {}
    for name in MODEL_ORDER:
        f = ROOT / "data" / "predictions" / f"{name}.parquet"
        if f.exists():
            out[name] = pd.read_parquet(f)
    return out


# ---- isotonic recalibration -------------------------------------------------------------
def isotonic_fit(p, y):
    """Monotone non-decreasing step fit of y on p (pool-adjacent-violators). Returns (xs, ys)."""
    from scipy.optimize import isotonic_regression
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="stable")
    fitted = isotonic_regression(y[order], increasing=True).x
    xs, inv = np.unique(p[order], return_inverse=True)
    ys = np.bincount(inv, fitted) / np.bincount(inv)
    return xs, ys


def isotonic_apply(model, p) -> np.ndarray:
    xs, ys = model
    return np.interp(np.asarray(p, float), xs, ys)


def recalibrate_walk_forward(preds: pd.DataFrame, min_prior_seasons: int = 2) -> pd.DataFrame:
    """Out-of-sample isotonic recalibration. Predictions for season s are recalibrated by one
    isotonic map per outcome, fitted on the (already out-of-sample) predictions and results
    of seasons before s only; rows are then renormalised to sum to 1. Seasons with fewer than
    `min_prior_seasons` earlier seasons are dropped from the output."""
    preds = preds.sort_values(KEY, kind="stable").reset_index(drop=True)
    seasons = sorted(preds["season"].unique())
    y = encode_outcome(preds["ftr"])
    parts = []
    for i, s in enumerate(seasons):
        if i < min_prior_seasons:
            continue
        prior, cur = (preds["season"] < s).to_numpy(), (preds["season"] == s).to_numpy()
        cal = np.column_stack([
            isotonic_apply(isotonic_fit(preds.loc[prior, c], (y[prior] == k).astype(float)), preds.loc[cur, c])
            for k, c in enumerate(PRED_COLS)])
        cal = np.clip(cal, 1e-4, 1.0)
        cal /= cal.sum(axis=1, keepdims=True)
        block = preds.loc[cur].copy()
        block[[f"{c}_raw" for c in PRED_COLS]] = block[PRED_COLS].to_numpy()
        block[PRED_COLS] = cal
        parts.append(block)
    return pd.concat(parts, ignore_index=True)


def paired_gap(p_a, p_b, y) -> tuple[float, float]:
    """Mean per-match RPS difference (a minus b) and its standard error."""
    d = rps_per_match(p_a, y) - rps_per_match(p_b, y)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def report_recalibration(preds: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        rc = recalibrate_walk_forward(p)
        y = encode_outcome(rc["ftr"])
        raw = rc[[f"{c}_raw" for c in PRED_COLS]].to_numpy()
        cal = rc[PRED_COLS].to_numpy()
        diff, se = paired_gap(cal, raw, y)
        rows.append({"model": name, "first_season": int(rc["season"].min()), "n": len(rc),
                     "rps_raw": rps(raw, y), "rps_isotonic": rps(cal, y), "rps_change": diff, "rps_change_se": se,
                     "log_loss_raw": log_loss(raw, y), "log_loss_isotonic": log_loss(cal, y),
                     "brier_raw": brier(raw, y), "brier_isotonic": brier(cal, y), "config_hash": config_hash(cfg)})
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER)
    out = out.sort_values("model").reset_index(drop=True)
    out.to_csv(ROOT / "reports" / "recalibration.csv", index=False, float_format="%.6f")
    return out


# ---- model vs market --------------------------------------------------------------------
def closing_common(preds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Every model restricted to the identical matches: those the market predicts."""
    keys = preds["market"][KEY]
    return {k: v.merge(keys, on=KEY).sort_values(KEY, kind="stable").reset_index(drop=True) for k, v in preds.items()}


def plot_rps_by_season(preds: dict[str, pd.DataFrame], path: Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    for name in reversed(MODEL_ORDER):         # market drawn last, on top
        if name not in preds:
            continue
        p = preds[name]
        y = encode_outcome(p["ftr"])
        g = pd.DataFrame({"season": p["season"].to_numpy(), "r": rps_per_match(p[PRED_COLS].to_numpy(), y)})
        s = g.groupby("season")["r"].agg(["mean", "size"])
        for season, r in s.iterrows():
            rows.append({"model": name, "season": int(season), "n": int(r["size"]), "rps": r["mean"]})
        ax.plot(s.index, s["mean"], color=MODEL_COLOR[name], marker=MODEL_MARKER[name], ms=5, lw=1.6,
                label=MODEL_LABEL[name])
    _style(ax)
    ax.set_xlabel("season (start year)")
    ax.set_ylabel("RPS (lower is better)")
    ax.set_xlim(2002.5, 2026.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(MODEL_LABEL[m]) for m in MODEL_ORDER if MODEL_LABEL[m] in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    ax.set_title("RPS by season and model, walk-forward. About 380 matches per season (2026 has 50); "
                 "market from 2012 only", fontsize=10, color=INK, loc="left")
    _save(fig, path)
    return pd.DataFrame(rows)


def plot_gap_to_market(table: pd.DataFrame, path: Path, n: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.sort_values("rps_gap", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor="white")
    for i, r in t.iterrows():
        ax.errorbar(r["rps_gap"], i, xerr=1.96 * r["rps_gap_se"], fmt=MODEL_MARKER[r["model"]], ms=8,
                    color=MODEL_COLOR[r["model"]], ecolor=MODEL_COLOR[r["model"]], elinewidth=1.6, capsize=3)
        ax.annotate(f"{r['rps_gap']:+.4f}", (r["rps_gap"], i), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK_MUTED, lw=1)
    ax.set_ylim(-0.5, len(t) - 0.2)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in t["model"]])
    ax.set_xlabel("RPS minus market RPS (paired; 0 = matches the closing line; bars are 95% intervals)")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Every model trails the closing line. Same {n:,} matches for all models", fontsize=10,
                 color=INK, loc="left")
    _save(fig, path)


def plot_vs_market_scatter(model: str, p_model: np.ndarray, p_market: np.ndarray, path: Path) -> float:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = float(np.corrcoef(p_model, p_market)[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 5.4), facecolor="white")
    ax.scatter(p_market, p_model, s=7, alpha=0.25, color=MODEL_COLOR[model], edgecolors="none")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("market P(home win), de-vigged closing line")
    ax.set_ylabel(f"{MODEL_LABEL[model]} P(home win)")
    _style(ax)
    ax.set_title(f"{MODEL_LABEL[model]} vs market, home win probability. n = {len(p_model):,}, r = {r:.3f}",
                 fontsize=9, color=INK, loc="left")
    _save(fig, path)
    return r


def report_models(cfg: dict | None = None) -> None:
    """M7 outputs: model-vs-market table, calibration curves and ECE for every model,
    RPS-by-season, gap-to-market, scatter plots, isotonic recalibration test."""
    cfg = cfg or load_config()
    preds = load_predictions()
    if "market" not in preds:
        raise SystemExit("run the backtests first (market predictions missing)")
    figs = ROOT / "reports" / "figures"
    bins = cfg["evaluate"]["calibration_bins"]
    cl = closing_common(preds)
    y = encode_outcome(cl["market"]["ftr"])
    pm = cl["market"][PRED_COLS].to_numpy()
    h = config_hash(cfg)

    rows, ece_rows = [], []
    for name in MODEL_ORDER:
        if name not in cl:
            continue
        p = cl[name][PRED_COLS].to_numpy()
        assert (cl[name][KEY].to_numpy() == cl["market"][KEY].to_numpy()).all()
        s = score(p, y)
        gap, se = paired_gap(p, pm, y)
        rows.append({"model": name, **s, "rps_gap": gap, "rps_gap_se": se,
                     "log_loss_gap": s["log_loss"] - log_loss(pm, y), "brier_gap": s["brier"] - brier(pm, y),
                     "config_hash": h})
        tab = plot_reliability(p, y, f"{MODEL_LABEL[name]}, calibration", figs / f"calibration_{name}.png", bins)
        for outcome, e in expected_calibration_error(tab).items():
            ece_rows.append({"model": name, "outcome": outcome, "ece": e, "n": len(y)})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "reports" / "model_vs_market.csv", index=False, float_format="%.6f")
    pd.DataFrame(ece_rows).to_csv(ROOT / "reports" / "calibration_ece.csv", index=False, float_format="%.6f")

    plot_rps_by_season(preds, figs / "rps_by_season.png").to_csv(
        ROOT / "reports" / "rps_by_season.csv", index=False, float_format="%.6f")
    plot_gap_to_market(table[table["model"] != "market"], figs / "rps_gap_to_market.png", len(y))
    for name in ("dixon_coles", "elo"):
        if name in cl:
            plot_vs_market_scatter(name, cl[name]["p_home"].to_numpy(), cl["market"]["p_home"].to_numpy(),
                                   figs / f"scatter_{name}_vs_market.png")
    report_recalibration(preds, cfg)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "market"
    if what == "market":
        print(report_market().to_string(index=False))
    elif what == "models":
        report_models()
        print(pd.read_csv(ROOT / "reports" / "model_vs_market.csv").to_string(index=False))
    else:
        raise SystemExit("usage: python src/evaluate.py [market|models]")
