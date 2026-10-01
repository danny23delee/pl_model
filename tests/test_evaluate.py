"""M2 tests: metrics, de-vigging, calibration, market-as-a-model. Hand-computed fixtures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluate as ev  # noqa: E402
from devig import devig, overround, proportional, shin  # noqa: E402
from models.market import MarketModel  # noqa: E402


# ---------------------------------------------------------------- RPS
def test_rps_perfect_is_zero():
    p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    assert ev.rps(p, [0, 1, 2]) == 0.0


def test_rps_maximally_wrong_is_one():
    assert ev.rps(np.array([[0, 0, 1.0]]), [0]) == 1.0
    assert ev.rps(np.array([[1.0, 0, 0]]), [2]) == 1.0


def test_rps_hand_computed():
    # forecast (0.5, 0.3, 0.2), outcome draw -> obs (0, 1, 0)
    # cum forecast (0.5, 0.8), cum obs (0, 1): ((0.5)^2 + (0.2)^2) / 2 = 0.145
    assert ev.rps([[0.5, 0.3, 0.2]], [1]) == pytest.approx(0.145)
    # outcome home: cum obs (1, 1): ((0.5)^2 + (0.2)^2) / 2 = 0.145 as well
    assert ev.rps([[0.5, 0.3, 0.2]], [0]) == pytest.approx(0.145)
    # outcome away: cum obs (0, 0): (0.25 + 0.64) / 2 = 0.445
    assert ev.rps([[0.5, 0.3, 0.2]], [2]) == pytest.approx(0.445)


def test_rps_is_ordinal():
    """Regression: on a home win, 'all away' must be punished more than 'all draw'."""
    away = ev.rps([[0.0, 0.0, 1.0]], [0])
    draw = ev.rps([[0.0, 1.0, 0.0]], [0])
    assert away == 1.0 and draw == 0.5
    assert away > draw
    # log loss and Brier cannot tell these two apart (both are 'certain and wrong')
    assert ev.brier([[0.0, 0.0, 1.0]], [0]) == ev.brier([[0.0, 1.0, 0.0]], [0])


def test_rps_symmetric_under_home_away_swap():
    p = np.array([[0.6, 0.25, 0.15]])
    assert ev.rps(p, [0]) == pytest.approx(ev.rps(p[:, ::-1], [2]))


# ---------------------------------------------------------------- log loss / Brier vs sklearn
@pytest.fixture
def random_case():
    rng = np.random.default_rng(42)
    p = rng.dirichlet([2, 1.5, 2], size=500)
    y = np.array([rng.choice(3, p=r) for r in p])
    return p, y


def test_log_loss_matches_sklearn(random_case):
    p, y = random_case
    assert ev.log_loss(p, y) == pytest.approx(sk_log_loss(y, p, labels=[0, 1, 2]), abs=1e-12)


def test_brier_matches_sklearn(random_case):
    p, y = random_case
    # multiclass Brier = sum over classes of the binary Brier score for that class
    expected = sum(brier_score_loss((y == k).astype(int), p[:, k]) for k in range(3))
    assert ev.brier(p, y) == pytest.approx(expected, abs=1e-12)


def test_log_loss_hand_computed():
    assert ev.log_loss([[0.5, 0.3, 0.2]], [1]) == pytest.approx(-np.log(0.3))


def test_log_loss_survives_zero_probability():
    assert np.isfinite(ev.log_loss([[0.0, 0.0, 1.0]], [0]))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        ev.rps([[0.5, 0.5, 0.5]], [0])          # does not sum to 1
    with pytest.raises(ValueError):
        ev.rps([[1.0, 0.0, 0.0]], [3])          # bad outcome code
    with pytest.raises(ValueError):
        ev.rps([[np.nan, 0.5, 0.5]], [0])


def test_no_accuracy_metric_exists():
    assert not any("accuracy" in n.lower() for n in dir(ev))


# ---------------------------------------------------------------- de-vig
ODDS = np.array([[2.0, 3.5, 4.0], [1.5, 4.5, 8.0], [3.1, 3.3, 2.4]])


@pytest.mark.parametrize("method", ["proportional", "shin"])
def test_devig_sums_to_one_and_strictly_inside(method):
    p = devig(ODDS, method)
    assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-9
    assert ((p > 0) & (p < 1)).all()


@pytest.mark.parametrize("method", ["proportional", "shin"])
def test_devig_extreme_odds(method):
    o = np.array([[1.01, 15.0, 101.0], [1.001, 500.0, 1000.0], [34.0, 12.0, 1.05]])
    p = devig(o, method)
    assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-9
    assert ((p > 0) & (p < 1)).all()


def test_proportional_hand_computed():
    p = proportional([[2.0, 4.0, 4.0]])  # implied 0.5, 0.25, 0.25; sum 1.0
    assert p[0] == pytest.approx([0.5, 0.25, 0.25])
    p = proportional([[1.8, 3.6, 4.5]])  # implied 0.5556, 0.2778, 0.2222; sum 1.0556
    q = np.array([1 / 1.8, 1 / 3.6, 1 / 4.5])
    assert p[0] == pytest.approx(q / q.sum())


def test_fair_book_is_unchanged():
    fair = np.array([[2.0, 4.0, 4.0]])
    assert overround(fair)[0] == pytest.approx(0.0)
    assert shin(fair)[0] == pytest.approx([0.5, 0.25, 0.25])
    assert proportional(fair)[0] == pytest.approx([0.5, 0.25, 0.25])


def test_shin_and_proportional_agree_on_low_margin_book():
    o = np.array([[2.55, 3.55, 3.05]])  # roughly 0.5% margin
    assert 0 < overround(o)[0] < 0.01
    assert np.abs(shin(o) - proportional(o)).max() < 5e-4


def test_shin_diverges_in_expected_direction_on_high_margin_book():
    """Favourite-longshot: Shin gives the favourite more and the longshot less than
    proportional, and the gap grows with the margin."""
    lo = np.array([[1.30, 5.50, 11.0]])
    hi = np.array([[1.20, 4.50, 8.00]])     # ~20% margin
    assert overround(hi)[0] > 0.15
    gap = lambda o: (shin(o) - proportional(o))[0]  # noqa: E731
    assert gap(hi)[0] > 0 and gap(hi)[2] < 0       # favourite up, longshot down
    assert abs(gap(hi)[0]) > abs(gap(lo)[0])


def test_devig_rejects_bad_odds():
    for bad in ([[1.0, 3.0, 3.0]], [[0.9, 3.0, 3.0]], [[np.nan, 3.0, 3.0]]):
        with pytest.raises(ValueError):
            devig(bad)
    with pytest.raises(ValueError):
        devig(ODDS, "nope")


# ---------------------------------------------------------------- calibration
def test_empty_bins_are_nan_not_zero():
    p = np.array([[0.05, 0.05, 0.9]] * 4)
    t = ev.reliability_table(p, [2, 2, 0, 2])
    home = t[t["outcome"] == "home"]
    assert home["n"].sum() == 4
    empty = home[home["n"] == 0]
    assert len(empty) == 9
    assert empty["mean_pred"].isna().all() and empty["obs_freq"].isna().all()


def test_reliability_hand_computed():
    # away prob 0.9 four times, away happens 3 of 4
    p = np.array([[0.05, 0.05, 0.9]] * 4)
    t = ev.reliability_table(p, [2, 2, 0, 2])
    a = t[(t["outcome"] == "away") & (t["n"] > 0)].iloc[0]
    assert a["bin"] == 9 and a["n"] == 4
    assert a["mean_pred"] == pytest.approx(0.9) and a["obs_freq"] == pytest.approx(0.75)
    assert ev.expected_calibration_error(t)["away"] == pytest.approx(0.15)


def test_probability_of_one_lands_in_last_bin():
    t = ev.reliability_table(np.array([[1.0, 0.0, 0.0]]), [0])
    h = t[t["outcome"] == "home"]
    assert h.loc[h["bin"] == 9, "n"].item() == 1 and h["n"].sum() == 1


def test_perfectly_calibrated_has_zero_ece():
    p = np.array([[0.5, 0.25, 0.25]] * 4)
    t = ev.reliability_table(p, [0, 0, 1, 2])  # home 2/4, draw 1/4, away 1/4
    assert all(v == pytest.approx(0.0) for v in ev.expected_calibration_error(t).values())


# ---------------------------------------------------------------- market as a model
def test_market_model_contract():
    df = pd.DataFrame({"mkt_h": [2.0, 1.5], "mkt_d": [3.5, 4.5], "mkt_a": [4.0, 8.0]})
    for method in ("proportional", "shin"):
        p = MarketModel(method).fit(df).predict_proba(df)
        assert p.shape == (2, 3)
        assert np.abs(p.sum(axis=1) - 1).max() < 1e-9 and ((p > 0) & (p < 1)).all()


def test_market_model_refuses_missing_odds():
    df = pd.DataFrame({"mkt_h": [2.0, np.nan], "mkt_d": [3.5, 4.5], "mkt_a": [4.0, 8.0]})
    with pytest.raises(ValueError):
        MarketModel().predict_proba(df)


def test_market_on_real_closing_subset():
    """Integration: the real closing-odds subset scores in a plausible range. A market RPS
    well below ~0.19 or above ~0.23 would indicate a data or scoring bug."""
    from clean import load_clean
    m = load_clean()
    sub = ev.market_subsets(m)["closing_any"]
    y = ev.encode_outcome(sub["ftr"])
    p = MarketModel("proportional").predict_proba(sub)
    s = ev.score(p, y)
    assert s["n"] == len(sub) > 4000
    assert 0.19 < s["rps"] < 0.23
    assert 0.9 < s["log_loss"] < 1.1
    assert sub["season"].min() == 2012


def test_metrics_upsert_is_deterministic(tmp_path):
    path = tmp_path / "metrics.csv"
    row = {"model": "a", "subset": "s", "n": 10, "rps": 0.2, "log_loss": 1.0, "brier": 0.6, "config_hash": "x"}
    ev.write_metrics([row], path)
    ev.write_metrics([{**row, "rps": 0.3}, {**row, "model": "b"}], path)
    out = pd.read_csv(path)
    assert len(out) == 2 and out.loc[out["model"] == "a", "rps"].item() == pytest.approx(0.3)


# ---------------------------------------------------------------- M7: isotonic recalibration and model-vs-market
def synthetic_predictions(n_seasons=6, per_season=200, seed=0, overconfident=False):
    rng = np.random.default_rng(seed)
    rows = []
    day = pd.Timestamp("2010-08-01")
    for s in range(n_seasons):
        for i in range(per_season):
            true = rng.dirichlet([3, 2, 3])
            y = rng.choice(3, p=true)
            p = true ** 1.6 / (true ** 1.6).sum() if overconfident else true
            rows.append({"season": 2010 + s, "date": day + pd.Timedelta(days=i % 250 + 365 * s), "home": f"h{i}",
                         "away": f"a{i}", "ftr": "HDA"[y], "p_home": p[0], "p_draw": p[1], "p_away": p[2]})
    return pd.DataFrame(rows)


def test_isotonic_fit_matches_sklearn():
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 400)
    y = (rng.uniform(0, 1, 400) < p ** 1.5).astype(float)
    xs, ys = ev.isotonic_fit(p, y)
    assert (np.diff(ys) >= -1e-12).all()                       # monotone non-decreasing
    grid = np.linspace(0.01, 0.99, 50)
    ref = IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(grid)
    np.testing.assert_allclose(ev.isotonic_apply((xs, ys), grid), ref, atol=1e-9)


def test_isotonic_fit_hand_computed():
    xs, ys = ev.isotonic_fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])   # 0.2 and 0.3 violate order -> pooled
    np.testing.assert_allclose(ys, [0.0, 0.5, 0.5, 1.0])


def test_recalibration_rows_sum_to_one_and_skip_first_seasons():
    rc = ev.recalibrate_walk_forward(synthetic_predictions())
    assert rc["season"].min() == 2012                           # two seasons of history required
    assert np.abs(rc[ev.PRED_COLS].sum(axis=1) - 1).max() < 1e-12
    assert ((rc[ev.PRED_COLS] > 0) & (rc[ev.PRED_COLS] < 1)).all().all()


def test_recalibration_uses_only_earlier_seasons():
    """Recalibrated predictions for season s must not change when results of s or later change."""
    preds = synthetic_predictions(seed=3)
    base = ev.recalibrate_walk_forward(preds)
    mutated = preds.copy()
    cut = mutated["season"] >= 2014
    mutated.loc[cut, "ftr"] = "A"
    alt = ev.recalibrate_walk_forward(mutated)
    early_a = base[base["season"] == 2012].reset_index(drop=True)
    early_b = alt[alt["season"] == 2012].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_a, early_b, check_exact=True)
    # seasons 2013 and 2014 are calibrated on earlier seasons only, so they are unaffected too
    for s in (2013, 2014):
        pd.testing.assert_frame_equal(base.loc[base["season"] == s, ev.PRED_COLS].reset_index(drop=True),
                                      alt.loc[alt["season"] == s, ev.PRED_COLS].reset_index(drop=True), check_exact=True)
    # season 2015 learns from the mutated 2014 results, so it must change (the test has teeth)
    assert not np.array_equal(base[base["season"] == 2015][ev.PRED_COLS], alt[alt["season"] == 2015][ev.PRED_COLS])


def test_recalibration_helps_an_overconfident_model_and_not_a_calibrated_one():
    over = ev.recalibrate_walk_forward(synthetic_predictions(n_seasons=8, per_season=600, seed=5, overconfident=True))
    y = ev.encode_outcome(over["ftr"])
    gain, se = ev.paired_gap(over[ev.PRED_COLS].to_numpy(), over[[f"{c}_raw" for c in ev.PRED_COLS]].to_numpy(), y)
    assert gain < -2 * se                                          # recalibrated RPS clearly lower
    good = ev.recalibrate_walk_forward(synthetic_predictions(n_seasons=8, per_season=600, seed=5))
    yg = ev.encode_outcome(good["ftr"])
    gain, se = ev.paired_gap(good[ev.PRED_COLS].to_numpy(), good[[f"{c}_raw" for c in ev.PRED_COLS]].to_numpy(), yg)
    assert gain > -2 * se                                          # no real improvement on a calibrated model


def test_paired_gap_hand_computed():
    pa = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    pb = np.array([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]])
    y = [0, 2]
    gap, se = ev.paired_gap(pa, pb, y)
    d = ev.rps_per_match(pa, y) - ev.rps_per_match(pb, y)
    assert d[1] == 0 and gap == pytest.approx(d.mean()) and gap < 0
    assert se == pytest.approx(d.std(ddof=1) / np.sqrt(2))


def test_closing_common_gives_every_model_identical_matches():
    a = synthetic_predictions(n_seasons=2, per_season=50)
    preds = {"market": a.iloc[10:40], "elo": a, "poisson": a.iloc[5:60]}
    out = ev.closing_common(preds)
    for name in out:
        assert len(out[name]) == 30
        pd.testing.assert_frame_equal(out[name][ev.KEY], out["market"][ev.KEY])


def test_reliability_plot_marks_thin_bins_and_writes_file(tmp_path):
    p = np.tile([0.5, 0.25, 0.25], (40, 1))
    y = np.array([0] * 20 + [1] * 10 + [2] * 10)
    f = tmp_path / "c.png"
    ev.plot_reliability(p, y, "t", f)
    assert f.exists() and f.stat().st_size > 5000
