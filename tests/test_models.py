"""Model tests. Grows with each milestone; M4 adds Elo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as bt  # noqa: E402
from ingest import load_config  # noqa: E402
from models import REGISTRY, TUNING  # noqa: E402
from models.elo import EloModel, goal_diff_multiplier  # noqa: E402
from test_leakage import synthetic_league  # noqa: E402

CFG = load_config()


@pytest.fixture(scope="module")
def league():
    return synthetic_league(n_seasons=5, n_teams=8, seed=3)


def match_frame(rows):
    return pd.DataFrame(rows, columns=["date", "home", "away", "fthg", "ftag"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


# ---------------------------------------------------------------- every model
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_predict_proba_valid(league, name):
    m = REGISTRY[name](CFG)
    train = league[league["season"] < 2003]
    fixtures = bt.make_fixtures(league[league["season"] == 2003], m.requires_odds)
    p = m.fit(train).predict_proba(fixtures)
    assert p.shape == (len(fixtures), 3)
    assert np.abs(p.sum(axis=1) - 1).max() < 1e-9
    assert ((p > 0) & (p < 1)).all()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_predict_proba_valid_with_no_training_data_or_unseen_teams(league, name):
    m = REGISTRY[name](CFG)
    fixtures = bt.make_fixtures(league.head(5), m.requires_odds).assign(home="new_a", away="new_b")
    p = m.fit(league.iloc[0:0]).predict_proba(fixtures)
    assert np.abs(p.sum(axis=1) - 1).max() < 1e-9 and ((p > 0) & (p < 1)).all()


# ---------------------------------------------------------------- Elo
def test_goal_diff_multiplier():
    assert [goal_diff_multiplier(g) for g in (0, 1, 2, 3, 4, 5)] == [1.0, 1.0, 1.5, 1.75, 15 / 8, 2.0]


def test_elo_update_hand_computed():
    m = EloModel(k=20, home_advantage=0)
    m._replay(match_frame([("2020-01-01", "a", "b", 1, 0)]))
    assert m.ratings_["a"] == pytest.approx(1510.0) and m.ratings_["b"] == pytest.approx(1490.0)


def test_elo_goal_difference_adjustment():
    on = EloModel(k=20, home_advantage=0, goal_diff_adjustment=True)
    off = EloModel(k=20, home_advantage=0, goal_diff_adjustment=False)
    frame = match_frame([("2020-01-01", "a", "b", 4, 0)])
    on._replay(frame)
    off._replay(frame)
    assert off.ratings_["a"] == pytest.approx(1510.0)
    assert on.ratings_["a"] == pytest.approx(1500 + 10 * 15 / 8)   # margin 4 -> (11+4)/8


def test_elo_home_advantage_reduces_reward_for_expected_home_win():
    with_ha = EloModel(k=20, home_advantage=100)
    with_ha._replay(match_frame([("2020-01-01", "a", "b", 1, 0)]))
    e = 1 / (1 + 10 ** (-100 / 400))
    assert with_ha.ratings_["a"] == pytest.approx(1500 + 20 * (1 - e))


def test_elo_is_zero_sum(league):
    m = EloModel(k=25)
    m._replay(league)
    assert sum(m.ratings_.values()) == pytest.approx(1500 * len(m.ratings_))


def test_elo_ratings_used_for_week_k_ignore_week_k_and_later(league):
    """Ratings at fit time depend only on the training rows given to fit()."""
    start = league.loc[league["season"] == 2003, "date"].sort_values().iloc[30]
    train = league[league["date"] < start]
    a = EloModel(k=20).fit(train)
    mutated = league.copy()
    fut = mutated["date"] >= start
    mutated.loc[fut, ["fthg", "ftag"]] = [5, 0]
    b = EloModel(k=20).fit(mutated[mutated["date"] < start])
    assert a.ratings_ == b.ratings_
    fx = bt.make_fixtures(league[league["date"] >= start].head(10), False)
    np.testing.assert_array_equal(a.predict_proba(fx), b.predict_proba(fx))


def test_elo_recovers_ordering_of_a_strictly_stronger_team():
    rng = np.random.default_rng(0)
    rows = []
    teams = ["strong", "mid1", "mid2", "weak"]
    skill = {"strong": 0.6, "mid1": 0.0, "mid2": 0.0, "weak": -0.6}
    day = pd.Timestamp("2015-08-01")
    for r in range(60):
        for h in teams:
            for a in teams:
                if h != a:
                    hg = rng.poisson(np.exp(0.2 + skill[h] - skill[a]))
                    ag = rng.poisson(np.exp(skill[a] - skill[h]))
                    rows.append((day, h, a, hg, ag))
                    day += pd.Timedelta(days=1)
    m = EloModel(k=10).fit(match_frame(rows).assign(ftr=lambda d: np.where(d.fthg > d.ftag, "H", np.where(d.fthg < d.ftag, "A", "D"))))
    r = m.ratings_
    assert r["strong"] > r["mid1"] and r["strong"] > r["mid2"] > r["weak"] and r["mid1"] > r["weak"]
    fx = match_frame([("2020-01-01", "strong", "weak", 0, 0), ("2020-01-01", "weak", "strong", 0, 0)])
    p = m.predict_proba(fx)
    assert p[0, 0] > p[0, 2] and p[1, 2] > p[1, 0]         # stronger side favoured at either venue
    assert p[0, 0] > p[1, 0]                                 # and better at home than away


def test_elo_draw_probability_peaks_for_evenly_matched_teams():
    rng = np.random.default_rng(1)
    rows, day = [], pd.Timestamp("2015-08-01")
    teams = [f"t{i}" for i in range(6)]
    skill = dict(zip(teams, rng.normal(0, 0.4, 6)))
    for _ in range(40):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append((day, h, a, rng.poisson(np.exp(0.2 + skill[h] - skill[a])),
                                 rng.poisson(np.exp(skill[a] - skill[h]))))
                    day += pd.Timedelta(days=1)
    m = EloModel(k=10).fit(match_frame(rows).assign(ftr="D"))
    even = m._probs(m.params_, np.array([0.0]))[0, 1]
    lopsided = m._probs(m.params_, np.array([400.0]))[0, 1]
    assert even > lopsided


def test_elo_beats_base_rate_on_real_data():
    """M4 exit criterion, on the cached backtest predictions written by `make backtest`."""
    import evaluate as ev
    root = Path(__file__).resolve().parent.parent
    m = pd.read_csv(root / "reports" / "metrics.csv")
    get = lambda model: m[(m["model"] == model) & (m["subset"] == "all_scored")]["rps"].item()  # noqa: E731
    assert get("elo") < get("baserate")
    assert (root / "reports" / "figures" / "elo_k_tuning.png").exists()


def test_tuning_selection_uses_only_earlier_seasons(league):
    """The K chosen for season s must not change if season s (or later) results change."""
    grid = [10, 20, 30]
    make = lambda v: EloModel(k=v)  # noqa: E731
    _, _, chosen_a = bt.walk_forward_tuned(league, make, grid, default=20, burn_in_seasons=2)
    mutated = league.copy()
    last = mutated["season"] == mutated["season"].max()
    mutated.loc[last, ["fthg", "ftag"]] = [7, 0]
    mutated["ftr"] = np.where(mutated.fthg > mutated.ftag, "H", np.where(mutated.fthg < mutated.ftag, "A", "D"))
    _, _, chosen_b = bt.walk_forward_tuned(mutated, make, grid, default=20, burn_in_seasons=2)
    # the final season's own results can't influence its own choice (nor earlier ones)
    pd.testing.assert_frame_equal(chosen_a, chosen_b)
    assert chosen_a.loc[0, "chosen"] == 20            # first scored season falls back to the default


def test_tuning_registry_entries_are_complete():
    for name, t in TUNING.items():
        assert name in REGISTRY and len(t["grid"](CFG)) >= 3 and t["default"](CFG) in t["grid"](CFG)


# ---------------------------------------------------------------- reducers (M5)
from scipy.optimize import check_grad  # noqa: E402

from models.poisson import PoissonModel, scoreline_matrix, sum_zero_map  # noqa: E402
from models.reducers import to_1x2, to_asian_handicap, to_btts, to_over_under  # noqa: E402


@pytest.fixture(scope="module")
def S():
    rng = np.random.default_rng(11)
    return rng.dirichlet(np.ones(64) * 0.4, size=25).reshape(25, 8, 8)


def test_over_under_matches_brute_force(S):
    for line in (0.5, 1.5, 2.5, 3.5):
        brute = np.array([sum(s[i, j] for i in range(8) for j in range(8) if i + j > line) for s in S])
        out = to_over_under(S, line)
        np.testing.assert_allclose(out[:, 0], brute, atol=1e-12)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_1x2_and_btts_match_brute_force(S):
    for k, s in enumerate(S):
        h = sum(s[i, j] for i in range(8) for j in range(8) if i > j)
        d = sum(s[i, j] for i in range(8) for j in range(8) if i == j)
        assert to_1x2(S)[k] == pytest.approx([h, d, 1 - h - d], abs=1e-12)
        yes = sum(s[i, j] for i in range(1, 8) for j in range(1, 8))
        assert to_btts(S)[k, 0] == pytest.approx(yes, abs=1e-12)


def test_asian_handicap_lines(S):
    for k, s in enumerate(S[:5]):
        def cell(cond):
            return sum(s[i, j] for i in range(8) for j in range(8) if cond(i - j))
        # half line -0.5: home must win
        assert to_asian_handicap(S, -0.5)[k] == pytest.approx([cell(lambda d: d > 0), 0, cell(lambda d: d <= 0)])
        # whole line 0 (draw no bet): the draw is a push
        assert to_asian_handicap(S, 0.0)[k] == pytest.approx(
            [cell(lambda d: d > 0), cell(lambda d: d == 0), cell(lambda d: d < 0)])
        # whole line -1: a one-goal win is a push
        assert to_asian_handicap(S, -1.0)[k] == pytest.approx(
            [cell(lambda d: d > 1), cell(lambda d: d == 1), cell(lambda d: d < 1)])
        # quarter line -0.25 = half the stake on 0 and half on -0.5
        expect = 0.5 * to_asian_handicap(S, 0.0)[k] + 0.5 * to_asian_handicap(S, -0.5)[k]
        assert to_asian_handicap(S, -0.25)[k] == pytest.approx(expect)
    assert np.allclose(to_asian_handicap(S, -0.75).sum(axis=1), 1.0)


def test_reducers_reject_bad_lines(S):
    for bad in (2.0, 2.25):
        with pytest.raises(ValueError):
            to_over_under(S, bad)
    with pytest.raises(ValueError):
        to_asian_handicap(S, 0.1)


# ---------------------------------------------------------------- Poisson (M5)
def simulate(n_seasons=30, n_teams=8, seed=5, mu=0.1, h=0.3):
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    att -= att.mean()
    dfn = rng.normal(0, 0.25, n_teams)
    dfn -= dfn.mean()
    rows, day = [], pd.Timestamp("2000-08-01")
    for _ in range(n_seasons):
        for i in range(n_teams):
            for j in range(n_teams):
                if i != j:
                    rows.append((day, teams[i], teams[j], rng.poisson(np.exp(mu + h + att[i] - dfn[j])),
                                 rng.poisson(np.exp(mu + att[j] - dfn[i]))))
                    day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows, columns=["date", "home", "away", "fthg", "ftag"])
    return df, dict(zip(teams, att)), dict(zip(teams, dfn)), mu, h


def test_poisson_identifiability_constraint():
    df, *_ = simulate(n_seasons=5)
    m = PoissonModel(lookback_years=None).fit(df)
    assert sum(m.attack_.values()) == pytest.approx(0.0, abs=1e-9)
    assert sum(m.defence_.values()) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(sum_zero_map(6).sum(axis=0), 0)


def test_poisson_recovers_true_parameters():
    df, att, dfn, mu, h = simulate(n_seasons=60)
    m = PoissonModel(lookback_years=None).fit(df)
    assert m.mu_ == pytest.approx(mu, abs=0.05) and m.home_adv_ == pytest.approx(h, abs=0.05)
    for t in att:
        assert m.attack_[t] == pytest.approx(att[t], abs=0.08)
        assert m.defence_[t] == pytest.approx(dfn[t], abs=0.08)


def test_poisson_symmetric_league_gives_zero_team_effects():
    teams = ["a", "b", "c", "d"]
    pairs = [(h, a) for h in teams for a in teams if h != a]
    rows = [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=k), h, a, 2, 1) for k, (h, a) in enumerate(pairs)]
    m = PoissonModel(lookback_years=None).fit(pd.DataFrame(rows, columns=["date", "home", "away", "fthg", "ftag"]))
    assert all(abs(v) < 1e-6 for v in list(m.attack_.values()) + list(m.defence_.values()))
    assert np.exp(m.mu_ + m.home_adv_) == pytest.approx(2.0, rel=1e-5)
    assert np.exp(m.mu_) == pytest.approx(1.0, rel=1e-5)


def test_poisson_gradient_is_correct():
    df, *_ = simulate(n_seasons=2, n_teams=5)
    teams = sorted(set(df.home))
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai = df.home.map(idx).to_numpy(), df.away.map(idx).to_numpy()
    x, y, w = df.fthg.to_numpy(float), df.ftag.to_numpy(float), np.linspace(0.5, 1.5, len(df))
    M = sum_zero_map(5)
    f = lambda th: PoissonModel.objective(th, M, hi, ai, x, y, w)[0]  # noqa: E731
    g = lambda th: PoissonModel.objective(th, M, hi, ai, x, y, w)[1]  # noqa: E731
    th = np.random.default_rng(0).normal(0, 0.2, 2 + 8)
    assert check_grad(f, g, th) < 1e-6


def test_poisson_fit_is_a_likelihood_maximum():
    df, *_ = simulate(n_seasons=8)
    m = PoissonModel(lookback_years=None).fit(df)
    teams = sorted(m.attack_)
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai = df.home.map(idx).to_numpy(), df.away.map(idx).to_numpy()
    x, y = df.fthg.to_numpy(float), df.ftag.to_numpy(float)
    A, D = np.array([m.attack_[t] for t in teams]), np.array([m.defence_[t] for t in teams])
    best = PoissonModel.loglik(m.mu_, m.home_adv_, A, D, hi, ai, x, y)
    rng = np.random.default_rng(0)
    for _ in range(20):
        dA = rng.normal(0, 0.05, len(teams))
        dA -= dA.mean()
        worse = PoissonModel.loglik(m.mu_ + rng.normal(0, 0.02), m.home_adv_ + rng.normal(0, 0.02),
                                    A + dA, D, hi, ai, x, y)
        assert worse < best


def test_poisson_orders_a_strictly_stronger_team():
    rng = np.random.default_rng(2)
    teams, skill = ["strong", "mid1", "mid2", "weak"], {"strong": .6, "mid1": 0, "mid2": 0, "weak": -.6}
    rows, day = [], pd.Timestamp("2015-08-01")
    for _ in range(40):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append((day, h, a, rng.poisson(np.exp(0.2 + skill[h] - skill[a])),
                                 rng.poisson(np.exp(skill[a] - skill[h]))))
                    day += pd.Timedelta(days=1)
    m = PoissonModel(lookback_years=None).fit(pd.DataFrame(rows, columns=["date", "home", "away", "fthg", "ftag"]))
    s = {t: m.attack_[t] + m.defence_[t] for t in teams}
    assert s["strong"] > s["mid1"] > s["weak"] and s["strong"] > s["mid2"] > s["weak"]
    p = m.predict_proba(pd.DataFrame({"home": ["strong", "weak"], "away": ["weak", "strong"]}))
    assert p[0, 0] > p[0, 2] and p[1, 2] > p[1, 0]


def test_poisson_scoreline_is_a_distribution_and_proba_derives_from_it():
    df, *_ = simulate(n_seasons=4)
    m = PoissonModel(lookback_years=None).fit(df)
    fx = df.head(12)[["home", "away"]]
    S = m.predict_scoreline(fx)
    assert S.shape == (12, 8, 8) and (S >= 0).all()
    np.testing.assert_allclose(S.sum(axis=(1, 2)), 1.0, atol=1e-12)
    np.testing.assert_array_equal(m.predict_proba(fx), to_1x2(S))       # by construction, not just close
    brute = [sum(s[i, j] for i in range(8) for j in range(8) if i + j >= 3) for s in S]
    np.testing.assert_allclose(to_over_under(S, 2.5)[:, 0], brute, atol=1e-12)


def test_scoreline_matrix_matches_scipy_pmf_before_truncation():
    from scipy.stats import poisson
    S = scoreline_matrix(np.array([1.4]), np.array([1.1]), max_goals=8)
    raw = np.outer(poisson.pmf(range(8), 1.4), poisson.pmf(range(8), 1.1))
    np.testing.assert_allclose(S[0], raw / raw.sum(), atol=1e-14)
    assert raw.sum() > 0.999


def test_poisson_lookback_window_and_unseen_teams():
    df, *_ = simulate(n_seasons=6)
    old = PoissonModel(lookback_years=None).fit(df)
    short = PoissonModel(lookback_years=0.5).fit(df)
    assert set(old.attack_) == set(df.home) and short.mu_ != old.mu_
    p = old.predict_proba(pd.DataFrame({"home": ["zzz"], "away": ["yyy"]}))     # two unseen clubs
    assert np.allclose(p.sum(), 1.0) and p[0, 0] > p[0, 2]                     # home advantage only


# ---------------------------------------------------------------- Dixon-Coles (M6)
from models.dixon_coles import DixonColesModel, tau, time_weights  # noqa: E402


def as_arrays(df, model):
    teams = sorted(model.attack_)
    idx = {t: i for i, t in enumerate(teams)}
    return (teams, df.home.map(idx).to_numpy(), df.away.map(idx).to_numpy(),
            df.fthg.to_numpy(float), df.ftag.to_numpy(float))


def test_tau_hand_computed():
    lh, la, rho = 1.2, 0.8, -0.1
    assert tau(0, 0, lh, la, rho) == pytest.approx(1 - 1.2 * 0.8 * -0.1)
    assert tau(0, 1, lh, la, rho) == pytest.approx(1 + 1.2 * -0.1)
    assert tau(1, 0, lh, la, rho) == pytest.approx(1 + 0.8 * -0.1)
    assert tau(1, 1, lh, la, rho) == pytest.approx(1 + 0.1)
    for x, y in [(2, 0), (0, 2), (2, 1), (1, 2), (3, 3), (0, 5)]:
        assert tau(x, y, lh, la, rho) == 1.0


def test_dc_gradient_is_correct():
    df, *_ = simulate(n_seasons=2, n_teams=5)
    teams = sorted(set(df.home))
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai = df.home.map(idx).to_numpy(), df.away.map(idx).to_numpy()
    x, y, w = df.fthg.to_numpy(float), df.ftag.to_numpy(float), np.linspace(0.3, 1.0, len(df))
    M = sum_zero_map(5)
    f = lambda th: DixonColesModel.objective(th, M, hi, ai, x, y, w)[0]  # noqa: E731
    g = lambda th: DixonColesModel.objective(th, M, hi, ai, x, y, w)[1]  # noqa: E731
    th = np.random.default_rng(0).normal(0, 0.2, 3 + 8)
    th[-1] = -0.08
    assert check_grad(f, g, th) < 1e-6


def test_dc_with_xi_zero_weights_every_match_equally():
    df, *_ = simulate(n_seasons=3)
    assert np.all(time_weights(df["date"], 0.0) == 1.0)
    a = DixonColesModel(xi=0.0, lookback_years=None).fit(df)
    teams, hi, ai, x, y = as_arrays(df, a)
    M = sum_zero_map(len(teams))
    theta = a._fit_arrays(hi, ai, x, y, len(teams), np.ones(len(x)))      # explicit equal weights
    assert a.rho_ == pytest.approx(theta[-1], abs=1e-9) and a.home_adv_ == pytest.approx(theta[1], abs=1e-9)
    # the tau term is genuinely active and helps the fit (nested model)
    A, D = np.array([a.attack_[t] for t in teams]), np.array([a.defence_[t] for t in teams])
    with_tau = DixonColesModel.full_loglik(a.mu_, a.home_adv_, a.rho_, A, D, hi, ai, x, y)
    without = DixonColesModel.full_loglik(a.mu_, a.home_adv_, 0.0, A, D, hi, ai, x, y)
    assert with_tau >= without


def test_dc_without_tau_reduces_to_independent_poisson():
    df, *_ = simulate(n_seasons=6)
    dc = DixonColesModel(xi=0.0, lookback_years=None, use_tau=False).fit(df)
    po = PoissonModel(lookback_years=None).fit(df)
    assert dc.rho_ == 0.0
    assert dc.mu_ == pytest.approx(po.mu_, abs=1e-5) and dc.home_adv_ == pytest.approx(po.home_adv_, abs=1e-5)
    for t in po.attack_:
        assert dc.attack_[t] == pytest.approx(po.attack_[t], abs=1e-5)
        assert dc.defence_[t] == pytest.approx(po.defence_[t], abs=1e-5)
    fx = df.head(10)[["home", "away"]]
    np.testing.assert_allclose(dc.predict_scoreline(fx), po.predict_scoreline(fx), atol=1e-6)


def test_dc_low_score_correction_direction():
    """rho < 0 (the empirical case) inflates 0-0 and 1-1 and deflates 1-0 and 0-1."""
    from models.dixon_coles import scoreline_matrix_dc
    lam_h, lam_a = np.array([1.4]), np.array([1.1])
    base = scoreline_matrix(lam_h, lam_a)[0]
    dc = scoreline_matrix_dc(lam_h, lam_a, -0.1)[0]
    assert dc[0, 0] > base[0, 0] and dc[1, 1] > base[1, 1]
    assert dc[1, 0] < base[1, 0] and dc[0, 1] < base[0, 1]
    assert dc.sum() == pytest.approx(1.0)


def test_dc_scoreline_and_proba_consistency():
    df, *_ = simulate(n_seasons=4)
    m = DixonColesModel(xi=0.001, lookback_years=None).fit(df)
    fx = df.head(12)[["home", "away"]]
    S = m.predict_scoreline(fx)
    assert S.shape == (12, 8, 8) and (S >= 0).all()
    np.testing.assert_allclose(S.sum(axis=(1, 2)), 1.0, atol=1e-12)
    np.testing.assert_array_equal(m.predict_proba(fx), to_1x2(S))
    brute = [sum(s[i, j] for i in range(8) for j in range(8) if i + j >= 3) for s in S]
    np.testing.assert_allclose(to_over_under(S, 2.5)[:, 0], brute, atol=1e-12)


def test_dc_time_decay_favours_recent_form():
    """A club that was strong long ago and weak lately is rated lower when xi is large."""
    rng = np.random.default_rng(4)
    rows, day = [], pd.Timestamp("2010-08-01")
    teams = ["x", "y", "z", "w"]
    for era in range(2):
        for _ in range(25):
            for h in teams:
                for a in teams:
                    if h != a:
                        sx = 0.8 if era == 0 else -0.8
                        sk = {"x": sx, "y": 0, "z": 0, "w": 0}
                        rows.append((day, h, a, rng.poisson(np.exp(0.2 + sk[h] - sk[a])),
                                     rng.poisson(np.exp(sk[a] - sk[h]))))
                        day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows, columns=["date", "home", "away", "fthg", "ftag"])
    flat = DixonColesModel(xi=0.0, lookback_years=None).fit(df)
    decayed = DixonColesModel(xi=0.005, lookback_years=None).fit(df)
    s = lambda m: m.attack_["x"] + m.defence_["x"]  # noqa: E731
    assert s(decayed) < s(flat) - 0.5


def test_dc_orders_a_strictly_stronger_team():
    rng = np.random.default_rng(2)
    teams, skill = ["strong", "mid1", "mid2", "weak"], {"strong": .6, "mid1": 0, "mid2": 0, "weak": -.6}
    rows, day = [], pd.Timestamp("2015-08-01")
    for _ in range(40):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append((day, h, a, rng.poisson(np.exp(0.2 + skill[h] - skill[a])),
                                 rng.poisson(np.exp(skill[a] - skill[h]))))
                    day += pd.Timedelta(days=1)
    m = DixonColesModel(xi=0.0005, lookback_years=None).fit(
        pd.DataFrame(rows, columns=["date", "home", "away", "fthg", "ftag"]))
    s = {t: m.attack_[t] + m.defence_[t] for t in teams}
    assert s["strong"] > s["mid1"] > s["weak"] and s["strong"] > s["mid2"] > s["weak"]
    assert sum(m.attack_.values()) == pytest.approx(0, abs=1e-9)
    assert sum(m.defence_.values()) == pytest.approx(0, abs=1e-9)


# ---- cross-check against penaltyblog (a test-only reference implementation, never imported by src/)
try:
    import penaltyblog as pb
except ImportError:          # test-only reference; the rest of the suite must run without it
    pb = None
needs_pb = pytest.mark.skipif(pb is None, reason="penaltyblog not installed (test-only dependency)")


@pytest.fixture(scope="module")
def pb_case():
    if pb is None:
        pytest.skip("penaltyblog not installed (test-only dependency)")
    df, *_ = simulate(n_seasons=6, n_teams=8, seed=21)
    xi = 0.0018
    w = pb.models.dixon_coles_weights(df["date"], xi)
    ref = pb.models.DixonColesGoalModel(df.fthg.to_numpy(), df.ftag.to_numpy(),
                                        df.home.to_numpy(), df.away.to_numpy(), weights=w)
    ref.fit()
    return df, xi, w, ref


def pb_to_ours(ref):
    """penaltyblog: lam_h = exp(hfa + att_h + def_a); ours: exp(mu + h + a_h - d_a)."""
    P = ref.get_params()
    teams = list(ref.teams)
    Apb = np.array([P[f"attack_{t}"] for t in teams])
    Dpb = np.array([P[f"defence_{t}"] for t in teams])
    return (teams, Apb.mean() + Dpb.mean(), P["home_advantage"], P["rho"], Apb - Apb.mean(), -(Dpb - Dpb.mean()))


@needs_pb
def test_time_weights_match_penaltyblog(pb_case):
    df, xi, w, _ = pb_case
    np.testing.assert_allclose(time_weights(df["date"], xi), w, rtol=1e-12)


@needs_pb
def test_dc_loglikelihood_matches_penaltyblog(pb_case):
    """Evaluate OUR log-likelihood at penaltyblog's fitted parameters: it must equal the value
    penaltyblog reports for the same data and weights."""
    df, xi, w, ref = pb_case
    teams, mu, h, rho, A, D = pb_to_ours(ref)
    idx = {t: i for i, t in enumerate(teams)}
    ours = DixonColesModel.full_loglik(mu, h, rho, A, D, df.home.map(idx).to_numpy(), df.away.map(idx).to_numpy(),
                                       df.fthg.to_numpy(float), df.ftag.to_numpy(float), w)
    assert ours == pytest.approx(ref.loglikelihood, abs=1e-6)


@needs_pb
def test_dc_fit_is_at_least_as_good_as_penaltyblog(pb_case):
    df, xi, w, ref = pb_case
    m = DixonColesModel(xi=xi, lookback_years=None).fit(df)
    teams, hi, ai, x, y = as_arrays(df, m)
    A, D = np.array([m.attack_[t] for t in teams]), np.array([m.defence_[t] for t in teams])
    ours = DixonColesModel.full_loglik(m.mu_, m.home_adv_, m.rho_, A, D, hi, ai, x, y, w)
    assert ours >= ref.loglikelihood - 1e-6
    assert ours == pytest.approx(ref.loglikelihood, abs=0.05)     # same optimum up to optimiser tolerance


@needs_pb
def test_dc_scoreline_matches_penaltyblog_at_identical_parameters(pb_case):
    df, xi, w, ref = pb_case
    teams, mu, h, rho, A, D = pb_to_ours(ref)
    m = DixonColesModel(xi=xi)
    m.mu_, m.home_adv_, m.rho_ = mu, h, rho
    m.attack_, m.defence_ = dict(zip(teams, A)), dict(zip(teams, D))
    fx = pd.DataFrame({"home": teams[:4], "away": teams[4:8]})
    S = m.predict_scoreline(fx)
    for k, (ht, at) in enumerate(zip(fx.home, fx.away)):
        g = ref.predict(ht, at, max_goals=8)
        np.testing.assert_allclose(S[k], np.asarray(g.grid), atol=1e-12)
        np.testing.assert_allclose(m.predict_proba(fx)[k], g.home_draw_away, atol=1e-12)
        assert to_over_under(S, 2.5)[k, 0] == pytest.approx(g.total_goals("over", 2.5), abs=1e-12)


@needs_pb
def test_rps_matches_penaltyblog():
    import evaluate as ev
    rng = np.random.default_rng(3)
    p = rng.dirichlet([2, 1.5, 2], size=200)
    y = rng.integers(0, 3, 200)
    ref = pb.metrics.rps_average(p.tolist(), y.tolist())
    assert ev.rps(p, y) == pytest.approx(ref, abs=1e-12)


# ---- end-to-end results (need `python src/backtest.py --model dixon_coles` to have been run)
def test_dc_backtest_outputs_and_ranking():
    root = Path(__file__).resolve().parent.parent
    m = pd.read_csv(root / "reports" / "metrics.csv")
    get = lambda model: m[(m["model"] == model) & (m["subset"] == "all_scored")]["rps"].item()  # noqa: E731
    assert get("dixon_coles") < get("baserate")
    assert (root / "reports" / "figures" / "dc_xi_tuning.png").exists()
    curve = pd.read_csv(root / "reports" / "tuning_dixon_coles.csv")
    assert list(curve["xi"]) == CFG["models"]["dixon_coles"]["xi_grid"]
    # the data is live (the current season grows), so compare with the data rather than a constant
    from clean import load_clean
    n_scored = (load_clean()["season"] >= load_clean()["season"].min() + CFG["backtest"]["burn_in_seasons"]).sum()
    assert (curve["n"] == n_scored).all()
