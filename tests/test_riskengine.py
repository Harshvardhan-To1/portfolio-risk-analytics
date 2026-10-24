"""Unit tests: each model is checked against a closed-form answer or a known property."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riskengine import attribution, metrics, optimize, stress  # noqa: E402
from riskengine import var as varlib  # noqa: E402


@pytest.fixture(scope="module")
def normal_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0004, 0.01, 200_000))


@pytest.fixture(scope="module")
def asset_returns():
    rng = np.random.default_rng(1)
    cov = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.1], [0.2, 0.1, 1.0]]) * 1e-4
    x = rng.multivariate_normal([3e-4, 2e-4, 1e-4], cov, 3000)
    idx = pd.bdate_range("2015-01-01", periods=3000)
    return pd.DataFrame(x, index=idx, columns=["A", "B", "C"])


# ------------------------------------------------------------------ metrics
def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2020-01-01", periods=3))
    assert metrics.max_drawdown_details(r)["max_drawdown"] == pytest.approx(-0.50)


def test_annualized_return_constant():
    r = pd.Series([0.0003] * 252 * 4)
    assert metrics.annualized_return(r) == pytest.approx(1.0003 ** 252 - 1)


def test_beta_of_series_with_itself_is_one(asset_returns):
    assert metrics.beta(asset_returns["A"], asset_returns["A"]) == pytest.approx(1.0)


def test_constant_weight_portfolio_matches_dot_product(asset_returns):
    w = {"A": 0.5, "B": 0.3, "C": 0.2}
    p = metrics.portfolio_returns(asset_returns, w, rebalance=None)
    np.testing.assert_allclose(p.values, asset_returns.values @ np.array([0.5, 0.3, 0.2]))


def test_drifting_weights_compound_like_buy_and_hold(asset_returns):
    # Within one rebalance period the portfolio must equal buy-and-hold of the assets.
    r = asset_returns.iloc[:20]
    r.index = pd.bdate_range("2021-03-01", periods=20)  # all inside March
    w = np.array([0.5, 0.3, 0.2])
    p = metrics.portfolio_returns(r, dict(zip(r.columns, w)), rebalance="M")
    bh = (w * (1 + r).prod()).sum() - 1
    assert (1 + p).prod() - 1 == pytest.approx(bh)


# ------------------------------------------------------------------ VaR
def test_historical_and_parametric_var_agree_on_normal_data(normal_returns):
    exact = -(0.0004 + stats.norm.ppf(0.01) * 0.01)
    assert varlib.historical_var(normal_returns, 0.99) == pytest.approx(exact, rel=0.02)
    assert varlib.parametric_var(normal_returns, 0.99) == pytest.approx(exact, rel=0.01)
    # Cornish-Fisher collapses to normal VaR when skew = kurtosis = 0
    assert varlib.cornish_fisher_var(normal_returns, 0.99) == pytest.approx(exact, rel=0.02)


def test_expected_shortfall_exceeds_var(normal_returns):
    assert varlib.historical_es(normal_returns, 0.99) > varlib.historical_var(normal_returns, 0.99)
    assert varlib.parametric_es(normal_returns, 0.99) > varlib.parametric_var(normal_returns, 0.99)


def test_monte_carlo_t_has_fatter_tail_than_normal(asset_returns):
    w = np.array([0.4, 0.4, 0.2])
    mc = varlib.monte_carlo_var(asset_returns, w, conf=0.999, n_sims=200_000)
    p = asset_returns @ w
    assert mc["var"] > varlib.parametric_var(p, 0.999)


def test_kupiec_accepts_correct_rate_and_rejects_wrong_rate():
    assert varlib.kupiec_pof(10, 1000, 0.99)[1] > 0.5     # exactly the expected 1%
    assert varlib.kupiec_pof(40, 1000, 0.99)[1] < 0.001   # 4x too many breaches


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=bool); clustered[500:510] = True
    spread = np.zeros(1000, dtype=bool); spread[::100] = True
    assert varlib.christoffersen_independence(clustered)[1] < 0.01
    assert varlib.christoffersen_independence(spread)[1] > 0.05


def test_rolling_var_has_no_lookahead(asset_returns):
    r = asset_returns["A"].copy()
    fc1 = varlib.rolling_var_forecasts(r, 0.99, 250)
    r2 = r.copy(); r2.iloc[-1] = -0.5  # a crash on the final day
    fc2 = varlib.rolling_var_forecasts(r2, 0.99, 250)
    assert fc1["Historical"].iloc[-1] == fc2["Historical"].iloc[-1]


# ------------------------------------------------------------------ optimization
def test_ledoit_wolf_is_valid_covariance(asset_returns):
    cov, delta = optimize.ledoit_wolf(asset_returns)
    assert 0 <= delta <= 1
    np.testing.assert_allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_min_variance_beats_equal_weight(asset_returns):
    cov = asset_returns.cov().values
    w = optimize.min_variance(cov, max_w=1.0)
    ew = np.full(3, 1 / 3)
    assert w.sum() == pytest.approx(1)
    assert w @ cov @ w <= ew @ cov @ ew + 1e-12


def test_risk_parity_equalizes_risk_contributions(asset_returns):
    cov = asset_returns.cov().values
    w = optimize.risk_parity(cov)
    rc = w * (cov @ w)
    np.testing.assert_allclose(rc / rc.sum(), np.full(3, 1 / 3), atol=1e-4)


def test_weight_cap_respected(asset_returns):
    mu, cov = asset_returns.mean().values * 252, asset_returns.cov().values * 252
    assert optimize.max_sharpe(mu, cov, max_w=0.4).max() <= 0.4 + 1e-8


def test_walk_forward_uses_only_past_data(asset_returns):
    res = optimize.walk_forward(asset_returns, ["Min Variance"], lookback=500)
    first_rebal = res["Min Variance"]["weights"].index[0]
    assert asset_returns.index.get_loc(first_rebal) >= 500 - 1
    assert res["Min Variance"]["returns"].index[0] > first_rebal


# ------------------------------------------------------------------ attribution & stress
def test_risk_contributions_sum_to_portfolio_vol(asset_returns):
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    cov = asset_returns.cov()
    rc = attribution.risk_contributions(w, cov)
    port_vol = np.sqrt(w.values @ cov.values @ w.values * 252)
    assert rc["Contribution to Vol"].sum() == pytest.approx(port_vol)
    assert rc["% of Risk"].sum() == pytest.approx(1.0)


def test_component_var_sums_to_parametric_var(asset_returns):
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    cvar = attribution.component_var(w, asset_returns, 0.99)
    p = asset_returns @ w
    expected = -stats.norm.ppf(0.01) * p.std()
    assert cvar.sum() == pytest.approx(expected, rel=1e-6)


def test_conditional_shock_reproduces_driver_and_beta(asset_returns):
    moves = stress.conditional_shock(asset_returns, {"A": -0.10}, "full", horizon_days=1)
    assert moves["A"] == pytest.approx(-0.10)
    assert moves["B"] < 0  # positively correlated asset falls too
    assert abs(moves["C"]) < abs(moves["B"])  # weaker correlation -> smaller move
