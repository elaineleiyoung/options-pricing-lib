import numpy as np
import pytest

from pricing import black_scholes
from implied_vol import implied_vol


# --- The deliverable: recover a known sigma from its own BS price ----------
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("true_sigma", [0.10, 0.25, 0.60, 1.20])
def test_round_trip(option_type, true_sigma):
    """price(sigma) -> implied_vol -> recovers sigma across a wide vol range."""
    S, K, T, r = 100.0, 105.0, 1.0, 0.05
    price = black_scholes(S, K, T, r, true_sigma, option_type=option_type)
    recovered = implied_vol(price, S, K, T, r, option_type=option_type)
    assert recovered == pytest.approx(true_sigma, abs=1e-4)


# --- Round-trip across strikes too (ITM, ATM, OTM), not just one strike ----
@pytest.mark.parametrize("K", [70.0, 90.0, 100.0, 110.0, 140.0])
def test_round_trip_across_strikes(K):
    """The solver must hold up across the whole smile, not just ATM."""
    S, T, r, true_sigma = 100.0, 0.5, 0.03, 0.35
    price = black_scholes(S, K, T, r, true_sigma, option_type="call")
    assert implied_vol(price, S, K, T, r, option_type="call") == pytest.approx(
        true_sigma, abs=1e-4
    )


# --- Fallback: deep ITM/OTM, vega ~ 0, NR alone would stall ----------------
def test_deep_itm_low_vega():
    S, K, T, r, true_sigma = 200.0, 50.0, 1.0, 0.05, 0.30
    price = black_scholes(S, K, T, r, true_sigma, option_type="call")
    assert implied_vol(price, S, K, T, r, option_type="call") == pytest.approx(
        true_sigma, abs=1e-3
    )


def test_deep_otm_low_vega():
    S, K, T, r, true_sigma = 50.0, 200.0, 1.0, 0.05, 0.30
    price = black_scholes(S, K, T, r, true_sigma, option_type="call")
    assert implied_vol(price, S, K, T, r, option_type="call") == pytest.approx(
        true_sigma, abs=1e-3
    )


# --- Puts recover too (same vega, but confirm the sign paths) --------------
def test_put_round_trip_deep_itm():
    S, K, T, r, true_sigma = 100.0, 160.0, 1.0, 0.05, 0.40
    price = black_scholes(S, K, T, r, true_sigma, option_type="put")
    assert implied_vol(price, S, K, T, r, option_type="put") == pytest.approx(
        true_sigma, abs=1e-3
    )


# --- Bad data returns nan, never a bogus number ----------------------------
def test_arbitrage_violations_return_nan():
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    assert np.isnan(implied_vol(0.0, S, K, T, r))       # zero price
    assert np.isnan(implied_vol(-5.0, S, K, T, r))      # negative price
    assert np.isnan(implied_vol(S + 1, S, K, T, r))     # call above underlying


def test_expired_option_returns_nan():
    assert np.isnan(implied_vol(5.0, 100.0, 100.0, 0.0, 0.05))  # T = 0


# --- A near-zero-tolerance sanity check on convergence quality -------------
def test_convergence_precision():
    """Solver should hit machine-ish precision, not just 'close enough'."""
    S, K, T, r, true_sigma = 100.0, 100.0, 1.0, 0.05, 0.22
    price = black_scholes(S, K, T, r, true_sigma, option_type="call")
    assert implied_vol(price, S, K, T, r, option_type="call") == pytest.approx(
        true_sigma, abs=1e-6
    )