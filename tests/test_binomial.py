import numpy as np
import pytest

from binomial import binomial_european
from binomial import binomial_american
from pricing import black_scholes  


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_converges_to_black_scholes(option_type):
    """Tree price -> BS price as N increases. This is the deliverable."""
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25
    bs = black_scholes(S, K, T, r, sigma, option_type=option_type)

    errors = [
        abs(binomial_european(S, K, T, r, sigma, N=N, option_type=option_type) - bs)
        for N in (10, 50, 200, 1000)
    ]

    # Error shrinks monotonically and lands close
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)), errors
    assert errors[-1] < 0.01


def test_put_call_parity():
    S, K, T, r, sigma, N = 100.0, 95.0, 0.75, 0.03, 0.30, 500
    c = binomial_european(S, K, T, r, sigma, N=N, option_type="call")
    p = binomial_european(S, K, T, r, sigma, N=N, option_type="put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-2)


def test_deep_itm_call_approaches_intrinsic():
    price = binomial_european(200.0, 50.0, 1.0, 0.05, 0.20, N=500, option_type="call")
    assert price == pytest.approx(200.0 - 50.0 * np.exp(-0.05), abs=0.5)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        binomial_european(100, 100, 1.0, 0.05, 0.25, N=0)


def test_american_put_ge_european_put():
    args = dict(S=100, K=110, T=1.0, r=0.05, sigma=0.25, N=500, option_type="put")
    assert binomial_american(**args) >= binomial_european(**args) - 1e-9

def test_american_call_no_div_equals_european():
    # with q=0, never optimal to exercise an American call early
    args = dict(S=100, K=95, T=1.0, r=0.05, sigma=0.25, N=500, option_type="call")
    am = binomial_american(**args, q=0.0)
    eu = binomial_european(**args)
    assert abs(am - eu) < 1e-4

def test_american_put_early_exercise_premium():
    # deep ITM American put should carry a premium over European
    args = dict(S=60, K=100, T=1.0, r=0.08, sigma=0.20, N=500, option_type="put")
    assert binomial_american(**args) > binomial_european(**args) + 1e-3