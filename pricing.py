from scipy.stats import norm
import numpy as np


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0,
) -> float:
    """Black-Scholes European option price.

    S: spot, K: strike, T: years to maturity, r: risk-free rate,
    sigma: annualized vol, q: continuous dividend yield (0 = non-dividend).
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def black_scholes_vega(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """Vega: dPrice/dSigma. Same for calls and puts.

    Per 1.0 (100%) change in vol — divide by 100 for the per-1-vol-point convention.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


# Back-compat alias so existing imports (`black_scholes`) keep working.
black_scholes = black_scholes_price