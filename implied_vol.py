"""Implied volatility solver: Newton-Raphson with bisection fallback."""

import numpy as np

from pricing import black_scholes
from pricing import black_scholes_vega  # adjust names to yours


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 5.0,
) -> float:
    """Back out implied volatility from an option price.

    Newton-Raphson using vega as the derivative; falls back to bisection
    whenever a step leaves the bracket or vega is too small to trust.
    Returns np.nan if the price is outside no-arbitrage bounds.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if price <= 0 or T <= 0:
        return np.nan

    # --- 1. No-arbitrage bounds -------------------------------------------
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        lower, upper = max(disc_S - disc_K, 0.0), disc_S
    else:
        lower, upper = max(disc_K - disc_S, 0.0), disc_K
    if not (lower - tol <= price <= upper + tol):
        return np.nan  # unpriceable — bad quote or stale data

    # --- 2. Bracket the root ----------------------------------------------
    def f(sig):
        return black_scholes(S, K, T, r, sig, option_type=option_type, q=q) - price

    lo, hi = sigma_lo, sigma_hi
    if f(lo) * f(hi) > 0:
        return np.nan  # no sign change → root not inside [sigma_lo, sigma_hi]

    # --- 3. Seed Newton-Raphson (Brenner–Subrahmanyam ATM approximation) ---
    sigma = np.sqrt(2.0 * np.pi / T) * price / S
    sigma = min(max(sigma, sigma_lo), sigma_hi)  # clamp into the bracket

    # --- 4. Iterate: NR when it behaves, bisection when it doesn't ---------
    for _ in range(max_iter):
        val = f(sigma)
        if abs(val) < tol:
            return sigma

        # Tighten the bracket with every evaluation.
        if val > 0:
            hi = sigma
        else:
            lo = sigma

        vega = black_scholes_vega(S, K, T, r, sigma, q=q)

        if vega < 1e-8:                       # vega collapsed → NR unreliable
            sigma = 0.5 * (lo + hi)
            continue

        sigma_next = sigma - val / vega       # the Newton step
        if sigma_next <= lo or sigma_next >= hi:
            sigma = 0.5 * (lo + hi)           # step escaped the bracket → bisect
        else:
            sigma = sigma_next                # step accepted

    return sigma  # bracket is tight by now; return best estimate