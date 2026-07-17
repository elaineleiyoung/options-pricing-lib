import numpy as np

def crr_params(sigma: float, r: float, q: float, dt: float):
    """ CRR up/down factors and risk-neutral probability for one step"""
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1.0 / u  # Down factor
    p = (np.exp((r - q) * dt) - d) / (u - d) # Risk-neutral probability
    return u, d, p

def binomial_european(S: float, K: float, T: float, r: float, sigma: float, N: int = 500, q: float = 0.0, option_type: str = "call") -> float:
    """
    Price a European option on a CRR binoimial tree.
    S: spot, K: strike, T: time to maturity in years, r: risk-free reate, sigma: volatility, N: number of steps, q: continuous dividend yield.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if T <= 0 or sigma <= 0 or N < 1:
        raise ValueError("T and sigma must be positive, N must be at least 1")
    
    dt = T/N
    u, d, p = crr_params(sigma, r, q, dt)
    if not 0.0 < p < 1.0:
        raise ValueError("Risk-neutral probability p must be in (0, 1). Check your parameters.")
    
    disc = np.exp(-r*dt)

    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))

    #Terminal payoffs
    if option_type == "call":
        V = np.maximum(ST - K, 0)
    else:
        V = np.maximum(K - ST, 0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])

    return float(V[0])

def binomial_american(S, K, T, r, sigma, N, option_type="call", q=0.0):
    """American option via CRR backward induction with early-exercise check."""
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if T <= 0 or sigma <= 0 or N < 1:
        raise ValueError("require T>0, sigma>0, N>=1")

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(f"risk-neutral prob out of (0,1): p={p:.4f} — check dt/params")

    # terminal asset prices: j up-moves, N-j down-moves
    j = np.arange(N + 1)
    ST = S * u**j * d**(N - j)
    payoff = (ST - K) if option_type == "call" else (K - ST)
    V = np.maximum(payoff, 0.0)

    # backward induction with early exercise
    for i in range(N - 1, -1, -1):
        j = np.arange(i + 1)
        Si = S * u**j * d**(i - j)
        cont = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])
        intrinsic = (Si - K) if option_type == "call" else (K - Si)
        V = np.maximum(cont, np.maximum(intrinsic, 0.0))
    return V[0]