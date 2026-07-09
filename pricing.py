from scipy.stats import norm
import numpy as np

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str ='call') -> float:
    """
    Black-Scholes European Option Price Function

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    T: float
        Time to maturity in years (ex: 0.25 for 3 months)
    r: float
        Risk-free interest rate (annualized, as a decimal)
    sigma: float
        Volatility of the underlying stock (annualized, as a decimal)
    option_type: str
        Type of option: 'call' or 'put'
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price