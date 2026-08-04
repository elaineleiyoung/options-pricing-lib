"""
Volatility smile from a real, cleaned option chain.

Pipeline:
  chains.get_clean_chain  ->  implied_vol (row-wise)  ->  IV vs strike plot

Produces figures/vol_smile.png: one curve per expiry, spot marked.
Runs end-to-end from the command line:

    python smile.py AAPL --expiry-lte 2026-10-01 --right otm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chains import get_clean_chain
from implied_vol import implied_vol

FIG_DIR = Path(__file__).parent / "figures"
_RIGHT_TO_TYPE = {"C": "call", "P": "put"}


def add_iv(
    df: pd.DataFrame,
    r: float,
    q: float = 0.0,
) -> pd.DataFrame:
    """Add an `iv` column by solving each row's mid price. Drops non-converged."""
    def solve(row) -> float:
        return implied_vol(
            price=row["mid"],
            S=row["spot"],
            K=row["strike"],
            T=row["t_years"],
            r=r,
            option_type=_RIGHT_TO_TYPE[row["right"]],
            q=q,
        )

    out = df.copy()
    out["iv"] = out.apply(solve, axis=1)
    out = out[out["iv"].notna()]
    # Sanity band: strip absurd solves (e.g. numerical escapes near the wings).
    out = out[(out["iv"] > 0.01) & (out["iv"] < 3.0)]
    return out.reset_index(drop=True)


def select_smile_rows(df: pd.DataFrame, right: str) -> pd.DataFrame:
    """
    Choose which contracts form the smile.

      "otm"  -> puts below spot, calls above (the clean, liquid convention)
      "call" -> calls only
      "put"  -> puts only
    """
    if right == "call":
        return df[df["right"] == "C"]
    if right == "put":
        return df[df["right"] == "P"]
    # otm: OTM puts (strike < spot) + OTM calls (strike > spot)
    otm_puts = (df["right"] == "P") & (df["strike"] < df["spot"])
    otm_calls = (df["right"] == "C") & (df["strike"] >= df["spot"])
    return df[otm_puts | otm_calls]


def plot_smile(
    df: pd.DataFrame,
    underlying: str,
    outfile: Path,
    x: str = "strike",
) -> Path:
    """Plot IV vs strike (or log_moneyness), one line per expiry."""
    if df.empty:
        raise ValueError("No IVs to plot — chain empty or all rows failed to solve.")

    spot = df["spot"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 6))

    for expiry, grp in df.groupby("expiry"):
        grp = grp.sort_values(x)
        label = f"{expiry}  ({grp['t_days'].iloc[0]}d)"
        ax.plot(grp[x], grp["iv"] * 100, marker="o", ms=4, lw=1.3, label=label)

    if x == "strike":
        ax.axvline(spot, color="k", ls="--", lw=1, alpha=0.6, label=f"spot {spot:.2f}")
        ax.set_xlabel("Strike")
    else:
        ax.axvline(0.0, color="k", ls="--", lw=1, alpha=0.6, label="ATM")
        ax.set_xlabel("log-moneyness  ln(K/S)")

    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"{underlying} implied volatility smile")
    ax.legend(fontsize=8, title="expiry")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def build_smile(
    underlying: str,
    r: float = 0.04,
    q: float = 0.0,
    right: str = "otm",
    x: str = "strike",
    expiry_gte: str | None = None,
    expiry_lte: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch -> clean -> solve IV -> plot. Returns the solved frame."""
    chain = get_clean_chain(
        underlying,
        expiry_gte=expiry_gte, expiry_lte=expiry_lte, use_cache=use_cache,
    )
    if chain.empty:
        raise ValueError("Cleaned chain is empty — check market hours or widen filters.")

    solved = add_iv(chain, r=r, q=q)
    smile = select_smile_rows(solved, right=right)
    plot_smile(smile, underlying, FIG_DIR / "vol_smile.png", x=x)
    return smile


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a vol smile from a real option chain.")
    p.add_argument("underlying")
    p.add_argument("--r", type=float, default=0.04, help="flat risk-free rate")
    p.add_argument("--q", type=float, default=0.0, help="dividend yield")
    p.add_argument("--right", choices=["otm", "call", "put"], default="otm")
    p.add_argument("--x", choices=["strike", "log_moneyness"], default="strike")
    p.add_argument("--expiry-gte", default=None)
    p.add_argument("--expiry-lte", default=None)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    smile = build_smile(
        args.underlying,
        r=args.r, q=args.q, right=args.right, x=args.x,
        expiry_gte=args.expiry_gte, expiry_lte=args.expiry_lte,
        use_cache=not args.no_cache,
    )
    print(f"solved {len(smile)} IVs across {smile['expiry'].nunique()} expiries")
    print(f"IV range: {smile['iv'].min()*100:.1f}% – {smile['iv'].max()*100:.1f}%")
    print(f"saved -> {FIG_DIR / 'vol_smile.png'}")