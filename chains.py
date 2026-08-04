"""
Option chain fetch + clean (yfinance backend, no API key required).

Fetches real option chains at runtime, caches to a gitignored dir,
returns a cleaned DataFrame ready for IV / smile work.

No data is committed to the repo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / ".cache" / "chains"
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6h — chains are stale-ish intraday, fine for smile work


@dataclass
class ChainQuote:
    """One cleaned option quote."""
    symbol: str
    underlying: str
    expiry: date
    strike: float
    right: str  # "C" or "P"
    bid: float
    ask: float
    mid: float
    spot: float
    t_years: float


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def _cache_path(underlying: str, tag: str) -> Path:
    return CACHE_DIR / f"{underlying.upper()}_{tag}.json"


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > CACHE_TTL_SECONDS:
        return None
    with path.open() as f:
        return json.load(f)


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f)


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

def fetch_chain_raw(
    underlying: str,
    expiry_gte: str | None = None,
    expiry_lte: str | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Fetch the raw option-chain snapshot for `underlying` via yfinance.

    Returns {occ_symbol: {expiry, strike, right, bid, ask}} — a flat dict
    keyed by contractSymbol, so the downstream shape matches the old
    Alpaca snapshot dict (one entry per contract).
    """
    underlying = underlying.upper()
    tag = f"{expiry_gte or 'any'}_{expiry_lte or 'any'}"
    cpath = _cache_path(underlying, tag)

    if use_cache:
        cached = _read_cache(cpath)
        if cached is not None:
            return cached

    tk = yf.Ticker(underlying)
    expiries = list(tk.options)  # tuple of 'YYYY-MM-DD' strings

    def _in_window(exp: str) -> bool:
        if expiry_gte and exp < expiry_gte:
            return False
        if expiry_lte and exp > expiry_lte:
            return False
        return True

    expiries = [e for e in expiries if _in_window(e)]

    snapshots: dict = {}
    for exp in expiries:
        oc = tk.option_chain(exp)  # namedtuple: .calls, .puts (DataFrames)
        for right, frame in (("C", oc.calls), ("P", oc.puts)):
            for row in frame.itertuples(index=False):
                snapshots[row.contractSymbol] = {
                    "expiry": exp,
                    "strike": float(row.strike),
                    "right": right,
                    "bid": None if pd.isna(row.bid) else float(row.bid),
                    "ask": None if pd.isna(row.ask) else float(row.ask),
                }

    _write_cache(cpath, snapshots)
    return snapshots


def fetch_spot(underlying: str) -> float:
    """Latest price for the underlying via yfinance fast_info."""
    tk = yf.Ticker(underlying.upper())
    px = tk.fast_info.get("last_price") or tk.fast_info.get("previous_close")
    if px is None:
        # fallback: last close from a 1d history pull
        px = float(tk.history(period="1d")["Close"].iloc[-1])
    return float(px)


# --------------------------------------------------------------------------
# clean
# --------------------------------------------------------------------------

def clean_chain(
    snapshots: dict,
    spot: float,
    asof: date | None = None,
    min_bid: float = 0.05,
    max_rel_spread: float = 0.25,
    min_t_days: int = 7,
    max_t_days: int = 400,
    moneyness_band: tuple[float, float] = (0.70, 1.30),
) -> pd.DataFrame:
    """
    Turn raw snapshots into a clean quote table.

    Filters applied (each one is a real source of garbage IVs):
      - missing/zero bid or ask               -> no two-sided market
      - bid < min_bid                         -> penny options, IV is noise
      - crossed/locked markets (bid >= ask)   -> stale feed
      - relative spread > max_rel_spread      -> untradeable, mid is meaningless
      - t < min_t_days                        -> near-expiry IV explodes
      - t > max_t_days                        -> illiquid LEAPS
      - strike outside moneyness band         -> deep wings, vega ~ 0
    """
    asof = asof or datetime.now(timezone.utc).date()
    rows = []

    for sym, snap in snapshots.items():
        bid = snap.get("bid")
        ask = snap.get("ask")
        if bid is None or ask is None:
            continue
        bid, ask = float(bid), float(ask)

        if bid <= 0 or ask <= 0:
            continue
        if bid >= ask:                      # crossed or locked
            continue
        if bid < min_bid:
            continue

        mid = 0.5 * (bid + ask)
        if (ask - bid) / mid > max_rel_spread:
            continue

        expiry = datetime.strptime(snap["expiry"], "%Y-%m-%d").date()
        right = snap["right"]
        strike = float(snap["strike"])

        t_days = (expiry - asof).days
        if t_days < min_t_days or t_days > max_t_days:
            continue

        m = strike / spot
        if not (moneyness_band[0] <= m <= moneyness_band[1]):
            continue

        rows.append({
            "symbol": sym,
            "underlying": sym[:sym.index(next(c for c in sym if c.isdigit()))],
            "expiry": expiry,
            "t_days": t_days,
            "t_years": t_days / 365.0,
            "strike": strike,
            "right": right,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": ask - bid,
            "rel_spread": (ask - bid) / mid,
            "spot": spot,
            "moneyness": m,
            "log_moneyness": np.log(m),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(["expiry", "right", "strike"]).reset_index(drop=True)


def get_clean_chain(
    underlying: str,
    expiry_gte: str | None = None,
    expiry_lte: str | None = None,
    use_cache: bool = True,
    **clean_kwargs,
) -> pd.DataFrame:
    """One-call convenience: fetch spot + chain, return cleaned frame."""
    spot = fetch_spot(underlying)
    raw = fetch_chain_raw(
        underlying,
        expiry_gte=expiry_gte, expiry_lte=expiry_lte, use_cache=use_cache,
    )
    return clean_chain(raw, spot, **clean_kwargs)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch and clean an option chain (yfinance).")
    p.add_argument("underlying")
    p.add_argument("--expiry-gte", default=None)
    p.add_argument("--expiry-lte", default=None)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    df = get_clean_chain(
        args.underlying,
        expiry_gte=args.expiry_gte,
        expiry_lte=args.expiry_lte,
        use_cache=not args.no_cache,
    )
    if df.empty:
        print("No quotes survived cleaning — try a wider moneyness band or check market hours.")
    else:
        print(f"spot={df['spot'].iloc[0]:.2f}  quotes={len(df)}  expiries={df['expiry'].nunique()}")
        print(df.head(20).to_string())