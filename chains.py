"""
Option chain fetch + clean.

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
import requests

CACHE_DIR = Path(__file__).parent / ".cache" / "chains"
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6h — chains are stale-ish intraday, fine for smile work

ALPACA_OPTIONS_BASE = "https://data.alpaca.markets/v1beta1/options"


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

def _headers(key: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def fetch_chain_raw(
    underlying: str,
    api_key: str,
    api_secret: str,
    expiry_gte: str | None = None,
    expiry_lte: str | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Fetch the raw option-chain snapshot for `underlying`.

    Paginates via next_page_token. Returns {occ_symbol: snapshot_dict}.
    """
    underlying = underlying.upper()
    tag = f"{expiry_gte or 'any'}_{expiry_lte or 'any'}"
    cpath = _cache_path(underlying, tag)

    if use_cache:
        cached = _read_cache(cpath)
        if cached is not None:
            return cached

    snapshots: dict = {}
    params = {"limit": 1000}
    if expiry_gte:
        params["expiration_date_gte"] = expiry_gte
    if expiry_lte:
        params["expiration_date_lte"] = expiry_lte

    url = f"{ALPACA_OPTIONS_BASE}/snapshots/{underlying}"
    token = None

    while True:
        if token:
            params["page_token"] = token
        r = requests.get(url, headers=_headers(api_key, api_secret), params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        snapshots.update(payload.get("snapshots", {}))
        token = payload.get("next_page_token")
        if not token:
            break

    _write_cache(cpath, snapshots)
    return snapshots


def fetch_spot(underlying: str, api_key: str, api_secret: str) -> float:
    """Latest trade price for the underlying."""
    url = f"https://data.alpaca.markets/v2/stocks/{underlying.upper()}/trades/latest"
    r = requests.get(url, headers=_headers(api_key, api_secret), timeout=30)
    r.raise_for_status()
    return float(r.json()["trade"]["p"])


# --------------------------------------------------------------------------
# parse + clean
# --------------------------------------------------------------------------

def parse_occ(symbol: str) -> tuple[str, date, str, float]:
    """
    Parse an OCC symbol: AAPL251219C00150000
      root | YYMMDD | C/P | strike * 1000, 8 digits
    """
    strike = float(symbol[-8:]) / 1000.0
    right = symbol[-9]
    yymmdd = symbol[-15:-9]
    root = symbol[:-15]
    expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    return root, expiry, right, strike


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
        quote = snap.get("latestQuote")
        if not quote:
            continue

        bid = quote.get("bp")
        ask = quote.get("ap")
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

        try:
            root, expiry, right, strike = parse_occ(sym)
        except (ValueError, IndexError):
            continue

        t_days = (expiry - asof).days
        if t_days < min_t_days or t_days > max_t_days:
            continue

        m = strike / spot
        if not (moneyness_band[0] <= m <= moneyness_band[1]):
            continue

        rows.append({
            "symbol": sym,
            "underlying": root,
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
    api_key: str,
    api_secret: str,
    expiry_gte: str | None = None,
    expiry_lte: str | None = None,
    use_cache: bool = True,
    **clean_kwargs,
) -> pd.DataFrame:
    """One-call convenience: fetch spot + chain, return cleaned frame."""
    spot = fetch_spot(underlying, api_key, api_secret)
    raw = fetch_chain_raw(
        underlying, api_key, api_secret,
        expiry_gte=expiry_gte, expiry_lte=expiry_lte, use_cache=use_cache,
    )
    return clean_chain(raw, spot, **clean_kwargs)


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Fetch and clean an option chain.")
    p.add_argument("underlying")
    p.add_argument("--expiry-gte", default=None)
    p.add_argument("--expiry-lte", default=None)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    key = os.environ["Key"]
    secret = os.environ["Secret"]

    df = get_clean_chain(
        args.underlying, key, secret,
        expiry_gte=args.expiry_gte,
        expiry_lte=args.expiry_lte,
        use_cache=not args.no_cache,
    )
    print(f"spot={df['spot'].iloc[0]:.2f}  quotes={len(df)}  expiries={df['expiry'].nunique()}")
    print(df.head(20).to_string())