"""
Utility functions for downloading raw OHLCV data from Binance
and (optionally) saving it to the project’s *raw_crypto* path.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Final

import requests
import yaml
import pandas as pd

from utils.logger import get_logger

# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #

# Load once, at import-time.  If the file is missing, raise a clear error that
# will surface in the Streamlit logs instead of silently swallowing the import.
CFG_PATH: Final = Path("config/settings.yaml")
if not CFG_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {CFG_PATH}.  Make sure it is committed and the relative "
        "path is correct in production."
    )

config = yaml.safe_load(CFG_PATH.read_text())
log    = get_logger(level=config["logging"]["level"])
API    = config["data"]["crypto_api"]

# ----------------------------------------------------------------------------- #
# Public functions
# ----------------------------------------------------------------------------- #
def fetch_binance(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Download **spot Candle/Kline** data from Binance’s public endpoint.

    Parameters
    ----------
    symbol   : e.g. ``"BTCUSDT"``
    interval : e.g. ``"1h"``, ``"15m"``, ``"1d"``
    limit    : number of rows to pull (max 1 000 per request)

    Returns
    -------
    pandas.DataFrame with columns
    ``open_time, open, high, low, close, volume`` and dtypes::

        open_time    datetime64[ns]
        open               float64
        high               float64
        low                float64
        close              float64
        volume             float64
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(API, params=params, timeout=10)
    r.raise_for_status()

    raw = pd.DataFrame(
        r.json(),
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "_1", "_2", "_3", "_4", "_5", "_6"             # throw-aways
        ],
    )

    df = raw[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.astype(
        {
            "open":   float,
            "high":   float,
            "low":    float,
            "close":  float,
            "volume": float,
        }
    )
    return df


def save_crypto(symbol: str) -> Path:
    """
    Convenience wrapper that calls ``fetch_binance`` and writes the result
    to *<raw_crypto>/<symbol>_<interval>.parquet*.

    Returns
    -------
    Path to the written file.
    """
    out_dir = Path(config["paths"]["raw_crypto"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df   = fetch_binance(symbol, config["data"]["interval"], config["data"]["limit"])
    file = out_dir / f"{symbol}_{config['data']['interval']}.parquet"
    df.to_parquet(file, index=False)
    log.info(f"Saved {symbol} → {file}  ({len(df):,} rows)")
    return file


# Make the symbol importable:  ``from ingestion.fetch_binance import fetch_binance``
__all__ = ["fetch_binance", "save_crypto"]


if __name__ == "__main__":
    for s in config["symbols"]["crypto"]:
        save_crypto(s)
