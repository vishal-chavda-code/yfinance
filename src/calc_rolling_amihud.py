"""
Compute rolling 252-day Amihud ILLIQ from combined 2024+2025 OHLCV data.

ILLIQ_t = |r_t| / dollar_volume_t
Rolling Amihud = rolling 252-day mean of daily ILLIQ per ticker

Output: daily rows for 2025 only (2024 serves as lookback).

Usage:
    python -m src.calc_rolling_amihud
"""

import pandas as pd
import numpy as np
from src.config import (
    OHLCV_2024, OHLCV_2025, ROLLING_AMIHUD,
    ROLLING_WINDOW, ROLLING_MIN_PERIODS,
)


def load_and_combine() -> pd.DataFrame:
    """Load 2024 + 2025 OHLCV and stack them."""
    print(f"Loading {OHLCV_2024}...")
    df24 = pd.read_parquet(OHLCV_2024)
    print(f"  2024: {len(df24):,} rows, {df24['ticker'].nunique()} tickers")

    print(f"Loading {OHLCV_2025}...")
    df25 = pd.read_parquet(OHLCV_2025)
    print(f"  2025: {len(df25):,} rows, {df25['ticker'].nunique()} tickers")

    # Only keep tickers present in both years for consistent rolling calculation
    common = set(df24["ticker"].unique()) & set(df25["ticker"].unique())
    print(f"  Common tickers in both years: {len(common)}")

    df = pd.concat([
        df24[df24["ticker"].isin(common)],
        df25[df25["ticker"].isin(common)],
    ], ignore_index=True)

    df = df.drop_duplicates(subset=["ticker", "date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_rolling_amihud(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily ILLIQ and rolling 252-day mean per ticker."""
    print("Computing daily returns and ILLIQ...")

    df["return"] = df.groupby("ticker")["close"].pct_change()
    df["dollar_volume"] = df["close"] * df["volume"]
    df["abs_return"] = df["return"].abs()

    df["illiq"] = np.where(
        df["dollar_volume"] > 0,
        df["abs_return"] / df["dollar_volume"],
        np.nan,
    )

    # Drop first row per ticker (no return)
    df = df.dropna(subset=["return"]).reset_index(drop=True)

    print(f"Computing rolling {ROLLING_WINDOW}-day Amihud (min_periods={ROLLING_MIN_PERIODS})...")
    df["illiq_252d"] = (
        df.groupby("ticker")["illiq"]
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean())
    )

    return df


def filter_to_2025(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 2025 rows — the 2024 data was just for lookback."""
    cutoff = pd.Timestamp("2025-01-01")
    out = df[df["date"] >= cutoff].copy().reset_index(drop=True)
    print(f"Filtered to 2025: {len(out):,} rows, {out['ticker'].nunique()} tickers")
    return out


if __name__ == "__main__":
    df = load_and_combine()
    df = compute_rolling_amihud(df)
    result = filter_to_2025(df)

    cols = [
        "ticker", "date", "close", "volume", "dollar_volume",
        "return", "abs_return", "illiq", "illiq_252d",
    ]
    result = result[cols]

    print(f"\nSaving → {ROLLING_AMIHUD}")
    result.to_parquet(ROLLING_AMIHUD, index=False, engine="pyarrow")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Rolling 252-day Amihud ILLIQ — 2025")
    print(f"{'='*60}")
    print(f"  Rows:       {len(result):,}")
    print(f"  Tickers:    {result['ticker'].nunique()}")
    print(f"  Date range: {result['date'].min().date()} → {result['date'].max().date()}")

    illiq = result["illiq_252d"]
    non_null = illiq.notna().sum()
    print(f"\n  illiq_252d (rolling annual Amihud):")
    print(f"    non-null:  {non_null:,} / {len(result):,} ({non_null/len(result)*100:.1f}%)")
    print(f"    mean:      {illiq.mean():.2e}")
    print(f"    median:    {illiq.median():.2e}")
    print(f"    p5:        {illiq.quantile(0.05):.2e}")
    print(f"    p95:       {illiq.quantile(0.95):.2e}")
    print(f"{'='*60}")
