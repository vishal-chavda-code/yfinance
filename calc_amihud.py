"""
Calculate Amihud ILLIQ ratios from the 2025 OHLCV dataset.

ILLIQ_t = |r_t| / (P_t * V_t)

where:
  r_t     = (close_t - close_{t-1}) / close_{t-1}  (simple daily return)
  P_t     = close price
  V_t     = share volume
  P_t*V_t = dollar volume

Output columns:
  ticker, date, close, volume, dollar_volume, return, abs_return,
  illiq            — raw daily Amihud ILLIQ
  illiq_21d        — 21-trading-day rolling mean (for cross-stock comparison)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"c:\Users\vmc30\OneDrive\Desktop\Personal_Repos\yfinance\data")
INPUT = DATA_DIR / "us_equities_2025_ohlcv.parquet"
OUTPUT = DATA_DIR / "us_equities_2025_amihud.parquet"

ROLLING_WINDOW = 21


def compute_amihud(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    # Daily return: simple return within each ticker
    df["return"] = df.groupby("ticker")["close"].pct_change()

    # Dollar volume = close * share volume
    df["dollar_volume"] = df["close"] * df["volume"]

    # |r_t|
    df["abs_return"] = df["return"].abs()

    # Raw daily ILLIQ = |r_t| / dollar_volume
    # Guard against zero/near-zero dollar volume -> set to NaN
    df["illiq"] = np.where(
        df["dollar_volume"] > 0,
        df["abs_return"] / df["dollar_volume"],
        np.nan,
    )

    # 21-day rolling mean per ticker (min_periods=15 to avoid edge noise)
    df["illiq_21d"] = (
        df.groupby("ticker")["illiq"]
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=15).mean())
    )

    # Drop first row per ticker (no return on day 1)
    df = df.dropna(subset=["return"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    print(f"Reading {INPUT}...")
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers")

    print("Computing Amihud ILLIQ...")
    result = compute_amihud(df)

    # Keep useful columns in clean order
    cols = [
        "ticker", "date", "close", "volume", "dollar_volume",
        "return", "abs_return", "illiq", "illiq_21d",
    ]
    result = result[cols]

    print(f"Saving {OUTPUT}...")
    result.to_parquet(OUTPUT, index=False, engine="pyarrow")

    # Summary stats
    print(f"\n{'='*60}")
    print(f"  Rows:           {len(result):,}")
    print(f"  Tickers:        {result['ticker'].nunique()}")
    print(f"  Date range:     {result['date'].min().date()} - {result['date'].max().date()}")

    illiq = result["illiq"]
    print(f"\n  ILLIQ (daily):")
    print(f"    mean:         {illiq.mean():.2e}")
    print(f"    median:       {illiq.median():.2e}")
    print(f"    p5:           {illiq.quantile(0.05):.2e}")
    print(f"    p95:          {illiq.quantile(0.95):.2e}")
    print(f"    NaN count:    {illiq.isna().sum():,}")

    illiq_21 = result["illiq_21d"]
    print(f"\n  ILLIQ (21d rolling avg):")
    print(f"    mean:         {illiq_21.mean():.2e}")
    print(f"    median:       {illiq_21.median():.2e}")
    print(f"    non-null:     {illiq_21.notna().sum():,}")
    print(f"    NaN count:    {illiq_21.isna().sum():,}")

    # Cross-sectional snapshot: median ILLIQ_21d by ticker on last date
    last_date = result["date"].max()
    snap = result[result["date"] == last_date].copy()
    print(f"\n  Cross-section on {last_date.date()}:")
    print(f"    Tickers with illiq_21d: {snap['illiq_21d'].notna().sum()}")
    q = snap["illiq_21d"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    for pct, val in q.items():
        print(f"    p{int(pct*100):02d}:          {val:.2e}")
    print(f"{'='*60}")
