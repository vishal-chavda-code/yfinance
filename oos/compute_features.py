"""
Compute features for 2026 OOS data using 2025 as rolling lookback.

Mirrors the feature engineering in plotly_dash.load_data() and
src/calc_rolling_amihud.py but for the 2026 evaluation period:
  - Combines 2025 OHLCV (lookback) + 2026 OHLCV
  - Computes daily ILLIQ, rolling 252d Amihud, 21d Amihud
  - Computes illiq_ratio, illiq_zscore (term structure)
  - Forward-fills the last known 2025 free-float onto 2026 rows
  - Outputs a single oos_features.parquet ready for evaluation

Usage:
    python -m oos.compute_features
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OOS_DIR = PROJECT_ROOT / "oos" / "data"

OHLCV_2025 = DATA_DIR / "us_equities_2025_ohlcv.parquet"
OHLCV_2026 = OOS_DIR / "us_equities_2026_ohlcv.parquet"
MERGED_2025 = DATA_DIR / "amihud_with_free_float.parquet"
OUTPUT = OOS_DIR / "oos_features.parquet"

ROLLING_WINDOW = 252
ROLLING_MIN_PERIODS = 200


def load_and_combine() -> pd.DataFrame:
    """Stack 2025 (lookback) + 2026 (evaluation) OHLCV."""
    print(f"Loading {OHLCV_2025}...")
    df25 = pd.read_parquet(OHLCV_2025)
    print(f"  2025: {len(df25):,} rows, {df25['ticker'].nunique()} tickers")

    print(f"Loading {OHLCV_2026}...")
    df26 = pd.read_parquet(OHLCV_2026)
    print(f"  2026: {len(df26):,} rows, {df26['ticker'].nunique()} tickers")

    # Only keep tickers present in both years
    common = set(df25["ticker"].unique()) & set(df26["ticker"].unique())
    print(f"  Common tickers: {len(common)}")

    df = pd.concat([
        df25[df25["ticker"].isin(common)],
        df26[df26["ticker"].isin(common)],
    ], ignore_index=True)

    df = df.drop_duplicates(subset=["ticker", "date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df, common


def compute_amihud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling Amihud + term structure features."""
    print("Computing daily returns and ILLIQ...")
    df["return"] = df.groupby("ticker")["close"].pct_change()
    df["dollar_volume"] = df["close"] * df["volume"]
    df["abs_return"] = df["return"].abs()
    df["illiq"] = np.where(df["dollar_volume"] > 0,
                           df["abs_return"] / df["dollar_volume"], np.nan)
    df = df.dropna(subset=["return"]).reset_index(drop=True)

    print(f"Computing rolling {ROLLING_WINDOW}d Amihud...")
    df["illiq_252d"] = (
        df.groupby("ticker")["illiq"]
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean())
    )

    print("Computing 21d Amihud and term structure...")
    df["illiq_21d"] = (
        df.groupby("ticker")["illiq"]
        .transform(lambda x: x.rolling(21, min_periods=15).mean())
    )
    df["illiq_ratio"] = df["illiq_21d"] / df["illiq_252d"]
    df["illiq_ratio"] = df["illiq_ratio"].replace([np.inf, -np.inf], np.nan)

    df["illiq_ratio_mean"] = (
        df.groupby("ticker")["illiq_ratio"]
        .transform(lambda x: x.rolling(252, min_periods=60).mean())
    )
    df["illiq_ratio_std"] = (
        df.groupby("ticker")["illiq_ratio"]
        .transform(lambda x: x.rolling(252, min_periods=60).std())
    )
    df["illiq_zscore"] = (
        (df["illiq_ratio"] - df["illiq_ratio_mean"]) / df["illiq_ratio_std"]
    )
    df["illiq_zscore"] = df["illiq_zscore"].replace([np.inf, -np.inf], np.nan)

    return df


def attach_free_float(df_2026: pd.DataFrame, common_tickers: set) -> pd.DataFrame:
    """Forward-fill the last known 2025 free-float values onto 2026 rows."""
    print("Attaching free-float from last known 2025 values...")
    merged_2025 = pd.read_parquet(MERGED_2025)

    # Get the last known free-float per ticker from 2025
    ff_cols = ["ticker", "cur_mkt_cap", "eqy_free_float_pct", "free_float_mkt_cap"]
    ff_latest = (
        merged_2025[merged_2025["ticker"].isin(common_tickers)]
        .dropna(subset=["eqy_free_float_pct"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker"], keep="last")[ff_cols]
    )
    print(f"  Last known FF for {ff_latest['ticker'].nunique()} tickers")

    df_2026 = df_2026.merge(ff_latest, on="ticker", how="left", suffixes=("", "_ff"))
    return df_2026


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns matching plotly_dash.load_data()."""
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["parkinson_vol"] = np.log(df["high"] / df["low"])
    df["ff_ratio"] = df["eqy_free_float_pct"] / 100
    df["cur_mkt_cap"] = df["cur_mkt_cap"] * 1e6   # source is in $M -> convert to $
    df["log_mktcap"] = np.log1p(df["cur_mkt_cap"])
    df["log_illiq"] = np.log(df["illiq_252d"]).replace(-np.inf, np.nan)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["signed_return"] = df["return"]
    return df


if __name__ == "__main__":
    # 1. Combine 2025 lookback + 2026 OHLCV
    df_combined, common = load_and_combine()

    # 2. Compute Amihud + term structure on the combined series
    df_combined = compute_amihud_features(df_combined)

    # 3. Filter to 2026 only
    cutoff = pd.Timestamp("2026-01-01")
    df_2026 = df_combined[df_combined["date"] >= cutoff].copy().reset_index(drop=True)
    print(f"2026 rows: {len(df_2026):,}, tickers: {df_2026['ticker'].nunique()}")

    # 4. Attach free-float (last known from 2025)
    df_2026 = attach_free_float(df_2026, common)

    # 5. Derive dashboard features
    df_2026 = derive_features(df_2026)

    # 6. Save
    print(f"\nSaving → {OUTPUT}")
    df_2026.to_parquet(OUTPUT, index=False, engine="pyarrow")

    # Summary
    print(f"\n{'='*60}")
    print(f"  OOS Feature Set — 2026 YTD")
    print(f"{'='*60}")
    print(f"  Rows:       {len(df_2026):,}")
    print(f"  Tickers:    {df_2026['ticker'].nunique()}")
    print(f"  Date range: {df_2026['date'].min().date()} → {df_2026['date'].max().date()}")
    for col in ["illiq_252d", "illiq_21d", "illiq_zscore", "log_illiq", "eqy_free_float_pct"]:
        nn = df_2026[col].notna().sum()
        print(f"  {col}: {nn:,} non-null ({nn/len(df_2026)*100:.1f}%)")
    print(f"{'='*60}")
