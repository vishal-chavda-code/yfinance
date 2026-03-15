"""
Merge monthly Bloomberg free float market cap onto daily rolling Amihud records.

Each daily row receives the most recent monthly free float mkt cap value
(forward-filled from the first business day of each month).

Usage:
    python -m src.merge_data
"""

import pandas as pd
from src.config import ROLLING_AMIHUD, BBG_FREE_FLOAT, FINAL_MERGED


def merge_free_float_onto_amihud() -> pd.DataFrame:
    """
    Merge strategy:
    1. Load daily rolling Amihud (2025, ~252k rows per ticker)
    2. Load monthly free float mkt cap (12 snapshots per ticker)
    3. For each ticker, assign each daily row the free float mkt cap
       from the most recent monthly snapshot (as-of join / forward fill).
    """
    print(f"Loading rolling Amihud from {ROLLING_AMIHUD}...")
    amihud = pd.read_parquet(ROLLING_AMIHUD)
    amihud["date"] = pd.to_datetime(amihud["date"])
    print(f"  {len(amihud):,} rows, {amihud['ticker'].nunique()} tickers")

    print(f"Loading Bloomberg free float from {BBG_FREE_FLOAT}...")
    ff = pd.read_parquet(BBG_FREE_FLOAT)
    ff["date"] = pd.to_datetime(ff["date"])

    # Keep only the columns we need for merging
    ff_slim = ff[["ticker", "date", "cur_mkt_cap", "eqy_free_float_pct", "free_float_mkt_cap"]].copy()
    ff_slim = ff_slim.rename(columns={"date": "ff_date"})
    print(f"  {len(ff_slim):,} records, {ff_slim['ticker'].nunique()} tickers")

    # Normalize datetime resolution to avoid merge_asof dtype mismatch
    amihud["date"] = amihud["date"].astype("datetime64[ns]")
    ff_slim["ff_date"] = ff_slim["ff_date"].astype("datetime64[ns]")

    # As-of merge: for each daily row, get the most recent monthly free float
    print("Performing as-of merge (forward fill monthly → daily)...")

    amihud = amihud.sort_values("date")
    ff_slim = ff_slim.sort_values("ff_date")

    merged = pd.merge_asof(
        amihud,
        ff_slim,
        left_on="date",
        right_on="ff_date",
        by="ticker",
        direction="backward",
    )

    # Report coverage
    total = len(merged)
    has_ff = merged["free_float_mkt_cap"].notna().sum()
    print(f"  Merged: {total:,} rows, {has_ff:,} with free_float_mkt_cap ({has_ff/total*100:.1f}%)")

    return merged


if __name__ == "__main__":
    result = merge_free_float_onto_amihud()

    cols = [
        "ticker", "date", "close", "volume", "dollar_volume",
        "return", "abs_return", "illiq", "illiq_252d",
        "ff_date", "cur_mkt_cap", "eqy_free_float_pct", "free_float_mkt_cap",
    ]
    result = result[cols]

    print(f"\nSaving → {FINAL_MERGED}")
    result.to_parquet(FINAL_MERGED, index=False, engine="pyarrow")

    print(f"\n{'='*60}")
    print(f"  Final Merged Dataset — Amihud + Free Float Mkt Cap")
    print(f"{'='*60}")
    print(f"  Rows:         {len(result):,}")
    print(f"  Tickers:      {result['ticker'].nunique()}")
    print(f"  Date range:   {result['date'].min().date()} → {result['date'].max().date()}")
    print(f"  Columns:      {result.columns.tolist()}")

    ff = result["free_float_mkt_cap"]
    print(f"\n  Free float mkt cap:")
    print(f"    coverage:   {ff.notna().sum():,} / {len(result):,}")
    print(f"    mean:       ${ff.mean():,.0f}")
    print(f"    median:     ${ff.median():,.0f}")

    illiq = result["illiq_252d"]
    print(f"\n  Rolling 252d Amihud:")
    print(f"    coverage:   {illiq.notna().sum():,} / {len(result):,}")
    print(f"    mean:       {illiq.mean():.2e}")
    print(f"    median:     {illiq.median():.2e}")
    print(f"{'='*60}")
