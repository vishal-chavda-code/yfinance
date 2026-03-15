"""
Download 2024 full-year OHLCV data for the same US equity universe.
Provides the 1-year lookback needed for rolling 252-day Amihud starting Jan 2025.

Usage:
    python -m src.download_2024
"""

import yfinance as yf
import pandas as pd
import json
import time
import logging
from pathlib import Path
from src.config import DATA_DIR, TICKER_FILE, OHLCV_2024, PRICE_CUTOFF

RAW_DIR_2024 = DATA_DIR / "raw_chunks_2024"
CHUNK_SIZE = 50
START = "2024-01-01"
END = "2025-01-01"
PROGRESS_FILE = DATA_DIR / "progress_2024.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "download_2024.log", mode="a"),
    ],
)
log = logging.getLogger("yf_2024")


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_chunks": [], "failed_chunks": {}}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def download_chunk(tickers: list[str]) -> pd.DataFrame:
    ticker_str = " ".join(tickers)
    df = yf.download(
        ticker_str, start=START, end=END,
        group_by="ticker", threads=True, progress=False,
    )
    if df.empty:
        return pd.DataFrame()

    records = []
    single = len(tickers) == 1
    for ticker in tickers:
        try:
            if single:
                tdf = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                tdf = df[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
            tdf = tdf.dropna(how="all")
            if len(tdf) < 10:
                continue
            tdf = tdf.reset_index()
            tdf.insert(0, "ticker", ticker)
            tdf.columns = ["ticker", "date", "open", "high", "low", "close", "volume"]
            records.append(tdf)
        except (KeyError, TypeError):
            continue

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def run_downloads(tickers: list[str]):
    RAW_DIR_2024.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    total = len(chunks)
    already_done = sum(1 for i in range(total) if f"chunk_{i:04d}" in progress["completed_chunks"])
    log.info(f"Tickers: {len(tickers)} | Chunks: {total} | Already done: {already_done}")

    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i:04d}"
        if chunk_name in progress["completed_chunks"]:
            continue

        log.info(f"[{i+1}/{total}] {chunk_name} — {len(chunk)} tickers: {chunk[0]}...{chunk[-1]}")
        try:
            df = download_chunk(chunk)
            if not df.empty:
                path = RAW_DIR_2024 / f"{chunk_name}.parquet"
                df.to_parquet(path, index=False, engine="pyarrow")
                log.info(f"  ✓ {len(df):,} rows, {df['ticker'].nunique()} tickers")
            else:
                log.warning(f"  ⚠ No data returned")

            progress["completed_chunks"].append(chunk_name)
            save_progress(progress)
            time.sleep(1.5)
        except Exception as e:
            log.error(f"  ✗ FAILED: {e}")
            progress["failed_chunks"][chunk_name] = {"tickers": chunk, "error": str(e)}
            save_progress(progress)
            time.sleep(5)


def retry_failures():
    progress = load_progress()
    failed = progress.get("failed_chunks", {})
    if not failed:
        log.info("No failed chunks to retry.")
        return

    log.info(f"Retrying {len(failed)} failed chunks...")
    retry_tickers = []
    for info in failed.values():
        retry_tickers.extend(info["tickers"])

    retry_tickers = sorted(set(retry_tickers))
    progress["failed_chunks"] = {}
    save_progress(progress)

    chunks = [retry_tickers[i:i + CHUNK_SIZE] for i in range(0, len(retry_tickers), CHUNK_SIZE)]
    for i, chunk in enumerate(chunks):
        chunk_name = f"retry_{i:04d}"
        log.info(f"  Retry {i+1}/{len(chunks)}: {len(chunk)} tickers")
        try:
            df = download_chunk(chunk)
            if not df.empty:
                path = RAW_DIR_2024 / f"{chunk_name}.parquet"
                df.to_parquet(path, index=False, engine="pyarrow")
                log.info(f"    ✓ {len(df):,} rows")
            progress["completed_chunks"].append(chunk_name)
            save_progress(progress)
            time.sleep(2)
        except Exception as e:
            log.error(f"    ✗ Retry failed: {e}")
            progress["failed_chunks"][chunk_name] = {"tickers": chunk, "error": str(e)}
            save_progress(progress)
            time.sleep(5)


def consolidate_and_filter():
    files = sorted(RAW_DIR_2024.glob("*.parquet"))
    if not files:
        log.error("No parquet files found in raw_chunks_2024/")
        return None

    log.info(f"Consolidating {len(files)} parquet files...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["ticker", "date"])
    log.info(f"Raw total: {len(df):,} rows, {df['ticker'].nunique()} unique tickers")

    # Full-year filter (2024 has ~251 trading days; use 98% threshold)
    days_per_ticker = df.groupby("ticker")["date"].nunique()
    expected_days = int(days_per_ticker.mode().iloc[0])
    min_days = int(expected_days * 0.98)
    full_year = days_per_ticker[days_per_ticker >= min_days].index
    n_before = df["ticker"].nunique()
    df = df[df["ticker"].isin(full_year)].copy()
    log.info(f"Full-year filter (>= {min_days}/{expected_days} days): {n_before} → {df['ticker'].nunique()}")

    # Penny stock filter
    avg_close = df.groupby("ticker")["close"].mean()
    non_penny = avg_close[avg_close >= PRICE_CUTOFF].index
    n_before = df["ticker"].nunique()
    df = df[df["ticker"].isin(non_penny)].copy()
    log.info(f"Penny stock filter (>= ${PRICE_CUTOFF}): {n_before} → {df['ticker'].nunique()}")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    df.to_parquet(OHLCV_2024, index=False, engine="pyarrow")
    log.info(f"Saved: {OHLCV_2024}  ({len(df):,} rows, {df['ticker'].nunique()} tickers)")
    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers = json.loads(TICKER_FILE.read_text())
    log.info(f"Loaded {len(tickers)} tickers from {TICKER_FILE}")

    run_downloads(tickers)
    retry_failures()
    df = consolidate_and_filter()

    if df is not None and not df.empty:
        print(f"\n{'='*60}")
        print(f"  2024 OHLCV Download Complete")
        print(f"  Tickers: {df['ticker'].nunique():,}")
        print(f"  Rows:    {len(df):,}")
        print(f"  Range:   {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"{'='*60}")
