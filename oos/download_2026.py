"""
Download 2026 YTD OHLCV data for the same US equity universe used in training.

Uses the existing ticker_universe.json (same universe to ensure comparability).
Downloads Jan 1 2026 → today, applies the same penny-stock and completeness
filters used for the 2025 data.

Usage:
    python -m oos.download_2026
"""

import yfinance as yf
import pandas as pd
import json
import time
import logging
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OOS_DIR = PROJECT_ROOT / "oos" / "data"
RAW_DIR = OOS_DIR / "raw_chunks_2026"
LOGS_DIR = OOS_DIR / "logs"
PROGRESS_FILE = LOGS_DIR / "progress_2026.json"
TICKER_FILE = DATA_DIR / "ticker_universe.json"
OUTPUT_FILE = OOS_DIR / "us_equities_2026_ohlcv.parquet"

CHUNK_SIZE = 50
PRICE_CUTOFF = 5.0
START = "2026-01-01"
END = "2026-03-16"  # exclusive upper bound — captures through Mar 15

for d in [OOS_DIR, RAW_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "download_2026.log", mode="a"),
    ],
)
log = logging.getLogger("yf_2026")


# ── Progress tracking ──────────────────────────────────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_chunks": [], "failed_chunks": {}}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ── Download ───────────────────────────────────────────────────────────
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
            if len(tdf) < 5:
                continue
            tdf = tdf.reset_index()
            tdf.insert(0, "ticker", ticker)
            tdf.columns = ["ticker", "date", "open", "high", "low", "close", "volume"]
            records.append(tdf)
        except (KeyError, TypeError):
            continue
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def run_downloads(tickers: list[str]):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()
    chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    total = len(chunks)
    already = sum(1 for i in range(total) if f"chunk_{i:04d}" in progress["completed_chunks"])
    log.info(f"Tickers: {len(tickers)} | Chunks: {total} | Already done: {already}")

    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i:04d}"
        if chunk_name in progress["completed_chunks"]:
            continue
        log.info(f"[{i+1}/{total}] {chunk_name} — {len(chunk)} tickers: {chunk[0]}...{chunk[-1]}")
        try:
            df = download_chunk(chunk)
            if not df.empty:
                path = RAW_DIR / f"{chunk_name}.parquet"
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
    log.info("Download pass complete.")


def consolidate_and_filter():
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        log.error("No parquet files found")
        return None

    log.info(f"Consolidating {len(files)} parquet files...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["ticker", "date"])
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Raw total: {len(df):,} rows, {df['ticker'].nunique()} tickers")

    # 2026 is partial year — require at least 80% of the mode trading days
    days_per_ticker = df.groupby("ticker")["date"].nunique()
    expected_days = int(days_per_ticker.mode().iloc[0])
    min_days = max(int(expected_days * 0.80), 30)
    full_tickers = days_per_ticker[days_per_ticker >= min_days].index
    n_before = df["ticker"].nunique()
    df = df[df["ticker"].isin(full_tickers)].copy()
    log.info(f"Completeness filter (>= {min_days}/{expected_days} days): {n_before} → {df['ticker'].nunique()}")

    # Penny stock filter
    avg_close = df.groupby("ticker")["close"].mean()
    non_penny = avg_close[avg_close >= PRICE_CUTOFF].index
    n_before = df["ticker"].nunique()
    df = df[df["ticker"].isin(non_penny)].copy()
    log.info(f"Penny stock filter (>= ${PRICE_CUTOFF}): {n_before} → {df['ticker'].nunique()}")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")
    log.info(f"Saved: {OUTPUT_FILE}  ({len(df):,} rows, {df['ticker'].nunique()} tickers)")
    return df


if __name__ == "__main__":
    if not TICKER_FILE.exists():
        log.error(f"Ticker universe not found: {TICKER_FILE}")
        raise SystemExit(1)

    tickers = json.loads(TICKER_FILE.read_text())
    log.info(f"Loaded {len(tickers)} tickers from {TICKER_FILE}")

    run_downloads(tickers)
    consolidate_and_filter()
