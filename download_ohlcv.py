"""
Download 2025 full-year OHLCV data for all US equities (non-penny stocks).
- Fetches ticker universe from NASDAQ screener API (SEC EDGAR fallback)
- Downloads daily OHLCV via yfinance in resumable chunks
- Filters: avg close >= $5, full calendar year 2025 only
- Saves chunked parquet files + consolidated final file
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
import logging
import re
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────
DATA_DIR = Path(r"c:\Users\vmc30\OneDrive\Desktop\Personal_Repos\yfinance\data")
STAGING_DIR = DATA_DIR / "staging"
LOGS_DIR = DATA_DIR / "logs"
RAW_DIR = STAGING_DIR / "raw_chunks"   # intermediate download chunks
CHUNK_SIZE = 50
PRICE_CUTOFF = 5.0
START = "2025-01-01"
END = "2026-01-01"  # exclusive
PROGRESS_FILE = LOGS_DIR / "progress.json"
TICKER_FILE = DATA_DIR / "ticker_universe.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "download.log", mode="a"),
    ],
)
log = logging.getLogger("yf_download")


# ── Step 1: Ticker universe ────────────────────────────────────────────

def fetch_nasdaq_tickers():
    """Fetch US-traded stock tickers from NASDAQ screener API."""
    url = "https://api.nasdaq.com/api/screener/stocks"
    params = {"tableType": "traded", "download": "true"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    log.info("Fetching tickers from NASDAQ screener API...")
    resp = requests.get(url, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    body = resp.json()
    rows = body["data"]["rows"]
    df = pd.DataFrame(rows)
    log.info(f"  NASDAQ API returned {len(df)} symbols")
    return df["symbol"].str.strip().tolist()


def fetch_sec_tickers():
    """Fallback: fetch tickers from SEC EDGAR company_tickers_exchange.json."""
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    headers = {"User-Agent": "PersonalResearch/1.0 contact@example.com"}

    log.info("Fetching tickers from SEC EDGAR (fallback)...")
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    body = resp.json()
    df = pd.DataFrame(body["data"], columns=body["fields"])
    # Filter to major US exchanges
    us_exchanges = {"NYSE", "NASDAQ", "CBOE", "AMEX", "BATS", "Cboe"}
    df = df[df["exchange"].isin(us_exchanges)]
    log.info(f"  SEC EDGAR returned {len(df)} US-exchange symbols")
    return df["ticker"].str.strip().tolist()


def clean_tickers(raw_tickers):
    """Filter out warrants, units, rights, preferred, and junk symbols."""
    clean = []
    # Pattern for valid equity tickers: letters, dots, hyphens; 1-5 primary chars
    valid_pattern = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?(-[A-Z]{1,2})?$")

    for t in raw_tickers:
        t = t.strip().upper()
        if not t:
            continue
        # Replace / with - (NASDAQ uses / for class shares: BRK/A -> BRK-A)
        t = t.replace("/", "-")
        # Skip symbols with special chars
        if not all(c.isalnum() or c in ".-" for c in t):
            continue
        # Skip warrants (5-letter ending in W), units (U), rights (R)
        if len(t) >= 5 and t[-1] in ("W", "U", "R") and t[:-1].isalpha():
            # But don't filter out legit short tickers like "SNOW", "SMFR"
            # Only filter if 5+ chars AND the base (without suffix) is 4+ alpha
            base = t[:-1]
            if len(base) >= 4 and base.isalpha():
                continue
        # Skip preferred stock indicators (ending in -P* or containing ^)
        if "^" in t:
            continue
        if "-P" in t and len(t) > 4:
            continue
        # Basic format check
        if valid_pattern.match(t):
            clean.append(t)

    clean = sorted(set(clean))
    return clean


def get_ticker_universe():
    """Get and cache ticker universe."""
    if TICKER_FILE.exists():
        tickers = json.loads(TICKER_FILE.read_text())
        log.info(f"Loaded {len(tickers)} tickers from cache ({TICKER_FILE})")
        return tickers

    # Try NASDAQ first, fall back to SEC
    try:
        raw = fetch_nasdaq_tickers()
    except Exception as e:
        log.warning(f"NASDAQ API failed ({e}), trying SEC EDGAR...")
        raw = fetch_sec_tickers()

    tickers = clean_tickers(raw)
    TICKER_FILE.write_text(json.dumps(tickers, indent=2))
    log.info(f"Saved {len(tickers)} cleaned tickers to {TICKER_FILE}")
    return tickers


# ── Step 2: Download in chunks ──────────────────────────────────────────

def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_chunks": [], "failed_chunks": {}}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def download_chunk(tickers):
    """Download OHLCV for a batch of tickers. Returns tidy DataFrame."""
    ticker_str = " ".join(tickers)

    df = yf.download(
        ticker_str,
        start=START,
        end=END,
        group_by="ticker",
        threads=True,
        progress=False,
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

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def run_downloads(tickers):
    """Download all tickers in chunks with resume capability."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    chunks = [tickers[i : i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    total = len(chunks)
    already_done = len([c for c in range(total) if f"chunk_{c:04d}" in progress["completed_chunks"]])
    log.info(f"Tickers: {len(tickers)} | Chunks: {total} | Already done: {already_done}")

    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i:04d}"

        if chunk_name in progress["completed_chunks"]:
            continue

        log.info(f"[{i + 1}/{total}] {chunk_name} — {len(chunk)} tickers: {chunk[0]}...{chunk[-1]}")

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
            time.sleep(1.5)  # rate limit

        except Exception as e:
            log.error(f"  ✗ FAILED: {e}")
            progress["failed_chunks"][chunk_name] = {
                "tickers": chunk,
                "error": str(e),
            }
            save_progress(progress)
            time.sleep(5)

    log.info("Download pass complete.")


def retry_failures():
    """Retry any chunks that failed in the initial pass."""
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

    chunks = [retry_tickers[i : i + CHUNK_SIZE] for i in range(0, len(retry_tickers), CHUNK_SIZE)]
    for i, chunk in enumerate(chunks):
        chunk_name = f"retry_{i:04d}"
        log.info(f"  Retry {i + 1}/{len(chunks)}: {len(chunk)} tickers")

        try:
            df = download_chunk(chunk)
            if not df.empty:
                path = RAW_DIR / f"{chunk_name}.parquet"
                df.to_parquet(path, index=False, engine="pyarrow")
                log.info(f"    ✓ {len(df):,} rows")
            progress["completed_chunks"].append(chunk_name)
            save_progress(progress)
            time.sleep(2)
        except Exception as e:
            log.error(f"    ✗ Retry failed: {e}")
            progress["failed_chunks"][chunk_name] = {
                "tickers": chunk,
                "error": str(e),
            }
            save_progress(progress)
            time.sleep(5)


# ── Step 3: Consolidate & filter ────────────────────────────────────────

def consolidate_and_filter():
    """Combine all raw chunks, apply quality filters, save final dataset."""
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        log.error("No parquet files found in raw_chunks/")
        return None

    log.info(f"Consolidating {len(files)} parquet files...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Deduplicate (in case of retry overlap)
    df = df.drop_duplicates(subset=["ticker", "date"])
    log.info(f"Raw total: {len(df):,} rows, {df['ticker'].nunique()} unique tickers")

    # ── Filter 1: Full year only ──
    days_per_ticker = df.groupby("ticker")["date"].nunique()
    expected_days = int(days_per_ticker.mode().iloc[0])
    # Use 98% threshold to allow for minor gaps (e.g. ticker halts)
    min_days = int(expected_days * 0.98)

    full_year_tickers = days_per_ticker[days_per_ticker >= min_days].index
    n_before = df["ticker"].nunique()
    df = df[df["ticker"].isin(full_year_tickers)].copy()
    log.info(
        f"Full-year filter (>= {min_days}/{expected_days} days): "
        f"{n_before} → {df['ticker'].nunique()} tickers"
    )

    # ── Filter 2: Not penny stock (avg close >= $5) ──
    avg_close = df.groupby("ticker")["close"].mean()
    non_penny = avg_close[avg_close >= PRICE_CUTOFF].index
    n_before = df["ticker"].nunique()
    df = df[df["ticker"].isin(non_penny)].copy()
    log.info(
        f"Penny stock filter (avg close >= ${PRICE_CUTOFF}): "
        f"{n_before} → {df['ticker'].nunique()} tickers"
    )

    # Sort and save
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    final_path = DATA_DIR / "us_equities_2025_ohlcv.parquet"
    df.to_parquet(final_path, index=False, engine="pyarrow")
    log.info(f"Final dataset saved: {final_path}")
    log.info(f"  {len(df):,} rows × {df['ticker'].nunique()} tickers")

    return df


# ── Step 4: Completeness report ────────────────────────────────────────

def completeness_report(df):
    """Print detailed completeness statistics."""
    progress = load_progress()
    days = df.groupby("ticker")["date"].nunique()
    avg_close = df.groupby("ticker")["close"].mean()

    print("\n" + "=" * 64)
    print("  COMPLETENESS REPORT — US Equities 2025 OHLCV")
    print("=" * 64)
    print(f"  Tickers:          {df['ticker'].nunique():,}")
    print(f"  Total rows:       {len(df):,}")
    print(f"  Date range:       {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Days/ticker:      min={days.min()}, max={days.max()}, median={days.median():.0f}")
    print(f"  Avg close range:  ${avg_close.min():.2f} — ${avg_close.max():,.2f}")
    print(f"  Close overall:    ${df['close'].mean():.2f} mean")

    nulls = df[["open", "high", "low", "close", "volume"]].isnull().sum()
    has_nulls = nulls.sum() > 0
    print(f"\n  Null values:      {'NONE ✓' if not has_nulls else ''}")
    if has_nulls:
        for col, n in nulls.items():
            if n > 0:
                print(f"    {col}: {n:,}")

    n_completed = len(progress.get("completed_chunks", []))
    n_failed = len(progress.get("failed_chunks", {}))
    print(f"\n  Chunks completed: {n_completed}")
    print(f"  Chunks failed:    {n_failed}")
    if n_failed:
        for name, info in progress["failed_chunks"].items():
            print(f"    {name}: {info['error'][:80]}")

    print("=" * 64)


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ticker universe
    tickers = get_ticker_universe()

    # 2. Download in chunks
    run_downloads(tickers)

    # 3. Retry failures
    retry_failures()

    # 4. Consolidate & filter
    df = consolidate_and_filter()

    # 5. Completeness report
    if df is not None and not df.empty:
        completeness_report(df)
    else:
        log.error("No data to report on.")
