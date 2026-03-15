"""
Pull monthly free float market cap from Bloomberg for all tickers.

For each of the 12 months in 2025, requests CUR_MKT_CAP and EQY_FREE_FLOAT_PCT
on the first business day, then computes:
    free_float_mkt_cap = CUR_MKT_CAP * EQY_FREE_FLOAT_PCT / 100

Requires Bloomberg Terminal running with blpapi access on localhost:8194.

Usage:
    python -m src.bbg_free_float
"""

import blpapi
import pandas as pd
import json
import time
import logging
from pathlib import Path
from src.config import (
    DATA_DIR, TICKER_FILE, BBG_FREE_FLOAT,
    BBG_HOST, BBG_PORT, MONTHLY_DATES_2025,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "bbg_free_float.log", mode="a"),
    ],
)
log = logging.getLogger("bbg_ff")

FIELDS = ["CUR_MKT_CAP", "EQY_FREE_FLOAT_PCT"]
BATCH_SIZE = 50  # Bloomberg typically allows up to ~500 securities per request
PROGRESS_FILE = DATA_DIR / "bbg_free_float_progress.json"


def yf_to_bbg(ticker: str) -> str:
    """Convert yfinance ticker to Bloomberg format."""
    # BRK-B -> BRK/B, then append " US Equity"
    bbg = ticker.replace("-", "/")
    return f"{bbg} US Equity"


def bbg_to_yf(bbg_ticker: str) -> str:
    """Convert Bloomberg ticker back to yfinance format."""
    # Remove " US Equity", then BRK/B -> BRK-B
    base = bbg_ticker.replace(" US Equity", "")
    return base.replace("/", "-")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": {}}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def start_session() -> blpapi.Session:
    options = blpapi.SessionOptions()
    options.setServerHost(BBG_HOST)
    options.setServerPort(BBG_PORT)

    session = blpapi.Session(options)
    if not session.start():
        raise RuntimeError("Failed to start Bloomberg session. Is the Terminal running?")
    if not session.openService("//blp/refdata"):
        session.stop()
        raise RuntimeError("Failed to open //blp/refdata service.")

    log.info("Bloomberg session started.")
    return session


def pull_historical_batch(
    session: blpapi.Session,
    tickers: list[str],
    ref_date: str,
) -> list[dict]:
    """
    Pull CUR_MKT_CAP and EQY_FREE_FLOAT_PCT for a batch of tickers
    on a single reference date using HistoricalDataRequest.
    """
    service = session.getService("//blp/refdata")
    request = service.createRequest("HistoricalDataRequest")

    for t in tickers:
        request.getElement("securities").appendValue(t)
    for f in FIELDS:
        request.getElement("fields").appendValue(f)

    request.set("startDate", ref_date.replace("-", ""))
    request.set("endDate", ref_date.replace("-", ""))
    request.set("periodicitySelection", "DAILY")

    session.sendRequest(request)

    results = []
    done = False
    while not done:
        ev = session.nextEvent(30000)
        for msg in ev:
            if msg.hasElement("securityData"):
                sec_data = msg.getElement("securityData")
                security = sec_data.getElementAsString("security")
                if sec_data.hasElement("fieldData"):
                    field_data = sec_data.getElement("fieldData")
                    if field_data.numValues() > 0:
                        row = field_data.getValueAsElement(0)
                        mkt_cap = None
                        ff_pct = None
                        if row.hasElement("CUR_MKT_CAP"):
                            try:
                                mkt_cap = row.getElementAsFloat("CUR_MKT_CAP")
                            except Exception:
                                pass
                        if row.hasElement("EQY_FREE_FLOAT_PCT"):
                            try:
                                ff_pct = row.getElementAsFloat("EQY_FREE_FLOAT_PCT")
                            except Exception:
                                pass
                        results.append({
                            "bbg_ticker": security,
                            "ticker": bbg_to_yf(security),
                            "date": ref_date,
                            "cur_mkt_cap": mkt_cap,
                            "eqy_free_float_pct": ff_pct,
                        })
        if ev.eventType() == blpapi.Event.RESPONSE:
            done = True
        elif ev.eventType() == blpapi.Event.TIMEOUT:
            log.warning("Timeout waiting for Bloomberg response")
            done = True

    return results


def pull_month(session: blpapi.Session, tickers: list[str], ref_date: str) -> pd.DataFrame:
    """Pull data for all tickers on one reference date, batched."""
    all_results = []
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        log.info(f"  Batch {i+1}/{len(batches)} ({len(batch)} tickers)")
        results = pull_historical_batch(session, batch, ref_date)
        all_results.extend(results)
        time.sleep(0.5)  # rate limit

    df = pd.DataFrame(all_results)
    if not df.empty:
        df["free_float_mkt_cap"] = df["cur_mkt_cap"] * df["eqy_free_float_pct"] / 100.0
    return df


if __name__ == "__main__":
    tickers_yf = json.loads(TICKER_FILE.read_text())
    tickers_bbg = [yf_to_bbg(t) for t in tickers_yf]
    log.info(f"Loaded {len(tickers_bbg)} tickers")

    progress = load_progress()
    session = start_session()

    all_dfs = []
    try:
        for ref_date in MONTHLY_DATES_2025:
            if ref_date in progress["completed"]:
                log.info(f"Skipping {ref_date} (already done)")
                cached = pd.read_parquet(DATA_DIR / f"bbg_ff_{ref_date}.parquet")
                all_dfs.append(cached)
                continue

            log.info(f"Pulling free float for {ref_date}...")
            df = pull_month(session, tickers_bbg, ref_date)

            if not df.empty:
                interim = DATA_DIR / f"bbg_ff_{ref_date}.parquet"
                df.to_parquet(interim, index=False, engine="pyarrow")
                all_dfs.append(df)
                log.info(f"  ✓ {len(df)} records, {df['free_float_mkt_cap'].notna().sum()} with ff_mkt_cap")
            else:
                log.warning(f"  ⚠ No data for {ref_date}")

            progress["completed"][ref_date] = True
            save_progress(progress)
    finally:
        session.stop()
        log.info("Bloomberg session stopped.")

    if all_dfs:
        final = pd.concat(all_dfs, ignore_index=True)
        final["date"] = pd.to_datetime(final["date"])
        final.to_parquet(BBG_FREE_FLOAT, index=False, engine="pyarrow")

        print(f"\n{'='*60}")
        print(f"  Bloomberg Free Float Market Cap — 2025 Monthly")
        print(f"{'='*60}")
        print(f"  Total records:   {len(final):,}")
        print(f"  Unique tickers:  {final['ticker'].nunique()}")
        print(f"  Months:          {final['date'].nunique()}")
        print(f"  ff_mkt_cap coverage: {final['free_float_mkt_cap'].notna().sum():,} / {len(final):,}")
        print(f"{'='*60}")
