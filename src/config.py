"""Shared configuration — paths, constants, helper utilities."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STAGING_DIR = DATA_DIR / "staging"    # intermediate / transient artifacts
LOGS_DIR = DATA_DIR / "logs"          # log files + progress JSON

# Inputs
OHLCV_2025 = DATA_DIR / "us_equities_2025_ohlcv.parquet"
OHLCV_2024 = DATA_DIR / "us_equities_2024_ohlcv.parquet"
TICKER_FILE = DATA_DIR / "ticker_universe.json"

# Outputs (final pipeline artifacts — consumed by dashboard / analysis)
OHLCV_COMBINED = DATA_DIR / "us_equities_2024_2025_ohlcv.parquet"
ROLLING_AMIHUD = DATA_DIR / "us_equities_2025_rolling_amihud_252d.parquet"
BBG_FREE_FLOAT = DATA_DIR / "bbg_free_float_monthly_2025.parquet"
FINAL_MERGED = DATA_DIR / "amihud_with_free_float.parquet"

# ── Constants ──────────────────────────────────────────────────────────
ROLLING_WINDOW = 252          # 1 year of trading days
ROLLING_MIN_PERIODS = 200     # require at least ~80% of window
PRICE_CUTOFF = 5.0            # exclude penny stocks (avg close < $5)
CHUNK_SIZE = 50               # tickers per yfinance batch

# ── Market Cap Bucket Thresholds (USD, round numbers) ─────────────────
# 6-bucket taxonomy aligned with standard index breakpoints.
# Thresholds are in raw dollars (not millions/billions) for direct
# comparison with cur_mkt_cap column.  Easily tunable via the dashboard.
#
#   Nano     <  $50M
#   Micro    $50M  – $250M
#   Small    $250M – $2B
#   Mid      $2B   – $10B
#   Large    $10B  – $200B
#   Mega     >  $200B

MKTCAP_BUCKETS = {
    "Nano":  (0,             50_000_000),
    "Micro": (50_000_000,    250_000_000),
    "Small": (250_000_000,   2_000_000_000),
    "Mid":   (2_000_000_000, 10_000_000_000),
    "Large": (10_000_000_000, 200_000_000_000),
    "Mega":  (200_000_000_000, float("inf")),
}

# Ordered labels for display / iteration
MKTCAP_BUCKET_ORDER = ["Nano", "Micro", "Small", "Mid", "Large", "Mega"]

# Bloomberg
BBG_HOST = "localhost"
BBG_PORT = 8194

# 2025 monthly first-business-day dates for free float pulls
MONTHLY_DATES_2025 = [
    "2025-01-02", "2025-02-03", "2025-03-03", "2025-04-01",
    "2025-05-01", "2025-06-02", "2025-07-01", "2025-08-01",
    "2025-09-02", "2025-10-01", "2025-11-03", "2025-12-01",
]
