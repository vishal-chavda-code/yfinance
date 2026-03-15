# Amihud ILLIQ vs Bloomberg Free-Float: Liquidity Risk Signals for US Equities

An interactive Plotly Dash dashboard that evaluates whether Bloomberg free-float
data adds predictive power over the classic Amihud (2002) illiquidity measure
for US equity liquidity risk.

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py            # downloads data & computes features
python plotly_dash.py             # open http://127.0.0.1:8050
```

Or in one shot:

```bash
python run_pipeline.py --dashboard
```

## Pipeline

```
 download_ohlcv.py          src.download_2024
 (2025 OHLCV)               (2024 look-back OHLCV)
       |                           |
       +----------+----------------+
                  |
                  v
        src.calc_rolling_amihud
        (252-day rolling Amihud)
                  |
                  v
    src.bbg_free_float ------+   * optional, needs Bloomberg Terminal
                  |          |
                  v          v
            src.merge_data
       (amihud_with_free_float.parquet)
                  |
                  v
           plotly_dash.py
        http://127.0.0.1:8050
```

### Steps

| # | Command                          | Output                                     | Notes              |
|---|----------------------------------|--------------------------------------------|--------------------|
| 1 | `python download_ohlcv.py`       | `data/us_equities_2025_ohlcv.parquet`      |                    |
| 2 | `python -m src.download_2024`    | `data/us_equities_2024_ohlcv.parquet`      |                    |
| 3 | `python -m src.bbg_free_float`   | `data/bbg_ff_*.parquet`                    | Bloomberg Terminal |
| 4 | `python -m src.calc_rolling_amihud` | `data/us_equities_2025_rolling_amihud_252d.parquet` |           |
| 5 | `python -m src.merge_data`       | `data/amihud_with_free_float.parquet`      |                    |
| 6 | `python plotly_dash.py`          | Dashboard on port 8050                     |                    |

> **Bloomberg is optional.** The dashboard runs without it but the free-float
> analysis panels will show reduced data. All Amihud-only chapters work fine.

### Out-of-Sample Validation (optional)

```bash
python -m oos.download_2026
python -m oos.compute_features
python -m oos.train_and_evaluate
```

## Dashboard Chapters

1. **Overview** -- summary statistics and universe coverage
2. **The Problem** -- staleness of free-float snapshots vs daily Amihud
3. **The Evidence** -- R-squared horse race (Amihud vs FF vs combined)
4. **Robustness** -- sub-period and size-bucket stability checks
5. **Tail Risk** -- behaviour during stress episodes
6. **Out-of-Sample** -- 2026 walk-forward results
7. **Interactive Tools** -- scatter plots, histograms, ticker drill-down
8. **Conclusion** -- takeaways and recommendations

## Project Structure

```
yfinance/
|-- download_ohlcv.py          # Step 1: fetch 2025 OHLCV via yfinance
|-- plotly_dash.py              # Step 6: Dash application
|-- run_pipeline.py             # Pipeline orchestrator (idempotent)
|-- Dockerfile
|-- src/
|   |-- config.py               # Paths, constants
|   |-- download_2024.py        # Step 2: 2024 look-back OHLCV
|   |-- bbg_free_float.py       # Step 3: Bloomberg free-float pull
|   |-- calc_rolling_amihud.py  # Step 4: 252d rolling Amihud
|   +-- merge_data.py           # Step 5: as-of merge
|-- oos/                        # Out-of-sample validation
|   |-- download_2026.py
|   |-- compute_features.py
|   +-- train_and_evaluate.py
|-- data/                       # Generated data (git-ignored)
+-- assets/                     # Dash CSS / static files
```

## Docker

```bash
docker build -t yfinance-liquidity .
docker run -p 8050:8050 yfinance-liquidity
```

The Docker image runs the full yfinance-only pipeline at build time (steps 1-2
and 4-5). Bloomberg free-float is skipped since there is no terminal in the
container. The dashboard launches automatically on `docker run`.

## Dependencies

Core: `pandas`, `numpy`, `scikit-learn`, `plotly`, `dash`,
`dash-bootstrap-components`, `yfinance`, `pyarrow`.

Optional: `blpapi` (Bloomberg Terminal must be running locally).

## License

Internal / research use.
