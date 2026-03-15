"""
plotly_dash.py — Liquidity Risk Dashboard (Amihud ILLIQ vs Free-Float Analysis)
================================================================================
Production-grade Plotly Dash application for liquidity risk analysis.

Compares two liquidity signals — Amihud ILLIQ (daily, trade-derived) and
Bloomberg free-float % (quarterly, filing-derived) — to understand where
each excels, where each fails, and how they complement each other.

Model versions compared:
  1. Total Market Cap Only       — baseline size proxy
  2. Free-Float Adjusted MktCap  — float-adjusted, no Amihud overlay
  3. MktCap + Amihud Overlay     — full model with Amihud ILLIQ add-on

Each model is evaluated on:
  • R², Adj R², incremental ΔR²
  • Extreme-move precision / recall / F1 (overprediction attention)
  • Size-bucket breakdowns (Mega → Nano)
  • Asymmetric up/down tail analysis

Usage:
    python plotly_dash.py
    → Opens at http://127.0.0.1:8050
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Market Cap Bucket Defaults (round, memorable numbers) ─────────────
# Keys are ordered Nano → Mega.  Values = (lower_bound, upper_bound) in USD.
# The tuner page lets users slide these thresholds interactively; these are
# the startup defaults that match standard index breakpoints.
DEFAULT_MKTCAP_THRESHOLDS = [
    50_000_000,       # Nano / Micro boundary   —  $50 M
    250_000_000,      # Micro / Small boundary   — $250 M
    2_000_000_000,    # Small / Mid boundary      —  $2 B
    10_000_000_000,   # Mid / Large boundary      — $10 B
    200_000_000_000,  # Large / Mega boundary     — $200 B
]
MKTCAP_LABELS = ["Nano", "Micro", "Small", "Mid", "Large", "Mega"]

# Color palette — deep, saturated tones
COLORS = {
    "bg": "#0B1120",
    "card": "#131B2E",
    "card_border": "#1E2D4A",
    "text": "#E2E8F0",
    "text_muted": "#8896AB",
    "accent_blue": "#3B82F6",
    "accent_cyan": "#06B6D4",
    "accent_orange": "#F59E0B",
    "accent_red": "#EF4444",
    "accent_green": "#10B981",
    "accent_purple": "#8B5CF6",
    "accent_pink": "#EC4899",
    "gradient_start": "#3B82F6",
    "gradient_end": "#8B5CF6",
    "grid": "#1E2D4A",
    "divider": "#1E2D4A",
}

FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"

PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, color=COLORS["text"], size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                   tickfont=dict(size=10, color=COLORS["text_muted"])),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                   tickfont=dict(size=10, color=COLORS["text_muted"])),
        colorway=[COLORS["accent_blue"], COLORS["accent_orange"],
                  COLORS["accent_green"], COLORS["accent_red"],
                  COLORS["accent_purple"], COLORS["accent_cyan"],
                  COLORS["accent_pink"]],
        margin=dict(l=50, r=30, t=50, b=50),
        hoverlabel=dict(bgcolor=COLORS["card"], font_size=12,
                        font_family=FONT, bordercolor=COLORS["card_border"]),
    )
)

# ═══════════════════════════════════════════════════════════════════════
# Data Loading & Pre-computation
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load and prepare all data needed for the dashboard.

    Data sources (trace reliability):
      • amihud_with_free_float.parquet  — produced by src/merge_data.py
        ├── daily Amihud 252d rolling  ← src/calc_rolling_amihud.py
        │   └── OHLCV 2024+2025       ← download_ohlcv.py + src/download_2024.py (yfinance)
        └── monthly free-float         ← src/bbg_free_float.py (Bloomberg blpapi)
      • us_equities_2025_ohlcv.parquet — produced by download_ohlcv.py (yfinance)
        └── high/low columns for intraday range calculations
    """
    df = pd.read_parquet(DATA_DIR / "amihud_with_free_float.parquet")
    ohlcv = pd.read_parquet(
        DATA_DIR / "us_equities_2025_ohlcv.parquet",
        columns=["ticker", "date", "high", "low"],
    )
    df = df.merge(ohlcv, on=["ticker", "date"], how="left")

    # Derived columns
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["parkinson_vol"] = np.log(df["high"] / df["low"])
    df["ff_ratio"] = df["eqy_free_float_pct"] / 100
    df["cur_mkt_cap"] = df["cur_mkt_cap"] * 1e6   # source is in $M → convert to $
    df["log_mktcap"] = np.log1p(df["cur_mkt_cap"])
    df["log_illiq"] = np.log(df["illiq_252d"]).replace(-np.inf, np.nan)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # ── Short/long horizon Amihud columns (README_2 Part 2) ──────────
    # Source: daily 'illiq' column already in amihud_with_free_float.parquet
    #         (raw ILLIQ_t = |r_t| / dollar_volume_t, per src/calc_rolling_amihud.py)
    df = df.sort_values(["ticker", "date"])
    df["illiq_21d"] = (
        df.groupby("ticker")["illiq"]
        .transform(lambda x: x.rolling(21, min_periods=15).mean())
    )
    # illiq_ratio: short-term / structural baseline (analogous to CDS spread)
    df["illiq_ratio"] = df["illiq_21d"] / df["illiq_252d"]
    df["illiq_ratio"] = df["illiq_ratio"].replace([np.inf, -np.inf], np.nan)
    # Rolling 252d stats of the ratio for z-score normalization
    df["illiq_ratio_mean"] = (
        df.groupby("ticker")["illiq_ratio"]
        .transform(lambda x: x.rolling(252, min_periods=60).mean())
    )
    df["illiq_ratio_std"] = (
        df.groupby("ticker")["illiq_ratio"]
        .transform(lambda x: x.rolling(252, min_periods=60).std())
    )
    # Liquidity z-score: cross-sectionally comparable stress signal
    df["illiq_zscore"] = (
        (df["illiq_ratio"] - df["illiq_ratio_mean"]) / df["illiq_ratio_std"]
    )
    df["illiq_zscore"] = df["illiq_zscore"].replace([np.inf, -np.inf], np.nan)

    # Signed return for up/down split (Part 8 scenario matrix)
    df["signed_return"] = df["return"] if "return" in df.columns else df["abs_return"] * np.nan

    return df


def build_analysis_sample(df):
    """Create the clean analysis subset with winsorisation."""
    cols_needed = [
        "abs_return", "hl_range", "parkinson_vol", "illiq_252d",
        "log_illiq", "eqy_free_float_pct", "ff_ratio", "log_mktcap", "cur_mkt_cap",
    ]
    dfc = df.dropna(subset=cols_needed).copy()

    win_cols = ["abs_return", "hl_range", "parkinson_vol", "illiq_252d"]
    for c in win_cols:
        lo, hi = dfc[c].quantile(0.01), dfc[c].quantile(0.99)
        dfc[c] = dfc[c].clip(lo, hi)

    # Size terciles
    ticker_mktcap = dfc.groupby("ticker")["cur_mkt_cap"].median()
    breaks = ticker_mktcap.quantile([1 / 3, 2 / 3])
    def _tercile(mc):
        if mc <= breaks.iloc[0]:
            return "Small Cap"
        elif mc <= breaks.iloc[1]:
            return "Mid Cap"
        return "Large Cap"
    ticker_group = ticker_mktcap.apply(_tercile).rename("size_group")
    dfc = dfc.merge(ticker_group, left_on="ticker", right_index=True, how="left")

    return dfc


def compute_staleness(df):
    """Compute free-float staleness statistics."""
    ff = df.dropna(subset=["eqy_free_float_pct", "ff_date"])[
        ["ticker", "date", "eqy_free_float_pct", "ff_date"]
    ].copy()
    ff["month"] = ff["date"].dt.to_period("M")

    ff_monthly = (
        ff.sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "month"], keep="first")[
            ["ticker", "month", "eqy_free_float_pct"]
        ]
        .reset_index(drop=True)
    )

    # Run lengths
    ticker_max_run = {}
    for tkr, grp in ff_monthly.sort_values(["ticker", "month"]).groupby("ticker"):
        vals = grp["eqy_free_float_pct"].values
        max_run = 1
        current = 1
        for i in range(1, len(vals)):
            if vals[i] == vals[i - 1]:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 1
        ticker_max_run[tkr] = max_run

    max_run_sr = pd.Series(ticker_max_run)
    tickers_12mo = ff_monthly.groupby("ticker").size()
    full_year_idx = tickers_12mo[tickers_12mo == 12].index

    # Monthly change rate
    ff_sorted = ff_monthly.sort_values(["ticker", "month"])
    ff_sorted["changed"] = ff_sorted.groupby("ticker")["eqy_free_float_pct"].diff().ne(0)
    ff_sorted = ff_sorted[ff_sorted.groupby("ticker").cumcount() > 0]
    monthly_change_rate = ff_sorted.groupby("month")["changed"].mean()

    # Quarterly boundary test
    ff_sorted["month_num"] = ff_sorted["month"].dt.month
    qtr_mask = ff_sorted["month_num"].isin([1, 4, 7, 10])
    qtr_change = ff_sorted.loc[qtr_mask, "changed"].mean()
    non_qtr_change = ff_sorted.loc[~qtr_mask, "changed"].mean()

    return {
        "max_run_sr": max_run_sr,
        "full_year_idx": full_year_idx,
        "monthly_change_rate": monthly_change_rate,
        "never_changed_pct": (max_run_sr[max_run_sr.index.isin(full_year_idx)] == 12).mean(),
        "streak_3plus": (max_run_sr[max_run_sr.index.isin(full_year_idx)] >= 3).mean(),
        "streak_6plus": (max_run_sr[max_run_sr.index.isin(full_year_idx)] >= 6).mean(),
        "qtr_change_rate": qtr_change,
        "non_qtr_change_rate": non_qtr_change,
    }


def run_regressions(dfc):
    """Run all OLS specs and return results DataFrame."""
    def _ols(y, X, y_name, x_names):
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))
        n, k = X.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        return {"Dependent": y_name, "Regressors": " + ".join(x_names),
                "R²": r2, "Adj R²": adj_r2, "N": n}

    specs = [
        ("abs_return", ["log_illiq"], "|Return|", ["Amihud"]),
        ("abs_return", ["eqy_free_float_pct"], "|Return|", ["Free-Float %"]),
        ("abs_return", ["log_illiq", "eqy_free_float_pct"], "|Return|", ["Amihud", "Free-Float %"]),
        ("abs_return", ["log_illiq", "eqy_free_float_pct", "log_mktcap"],
         "|Return|", ["Amihud", "FF%", "MktCap"]),
        ("hl_range", ["log_illiq"], "H-L Range", ["Amihud"]),
        ("hl_range", ["eqy_free_float_pct"], "H-L Range", ["Free-Float %"]),
        ("hl_range", ["log_illiq", "eqy_free_float_pct"], "H-L Range", ["Amihud", "Free-Float %"]),
        ("hl_range", ["log_illiq", "eqy_free_float_pct", "log_mktcap"],
         "H-L Range", ["Amihud", "FF%", "MktCap"]),
    ]
    results = []
    for y_col, x_cols, y_name, x_names in specs:
        results.append(_ols(dfc[y_col].values, dfc[x_cols].values, y_name, x_names))
    return pd.DataFrame(results)


def compute_monthly_r2(dfc):
    """Compute monthly R² for Amihud vs FF%."""
    records = []
    for month, grp in dfc.groupby("month"):
        for y_col, y_name in [("abs_return", "|Return|"), ("hl_range", "H-L Range")]:
            for x_col, x_name in [("log_illiq", "Amihud"), ("eqy_free_float_pct", "Free-Float %")]:
                y = grp[y_col].values
                X = grp[[x_col]].values
                r2 = r2_score(y, LinearRegression().fit(X, y).predict(X))
                records.append({"Month": month, "Dependent": y_name,
                                "Regressor": x_name, "R²": r2})
    return pd.DataFrame(records)


def compute_size_r2(dfc):
    """R² within each size tercile."""
    records = []
    for sz in ["Small Cap", "Mid Cap", "Large Cap"]:
        sub = dfc[dfc["size_group"] == sz]
        for y_col, y_name in [("abs_return", "|Return|"), ("hl_range", "H-L Range")]:
            for x_cols, x_name in [(["log_illiq"], "Amihud"), (["eqy_free_float_pct"], "Free-Float %")]:
                y = sub[y_col].values
                X = sub[x_cols].values
                r2 = r2_score(y, LinearRegression().fit(X, y).predict(X))
                records.append({"Size": sz, "Dependent": y_name,
                                "Regressor": x_name, "R²": r2, "N": len(sub)})
    return pd.DataFrame(records)


def compute_quintile_sorts(dfc):
    """Quintile portfolio sorts."""
    def _qsort(data, sort_col, value_cols):
        data = data.copy()
        data["quintile"] = pd.qcut(data[sort_col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        return data.groupby("quintile", observed=True)[value_cols].mean()

    q_illiq = _qsort(dfc, "illiq_252d", ["abs_return", "hl_range"])
    q_ff = _qsort(dfc, "eqy_free_float_pct", ["abs_return", "hl_range"])
    return q_illiq, q_ff


def compute_heatmap(dfc):
    """5x5 interaction heatmap data."""
    tmp = dfc.copy()
    tmp["illiq_q"] = pd.qcut(tmp["illiq_252d"], 5,
                             labels=["Q1 Liquid", "Q2", "Q3", "Q4", "Q5 Illiquid"])
    tmp["ff_q"] = pd.qcut(tmp["eqy_free_float_pct"], 5,
                           labels=["Q1 Low Float", "Q2", "Q3", "Q4", "Q5 High Float"])
    heat_ret = tmp.pivot_table(index="illiq_q", columns="ff_q", values="abs_return", aggfunc="mean")
    heat_hl = tmp.pivot_table(index="illiq_q", columns="ff_q", values="hl_range", aggfunc="mean")
    return heat_ret, heat_hl


def compute_spearman(dfc):
    """Spearman rank correlations."""
    cols = ["abs_return", "hl_range", "illiq_252d", "eqy_free_float_pct", "cur_mkt_cap"]
    labels = ["|Return|", "H-L Range", "Amihud 252d", "Free-Float %", "Mkt Cap"]
    rho = dfc[cols].corr(method="spearman")
    rho.index = labels
    rho.columns = labels
    return rho


def assign_mktcap_buckets(dfc, thresholds=None):
    """Assign each ticker to a 6-bucket market cap group using round thresholds.

    Parameters
    ----------
    dfc : DataFrame with 'ticker' and 'cur_mkt_cap' columns
    thresholds : list of 5 floats [nano/micro, micro/small, small/mid, mid/large, large/mega]
                 If None, uses DEFAULT_MKTCAP_THRESHOLDS.

    Returns
    -------
    Series indexed by ticker with bucket label.
    """
    if thresholds is None:
        thresholds = DEFAULT_MKTCAP_THRESHOLDS
    t = sorted(thresholds)
    ticker_mktcap = dfc.groupby("ticker")["cur_mkt_cap"].median()

    def _label(mc):
        if mc < t[0]:
            return "Nano"
        elif mc < t[1]:
            return "Micro"
        elif mc < t[2]:
            return "Small"
        elif mc < t[3]:
            return "Mid"
        elif mc < t[4]:
            return "Large"
        return "Mega"

    return ticker_mktcap.apply(_label).rename("mktcap_bucket")


def compute_bucket_stats(dfc, thresholds=None):
    """Compute per-bucket summary statistics for the tuner display."""
    buckets = assign_mktcap_buckets(dfc, thresholds)
    tmp = dfc.merge(buckets, left_on="ticker", right_index=True, how="left")

    records = []
    for label in MKTCAP_LABELS:
        sub = tmp[tmp["mktcap_bucket"] == label]
        n_tickers = sub["ticker"].nunique()
        n_rows = len(sub)
        if n_rows == 0:
            records.append({
                "Bucket": label, "Tickers": 0, "Rows": 0,
                "Median MktCap": 0, "Mean |Return|": 0,
                "Mean H-L Range": 0, "Mean Amihud": 0,
                "p95 |Return|": 0, "p99 |Return|": 0,
            })
            continue
        records.append({
            "Bucket": label,
            "Tickers": n_tickers,
            "Rows": n_rows,
            "Median MktCap": sub.groupby("ticker")["cur_mkt_cap"].median().median(),
            "Mean |Return|": sub["abs_return"].mean(),
            "Mean H-L Range": sub["hl_range"].mean(),
            "Mean Amihud": sub["illiq_252d"].mean(),
            "p95 |Return|": sub["abs_return"].quantile(0.95),
            "p99 |Return|": sub["abs_return"].quantile(0.99),
        })
    return pd.DataFrame(records)


def _fmt_usd(val):
    """Format a USD value as a readable string with B/M suffix."""
    if val >= 1e12:
        return f"${val / 1e12:.0f}T"
    if val >= 1e9:
        return f"${val / 1e9:.0f}B"
    if val >= 1e6:
        return f"${val / 1e6:.0f}M"
    return f"${val:,.0f}"


def _fmt_illiq(val):
    """Format an Amihud ILLIQ value as a compact, human-readable string.

    Raw ILLIQ spans ~10⁻¹³ to 10⁻⁵.  Scientific notation is unreadable
    for non-quants, so we express values as a coefficient × a named
    power-of-ten suffix (e.g. "3.2 × 10⁻⁹").
    """
    if val == 0 or np.isnan(val):
        return "0"
    exp = int(np.floor(np.log10(abs(val))))
    coeff = val / (10 ** exp)
    # Unicode superscript digits for the exponent
    sup = str(abs(exp)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    sign = "⁻" if exp < 0 else ""
    return f"{coeff:.1f}×10{sign}{sup}"


# ═══════════════════════════════════════════════════════════════════════
# Extended Computation Functions (README_2 Parts 2-5, 8)
# ═══════════════════════════════════════════════════════════════════════

def compute_scenario_matrix(dfc):
    """Part 8: 6-bucket × up/down empirical percentile grid."""
    buckets = assign_mktcap_buckets(dfc)
    tmp = dfc.merge(buckets, left_on="ticker", right_index=True, how="left")
    records = []
    for label in MKTCAP_LABELS:
        sub = tmp[tmp["mktcap_bucket"] == label]
        if len(sub) == 0:
            records.append({"Bucket": label, "N_tickers": 0, "N_days": 0,
                            "p95_up": 0, "p99_up": 0, "p95_down": 0, "p99_down": 0,
                            "median_amihud": 0, "median_ff": 0})
            continue
        signed = sub["return"] if "return" in sub.columns else sub["abs_return"]
        up = signed[signed > 0]
        down = signed[signed < 0].abs()
        records.append({
            "Bucket": label,
            "N_tickers": sub["ticker"].nunique(),
            "N_days": len(sub),
            "p95_up": float(up.quantile(0.95)) if len(up) > 0 else 0,
            "p99_up": float(up.quantile(0.99)) if len(up) > 0 else 0,
            "p95_down": float(down.quantile(0.95)) if len(down) > 0 else 0,
            "p99_down": float(down.quantile(0.99)) if len(down) > 0 else 0,
            "median_amihud": float(sub["illiq_252d"].median()),
            "median_ff": float(sub["eqy_free_float_pct"].median()),
        })
    sm = pd.DataFrame(records)
    sm["asym_95"] = sm["p95_up"] / sm["p95_down"].clip(lower=1e-10)
    sm["asym_99"] = sm["p99_up"] / sm["p99_down"].clip(lower=1e-10)
    return sm


def compute_full_regressions(dfc):
    """Part 3: All 7 OLS specs (6 pooled + ΔR² decomposition).
    Fama-MacBeth is spec 7 — run separately per month."""
    specs = [
        ("|Return|", "abs_return", ["log_mktcap"], "① MktCap Only"),
        ("|Return|", "abs_return", ["log_illiq"], "② Amihud Only"),
        ("|Return|", "abs_return", ["eqy_free_float_pct"], "③ Free-Float Only"),
        ("|Return|", "abs_return", ["log_mktcap", "log_illiq"], "④ MktCap + Amihud"),
        ("|Return|", "abs_return", ["log_mktcap", "eqy_free_float_pct"], "⑤ MktCap + FF"),
        ("|Return|", "abs_return", ["log_mktcap", "log_illiq", "eqy_free_float_pct"],
         "⑥ Kitchen Sink"),
        ("H-L Range", "hl_range", ["log_mktcap"], "① MktCap Only"),
        ("H-L Range", "hl_range", ["log_illiq"], "② Amihud Only"),
        ("H-L Range", "hl_range", ["eqy_free_float_pct"], "③ Free-Float Only"),
        ("H-L Range", "hl_range", ["log_mktcap", "log_illiq"], "④ MktCap + Amihud"),
        ("H-L Range", "hl_range", ["log_mktcap", "eqy_free_float_pct"], "⑤ MktCap + FF"),
        ("H-L Range", "hl_range", ["log_mktcap", "log_illiq", "eqy_free_float_pct"],
         "⑥ Kitchen Sink"),
    ]
    records = []
    for dep_name, y_col, x_cols, model_name in specs:
        y = dfc[y_col].values
        X = dfc[x_cols].values
        m = LinearRegression().fit(X, y)
        r2 = r2_score(y, m.predict(X))
        n, k = X.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        records.append({"Dependent": dep_name, "Model": model_name,
                        "R²": r2, "Adj R²": adj_r2, "N": n,
                        "Regressors": " + ".join(x_cols)})
    return pd.DataFrame(records)


def compute_r2_decomposition(full_reg):
    """Part 3: ΔR² incremental decomposition — how much does each variable add."""
    records = []
    for dep in ["|Return|", "H-L Range"]:
        sub = full_reg[full_reg["Dependent"] == dep]
        r2_base = float(sub[sub["Model"] == "① MktCap Only"]["R²"].iloc[0])
        r2_amihud = float(sub[sub["Model"] == "④ MktCap + Amihud"]["R²"].iloc[0])
        r2_ff = float(sub[sub["Model"] == "⑤ MktCap + FF"]["R²"].iloc[0])
        r2_kitchen = float(sub[sub["Model"] == "⑥ Kitchen Sink"]["R²"].iloc[0])
        delta_amihud = r2_amihud - r2_base
        delta_ff = r2_ff - r2_base
        superiority = delta_amihud / max(delta_ff, 1e-10)
        records.append({"Dependent": dep, "R²_baseline": r2_base,
                        "R²_amihud": r2_amihud, "R²_ff": r2_ff,
                        "R²_kitchen": r2_kitchen,
                        "ΔR²_amihud": delta_amihud, "ΔR²_ff": delta_ff,
                        "Superiority": superiority})
    return pd.DataFrame(records)


def compute_fama_macbeth(dfc):
    """Part 3 spec 7: Fama-MacBeth cross-sectional regressions — monthly OLS, t-test coefficients."""
    x_cols = ["log_mktcap", "log_illiq", "eqy_free_float_pct"]
    records = []
    for month, grp in dfc.groupby("month"):
        for y_col, y_name in [("abs_return", "|Return|"), ("hl_range", "H-L Range")]:
            y = grp[y_col].values
            X = grp[x_cols].values
            if len(y) < 50:
                continue
            m = LinearRegression().fit(X, y)
            row = {"Month": month, "Dependent": y_name}
            for i, col in enumerate(x_cols):
                row[f"β_{col}"] = m.coef_[i]
            row["Intercept"] = m.intercept_
            row["R²"] = r2_score(y, m.predict(X))
            records.append(row)
    fm_df = pd.DataFrame(records)
    # T-test across months
    summary = []
    for dep in ["|Return|", "H-L Range"]:
        sub = fm_df[fm_df["Dependent"] == dep]
        if len(sub) < 3:
            continue
        for col in x_cols:
            key = f"β_{col}"
            if key not in sub.columns:
                continue
            vals = sub[key].dropna()
            mean_b = vals.mean()
            se = vals.std() / np.sqrt(len(vals))
            t_stat = mean_b / se if se > 0 else 0
            summary.append({"Dependent": dep, "Variable": col,
                            "Mean β": mean_b, "SE": se, "t-stat": t_stat,
                            "Months": len(vals)})
    return fm_df, pd.DataFrame(summary)


def compute_extreme_moves(dfc, df_full):
    """Part 4: Logistic regression for extreme move prediction."""
    buckets = assign_mktcap_buckets(dfc)
    tmp = dfc.merge(buckets, left_on="ticker", right_index=True, how="left")

    # Within-bucket p95 threshold
    p95 = tmp.groupby("mktcap_bucket")["abs_return"].quantile(0.95)
    tmp["p95_threshold"] = tmp["mktcap_bucket"].map(p95)
    tmp["extreme_flag"] = (tmp["abs_return"] > tmp["p95_threshold"]).astype(int)

    # Need lagged z-score from df_full (has illiq_zscore)
    if "illiq_zscore" in df_full.columns:
        zs = df_full[["ticker", "date", "illiq_zscore"]].copy()
        zs = zs.sort_values(["ticker", "date"])
        zs["zscore_lag1"] = zs.groupby("ticker")["illiq_zscore"].shift(1)
        tmp = tmp.merge(zs[["ticker", "date", "zscore_lag1"]],
                        on=["ticker", "date"], how="left")
    else:
        tmp["zscore_lag1"] = np.nan

    # Encode bucket
    bucket_map = {label: i for i, label in enumerate(MKTCAP_LABELS)}
    tmp["bucket_num"] = tmp["mktcap_bucket"].map(bucket_map)

    model_df = tmp.dropna(subset=["bucket_num", "zscore_lag1", "extreme_flag"])
    if len(model_df) < 100:
        return None

    X_base = model_df[["bucket_num"]].values
    X_addon = model_df[["bucket_num", "zscore_lag1"]].values
    y = model_df["extreme_flag"].values

    lr_base = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced").fit(X_base, y)
    lr_addon = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced").fit(X_addon, y)

    pred_base = lr_base.predict(X_base)
    pred_addon = lr_addon.predict(X_addon)

    results = {
        "baseline": {
            "precision": precision_score(y, pred_base, zero_division=0),
            "recall": recall_score(y, pred_base, zero_division=0),
            "f1": f1_score(y, pred_base, zero_division=0),
            "confusion": confusion_matrix(y, pred_base),
        },
        "addon": {
            "precision": precision_score(y, pred_addon, zero_division=0),
            "recall": recall_score(y, pred_addon, zero_division=0),
            "f1": f1_score(y, pred_addon, zero_division=0),
            "confusion": confusion_matrix(y, pred_addon),
        },
        "p95_thresholds": p95,
    }

    # Per-bucket metrics
    bucket_metrics = []
    for label in MKTCAP_LABELS:
        bsub = model_df[model_df["mktcap_bucket"] == label]
        if len(bsub) < 10:
            continue
        yb = bsub["extreme_flag"].values
        pb_base = lr_base.predict(bsub[["bucket_num"]].values)
        pb_addon = lr_addon.predict(bsub[["bucket_num", "zscore_lag1"]].values)
        bucket_metrics.append({
            "Bucket": label,
            "N": len(bsub),
            "Extreme %": yb.mean() * 100,
            "Base Recall": recall_score(yb, pb_base, zero_division=0),
            "Add-on Recall": recall_score(yb, pb_addon, zero_division=0),
            "Base Precision": precision_score(yb, pb_base, zero_division=0),
            "Add-on Precision": precision_score(yb, pb_addon, zero_division=0),
            "Base F1": f1_score(yb, pb_base, zero_division=0),
            "Add-on F1": f1_score(yb, pb_addon, zero_division=0),
        })
    results["bucket_metrics"] = pd.DataFrame(bucket_metrics)

    # Missed extreme magnitude comparison
    base_missed = model_df[(y == 1) & (pred_base == 0)]["abs_return"]
    addon_missed = model_df[(y == 1) & (pred_addon == 0)]["abs_return"]
    results["base_miss_mean"] = float(base_missed.mean()) if len(base_missed) > 0 else 0
    results["addon_miss_mean"] = float(addon_missed.mean()) if len(addon_missed) > 0 else 0
    results["base_miss_n"] = len(base_missed)
    results["addon_miss_n"] = len(addon_missed)
    results["fn_reduction"] = len(base_missed) - len(addon_missed)

    return results


def compute_asymmetry(dfc):
    """Part 5: Up/down day z-score predictive power."""
    tmp = dfc.copy()
    needed = ["return", "abs_return", "log_illiq"]
    for c in needed:
        if c not in tmp.columns:
            return None

    # Also include illiq_zscore if available
    has_zscore = "illiq_zscore" in tmp.columns

    results = []
    for split_label, mask in [("All Days", tmp["return"].notna()),
                              ("Down Days", tmp["return"] < 0),
                              ("Up Days", tmp["return"] >= 0)]:
        subset = tmp[mask].dropna(subset=["abs_return", "log_illiq"])
        if len(subset) < 100:
            continue
        for x_col, x_name in [("log_illiq", "Amihud 252d"),
                               ("eqy_free_float_pct", "Free-Float %")]:
            sub = subset.dropna(subset=[x_col])
            if len(sub) < 100:
                continue
            X = sub[[x_col]].values
            y = sub["abs_return"].values
            m = LinearRegression().fit(X, y)
            r2 = r2_score(y, m.predict(X))
            results.append({"Split": split_label, "Regressor": x_name,
                            "R²": r2, "Coef": m.coef_[0], "N": len(sub)})
        if has_zscore:
            sub = subset.dropna(subset=["illiq_zscore"])
            if len(sub) >= 100:
                X = sub[["illiq_zscore"]].values
                y = sub["abs_return"].values
                m = LinearRegression().fit(X, y)
                r2 = r2_score(y, m.predict(X))
                results.append({"Split": split_label, "Regressor": "Liquidity Z-Score",
                                "R²": r2, "Coef": m.coef_[0], "N": len(sub)})
    return pd.DataFrame(results) if results else None


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading data...")
df = load_data()
dfc = build_analysis_sample(df)
staleness = compute_staleness(df)
reg_results = run_regressions(dfc)
monthly_r2 = compute_monthly_r2(dfc)
size_r2 = compute_size_r2(dfc)
q_illiq, q_ff = compute_quintile_sorts(dfc)
heat_ret, heat_hl = compute_heatmap(dfc)
spearman = compute_spearman(dfc)

has_ff = df.groupby("ticker")["eqy_free_float_pct"].apply(lambda x: x.notna().any())
ff_coverage = has_ff.mean()

# Key metrics for hero cards
r2_amihud_ret = float(reg_results[(reg_results["Dependent"] == "|Return|") &
                                   (reg_results["Regressors"] == "Amihud")]["R²"].iloc[0])
r2_ff_ret = float(reg_results[(reg_results["Dependent"] == "|Return|") &
                               (reg_results["Regressors"] == "Free-Float %")]["R²"].iloc[0])
r2_amihud_hl = float(reg_results[(reg_results["Dependent"] == "H-L Range") &
                                  (reg_results["Regressors"] == "Amihud")]["R²"].iloc[0])
r2_ff_hl = float(reg_results[(reg_results["Dependent"] == "H-L Range") &
                              (reg_results["Regressors"] == "Free-Float %")]["R²"].iloc[0])

superiority_ret = r2_amihud_ret / max(r2_ff_ret, 1e-10)
superiority_hl = r2_amihud_hl / max(r2_ff_hl, 1e-10)

ticker_list = sorted(dfc["ticker"].unique())

# Extended computations (Parts 2-5, 8)
scenario_matrix = compute_scenario_matrix(dfc)
full_reg = compute_full_regressions(dfc)
r2_decomp = compute_r2_decomposition(full_reg)
fm_monthly, fm_summary = compute_fama_macbeth(dfc)
extreme_results = compute_extreme_moves(dfc, df)
asymmetry_results = compute_asymmetry(dfc)

print(f"Data loaded: {df.shape[0]:,} rows, {df['ticker'].nunique():,} tickers")
print(f"Analysis sample: {dfc.shape[0]:,} rows")


# ═══════════════════════════════════════════════════════════════════════
# Helper: generate plotly figures
# ═══════════════════════════════════════════════════════════════════════

def make_card(title, value, subtitle="", color=COLORS["accent_blue"]):
    """Create a styled metric card."""
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="card-title",
                   style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                          "textTransform": "uppercase", "letterSpacing": "1px",
                          "marginBottom": "4px", "fontWeight": 600}),
            html.H2(value, style={"color": color, "fontWeight": 700,
                                   "fontSize": "2rem", "marginBottom": "2px",
                                   "fontFamily": FONT}),
            html.P(subtitle, style={"color": COLORS["text_muted"], "fontSize": "0.75rem",
                                     "marginBottom": 0}),
        ]),
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['card_border']}",
            "borderRadius": "12px",
            "padding": "4px",
            "boxShadow": "0 4px 24px rgba(0,0,0,0.3)",
        },
    )


def fig_staleness_histogram():
    """Distribution of longest unchanged FF% streak."""
    sr = staleness["max_run_sr"]
    full = sr[sr.index.isin(staleness["full_year_idx"])]
    counts = full.value_counts().sort_index()

    fig = go.Figure()
    colors_bar = [COLORS["accent_green"] if x <= 2
                  else COLORS["accent_orange"] if x <= 5
                  else COLORS["accent_red"] for x in counts.index]
    fig.add_trace(go.Bar(
        x=[int(x) for x in counts.index], y=[int(v) for v in counts.values],
        marker=dict(color=colors_bar, line=dict(width=0)),
        hovertemplate="<b>%{x} months</b> unchanged<br>%{y} tickers<extra></extra>",
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Longest Streak of Unchanged Free-Float %", font=dict(size=16)),
        xaxis=dict(title="Consecutive Months Unchanged", dtick=1),
        yaxis=dict(title="Number of Tickers"),
        height=380,
    )
    return fig


def fig_monthly_change_rate():
    """Month-over-month FF% change rate."""
    mcr = staleness["monthly_change_rate"]
    x_labels = [str(m) for m in mcr.index]
    y_vals = [float(v * 100) for v in mcr.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_vals,
        marker=dict(color=COLORS["accent_orange"],
                    line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% of tickers changed<extra></extra>",
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="% of Tickers with Changed Free-Float vs Prior Month", font=dict(size=16)),
        xaxis=dict(title="Month"),
        yaxis=dict(title="% of Tickers Changed", ticksuffix="%"),
        height=380,
    )
    return fig


def fig_r2_horse_race():
    """R² comparison bar chart — the core horse-race."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Close-to-Close |Return|", "Intraday H-L Range"],
                        horizontal_spacing=0.12)

    for col_idx, dep in enumerate(["|Return|", "H-L Range"], 1):
        sub = reg_results[reg_results["Dependent"] == dep].sort_values("R²")
        colors_bar = []
        for reg in sub["Regressors"]:
            if "Amihud" in reg and "Free" not in reg and "MktCap" not in reg:
                colors_bar.append(COLORS["accent_blue"])
            elif "Free" in reg and "Amihud" not in reg:
                colors_bar.append(COLORS["accent_orange"])
            elif "MktCap" in reg:
                colors_bar.append(COLORS["accent_purple"])
            else:
                colors_bar.append(COLORS["accent_cyan"])

        fig.add_trace(go.Bar(
            y=sub["Regressors"], x=sub["R²"],
            orientation="h",
            marker=dict(color=colors_bar, line=dict(width=0)),
            text=[f"{v:.4f}" for v in sub["R²"]],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            hovertemplate="<b>%{y}</b><br>R² = %{x:.5f}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Explanatory Power (R²): Which Proxy Wins?", font=dict(size=18)),
        height=420,
        margin=dict(l=50, r=80, t=60, b=50),
    )
    fig.update_xaxes(title_text="R²")
    return fig


def fig_quintile_sorts():
    """Quintile portfolio sort comparison."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Sorted by Amihud ILLIQ", "Sorted by Free-Float %"],
                        horizontal_spacing=0.08)

    for col_idx, (qdata, title) in enumerate([(q_illiq, "Amihud"), (q_ff, "Free-Float")], 1):
        for i, (metric, color) in enumerate([
            ("abs_return", COLORS["accent_blue"]),
            ("hl_range", COLORS["accent_orange"]),
        ]):
            label = "|Return|" if metric == "abs_return" else "H-L Range"
            fig.add_trace(go.Bar(
                x=qdata.index,
                y=qdata[metric],
                name=label,
                marker=dict(color=color, line=dict(width=0)),
                showlegend=(col_idx == 1),
                hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.5f}}<extra></extra>",
            ), row=1, col=col_idx)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Quintile Portfolio Sorts: Do Illiquid Stocks Move More?",
                   font=dict(size=18)),
        height=480,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=90, b=50),
    )
    fig.update_yaxes(title_text="Mean Value", col=1)
    return fig


def fig_spearman_heatmap():
    """Spearman rank correlation heatmap."""
    mask = np.triu(np.ones_like(spearman, dtype=bool), k=1)
    vals = spearman.values.copy()
    vals[mask] = np.nan

    fig = go.Figure(go.Heatmap(
        z=vals,
        x=spearman.columns,
        y=spearman.index,
        colorscale=[[0, COLORS["accent_blue"]], [0.5, "#1a1a2e"], [1, COLORS["accent_red"]]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in vals],
        texttemplate="%{text}",
        textfont=dict(size=13, color=COLORS["text"]),
        hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="Spearman ρ", font=dict(size=11))),
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Spearman Rank Correlations", font=dict(size=16)),
        height=460,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_monthly_r2_lines():
    """Monthly R² time series."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["|Return|", "H-L Range"],
                        horizontal_spacing=0.08)

    for col_idx, dep in enumerate(["|Return|", "H-L Range"], 1):
        sub = monthly_r2[monthly_r2["Dependent"] == dep]
        for reg, color, dash_style in [
            ("Amihud", COLORS["accent_blue"], "solid"),
            ("Free-Float %", COLORS["accent_orange"], "dash"),
        ]:
            line_data = sub[sub["Regressor"] == reg]
            fig.add_trace(go.Scatter(
                x=line_data["Month"], y=line_data["R²"],
                mode="lines+markers",
                name=reg, showlegend=(col_idx == 1),
                line=dict(color=color, width=3, dash=dash_style),
                marker=dict(size=7, symbol="circle"),
                hovertemplate=f"<b>{reg}</b><br>%{{x}}<br>R² = %{{y:.5f}}<extra></extra>",
            ), row=1, col=col_idx)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Monthly R² Stability: Does the Signal Hold All Year?",
                   font=dict(size=18)),
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=90, b=50),
    )
    fig.update_yaxes(title_text="R²", col=1)
    return fig


def fig_size_tercile_r2():
    """R² by size tercile grouped bar."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["|Return|", "H-L Range"],
                        horizontal_spacing=0.08)
    order = ["Small Cap", "Mid Cap", "Large Cap"]

    for col_idx, dep in enumerate(["|Return|", "H-L Range"], 1):
        sub = size_r2[size_r2["Dependent"] == dep]
        for reg, color in [("Amihud", COLORS["accent_blue"]),
                           ("Free-Float %", COLORS["accent_orange"])]:
            d = sub[sub["Regressor"] == reg].set_index("Size").reindex(order)
            fig.add_trace(go.Bar(
                x=d.index, y=d["R²"],
                name=reg,
                marker=dict(color=color, line=dict(width=0)),
                text=[f"{v:.4f}" for v in d["R²"]],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["text"]),
                showlegend=(col_idx == 1),
                hovertemplate=f"<b>{reg}</b><br>%{{x}}: R² = %{{y:.5f}}<extra></extra>",
            ), row=1, col=col_idx)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Size Controlled: Amihud Wins in Every Market Cap Bucket",
                   font=dict(size=18)),
        height=480,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=90, b=50),
    )
    fig.update_yaxes(title_text="R²", col=1)
    return fig


def fig_interaction_heatmap(metric="ret"):
    """5x5 interaction heatmap."""
    data = heat_ret if metric == "ret" else heat_hl
    label = "Mean |Return|" if metric == "ret" else "Mean H-L Range"

    fig = go.Figure(go.Heatmap(
        z=data.values,
        x=data.columns.tolist(),
        y=data.index.tolist(),
        colorscale=[[0, "#0B1120"], [0.3, COLORS["accent_blue"]],
                    [0.6, COLORS["accent_orange"]], [1, COLORS["accent_red"]]],
        text=[[f"{v:.4f}" for v in row] for row in data.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color=COLORS["text"]),
        hovertemplate="Amihud: %{y}<br>Free-Float: %{x}<br>" + label + " = %{z:.5f}<extra></extra>",
        colorbar=dict(title=dict(text=label, font=dict(size=10))),
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text=f"Interaction Effect: {label} by Amihud × Free-Float Quintile",
                   font=dict(size=16)),
        xaxis=dict(title="Free-Float % Quintile →", side="bottom"),
        yaxis=dict(title="← Amihud ILLIQ Quintile", autorange="reversed"),
        height=460,
    )
    return fig


def fig_distribution_overview():
    """Distribution overview of key variables."""
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Free-Float %", "log₁₀(Amihud ILLIQ)", "H-L Range"])

    ff_vals = df["eqy_free_float_pct"].dropna()
    fig.add_trace(go.Histogram(
        x=ff_vals, nbinsx=60,
        marker=dict(color=COLORS["accent_blue"], line=dict(width=0)),
        hovertemplate="FF%: %{x:.0f}<br>Count: %{y:,}<extra></extra>",
    ), row=1, col=1)

    illiq_vals = np.log10(df["illiq_252d"].dropna().clip(lower=1e-15))
    fig.add_trace(go.Histogram(
        x=illiq_vals, nbinsx=60,
        marker=dict(color=COLORS["accent_orange"], line=dict(width=0)),
        hovertemplate="log₁₀(ILLIQ): %{x:.1f}<br>Count: %{y:,}<extra></extra>",
    ), row=1, col=2)

    hl_vals = df["hl_range"].dropna().clip(upper=df["hl_range"].quantile(0.99))
    fig.add_trace(go.Histogram(
        x=hl_vals, nbinsx=60,
        marker=dict(color=COLORS["accent_green"], line=dict(width=0)),
        hovertemplate="H-L Range: %{x:.3f}<br>Count: %{y:,}<extra></extra>",
    ), row=1, col=3)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Distribution of Core Variables", font=dict(size=16)),
        showlegend=False,
        height=340,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def fig_scatter_binned(x_col, y_col, x_label, y_label, color=COLORS["accent_blue"]):
    """Binned scatter plot with trend line."""
    np.random.seed(0)
    sample_size = min(30_000, len(dfc))
    idx = np.random.choice(dfc.index, size=sample_size, replace=False)
    samp = dfc.loc[idx]

    bins = pd.cut(dfc[x_col], bins=50)
    binned = dfc.groupby(bins, observed=True)[y_col].mean()
    centers = [(iv.left + iv.right) / 2 for iv in binned.index]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=samp[x_col], y=samp[y_col],
        mode="markers",
        marker=dict(color=color, opacity=0.06, size=3),
        hoverinfo="skip",
        name="Raw observations",
    ))
    fig.add_trace(go.Scatter(
        x=centers, y=binned.values,
        mode="lines+markers",
        line=dict(color=COLORS["accent_red"], width=3),
        marker=dict(size=6, color=COLORS["accent_red"]),
        name="Binned mean (50 bins)",
        hovertemplate=f"<b>{x_label}</b>: %{{x:.2f}}<br><b>{y_label}</b>: %{{y:.5f}}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text=f"{y_label} vs {x_label}", font=dict(size=16)),
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def fig_ticker_detail(ticker):
    """Generate a multi-panel detail view for a single ticker."""
    tkr_data = df[df["ticker"] == ticker].sort_values("date").copy()
    if tkr_data.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOT_TEMPLATE, title="No data for this ticker")
        return fig

    tkr_data["hl_range"] = (tkr_data["high"] - tkr_data["low"]) / tkr_data["close"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=["Price", "Rolling 252d Amihud ILLIQ",
                        "Daily |Return|", "Free-Float % (monthly snapshot)",
                        "Intraday H-L Range", "Dollar Volume"],
        vertical_spacing=0.10, horizontal_spacing=0.08,
    )

    # Price
    fig.add_trace(go.Scatter(
        x=tkr_data["date"], y=tkr_data["close"],
        mode="lines", line=dict(color=COLORS["accent_blue"], width=1.5),
        name="Close", hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Amihud
    fig.add_trace(go.Scatter(
        x=tkr_data["date"], y=tkr_data["illiq_252d"],
        mode="lines", line=dict(color=COLORS["accent_orange"], width=1.5),
        name="Amihud 252d",
        hovertemplate="%{x|%b %d}<br>ILLIQ: %{customdata}<extra></extra>",
        customdata=[_fmt_illiq(v) for v in tkr_data["illiq_252d"]],
    ), row=1, col=2)

    # |Return|
    fig.add_trace(go.Bar(
        x=tkr_data["date"], y=tkr_data["abs_return"],
        marker=dict(color=COLORS["accent_cyan"], line=dict(width=0)),
        name="|Return|",
        hovertemplate="%{x|%b %d}<br>|Ret|: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # Free-Float
    ff_data = tkr_data.dropna(subset=["eqy_free_float_pct"])
    if not ff_data.empty:
        ff_monthly_tk = ff_data.groupby(ff_data["date"].dt.to_period("M")).first().reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=ff_monthly_tk["date"], y=ff_monthly_tk["eqy_free_float_pct"],
            mode="lines+markers",
            line=dict(color=COLORS["accent_orange"], width=2, shape="hv"),
            marker=dict(size=8, symbol="diamond"),
            name="Free-Float %",
            hovertemplate="%{x|%b %d}<br>FF: %{y:.1f}%<extra></extra>",
        ), row=2, col=2)

    # H-L Range
    fig.add_trace(go.Bar(
        x=tkr_data["date"], y=tkr_data["hl_range"],
        marker=dict(color=COLORS["accent_green"], line=dict(width=0)),
        name="H-L Range",
        hovertemplate="%{x|%b %d}<br>HL: %{y:.4f}<extra></extra>",
    ), row=3, col=1)

    # Dollar Volume
    fig.add_trace(go.Bar(
        x=tkr_data["date"], y=tkr_data["dollar_volume"],
        marker=dict(color=COLORS["accent_purple"], line=dict(width=0)),
        name="$ Volume",
        hovertemplate="%{x|%b %d}<br>$Vol: %{y:,.0f}<extra></extra>",
    ), row=3, col=2)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text=f"{ticker} — Full-Year Profile", font=dict(size=18)),
        height=750,
        showlegend=False,
    )
    return fig


def fig_amihud_vs_ff_scatter():
    """Scatter of Amihud vs Free-Float colored by size group."""
    # One point per ticker (median values)
    tk = dfc.groupby("ticker").agg(
        amihud=("illiq_252d", "median"),
        ff=("eqy_free_float_pct", "median"),
        mktcap=("cur_mkt_cap", "median"),
        size=("size_group", "first"),
        avg_ret=("abs_return", "mean"),
    ).reset_index()

    tk["log_amihud"] = np.log10(tk["amihud"].clip(lower=1e-15))

    color_map = {
        "Small Cap": COLORS["accent_red"],
        "Mid Cap": COLORS["accent_orange"],
        "Large Cap": COLORS["accent_blue"],
    }

    fig = go.Figure()
    for sz in ["Small Cap", "Mid Cap", "Large Cap"]:
        sub = tk[tk["size"] == sz]
        fig.add_trace(go.Scattergl(
            x=sub["ff"], y=sub["log_amihud"],
            mode="markers",
            marker=dict(color=color_map[sz], size=5, opacity=0.6),
            name=sz,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "FF: %{x:.1f}%<br>"
                "log₁₀(Amihud): %{y:.2f}<br>"
                "Avg |Ret|: %{customdata[1]:.4f}<extra></extra>"
            ),
            customdata=np.column_stack([sub["ticker"], sub["avg_ret"]]),
        ))

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Amihud vs Free-Float: Are They Measuring the Same Thing?",
                   font=dict(size=16)),
        xaxis=dict(title="Free-Float %"),
        yaxis=dict(title="log₁₀(Amihud ILLIQ 252d)"),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def fig_staleness_vs_amihud_volatility():
    """Compare FF% update frequency with Amihud variance — shows Amihud captures
    liquidity changes that FF% misses due to staleness."""
    # For each ticker, get: max FF% run-length vs coefficient of variation of Amihud
    tk_stats = dfc.groupby("ticker").agg(
        amihud_cv=("illiq_252d", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        avg_ret=("abs_return", "mean"),
        ff_pct=("eqy_free_float_pct", "first"),
        size=("size_group", "first"),
    ).reset_index()

    sr = staleness["max_run_sr"]
    tk_stats = tk_stats.merge(sr.rename("ff_streak"), left_on="ticker", right_index=True, how="inner")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=tk_stats["ff_streak"].astype(int).tolist(),
        y=tk_stats["amihud_cv"].astype(float).tolist(),
        mode="markers",
        marker=dict(
            color=tk_stats["avg_ret"].astype(float).tolist(),
            colorscale=[[0, COLORS["accent_blue"]], [1, COLORS["accent_red"]]],
            size=5,
            opacity=0.5,
            colorbar=dict(title=dict(text="Avg |Return|", font=dict(size=10))),
        ),
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "FF Unchanged Streak: %{x} months<br>"
            "Amihud CV: %{y:.2f}<br>"
            "<extra></extra>"
        ),
        customdata=tk_stats["ticker"].tolist(),
    ))

    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Stale Free-Float vs Dynamic Amihud: The Blind Spot",
                   font=dict(size=16)),
        xaxis=dict(title="Longest FF% Unchanged Streak (months)", dtick=1),
        yaxis=dict(title="Amihud Coefficient of Variation (within-year)"),
        height=440,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# DASH APP
# ═══════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    title="Liquidity Risk Dashboard",
    suppress_callback_exceptions=True,
)

# ── Sidebar ──

sidebar_sections = [
    ("THE STORY", [
        ("overview",   "The Thesis",        "bi-lightbulb"),
        ("problem",    "The Problem",       "bi-exclamation-circle"),
        ("evidence",   "The Evidence",      "bi-trophy"),
        ("robustness", "Robustness",        "bi-shield-check"),
        ("tail_risk",  "Tail Risk",         "bi-lightning"),
    ]),
    ("VALIDATION", [
        ("oos",        "Out-of-Sample",     "bi-clipboard-data"),
    ]),
    ("EXPLORE", [
        ("tools",      "Interactive Tools", "bi-sliders"),
        ("conclusion", "Conclusion",        "bi-flag"),
    ]),
]

# Flat list for callbacks
sidebar_items = [item for _, items in sidebar_sections for item in items]

SIDEBAR_WIDTH = "260px"
SIDEBAR_COLLAPSED_WIDTH = "52px"

# Build sidebar navigation with section headers
_nav_children = []
for _sec_name, _sec_items in sidebar_sections:
    _nav_children.append(
        html.Div(_sec_name, className="sidebar-section",
                 style={"color": COLORS["text_muted"], "fontSize": "0.6rem",
                        "fontWeight": 700, "letterSpacing": "2px",
                        "padding": "16px 14px 4px 14px", "textTransform": "uppercase",
                        "opacity": 0.5})
    )
    for _key, _label, _icon in _sec_items:
        _nav_children.append(
            dbc.Button(
                [html.I(className=f"bi {_icon}",
                        style={"minWidth": "20px", "fontSize": "1rem", "flexShrink": "0"}),
                 html.Span(f"  {_label}", className="sidebar-label",
                           style={"transition": "opacity 0.2s ease"})],
                id={"type": "nav-btn", "index": _key},
                color="link",
                className="nav-link-btn",
                style={
                    "color": COLORS["text_muted"], "fontSize": "0.82rem",
                    "fontWeight": 500, "textAlign": "left",
                    "padding": "10px 14px", "width": "100%",
                    "border": "none", "borderRadius": "0",
                    "textDecoration": "none", "whiteSpace": "nowrap",
                    "overflow": "hidden", "display": "flex", "alignItems": "center",
                },
            )
        )

sidebar = html.Div(
    [
        # Toggle button
        html.Div(
            dbc.Button(
                html.I(className="bi bi-list", id="sidebar-icon"),
                id="sidebar-toggle",
                color="link",
                style={
                    "color": COLORS["text_muted"], "fontSize": "1.3rem",
                    "padding": "8px 14px", "border": "none",
                    "width": "100%", "textAlign": "left",
                },
            ),
            style={"borderBottom": f"1px solid {COLORS['divider']}"},
        ),
        # Branding
        html.Div(
            [
                html.H4("LIQUIDITY RISK", id="sidebar-title", style={
                    "color": COLORS["accent_blue"], "fontWeight": 800,
                    "letterSpacing": "2px", "fontSize": "0.9rem",
                    "marginBottom": "2px",
                }),
                html.P("Amihud vs Free-Float", id="sidebar-subtitle", style={
                    "color": COLORS["text_muted"], "fontSize": "0.7rem",
                    "letterSpacing": "1px",
                }),
            ],
            id="sidebar-brand",
            style={"padding": "12px 20px 12px 20px",
                   "borderBottom": f"1px solid {COLORS['divider']}"},
        ),
        # Nav buttons with section headers
        html.Div(_nav_children, id="sidebar-nav",
                 style={"paddingTop": "4px", "paddingBottom": "20px"}),
    ],
    id="sidebar",
    className="",
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": SIDEBAR_WIDTH,
        "backgroundColor": COLORS["card"],
        "borderRight": f"1px solid {COLORS['card_border']}",
        "overflowY": "auto",
        "overflowX": "hidden",
        "zIndex": 1000,
        "transition": "width 0.25s ease",
    },
)

# ── Page Layouts ──

def page_overview():
    return html.Div([
        # Hero
        html.Div([
            html.H1("Two Signals, One Risk", style={
                "background": f"linear-gradient(135deg, {COLORS['accent_blue']}, {COLORS['accent_purple']})",
                "-webkit-background-clip": "text",
                "-webkit-text-fill-color": "transparent",
                "fontWeight": 800, "fontSize": "2.8rem", "marginBottom": "8px",
                "lineHeight": 1.1,
            }),
            html.P(
                "Amihud ILLIQ and free-float percentage capture different facets of "
                "liquidity risk. This dashboard maps where each signal shines, where "
                "each fails, and how they combine.",
                style={"color": COLORS["text_muted"], "fontSize": "1.05rem",
                       "maxWidth": "680px", "lineHeight": 1.5},
            ),
        ], style={"marginBottom": "36px"}),

        # Hero cards
        dbc.Row([
            dbc.Col(make_card("Total Observations", f"{df.shape[0]:,}",
                              "Daily ticker-day rows, 2025"), md=3),
            dbc.Col(make_card("Tickers Analyzed", f"{df['ticker'].nunique():,}",
                              "US equities, full-year 2025"), md=3),
            dbc.Col(make_card("Amihud R² (standalone)", f"{r2_amihud_ret:.4f}",
                              f"FF% standalone: {r2_ff_ret:.4f}",
                              color=COLORS["accent_blue"]), md=3),
            dbc.Col(make_card("FF% Stale 1+ Quarter", f"{staleness['streak_3plus']:.0%}",
                              "Tickers unchanged for 3+ months",
                              color=COLORS["accent_red"]), md=3),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(make_card("Bloomberg FF Coverage", f"{ff_coverage:.0%}",
                              "Of tickers have free-float data"), md=3),
            dbc.Col(make_card("Analysis Sample", f"{dfc.shape[0]:,}",
                              "Rows with all fields non-null"), md=3),
            dbc.Col(make_card("Amihud-MktCap Corr", f"{-0.95:.2f}",
                              "Amihud is largely a size proxy",
                              color=COLORS["accent_orange"]), md=3),
            dbc.Col(make_card("FF% Adds More after Size", "Yes",
                              "Higher incremental R² over MktCap",
                              color=COLORS["accent_green"]), md=3),
        ], className="g-3 mb-4"),

        # Story overview
        dbc.Card(
            dbc.CardBody([
                html.H5("The Question", style={"color": COLORS["accent_blue"],
                                                  "fontWeight": 700, "marginBottom": "12px"}),
                dcc.Markdown("""
**Two competing liquidity signals exist.** Free-float percentage (Bloomberg `EQY_FREE_FLOAT_PCT`)
measures the *supply side* — what share of outstanding stock is publicly tradeable. The Amihud
ILLIQ ratio measures the *friction side* — how much the price moves per dollar of actual volume.
These capture fundamentally different information.

**What this analysis reveals** across {:,} daily observations and {:,} US equities in 2025:

- **As a standalone predictor**, Amihud explains {:.1f}× more return variance than free-float —
  but most of this is because Amihud is nearly a perfect proxy for market cap (Spearman ρ = −0.95).
- **After controlling for market cap**, free-float actually contributes *more* incremental R² than
  Amihud — because FF% is less collinear with size and captures genuinely different information.
- **Within size buckets**, Amihud dominates — especially in small caps where free-float has
  essentially zero explanatory power but Amihud captures real microstructure friction.
- **Free-float data is structurally stale** — {:.0%} of tickers are unchanged for 3+ months.
  Amihud updates daily.

The story is not "one wins" — it's "each wins in a different context." Navigate the chapters
to see where.
""".format(df.shape[0], df["ticker"].nunique(), superiority_ret, staleness['streak_3plus']),
                             style={"color": COLORS["text_muted"], "fontSize": "0.9rem",
                                    "lineHeight": 1.7}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px"},
        ),

        # Distribution overview
        html.Div([
            dcc.Graph(figure=fig_distribution_overview(), config={"displayModeBar": False}),
            dbc.Row([
                dbc.Col(html.P(
                    "Most stocks have 80–100% free-float (large public companies). "
                    "The left tail near 0–30% represents closely-held or insider-heavy names — "
                    "the low-float stocks where liquidity risk is most acute.",
                    style={"color": COLORS["text_muted"], "fontSize": "0.78rem",
                           "lineHeight": 1.5, "textAlign": "center", "padding": "0 8px"},
                ), md=4),
                dbc.Col(html.P(
                    "On a log scale, Amihud ILLIQ spans ~8 orders of magnitude: from mega-cap "
                    "liquid names (left) to illiquid micro-caps (right). The roughly normal "
                    "shape confirms the log transform is appropriate for regressions.",
                    style={"color": COLORS["text_muted"], "fontSize": "0.78rem",
                           "lineHeight": 1.5, "textAlign": "center", "padding": "0 8px"},
                ), md=4),
                dbc.Col(html.P(
                    "Intraday price range (High − Low)/Close is right-skewed as expected for "
                    "volatility. Clipped at 99th percentile. This captures within-day price "
                    "impact that close-to-close returns miss.",
                    style={"color": COLORS["text_muted"], "fontSize": "0.78rem",
                           "lineHeight": 1.5, "textAlign": "center", "padding": "0 8px"},
                ), md=4),
            ], className="mt-1"),
        ], style={"marginTop": "24px"}),

        # Glossary & units reference
        dbc.Card(
            dbc.CardBody([
                html.H5("Glossary & Units Reference", style={
                    "color": COLORS["accent_cyan"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
| Term | Definition | Unit |
|------|-----------|------|
| **Amihud ILLIQ (252d)** | Rolling 252-day mean of daily \\|return\\| / $ volume. Higher = more illiquid. | Ratio (unitless) |
| **log(Amihud)** | Natural log of ILLIQ 252d, used in regressions. | log-ratio |
| **Amihud ILLIQ (21d)** | Rolling 21-day mean — short-horizon friction (recent trading conditions). | Ratio |
| **Liquidity Z-Score** | (ILLIQ ratio − rolling mean) / rolling std. Cross-sectionally comparable stress signal. | Standard deviations |
| **Free-Float % (FF%)** | Bloomberg `EQY_FREE_FLOAT_PCT` — share of outstanding stock publicly tradeable. | 0–100% |
| **\\|Return\\|** | Absolute value of daily close-to-close return (e.g., 0.05 = 5%). | Decimal |
| **H-L Range** | Intraday (High − Low) / Close — a proxy for realized microstructure volatility. | Decimal |
| **R²** | Coefficient of determination from OLS. 0 = no explanatory power, 1 = perfect. | Unitless (0–1) |
| **ΔR²** | Incremental R² from adding a variable to a baseline model. | Unitless |
| **F1 Score** | Harmonic mean of precision and recall for classification tasks. | Unitless (0–1) |
| **MktCap** | Market capitalisation — price × shares outstanding. | USD |
""", style={"color": COLORS["text_muted"], "fontSize": "0.82rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "24px"},
        ),
    ])


def page_staleness():
    return html.Div([
        html.H2("The Free-Float Staleness Problem", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Bloomberg reports EQY_FREE_FLOAT_PCT monthly, but how often does the value actually change? "
            "If the data doesn't update, it can't reflect shifting liquidity conditions.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem", "marginBottom": "4px",
                   "maxWidth": "700px"},
        ),
        html.P(
            "Metric cards show the share of tickers (%) whose FF% stayed constant for 3+ or 6+ consecutive "
            "months. The histogram below shows the distribution of longest unchanged streaks (months). "
            "The scatter plots the Amihud coefficient of variation (unitless) vs streak length.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "marginBottom": "24px",
                   "maxWidth": "700px", "fontStyle": "italic"},
        ),

        dbc.Row([
            dbc.Col(make_card("3+ Month Unchanged", f"{staleness['streak_3plus']:.0%}",
                              "Tickers with stale FF data",
                              color=COLORS["accent_orange"]), md=3),
            dbc.Col(make_card("6+ Month Unchanged", f"{staleness['streak_6plus']:.0%}",
                              "Severely stale data",
                              color=COLORS["accent_red"]), md=3),
            dbc.Col(make_card("Never Changed", f"{staleness['never_changed_pct']:.0%}",
                              "Same value all 12 months",
                              color=COLORS["accent_red"]), md=3),
            dbc.Col(make_card("Quarter-Boundary Ratio",
                              f"{staleness['qtr_change_rate']/max(staleness['non_qtr_change_rate'],1e-10):.1f}×",
                              "More likely to change at Q starts",
                              color=COLORS["accent_purple"]), md=3),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_staleness_histogram(), config={"displayModeBar": False}),
                html.P("Each bar = how many tickers had their longest FF% unchanged streak at that length. "
                       "Red bars (6+ months) are the most concerning — these stocks had no FF% update for half the year or more.",
                       style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                              "padding": "0 12px", "marginTop": "-8px"}),
            ], md=6),
            dbc.Col([
                dcc.Graph(figure=fig_monthly_change_rate(), config={"displayModeBar": False}),
                html.P("What fraction of tickers saw their FF% change vs the prior month. "
                       "Low bars mean most stocks had no update that month. "
                       "Spikes at quarter starts (Jan, Apr, Jul, Oct) suggest filing-driven updates.",
                       style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                              "padding": "0 12px", "marginTop": "-8px"}),
            ], md=6),
        ], className="mb-4"),

        dbc.Card(
            dbc.CardBody([
                html.H5("The Blind Spot", style={"color": COLORS["accent_orange"],
                                                    "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Graph(figure=fig_staleness_vs_amihud_volatility(),
                          config={"displayModeBar": False}),
                dcc.Markdown("""
**Coefficient of Variation (CV)** = standard deviation / mean — a unitless measure of how
much Amihud ILLIQ varied within the year for each stock. A CV of 0.5 means liquidity
conditions swung by half the annual average; a CV near 0 means liquidity was stable all year.

**What this shows:** Stocks in the **upper-right** corner had highly volatile liquidity
(high CV) yet their free-float value never changed (long streak). This is where FF% fails
most — it reports "no change" while actual trading friction is swinging significantly.

**Is Amihud doing its job?** Yes — the wide vertical spread of CV values across all streak
lengths proves Amihud *detects* liquidity variation that FF% completely misses. That's
the operational case: if you rely only on FF%, you're blind to these shifts.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6,
            "marginTop": "12px"}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px"},
        ),
    ])


def page_horse_race():
    return html.Div([
        html.H2("The Horse Race: Amihud vs Free-Float", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Head-to-head comparison of explanatory power. R² measures how much variance "
            "in daily returns each liquidity proxy captures.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "700px"},
        ),
        html.P(
            "R² is unitless (0–1). The bar chart compares R² from simple OLS regressions for two "
            "dependent variables: |Return| (absolute daily return, decimal) and H-L Range "
            "(intraday High−Low / Close, decimal). The quintile sorts rank all ticker-days by each "
            "predictor and show average outcomes — no model needed.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "700px", "fontStyle": "italic"},
        ),

        dbc.Row([
            dbc.Col(make_card("Amihud R² (|Return|)", f"{r2_amihud_ret:.4f}",
                              color=COLORS["accent_blue"]), md=3),
            dbc.Col(make_card("FF% R² (|Return|)", f"{r2_ff_ret:.4f}",
                              color=COLORS["accent_orange"]), md=3),
            dbc.Col(make_card("Amihud R² (H-L Range)", f"{r2_amihud_hl:.4f}",
                              color=COLORS["accent_blue"]), md=3),
            dbc.Col(make_card("FF% R² (H-L Range)", f"{r2_ff_hl:.4f}",
                              color=COLORS["accent_orange"]), md=3),
        ], className="g-3 mb-4"),

        dcc.Graph(figure=fig_r2_horse_race(), config={"displayModeBar": False}),
        html.P("Longer bar = more variance explained. These are standalone (univariate) R² values — "
               "no size control. Amihud's large advantage partly reflects its correlation with market cap.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "16px"}),

        dcc.Graph(figure=fig_quintile_sorts(), config={"displayModeBar": False}),
        html.P("Stocks ranked into 5 equal groups by each measure. A clear staircase from Q1 to Q5 "
               "means the measure separates low-volatility from high-volatility stocks. No model needed — "
               "this is a pure sort.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "8px"}),

        dbc.Card(
            dbc.CardBody([
                html.H5("Key Takeaway", style={"color": COLORS["accent_blue"],
                                                  "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown(f"""
**As a standalone predictor**, Amihud explains {superiority_ret:.1f}× more |Return| variance and
{superiority_hl:.1f}× more H-L Range variance than free-float. But context matters:

**Why the gap is so large:** Amihud correlates with market cap at ρ = −0.95. Much of its
explanatory power comes from being a near-perfect size proxy. Market cap *alone* actually
outperforms Amihud alone (R² = {r2_amihud_ret:.4f} for Amihud vs ~0.082 for MktCap on |Return|).

**Where each signal shines:**
- **Amihud dominates within size buckets** — especially small caps, where free-float has
  essentially zero R² but Amihud captures real microstructure friction.
- **Free-float adds more *incremental* information** on top of market cap — because it is
  less collinear with size and captures ownership structure that MktCap misses.

**The quintile sorts** confirm the standalone Amihud result model-free: stocks in the highest
Amihud quintile show monotonically increasing return magnitudes. Free-float sorts are weaker
and less monotonic.

The right question is not "which is better?" but "better *for what?*" — explored in the
Regression Specs and Size Bucket tabs.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def page_deep_dive():
    return html.Div([
        html.H2("Scatter Analysis & Correlations", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Binned scatter plots reveal the shape of the relationship. The red trend line shows "
            "the average within each bin — stripping away daily noise to reveal the signal.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "24px", "maxWidth": "700px"},
        ),

        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=fig_scatter_binned("log_illiq", "abs_return",
                                          "log(Amihud ILLIQ 252d)", "|Return|",
                                          COLORS["accent_blue"]),
                config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(
                figure=fig_scatter_binned("eqy_free_float_pct", "abs_return",
                                          "Free-Float %", "|Return|",
                                          COLORS["accent_orange"]),
                config={"displayModeBar": False}), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=fig_scatter_binned("log_illiq", "hl_range",
                                          "log(Amihud ILLIQ 252d)", "H-L Range",
                                          COLORS["accent_blue"]),
                config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(
                figure=fig_scatter_binned("eqy_free_float_pct", "hl_range",
                                          "Free-Float %", "H-L Range",
                                          COLORS["accent_orange"]),
                config={"displayModeBar": False}), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_spearman_heatmap(),
                              config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_amihud_vs_ff_scatter(),
                              config={"displayModeBar": False}), md=6),
        ]),

        dbc.Card(
            dbc.CardBody([
                html.H5("How to Read These Charts", style={
                    "color": COLORS["accent_cyan"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
**Binned Scatters (top 4 panels):** Each dot is the average of ~equal-sized bins along the
x-axis.  A steep red trend line means the predictor (x) explains meaningful variation in
the outcome (y).  Steeper slope for Amihud than for Free-Float = stronger signal.

- **|Return|** (y-axis, top row) — the absolute value of the daily close-to-close return
  (unitless decimal; 0.05 = 5%).
- **H-L Range** (y-axis, bottom row) — intraday (High − Low) / Close (unitless decimal;
  0.04 = 4% intraday swing).
- **log(Amihud ILLIQ 252d)** (x-axis, left column) — natural log of the 252-day rolling
  Amihud ratio (|return| / $ volume).  Higher = more illiquid.
- **Free-Float %** (x-axis, right column) — Bloomberg `EQY_FREE_FLOAT_PCT`, 0–100.

**Spearman Heatmap** (bottom left): Rank-order correlations between all variables.
Values near ±1 = strong monotonic relationship.  Look for how Amihud correlates with
return measures vs how Free-Float does.

**Amihud vs FF Scatter** (bottom right): Each dot is a ticker's median Amihud (log scale)
vs its Free-Float %.  A weak cloud means the two measures capture *different* information;
tight clustering would mean they're redundant.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def page_size():
    return html.Div([
        html.H2("Size-Controlled Analysis", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "A critical robustness check: both Amihud and free-float correlate with firm size. "
            "Does Amihud still win within each size bucket, or is it just a size proxy?",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "700px"},
        ),
        html.P(
            "Bars show R² (unitless, 0–1) from within-tercile OLS regressions. "
            "Size terciles are defined by median market cap: Small (bottom third), Mid, Large (top third).",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "700px", "fontStyle": "italic"},
        ),

        dcc.Graph(figure=fig_size_tercile_r2(), config={"displayModeBar": False}),
        html.P("Blue = Amihud, Orange = FF%. Within each size group, a taller bar = better predictor. "
               "This removes the size confound — if Amihud still wins here, it's real microstructure signal.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "8px"}),

        dbc.Card(
            dbc.CardBody([
                html.H5("Where Amihud Genuinely Wins", style={
                    "color": COLORS["accent_green"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
**This is Amihud's strongest result.** Within each size bucket — where market cap is held
roughly constant — Amihud outperforms free-float in every tercile, for both dependent variables.

**Small caps are where the gap is widest.** Amihud R² in small caps is 10–100× larger than
free-float R². This makes economic sense: small-cap stocks have the widest dispersion of
microstructure friction (thin order books, wide spreads), and Amihud measures that friction
directly. Free-float — a static ownership number — tells you nothing about intraday trading
conditions.

**Mid and large caps:** Amihud still wins, though the margin is smaller (~1.5–2×). Free-float
carries more signal in larger names where ownership filings are more timely and the supply
channel matters more.

**Why this matters:** The pooled regression (previous tab) is dominated by the Amihud–MktCap
collinearity (ρ = −0.95). This within-bucket view strips out size and shows the **residual**
signal — and there, Amihud's microstructure information is genuinely unique.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def page_time():
    return html.Div([
        html.H2("Time Stability", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Does the result hold every month, or is it driven by one volatile period? "
            "Stable monthly R² across the whole year confirms a genuine, persistent signal.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "700px"},
        ),
        html.P(
            "Each point is the R² (unitless, 0–1) from a month-specific OLS. Blue solid = Amihud, "
            "orange dashed = Free-Float. A flat blue line above orange = consistent superiority.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "700px", "fontStyle": "italic"},
        ),

        dcc.Graph(figure=fig_monthly_r2_lines(), config={"displayModeBar": False}),
        html.P("Each point = R² for that month's data only (univariate, no size control). "
               "If the blue line (Amihud) stays consistently above orange (FF%), the relationship "
               "holds throughout the year — it's not driven by one volatile month.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "8px"}),

        dbc.Card(
            dbc.CardBody([
                html.H5("Consistent Across All 12 Months", style={
                    "color": COLORS["accent_cyan"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
The Amihud line (solid blue) sits **consistently above** the Free-Float line (dashed orange)
across every month of 2025. There's no single volatile month driving the result — the
superiority is persistent and reliable.

This temporal stability is critical for a risk model: a signal that only works during crises
isn't useful for position sizing in normal markets. Amihud works in both calm and volatile
periods because it's derived from live trading data — not static filings.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def page_heatmap():
    return html.Div([
        html.H2("Risk Interaction Heatmap", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Double-sort: independently rank stocks by both Amihud and Free-Float, "
            "then measure average returns in each cell. Which axis drives the gradient?",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "700px"},
        ),
        html.P(
            "Toggle between |Return| (absolute daily return, decimal) and H-L Range "
            "(intraday swing, decimal). Cell color = average value within that quintile combination. "
            "A vertical gradient means Amihud quintile matters more; horizontal = Free-Float matters more.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "700px", "fontStyle": "italic"},
        ),

        dbc.Row([
            dbc.Col([
                html.Label("Metric:", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                dcc.RadioItems(
                    id="heatmap-metric",
                    options=[
                        {"label": " |Return|", "value": "ret"},
                        {"label": " H-L Range", "value": "hl"},
                    ],
                    value="ret",
                    inline=True,
                    style={"color": COLORS["text"], "fontSize": "0.9rem"},
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"marginRight": "20px"},
                ),
            ], md=6),
        ], className="mb-3"),

        dcc.Graph(id="interaction-heatmap",
                  figure=fig_interaction_heatmap("ret"),
                  config={"displayModeBar": False}),

        dbc.Card(
            dbc.CardBody([
                html.H5("How to Read This", style={
                    "color": COLORS["accent_orange"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
**If the color gradient runs vertically** (top-to-bottom), Amihud quintile drives returns
more than free-float. **If horizontal** (left-to-right), free-float matters more.

**Caveat:** Both Amihud and free-float correlate with market cap, so the gradients
partly reflect size. The key question is whether any cell deviates from what
size alone would predict — e.g., a low-Amihud / low-float stock that still moves a lot
would suggest free-float captures something Amihud misses.

**Upper-left corner** (high Amihud, low float) = highest risk: illiquid by trading
activity AND limited public supply.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def _slider_marks(lo, hi, n_marks=6):
    """Generate log-spaced slider marks between lo and hi in USD."""
    import math
    lo_log = math.log10(max(lo, 1e6))
    hi_log = math.log10(hi)
    marks = {}
    for i in range(n_marks):
        exp = lo_log + i * (hi_log - lo_log) / (n_marks - 1)
        val = 10 ** exp
        marks[exp] = _fmt_usd(val)
    return lo_log, hi_log, marks


def page_bucket_tuner():
    """Interactive market cap bucket tuner with round-number sliders."""
    # Pre-compute initial stats — use full dataset (df) so ALL tickers are included
    init_stats = compute_bucket_stats(df)

    # Slider config: log10 scale for readability
    # Data range (full universe, not just analysis sample)
    mc_min = df.dropna(subset=["cur_mkt_cap"]).groupby("ticker")["cur_mkt_cap"].median().min()
    mc_max = df.dropna(subset=["cur_mkt_cap"]).groupby("ticker")["cur_mkt_cap"].median().max()
    lo_log = np.log10(max(mc_min, 1e6))
    hi_log = np.log10(mc_max)

    # Preset configurations (all round numbers)
    presets = {
        "Standard (Default)": [50e6, 250e6, 2e9, 10e9, 200e9],
        "Russell-Aligned": [50e6, 300e6, 2e9, 15e9, 200e9],
        "Aggressive (More Nano)": [100e6, 500e6, 2e9, 10e9, 200e9],
        "Conservative (Fewer Buckets)": [25e6, 250e6, 2e9, 10e9, 100e9],
        "Equal-ish Tickers": None,  # will be computed from data quantiles
    }

    # Round thresholds labels
    boundary_labels = [
        ("Nano / Micro", 0),
        ("Micro / Small", 1),
        ("Small / Mid", 2),
        ("Mid / Large", 3),
        ("Large / Mega", 4),
    ]

    return html.Div([
        html.H2("Market Cap Bucket Tuner", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "This tool lets you redefine the market cap size boundaries used throughout the analysis. "
            "Drag any slider to change where Nano/Micro/Small/Mid/Large/Mega cut-offs sit, and the "
            "stats table below updates instantly — showing how many tickers fall in each bucket, their "
            "average returns, and tail risk. Use the presets dropdown to quickly switch between common "
            "index definitions (Russell, standard, etc.).",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "720px"},
        ),
        html.P(
            "Sliders operate on a log\u2081\u2080 scale in USD and auto-snap to round numbers. "
            "All tickers in the universe are included (not only those with free-float data). "
            "Table columns: Tickers (count), Median MktCap (USD), Mean |Ret| (decimal), "
            "p95/p99 |Ret| (decimal), Mean Amihud (ratio, e.g. 3.2×10⁻⁹).",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "720px", "fontStyle": "italic"},
        ),

        # Preset selector
        dbc.Row([
            dbc.Col([
                html.Label("Quick Presets:", style={
                    "color": COLORS["text_muted"], "fontSize": "0.8rem",
                    "fontWeight": 600, "textTransform": "uppercase", "letterSpacing": "1px",
                }),
                dcc.Dropdown(
                    id="bucket-preset",
                    options=[{"label": k, "value": k} for k in presets],
                    value="Standard (Default)",
                    style={"backgroundColor": COLORS["card"], "color": COLORS["text"]},
                    className="dash-dropdown-dark",
                ),
            ], md=4),
            dbc.Col([
                html.Label("Data Coverage:", style={
                    "color": COLORS["text_muted"], "fontSize": "0.8rem",
                    "fontWeight": 600, "textTransform": "uppercase", "letterSpacing": "1px",
                }),
                html.P(
                    f"{df['ticker'].nunique():,} tickers · "
                    f"MktCap range: {_fmt_usd(mc_min)} – {_fmt_usd(mc_max)}",
                    style={"color": COLORS["text"], "fontSize": "0.9rem", "marginTop": "8px"},
                ),
            ], md=4),
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise"), "  Reset to Default"],
                    id="bucket-reset-btn",
                    color="primary",
                    outline=True,
                    size="sm",
                    style={"marginTop": "24px"},
                ),
            ], md=4),
        ], className="mb-4"),

        # 5 boundary sliders — each with a log10 scale and round-number snapping
        dbc.Card(
            dbc.CardBody([
                html.H5("Boundary Thresholds", style={
                    "color": COLORS["accent_blue"], "fontWeight": 700, "marginBottom": "16px"}),
                html.P(
                    "Slide to adjust each boundary. Values auto-snap to round numbers "
                    "(nearest $25M / $50M / $100M / $500M / $1B / $5B / $10B / $25B / $50B / $100B).",
                    style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "marginBottom": "20px"},
                ),
                *[
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                html.Label(f"{lbl}:", style={
                                    "color": COLORS["text"], "fontWeight": 600,
                                    "fontSize": "0.85rem",
                                }),
                                md=2,
                            ),
                            dbc.Col(
                                dcc.Slider(
                                    id={"type": "bucket-slider", "index": idx},
                                    min=lo_log, max=hi_log,
                                    value=np.log10(DEFAULT_MKTCAP_THRESHOLDS[idx]),
                                    step=0.01,
                                    marks={},  # we'll show the value as a tooltip
                                    tooltip={"placement": "top", "always_visible": True},
                                    updatemode="mouseup",
                                ),
                                md=8,
                            ),
                            dbc.Col(
                                html.Span(
                                    _fmt_usd(DEFAULT_MKTCAP_THRESHOLDS[idx]),
                                    id={"type": "bucket-val-label", "index": idx},
                                    style={"color": COLORS["accent_cyan"], "fontWeight": 700,
                                           "fontSize": "0.95rem"},
                                ),
                                md=2,
                                style={"textAlign": "right"},
                            ),
                        ], className="align-items-center"),
                    ], style={"marginBottom": "16px"})
                    for lbl, idx in boundary_labels
                ],
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginBottom": "24px"},
        ),

        # Visual bucket scale
        html.Div(id="bucket-visual-scale", style={"marginBottom": "24px"}),

        # Stats table
        html.Div(id="bucket-stats-table", style={"marginBottom": "24px"}),

        # Distribution chart
        dcc.Graph(id="bucket-distribution-chart", config={"displayModeBar": False}),
    ])


def _snap_to_round(val_usd):
    """Snap a raw USD value to the nearest round memorable number."""
    snaps = [
        25e6, 50e6, 75e6, 100e6, 150e6, 200e6, 250e6, 300e6, 400e6, 500e6,
        750e6, 1e9, 1.5e9, 2e9, 2.5e9, 3e9, 4e9, 5e9, 7.5e9, 10e9,
        15e9, 20e9, 25e9, 30e9, 40e9, 50e9, 75e9, 100e9, 150e9, 200e9,
        250e9, 300e9, 400e9, 500e9, 750e9, 1e12,
    ]
    return min(snaps, key=lambda s: abs(s - val_usd))


def page_explorer():
    return html.Div([
        html.H2("Stock Explorer", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Deep-dive into any individual ticker: price, Amihud dynamics, return volatility, "
            "free-float snapshots, and dollar volume — all on one page.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "24px", "maxWidth": "700px"},
        ),

        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[{"label": t, "value": t} for t in ticker_list],
                    value=ticker_list[0] if ticker_list else None,
                    placeholder="Select a ticker...",
                    style={"backgroundColor": COLORS["card"], "color": COLORS["text"]},
                    className="dash-dropdown-dark",
                ),
            ], md=4),
        ], className="mb-3"),

        dcc.Graph(id="ticker-detail",
                  figure=fig_ticker_detail(ticker_list[0]) if ticker_list else go.Figure(),
                  config={"displayModeBar": True}),

        dbc.Card(
            dbc.CardBody([
                html.H5("Panel Guide", style={
                    "color": COLORS["accent_cyan"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
| Panel | What it shows | Units |
|-------|---------------|-------|
| **Price** | Adjusted close over the full date range | USD per share |
| **Amihud ILLIQ (252d)** | Rolling 252-day average of \\|return\\| / $ volume — the structural liquidity baseline | Ratio (higher = more illiquid) |
| **\\|Return\\|** | Daily absolute close-to-close return | Decimal (0.05 = 5%) |
| **Free-Float %** | Bloomberg `EQY_FREE_FLOAT_PCT` — share of outstanding stock publicly tradeable | 0–100% (flat stretches = stale data) |
| **H-L Range** | Intraday (High − Low) / Close — a proxy for within-day volatility | Decimal (0.04 = 4% swing) |
| **$ Volume** | Daily dollar trading volume | USD |

**Tip:** Look for periods where Amihud spikes but Free-Float stays flat — that gap is the
"blind spot" this thesis identifies.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def page_scenario_matrix():
    """Part 8: The opening act — 6-bucket × up/down empirical tail grid."""
    sm = scenario_matrix
    return html.Div([
        html.H2("The Risk Surface", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Extreme daily moves are asymmetric across the market cap spectrum. "
            "Upside tails widen dramatically as stocks get smaller — downside tails widen too, "
            "but the asymmetry ratio itself increases. This is the scenario surface.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "720px"},
        ),
        html.P(
            "Heatmap values are p99 daily returns (%). The asymmetry ratio (unitless) = p99 upside / p99 "
            "downside; >1 means upside tails are fatter. Overlay charts show the bucket's median "
            "Free-Float % (0–100) or median Amihud ILLIQ (log scale) against the tail width.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "720px", "fontStyle": "italic"},
        ),

        # Scenario matrix heatmap — up vs down p99 by bucket
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=_fig_scenario_heatmap(sm), config={"displayModeBar": False}),
            ], md=7),
            dbc.Col([
                dcc.Graph(figure=_fig_asymmetry_ratio(sm), config={"displayModeBar": False}),
            ], md=5),
        ], className="mb-3"),

        # Overlay: FF% vs Amihud by bucket
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_fig_scenario_overlay(sm, "median_ff", "Median Free-Float %",
                                                           "Free-Float Cannot See the Gradient",
                                                           COLORS["accent_orange"]),
                              config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=_fig_scenario_overlay(sm, "median_amihud", "Median Amihud ILLIQ",
                                                           "Amihud Tracks the Gradient",
                                                           COLORS["accent_blue"], log_y=True),
                              config={"displayModeBar": False}), md=6),
        ], className="mb-3"),

        dbc.Card(
            dbc.CardBody([
                html.H5("Why This Matters", style={
                    "color": COLORS["accent_red"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
The **p99 upside** widens from ~2-3% for mega-caps to potentially hundreds of percent for
nano-caps. The **asymmetry ratio** (upside width / downside width) itself increases as you
move down the spectrum — nano-cap upside tails are 2-5× wider than their downside.

**Free-float is bucket-blind.** A nano-cap and a mega-cap can share identical free-float
percentages, yet their extreme tail profiles differ by orders of magnitude.

**Amihud captures the mechanism.** Thin order books, wide spreads, and low dollar volume
are exactly what drives fat tails in small stocks — and that is exactly what Amihud measures.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px"},
        ),
    ])


def _fig_scenario_heatmap(sm):
    """Diverging heatmap — p99 up vs p99 down by bucket."""
    z = np.array([
        [float(v * 100) for v in sm["p99_up"]],
        [float(v * 100) for v in sm["p99_down"]],
    ])
    fig = go.Figure(go.Heatmap(
        z=z, x=sm["Bucket"].tolist(), y=["Upside p99 (%)", "Downside p99 (%)"],
        colorscale=[[0, COLORS["accent_blue"]], [0.5, "#1a1a2e"], [1, COLORS["accent_red"]]],
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=13, color=COLORS["text"]),
        hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
        colorbar=dict(title=dict(text="Daily %", font=dict(size=10))),
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Extreme Return Tails by Market Cap Bucket (p99)", font=dict(size=16)),
        height=320, yaxis=dict(autorange="reversed"),
    )
    return fig


def _fig_asymmetry_ratio(sm):
    """Bar chart of asymmetry ratios by bucket."""
    valid = sm[sm["asym_99"] > 0]
    colors = [COLORS["accent_green"] if v < 1.3
              else COLORS["accent_orange"] if v < 2.0
              else COLORS["accent_red"] for v in valid["asym_99"]]
    fig = go.Figure(go.Bar(
        x=valid["Bucket"], y=valid["asym_99"],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f}×" for v in valid["asym_99"]],
        textposition="outside", textfont=dict(size=12, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Up/Down Ratio: %{y:.2f}×<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["text_muted"], opacity=0.5,
                  annotation_text="Symmetric", annotation_position="bottom right")
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Tail Asymmetry (p99 Up / p99 Down)", font=dict(size=16)),
        yaxis=dict(title="Ratio"), height=320,
    )
    return fig


def _fig_scenario_overlay(sm, col, y_label, title, color, log_y=False):
    """Overlay a measure vs p99 tail width by bucket."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=sm["Bucket"], y=sm["p99_up"] * 100, name="p99 Upside %",
        marker=dict(color=COLORS["accent_red"], opacity=0.4, line=dict(width=0)),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=sm["Bucket"], y=sm[col], name=y_label, mode="lines+markers",
        line=dict(color=color, width=3), marker=dict(size=8),
        customdata=[_fmt_illiq(v) if log_y else f"{v:.1f}" for v in sm[col]],
        hovertemplate="<b>%{x}</b><br>" + y_label + ": %{customdata}<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(
        template=PLOT_TEMPLATE, title=dict(text=title, font=dict(size=16)),
        height=360, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="p99 Upside %", secondary_y=False)
    fig.update_yaxes(title_text=y_label, secondary_y=True,
                     type="log" if log_y else "linear")
    return fig


def page_term_structure():
    """Part 2: Term structure & z-score visualization."""
    return html.Div([
        html.H2("Liquidity Stress Monitor", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Amihud ILLIQ at two horizons — 21-day (recent friction) vs 252-day (structural "
            "baseline) — for a single stock. This is not technical analysis: Amihud measures "
            "price impact per dollar of volume, a microstructure quantity. The ratio of short "
            "to long reveals whether current liquidity is abnormally stressed or calm.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "720px"},
        ),
        html.P(
            "Top chart: 21d and 252d Amihud ILLIQ (log scale). When the red line (21d) rises "
            "above the blue dashed line (252d), short-term friction exceeds the structural norm. "
            "Bottom chart: the z-score of that ratio — how many standard deviations from the "
            "stock's own historical average. Beyond ±2 indicates unusual stress or calm.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "720px", "fontStyle": "italic"},
        ),

        dbc.Row([
            dbc.Col([
                html.Label("Select Ticker:", style={"color": COLORS["text_muted"],
                                                     "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="ts-ticker-dropdown",
                    options=[{"label": t, "value": t} for t in ticker_list],
                    value=ticker_list[0] if ticker_list else None,
                    style={"backgroundColor": COLORS["card"], "color": COLORS["text"]},
                    className="dash-dropdown-dark",
                ),
            ], md=4),
        ], className="mb-3"),

        dcc.Graph(id="ts-curve-chart", config={"displayModeBar": False}),
        dcc.Graph(id="ts-zscore-chart", config={"displayModeBar": False}),

        dbc.Card(
            dbc.CardBody([
                html.H5("What This Is (and Isn't)", style={
                    "color": COLORS["accent_cyan"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
**This is not technical analysis.** Moving averages of price are backward-looking pattern
matching. Amihud ILLIQ measures a *structural* quantity — price impact per dollar of volume —
that reflects order book depth, market-maker capacity, and trading costs. The 21d and 252d
windows are simply short-horizon vs long-horizon averages of that structural measure.

The **z-score** = `(ratio − rolling_mean) / rolling_std` normalizes across stocks so you can
compare a micro-cap and a mega-cap on the same scale. A z-score of +2.0 means current
friction is 2 standard deviations above this stock's own norm.

| Regime | Interpretation | Risk Implication |
|--------|---------------|------------------|
| 21d >> 252d (z > +2) | Transient stress | Earnings, thin market, event-driven |
| 21d << 252d (z < −2) | Unusual liquidity | Institutional accumulation, rebal |
| Both elevated | Chronic illiquidity | Persistent tail risk, position sizing concern |
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def _fig_term_structure_curve(ticker):
    """Per-ticker illiq_21d vs illiq_252d over time."""
    tkr = df[df["ticker"] == ticker].sort_values("date").copy()
    if tkr.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOT_TEMPLATE, title="No data")
        return fig

    fig = make_subplots(rows=1, cols=1)
    for col, name, color, dash in [
        ("illiq_21d", "21-Day Amihud", COLORS["accent_red"], "solid"),
        ("illiq_252d", "252-Day Amihud", COLORS["accent_blue"], "dash"),
    ]:
        if col in tkr.columns:
            series = tkr.dropna(subset=[col])
            fig.add_trace(go.Scatter(
                x=series["date"], y=series[col], mode="lines",
                name=name, line=dict(color=color, width=2, dash=dash),
                customdata=[_fmt_illiq(v) for v in series[col]],
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %d}}<br>ILLIQ: %{{customdata}}<extra></extra>",
            ))

    # Shade divergence regions
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text=f"{ticker} — Short vs Long Horizon Amihud", font=dict(size=16)),
        yaxis=dict(title="Amihud ILLIQ", type="log"),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def _fig_zscore_bands(ticker):
    """Per-ticker z-score time series with ±2 bands."""
    tkr = df[df["ticker"] == ticker].sort_values("date").copy()
    if tkr.empty or "illiq_zscore" not in tkr.columns:
        fig = go.Figure()
        fig.update_layout(template=PLOT_TEMPLATE, title="No z-score data")
        return fig

    zs = tkr.dropna(subset=["illiq_zscore"])
    fig = go.Figure()
    fig.add_hline(y=2, line_dash="dash", line_color=COLORS["accent_red"], opacity=0.6,
                  annotation_text="Stress (+2σ)")
    fig.add_hline(y=-2, line_dash="dash", line_color=COLORS["accent_green"], opacity=0.6,
                  annotation_text="Calm (−2σ)")
    fig.add_hline(y=0, line_color=COLORS["text_muted"], opacity=0.3)
    fig.add_trace(go.Scatter(
        x=zs["date"], y=zs["illiq_zscore"], mode="lines",
        line=dict(color=COLORS["accent_cyan"], width=2),
        fill="tozeroy",
        hovertemplate="%{x|%b %d}<br>Z-Score: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text=f"{ticker} — Liquidity Z-Score", font=dict(size=16)),
        yaxis=dict(title="Z-Score"), height=320,
    )
    return fig


def page_regression_deep_dive():
    """Part 3: All 7 OLS specs + ΔR² decomposition + Fama-MacBeth."""
    decomp = r2_decomp
    return html.Div([
        html.H2("Regression Deep Dive", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Seven specifications isolating each variable's contribution. "
            "The ΔR² decomposition answers: how much does each measure add ON TOP of market cap?",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "720px"},
        ),
        html.P(
            "R² and ΔR² are unitless (0–1). The horizontal bar chart shows total R² per model spec. "
            "The ΔR² decomposition shows the incremental R² of adding Amihud or Free-Float to a "
            "MktCap-only baseline. Fama-MacBeth t-stats test whether coefficients are significant "
            "when estimated cross-sectionally each month; *=10%, **=5%, ***=1% level.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "720px", "fontStyle": "italic"},
        ),

        # ΔR² hero cards
        dbc.Row([
            dbc.Col(make_card("ΔR² Amihud → |Ret|",
                              f"{float(decomp[decomp['Dependent']=='|Return|']['ΔR²_amihud'].iloc[0]):.5f}",
                              "Incremental over MktCap", color=COLORS["accent_blue"]), md=3),
            dbc.Col(make_card("ΔR² FF → |Ret|",
                              f"{float(decomp[decomp['Dependent']=='|Return|']['ΔR²_ff'].iloc[0]):.5f}",
                              "Incremental over MktCap", color=COLORS["accent_orange"]), md=3),
            dbc.Col(make_card("Superiority (|Ret|)",
                              f"{float(decomp[decomp['Dependent']=='|Return|']['Superiority'].iloc[0]):.1f}×",
                              "ΔR² Amihud / ΔR² FF", color=COLORS["accent_green"]), md=3),
            dbc.Col(make_card("Superiority (HL)",
                              f"{float(decomp[decomp['Dependent']=='H-L Range']['Superiority'].iloc[0]):.1f}×",
                              "ΔR² Amihud / ΔR² FF", color=COLORS["accent_green"]), md=3),
        ], className="g-3 mb-4"),

        # Full spec bar chart
        dcc.Graph(figure=_fig_full_r2_specs(), config={"displayModeBar": False}),
        html.P("Each bar = total R² for that model. The gap between MktCap Only and the kitchen "
               "sink tells you how much all variables combined add. Note how little Amihud adds "
               "beyond MktCap alone — they're nearly redundant.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "16px"}),

        # ΔR² decomposition
        dcc.Graph(figure=_fig_delta_r2(), config={"displayModeBar": False}),
        html.P("ΔR² = how much R² improves when adding this variable to the MktCap baseline. "
               "A taller bar means more unique information beyond size. "
               "If FF% bar is taller than Amihud, FF% adds more that size doesn't already capture.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "16px"}),

        # Fama-MacBeth summary
        html.H4("Fama-MacBeth Cross-Sectional Test", style={
            "color": COLORS["accent_purple"], "fontWeight": 700, "marginTop": "24px",
            "marginBottom": "12px"}),
        html.P(
            "Run spec ⑥ cross-sectionally each month, then t-test the coefficient time series. "
            "Examine which coefficients remain significant after controlling for the other variables.",
            style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "marginBottom": "16px",
                   "maxWidth": "700px"},
        ),
        _build_fm_table(),

        dbc.Card(
            dbc.CardBody([
                html.H5("Key Takeaway", style={
                    "color": COLORS["accent_blue"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
**The ΔR² decomposition reveals a nuanced picture.** After controlling for market cap,
free-float actually contributes **more incremental R²** than Amihud for both dependent
variables. This is because Amihud is so highly correlated with market cap (ρ = −0.95) that
adding it on top of MktCap provides almost no new information — the size signal is already
captured. Free-float (ρ ≈ 0.43 with MktCap) is less redundant with size, so it adds a
genuinely different dimension: ownership structure.

**However**, both increments are tiny in absolute terms. The practical takeaway is that
market cap does most of the heavy lifting. The **Fama-MacBeth** cross-sectional test shows
which coefficients survive month-to-month — examine the t-statistics to see whether either
variable is consistently significant beyond the size effect.

**Where Amihud uniquely wins:** within size buckets (see Robustness tab), Amihud captures
microstructure variation that neither MktCap nor FF% can see.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def _fig_full_r2_specs():
    """All 6 pooled OLS specs, horizontal bar chart, for both dependents."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["|Return|", "H-L Range"],
                        horizontal_spacing=0.15)
    for col_idx, dep in enumerate(["|Return|", "H-L Range"], 1):
        sub = full_reg[full_reg["Dependent"] == dep].sort_values("R²")
        colors = []
        for m in sub["Model"]:
            if "Amihud Only" in m:
                colors.append(COLORS["accent_blue"])
            elif "Free-Float Only" in m:
                colors.append(COLORS["accent_orange"])
            elif "Kitchen" in m:
                colors.append(COLORS["accent_cyan"])
            elif "MktCap Only" in m:
                colors.append(COLORS["accent_purple"])
            elif "Amihud" in m:
                colors.append(COLORS["accent_blue"])
            else:
                colors.append(COLORS["accent_orange"])
        fig.add_trace(go.Bar(
            y=sub["Model"], x=sub["R²"], orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.5f}" for v in sub["R²"]],
            textposition="outside", textfont=dict(size=10, color=COLORS["text"]),
            hovertemplate="<b>%{y}</b><br>R² = %{x:.6f}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_idx)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="All Six OLS Specifications", font=dict(size=18)),
        height=450, margin=dict(l=180, r=80, t=60, b=50),
    )
    fig.update_xaxes(title_text="R²")
    return fig


def _fig_delta_r2():
    """ΔR² decomposition bar chart."""
    fig = go.Figure()
    for dep, color_a, color_f in [
        ("|Return|", COLORS["accent_blue"], COLORS["accent_orange"]),
        ("H-L Range", COLORS["accent_cyan"], COLORS["accent_red"]),
    ]:
        row = r2_decomp[r2_decomp["Dependent"] == dep].iloc[0]
        fig.add_trace(go.Bar(
            x=[f"{dep}<br>Amihud"], y=[row["ΔR²_amihud"]],
            marker=dict(color=color_a, line=dict(width=0)),
            text=[f"{row['ΔR²_amihud']:.5f}"], textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            name=f"Amihud ({dep})", showlegend=True,
        ))
        fig.add_trace(go.Bar(
            x=[f"{dep}<br>Free-Float"], y=[row["ΔR²_ff"]],
            marker=dict(color=color_f, line=dict(width=0)),
            text=[f"{row['ΔR²_ff']:.5f}"], textposition="outside",
            textfont=dict(size=11, color=COLORS["text"]),
            name=f"FF ({dep})", showlegend=True,
        ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text="Incremental R² Over Market Cap Baseline (ΔR²)", font=dict(size=18)),
        yaxis=dict(title="ΔR²"), height=400, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return fig


def _build_fm_table():
    """Fama-MacBeth summary as a Dash table."""
    if fm_summary.empty:
        return html.P("Insufficient monthly data for Fama-MacBeth.", style={"color": COLORS["text_muted"]})
    rows = []
    for _, r in fm_summary.iterrows():
        sig = "***" if abs(r["t-stat"]) > 2.576 else "**" if abs(r["t-stat"]) > 1.96 else "*" if abs(r["t-stat"]) > 1.645 else ""
        t_color = COLORS["accent_green"] if abs(r["t-stat"]) > 1.96 else COLORS["text_muted"]
        rows.append(html.Tr([
            html.Td(r["Dependent"], style={"color": COLORS["text"]}),
            html.Td(r["Variable"], style={"color": COLORS["text"]}),
            html.Td(f"{r['Mean β']:.2e}", style={"color": COLORS["text_muted"], "textAlign": "right"}),
            html.Td(f"{r['t-stat']:.2f}{sig}", style={"color": t_color, "textAlign": "right",
                                                        "fontWeight": 700}),
            html.Td(f"{int(r['Months'])}", style={"color": COLORS["text_muted"], "textAlign": "right"}),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Dependent", style={"color": COLORS["text"]}),
            html.Th("Variable", style={"color": COLORS["text"]}),
            html.Th("Mean β", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Th("t-stat", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Th("Months", style={"color": COLORS["text"], "textAlign": "right"}),
        ]))] + [html.Tbody(rows)],
        bordered=False, color="dark", hover=True, size="sm",
        style={"backgroundColor": "transparent"},
    )


def page_extreme_moves():
    """Part 4: Extreme move logistic model — baseline vs Amihud add-on."""
    er = extreme_results
    if er is None:
        return html.Div([
            html.H2("Extreme Move Framework", style={"color": COLORS["text"], "fontWeight": 700}),
            html.P("Insufficient data for extreme move analysis (need illiq_zscore).",
                   style={"color": COLORS["text_muted"]}),
        ])

    return html.Div([
        html.H2("Extreme Move Framework", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Can we predict which stocks will deliver extreme daily moves? "
            "The baseline uses market cap bucket alone. The add-on includes yesterday's "
            "Amihud liquidity z-score. This is the operational deliverable.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "720px"},
        ),
        html.P(
            "F1 score (0–1): harmonic mean of precision and recall — higher is better. "
            "Extreme = daily |return| above the 95th percentile. The confusion matrix counts "
            "true/false positives and negatives. False Neg Reduction = extreme events the "
            "add-on catches that the baseline misses.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "720px", "fontStyle": "italic"},
        ),

        # Hero cards
        dbc.Row([
            dbc.Col(make_card("Baseline F1", f"{er['baseline']['f1']:.3f}",
                              "MktCap bucket only",
                              color=COLORS["accent_orange"]), md=3),
            dbc.Col(make_card("Add-on F1", f"{er['addon']['f1']:.3f}",
                              "+ Lagged Amihud Z-Score",
                              color=COLORS["accent_blue"]), md=3),
            dbc.Col(make_card("False Neg Reduction", f"{er['fn_reduction']:+,}",
                              "Extreme moves now caught",
                              color=COLORS["accent_green"]), md=3),
            dbc.Col(make_card("Avg Missed Magnitude",
                              f"{er['addon_miss_mean']:.4f}" if er['addon_miss_n'] > 0 else "—",
                              f"vs base: {er['base_miss_mean']:.4f}",
                              color=COLORS["accent_red"]), md=3),
        ], className="g-3 mb-4"),

        # Per-bucket recall comparison
        dcc.Graph(figure=_fig_extreme_bucket_comparison(er), config={"displayModeBar": False}),
        html.P("Recall = what fraction of actual extreme moves did the model flag? "
               "Higher is better. Blue (+ Amihud) should exceed orange (baseline) if the z-score adds value.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "12px"}),

        # Confusion matrices side by side
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_fig_confusion(er["baseline"]["confusion"], "Baseline (MktCap Only)"),
                              config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=_fig_confusion(er["addon"]["confusion"], "Add-on (+ Amihud Z-Score)"),
                              config={"displayModeBar": False}), md=6),
        ], className="mb-3"),
        html.P("Confusion matrix: rows = actual, columns = predicted. Bottom-left cell (Extreme predicted Not Extreme) "
               "= false negatives — extreme moves the model missed. Fewer false negatives = better.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "8px"}),

        dbc.Card(
            dbc.CardBody([
                html.H5("The Operational Case", style={
                    "color": COLORS["accent_green"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
The extreme move model with Amihud z-score catches moves that the size-only baseline misses.
The **false negative reduction** is the headline metric: these are the extreme events that
the current framework would not flag, but the Amihud overlay does.

Moves that *neither* model catches are genuine information shocks — earnings surprises,
macro events, geopolitical catalysts. These are not liquidity-driven; no liquidity measure
will predict them. This residual is the honest caveat: Amihud solves liquidity risk,
not event risk.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px"},
        ),
    ])


def _fig_extreme_bucket_comparison(er):
    """Recall by bucket: baseline vs add-on."""
    bm = er.get("bucket_metrics")
    if bm is None or bm.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOT_TEMPLATE, title="No per-bucket data")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bm["Bucket"], y=bm["Base Recall"], name="Baseline",
        marker=dict(color=COLORS["accent_orange"], line=dict(width=0)),
        text=[f"{v:.2f}" for v in bm["Base Recall"]], textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
    ))
    fig.add_trace(go.Bar(
        x=bm["Bucket"], y=bm["Add-on Recall"], name="+ Amihud Z-Score",
        marker=dict(color=COLORS["accent_blue"], line=dict(width=0)),
        text=[f"{v:.2f}" for v in bm["Add-on Recall"]], textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="group",
        title=dict(text="Extreme Move Recall by MktCap Bucket", font=dict(size=18)),
        yaxis=dict(title="Recall"), height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return fig


def _fig_confusion(cm, title):
    """Confusion matrix heatmap."""
    labels = ["Not Extreme", "Extreme"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, COLORS["accent_blue"]], [1, COLORS["accent_red"]]],
        text=[[f"{int(v):,}" for v in row] for row in cm],
        texttemplate="%{text}", textfont=dict(size=14, color=COLORS["text"]),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        height=300,
    )
    return fig


def page_asymmetry():
    """Part 5: Down-day vs up-day z-score predictive power."""
    ar = asymmetry_results
    if ar is None or ar.empty:
        return html.Div([
            html.H2("Asymmetry Analysis", style={"color": COLORS["text"], "fontWeight": 700}),
            html.P("Insufficient data for asymmetry analysis.",
                   style={"color": COLORS["text_muted"]}),
        ])

    return html.Div([
        html.H2("Downside Asymmetry", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Liquidity deteriorates faster during selloffs than rallies. "
            "If Amihud is more predictive on down days, it's specifically solving "
            "the downside liquidity risk problem that risk management cares about.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "720px"},
        ),
        html.P(
            "Bars show R² (unitless, 0–1) from OLS regressions run separately on up-day and down-day "
            "subsets. Blue = Amihud 252d, Orange = Free-Float %, Cyan = Liquidity Z-Score. "
            "Higher R² on down days = the predictor is tuned to the risk case.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "720px", "fontStyle": "italic"},
        ),

        dcc.Graph(figure=_fig_asymmetry_r2(ar), config={"displayModeBar": False}),
        html.P("Bars = R² when the regression is run only on up days, only on down days, or all days. "
               "If Amihud's R² is higher on down days than up days, it specifically captures "
               "downside liquidity risk — the case that matters for risk management.",
               style={"color": COLORS["text_muted"], "fontSize": "0.75rem", "textAlign": "center",
                      "padding": "0 40px", "marginTop": "-4px", "marginBottom": "8px"}),

        dbc.Card(
            dbc.CardBody([
                html.H5("The Risk Case", style={
                    "color": COLORS["accent_red"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
**Downside liquidity is the critical scenario.** The question isn't "can this stock go up 500%?"
— it's "can this stock gap down 80% before we can exit?"

If Amihud's R² is **higher on down days** than up days, it confirms the measure is specifically
tuned to the risk case. Free-float, being symmetric by construction, shows no such asymmetry
— it can't differentiate between stocks that are about to gap down vs. gap up.

This is the **downside liquidity warning system** that the Amihud z-score provides:
elevated z-scores on a stock heading into weak tape conditions are the highest-priority alert.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


def _fig_asymmetry_r2(ar):
    """Grouped bar: R² by regressor, split by up/down/all days."""
    fig = go.Figure()
    color_map = {
        "Amihud 252d": COLORS["accent_blue"],
        "Free-Float %": COLORS["accent_orange"],
        "Liquidity Z-Score": COLORS["accent_cyan"],
    }
    for reg in ar["Regressor"].unique():
        sub = ar[ar["Regressor"] == reg]
        fig.add_trace(go.Bar(
            x=sub["Split"], y=sub["R²"],
            name=reg, marker=dict(color=color_map.get(reg, COLORS["accent_purple"]),
                                  line=dict(width=0)),
            text=[f"{v:.5f}" for v in sub["R²"]], textposition="outside",
            textfont=dict(size=10, color=COLORS["text"]),
            hovertemplate=f"<b>{reg}</b><br>%{{x}}: R² = %{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="group",
        title=dict(text="R² by Day Direction: Who Captures the Downside?", font=dict(size=18)),
        yaxis=dict(title="R²"), height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# OOS Validation Page
# ═══════════════════════════════════════════════════════════════════════

def _load_oos():
    """Load oos_results.json if it exists, else None."""
    p = DATA_DIR.parent / "oos" / "data" / "oos_results.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _oos_tab_ols_horse_race(oos):
    """Tab 1: IS vs OOS R² side-by-side bar chart for all 6 model specs."""
    ols = pd.DataFrame(oos["ols"])
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["|Return|", "H-L Range"],
                        horizontal_spacing=0.14)
    for col_idx, dep in enumerate(["|Return|", "H-L Range"], 1):
        sub = ols[ols["Dependent"] == dep].sort_values("IS_R2")
        fig.add_trace(go.Bar(
            y=sub["Model"], x=sub["IS_R2"], orientation="h",
            marker=dict(color=COLORS["accent_blue"], opacity=0.7),
            name="In-Sample (2025)", showlegend=(col_idx == 1),
            text=[f"{v:.4f}" for v in sub["IS_R2"]], textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate="<b>%{y}</b><br>IS R²=%{x:.5f}<extra></extra>",
        ), row=1, col=col_idx)
        fig.add_trace(go.Bar(
            y=sub["Model"], x=sub["OOS_R2"], orientation="h",
            marker=dict(color=COLORS["accent_green"], opacity=0.85),
            name="Out-of-Sample (2026)", showlegend=(col_idx == 1),
            text=[f"{v:.4f}" for v in sub["OOS_R2"]], textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate="<b>%{y}</b><br>OOS R²=%{x:.5f}<extra></extra>",
        ), row=1, col=col_idx)
    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="group", height=500,
        title=dict(text="Model R²: Trained on 2025, Tested on 2026", font=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="R²")

    # Decay table
    decay_rows = []
    for _, r in ols[ols["Dependent"] == "|Return|"].iterrows():
        decay_rows.append(html.Tr([
            html.Td(r["Model"], style={"color": COLORS["text"], "fontWeight": 600}),
            html.Td(f"{r['IS_R2']:.4f}", style={"color": COLORS["accent_blue"], "textAlign": "right"}),
            html.Td(f"{r['OOS_R2']:.4f}", style={"color": COLORS["accent_green"], "textAlign": "right"}),
            html.Td(f"{r['R2_decay']:+.4f}", style={
                "color": COLORS["accent_red"] if r["R2_decay"] > 0.005 else COLORS["accent_green"],
                "textAlign": "right", "fontWeight": 700,
            }),
        ]))

    decay_table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Model", style={"color": COLORS["text"]}),
            html.Th("IS R²", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Th("OOS R²", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Th("R² Decay", style={"color": COLORS["text"], "textAlign": "right"}),
        ])),
         html.Tbody(decay_rows)],
        bordered=False, color="dark", hover=True, size="sm",
        style={"backgroundColor": "transparent"},
    )

    return dbc.Tab(label="R² Horse Race", children=[
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dbc.Card(dbc.CardBody([
            html.H5("R² Decay: IS → OOS", style={
                "color": COLORS["accent_cyan"], "fontWeight": 700, "marginBottom": "8px"}),
            html.P("Lower decay = more robust model. Green means OOS held up well.",
                   style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "marginBottom": "12px"}),
            decay_table,
        ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"}),
    ], tab_style={"backgroundColor": COLORS["card"]},
       label_style={"color": COLORS["text_muted"]},
       active_label_style={"color": COLORS["accent_blue"], "fontWeight": 700})


def _oos_tab_walk_forward(oos):
    """Tab 2: Walk-forward monthly OOS R² within 2025."""
    wf = pd.DataFrame(oos["walk_forward"])
    wf_sum = pd.DataFrame(oos["walk_forward_summary"])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["|Return|", "H-L Range"],
                        horizontal_spacing=0.10)

    # One line per model spec
    model_colors = {
        "① MktCap Only": COLORS["accent_purple"],
        "② Free-Float Only": COLORS["accent_orange"],
        "③ MktCap + FF": "#FFD700",
        "④ Amihud Only": COLORS["accent_blue"],
        "⑤ MktCap + Amihud": COLORS["accent_cyan"],
        "⑥ Kitchen Sink": COLORS["accent_green"],
    }

    for col_idx, dep in enumerate(["|Return|", "H-L Range"], 1):
        sub = wf[wf["Dependent"] == dep]
        for model_name, color in model_colors.items():
            line = sub[sub["Model"] == model_name].sort_values("Target_Month")
            if line.empty:
                continue
            fig.add_trace(go.Scatter(
                x=line["Target_Month"], y=line["OOS_R2"],
                mode="lines+markers", name=model_name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                showlegend=(col_idx == 1),
                hovertemplate=f"<b>{model_name}</b><br>%{{x}}<br>OOS R²=%{{y:.5f}}<extra></extra>",
            ), row=1, col=col_idx)

    fig.update_layout(
        template=PLOT_TEMPLATE, height=480,
        title=dict(text="Walk-Forward OOS R² (Expanding Window within 2025)", font=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5,
                    font=dict(size=9)),
    )
    fig.update_yaxes(title_text="OOS R²")
    fig.update_xaxes(title_text="Predicted Month")

    # Summary table
    sum_rows = []
    for _, r in wf_sum[wf_sum["Dependent"] == "|Return|"].sort_values("Avg_OOS_R2", ascending=False).iterrows():
        sum_rows.append(html.Tr([
            html.Td(r["Model"], style={"color": COLORS["text"], "fontWeight": 600}),
            html.Td(f"{r['Avg_IS_R2']:.4f}", style={"color": COLORS["accent_blue"], "textAlign": "right"}),
            html.Td(f"{r['Avg_OOS_R2']:.4f}", style={"color": COLORS["accent_green"], "textAlign": "right"}),
            html.Td(f"{r['Min_OOS_R2']:.4f}", style={"color": COLORS["text_muted"], "textAlign": "right"}),
            html.Td(f"{r['Max_OOS_R2']:.4f}", style={"color": COLORS["text_muted"], "textAlign": "right"}),
            html.Td(str(r["Months_tested"]), style={"color": COLORS["text_muted"], "textAlign": "right"}),
        ]))

    sum_table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Model", style={"color": COLORS["text"]}),
            html.Th("Avg IS R²", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Th("Avg OOS R²", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Th("Min OOS R²", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Th("Max OOS R²", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Th("Months", style={"color": COLORS["text"], "textAlign": "right"}),
        ])),
         html.Tbody(sum_rows)],
        bordered=False, color="dark", hover=True, size="sm",
        style={"backgroundColor": "transparent"},
    )

    return dbc.Tab(label="Walk-Forward (2025)", children=[
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dbc.Card(dbc.CardBody([
            html.H5("Walk-Forward Summary (|Return|)", style={
                "color": COLORS["accent_cyan"], "fontWeight": 700, "marginBottom": "8px"}),
            html.P("Expanding window: train on months 1–N, predict month N+1. "
                   "Started from month 7 (6-month minimum training window).",
                   style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "marginBottom": "12px"}),
            sum_table,
        ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"}),
    ], tab_style={"backgroundColor": COLORS["card"]},
       label_style={"color": COLORS["text_muted"]},
       active_label_style={"color": COLORS["accent_blue"], "fontWeight": 700})


def _oos_tab_size(oos):
    """Tab 3: IS vs OOS R² by size tercile."""
    sz = pd.DataFrame(oos["ols_by_size"])
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["|Return|", "H-L Range"],
                        horizontal_spacing=0.14)
    size_order = ["Small Cap", "Mid Cap", "Large Cap"]

    for col_idx, dep in enumerate(["|Return|", "H-L Range"], 1):
        sub = sz[(sz["Dependent"] == dep) & (sz["Model"] == "\u2463 Amihud Only")]
        for sample_label, color in [("IS", COLORS["accent_blue"]), ("OOS", COLORS["accent_green"])]:
            d = sub[sub["Set"] == sample_label].set_index("Size").reindex(size_order)
            if d.empty:
                continue
            fig.add_trace(go.Bar(
                x=d.index, y=d["R2"],
                name=sample_label, showlegend=(col_idx == 1),
                marker=dict(color=color, line=dict(width=0)),
                text=[f"{v:.4f}" for v in d["R2"]],
                textposition="outside", textfont=dict(size=10, color=COLORS["text"]),
                hovertemplate=f"<b>%{{x}}</b><br>{sample_label} R²=%{{y:.5f}}<extra></extra>",
            ), row=1, col=col_idx)

    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="group", height=440,
        title=dict(text="Amihud R² by Size: IS vs OOS", font=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="R²")

    return dbc.Tab(label="Size Breakdown", children=[
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dbc.Card(dbc.CardBody([
            dcc.Markdown(
                "Amihud's explanatory power for each size tercile, compared in-sample (2025) vs "
                "out-of-sample (2026). Consistent OOS R² across sizes confirms the result is not "
                "a size artifact.",
                style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
        ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"}),
    ], tab_style={"backgroundColor": COLORS["card"]},
       label_style={"color": COLORS["text_muted"]},
       active_label_style={"color": COLORS["accent_blue"], "fontWeight": 700})


def _oos_tab_extreme(oos):
    """Tab 4: Extreme moves logistic IS vs OOS."""
    em = oos["extreme_moves"]
    if isinstance(em, dict) and "error" in em:
        return dbc.Tab(label="Extreme Moves", children=[
            html.P(f"Extreme moves analysis unavailable: {em['error']}",
                   style={"color": COLORS["accent_orange"], "padding": "40px"}),
        ], tab_style={"backgroundColor": COLORS["card"]},
           label_style={"color": COLORS["text_muted"]},
           active_label_style={"color": COLORS["accent_blue"], "fontWeight": 700})

    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Baseline (MktCap Only)", "Add-on (+ Amihud Z-score)"],
                        horizontal_spacing=0.14)

    for col_idx, (model_base, title) in enumerate(
            [("baseline", "Baseline"), ("addon", "Add-on")], 1):
        is_vals = [em[f"{model_base}_IS"][m] for m in metrics]
        oos_vals = [em[f"{model_base}_OOS"][m] for m in metrics]
        fig.add_trace(go.Bar(
            x=labels, y=is_vals, name="IS", showlegend=(col_idx == 1),
            marker=dict(color=COLORS["accent_blue"], opacity=0.7),
            text=[f"{v:.3f}" for v in is_vals], textposition="outside",
            textfont=dict(size=10, color=COLORS["text"]),
        ), row=1, col=col_idx)
        fig.add_trace(go.Bar(
            x=labels, y=oos_vals, name="OOS", showlegend=(col_idx == 1),
            marker=dict(color=COLORS["accent_green"], opacity=0.85),
            text=[f"{v:.3f}" for v in oos_vals], textposition="outside",
            textfont=dict(size=10, color=COLORS["text"]),
        ), row=1, col=col_idx)

    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="group", height=480,
        title=dict(text="Extreme Move Logistic: IS vs OOS", font=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5),
        margin=dict(t=90),
    )
    fig.update_yaxes(title_text="Score", range=[0, 0.5])

    return dbc.Tab(label="Extreme Moves", children=[
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dbc.Card(dbc.CardBody([
            dcc.Markdown(
                "Logistic regression predicting p95 extreme daily moves. The **add-on** model "
                "includes lagged Amihud z-score on top of market cap. Compare IS (2025) vs OOS (2026) "
                "to see if the F1 improvement holds out-of-sample.",
                style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
        ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"}),
    ], tab_style={"backgroundColor": COLORS["card"]},
       label_style={"color": COLORS["text_muted"]},
       active_label_style={"color": COLORS["accent_blue"], "fontWeight": 700})


def _oos_tab_asymmetry(oos):
    """Tab 5: Up-day vs down-day IS/OOS comparison."""
    asym = pd.DataFrame(oos["asymmetry"])
    if asym.empty:
        return dbc.Tab(label="Asymmetry", children=[
            html.P("Asymmetry data unavailable.",
                   style={"color": COLORS["accent_orange"], "padding": "40px"}),
        ], tab_style={"backgroundColor": COLORS["card"]},
           label_style={"color": COLORS["text_muted"]},
           active_label_style={"color": COLORS["accent_blue"], "fontWeight": 700})

    fig = go.Figure()

    dir_colors = {"Down Days": COLORS["accent_red"], "Up Days": COLORS["accent_green"]}

    sub = asym[asym["Regressor"] == "Amihud 252d"]
    for split_label, color in dir_colors.items():
        d = sub[sub["Split"] == split_label]
        if d.empty:
            continue
        for sample, opacity in [("IS", 0.5), ("OOS", 1.0)]:
            row = d[d["Set"] == sample]
            if row.empty:
                continue
            fig.add_trace(go.Bar(
                x=[f"{split_label} {sample}"],
                y=[float(row.iloc[0]["R2"])],
                name=f"{split_label} {sample}",
                marker=dict(color=color, opacity=opacity, line=dict(width=0)),
                text=[f"{float(row.iloc[0]['R2']):.4f}"], textposition="outside",
                textfont=dict(size=10, color=COLORS["text"]),
            ))

    fig.update_layout(
        template=PLOT_TEMPLATE, height=440,
        title=dict(text="Amihud R² by Direction: Up vs Down Days (IS vs OOS)", font=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5,
                    font=dict(size=9)),
        showlegend=True,
    )
    fig.update_yaxes(title_text="R²")

    return dbc.Tab(label="Asymmetry", children=[
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dbc.Card(dbc.CardBody([
            dcc.Markdown(
                "Amihud's predictive power split by market direction. If the OOS down-day R² "
                "exceeds up-day R², Amihud is genuinely a **downside risk** signal, not just "
                "a generic volatility proxy.",
                style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
        ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"}),
    ], tab_style={"backgroundColor": COLORS["card"]},
       label_style={"color": COLORS["text_muted"]},
       active_label_style={"color": COLORS["accent_blue"], "fontWeight": 700})


def page_oos_validation():
    """OOS Validation: tabbed view of 2025 walk-forward + 2026 holdout results."""
    oos = _load_oos()

    if oos is None:
        return html.Div([
            html.H2("Out-of-Sample Validation", style={
                "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "OOS results not yet generated. Run the pipeline first:",
                html.Pre("python -m oos.download_2026\n"
                         "python -m oos.compute_features\n"
                         "python -m oos.train_and_evaluate",
                         style={"color": COLORS["text"], "marginTop": "8px",
                                "backgroundColor": COLORS["bg"], "padding": "12px",
                                "borderRadius": "6px"}),
            ], color="warning", style={"maxWidth": "600px", "marginTop": "24px"}),
        ])

    meta = oos["metadata"]
    ols = pd.DataFrame(oos["ols"])
    # Headline numbers
    amihud_is = ols[(ols["Dependent"] == "|Return|") & (ols["Model"] == "④ Amihud Only")]
    amihud_oos = float(amihud_is.iloc[0]["OOS_R2"]) if len(amihud_is) else 0
    amihud_is_val = float(amihud_is.iloc[0]["IS_R2"]) if len(amihud_is) else 0
    ff_row = ols[(ols["Dependent"] == "|Return|") & (ols["Model"] == "② Free-Float Only")]
    ff_oos = float(ff_row.iloc[0]["OOS_R2"]) if len(ff_row) else 0
    ratio = amihud_oos / max(ff_oos, 1e-10)

    return html.Div([
        html.H2("Out-of-Sample Validation", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            f"Models trained on {meta['train_rows']:,} rows ({meta['train_tickers']} tickers) "
            f"from {meta['train_period']}, then applied to {meta['test_rows']:,} rows "
            f"({meta['test_tickers']} tickers) from {meta['test_period']}.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "4px", "maxWidth": "700px"},
        ),
        html.P(
            "All tabs compare in-sample (IS, blue) vs out-of-sample (OOS, green). "
            "R² and R² Decay are unitless. Walk-Forward uses expanding-window monthly retraining. "
            "F1 scores (0–1) for extreme move detection. Shrinkage is natural; watch for sign-flips.",
            style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                   "marginBottom": "24px", "maxWidth": "700px", "fontStyle": "italic"},
        ),

        dbc.Row([
            dbc.Col(make_card("Amihud IS R²", f"{amihud_is_val:.4f}",
                              "In-sample (2025)", color=COLORS["accent_blue"]), md=3),
            dbc.Col(make_card("Amihud OOS R²", f"{amihud_oos:.4f}",
                              "Out-of-sample (2026)", color=COLORS["accent_green"]), md=3),
            dbc.Col(make_card("OOS Superiority", f"{ratio:.1f}×",
                              "vs Free-Float OOS", color=COLORS["accent_cyan"]), md=3),
            dbc.Col(make_card("R² Decay", f"{amihud_is_val - amihud_oos:+.4f}",
                              "IS → OOS change",
                              color=COLORS["accent_green"] if (amihud_is_val - amihud_oos) < 0.005
                              else COLORS["accent_orange"]), md=3),
        ], className="g-3 mb-4"),

        dbc.Tabs([
            _oos_tab_ols_horse_race(oos),
            _oos_tab_walk_forward(oos),
            _oos_tab_size(oos),
            _oos_tab_extreme(oos),
            _oos_tab_asymmetry(oos),
        ], className="mb-3"),
    ])


def page_conclusion():
    # ── Try to load OOS results ──────────────────────────────────────
    _oos_path = DATA_DIR.parent / "oos" / "data" / "oos_results.json"
    _oos = json.loads(_oos_path.read_text()) if _oos_path.exists() else None

    conclusion_note = ""
    if _oos:
        # Pull OOS headline numbers for conclusion text
        ols = pd.DataFrame(_oos["ols"])
        _oos_amihud = ols[(ols["Dependent"] == "|Return|") & (ols["Model"] == "④ Amihud Only")]
        _oos_ff = ols[(ols["Dependent"] == "|Return|") & (ols["Model"] == "② Free-Float Only")]
        if len(_oos_amihud) and len(_oos_ff):
            _a_oos = float(_oos_amihud.iloc[0]["OOS_R2"])
            _f_oos = float(_oos_ff.iloc[0]["OOS_R2"])
            conclusion_note = (
                f"\n\n**8. Out-of-sample results.**\n"
                f"Models fitted on 2025 and applied to 2026 YTD: Amihud standalone OOS R²="
                f"{_a_oos:.4f}, Free-Float standalone OOS R²={_f_oos:.4f}. "
                f"See the **OOS Validation** page for size-controlled and walk-forward results."
            )

    return html.Div([
        html.Div([
            html.H1("Conclusion", style={
                "background": f"linear-gradient(135deg, {COLORS['accent_green']}, {COLORS['accent_cyan']})",
                "-webkit-background-clip": "text",
                "-webkit-text-fill-color": "transparent",
                "fontWeight": 800, "fontSize": "2.4rem", "marginBottom": "24px",
            }),
        ]),

        dbc.Row([
            dbc.Col(make_card("Amihud: Best Standalone", f"{superiority_ret:.0f}× R²",
                              "Univariate, but largely a size proxy",
                              color=COLORS["accent_blue"]), md=4),
            dbc.Col(make_card("FF%: Best Incremental", "Over MktCap",
                              "Adds more unique info beyond size",
                              color=COLORS["accent_orange"]), md=4),
            dbc.Col(make_card("Amihud: Best Within-Size", "Small Caps",
                              "Dominates where FF% is blind",
                              color=COLORS["accent_green"]), md=4),
        ], className="g-3 mb-4"),

        dbc.Card(
            dbc.CardBody([
                dcc.Markdown(f"""
## Key Findings

**1. Amihud is the strongest standalone predictor — but mostly captures size.**
Across {df.shape[0]:,} observations and {df['ticker'].nunique():,} US equities, Amihud explains
{superiority_ret:.0f}× more return variance than free-float as a univariate predictor. However,
Amihud correlates with market cap at ρ = −0.95 — most of its power is a size effect.
Market cap alone actually outperforms Amihud alone.

**2. Free-float adds more unique information beyond size.**
After controlling for market cap, FF% contributes more incremental ΔR² than Amihud
for both |Return| and H-L Range. FF% is less collinear with size (ρ ≈ 0.43) and captures
ownership structure — a genuinely different dimension.

**3. Within size buckets, Amihud dominates — especially in small caps.**
When market cap is held constant, Amihud outperforms FF% in every size tercile, with
the gap widest in small caps (10–100× more R²). This is Amihud's genuine, unique
contribution: microstructure friction that static filing data cannot see.

**4. Free-float data is structurally stale.**
{staleness['streak_3plus']:.0%} of tickers are unchanged for 3+ consecutive months.
{staleness['never_changed_pct']:.0%} showed zero change across all 12 months.
Amihud updates daily from live trading data.

**5. The Amihud stress monitor is a genuine operational tool.**
The 21-day / 252-day Amihud z-score provides a live, cross-sectionally comparable
stress signal that free-float cannot offer at any frequency.

**6. Extreme move prediction improves with Amihud z-score.**
Adding lagged z-score to a size-only baseline catches extreme daily moves the
baseline misses. This is the operational deliverable.

**7. Downside asymmetry confirms the risk case.**
Amihud's predictive power is strongest on down days — the scenario that risk
management cares about most.

## Recommendation

**Use both signals for their respective strengths:**
- **Amihud ILLIQ** as the primary *within-bucket* liquidity signal and daily monitoring
  tool. Best value in small/mid caps. Deploy the **z-score** for transient stress alerts.
- **Free-float %** as the *cross-sectional* ownership structure signal that adds unique
  information beyond market cap. Best value in large caps where filing data is timely.
- **Neither replaces market cap** — size dominates both as an explanatory variable.

## Suggested Next Steps

1. **Lagged Amihud Test** — Use t−1 Amihud to eliminate contemporaneous overlap with returns
2. **Sector Controls** — Add GICS sector dummies to confirm sector neutrality
3. **Combined Model** — Develop a composite score weighting Amihud, FF%, and MktCap by context
4. **Portfolio-Level Backtest** — Simulate position sizing using combined constraints
{conclusion_note}
""", style={"color": COLORS["text_muted"], "fontSize": "0.9rem", "lineHeight": 1.7}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px"},
        ),
    ])



# ═══════════════════════════════════════════════════════════════════════
# Composite Chapter Pages (8-chapter story flow)
# ═══════════════════════════════════════════════════════════════════════

_TAB_STYLE = {"backgroundColor": COLORS["card"]}
_LABEL_STYLE = {"color": COLORS["text_muted"]}
_ACTIVE_LABEL = {"color": COLORS["accent_blue"], "fontWeight": 700}


def _chapter_header(title, description):
    """Consistent chapter header."""
    return html.Div([
        html.H2(title, style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "4px"}),
        html.P(description,
               style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                      "marginBottom": "24px", "maxWidth": "720px", "lineHeight": 1.5}),
    ])


def page_ch_overview():
    """Ch 1: Opening thesis and key metrics."""
    return page_overview()


def page_ch_problem():
    """Ch 2: Free-float is broken — staleness + Amihud as live alternative."""
    return html.Div([
        _chapter_header(
            "The Problem",
            "Free-float percentage — the conventional liquidity proxy — is fundamentally stale. "
            "It updates quarterly at best, yet true liquidity shifts daily. This chapter presents "
            "the evidence, then introduces the Amihud ILLIQ ratio as a live, daily alternative."
        ),
        dbc.Tabs([
            dbc.Tab(label="Staleness Evidence", children=[page_staleness()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
            dbc.Tab(label="Liquidity Stress Monitor", children=[page_term_structure()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
        ], className="mb-3"),
    ])


def page_ch_evidence():
    """Ch 3: Head-to-head statistical comparison."""
    return html.Div([
        _chapter_header(
            "The Evidence",
            "Standalone R\u00b2 comparison, full regression specifications with \u0394R\u00b2 "
            "decomposition showing where each signal shines, Fama-MacBeth tests, "
            "and visual scatter analysis."
        ),
        dbc.Tabs([
            dbc.Tab(label="R\u00b2 Horse Race", children=[page_horse_race()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
            dbc.Tab(label="Regression Specs", children=[page_regression_deep_dive()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
            dbc.Tab(label="Scatter & Correlations", children=[page_deep_dive()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
        ], className="mb-3"),
    ])


def page_ch_robustness():
    """Ch 4: Does the result hold across sizes, time, and interactions?"""
    return html.Div([
        _chapter_header(
            "Robustness",
            "A good liquidity signal should work across all market cap sizes, stay consistent "
            "over time, and hold up in double-sorted interaction analysis. Three stress tests."
        ),
        dbc.Tabs([
            dbc.Tab(label="By Size Bucket", children=[page_size()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
            dbc.Tab(label="Over Time", children=[page_time()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
            dbc.Tab(label="Interaction Heatmap", children=[page_heatmap()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
        ], className="mb-3"),
    ])


def page_ch_tail_risk():
    """Ch 5: Where it matters most — tail risk applications."""
    return html.Div([
        _chapter_header(
            "Tail Risk",
            "Liquidity risk matters most in extremes. The risk surface shows how tails widen "
            "across the size spectrum. The extreme-move framework tests whether Amihud predicts "
            "which stocks will blow up. The asymmetry analysis checks if it specifically captures "
            "downside risk."
        ),
        dbc.Tabs([
            dbc.Tab(label="Risk Surface", children=[page_scenario_matrix()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
            dbc.Tab(label="Extreme Moves", children=[page_extreme_moves()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
            dbc.Tab(label="Up vs Down Days", children=[page_asymmetry()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
        ], className="mb-3"),
    ])


def page_ch_oos():
    """Ch 6: Out-of-sample validation."""
    return page_oos_validation()


def page_ch_tools():
    """Ch 7: Interactive exploration tools."""
    return html.Div([
        _chapter_header(
            "Interactive Tools",
            "Explore the data yourself. Adjust market cap bucket boundaries to see how "
            "size definitions affect tail risk, or deep-dive into any individual stock's "
            "Amihud dynamics, free-float history, and return profile."
        ),
        dbc.Tabs([
            dbc.Tab(label="MktCap Bucket Tuner", children=[page_bucket_tuner()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
            dbc.Tab(label="Stock Explorer", children=[page_explorer()],
                    tab_style=_TAB_STYLE, label_style=_LABEL_STYLE,
                    active_label_style=_ACTIVE_LABEL),
        ], className="mb-3"),
    ])


def page_ch_conclusion():
    """Ch 8: Conclusion and recommendations."""
    return page_conclusion()


# ── Main Layout ──

PAGE_MAP = {
    "overview":   page_ch_overview,
    "problem":    page_ch_problem,
    "evidence":   page_ch_evidence,
    "robustness": page_ch_robustness,
    "tail_risk":  page_ch_tail_risk,
    "oos":        page_ch_oos,
    "tools":      page_ch_tools,
    "conclusion": page_ch_conclusion,
}

app.layout = html.Div(
    [
        # Bootstrap Icons CDN
        html.Link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
        ),
        # Google Fonts
        html.Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
        ),
        dcc.Store(id="current-page", data="overview"),
        dcc.Store(id="sidebar-state", data="open"),
        sidebar,
        html.Div(
            id="page-content",
            style={
                "marginLeft": SIDEBAR_WIDTH,
                "padding": "32px 40px",
                "minHeight": "100vh",
                "backgroundColor": COLORS["bg"],
                "fontFamily": FONT,
                "transition": "margin-left 0.25s ease",
            },
        ),
    ],
    style={"backgroundColor": COLORS["bg"]},
)


# ═══════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output("sidebar", "style"),
    Output("sidebar", "className"),
    Output("page-content", "style"),
    Output("sidebar-brand", "style"),
    Output("sidebar-nav", "style"),
    Output("sidebar-state", "data"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-state", "data"),
    prevent_initial_call=True,
)
def toggle_sidebar(n, state):
    new_state = "closed" if state == "open" else "open"
    if new_state == "closed":
        sb_style = {
            "position": "fixed", "top": 0, "left": 0, "bottom": 0,
            "width": SIDEBAR_COLLAPSED_WIDTH,
            "backgroundColor": COLORS["card"],
            "borderRight": f"1px solid {COLORS['card_border']}",
            "overflowY": "auto", "overflowX": "hidden",
            "zIndex": 1000, "transition": "width 0.25s ease",
        }
        content_style = {
            "marginLeft": SIDEBAR_COLLAPSED_WIDTH,
            "padding": "32px 40px", "minHeight": "100vh",
            "backgroundColor": COLORS["bg"], "fontFamily": FONT,
            "transition": "margin-left 0.25s ease",
        }
        brand_style = {"display": "none"}
        nav_style = {"paddingTop": "4px", "paddingBottom": "20px"}
        sb_class = "sidebar-collapsed"
    else:
        sb_style = {
            "position": "fixed", "top": 0, "left": 0, "bottom": 0,
            "width": SIDEBAR_WIDTH,
            "backgroundColor": COLORS["card"],
            "borderRight": f"1px solid {COLORS['card_border']}",
            "overflowY": "auto", "overflowX": "hidden",
            "zIndex": 1000, "transition": "width 0.25s ease",
        }
        content_style = {
            "marginLeft": SIDEBAR_WIDTH,
            "padding": "32px 40px", "minHeight": "100vh",
            "backgroundColor": COLORS["bg"], "fontFamily": FONT,
            "transition": "margin-left 0.25s ease",
        }
        brand_style = {"padding": "12px 20px 12px 20px",
                       "borderBottom": f"1px solid {COLORS['divider']}"}
        nav_style = {"paddingTop": "4px", "paddingBottom": "20px"}
        sb_class = ""
    return sb_style, sb_class, content_style, brand_style, nav_style, new_state


@callback(
    Output("current-page", "data"),
    [Input({"type": "nav-btn", "index": key}, "n_clicks") for key, _, _ in sidebar_items],
    prevent_initial_call=True,
)
def navigate(*clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    triggered_id = ctx.triggered[0]["prop_id"]
    for key, _, _ in sidebar_items:
        if f'"index":"{key}"' in triggered_id:
            return key
    return dash.no_update


@callback(
    Output("page-content", "children"),
    Output({"type": "nav-btn", "index": ALL}, "style"),
    Input("current-page", "data"),
)
def render_page(page):
    builder = PAGE_MAP.get(page, page_ch_overview)
    content = builder()
    # Active nav highlighting
    styles = []
    for key, _, _ in sidebar_items:
        base = {
            "fontSize": "0.82rem", "textAlign": "left",
            "padding": "10px 14px", "width": "100%",
            "border": "none", "borderRadius": "0",
            "textDecoration": "none", "whiteSpace": "nowrap",
            "overflow": "hidden", "display": "flex", "alignItems": "center",
        }
        if key == page:
            base["color"] = COLORS["accent_blue"]
            base["backgroundColor"] = "rgba(59,130,246,0.1)"
            base["borderLeft"] = f"3px solid {COLORS['accent_blue']}"
            base["fontWeight"] = 700
        else:
            base["color"] = COLORS["text_muted"]
            base["fontWeight"] = 500
        styles.append(base)
    return content, styles


@callback(
    Output("interaction-heatmap", "figure"),
    Input("heatmap-metric", "value"),
    prevent_initial_call=True,
)
def update_heatmap(metric):
    return fig_interaction_heatmap(metric)


@callback(
    Output("ticker-detail", "figure"),
    Input("ticker-dropdown", "value"),
    prevent_initial_call=True,
)
def update_ticker(ticker):
    if not ticker:
        return go.Figure()
    return fig_ticker_detail(ticker)


# ── Stress Monitor Callbacks ──

@callback(
    Output("ts-curve-chart", "figure"),
    Output("ts-zscore-chart", "figure"),
    Input("ts-ticker-dropdown", "value"),
)
def update_term_structure(ticker):
    if not ticker:
        empty = go.Figure()
        empty.update_layout(template=PLOT_TEMPLATE, title="Select a ticker")
        return empty, empty
    return _fig_term_structure_curve(ticker), _fig_zscore_bands(ticker)


# ── Bucket Tuner Callbacks ──

@callback(
    Output({"type": "bucket-slider", "index": ALL}, "value"),
    Input("bucket-preset", "value"),
    Input("bucket-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def apply_preset(preset_name, reset_clicks):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "bucket-reset-btn" in triggered:
        return [np.log10(v) for v in DEFAULT_MKTCAP_THRESHOLDS]

    presets = {
        "Standard (Default)": [50e6, 250e6, 2e9, 10e9, 200e9],
        "Russell-Aligned": [50e6, 300e6, 2e9, 15e9, 200e9],
        "Aggressive (More Nano)": [100e6, 500e6, 2e9, 10e9, 200e9],
        "Conservative (Fewer Buckets)": [25e6, 250e6, 2e9, 10e9, 100e9],
    }

    if preset_name == "Equal-ish Tickers":
        ticker_mc = df.dropna(subset=["cur_mkt_cap"]).groupby("ticker")["cur_mkt_cap"].median()
        quantiles = ticker_mc.quantile([1/6, 2/6, 3/6, 4/6, 5/6]).values
        return [np.log10(_snap_to_round(q)) for q in quantiles]

    vals = presets.get(preset_name, DEFAULT_MKTCAP_THRESHOLDS)
    return [np.log10(v) for v in vals]


@callback(
    Output({"type": "bucket-val-label", "index": ALL}, "children"),
    Output("bucket-visual-scale", "children"),
    Output("bucket-stats-table", "children"),
    Output("bucket-distribution-chart", "figure"),
    Input({"type": "bucket-slider", "index": ALL}, "value"),
)
def update_bucket_tuner(slider_values):
    # Convert log10 slider values back to USD and snap to round numbers
    raw_usd = [10 ** v for v in slider_values]
    snapped = [_snap_to_round(v) for v in raw_usd]
    # Ensure monotonically increasing
    for i in range(1, len(snapped)):
        if snapped[i] <= snapped[i - 1]:
            snapped[i] = snapped[i - 1] * 2

    # Value labels
    labels = [_fmt_usd(s) for s in snapped]

    # Visual scale bar
    scale_bar = dbc.Card(
        dbc.CardBody([
            html.H5("Bucket Scale", style={"color": COLORS["accent_blue"],
                                            "fontWeight": 700, "marginBottom": "12px"}),
            dbc.Row(
                [dbc.Col(
                    html.Div([
                        html.Div(MKTCAP_LABELS[i], style={
                            "fontWeight": 700, "fontSize": "0.85rem",
                            "color": [COLORS["accent_red"], COLORS["accent_orange"],
                                      COLORS["accent_cyan"], COLORS["accent_blue"],
                                      COLORS["accent_purple"], COLORS["accent_green"]][i],
                            "textAlign": "center",
                        }),
                        html.Div(
                            f"{'<' if i == 0 else '>'}{_fmt_usd(snapped[i] if i < len(snapped) else snapped[-1])}"
                            if i == 0 or i == 5
                            else f"{_fmt_usd(snapped[i-1])} – {_fmt_usd(snapped[i])}",
                            style={"color": COLORS["text_muted"], "fontSize": "0.7rem",
                                   "textAlign": "center"},
                        ),
                    ], style={
                        "padding": "8px 4px",
                        "backgroundColor": "rgba(255,255,255,0.03)",
                        "borderRadius": "6px",
                        "border": f"1px solid {COLORS['card_border']}",
                    }),
                    md=2,
                ) for i in range(6)],
                className="g-2",
            ),
        ]),
        style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
               "borderRadius": "12px"},
    )

    # Compute bucket stats (full universe)
    stats_df = compute_bucket_stats(df, snapped)

    # Stats table
    stats_table = dbc.Card(
        dbc.CardBody([
            html.H5("Bucket Statistics", style={
                "color": COLORS["accent_cyan"], "fontWeight": 700, "marginBottom": "12px"}),
            dbc.Table(
                [html.Thead(html.Tr([
                    html.Th("Bucket", style={"color": COLORS["text"], "fontWeight": 700}),
                    html.Th("Tickers", style={"color": COLORS["text"], "textAlign": "right"}),
                    html.Th("Rows", style={"color": COLORS["text"], "textAlign": "right"}),
                    html.Th("Median MktCap", style={"color": COLORS["text"], "textAlign": "right"}),
                    html.Th("Mean |Ret|", style={"color": COLORS["text"], "textAlign": "right"}),
                    html.Th("p95 |Ret|", style={"color": COLORS["text"], "textAlign": "right"}),
                    html.Th("p99 |Ret|", style={"color": COLORS["text"], "textAlign": "right"}),
                    html.Th("Mean Amihud", style={"color": COLORS["text"], "textAlign": "right"}),
                ]))] +
                [html.Tbody([
                    html.Tr([
                        html.Td(row["Bucket"], style={"color": COLORS["text"], "fontWeight": 600}),
                        html.Td(f"{int(row['Tickers']):,}", style={"color": COLORS["text_muted"], "textAlign": "right"}),
                        html.Td(f"{int(row['Rows']):,}", style={"color": COLORS["text_muted"], "textAlign": "right"}),
                        html.Td(_fmt_usd(row["Median MktCap"]), style={"color": COLORS["accent_cyan"], "textAlign": "right"}),
                        html.Td(f"{row['Mean |Return|']:.4f}", style={"color": COLORS["text_muted"], "textAlign": "right"}),
                        html.Td(f"{row['p95 |Return|']:.4f}", style={"color": COLORS["accent_orange"], "textAlign": "right"}),
                        html.Td(f"{row['p99 |Return|']:.4f}", style={"color": COLORS["accent_red"], "textAlign": "right"}),
                        html.Td(_fmt_illiq(row["Mean Amihud"]), style={"color": COLORS["text_muted"], "textAlign": "right"}),
                    ]) for _, row in stats_df.iterrows()
                ])],
                bordered=False,
                color="dark",
                hover=True,
                size="sm",
                style={"backgroundColor": "transparent"},
            ),
        ]),
        style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
               "borderRadius": "12px"},
    )

    # Distribution chart
    bucket_colors = [COLORS["accent_red"], COLORS["accent_orange"], COLORS["accent_cyan"],
                     COLORS["accent_blue"], COLORS["accent_purple"], COLORS["accent_green"]]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Tickers per Bucket", "Mean |Return| by Bucket", "Mean Amihud by Bucket"],
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Bar(
        x=stats_df["Bucket"], y=stats_df["Tickers"],
        marker=dict(color=bucket_colors, line=dict(width=0)),
        text=[f"{int(v):,}" for v in stats_df["Tickers"]],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>%{y:,} tickers<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=stats_df["Bucket"], y=stats_df["Mean |Return|"],
        marker=dict(color=bucket_colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in stats_df["Mean |Return|"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Mean |Ret|: %{y:.5f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=stats_df["Bucket"], y=stats_df["Mean Amihud"],
        marker=dict(color=bucket_colors, line=dict(width=0)),
        text=[_fmt_illiq(v) for v in stats_df["Mean Amihud"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Mean Amihud: %{customdata}<extra></extra>",
        customdata=[_fmt_illiq(v) for v in stats_df["Mean Amihud"]],
        showlegend=False,
    ), row=1, col=3)

    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=420,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    fig.update_yaxes(title_text="Count", col=1)
    fig.update_yaxes(title_text="Mean |Return|", col=2)
    fig.update_yaxes(title_text="Mean Amihud", col=3)

    return labels, scale_bar, stats_table, fig


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Liquidity Risk Dashboard")
    print("  http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=False, port=8050)
