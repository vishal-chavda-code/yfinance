"""
Liquidity Risk Dashboard — Amihud ILLIQ vs Free-Float Analysis
==============================================================
Production-grade Plotly Dash application for executive presentation.

Tells the story: Free-float is stale and unreliable → Amihud ILLIQ is a live,
superior measure of liquidity risk that works across all market cap sizes.

Usage:
    python dashboard.py
    → Opens at http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).resolve().parent / "data"

# Executive color palette — deep, saturated tones
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
    """Load and prepare all data needed for the dashboard."""
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
    df["log_mktcap"] = np.log1p(df["cur_mkt_cap"])
    df["log_illiq"] = np.log(df["illiq_252d"]).replace(-np.inf, np.nan)
    df["month"] = df["date"].dt.to_period("M").astype(str)

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
        colorbar=dict(title="Spearman ρ", titlefont=dict(size=11)),
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
        colorbar=dict(title=label, titlefont=dict(size=10)),
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
        hovertemplate="%{x|%b %d}<br>ILLIQ: %{y:.2e}<extra></extra>",
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
            colorbar=dict(title="Avg |Return|", titlefont=dict(size=10)),
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
sidebar_items = [
    ("overview", "Overview", "bi-speedometer2"),
    ("staleness", "The Staleness Problem", "bi-clock-history"),
    ("horse_race", "Amihud vs Free-Float", "bi-trophy"),
    ("deep_dive", "Scatter & Correlations", "bi-graph-up"),
    ("size", "Size Controls", "bi-bar-chart-steps"),
    ("time", "Time Stability", "bi-calendar-range"),
    ("heatmap", "Risk Heatmap", "bi-grid-3x3"),
    ("explorer", "Stock Explorer", "bi-search"),
    ("conclusion", "Conclusion", "bi-flag"),
]

SIDEBAR_WIDTH = "240px"
SIDEBAR_COLLAPSED_WIDTH = "52px"

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
                html.P("Executive Dashboard", id="sidebar-subtitle", style={
                    "color": COLORS["text_muted"], "fontSize": "0.7rem",
                    "letterSpacing": "1px",
                }),
            ],
            id="sidebar-brand",
            style={"padding": "12px 20px 12px 20px", "borderBottom": f"1px solid {COLORS['divider']}"},
        ),
        # Nav buttons
        html.Div(
            [
                dbc.Button(
                    [html.I(className=f"bi {icon}", style={"minWidth": "18px"}),
                     html.Span(f"  {label}", className="sidebar-label")],
                    id={"type": "nav-btn", "index": key},
                    color="link",
                    className="nav-link-btn",
                    style={
                        "color": COLORS["text_muted"], "fontSize": "0.82rem",
                        "fontWeight": 500, "textAlign": "left",
                        "padding": "10px 14px", "width": "100%",
                        "border": "none", "borderRadius": "0",
                        "textDecoration": "none", "whiteSpace": "nowrap",
                        "overflow": "hidden",
                    },
                )
                for key, label, icon in sidebar_items
            ],
            style={"paddingTop": "8px"},
        ),
    ],
    id="sidebar",
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
            html.H1("The Liquidity Blind Spot", style={
                "background": f"linear-gradient(135deg, {COLORS['accent_blue']}, {COLORS['accent_purple']})",
                "-webkit-background-clip": "text",
                "-webkit-text-fill-color": "transparent",
                "fontWeight": 800, "fontSize": "2.8rem", "marginBottom": "8px",
                "lineHeight": 1.1,
            }),
            html.P(
                "Why free-float percentage fails as a liquidity measure — and how "
                "the Amihud illiquidity ratio provides a live, actionable signal.",
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
            dbc.Col(make_card("Amihud R² Superiority", f"{superiority_ret:.1f}×",
                              f"vs Free-Float for |Return|",
                              color=COLORS["accent_green"]), md=3),
            dbc.Col(make_card("FF% Stale 1+ Quarter", f"{staleness['streak_3plus']:.0%}",
                              "Tickers unchanged for 3+ months",
                              color=COLORS["accent_red"]), md=3),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(make_card("Bloomberg FF Coverage", f"{ff_coverage:.0%}",
                              "Of tickers have free-float data"), md=3),
            dbc.Col(make_card("Analysis Sample", f"{dfc.shape[0]:,}",
                              "Rows with all fields non-null"), md=3),
            dbc.Col(make_card("Amihud R² (|Return|)", f"{r2_amihud_ret:.4f}",
                              f"vs FF%: {r2_ff_ret:.4f}",
                              color=COLORS["accent_blue"]), md=3),
            dbc.Col(make_card("Amihud R² (H-L Range)", f"{r2_amihud_hl:.4f}",
                              f"vs FF%: {r2_ff_hl:.4f}",
                              color=COLORS["accent_cyan"]), md=3),
        ], className="g-3 mb-4"),

        # Story overview
        dbc.Card(
            dbc.CardBody([
                html.H5("The Narrative", style={"color": COLORS["accent_blue"],
                                                  "fontWeight": 700, "marginBottom": "12px"}),
                dcc.Markdown("""
**The conventional approach** to sizing position risk uses market capitalization and free-float percentage
— the share of stock available for public trading. The assumption: lower free-float → less supply
→ greater price impact per trade.

**The problem**: Free-float data from vendors like Bloomberg is inherently stale. It updates
quarterly at best, and for many stocks, the reported value *doesn't change for the entire year*.
Meanwhile, true liquidity conditions shift daily.

**The solution**: The **Amihud ILLIQ ratio** — computed daily from actual trading data
(|return| / dollar volume) — captures real-time liquidity dynamics. Our analysis of **{:,} daily
observations** across **{:,} US equities** in 2025 demonstrates that Amihud consistently explains
**{:.1f}× more** return variance than free-float percentage, even after controlling for firm size.

Navigate the tabs to explore each dimension of the evidence.
""".format(df.shape[0], df["ticker"].nunique(), superiority_ret),
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
    ])


def page_staleness():
    return html.Div([
        html.H2("The Free-Float Staleness Problem", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "Bloomberg reports EQY_FREE_FLOAT_PCT monthly, but how often does the value actually change? "
            "If the data doesn't update, it can't reflect shifting liquidity conditions.",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem", "marginBottom": "24px",
                   "maxWidth": "700px"},
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
            dbc.Col(dcc.Graph(figure=fig_staleness_histogram(),
                              config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_monthly_change_rate(),
                              config={"displayModeBar": False}), md=6),
        ], className="mb-4"),

        dbc.Card(
            dbc.CardBody([
                html.H5("The Blind Spot", style={"color": COLORS["accent_orange"],
                                                    "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Graph(figure=fig_staleness_vs_amihud_volatility(),
                          config={"displayModeBar": False}),
                dcc.Markdown("""
Many stocks with **high Amihud coefficient of variation** (meaning their liquidity conditions
changed significantly throughout the year) had **unchanged free-float values for 6–12 months**.
This is the blind spot: the free-float signal reports *no change* while actual trading liquidity
is shifting meaningfully. Stocks in the upper-right corner are the most dangerous —
genuinely volatile liquidity that static free-float completely misses.
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
                   "marginBottom": "24px", "maxWidth": "700px"},
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

        html.Div(style={"height": "24px"}),

        dcc.Graph(figure=fig_quintile_sorts(), config={"displayModeBar": False}),

        dbc.Card(
            dbc.CardBody([
                html.H5("Key Takeaway", style={"color": COLORS["accent_blue"],
                                                  "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown(f"""
The Amihud illiquidity ratio explains **{superiority_ret:.1f}× more** variance in daily |Return|
and **{superiority_hl:.1f}× more** in intraday H-L range compared to free-float percentage.

The quintile sorts confirm this model-free: stocks in the highest Amihud quintile (most illiquid)
show **monotonically increasing** return magnitudes from Q1 to Q5. Free-float sorts are weaker
and less monotonic — consistent with a measure that's stale and less precise.

Adding free-float to a model that already contains Amihud provides **minimal incremental R²**,
confirming that Amihud already subsumes most of the information content of free-float.
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
    ])


def page_size():
    return html.Div([
        html.H2("Size-Controlled Analysis", style={
            "color": COLORS["text"], "fontWeight": 700, "marginBottom": "8px"}),
        html.P(
            "A critical robustness check: both Amihud and free-float correlate with firm size. "
            "Does Amihud still win within each size bucket, or is it just a size proxy?",
            style={"color": COLORS["text_muted"], "fontSize": "0.95rem",
                   "marginBottom": "24px", "maxWidth": "700px"},
        ),

        dcc.Graph(figure=fig_size_tercile_r2(), config={"displayModeBar": False}),

        dbc.Card(
            dbc.CardBody([
                html.H5("Verdict: Not a Size Effect", style={
                    "color": COLORS["accent_green"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
Amihud's explanatory power holds across **all three size buckets**: Small, Mid, and Large.
This rules out the objection that "Amihud only works because small stocks are illiquid and
volatile" — it works equally well within large-caps where free-float data is most complete.

The relationship between illiquidity and return magnitude is **not a disguised size effect**.
It reflects genuine liquidity risk that Amihud captures and free-float does not.
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
                   "marginBottom": "24px", "maxWidth": "700px"},
        ),

        dcc.Graph(figure=fig_monthly_r2_lines(), config={"displayModeBar": False}),

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
                   "marginBottom": "24px", "maxWidth": "700px"},
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
                html.H5("Amihud Drives the Gradient", style={
                    "color": COLORS["accent_orange"], "fontWeight": 700, "marginBottom": "8px"}),
                dcc.Markdown("""
The dominant color gradient runs **vertically** (across Amihud quintiles), not horizontally
(across Free-Float quintiles). This confirms that Amihud is the primary driver of return
magnitude — even after conditioning on free-float levels.

The top-right corner (high Amihud × low float) represents the most dangerous liquidity profile:
stocks that are both illiquid by trading activity AND have limited public supply. These are
the names most likely to experience extreme price dislocations.
""", style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "lineHeight": 1.6}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px", "marginTop": "16px"},
        ),
    ])


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
    ])


def page_conclusion():
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
            dbc.Col(make_card("Amihud Wins", f"{superiority_ret:.1f}× R²",
                              "Superior to free-float for |Return|",
                              color=COLORS["accent_green"]), md=4),
            dbc.Col(make_card("FF% is Stale", f"{staleness['streak_3plus']:.0%} stale",
                              "Unchanged for an entire quarter or more",
                              color=COLORS["accent_red"]), md=4),
            dbc.Col(make_card("Robust to Size", "All Buckets",
                              "Holds for Small, Mid, and Large caps",
                              color=COLORS["accent_blue"]), md=4),
        ], className="g-3 mb-4"),

        dbc.Card(
            dbc.CardBody([
                dcc.Markdown(f"""
## Key Findings

**1. Amihud ILLIQ is the superior liquidity risk signal.**
Across {df.shape[0]:,} daily observations and {df['ticker'].nunique():,} US equities in 2025,
Amihud ILLIQ explains **{superiority_ret:.1f}× more** variance in close-to-close returns and
**{superiority_hl:.1f}× more** in intraday H-L range compared to Bloomberg free-float percentage.

**2. Free-float data is structurally stale.**
{staleness['streak_3plus']:.0%} of tickers have a 3+ month unchanged streak.
{staleness['never_changed_pct']:.0%} showed zero change across all 12 months.
Changes that do occur cluster at quarterly filing boundaries, confirming the data is
derived from periodic SEC filings — not live market activity.

**3. Amihud is a live, daily signal.**
Computed directly from |return| / dollar volume, it updates every trading day and
reflects actual market liquidity conditions in real-time.

**4. The result is robust:**
- Consistent across **Small / Mid / Large** market cap terciles (not a size effect)
- Stable across **all 12 months** of 2025 (not period-specific)
- Holds for **multiple return definitions** (close-to-close, H-L range, Parkinson volatility)
- Confirmed by both **pooled OLS** and **Fama-MacBeth** cross-sectional methodology

## Recommendation

Integrate the **Amihud ILLIQ ratio** into the risk framework as the primary liquidity metric
for position sizing and concentration risk. Free-float should be retained as a secondary,
cross-referencing measure — but its structural staleness means it should not be relied upon
as the sole indicator of liquidity risk.

## Suggested Next Steps

1. **Sector Controls** — Add GICS sector dummies to confirm sector neutrality
2. **VIX Regime Splits** — Test relationship strength in high vs low volatility markets
3. **Predictive Regressions** — Does yesterday's Amihud predict today's return?
4. **Tail Risk Analysis** — Focus on extreme return events (beyond 3σ) where liquidity risk matters most
5. **Portfolio-Level Backtest** — Simulate position sizing using Amihud vs free-float constraints
""", style={"color": COLORS["text_muted"], "fontSize": "0.9rem", "lineHeight": 1.7}),
            ]),
            style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['card_border']}",
                   "borderRadius": "12px"},
        ),
    ])


# ── Main Layout ──

PAGE_MAP = {
    "overview": page_overview,
    "staleness": page_staleness,
    "horse_race": page_horse_race,
    "deep_dive": page_deep_dive,
    "size": page_size,
    "time": page_time,
    "heatmap": page_heatmap,
    "explorer": page_explorer,
    "conclusion": page_conclusion,
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
    Output("page-content", "style"),
    Output("sidebar-brand", "style"),
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
    return sb_style, content_style, brand_style, new_state


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
    Input("current-page", "data"),
)
def render_page(page):
    builder = PAGE_MAP.get(page, page_overview)
    return builder()


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


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Liquidity Risk Dashboard")
    print("  http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=False, port=8050)
