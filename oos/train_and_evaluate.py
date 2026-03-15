"""
Train on 2025, evaluate on 2026 — Out-of-Sample validation.

Fits all OLS and logistic regression formulas on the 2025 in-sample data,
freezes the coefficients, then applies those frozen formulas to the 2026
out-of-sample data.  Compares predicted vs realized to produce IS vs OOS
metrics side-by-side.

Models evaluated:
  OLS Group 1 — Market Cap baseline:
    ① |Return| ~ log(mktcap)
  OLS Group 2 — Free-Float:
    ② |Return| ~ eqy_free_float_pct
    ③ |Return| ~ log(mktcap) + eqy_free_float_pct
  OLS Group 3 — Amihud:
    ④ |Return| ~ log(illiq_252d)
    ⑤ |Return| ~ log(mktcap) + log(illiq_252d)
  OLS Group 4 — Kitchen Sink:
    ⑥ |Return| ~ log(mktcap) + log(illiq_252d) + eqy_free_float_pct

  Logistic — Extreme move prediction:
    Baseline: P(extreme) ~ mktcap_bucket
    Add-on:   P(extreme) ~ mktcap_bucket + illiq_zscore_lag1

  All of the above repeated for H-L Range as dependent variable.

Usage:
    python -m oos.train_and_evaluate
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, precision_score,
    recall_score, f1_score, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OOS_DIR = PROJECT_ROOT / "oos" / "data"

MERGED_2025 = DATA_DIR / "amihud_with_free_float.parquet"
OHLCV_2025 = DATA_DIR / "us_equities_2025_ohlcv.parquet"
OOS_FEATURES = OOS_DIR / "oos_features.parquet"
OUTPUT_JSON = OOS_DIR / "oos_results.json"

MKTCAP_LABELS = ["Nano", "Micro", "Small", "Mid", "Large", "Mega"]
DEFAULT_MKTCAP_THRESHOLDS = [
    50_000_000, 250_000_000, 2_000_000_000,
    10_000_000_000, 200_000_000_000,
]


# ═══════════════════════════════════════════════════════════════════════
# Data prep — mirrors plotly_dash.py load_data + build_analysis_sample
# ═══════════════════════════════════════════════════════════════════════

def load_2025_training_data() -> pd.DataFrame:
    """Load the same 2025 dataset the dashboard uses."""
    df = pd.read_parquet(MERGED_2025)
    ohlcv = pd.read_parquet(OHLCV_2025, columns=["ticker", "date", "high", "low"])
    df = df.merge(ohlcv, on=["ticker", "date"], how="left")

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["ff_ratio"] = df["eqy_free_float_pct"] / 100
    df["cur_mkt_cap"] = df["cur_mkt_cap"] * 1e6   # source is in $M -> convert to $
    df["log_mktcap"] = np.log1p(df["cur_mkt_cap"])
    df["log_illiq"] = np.log(df["illiq_252d"]).replace(-np.inf, np.nan)

    # Term structure
    df = df.sort_values(["ticker", "date"])
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


def build_clean_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise and drop NAs — same as build_analysis_sample."""
    cols_needed = [
        "abs_return", "hl_range", "illiq_252d",
        "log_illiq", "eqy_free_float_pct", "ff_ratio",
        "log_mktcap", "cur_mkt_cap",
    ]
    dfc = df.dropna(subset=cols_needed).copy()
    for c in ["abs_return", "hl_range", "illiq_252d"]:
        lo, hi = dfc[c].quantile(0.01), dfc[c].quantile(0.99)
        dfc[c] = dfc[c].clip(lo, hi)
    return dfc


def assign_buckets(dfc: pd.DataFrame) -> pd.Series:
    t = sorted(DEFAULT_MKTCAP_THRESHOLDS)
    ticker_mc = dfc.groupby("ticker")["cur_mkt_cap"].median()

    def _label(mc):
        if mc < t[0]: return "Nano"
        elif mc < t[1]: return "Micro"
        elif mc < t[2]: return "Small"
        elif mc < t[3]: return "Mid"
        elif mc < t[4]: return "Large"
        return "Mega"

    return ticker_mc.apply(_label).rename("mktcap_bucket")


# ═══════════════════════════════════════════════════════════════════════
# OLS training + OOS evaluation
# ═══════════════════════════════════════════════════════════════════════

OLS_SPECS = [
    ("① MktCap Only",      ["log_mktcap"]),
    ("② Free-Float Only",  ["eqy_free_float_pct"]),
    ("③ MktCap + FF",      ["log_mktcap", "eqy_free_float_pct"]),
    ("④ Amihud Only",      ["log_illiq"]),
    ("⑤ MktCap + Amihud",  ["log_mktcap", "log_illiq"]),
    ("⑥ Kitchen Sink",     ["log_mktcap", "log_illiq", "eqy_free_float_pct"]),
]


def run_ols_is_oos(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Fit each OLS spec on train, evaluate on both train and test."""
    results = []
    for dep_label, y_col in [("|Return|", "abs_return"), ("H-L Range", "hl_range")]:
        for model_name, x_cols in OLS_SPECS:
            # Fit on training data only
            tr = train.dropna(subset=[y_col] + x_cols)
            te = test.dropna(subset=[y_col] + x_cols)
            if len(tr) < 100 or len(te) < 50:
                continue

            X_tr, y_tr = tr[x_cols].values, tr[y_col].values
            X_te, y_te = te[x_cols].values, te[y_col].values

            model = LinearRegression().fit(X_tr, y_tr)

            # In-sample metrics (on training set)
            pred_tr = model.predict(X_tr)
            is_r2 = r2_score(y_tr, pred_tr)
            is_mae = mean_absolute_error(y_tr, pred_tr)
            n_tr, k = X_tr.shape
            is_adj_r2 = 1 - (1 - is_r2) * (n_tr - 1) / (n_tr - k - 1)

            # Out-of-sample metrics (frozen coefficients applied to test set)
            pred_te = model.predict(X_te)
            oos_r2 = r2_score(y_te, pred_te)
            oos_mae = mean_absolute_error(y_te, pred_te)
            n_te = X_te.shape[0]
            oos_adj_r2 = 1 - (1 - oos_r2) * (n_te - 1) / (n_te - k - 1)

            results.append({
                "Dependent": dep_label,
                "Model": model_name,
                "Regressors": " + ".join(x_cols),
                "IS_R2": round(is_r2, 6),
                "IS_Adj_R2": round(is_adj_r2, 6),
                "IS_MAE": round(is_mae, 6),
                "IS_N": int(n_tr),
                "OOS_R2": round(oos_r2, 6),
                "OOS_Adj_R2": round(oos_adj_r2, 6),
                "OOS_MAE": round(oos_mae, 6),
                "OOS_N": int(n_te),
                "R2_decay": round(is_r2 - oos_r2, 6),
                "Coefficients": {col: round(float(model.coef_[i]), 8)
                                 for i, col in enumerate(x_cols)},
                "Intercept": round(float(model.intercept_), 8),
            })
    return results


def run_ols_by_size(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Same OLS specs but broken down by size tercile."""
    results = []
    for df_set, label in [(train, "IS"), (test, "OOS")]:
        ticker_mc = df_set.groupby("ticker")["cur_mkt_cap"].median()
        breaks = ticker_mc.quantile([1/3, 2/3])
        def _terc(mc):
            if mc <= breaks.iloc[0]: return "Small Cap"
            elif mc <= breaks.iloc[1]: return "Mid Cap"
            return "Large Cap"
        tg = ticker_mc.apply(_terc).rename("size_group")
        df_tmp = df_set.merge(tg, left_on="ticker", right_index=True, how="left")

        for sz in ["Small Cap", "Mid Cap", "Large Cap"]:
            sub = df_tmp[df_tmp["size_group"] == sz]
            for dep_label, y_col in [("|Return|", "abs_return"), ("H-L Range", "hl_range")]:
                for model_name, x_cols in OLS_SPECS:
                    s = sub.dropna(subset=[y_col] + x_cols)
                    if len(s) < 50:
                        continue
                    X, y = s[x_cols].values, s[y_col].values
                    m = LinearRegression().fit(X, y)
                    r2 = r2_score(y, m.predict(X))
                    results.append({
                        "Set": label, "Size": sz, "Dependent": dep_label,
                        "Model": model_name, "R2": round(r2, 6), "N": int(len(s)),
                    })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Logistic — Extreme move prediction
# ═══════════════════════════════════════════════════════════════════════

def run_extreme_is_oos(train_full: pd.DataFrame, test_full: pd.DataFrame,
                       train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> dict:
    """Fit extreme-move logistic models on train, apply frozen to test."""
    bucket_map = {label: i for i, label in enumerate(MKTCAP_LABELS)}

    def _prep(dfc, df_full):
        buckets = assign_buckets(dfc)
        tmp = dfc.merge(buckets, left_on="ticker", right_index=True, how="left")
        # Within-bucket p95 (use TRAINING thresholds for both sets)
        return tmp, buckets

    # Prep training set — compute thresholds
    train_tmp, _ = _prep(train_clean, train_full)
    p95_train = train_tmp.groupby("mktcap_bucket")["abs_return"].quantile(0.95)
    train_tmp["p95_threshold"] = train_tmp["mktcap_bucket"].map(p95_train)
    train_tmp["extreme_flag"] = (train_tmp["abs_return"] > train_tmp["p95_threshold"]).astype(int)

    # Lagged z-score from full df
    zs = train_full[["ticker", "date", "illiq_zscore"]].copy()
    zs = zs.sort_values(["ticker", "date"])
    zs["zscore_lag1"] = zs.groupby("ticker")["illiq_zscore"].shift(1)
    train_tmp = train_tmp.merge(zs[["ticker", "date", "zscore_lag1"]], on=["ticker", "date"], how="left")
    train_tmp["bucket_num"] = train_tmp["mktcap_bucket"].map(bucket_map)
    train_model = train_tmp.dropna(subset=["bucket_num", "zscore_lag1", "extreme_flag"])

    if len(train_model) < 100:
        return {"error": "insufficient training data"}

    # FIT on train
    X_base_tr = train_model[["bucket_num"]].values
    X_addon_tr = train_model[["bucket_num", "zscore_lag1"]].values
    y_tr = train_model["extreme_flag"].values

    lr_base = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced").fit(X_base_tr, y_tr)
    lr_addon = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced").fit(X_addon_tr, y_tr)

    # In-sample evaluation
    pred_base_tr = lr_base.predict(X_base_tr)
    pred_addon_tr = lr_addon.predict(X_addon_tr)

    # Prep test set — APPLY TRAINING p95 thresholds
    test_tmp, _ = _prep(test_clean, test_full)
    test_tmp["p95_threshold"] = test_tmp["mktcap_bucket"].map(p95_train)
    test_tmp["extreme_flag"] = (test_tmp["abs_return"] > test_tmp["p95_threshold"]).astype(int)

    zs_te = test_full[["ticker", "date", "illiq_zscore"]].copy()
    zs_te = zs_te.sort_values(["ticker", "date"])
    zs_te["zscore_lag1"] = zs_te.groupby("ticker")["illiq_zscore"].shift(1)
    test_tmp = test_tmp.merge(zs_te[["ticker", "date", "zscore_lag1"]], on=["ticker", "date"], how="left")
    test_tmp["bucket_num"] = test_tmp["mktcap_bucket"].map(bucket_map)
    test_model = test_tmp.dropna(subset=["bucket_num", "zscore_lag1", "extreme_flag"])

    if len(test_model) < 50:
        return {"error": "insufficient test data"}

    # FROZEN model applied to test
    X_base_te = test_model[["bucket_num"]].values
    X_addon_te = test_model[["bucket_num", "zscore_lag1"]].values
    y_te = test_model["extreme_flag"].values

    pred_base_te = lr_base.predict(X_base_te)
    pred_addon_te = lr_addon.predict(X_addon_te)

    def _metrics(y_true, y_pred, label):
        return {
            "set": label,
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "N": int(len(y_true)),
            "N_extreme": int(y_true.sum()),
            "confusion": confusion_matrix(y_true, y_pred).tolist(),
        }

    return {
        "p95_thresholds": {k: round(float(v), 6) for k, v in p95_train.items()},
        "baseline_IS": _metrics(y_tr, pred_base_tr, "IS"),
        "addon_IS": _metrics(y_tr, pred_addon_tr, "IS"),
        "baseline_OOS": _metrics(y_te, pred_base_te, "OOS"),
        "addon_OOS": _metrics(y_te, pred_addon_te, "OOS"),
        "baseline_coef": lr_base.coef_.tolist(),
        "addon_coef": lr_addon.coef_.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Asymmetry — IS vs OOS
# ═══════════════════════════════════════════════════════════════════════

def run_asymmetry_is_oos(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Up/down day R² comparison IS vs OOS."""
    results = []
    for df_set, set_label in [(train, "IS"), (test, "OOS")]:
        for split_label, mask_fn in [
            ("All Days", lambda d: d["return"].notna()),
            ("Down Days", lambda d: d["return"] < 0),
            ("Up Days", lambda d: d["return"] >= 0),
        ]:
            subset = df_set[mask_fn(df_set)].dropna(subset=["abs_return", "log_illiq"])
            if len(subset) < 100:
                continue
            for x_col, x_name in [("log_illiq", "Amihud 252d"),
                                   ("eqy_free_float_pct", "Free-Float %")]:
                sub = subset.dropna(subset=[x_col])
                if len(sub) < 100:
                    continue
                X, y = sub[[x_col]].values, sub["abs_return"].values
                m = LinearRegression().fit(X, y)
                r2 = r2_score(y, m.predict(X))
                results.append({
                    "Set": set_label, "Split": split_label,
                    "Regressor": x_name, "R2": round(r2, 6), "N": int(len(sub)),
                })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Scenario matrix — IS vs OOS
# ═══════════════════════════════════════════════════════════════════════

def run_scenario_is_oos(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """6-bucket tail percentiles IS vs OOS."""
    results = []
    for df_set, set_label in [(train, "IS"), (test, "OOS")]:
        buckets = assign_buckets(df_set)
        tmp = df_set.merge(buckets, left_on="ticker", right_index=True, how="left")
        for label in MKTCAP_LABELS:
            sub = tmp[tmp["mktcap_bucket"] == label]
            if len(sub) < 50:
                continue
            signed = sub["return"] if "return" in sub.columns else sub["abs_return"]
            up = signed[signed > 0]
            down = signed[signed < 0].abs()
            results.append({
                "Set": set_label, "Bucket": label,
                "N_tickers": int(sub["ticker"].nunique()),
                "N_days": int(len(sub)),
                "p95_up": round(float(up.quantile(0.95)), 6) if len(up) > 0 else 0,
                "p99_up": round(float(up.quantile(0.99)), 6) if len(up) > 0 else 0,
                "p95_down": round(float(down.quantile(0.95)), 6) if len(down) > 0 else 0,
                "p99_down": round(float(down.quantile(0.99)), 6) if len(down) > 0 else 0,
            })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Walk-forward within 2025 (expanding window, predict next month)
# ═══════════════════════════════════════════════════════════════════════

def run_walk_forward(dfc: pd.DataFrame) -> list[dict]:
    """Expanding-window walk-forward on 2025 monthly data.

    For each target month M (starting from month 7 = Jul 2025):
      - Training window: all months < M  (expanding)
      - Test window: month M only
      - Fit each OLS spec on training, evaluate on test
    This tests temporal stability of the Amihud/FF relationship.
    """
    dfc = dfc.copy()
    dfc["month"] = pd.to_datetime(dfc["date"] if "date" in dfc.columns
                                  else dfc.index).dt.to_period("M")
    months = sorted(dfc["month"].unique())

    if len(months) < 7:
        print("  ⚠ Need at least 7 months for walk-forward (6 train + 1 test)")
        return []

    results = []
    # Start predicting from month index 6 (= 7th month, ~Jul) onwards
    for i in range(6, len(months)):
        target_month = months[i]
        train_months = months[:i]

        train_set = dfc[dfc["month"].isin(train_months)]
        test_set = dfc[dfc["month"] == target_month]

        if len(train_set) < 1000 or len(test_set) < 100:
            continue

        for dep_label, y_col in [("|Return|", "abs_return"), ("H-L Range", "hl_range")]:
            for model_name, x_cols in OLS_SPECS:
                tr = train_set.dropna(subset=[y_col] + x_cols)
                te = test_set.dropna(subset=[y_col] + x_cols)
                if len(tr) < 100 or len(te) < 50:
                    continue

                X_tr, y_tr = tr[x_cols].values, tr[y_col].values
                X_te, y_te = te[x_cols].values, te[y_col].values

                m = LinearRegression().fit(X_tr, y_tr)

                is_r2 = r2_score(y_tr, m.predict(X_tr))
                oos_r2 = r2_score(y_te, m.predict(X_te))

                results.append({
                    "Target_Month": str(target_month),
                    "Train_Months": i,
                    "Dependent": dep_label,
                    "Model": model_name,
                    "IS_R2": round(is_r2, 6),
                    "OOS_R2": round(oos_r2, 6),
                    "R2_decay": round(is_r2 - oos_r2, 6),
                    "Train_N": int(len(tr)),
                    "Test_N": int(len(te)),
                })

    return results


def summarise_walk_forward(wf_results: list[dict]) -> list[dict]:
    """Average walk-forward OOS R² across months for each model spec."""
    if not wf_results:
        return []
    df = pd.DataFrame(wf_results)
    summary = []
    for (dep, model), grp in df.groupby(["Dependent", "Model"]):
        summary.append({
            "Dependent": dep,
            "Model": model,
            "Avg_OOS_R2": round(float(grp["OOS_R2"].mean()), 6),
            "Avg_IS_R2": round(float(grp["IS_R2"].mean()), 6),
            "Avg_R2_decay": round(float(grp["R2_decay"].mean()), 6),
            "Min_OOS_R2": round(float(grp["OOS_R2"].min()), 6),
            "Max_OOS_R2": round(float(grp["OOS_R2"].max()), 6),
            "Months_tested": int(len(grp)),
        })
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  OOS Validation: Train on 2025, Evaluate on 2026")
    print("=" * 60)

    # ── Load training data (2025) ─────────────────────────────────────
    print("\n[1/6] Loading 2025 training data...")
    train_full = load_2025_training_data()
    train_clean = build_clean_sample(train_full)
    print(f"  Train:  {train_clean.shape[0]:,} rows, {train_clean['ticker'].nunique()} tickers")

    # ── Load test data (2026) ─────────────────────────────────────────
    print("\n[2/6] Loading 2026 OOS features...")
    test_full = pd.read_parquet(OOS_FEATURES)
    test_full["date"] = pd.to_datetime(test_full["date"])
    test_clean = build_clean_sample(test_full)
    print(f"  Test:   {test_clean.shape[0]:,} rows, {test_clean['ticker'].nunique()} tickers")

    # ── OLS IS vs OOS ─────────────────────────────────────────────────
    print("\n[3/6] Running OLS evaluations (6 specs × 2 DVs)...")
    ols_results = run_ols_is_oos(train_clean, test_clean)
    print(f"  Generated {len(ols_results)} spec results")

    # Quick summary
    for r in ols_results:
        if r["Dependent"] == "|Return|":
            print(f"    {r['Model']:20s}  IS R²={r['IS_R2']:.4f}  OOS R²={r['OOS_R2']:.4f}  "
                  f"decay={r['R2_decay']:+.4f}")

    # ── OLS by size ───────────────────────────────────────────────────
    print("\n[4/6] Running OLS by size tercile...")
    size_results = run_ols_by_size(train_clean, test_clean)
    print(f"  Generated {len(size_results)} size-split results")

    # ── Extreme moves IS vs OOS ──────────────────────────────────────
    print("\n[5/6] Running extreme move logistic evaluation...")
    extreme_results = run_extreme_is_oos(train_full, test_full, train_clean, test_clean)
    if "error" not in extreme_results:
        print(f"  Baseline IS: F1={extreme_results['baseline_IS']['f1']:.3f}  "
              f"OOS: F1={extreme_results['baseline_OOS']['f1']:.3f}")
        print(f"  Add-on   IS: F1={extreme_results['addon_IS']['f1']:.3f}  "
              f"OOS: F1={extreme_results['addon_OOS']['f1']:.3f}")
    else:
        print(f"  ⚠ {extreme_results['error']}")

    # ── Asymmetry IS vs OOS ──────────────────────────────────────────
    print("\n[6/8] Running asymmetry evaluation...")
    asym_results = run_asymmetry_is_oos(train_clean, test_clean)
    print(f"  Generated {len(asym_results)} asymmetry results")

    # ── Scenario matrix IS vs OOS ────────────────────────────────────
    print("\n[7/8] Running scenario matrix IS vs OOS...")
    scenario_results = run_scenario_is_oos(train_clean, test_clean)
    print(f"  Generated {len(scenario_results)} scenario results")

    # ── Walk-forward within 2025 ─────────────────────────────────────
    print("\n[8/8] Running walk-forward backtest within 2025...")
    wf_results = run_walk_forward(train_clean)
    wf_summary = summarise_walk_forward(wf_results)
    print(f"  {len(wf_results)} monthly predictions, {len(wf_summary)} model summaries")
    if wf_summary:
        for s in wf_summary:
            if s["Dependent"] == "|Return|":
                print(f"    {s['Model']:20s}  Avg OOS R²={s['Avg_OOS_R2']:.4f}  "
                      f"decay={s['Avg_R2_decay']:+.4f}")

    # ── Save all results ──────────────────────────────────────────────
    output = {
        "metadata": {
            "train_period": "2025",
            "test_period": "2026-01-01 to 2026-03-15",
            "train_rows": int(train_clean.shape[0]),
            "train_tickers": int(train_clean["ticker"].nunique()),
            "test_rows": int(test_clean.shape[0]),
            "test_tickers": int(test_clean["ticker"].nunique()),
        },
        "ols": ols_results,
        "ols_by_size": size_results,
        "extreme_moves": extreme_results,
        "asymmetry": asym_results,
        "scenario": scenario_results,
        "walk_forward": wf_results,
        "walk_forward_summary": wf_summary,
    }

    OOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Results saved → {OUTPUT_JSON}")
    print("=" * 60)
