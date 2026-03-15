# Amihud ILLIQ vs Free-Float: A Liquidity Risk Thesis

> Full analytical framework for why Amihud ILLIQ outperforms free-float percentage as a daily liquidity risk signal, including the liquidity term structure concept, extreme move modeling, asymmetric scenario analysis, and construction of the Amihud add-on overlay.

**Domain:** Risk analytics · Executive presentation · US equities 2025  
**Data sources:** Bloomberg Terminal (blpapi) + yfinance  
**Stack:** Python · Plotly Dash · OLS/Logistic regression · SQLite

---

## Table of Contents

1. [Core Thesis](#part-1--core-thesis)
2. [The Liquidity Term Structure](#part-2--the-liquidity-term-structure)
3. [Regression Framework](#part-3--regression-framework)
4. [Extreme Move Framework](#part-4--extreme-move-framework)
5. [Additional Amihud Advantages](#part-5--additional-amihud-advantages)
6. [Visualization Plan](#part-6--visualization-plan)
7. [What to Add Next](#part-7--what-to-add-next)
8. [Asymmetric Scenario Framework](#part-8--asymmetric-scenario-framework)
9. [Codebase Inventory](#codebase-inventory)

---

## Part 1 — Core Thesis

### What each measure actually captures

**Free-float percentage** is a *supply* measure — it tells you what share of outstanding stock could theoretically trade. It says nothing about whether those shares are trading, at what cost, or with what price impact. A stock can have 95% free float and still be severely illiquid if nobody wants to trade it.

**Amihud ILLIQ** is a *friction* measure — computed from actual daily transactions, it captures how much the price moves per dollar of realized volume. These are fundamentally different quantities. One is structural, one is behavioral.

```
ILLIQ_t = |r_t| / (P_t × V_t)

where r_t = daily return, P_t × V_t = dollar volume
```

### Why free-float data is structurally stale

Bloomberg's `EQY_FREE_FLOAT_PCT` derives from SEC ownership filings (13F, 13D, 13G, Schedule 14A). These are filed quarterly at best; many smaller institutions file annually. Bloomberg interpolates or carries forward the last known value. You cannot answer "what was the free float on March 15th?" — you can only answer "what was the last reported filing value near March 15th?" That is a categorically weaker signal.

**Empirically:** a large fraction of tickers show zero change in free float across 3+ consecutive months, clusters of changes occur at quarterly filing boundaries, and many tickers show no change across the entire year.

### The epistemological nuance

The honest framing: even if daily free-float data existed, it would still be the wrong variable. Liquidity is not about supply availability — it is about demand-side friction (bid-ask spreads, market depth, price impact). Amihud captures this directly. Free float is at best a second-order proxy that works only through its correlation with trading activity.

A large-cap stock with 99% free float can become severely illiquid in a crisis. Amihud captures that immediately. Free float does not change.

### Why Amihud works as a return overlay on market cap

Market cap captures size, which correlates with liquidity but is not identical to it. Adding Amihud on top of market cap contributes the residual liquidity variation within size buckets — the component that market cap alone misses. Illiquid stocks require a liquidity premium compensating investors for the cost and risk of exiting positions. This premium shows up as higher expected absolute returns, higher intraday ranges, and higher sensitivity to volume shocks.

---

## Part 2 — The Liquidity Term Structure

### Two-point term structure of liquidity

Amihud at different rolling windows reveals different regimes of the same underlying quantity — analogous to a yield curve with a short end and a long end:

- **21-day Amihud** — tactical signal. Captures current microstructure friction, post-earnings illiquidity spikes, volume drying up ahead of a catalyst, short squeeze building. Noise-sensitive but highly responsive.
- **252-day Amihud** — structural baseline. Captures the chronic liquidity character of the stock. Stable but slow to react.

### What the divergence tells you

| Regime | Meaning | Risk Interpretation |
|--------|---------|---------------------|
| Short >> Long | Illiquidity spike | Transient liquidity stress — earnings, news, thin market |
| Short << Long | Liquidity improvement | Unusual volume — institutional accumulation, index rebal |
| Both elevated | Chronic structural illiquidity | Highest persistent tail risk |

### The liquidity z-score (CDS analog)

The ratio `illiq_21d / illiq_252d` is a live spread between short-term and structural liquidity. To make it cross-sectionally comparable, standardize it against the stock's own history:

```
illiq_ratio_t   = illiq_21d_t / illiq_252d_t
μ_ratio_t       = rolling 252d mean of illiq_ratio
σ_ratio_t       = rolling 252d std of illiq_ratio
liquidity_zscore_t = (illiq_ratio_t − μ_ratio_t) / σ_ratio_t
```

A z-score of +2.0 means this stock's short-term illiquidity is 2 standard deviations above its own historical norm — comparable across a micro-cap and a mega-cap. This is the CDS analog: just as a CDS spread measures the market's live assessment of near-term credit stress relative to the issuer's long-run rating, the liquidity z-score measures the market's live assessment of near-term liquidity stress relative to the stock's structural baseline.

### Five columns to compute daily

| Column | Definition |
|--------|-----------|
| `illiq_21d` | Rolling 21-day mean of daily ILLIQ |
| `illiq_252d` | Rolling 252-day mean of daily ILLIQ |
| `illiq_ratio` | `illiq_21d / illiq_252d` |
| `illiq_ratio_mean` | Rolling 252d mean of `illiq_ratio` |
| `illiq_zscore` | `(illiq_ratio − illiq_ratio_mean) / rolling_252d_std(illiq_ratio)` |

### Two-factor risk score

- **Structural risk:** where does `illiq_252d` sit in the cross-sectional distribution? Chronic illiquidity.
- **Stress indicator:** is `illiq_zscore` currently above ~+2? Current liquidity anomalously worse than norm.

High on both = highest priority risk name. High on z-score only = transient alert. High on structural only = chronic position-sizing concern.

---

## Part 3 — Regression Framework

### Seven OLS specifications

| Spec | Model |
|------|-------|
| 1 | `\|return\| ~ log(mktcap)` — baseline, market cap only |
| 2 | `\|return\| ~ log(illiq_252d)` — Amihud alone |
| 3 | `\|return\| ~ eqy_free_float_pct` — free float alone |
| 4 | `\|return\| ~ log(mktcap) + log(illiq_252d)` — Amihud adds to mktcap |
| 5 | `\|return\| ~ log(mktcap) + eqy_free_float_pct` — FF adds to mktcap |
| 6 | `\|return\| ~ log(mktcap) + log(illiq_252d) + eqy_free_float_pct` — kitchen sink |
| 7 | Fama-MacBeth: run spec 6 cross-sectionally each month, t-test coefficients across months |

In spec 6, if Amihud's coefficient stays significant while the free-float t-stat drops toward zero, that is the cleanest evidence that Amihud subsumes free float's information content. If free-float's coefficient flips sign after controlling for Amihud, it was a spurious size correlation all along.

### Dependent variables

| Variable | Definition |
|----------|-----------|
| `abs_return` | Close-to-close absolute return |
| `hl_range` | `(high − low) / close` — intraday price range |
| `liquidity_adj_return` | `abs_return / illiq_252d` — strips chronic component |

### R² incremental decomposition (executive metric)

```
R²_baseline  = R² of log(mktcap) alone
R²_amihud    = R² of log(mktcap) + log(illiq_252d)
R²_ff        = R² of log(mktcap) + eqy_free_float_pct
R²_full      = R² of kitchen sink spec

ΔR²_amihud   = R²_amihud − R²_baseline   (Amihud's incremental contribution)
ΔR²_ff       = R²_ff − R²_baseline        (free float's incremental contribution)
Superiority  = ΔR²_amihud / ΔR²_ff        (how many times better is Amihud)
```

---

## Part 4 — Extreme Move Framework

### Market cap bucket thresholds

For each market cap bucket (see [Part 8](#part-8--asymmetric-scenario-framework) for the full 6-bucket taxonomy), compute the empirical 95th and 99th percentile of `abs_return` within that bucket. An extreme move is defined as a daily return exceeding the within-bucket 95th percentile. This is not a one-size-fits-all cutoff — large caps have structurally smaller daily moves than small caps.

```
extreme_flag_t = 1  if abs_return_t > p95_within_mktcap_bucket  else 0
```

### Amihud add-on model

Logistic regression predicting extreme moves, comparing baseline (market cap bucket only) versus add-on (market cap bucket + Amihud z-score lagged one day):

```
P(extreme_t) = logistic(β₀ + β₁·mktcap_bucket + β₂·illiq_zscore_{t−1})
```

### Evaluation metrics

| Metric | What It Measures |
|--------|-----------------|
| McFadden pseudo-R² | Overall fit improvement over baseline |
| False negative reduction | Extreme moves mktcap missed that Amihud catches |
| Precision / recall by bucket | Where does Amihud help most? |
| Average miss magnitude | When both models miss, how large is the missed move? |
| Lag decay | Test 1, 3, 5 day lags — where does signal decay? |

### Residual analysis

The extreme moves that neither model catches are the genuine information shocks — macro events, earnings surprises, geopolitical catalysts. These are not liquidity-driven; no liquidity measure will predict them. Quantifying the size of this irreducible residual is an important caveat for the executive presentation: it tells management what Amihud is actually solving and what it is not claiming to solve.

---

## Part 5 — Additional Amihud Advantages

### Things Amihud captures that free float cannot

- **Intraday liquidity collapse** — on days with low volume, Amihud spikes even if float is unchanged
- **Asymmetric liquidity** — Amihud on down days is typically 2–3× higher for illiquid stocks; free float is symmetric by construction
- **Liquidity commonality** — market-wide illiquidity shocks hit high-Amihud stocks disproportionately; free float doesn't capture this co-movement
- **Post-earnings illiquidity** — Amihud spikes predictably around earnings for small caps; free float shows nothing
- **Short squeeze early warning** — rising Amihud can precede a squeeze completion; free float is static until the next filing
- **Lag structure is testable** — Amihud's predictive power can be measured at 1, 3, 5 day horizons; free float has no meaningful lag structure

### Asymmetry test (critical for risk frameworks)

Test the z-score separately on down days vs up days. Liquidity deteriorates faster during selloffs than rallies. If z-score is predictive on down days but not up days, that is the correct result for a risk framework — it is a downside liquidity warning system, not a symmetric one. This should be surfaced explicitly in the presentation.

---

## Part 6 — Visualization Plan

### Charts to build (Dash)

1. **Term structure curve** — per-stock plot of `illiq_21d` vs `illiq_252d` over time, with divergence highlighted. Side-by-side with FF% equivalent (21d rolling std vs annual mean).
2. **Liquidity z-score time series** — per ticker, z-score with ±2 bands. Stress events annotated.
3. **R² horse race** — all 7 specs, horizontal bar chart, for both dependent variables.
4. **Incremental R² panel** — ΔR² from adding Amihud vs ΔR² from adding FF%, by size bucket and by month.
5. **Extreme move confusion matrix** — baseline model vs Amihud add-on: true positives, false negatives, reduction in misses.
6. **Average miss magnitude** — histogram of missed extreme moves, baseline vs add-on, overlaid.
7. **Asymmetry panel** — z-score predictive power on up days vs down days.
8. **5×5 interaction heatmap** — Amihud quintile × FF quintile, colored by mean `abs_return`. Shows which axis drives the gradient.
9. **Staleness diagnostic** — distribution of longest unchanged FF% streak; monthly change rate bar chart.
10. **Stock explorer** — per-ticker deep dive: price, `illiq_21d`, `illiq_252d`, z-score, FF%, dollar volume on one page.
11. **Asymmetric scenario dashboard** — the 6-bucket × up/down scenario matrix from Part 8. Core narrative page.

---

## Part 7 — What to Add Next

### Open items not yet in the codebase

- [ ] Compute `illiq_ratio`, `illiq_ratio_mean`, `illiq_zscore` columns in `calc_rolling_amihud.py`
- [ ] Add 21-day rolling window as a second pass in the same script
- [ ] Fama-MacBeth cross-sectional regressions (monthly, t-test coefficients)
- [ ] Logistic regression for extreme move prediction (baseline vs add-on)
- [ ] Asymmetry split: down-day vs up-day z-score predictive power
- [ ] Lag structure decay test (1, 3, 5 days)
- [ ] Liquidity-adjusted return as additional dependent variable
- [ ] Sector controls (GICS dummies) for robustness
- [ ] VIX regime split: does Amihud superiority increase in high-vol environments?
- [ ] Term structure curve visualization in Dash explorer tab
- [ ] Asymmetric scenario framework (Part 8) — bucket assignment, empirical percentile computation, scenario matrix page in Dash

---

## Part 8 — Asymmetric Scenario Framework

### The core observation

Extreme daily moves are not symmetric across the market cap spectrum, and they are not symmetric between up and down. This section formalizes that asymmetry into a scenario matrix that serves as the narrative spine of the Dash presentation.

The directional framing:

- **Upside tails widen dramatically as market cap shrinks.** A mega-cap might see a 2–3% daily move as extreme. A nano-cap can move 100–500% in a single session on a catalyst — a biotech FDA approval, a short squeeze, a viral social-media event. The upside tail is essentially unbounded for the smallest names.
- **Downside tails compress less uniformly.** All stocks are bounded at −100% (equity goes to zero), but the practical shape of the downside distribution differs. Mega-caps rarely gap down more than 10% absent a systemic event. Nano-caps can and do lose 50–90% in a session — but the gradient from mega to nano is less steep on the downside than on the upside.

This creates a **skewed risk surface** across market cap buckets that free float cannot see and that Amihud captures directly through its sensitivity to volume-adjusted price impact.

### Market cap bucket taxonomy

| Bucket | Label | Float-Adjusted Market Cap Range | Typical Example |
|--------|-------|---------------------------------|-----------------|
| 1 | **Mega** | > $200B | AAPL, MSFT, NVDA |
| 2 | **Large** | $10B – $200B | Broad S&P 500 |
| 3 | **Mid** | $2B – $10B | S&P 400 / Russell Mid |
| 4 | **Small** | $300M – $2B | Russell 2000 core |
| 5 | **Micro** | $50M – $300M | Russell Micro, OTC graduated |
| 6 | **Nano** | < $50M | OTC, pink sheets, pre-revenue |

> **Note:** Thresholds are directional starting points. Final cutoffs should be calibrated to the empirical distribution of float-adjusted market cap in the dataset to ensure roughly balanced bucket sizes, or alternatively aligned to standard index breakpoints (Russell 2000/3000 boundaries).

### Scenario matrix (directional framing)

The table below captures the conceptual shape of extreme tail behavior by bucket. These are not hard targets — they represent the order-of-magnitude gradient that the data should confirm or refine.

| Bucket | Upside Extreme (p99 daily) | Downside Extreme (p99 daily) | Asymmetry Ratio (Up/Down) |
|--------|---------------------------|------------------------------|---------------------------|
| Mega | ~2–3% | ~2–3% | ~1.0× (roughly symmetric) |
| Large | ~5–8% | ~4–6% | ~1.2–1.5× |
| Mid | ~10–15% | ~8–12% | ~1.3–1.5× |
| Small | ~20–40% | ~15–25% | ~1.5–2.0× |
| Micro | ~50–100% | ~30–50% | ~2.0–2.5× |
| Nano | ~100–500% | ~50–90% | ~2.0–5.0× |

The key insight: **the asymmetry ratio itself increases as you move down the market cap spectrum.** Mega-caps have roughly symmetric tails. Nano-caps have grotesquely asymmetric tails — the upside is 2–5× wider than the downside. This is the scenario surface that the Dash app needs to make viscerally obvious.

### Why this matters for the Amihud thesis

1. **Free float is bucket-blind to tail shape.** A nano-cap and a mega-cap can have the same free-float percentage. Free float tells you nothing about whether that stock's extreme moves are ±3% or −80%/+400%. Bucket membership alone gets you part of the way — Amihud gets you the rest.

2. **Amihud captures the mechanism.** The reason nano-cap upside tails are so fat is precisely the microstructure friction Amihud measures: thin order books, wide spreads, low dollar volume. A buy order that would move AAPL by 1 basis point can move a nano-cap by 20%. Amihud quantifies this directly.

3. **The z-score differentiates within buckets.** Two nano-caps can have very different tail behavior on any given day. The one whose `illiq_zscore` is at +3 is the one about to deliver the extreme move. Bucket membership is necessary but not sufficient; Amihud provides the within-bucket signal.

4. **Downside asymmetry is the risk case.** For a risk framework, the critical question is not "can this stock go up 500%?" — it's "can this stock gap down 80% before we can exit?" The asymmetry test from Part 5 feeds directly into this: if the z-score is more predictive on down days, then it is specifically solving the downside liquidity risk problem that management cares about.

### Dash app narrative flow

The asymmetric scenario framework should serve as the **opening act** of the Dash presentation, not a supplementary page. The story arc:

1. **Open with the scenario matrix.** Show the 6-bucket × up/down grid with empirical percentiles from the data. Let the visual asymmetry speak for itself. Executive takeaway: "The risk surface is not flat — it is dramatically skewed, and it gets worse as you go smaller."

2. **Show that free float can't see it.** Overlay free-float percentage onto the same grid. Demonstrate that FF% varies across buckets but has no relationship to within-bucket tail width or asymmetry. The gradient in the scenario matrix is invisible to free float.

3. **Show that Amihud can see it.** Overlay `illiq_252d` and `illiq_zscore` onto the grid. Demonstrate that Amihud tracks the tail gradient — higher structural illiquidity maps to wider tails, and elevated z-scores predict when a stock is about to deliver an extreme move within its bucket.

4. **Transition to the regression horse race.** Now the R² comparison (Part 3) has context: you are not just comparing abstract fit statistics — you are comparing which measure captures the risk surface that you just showed management.

5. **Close with the extreme move add-on.** The logistic model (Part 4) is the operational deliverable: "Here is the tool we can deploy. Here is how many extreme moves it catches that the current framework misses. Here is what it costs us when we miss."

### Implementation tasks

- [ ] Assign each ticker to one of the 6 market cap buckets (float-adjusted) with configurable thresholds
- [ ] Compute empirical p95 and p99 of daily returns within each bucket, split by sign (up vs down)
- [ ] Compute the asymmetry ratio (`p99_up / |p99_down|`) per bucket
- [ ] Build the scenario matrix visualization in Dash (heatmap or diverging bar chart)
- [ ] Build the overlay views: FF% vs bucket tail width, Amihud vs bucket tail width
- [ ] Wire the scenario page as the opening tab in `plotly_dash.py`

---

## Codebase Inventory

| File | Purpose | Status |
|------|---------|--------|
| `download_ohlcv.py` | yfinance 2025 OHLCV download, chunked with resume | ✅ Exists |
| `src/download_2024.py` | 2024 lookback year download | ✅ Exists |
| `src/calc_rolling_amihud.py` | 252d rolling Amihud — needs 21d + ratio + z-score columns | ⚠️ Needs update |
| `src/bbg_free_float.py` | Bloomberg monthly free float pull via blpapi | ✅ Exists |
| `src/merge_data.py` | As-of merge: monthly FF onto daily Amihud | ✅ Exists |
| `calc_amihud_21d.py` | 21-day rolling Amihud + annual mean (standalone, no 2024 lookback) | ✅ Exists |
| `plotly_dash.py` | Plotly Dash executive dashboard — model comparison + scenario matrix + narrative | ⚠️ Needs update |


