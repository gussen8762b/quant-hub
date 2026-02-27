"""
Market Intelligence Hub — Streamlit Cloud App
All 7 sections, dark theme, sidebar navigation, cached data loading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Intelligence Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent / "data"

C_GREEN  = "#22c55e"
C_YELLOW = "#eab308"
C_RED    = "#ef4444"
C_BLUE   = "#4f8ef7"
C_GREY   = "#9ca3af"
C_BG     = "#0f1117"
C_CARD   = "#1a1f2e"

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=C_BG,
    plot_bgcolor=C_CARD,
    font=dict(color="#ffffff", family="Inter, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=50, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor=C_CARD, font_color="#ffffff"),
)

HORIZON_MAP = {
    "short":  "3-Day",
    "medium": "10-Day",
    "long":   "40-Day",
}

SECTOR_ETF = {
    "Energy":                  "XLE",
    "Communication Services":  "XLC",
    "Comm Services":           "XLC",
    "Technology":              "XLK",
    "Information Technology":  "XLK",
    "Materials":               "XLB",
    "Financials":              "XLF",
    "Financial Services":      "XLF",
    "Finance":                 "XLF",
    "Health Care":             "XLV",
    "Healthcare":              "XLV",
    "Industrials":             "XLI",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def sig_color(sig: str) -> str:
    s = str(sig).lower().strip()
    return {
        "green":  C_GREEN,
        "yellow": C_YELLOW,
        "red":    C_RED,
        "long":   C_GREEN,
        "short":  C_RED,
        "flat":   C_GREY,
        "neutral":C_GREY,
    }.get(s, C_GREY)


def badge_html(text: str, color: str, size: str = "0.8em") -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 9px;'
        f'border-radius:4px;font-size:{size};font-weight:700;'
        f'letter-spacing:0.03em">{text}</span>'
    )


def card_html(title: str, value: str, value_color: str = "#fff",
              subtitle: str = "", border_color: str = C_BLUE) -> str:
    return (
        f'<div style="background:{C_CARD};border-left:4px solid {border_color};'
        f'border-radius:8px;padding:14px 18px;margin:6px 0">'
        f'<div style="color:{C_GREY};font-size:0.78em;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:4px">{title}</div>'
        f'<div style="color:{value_color};font-size:1.6em;font-weight:700;'
        f'line-height:1.1">{value}</div>'
        f'{"<div style=color:" + C_GREY + ";font-size:0.8em;margin-top:3px>" + subtitle + "</div>" if subtitle else ""}'
        f'</div>'
    )


def score_color(score: float) -> str:
    if score >= 7:   return C_GREEN
    if score >= 4:   return C_YELLOW
    return C_RED


def regime_info(pct: float) -> tuple:
    if pct >= 70:  return "Bull Market",  C_GREEN
    if pct >= 50:  return "Mixed",         C_YELLOW
    if pct >= 30:  return "Caution",       C_YELLOW
    return "Bear Market", C_RED


def add_rangeselector(fig, periods=None, default_range=None):
    periods = periods or ["1M", "3M", "6M", "1Y", "All"]
    buttons = []
    for p in periods:
        if p == "All":
            buttons.append(dict(step="all", label="All"))
        elif p.endswith("M"):
            buttons.append(dict(count=int(p[:-1]), label=p, step="month", stepmode="backward"))
        elif p.endswith("Y"):
            buttons.append(dict(count=int(p[:-1]), label=p, step="year", stepmode="backward"))
    fig.update_xaxes(
        rangeselector=dict(
            buttons=buttons,
            bgcolor=C_CARD,
            activecolor=C_BLUE,
            font=dict(color="#fff"),
        ),
        rangeslider=dict(visible=False),
        type="date",
    )
    if default_range:
        fig.update_xaxes(range=default_range)
    return fig


def section_divider(text: str):
    st.markdown(
        f'<div style="border-bottom:1px solid #2d3748;padding-bottom:4px;'
        f'margin:24px 0 16px;color:{C_GREY};font-size:0.8em;'
        f'text-transform:uppercase;letter-spacing:0.08em">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Data loaders (all cached) ─────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_ml_signals() -> pd.DataFrame:
    p = DATA_ROOT / "ml_signals" / "signal_log_meta.csv"
    # Columns: date, asset, horizon, tft_pred, lgb_pred, xgb_pred,
    #          ridge_pred, rf_pred, ensemble_pred, ensemble_direction,
    #          vote_count, n_models, confidence, confidence_level
    df = pd.read_csv(p, parse_dates=["date"])
    df["horizon_label"] = df["horizon"].map(HORIZON_MAP).fillna(df["horizon"].str.title())
    return df


@st.cache_data(ttl=3600)
def load_canslim() -> pd.DataFrame:
    p = DATA_ROOT / "canslim" / "composite_rankings.csv"
    # Columns: ticker, composite_score, tier, technical_score,
    #          fundamental_score, pattern_score, market_score,
    #          best_pattern, rs_rating, trend_template_pass,
    #          squeeze_fired, price, sector
    df = pd.read_csv(p)
    return df


@st.cache_data(ttl=3600)
def load_breadth_index() -> pd.DataFrame:
    p = DATA_ROOT / "breadth" / "breadth_index_weekly_with_wow.csv"
    # Columns: date, above_50sma, above_200sma, new_52w_high, new_52w_low,
    #          above_50sma_wow, above_200sma_wow, new_52w_high_wow, new_52w_low_wow
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["above_200sma"])
    return df


@st.cache_data(ttl=3600)
def load_breadth_sector() -> pd.DataFrame:
    p = DATA_ROOT / "breadth" / "breadth_sector_weekly.csv"
    # Columns: date, sector, above_50sma, above_200sma, new_52w_high, new_52w_low
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values(["sector", "date"])
    return df


@st.cache_data(ttl=3600)
def load_risk_factors() -> list:
    p = DATA_ROOT / "risk_factors" / "latest_scores.json"
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_risk_timeseries() -> pd.DataFrame:
    p = DATA_ROOT / "risk_factors" / "composite_timeseries.parquet"
    # Columns: date (datetime), composite_20, composite_50
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_crypto() -> pd.DataFrame:
    p = DATA_ROOT / "crypto" / "history.parquet"
    # Columns: Rank, Symbol, Composite, RS Score, Tech Score, Fund Score,
    #          RRG Regime, RS %ile, RS Momentum, RS Breakout,
    #          Golden ✕, Death ✕, run_date, run_ts, Tier
    df = pd.read_parquet(p)
    df["run_date"] = pd.to_datetime(df["run_date"])
    latest = df["run_date"].max()
    return df[df["run_date"] == latest].copy().sort_values("Rank")


# ── Section 1: Overview ───────────────────────────────────────────────────────

def render_overview():
    st.title("📊 Market Intelligence Hub")
    st.markdown("Real-time dashboard status — all modules at a glance")
    st.markdown("---")

    panels = [
        {
            "name": "ML Signals",
            "icon": "🤖",
            "loader": load_ml_signals,
            "metric": lambda df: (
                f"{len(df['asset'].unique())} assets tracked",
                df["date"].max().strftime("%Y-%m-%d"),
                C_BLUE,
            ),
        },
        {
            "name": "CAN SLIM / SEPA",
            "icon": "📈",
            "loader": load_canslim,
            "metric": lambda df: (
                f"{(df['composite_score'] >= 80).sum()} A/A+ setups",
                f"{len(df)} stocks screened",
                C_GREEN,
            ),
        },
        {
            "name": "Crypto",
            "icon": "🪙",
            "loader": load_crypto,
            "metric": lambda df: (
                f"#{int(df.iloc[0]['Rank'])} {df.iloc[0]['Symbol']}",
                df["run_date"].iloc[0].strftime("%Y-%m-%d"),
                C_YELLOW,
            ),
        },
        {
            "name": "S&P 500 Breadth",
            "icon": "🌊",
            "loader": load_breadth_index,
            "metric": lambda df: (
                f"{df.iloc[-1]['above_200sma']:.1f}% >200 SMA",
                df.iloc[-1]["date"].strftime("%Y-%m-%d"),
                score_color(df.iloc[-1]["above_200sma"] / 10),
            ),
        },
        {
            "name": "Market Risk Factors",
            "icon": "⚠️",
            "loader": load_risk_factors,
            "metric": lambda factors: (
                f"{sum(1 for f in factors if f.get('signal') == 'green')} green / "
                f"{sum(1 for f in factors if f.get('signal') == 'red')} red",
                "15 macro factors",
                C_GREEN if sum(1 for f in factors if f.get('signal') == 'green') >= 8 else C_YELLOW,
            ),
        },
        {
            "name": "Smart Money",
            "icon": "💰",
            "loader": None,
            "metric": None,
        },
        {
            "name": "ML Risk Timeseries",
            "icon": "📉",
            "loader": load_risk_timeseries,
            "metric": lambda df: (
                f"20d: {df.iloc[-1]['composite_20']:.2f} | 50d: {df.iloc[-1]['composite_50']:.2f}",
                df.iloc[-1]["date"].strftime("%Y-%m-%d"),
                score_color(df.iloc[-1]["composite_20"]),
            ),
        },
    ]

    cols = st.columns(3)
    for i, panel in enumerate(panels[:6]):
        with cols[i % 3]:
            try:
                if panel["loader"] is None:
                    smart_dir = DATA_ROOT / "smart_money"
                    files = list(smart_dir.glob("*")) if smart_dir.exists() else []
                    status = f"{len(files)} files" if files else "Pipeline scheduled"
                    st.markdown(
                        card_html(f"{panel['icon']} {panel['name']}", status,
                                  C_GREY, "Data loads on next run", C_GREY),
                        unsafe_allow_html=True,
                    )
                else:
                    data = panel["loader"]()
                    value, subtitle, color = panel["metric"](data)
                    st.markdown(
                        card_html(f"{panel['icon']} {panel['name']}", value,
                                  color, subtitle, color),
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.markdown(
                    card_html(f"{panel['icon']} {panel['name']}",
                              "No data", C_RED, str(e)[:60], C_RED),
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(
        f'<div style="color:{C_GREY};font-size:0.8em">Last refreshed: '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")} — '
        f'Data cached 1 hour (ttl=3600)</div>',
        unsafe_allow_html=True,
    )


# ── Section 2: ML Signals ─────────────────────────────────────────────────────

def render_ml_signals():
    st.title("🤖 ML Signal Engine")
    st.markdown("Multi-model ensemble predictions: TFT · LightGBM · XGBoost · Ridge · Random Forest")

    try:
        df = load_ml_signals()
    except Exception as e:
        st.error(f"Failed to load ML signals: {e}")
        return

    # Latest signal per asset × horizon
    latest = (
        df.sort_values("date")
        .groupby(["asset", "horizon"], as_index=False)
        .last()
    )
    latest["horizon_label"] = latest["horizon"].map(HORIZON_MAP).fillna(latest["horizon"])

    section_divider("Latest Signals — Asset × Horizon Grid")

    assets = sorted(latest["asset"].unique())
    horizons = ["short", "medium", "long"]

    # Grid of signal cards
    asset_cols = st.columns(len(assets))
    for col, asset in zip(asset_cols, assets):
        with col:
            st.markdown(f"**{asset}**", help="Asset tracked by the ML ensemble")
            for h in horizons:
                row = latest[(latest["asset"] == asset) & (latest["horizon"] == h)]
                if row.empty:
                    continue
                r = row.iloc[0]
                direction = str(r.get("ensemble_direction", "FLAT")).upper()
                conf = float(r.get("confidence", 0))
                conf_level = str(r.get("confidence_level", ""))
                hlabel = HORIZON_MAP.get(h, h)
                dc = sig_color(direction)
                d_text = direction if direction != "FLAT" else "NEUTRAL"
                st.markdown(
                    f'<div style="background:{C_CARD};border-left:3px solid {dc};'
                    f'padding:8px 10px;margin:4px 0;border-radius:5px">'
                    f'<div style="color:{dc};font-weight:700;font-size:0.9em">{d_text}</div>'
                    f'<div style="color:{C_GREY};font-size:0.75em">{hlabel}</div>'
                    f'<div style="color:#fff;font-size:0.8em">'
                    f'{conf:.0%} &nbsp;'
                    f'<span style="color:{sig_color(conf_level.lower() if conf_level=="HIGH" else "yellow" if conf_level=="MED" else "red")}">'
                    f'{conf_level}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    section_divider("Summary Table — All Latest Signals")

    disp = latest[["asset", "horizon_label", "ensemble_direction",
                   "confidence_level", "confidence", "vote_count",
                   "n_models", "date"]].copy()
    disp.columns = ["Asset", "Horizon", "Direction", "Conf Level",
                    "Confidence", "Votes", "Models", "Signal Date"]
    disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{x:.0%}")
    disp["Signal Date"] = pd.to_datetime(disp["Signal Date"]).dt.strftime("%Y-%m-%d")

    def _dir_style(val):
        v = str(val).upper()
        if v == "LONG":   return f"color:{C_GREEN};font-weight:700"
        if v == "SHORT":  return f"color:{C_RED};font-weight:700"
        return f"color:{C_GREY}"

    def _cl_style(val):
        return {"HIGH": f"color:{C_GREEN}", "MED": f"color:{C_YELLOW}",
                "LOW": f"color:{C_RED}"}.get(str(val), "")

    styled = (
        disp.style
        .map(_dir_style, subset=["Direction"])
        .map(_cl_style, subset=["Conf Level"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    section_divider("Cumulative Ensemble Prediction by Asset")

    col1, col2 = st.columns([2, 1])
    with col1:
        sel_assets = st.multiselect(
            "Assets", options=sorted(df["asset"].unique()),
            default=sorted(df["asset"].unique()),
        )
    with col2:
        sel_horizon = st.selectbox(
            "Horizon", options=["short", "medium", "long"],
            format_func=lambda x: HORIZON_MAP.get(x, x),
        )

    fig = go.Figure()
    palette = [C_BLUE, C_GREEN, C_YELLOW, C_RED, "#a855f7", "#06b6d4"]
    for i, asset in enumerate(sel_assets):
        adf = df[(df["asset"] == asset) & (df["horizon"] == sel_horizon)].sort_values("date").copy()
        if adf.empty:
            continue
        adf["cumulative"] = adf["ensemble_pred"].cumsum()
        fig.add_trace(go.Scatter(
            x=adf["date"], y=adf["cumulative"],
            name=asset, mode="lines",
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"<b>{asset}</b><br>%{{x|%Y-%m-%d}}<br>Cumulative: %{{y:.4f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=C_GREY, opacity=0.5)
    fig.update_layout(
        **PLOTLY_BASE,
        title=f"Cumulative Ensemble Prediction — {HORIZON_MAP.get(sel_horizon, sel_horizon)}",
        xaxis_title="Date", yaxis_title="Cumulative Prediction Signal",
    )
    add_rangeselector(fig, ["1M", "3M", "6M", "1Y", "All"])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Chart shows cumulative sum of ensemble prediction values — used as directional signal trend, not realized P&L.")

    section_divider("Model Agreement by Asset & Horizon")

    latest["agreement_pct"] = (latest["vote_count"] / latest["n_models"] * 100).round(1)
    fig2 = px.bar(
        latest.sort_values("asset"),
        x="asset", y="agreement_pct", color="horizon_label",
        barmode="group",
        title="Model Vote Agreement (%)",
        labels={"agreement_pct": "Agreement (%)", "asset": "Asset", "horizon_label": "Horizon"},
        color_discrete_sequence=[C_BLUE, C_GREEN, C_YELLOW],
    )
    fig2.update_layout(**PLOTLY_BASE, yaxis=dict(range=[0, 105]))
    st.plotly_chart(fig2, use_container_width=True)


# ── Section 3: CAN SLIM / SEPA ────────────────────────────────────────────────

def render_canslim():
    st.title("📈 CAN SLIM / SEPA Setups")
    st.markdown("Composite stock rankings using CAN SLIM criteria + SEPA trend template")

    try:
        df = load_canslim()
    except Exception as e:
        st.error(f"Failed to load CAN SLIM data: {e}")
        return

    top_df = df[df["composite_score"] >= 80].sort_values("composite_score", ascending=False).copy()

    section_divider(f"Top Setups — A / A+ Tier  ·  Score ≥ 80  ·  {len(top_df)} stocks")

    if top_df.empty:
        st.info("No A/A+ setups currently. Showing top 20 by composite score instead.")
        top_df = df.sort_values("composite_score", ascending=False).head(20).copy()

    # Build HTML table with TradingView links and tier badges
    def tier_badge_html(score: float, tier) -> str:
        if score >= 90:
            return badge_html("A+", C_GREEN)
        elif score >= 80:
            return badge_html("A", "#16a34a")
        elif score >= 70:
            return badge_html("B+", C_YELLOW)
        elif score >= 60:
            return badge_html("B", "#ca8a04")
        else:
            return badge_html(str(tier) if pd.notna(tier) else "C", C_GREY)

    def tv_link(ticker: str) -> str:
        return (
            f'<a href="https://www.tradingview.com/chart/?symbol={ticker}" '
            f'target="_blank" style="color:{C_BLUE};font-weight:600;'
            f'text-decoration:none">{ticker} ↗</a>'
        )

    rows_html = ""
    for _, r in top_df.iterrows():
        ticker  = str(r["ticker"])
        score   = float(r["composite_score"])
        tier    = r.get("tier", "")
        pattern = str(r.get("best_pattern", "—")).upper() if pd.notna(r.get("best_pattern")) else "—"
        sector  = str(r.get("sector", "—")) if pd.notna(r.get("sector")) else "—"
        price   = r.get("price", None)
        price_s = f"${price:.2f}" if price is not None and pd.notna(price) else "—"
        rows_html += (
            f"<tr>"
            f"<td>{tv_link(ticker)}</td>"
            f"<td style='text-align:center'>{tier_badge_html(score, tier)}</td>"
            f"<td style='text-align:right;color:{score_color(score/10)};font-weight:700'>{score:.1f}</td>"
            f"<td style='color:{C_GREY}'>{pattern}</td>"
            f"<td style='color:#fff'>{sector}</td>"
            f"<td style='color:#aaa'>{price_s}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <style>
        .cs-table {{width:100%;border-collapse:collapse;font-size:0.9em;margin-bottom:12px}}
        .cs-table th {{background:{C_CARD};color:{C_GREY};padding:10px 12px;
                       text-align:left;font-size:0.78em;text-transform:uppercase;
                       letter-spacing:0.05em;border-bottom:1px solid #2d3748}}
        .cs-table td {{padding:8px 12px;border-bottom:1px solid #1e2433;vertical-align:middle}}
        .cs-table tr:hover td {{background:{C_CARD}88}}
        </style>
        <table class="cs-table">
          <thead>
            <tr>
              <th>Ticker</th><th>Tier</th><th style="text-align:right">Score</th>
              <th>Pattern</th><th>Sector</th><th>Price</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    section_divider("Sector Breakdown — A/A+ Setups")

    col_chart, col_score = st.columns([3, 2])
    with col_chart:
        sector_counts = (
            top_df.groupby("sector").size()
            .reset_index(name="count")
            .sort_values("count", ascending=True)
        )
        fig = px.bar(
            sector_counts, x="count", y="sector", orientation="h",
            title="A/A+ Setups by Sector",
            labels={"count": "# Setups", "sector": "Sector"},
            color="count",
            color_continuous_scale=[[0, C_BLUE], [0.5, C_GREEN], [1, C_GREEN]],
        )
        fig.update_layout(**PLOTLY_BASE, showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_score:
        fig2 = px.histogram(
            df, x="composite_score", nbins=20,
            title="Score Distribution (all stocks)",
            labels={"composite_score": "Composite Score"},
            color_discrete_sequence=[C_BLUE],
        )
        fig2.add_vline(x=80, line_dash="dash", line_color=C_GREEN,
                       annotation_text="A threshold (80)")
        fig2.update_layout(**PLOTLY_BASE, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    section_divider("Stock Drill-Down")

    all_tickers = df.sort_values("composite_score", ascending=False)["ticker"].tolist()
    selected = st.selectbox("Select a stock for detailed breakdown", all_tickers)

    if selected:
        row = df[df["ticker"] == selected].iloc[0]
        score = float(row["composite_score"])
        tier  = str(row.get("tier", ""))
        sc    = score_color(score / 10)

        # Score summary header
        st.markdown(
            f'<div style="background:{C_CARD};border-radius:10px;padding:16px 24px;'
            f'margin:10px 0;display:flex;align-items:center;gap:20px">'
            f'<div><span style="font-size:2.5em;font-weight:800;color:{sc}">{score:.1f}</span>'
            f'<span style="color:{C_GREY};font-size:1em"> / 100</span></div>'
            f'<div>{badge_html(tier if tier else "—", sc, "1em")}'
            f'<div style="color:{C_GREY};font-size:0.8em;margin-top:4px">'
            f'Composite Score — {selected}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_tech, col_fund, col_mkt = st.columns(3)

        with col_tech:
            st.markdown("**Technical Sub-Scores**")
            rs = row.get("rs_rating", None)
            tt = row.get("trend_template_pass", None)
            sq = row.get("squeeze_fired", None)
            pt = row.get("best_pattern", None)
            ts = row.get("technical_score", None)
            tech_items = [
                ("Technical Score", f"{ts:.1f}/100" if ts is not None and pd.notna(ts) else "—"),
                ("RS Rating", f"{rs:.0f}" if rs is not None and pd.notna(rs) else "—"),
                ("Trend Template", "✅ Pass" if tt else "❌ Fail"),
                ("Squeeze Fired", "✅ Yes" if sq else "❌ No"),
                ("Best Pattern", str(pt).upper() if pd.notna(pt) else "—"),
            ]
            for k, v in tech_items:
                st.metric(k, v)

        with col_fund:
            st.markdown("**Fundamental Sub-Scores**")
            fs  = row.get("fundamental_score", None)
            ms  = row.get("market_score", None)
            ps  = row.get("pattern_score", None)
            pr  = row.get("price", None)
            sec = row.get("sector", None)
            fund_items = [
                ("Fundamental Score", f"{fs:.1f}/100" if fs is not None and pd.notna(fs) else "—"),
                ("Market Score",      f"{ms:.1f}/100" if ms is not None and pd.notna(ms) else "—"),
                ("Pattern Score",     f"{ps:.1f}/100" if ps is not None and pd.notna(ps) else "—"),
                ("Price",             f"${pr:.2f}" if pr is not None and pd.notna(pr) else "—"),
                ("Sector",            str(sec) if pd.notna(sec) else "—"),
            ]
            for k, v in fund_items:
                st.metric(k, v)

        with col_mkt:
            st.markdown("**Score Component Chart**")
            cats = ["Technical", "Fundamental", "Pattern", "Market"]
            vals = [
                float(row.get("technical_score", 0) or 0),
                float(row.get("fundamental_score", 0) or 0),
                float(row.get("pattern_score", 0) or 0),
                float(row.get("market_score", 0) or 0),
            ]
            colors = [C_BLUE, C_GREEN, C_YELLOW, "#a855f7"]
            fig3 = go.Figure(go.Bar(
                x=cats, y=vals,
                marker_color=colors,
                text=[f"{v:.1f}" for v in vals],
                textposition="outside",
                textfont=dict(color="#fff"),
            ))
            fig3.update_layout(
                **PLOTLY_BASE,
                title=f"{selected} — Sub-Scores",
                yaxis=dict(range=[0, 110]),
                height=320,
            )
            st.plotly_chart(fig3, use_container_width=True)

    section_divider("Backtest Results")
    st.info(
        "Backtest data not included in current data snapshot. "
        "When backtest runs are available, results will display here including: "
        "holding period, entry/exit dates, return per setup, win rate."
    )


# ── Section 4: Crypto ─────────────────────────────────────────────────────────

def render_crypto():
    st.title("🪙 Crypto Rankings")
    st.markdown("Multi-factor analysis: RS · Technical · Fundamental · RRG Regime")

    try:
        df = load_crypto()
    except Exception as e:
        st.error(f"Failed to load crypto data: {e}")
        return

    run_date = df["run_date"].iloc[0].strftime("%Y-%m-%d") if "run_date" in df.columns else "latest"
    st.markdown(f"**Run date:** `{run_date}` — **{len(df)} assets** analysed")

    # Derive tier if missing
    def derive_tier(score):
        if pd.isna(score): return "N/A"
        if score >= 60:    return "A"
        if score >= 50:    return "B"
        if score >= 40:    return "C"
        return "D"

    tier_col = df.get("Tier", pd.Series(dtype=str))
    if tier_col.isna().all() or (tier_col.astype(str) == "None").all():
        df["Tier"] = df["Composite"].apply(derive_tier)

    TIER_COLOR = {"A": C_GREEN, "B": C_YELLOW, "C": C_GREY, "D": C_RED, "N/A": C_GREY}

    section_divider("Rankings Table")

    show_cols = [c for c in ["Rank", "Symbol", "Composite", "Tier",
                              "RS Score", "Tech Score", "Fund Score",
                              "RRG Regime", "RS Momentum", "RS %ile"]
                 if c in df.columns]
    disp = df[show_cols].copy()

    def _tier_style(val):
        c = TIER_COLOR.get(str(val), C_GREY)
        return f"background:{c}25;color:{c};font-weight:700"

    def _comp_style(val):
        try:
            v = float(val)
            if v >= 60: return f"color:{C_GREEN};font-weight:700"
            if v >= 40: return f"color:{C_YELLOW}"
            return f"color:{C_RED}"
        except Exception:
            return ""

    styled = disp.style
    if "Tier" in disp.columns:
        styled = styled.map(_tier_style, subset=["Tier"])
    if "Composite" in disp.columns:
        styled = styled.map(_comp_style, subset=["Composite"])
        styled = styled.format({"Composite": "{:.2f}"}, na_rep="—")
    for nc in ["RS Score", "Tech Score", "Fund Score", "RS Momentum", "RS %ile"]:
        if nc in disp.columns:
            styled = styled.format({nc: "{:.2f}"}, na_rep="—")

    st.dataframe(styled, use_container_width=True, hide_index=True)

    section_divider("Momentum vs Composite Score")

    scatter_df = df.dropna(subset=["RS Momentum", "Composite"]).copy()
    if not scatter_df.empty:
        fig = px.scatter(
            scatter_df,
            x="RS Momentum",
            y="Composite",
            color="Tier",
            text="Symbol",
            color_discrete_map=TIER_COLOR,
            title="RS Momentum vs Composite Score",
            labels={"RS Momentum": "RS Momentum", "Composite": "Composite Score"},
            hover_data=[c for c in ["Rank", "RRG Regime", "RS Score", "Tech Score"]
                        if c in scatter_df.columns],
        )
        fig.update_traces(textposition="top center", marker=dict(size=10, opacity=0.85))
        fig.add_hline(y=50, line_dash="dash", line_color=C_GREY, opacity=0.4,
                      annotation_text="Score = 50")
        fig.add_vline(x=0, line_dash="dash", line_color=C_GREY, opacity=0.4,
                      annotation_text="Momentum = 0")
        fig.update_layout(**PLOTLY_BASE, height=520)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section_divider("Score Distributions")
        fig2 = go.Figure()
        for col_name, color in [("RS Score", C_BLUE), ("Tech Score", C_GREEN),
                                 ("Fund Score", C_YELLOW)]:
            if col_name in df.columns:
                vals = df[col_name].dropna()
                fig2.add_trace(go.Histogram(
                    x=vals, name=col_name, opacity=0.7,
                    marker_color=color, nbinsx=12,
                ))
        fig2.update_layout(**PLOTLY_BASE, title="Score Distributions",
                           barmode="overlay", height=300,
                           xaxis_title="Score", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        if "RRG Regime" in df.columns and not df["RRG Regime"].isna().all():
            section_divider("RRG Regime Breakdown")
            regime_counts = df["RRG Regime"].value_counts().reset_index()
            regime_counts.columns = ["Regime", "Count"]
            fig3 = px.pie(
                regime_counts, values="Count", names="Regime",
                title="Assets by RRG Regime",
                color_discrete_sequence=[C_GREEN, C_YELLOW, C_RED, C_BLUE, C_GREY],
            )
            fig3.update_layout(**PLOTLY_BASE, height=300)
            st.plotly_chart(fig3, use_container_width=True)

    st.info("Fear & Greed equivalent metric: not available in current data snapshot.")


# ── Section 5: S&P 500 Breadth ────────────────────────────────────────────────

def render_breadth():
    st.title("🌊 S&P 500 Market Breadth")
    st.markdown("Weekly internal breadth — index-level and sector-level indicators")

    try:
        bdf = load_breadth_index()
        sdf = load_breadth_sector()
    except Exception as e:
        st.error(f"Failed to load breadth data: {e}")
        return

    latest = bdf.iloc[-1]
    above200 = float(latest["above_200sma"])
    regime, regime_color = regime_info(above200)

    # Regime badge
    st.markdown(
        f'<div style="background:{regime_color}22;border:1px solid {regime_color};'
        f'border-radius:10px;padding:14px 22px;display:inline-block;margin-bottom:18px">'
        f'<span style="color:{regime_color};font-size:1.25em;font-weight:700">'
        f'Market Regime: {regime}</span>'
        f'<span style="color:{C_GREY};font-size:0.85em;margin-left:14px">'
        f'{above200:.1f}% of S&P 500 stocks above 200-day SMA</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_divider("4 Key Breadth Metrics — WoW Delta")

    col1, col2, col3, col4 = st.columns(4)
    metric_cfg = [
        ("% Above 50 SMA",  "above_50sma",  "above_50sma_wow",  col1),
        ("% Above 200 SMA", "above_200sma", "above_200sma_wow", col2),
        ("52W Highs %",     "new_52w_high", "new_52w_high_wow", col3),
        ("52W Lows %",      "new_52w_low",  "new_52w_low_wow",  col4),
    ]
    for label, col_name, wow_col, widget in metric_cfg:
        val = latest.get(col_name, None)
        wow = latest.get(wow_col, None)
        with widget:
            if val is not None and not pd.isna(val):
                delta_str = f"{float(wow):+.1f} pp WoW" if wow is not None and not pd.isna(wow) else None
                st.metric(label, f"{float(val):.1f}%", delta=delta_str)
            else:
                st.metric(label, "—")

    section_divider("Sector Health Table — ETF Tickers")

    if not sdf.empty:
        latest_sector = sdf.sort_values("date").groupby("sector", as_index=False).last()
        max_date = sdf["date"].max()
        four_weeks_ago = max_date - pd.Timedelta(weeks=4)

        rows = []
        for _, sr in latest_sector.iterrows():
            sector_name = str(sr["sector"])
            etf = SECTOR_ETF.get(sector_name, "")
            display_name = f"{sector_name} ({etf})" if etf else sector_name

            a200   = sr.get("above_200sma", None)
            a50    = sr.get("above_50sma", None)
            highs  = sr.get("new_52w_high", None)
            lows   = sr.get("new_52w_low", None)

            # WoW from historical
            sec_hist = sdf[sdf["sector"] == sector_name].sort_values("date")
            if len(sec_hist) >= 2:
                wow = float(sec_hist["above_200sma"].iloc[-1]) - float(sec_hist["above_200sma"].iloc[-2])
            else:
                wow = None

            mom_hist = sec_hist[sec_hist["date"] <= four_weeks_ago]
            if not mom_hist.empty:
                mom = float(sec_hist["above_200sma"].iloc[-1]) - float(mom_hist["above_200sma"].iloc[-1])
            else:
                mom = None

            # Trend arrows
            if wow is not None:
                if   wow >=  5: trend = "↑↑"
                elif wow >=  1: trend = "↑"
                elif wow <= -5: trend = "↓↓"
                elif wow <= -1: trend = "↓"
                else:           trend = "→"
            else:
                trend = "—"

            rows.append({
                "Sector":       display_name,
                "% >200 SMA":  f"{a200:.1f}%" if a200 is not None and pd.notna(a200) else "—",
                "% >50 SMA":   f"{a50:.1f}%"  if a50  is not None and pd.notna(a50)  else "—",
                "WoW pp":       f"{wow:+.1f}" if wow is not None else "—",
                "MoM pp (4W)":  f"{mom:+.1f}" if mom is not None else "—",
                "52W Highs%":  f"{highs:.1f}%" if highs is not None and pd.notna(highs) else "—",
                "52W Lows%":   f"{lows:.1f}%"  if lows  is not None and pd.notna(lows)  else "—",
                "Trend":        trend,
            })

        sector_table = pd.DataFrame(rows)

        def _wow_style(val):
            try:
                v = float(str(val).replace("+", "").replace("%", ""))
                if v > 0: return f"color:{C_GREEN}"
                if v < 0: return f"color:{C_RED}"
            except Exception:
                pass
            return ""

        def _trend_style(val):
            if val in ("↑↑", "↑"): return f"color:{C_GREEN};font-weight:700"
            if val in ("↓↓", "↓"): return f"color:{C_RED};font-weight:700"
            return f"color:{C_GREY}"

        styled_sector = (
            sector_table.style
            .map(_wow_style, subset=["WoW pp", "MoM pp (4W)"])
            .map(_trend_style, subset=["Trend"])
        )
        st.dataframe(styled_sector, use_container_width=True, hide_index=True)

    section_divider("Breadth Time Series")

    metric_options = {
        "above_50sma":  ("% >50 SMA",  C_BLUE),
        "above_200sma": ("% >200 SMA", C_GREEN),
        "new_52w_high": ("52W Highs%", C_YELLOW),
        "new_52w_low":  ("52W Lows%",  C_RED),
    }
    sel_metrics = st.multiselect(
        "Select metrics to display",
        options=list(metric_options.keys()),
        default=["above_50sma", "above_200sma"],
        format_func=lambda x: metric_options[x][0],
    )

    if sel_metrics:
        fig = go.Figure()
        for m in sel_metrics:
            label, color = metric_options[m]
            plot_df = bdf.dropna(subset=[m])
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df[m],
                name=label, mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
            ))
        fig.add_hrect(y0=70, y1=100, fillcolor=C_GREEN, opacity=0.04, line_width=0)
        fig.add_hrect(y0=0,  y1=30,  fillcolor=C_RED,   opacity=0.04, line_width=0)
        fig.add_hline(y=70, line_dash="dot", line_color=C_GREEN, opacity=0.35,
                      annotation_text="Bull (70%)")
        fig.add_hline(y=30, line_dash="dot", line_color=C_RED, opacity=0.35,
                      annotation_text="Bear (30%)")
        fig.update_layout(
            **PLOTLY_BASE,
            title="S&P 500 Market Breadth — Weekly",
            xaxis_title="Date", yaxis_title="Percentage (%)",
            yaxis=dict(range=[0, 100]),
        )
        add_rangeselector(
            fig, ["1M", "3M", "6M", "1Y", "All"],
            default_range=["2024-01-01", bdf["date"].max().strftime("%Y-%m-%d")],
        )
        st.plotly_chart(fig, use_container_width=True)

    section_divider("Sector Comparison Chart")

    metric_choice = st.selectbox(
        "Metric",
        options=["above_200sma", "above_50sma", "new_52w_high", "new_52w_low"],
        format_func=lambda x: metric_options[x][0],
    )

    if not sdf.empty:
        palette11 = [
            "#4f8ef7", "#22c55e", "#eab308", "#ef4444", "#a855f7",
            "#06b6d4", "#f97316", "#ec4899", "#14b8a6", "#8b5cf6", "#84cc16",
        ]
        fig2 = go.Figure()
        for i, sector in enumerate(sorted(sdf["sector"].unique())):
            sdata = sdf[sdf["sector"] == sector].sort_values("date").dropna(subset=[metric_choice])
            etf = SECTOR_ETF.get(sector, "")
            label = f"{sector} ({etf})" if etf else sector
            fig2.add_trace(go.Scatter(
                x=sdata["date"], y=sdata[metric_choice],
                name=label, mode="lines",
                line=dict(color=palette11[i % len(palette11)], width=1.5),
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
            ))
        fig2.update_layout(
            **PLOTLY_BASE,
            title=f"Sector Breadth — {metric_options[metric_choice][0]}",
            xaxis_title="Date", yaxis_title="Percentage (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)),
        )
        add_rangeselector(
            fig2, ["1M", "3M", "6M", "1Y", "All"],
            default_range=["2024-01-01", sdf["date"].max().strftime("%Y-%m-%d")],
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Section 6: Market Risk Factors ────────────────────────────────────────────

def render_risk_factors():
    st.title("⚠️ Market Risk Factors")
    st.markdown("15 macro risk factors — composite scoring (0–10) · green ≥7 · yellow 4–7 · red <4")

    try:
        factors = load_risk_factors()
        ts_df   = load_risk_timeseries()
    except Exception as e:
        st.error(f"Failed to load risk factor data: {e}")
        return

    latest_ts = ts_df.iloc[-1]
    score_20  = float(latest_ts["composite_20"])
    score_50  = float(latest_ts["composite_50"])
    sc20      = score_color(score_20)
    sc50      = score_color(score_50)

    section_divider("Composite Score Cards")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.markdown(
            card_html("20-Day Composite", f"{score_20:.2f} / 10", sc20,
                      f"{'Bullish' if score_20>=7 else 'Neutral' if score_20>=4 else 'Bearish'}",
                      sc20),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            card_html("50-Day Composite", f"{score_50:.2f} / 10", sc50,
                      f"{'Bullish' if score_50>=7 else 'Neutral' if score_50>=4 else 'Bearish'}",
                      sc50),
            unsafe_allow_html=True,
        )
    with c3:
        greens  = sum(1 for f in factors if f.get("signal") == "green")
        yellows = sum(1 for f in factors if f.get("signal") == "yellow")
        reds    = sum(1 for f in factors if f.get("signal") == "red")
        st.markdown(
            f'<div style="background:{C_CARD};border-radius:10px;padding:16px 20px;margin:6px 0">'
            f'<div style="color:{C_GREY};font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em">Signal Summary — 15 Factors</div>'
            f'<div style="margin-top:8px;font-size:1.1em">'
            f'<span style="color:{C_GREEN};font-weight:700">● {greens} Green</span>&nbsp;&nbsp;'
            f'<span style="color:{C_YELLOW};font-weight:700">● {yellows} Yellow</span>&nbsp;&nbsp;'
            f'<span style="color:{C_RED};font-weight:700">● {reds} Red</span>'
            f'</div>'
            f'<div style="color:{C_GREY};font-size:0.8em;margin-top:6px">'
            f'As of {ts_df.iloc[-1]["date"].strftime("%Y-%m-%d")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    section_divider("Composite Risk Score — Historical Trend (2017–Present)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df["date"], y=ts_df["composite_20"],
        name="20-Day Composite", mode="lines",
        line=dict(color=C_BLUE, width=2),
        hovertemplate="<b>20-Day</b><br>%{x|%Y-%m-%d}<br>Score: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ts_df["date"], y=ts_df["composite_50"],
        name="50-Day Composite", mode="lines",
        line=dict(color=C_GREEN, width=2, dash="dot"),
        hovertemplate="<b>50-Day</b><br>%{x|%Y-%m-%d}<br>Score: %{y:.2f}<extra></extra>",
    ))
    fig.add_hrect(y0=7, y1=10, fillcolor=C_GREEN, opacity=0.05, line_width=0)
    fig.add_hrect(y0=0,  y1=4,  fillcolor=C_RED,   opacity=0.05, line_width=0)
    fig.add_hline(y=7, line_dash="dot", line_color=C_GREEN, opacity=0.4,
                  annotation_text="Bullish ≥7")
    fig.add_hline(y=4, line_dash="dot", line_color=C_RED, opacity=0.4,
                  annotation_text="Bearish <4")
    fig.update_layout(
        **PLOTLY_BASE,
        title="Composite Risk Scores — 20-Day & 50-Day",
        xaxis_title="Date", yaxis_title="Composite Score (0–10)",
        yaxis=dict(range=[0, 10]),
    )
    one_year_ago = (ts_df["date"].max() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    add_rangeselector(
        fig, ["1Y", "3Y", "5Y", "All"],
        default_range=[one_year_ago, ts_df["date"].max().strftime("%Y-%m-%d")],
    )
    st.plotly_chart(fig, use_container_width=True)

    section_divider("Multi-Horizon Signal Grid — 15 Factors")

    # Derive 3M signal: if signal == signal_50, use that; otherwise "mixed" (yellow)
    grid_rows = []
    for f in factors:
        sig_1w  = f.get("signal", "—")
        sig_50d = f.get("signal_50", "—")
        # 3M: agree → same, disagree → yellow
        if sig_1w == sig_50d and sig_1w in ("green", "yellow", "red"):
            sig_3m = sig_1w
        elif sig_1w in ("green", "yellow", "red") and sig_50d in ("green", "yellow", "red"):
            sig_3m = "yellow"
        else:
            sig_3m = "—"
        grid_rows.append({
            "Factor":        f.get("factor_name", "Unknown"),
            "Current (1W)":  sig_1w,
            "50-Day (1M)":   sig_50d,
            "3M (derived)":  sig_3m,
            "Value":         f.get("display_value", "—"),
        })

    grid_df = pd.DataFrame(grid_rows)

    def _sig_style(val):
        v = str(val).lower()
        if v == "green":  return f"background:{C_GREEN}25;color:{C_GREEN};font-weight:700"
        if v == "yellow": return f"background:{C_YELLOW}25;color:{C_YELLOW};font-weight:700"
        if v == "red":    return f"background:{C_RED}25;color:{C_RED};font-weight:700"
        return f"color:{C_GREY}"

    styled_grid = (
        grid_df.style
        .map(_sig_style, subset=["Current (1W)", "50-Day (1M)", "3M (derived)"])
    )
    st.dataframe(styled_grid, use_container_width=True, hide_index=True)
    st.caption("3M column derived: matches 1W+50D signal if they agree, otherwise shows yellow (mixed).")

    section_divider("Factor Detail Cards")

    CPR = 3  # cards per row
    for chunk_start in range(0, len(factors), CPR):
        chunk = factors[chunk_start:chunk_start + CPR]
        cols = st.columns(CPR)
        for col, f in zip(cols, chunk):
            with col:
                sig  = f.get("signal", "")
                sc   = sig_color(sig)
                name = f.get("factor_name", "Unknown")
                val  = f.get("display_value", "—")
                txt  = f.get("text", "")
                desc = f.get("description", "")
                s50  = f.get("signal_50", "")
                sc50_c = sig_color(s50)

                st.markdown(
                    f'<div style="background:{C_CARD};border-left:3px solid {sc};'
                    f'padding:12px 14px;border-radius:7px;margin-bottom:10px;min-height:120px">'
                    f'<div style="font-weight:700;font-size:0.88em;color:#fff;'
                    f'margin-bottom:4px">{name}</div>'
                    f'<div style="font-size:1.3em;font-weight:800;color:{sc};'
                    f'margin:3px 0">{val}</div>'
                    f'<div style="margin:4px 0">'
                    f'{badge_html(sig.upper() if sig else "—", sc, "0.72em")}'
                    f'&nbsp;'
                    f'<span style="color:{C_GREY};font-size:0.72em">50d: </span>'
                    f'{badge_html(s50.upper() if s50 else "—", sc50_c, "0.72em")}'
                    f'</div>'
                    f'<div style="font-size:0.75em;color:#aaa;line-height:1.4;margin-top:5px">'
                    f'{txt[:130]}{"…" if len(txt) > 130 else ""}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Section 7: Smart Money ────────────────────────────────────────────────────

def render_smart_money():
    st.title("💰 Smart Money Flows")
    st.markdown("Institutional positioning · Insider transactions · Fund movements")

    smart_dir = DATA_ROOT / "smart_money"
    data_files = []
    if smart_dir.exists():
        data_files = (
            list(smart_dir.glob("*.csv"))
            + list(smart_dir.glob("*.parquet"))
            + list(smart_dir.glob("*.json"))
        )

    if not data_files:
        st.markdown(
            f'<div style="background:{C_CARD};border:1px dashed {C_BLUE};'
            f'border-radius:12px;padding:48px 32px;text-align:center;margin:20px 0">'
            f'<div style="font-size:2.5em;margin-bottom:12px">📋</div>'
            f'<div style="font-size:1.2em;font-weight:700;color:#fff;margin-bottom:8px">'
            f'Insider Pipeline Scheduled — Data Loads Next Run</div>'
            f'<div style="color:{C_GREY};max-width:480px;margin:0 auto;line-height:1.6">'
            f'Institutional flow tracking and insider transaction analysis will '
            f'appear here once the data pipeline runs. Expected data includes '
            f'SEC Form 4 filings, 13F changes, and unusual options activity.</div>'
            f'<div style="margin-top:20px">'
            f'{badge_html("Top Insider Buys", C_BLUE, "0.85em")}&nbsp;'
            f'{badge_html("Fund Movements", C_GREEN, "0.85em")}&nbsp;'
            f'{badge_html("Options Flow", C_YELLOW, "0.85em")}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Planned: Top Insider Buys")
            st.dataframe(
                pd.DataFrame({
                    "Ticker":      ["—", "—", "—"],
                    "Insider":     ["—", "—", "—"],
                    "Transaction": ["Purchase", "Purchase", "Purchase"],
                    "Value ($M)":  ["—", "—", "—"],
                    "Date":        ["—", "—", "—"],
                }),
                hide_index=True,
            )
        with col2:
            st.subheader("Planned: Fund Movements (13F Changes)")
            st.dataframe(
                pd.DataFrame({
                    "Fund":    ["—", "—", "—"],
                    "Action":  ["New Position", "Increase", "Decrease"],
                    "Ticker":  ["—", "—", "—"],
                    "Shares":  ["—", "—", "—"],
                    "Value":   ["—", "—", "—"],
                }),
                hide_index=True,
            )
    else:
        st.success(f"Data available — {len(data_files)} file(s) loaded")
        for fp in sorted(data_files):
            try:
                if fp.suffix == ".csv":
                    fdf = pd.read_csv(fp)
                elif fp.suffix == ".parquet":
                    fdf = pd.read_parquet(fp)
                elif fp.suffix == ".json":
                    with open(fp) as f:
                        fdf = pd.DataFrame(json.load(f))
                else:
                    continue
                st.subheader(fp.stem.replace("_", " ").title())
                st.dataframe(fdf, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not load {fp.name}: {e}")


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = {
    "📊 Overview":           render_overview,
    "🤖 ML Signals":          render_ml_signals,
    "📈 CAN SLIM / SEPA":     render_canslim,
    "🪙 Crypto":              render_crypto,
    "🌊 S&P 500 Breadth":     render_breadth,
    "⚠️ Market Risk Factors": render_risk_factors,
    "💰 Smart Money":         render_smart_money,
}


def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            f'<div style="padding:10px 0 4px;font-size:1.1em;font-weight:700;'
            f'color:#fff">Market Intelligence Hub</div>'
            f'<div style="color:{C_GREY};font-size:0.75em;margin-bottom:16px">'
            f'Powered by Streamlit Cloud</div>',
            unsafe_allow_html=True,
        )

        selection = st.radio(
            "Navigate",
            options=list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            f'<div style="color:{C_GREY};font-size:0.72em;line-height:1.6">'
            f'Data cached: 1h TTL<br>'
            f'Refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Main content ──
    try:
        PAGES[selection]()
    except Exception as e:
        st.error(f"Error rendering section: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
