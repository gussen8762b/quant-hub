"""
Market Intelligence Hub — Streamlit Cloud App
Full subsection structure: 7 sections × multiple tabs, dark theme, sidebar navigation.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
C_BLUE   = "#3b82f6"
C_GREY   = "#6b7280"
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

ASSETS = ["SPY", "QQQ", "GC=F", "BTC-USD"]
ASSET_LABELS = {"SPY": "SPY", "QQQ": "QQQ", "GC=F": "Gold", "BTC-USD": "Bitcoin"}
ASSET_SUBTEXT = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "GC=F": "Gold Futures", "BTC-USD": "BTC/USD"}

HORIZONS = [
    ("short",  "3-day",  "3-Day"),
    ("medium", "10-day", "10-Day"),
    ("long",   "40-day", "40-Day"),
]
HORIZON_LABEL = {"short": "3-Day", "medium": "10-Day", "long": "40-Day"}

DIRECTION_COLOR = {"LONG": C_GREEN, "SHORT": C_RED, "NEUTRAL": C_GREY, "FLAT": C_GREY}
DIRECTION_ICON  = {"LONG": "↑", "SHORT": "↓", "NEUTRAL": "↔", "FLAT": "↔"}

MODEL_COLS = ["tft_pred", "lgb_pred", "xgb_pred", "ridge_pred", "rf_pred", "ensemble_pred"]
MODEL_NAMES = {
    "tft_pred": "TFT",
    "lgb_pred": "LightGBM",
    "xgb_pred": "XGBoost",
    "ridge_pred": "Ridge",
    "rf_pred": "RF",
    "ensemble_pred": "Ensemble",
}

CONF_COLOR = {"HIGH": C_GREEN, "MED": C_YELLOW, "LOW": C_RED}
CONF_BADGE = {"HIGH": "🟢 HIGH", "MED": "🟡 MED", "LOW": "🔴 LOW"}

TIER_COLOR_MAP = {
    "A+": C_GREEN,
    "A":  "#16a34a",
    "B":  C_YELLOW,
    "C":  "#f97316",
    "D":  C_RED,
}

SECTOR_ETF = {
    "Energy": "XLE",
    "Information Technology": "XLK",
    "Technology": "XLK",
    "Materials": "XLB",
    "Financials": "XLF",
    "Financial Services": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Comm Services": "XLC",
}

RRG_COLOR = {
    "leading":   C_GREEN,
    "improving": C_BLUE,
    "weakening": C_YELLOW,
    "lagging":   C_RED,
}
RRG_BADGE = {
    "leading":   "🟢 Leading",
    "improving": "🔵 Improving",
    "weakening": "🟡 Weakening",
    "lagging":   "🔴 Lagging",
}

# ── CSS (dark theme overrides) ────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0d1117; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.stApp { background-color: #0f1117; }
h1,h2,h3,h4,h5,h6 { color: #f1f5f9 !important; }
p, li, span, div, label { color: #cbd5e1; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
.block-container { padding-top: 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; }
.stTabs [data-baseweb="tab"] { padding: 6px 14px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_ml_signals() -> pd.DataFrame:
    p = DATA_ROOT / "ml_signals" / "signal_log_meta.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_signal_history() -> pd.DataFrame:
    p = DATA_ROOT / "ml_signals" / "signal_history_full.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # Mixed formats: "2012-01-02" and "2012-01-02 00:00:00" — use errors='coerce' + strip time
        df["date"] = pd.to_datetime(df["date"].astype(str).str[:10], errors="coerce")
        df = df.dropna(subset=["date"])
        return df.sort_values("date").reset_index(drop=True)
    p = DATA_ROOT / "ml_signals" / "signal_history_full.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_canslim_rankings() -> pd.DataFrame:
    p = DATA_ROOT / "canslim" / "composite_rankings.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for col in ["trend_template_pass", "squeeze_fired"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


@st.cache_data(ttl=3600)
def load_canslim_full() -> pd.DataFrame:
    """Full CAN SLIM screening universe (164 rows, all tiers B/A/C)."""
    p = DATA_ROOT / "canslim" / "screening_results.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        for col in ["trend_template_pass", "squeeze_fired"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_canslim_patterns() -> pd.DataFrame:
    p = DATA_ROOT / "canslim" / "pattern_results.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(ttl=3600)
def load_canslim_backtest() -> dict | None:
    p = DATA_ROOT / "canslim" / "backtest_results.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_crypto() -> pd.DataFrame:
    p = DATA_ROOT / "crypto" / "history.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        if "run_date" in df.columns:
            df["run_date"] = pd.to_datetime(df["run_date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_breadth_index() -> pd.DataFrame:
    p = DATA_ROOT / "breadth" / "breadth_index_weekly_with_wow.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_breadth_sector() -> pd.DataFrame:
    p = DATA_ROOT / "breadth" / "breadth_sector_weekly.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["date"]).sort_values(["date", "sector"]).reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_breadth_ath() -> pd.DataFrame:
    p = DATA_ROOT / "breadth" / "breadth_sector_ath.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_risk_scores() -> list[dict]:
    p = DATA_ROOT / "risk_factors" / "latest_scores.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


@st.cache_data(ttl=3600)
def load_risk_timeseries() -> pd.DataFrame:
    p = DATA_ROOT / "risk_factors" / "composite_timeseries.parquet"
    p_csv = DATA_ROOT / "risk_factors" / "composite_timeseries.csv"
    try:
        if p.exists():
            df = pd.read_parquet(p)
        elif p_csv.exists():
            df = pd.read_csv(p_csv)
        else:
            return pd.DataFrame()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_smart_money_db() -> pd.DataFrame:
    """Load insider transactions from smartmoney.db."""
    import sqlite3
    db_path = DATA_ROOT / "smart_money" / "smartmoney.db"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(db_path))
        query = """
            SELECT
                c.ticker,
                c.name as company_name,
                it.owner_name as insider_name,
                it.owner_role as role_title,
                it.shares,
                it.total_value,
                it.transaction_date,
                it.price_per_share,
                it.security_title,
                it.role_weight,
                COALESCE(cs.total_score, it.role_weight * 1.0) as conviction_score
            FROM insider_transactions it
            JOIN companies c ON it.company_id = c.id
            LEFT JOIN conviction_scores cs ON cs.ticker = c.ticker
            WHERE it.transaction_code = 'P'
            ORDER BY conviction_score DESC, it.transaction_date DESC
            LIMIT 200
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_smart_money_13f() -> pd.DataFrame:
    """Load 13F fund holdings from smartmoney.db."""
    import sqlite3
    db_path = DATA_ROOT / "smart_money" / "smartmoney.db"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(db_path))
        # Check if fund_holdings table exists
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if "fund_holdings" not in tables:
            conn.close()
            return pd.DataFrame()
        query = """
            SELECT
                f.fund_name,
                c.ticker,
                c.name as company_name,
                fh.shares,
                fh.market_value,
                fh.pct_portfolio,
                fh.report_date,
                fh.change_type
            FROM fund_holdings fh
            JOIN funds f ON fh.fund_id = f.id
            JOIN companies c ON fh.company_id = c.id
            ORDER BY fh.market_value DESC
            LIMIT 200
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _file_mtime(path: Path) -> str:
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return "—"


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dir_badge(direction: str, size: str = "0.85rem") -> str:
    col = DIRECTION_COLOR.get(str(direction).upper(), C_GREY)
    icon = DIRECTION_ICON.get(str(direction).upper(), "?")
    return (
        f"<span style='background:{col};color:#fff;padding:2px 10px;"
        f"border-radius:10px;font-weight:700;font-size:{size};'>"
        f"{icon} {direction}</span>"
    )


def _conf_badge(level: str) -> str:
    col = CONF_COLOR.get(str(level).upper(), C_GREY)
    return (
        f"<span style='background:{col}22;color:{col};border:1px solid {col}66;"
        f"padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.78rem;'>"
        f"{CONF_BADGE.get(str(level).upper(), level)}</span>"
    )


def _tier_badge(tier: str) -> str:
    col = TIER_COLOR_MAP.get(tier, C_GREY)
    return (
        f"<span style='background:{col}33;color:{col};border:1px solid {col}88;"
        f"padding:2px 10px;border-radius:6px;font-weight:700;font-size:0.82rem;'>{tier}</span>"
    )


def _rrg_badge(regime: str) -> str:
    col = RRG_COLOR.get(str(regime).lower(), C_GREY)
    label = RRG_BADGE.get(str(regime).lower(), regime)
    return (
        f"<span style='background:{col}22;color:{col};border:1px solid {col}66;"
        f"padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.82rem;'>{label}</span>"
    )


def _pred_arrow(val) -> str:
    """Format raw return prediction (e.g. 0.0527) as coloured percentage arrow."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    pct = v * 100
    if pct > 0.001:
        return f"<span style='color:{C_GREEN};font-weight:700;'>↑ {pct:+.2f}%</span>"
    elif pct < -0.001:
        return f"<span style='color:{C_RED};font-weight:700;'>↓ {pct:+.2f}%</span>"
    return f"<span style='color:{C_GREY};'>→ {pct:.2f}%</span>"


def _rangeselector() -> dict:
    return dict(
        buttons=[
            dict(count=1,  label="1M",  step="month", stepmode="backward"),
            dict(count=3,  label="3M",  step="month", stepmode="backward"),
            dict(count=6,  label="6M",  step="month", stepmode="backward"),
            dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Market Hub")
    st.markdown("---")
    section = st.radio(
        "Navigate",
        options=[
            "🏠 Overview",
            "🤖 ML Signals",
            "📈 CAN SLIM",
            "🪙 Crypto",
            "📊 S&P Breadth",
            "⚠️ Risk Factors",
            "💰 Smart Money",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW SECTION
# ══════════════════════════════════════════════════════════════════════════════

if section == "🏠 Overview":
    st.title("📊 Market Intelligence Hub")
    st.caption("Unified dashboard — ML signals · CAN SLIM · Crypto · Breadth · Risk · Smart Money")
    st.markdown("---")

    cols = st.columns(3)

    # ML Signals card
    with cols[0]:
        try:
            ml = load_ml_signals()
            if not ml.empty:
                latest_date = ml["date"].max().strftime("%Y-%m-%d")
                n_long = int((ml.groupby(["asset","horizon"]).last().reset_index()["ensemble_direction"] == "LONG").sum())
                st.metric("🤖 ML Signals", f"{n_long}/12 LONG", delta=f"As of {latest_date}")
            else:
                st.metric("🤖 ML Signals", "No data")
        except Exception as e:
            st.metric("🤖 ML Signals", "Error")

    # CAN SLIM card
    with cols[1]:
        try:
            cs = load_canslim_rankings()
            if not cs.empty:
                n_elite = int((cs["composite_score"].fillna(0) >= 80).sum())
                last_run = _file_mtime(DATA_ROOT / "canslim" / "composite_rankings.csv")
                st.metric("📈 CAN SLIM", f"{n_elite} A/A+ setups", delta=last_run)
            else:
                st.metric("📈 CAN SLIM", "No data")
        except Exception:
            st.metric("📈 CAN SLIM", "Error")

    # Crypto card
    with cols[2]:
        try:
            cr = load_crypto()
            if not cr.empty:
                latest_rd = cr["run_date"].max()
                cr_latest = cr[cr["run_date"] == latest_rd]
                top = cr_latest.sort_values("Composite", ascending=False).iloc[0]
                sym = str(top.get("Symbol", "?")).replace("/USDT","")
                st.metric("🪙 Crypto", f"#{1} {sym}", delta=f"Composite {top['Composite']:.1f}")
            else:
                st.metric("🪙 Crypto", "No data")
        except Exception:
            st.metric("🪙 Crypto", "Error")

    cols2 = st.columns(3)

    # Breadth card
    with cols2[0]:
        try:
            brd = load_breadth_index()
            if not brd.empty:
                latest_row = brd.iloc[-1]
                pct200 = float(latest_row.get("above_200sma", 0))
                regime = "🟢 BULL" if pct200 > 70 else ("🟡 MIXED" if pct200 > 50 else ("🟠 CAUTION" if pct200 > 30 else "🔴 BEAR"))
                st.metric("📊 S&P Breadth", regime, delta=f"{pct200:.1f}% >200SMA")
            else:
                st.metric("📊 S&P Breadth", "No data")
        except Exception:
            st.metric("📊 S&P Breadth", "Error")

    # Risk card
    with cols2[1]:
        try:
            scores = load_risk_scores()
            if scores:
                green = sum(1 for s in scores if s.get("signal") == "green")
                red   = sum(1 for s in scores if s.get("signal") == "red")
                total = len(scores)
                label = "🟢 LOW" if green >= total * 0.6 else ("🔴 HIGH" if red >= total * 0.6 else "🟡 MIXED")
                st.metric("⚠️ Risk Factors", f"{label}", delta=f"{green}G / {red}R of {total}")
            else:
                st.metric("⚠️ Risk Factors", "No data")
        except Exception:
            st.metric("⚠️ Risk Factors", "Error")

    # Smart Money card
    with cols2[2]:
        st.metric("💰 Smart Money", "Scheduled", delta="Next daily run")

    st.markdown("---")
    st.subheader("Data Status")
    status_data = [
        ("ML Signals",  DATA_ROOT / "ml_signals" / "signal_log_meta.csv"),
        ("CAN SLIM",    DATA_ROOT / "canslim" / "composite_rankings.csv"),
        ("Crypto",      DATA_ROOT / "crypto" / "history.parquet"),
        ("Breadth",     DATA_ROOT / "breadth" / "breadth_index_weekly_with_wow.csv"),
        ("Risk Factors",DATA_ROOT / "risk_factors" / "latest_scores.json"),
    ]
    stat_df = pd.DataFrame([
        {"Source": name, "File": p.name, "Last Updated": _file_mtime(p), "Exists": "✅" if p.exists() else "❌"}
        for name, p in status_data
    ])
    st.dataframe(stat_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# ML SIGNALS SECTION
# ══════════════════════════════════════════════════════════════════════════════

elif section == "🤖 ML Signals":
    st.title("🤖 ML Signals")

    try:
        meta_df = load_ml_signals()
    except Exception as e:
        st.error(f"Failed to load ML signals: {e}")
        meta_df = pd.DataFrame()

    tabs = st.tabs([
        "🎯 Signals",
        "📊 Summary Table",
        "📋 History",
        "📈 P&L",
        "🔄 Performance",
        "🔭 Indicators",
        "🤖 Models",
        "🧩 Ensemble",
        "📖 Overview",
    ])

    # ── Tab 1: Signal Cards ────────────────────────────────────────────────
    with tabs[0]:
        try:
            if meta_df.empty:
                st.warning("No ML signal data found at data/ml_signals/signal_log_meta.csv")
            else:
                latest_date = meta_df["date"].max()
                st.caption(f"Latest signals: **{latest_date.strftime('%Y-%m-%d')}** · {len(ASSETS)} assets × 3 horizons")

                # Get latest per asset+horizon
                latest = (
                    meta_df.sort_values("date")
                    .groupby(["asset", "horizon"])
                    .last()
                    .reset_index()
                )

                # Compute conviction score and sort cards by it
                def _conviction(row):
                    try:
                        vc = float(row.get("vote_count", 0) or 0)
                        nm = float(row.get("n_models", 5) or 5)
                        ep = float(row.get("ensemble_pred", 0) or 0)
                        return (vc / nm * 0.6) + (abs(ep) * 0.4)
                    except Exception:
                        return 0.0

                if not latest.empty:
                    latest["_conviction"] = latest.apply(_conviction, axis=1)
                    if "ensemble_pred" in latest.columns:
                        latest["_abs_ens"] = latest["ensemble_pred"].apply(lambda v: abs(float(v)) if pd.notna(v) else 0)
                    else:
                        latest["_abs_ens"] = 0
                    latest = latest.sort_values(["_conviction", "_abs_ens"], ascending=False)

                # Build ordered list of (asset, horizon) pairs for card layout
                card_order = [(row["asset"], row["horizon"]) for _, row in latest.iterrows()]

                # Group cards by asset for display (preserving conviction sort within each asset)
                for asset in ASSETS:
                    label   = ASSET_LABELS.get(asset, asset)
                    subtext = ASSET_SUBTEXT.get(asset, asset)
                    st.markdown(f"### {label} <span style='color:{C_GREY};font-size:0.8rem'>({subtext})</span>", unsafe_allow_html=True)

                    # Sort this asset's horizons by conviction
                    asset_rows = latest[latest["asset"] == asset]

                    hz_cols = st.columns(3)
                    for col_idx, (_, row_data_s) in enumerate(asset_rows.iterrows()):
                        if col_idx >= 3:
                            break
                        with hz_cols[col_idx]:
                            r = row_data_s
                            hz_key   = r.get("horizon", "")
                            hz_label = HORIZON_LABEL.get(hz_key, hz_key)
                            ens_dir  = str(r.get("ensemble_direction", "NEUTRAL")).upper()
                            conf_lvl = str(r.get("confidence_level", "LOW")).upper()
                            votes    = r.get("vote_count", "?")
                            n_models = r.get("n_models", 5)
                            sig_date = r.get("date", "—")
                            if hasattr(sig_date, "strftime"):
                                sig_date = sig_date.strftime("%Y-%m-%d")
                            conv_val = r.get("_conviction", 0)

                            dir_col  = DIRECTION_COLOR.get(ens_dir, C_GREY)

                            # Build model predictions rows
                            pred_rows = []
                            for mc in MODEL_COLS:
                                if mc in r.index:
                                    mn = MODEL_NAMES.get(mc, mc)
                                    pred_rows.append(f"<span style='color:{C_GREY};font-size:0.72rem;'>{mn}:</span> {_pred_arrow(r[mc])}")

                            preds_html = " &nbsp; ".join(pred_rows)

                            st.markdown(
                                f"<div style='background:{C_CARD};border-radius:8px;padding:14px;"
                                f"border:1px solid {dir_col}55;border-left:3px solid {dir_col};'>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
                                f"<span style='font-weight:700;color:#e2e8f0;'>{hz_label}</span>"
                                f"{_dir_badge(ens_dir)}"
                                f"</div>"
                                f"<div style='margin-bottom:6px;'>{_conf_badge(conf_lvl)}"
                                f" &nbsp; <span style='color:{C_GREY};font-size:0.75rem;'>{votes}/{n_models} agree</span>"
                                f" &nbsp; <span style='color:{C_GREY};font-size:0.7rem;'>conv: {conv_val:.2f}</span>"
                                f"</div>"
                                f"<div style='font-size:0.72rem;line-height:1.8;'>{preds_html}</div>"
                                f"<div style='color:{C_GREY};font-size:0.7rem;margin-top:6px;'>📅 {sig_date}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    # Fill remaining columns if fewer than 3 horizons
                    for col_idx in range(len(asset_rows), 3):
                        with hz_cols[col_idx]:
                            st.markdown(
                                f"<div style='background:{C_CARD};border-radius:8px;padding:12px;"
                                f"border:1px solid #2d3748;'>"
                                f"<div style='color:{C_GREY};'>No data</div></div>",
                                unsafe_allow_html=True,
                            )
                    st.markdown("---")
        except Exception as e:
            st.error(f"Signals tab error: {e}")

    # ── Tab 2: Summary Table ───────────────────────────────────────────────
    with tabs[1]:
        try:
            if meta_df.empty:
                st.info("No data.")
            else:
                latest = (
                    meta_df.sort_values("date")
                    .groupby(["asset", "horizon"])
                    .last()
                    .reset_index()
                )
                latest["Horizon"] = latest["horizon"].map(HORIZON_LABEL).fillna(latest["horizon"])
                latest["Asset"]   = latest["asset"].map(ASSET_LABELS).fillna(latest["asset"])

                display_cols = ["Asset", "Horizon"]
                for mc in MODEL_COLS:
                    if mc in latest.columns:
                        display_cols.append(mc)
                for extra in ["ensemble_direction", "confidence_level", "vote_count"]:
                    if extra in latest.columns:
                        display_cols.append(extra)

                disp = latest[[c for c in display_cols if c in latest.columns]].copy()
                # Format model predictions as percentage strings before rename
                for mc in MODEL_COLS:
                    if mc in disp.columns:
                        disp[mc] = disp[mc].apply(
                            lambda v: f"{float(v)*100:.1f}%" if pd.notna(v) and v != "" else "—"
                        )
                disp = disp.rename(columns={mc: MODEL_NAMES[mc] for mc in MODEL_COLS if mc in disp.columns})
                disp = disp.rename(columns={
                    "ensemble_direction": "Direction",
                    "confidence_level": "Confidence",
                    "vote_count": "Votes",
                })

                def _color_dir(val):
                    col = DIRECTION_COLOR.get(str(val).upper(), "")
                    if col:
                        return f"color: {col}; font-weight: bold"
                    return ""

                styled = disp.style
                if "Direction" in disp.columns:
                    styled = styled.applymap(_color_dir, subset=["Direction"])

                st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Summary table error: {e}")

    # ── Tab 3: History ─────────────────────────────────────────────────────
    with tabs[2]:
        try:
            hist_df = load_signal_history()
            if hist_df.empty:
                # Fall back to meta_df if full history not synced yet
                if meta_df.empty:
                    st.info("No signal history data found. Run sync_data.sh to copy signal_history_full.csv")
                else:
                    st.info("signal_history_full.csv not found — showing signal_log_meta.csv (12 rows). Run sync_data.sh for full history.")
                    hist_df = meta_df.copy()
            else:
                st.caption(f"**signal_history_full.csv** — {len(hist_df):,} rows · dates {hist_df['date'].min().strftime('%Y-%m-%d')} → {hist_df['date'].max().strftime('%Y-%m-%d')}")

            if not hist_df.empty:
                # OOS toggle
                has_oos = "is_oos" in hist_df.columns
                fc0, fc1, fc2, fc3, fc4 = st.columns([1, 2, 2, 2, 1])
                with fc0:
                    if has_oos:
                        oos_only = st.checkbox("OOS only", value=True, key="hist_oos")
                    else:
                        oos_only = False
                with fc1:
                    asset_opts = sorted(hist_df["asset"].dropna().unique().tolist()) if "asset" in hist_df.columns else list(ASSETS)
                    sel_assets = st.multiselect("Asset", options=asset_opts,
                                                default=asset_opts, key="hist_asset2")
                with fc2:
                    hz_opts = sorted(hist_df["horizon"].dropna().unique().tolist()) if "horizon" in hist_df.columns else ["short","medium","long"]
                    sel_hz = st.multiselect("Horizon", options=hz_opts,
                                            default=hz_opts, key="hist_hz2")
                with fc3:
                    model_opts = sorted(hist_df["model"].dropna().unique().tolist()) if "model" in hist_df.columns else []
                    sel_models = st.multiselect("Model", options=model_opts,
                                                default=model_opts, key="hist_model")
                with fc4:
                    show_rows = st.number_input("Rows", min_value=50, max_value=5000, value=500, step=100, key="hist_rows")

                disp = hist_df.copy()
                if oos_only and has_oos:
                    disp = disp[disp["is_oos"] == True]
                if sel_assets and "asset" in disp.columns:
                    disp = disp[disp["asset"].isin(sel_assets)]
                if sel_hz and "horizon" in disp.columns:
                    disp = disp[disp["horizon"].isin(sel_hz)]
                if sel_models and "model" in disp.columns:
                    disp = disp[disp["model"].isin(sel_models)]

                disp = disp.sort_values("date", ascending=False).head(int(show_rows))

                # Format prediction column
                if "prediction" in disp.columns:
                    disp["prediction"] = disp["prediction"].apply(
                        lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "—"
                    )
                if "actual_return" in disp.columns:
                    disp["actual_return"] = disp["actual_return"].apply(
                        lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "—"
                    )
                if "date" in disp.columns:
                    disp["date"] = disp["date"].dt.strftime("%Y-%m-%d") if hasattr(disp["date"].iloc[0], "strftime") else disp["date"].astype(str).str[:10]

                if "is_correct" in disp.columns:
                    disp["is_correct"] = disp["is_correct"].apply(
                        lambda v: "✅ Correct" if v == 1 or v == 1.0 or v == "1" or v == "1.0"
                        else ("❌ Wrong" if v == 0 or v == 0.0 or v == "0" or v == "0.0"
                        else "—")
                    )

                show_cols = [c for c in ["date","asset","horizon","model","direction","prediction","actual_return","is_correct","is_oos"] if c in disp.columns]
                st.dataframe(disp[show_cols], use_container_width=True, hide_index=True, height=500)
                st.caption(f"Showing {len(disp):,} rows (most recent first)")
        except Exception as e:
            st.error(f"History tab error: {e}")

    # ── Tab 4: P&L ────────────────────────────────────────────────────────
    with tabs[3]:
        try:
            hist_df = load_signal_history()
            if hist_df.empty:
                st.info("P&L requires signal_history_full.csv — run sync_data.sh to copy it.")
            else:
                st.markdown("**Model P&L** — cumulative returns using actual_return × directional position (LONG=+1, SHORT/FLAT=-1, NEUTRAL=0)")

                pnl_df = hist_df[hist_df["actual_return"].notna()].copy()
                if pnl_df.empty:
                    st.warning("No rows with actual_return populated yet.")
                else:
                    # Compute position and pnl
                    pnl_df["position"] = pnl_df["direction"].map(
                        {"LONG": 1, "SHORT": -1, "FLAT": -1, "NEUTRAL": 0}
                    ).fillna(0)
                    pnl_df["pnl"] = pnl_df["position"] * pnl_df["actual_return"]

                    # Filters
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        asset_opts = sorted(pnl_df["asset"].dropna().unique().tolist())
                        sel_asset_pnl = st.selectbox("Asset", options=asset_opts, key="pnl_asset")
                    with fc2:
                        hz_opts_pnl = sorted(pnl_df["horizon"].dropna().unique().tolist())
                        sel_hz_pnl = st.selectbox("Horizon", options=hz_opts_pnl, key="pnl_hz")

                    sub = pnl_df[(pnl_df["asset"] == sel_asset_pnl) & (pnl_df["horizon"] == sel_hz_pnl)].copy()
                    if sub.empty:
                        st.info(f"No data for {sel_asset_pnl} / {sel_hz_pnl}")
                    else:
                        # Cumulative P&L per model
                        models_avail = sorted(sub["model"].dropna().unique().tolist()) if "model" in sub.columns else []
                        fig = go.Figure()
                        summary_rows = []

                        for mdl in models_avail:
                            mdl_df = sub[sub["model"] == mdl].sort_values("date").copy()
                            mdl_df["cum_pnl"] = mdl_df["pnl"].cumsum()
                            fig.add_trace(go.Scatter(
                                x=mdl_df["date"], y=mdl_df["cum_pnl"] * 100,
                                mode="lines", name=mdl,
                                hovertemplate=f"{mdl}<br>%{{x|%Y-%m-%d}}<br>Cum P&L: %{{y:.1f}}%<extra></extra>",
                            ))
                            # Summary stats
                            pnl_s = mdl_df["pnl"]
                            cum = mdl_df["cum_pnl"]
                            n = len(pnl_s)
                            total_ret = float(pnl_s.sum()) * 100
                            win_rate = float((pnl_s > 0).mean() * 100) if n > 0 else 0
                            sharpe = float(pnl_s.mean() / pnl_s.std() * math.sqrt(252)) if (n > 1 and pnl_s.std() > 0) else 0
                            max_dd = float(((cum.cummax() - cum) / (cum.cummax() + 1e-9)).max() * 100) if n > 0 else 0
                            summary_rows.append({
                                "Model": mdl,
                                "N Signals": n,
                                "Total Return %": round(total_ret, 2),
                                "Sharpe": round(sharpe, 2),
                                "Max Drawdown %": round(max_dd, 2),
                                "Win Rate %": round(win_rate, 1),
                            })

                        # Buy & Hold benchmark (per model's date range — use first model's dates)
                        if models_avail:
                            bh_df = sub[sub["model"] == models_avail[0]].sort_values("date").copy()
                            bh_df["bh_cum"] = bh_df["actual_return"].cumsum()
                            fig.add_trace(go.Scatter(
                                x=bh_df["date"], y=bh_df["bh_cum"] * 100,
                                mode="lines", name=f"Buy & Hold {sel_asset_pnl}",
                                line=dict(color=C_GREY, dash="dash", width=1.5),
                                hovertemplate=f"Buy & Hold<br>%{{x|%Y-%m-%d}}<br>Cum: %{{y:.1f}}%<extra></extra>",
                            ))

                        rs_pnl = dict(buttons=[
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year",  stepmode="backward"),
                            dict(count=3, label="3Y", step="year",  stepmode="backward"),
                            dict(step="all", label="All"),
                        ])
                        fig.update_layout(
                            **PLOTLY_BASE,
                            height=450,
                            title=f"{sel_asset_pnl} {sel_hz_pnl} — Cumulative P&L",
                            xaxis=dict(rangeselector=rs_pnl, rangeslider=dict(visible=False)),
                            yaxis_title="Cumulative Return %",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        if summary_rows:
                            st.markdown("#### Summary Statistics")
                            sum_df = pd.DataFrame(summary_rows).sort_values("Sharpe", ascending=False)
                            def _ret_color_pnl(val):
                                try:
                                    return f"color: {C_GREEN}" if float(val) > 0 else f"color: {C_RED}"
                                except Exception:
                                    return ""
                            styled_sum = sum_df.style
                            for col in ["Total Return %", "Sharpe"]:
                                if col in sum_df.columns:
                                    styled_sum = styled_sum.applymap(_ret_color_pnl, subset=[col])
                            st.dataframe(styled_sum, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"P&L tab error: {e}")

    # ── Tab 5: Performance (signal flips) ─────────────────────────────────
    with tabs[4]:
        try:
            hist_df = load_signal_history()
            src_df = hist_df if not hist_df.empty else meta_df

            if src_df.empty:
                st.info("No data.")
            else:
                # Last signal detected per asset+horizon+model
                st.markdown("### Signal Status by Asset & Horizon")
                if "asset" in src_df.columns and "horizon" in src_df.columns:
                    grp_cols = ["asset", "horizon"]
                    if "model" in src_df.columns:
                        grp_cols.append("model")
                    last_signals = src_df.sort_values("date").groupby(grp_cols).last().reset_index()
                    for _, row in last_signals.head(12).iterrows():
                        dt = row["date"]
                        dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
                        direction_raw = str(row.get("direction", row.get("ensemble_direction", "—"))).upper()
                        direction_disp = ("🟢 ON (LONG)" if direction_raw == "LONG"
                                          else ("🔴 OFF (SHORT)" if direction_raw == "SHORT"
                                          else direction_raw))
                        asset_label = ASSET_LABELS.get(row.get("asset",""), row.get("asset",""))
                        mdl = row.get("model", "ensemble")
                        hz = HORIZON_LABEL.get(row.get("horizon",""), row.get("horizon",""))
                        st.info(f"{asset_label} {hz} — Model: {mdl} | Direction: {direction_disp} | Date: {dt_str}")

                st.markdown("---")

                # Signal flips
                st.markdown("### Direction Flip History")
                dir_col = "direction" if "direction" in src_df.columns else "ensemble_direction"
                if dir_col not in src_df.columns:
                    st.info("No direction column found.")
                else:
                    flip_group_cols = ["asset", "horizon"]
                    if "model" in src_df.columns:
                        flip_group_cols.append("model")

                    flips = []
                    for keys, grp in src_df.groupby(flip_group_cols):
                        grp = grp.sort_values("date").copy()
                        prev_dir = grp[dir_col].shift(1)
                        changed  = grp[dir_col] != prev_dir
                        changed_rows = grp[changed & prev_dir.notna()]
                        for idx, row in changed_rows.iterrows():
                            dt = row["date"]
                            # 5-day actual return after flip
                            future = grp[grp["date"] > dt].head(5)
                            ret5 = None
                            if "actual_return" in grp.columns and not future.empty:
                                ret5_vals = future["actual_return"].dropna()
                                if not ret5_vals.empty:
                                    ret5 = float(ret5_vals.mean()) * 100

                            asset_k = keys[0] if isinstance(keys, tuple) else keys
                            hz_k    = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else ""
                            mdl_k   = keys[2] if isinstance(keys, tuple) and len(keys) > 2 else ""
                            flips.append({
                                "Date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10],
                                "Asset": ASSET_LABELS.get(asset_k, asset_k),
                                "Horizon": HORIZON_LABEL.get(hz_k, hz_k),
                                "Model": mdl_k,
                                "From": prev_dir[idx],
                                "To": row[dir_col],
                                "5D Return %": f"{ret5:+.2f}%" if ret5 is not None else "—",
                            })

                    if not flips:
                        st.info("No signal flips detected.")
                    else:
                        flip_df = pd.DataFrame(flips).sort_values("Date", ascending=False).head(200)
                        st.markdown(f"**{len(flip_df)} signal flips detected** (showing last 200)")

                        def _color_flip(val):
                            col = DIRECTION_COLOR.get(str(val).upper(), "")
                            return f"color: {col}; font-weight: bold" if col else ""

                        styled = flip_df.style
                        for col in ["From", "To"]:
                            if col in flip_df.columns:
                                styled = styled.applymap(_color_flip, subset=[col])
                        st.dataframe(styled, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("### Per-Model Signal Chart")
                pc1, pc2 = st.columns(2)
                with pc1:
                    sel_asset_perf = st.selectbox("Asset", ASSETS, key="perf_asset")
                with pc2:
                    sel_model_perf = st.selectbox("Model", ["lgb","rf","ridge","xgb","tft"], key="perf_model")

                try:
                    hist_perf = load_signal_history()
                    if not hist_perf.empty and "model" in hist_perf.columns:
                        sub_perf = hist_perf[
                            (hist_perf["asset"] == sel_asset_perf) &
                            (hist_perf["model"] == sel_model_perf)
                        ].sort_values("date").copy()

                        if sub_perf.empty:
                            st.info(f"No history for {sel_asset_perf} / {sel_model_perf}")
                        else:
                            sub_perf["price_proxy"] = sub_perf["actual_return"].cumsum() * 100
                            dir_col_name = "direction" if "direction" in sub_perf.columns else "ensemble_direction"

                            fig_perf = go.Figure()
                            # Price proxy line
                            fig_perf.add_trace(go.Scatter(
                                x=sub_perf["date"], y=sub_perf["price_proxy"],
                                mode="lines", name="Cumulative Return (price proxy)",
                                line=dict(color=C_BLUE, width=2),
                                hovertemplate="%{x|%Y-%m-%d}<br>Cum: %{y:.1f}%<extra></extra>",
                            ))
                            # Background shading for LONG/SHORT
                            if dir_col_name in sub_perf.columns:
                                sub_perf["_dir"] = sub_perf[dir_col_name].str.upper()
                                for i in range(len(sub_perf) - 1):
                                    row_s = sub_perf.iloc[i]
                                    d = row_s["_dir"]
                                    fill_col = "rgba(34,197,94,0.12)" if d == "LONG" else ("rgba(239,68,68,0.12)" if d == "SHORT" else "rgba(0,0,0,0)")
                                    fig_perf.add_vrect(
                                        x0=row_s["date"], x1=sub_perf.iloc[i+1]["date"],
                                        fillcolor=fill_col, opacity=1, layer="below", line_width=0,
                                    )
                            rs_perf = dict(buttons=[
                                dict(count=1, label="1Y", step="year",  stepmode="backward"),
                                dict(count=3, label="3Y", step="year",  stepmode="backward"),
                                dict(count=5, label="5Y", step="year",  stepmode="backward"),
                                dict(step="all", label="All"),
                            ])
                            fig_perf.update_layout(
                                **PLOTLY_BASE, height=450,
                                title=f"{ASSET_LABELS.get(sel_asset_perf, sel_asset_perf)} — {sel_model_perf} signal history",
                                xaxis=dict(rangeselector=rs_perf, rangeslider=dict(visible=False)),
                                yaxis_title="Cumulative Return %",
                            )
                            st.plotly_chart(fig_perf, use_container_width=True)
                            st.caption("🟢 Green background = LONG signal · 🔴 Red background = SHORT signal")
                except Exception as e_perf:
                    st.error(f"Signal chart error: {e_perf}")
        except Exception as e:
            st.error(f"Performance tab error: {e}")

    # ── Tab 6: Indicators ─────────────────────────────────────────────────
    with tabs[5]:
        try:
            st.markdown("### Intermarket Indicator Framework")
            st.markdown("""
The ML pipeline derives features from **intermarket ratios** and macroeconomic indicators
that historically lead or confirm equity and crypto market direction.

**Core ratio clusters:**
""")
            indicators_data = [
                ("FXI / RCD", "China vs US Consumer Discretionary", "Global growth risk appetite"),
                ("EEM / SPY", "Emerging Markets vs US Equities", "Global vs domestic risk"),
                ("HYG / IEI", "High Yield vs Intermediate Treasuries", "Credit risk / risk-on gauge"),
                ("TLT / SPY", "Long Bonds vs S&P 500", "Flight-to-safety signal"),
                ("GLD / USO", "Gold vs Oil", "Commodity regime shift"),
                ("XLE / XLU", "Energy vs Utilities", "Cyclical vs defensive rotation"),
                ("DBC / DXY", "Commodities vs USD", "Dollar vs commodity regime"),
                ("VIX",       "CBOE Volatility Index", "Fear gauge / regime detection"),
                ("EUROSTOXX vs USD", "European equity cluster", "Cross-market confirmation"),
                ("BTC / ETH", "Bitcoin vs Ethereum", "Crypto dominance regime"),
                ("QQQ / SPY", "Nasdaq vs S&P 500", "Tech leadership signal"),
                ("IWM / SPY", "Small Cap vs Large Cap", "Risk appetite / breadth"),
            ]
            ind_df = pd.DataFrame(indicators_data, columns=["Indicator", "Pair Description", "Signal Type"])
            st.dataframe(ind_df, use_container_width=True, hide_index=True)

            if not meta_df.empty and "date" in meta_df.columns:
                st.markdown("### Recent Signal Context")
                latest = meta_df.sort_values("date").groupby(["asset","horizon"]).last().reset_index()
                context_rows = []
                for _, row in latest.iterrows():
                    context_rows.append({
                        "Asset": ASSET_LABELS.get(row["asset"], row["asset"]),
                        "Horizon": HORIZON_LABEL.get(row["horizon"], row["horizon"]),
                        "Ensemble": row.get("ensemble_direction", "—"),
                        "Confidence": row.get("confidence_level", "—"),
                        "Votes": f"{row.get('vote_count','?')}/{row.get('n_models',5)}",
                        "Date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"],"strftime") else str(row["date"])[:10],
                    })
                st.dataframe(pd.DataFrame(context_rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Indicators tab error: {e}")

    # ── Tab 7: Models ─────────────────────────────────────────────────────
    with tabs[6]:
        try:
            st.markdown("### Model Architecture")
            model_cards = [
                ("TFT", "Temporal Fusion Transformer", "Deep learning · attention mechanism · handles multi-horizon sequences. Input: 60-step windows across hourly/daily/weekly bars."),
                ("LightGBM", "LightGBM (Gradient Boosting)", "Fast gradient boosted trees. 200+ features including intermarket ratios, momentum, volatility."),
                ("XGBoost", "XGBoost", "Regularized gradient boosting. Separate models per asset × horizon. Strong on tabular features."),
                ("Ridge", "Ridge Regression (L2)", "Linear baseline with L2 regularization. Interpretable. Useful for momentum/trend feature combinations."),
                ("Random Forest", "Random Forest", "Bootstrap ensemble of decision trees. Robust to noisy features. 500 trees per model."),
                ("Ensemble", "Meta-Ensemble (Voting)", "Weighted vote across TFT+LGB+XGB+Ridge+RF. Weight = directional accuracy on rolling 20-day window."),
            ]
            c1, c2 = st.columns(2)
            for i, (short, full, desc) in enumerate(model_cards):
                col = c1 if i % 2 == 0 else c2
                with col:
                    st.markdown(
                        f"<div style='background:{C_CARD};border-radius:8px;padding:14px;"
                        f"border:1px solid #2d3748;margin-bottom:10px;'>"
                        f"<div style='font-weight:700;color:#f1f5f9;font-size:1rem;'>{short}</div>"
                        f"<div style='color:{C_BLUE};font-size:0.82rem;margin-bottom:4px;'>{full}</div>"
                        f"<div style='color:{C_GREY};font-size:0.78rem;'>{desc}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Per-model directional accuracy from signal_history_full
            try:
                hist_df = load_signal_history()
                if not hist_df.empty and "model" in hist_df.columns and "is_correct" in hist_df.columns:
                    st.markdown("### Per-Model Directional Accuracy (Historical)")
                    st.markdown("Mean(is_correct) per model × asset from signal_history_full.csv")

                    with st.spinner("Computing model accuracy..."):
                        acc_df = (
                            hist_df.groupby(["model", "asset"])["is_correct"]
                            .mean()
                            .reset_index()
                        )
                    acc_df["is_correct"] = (acc_df["is_correct"] * 100).round(1)
                    acc_df["asset_label"] = acc_df["asset"].map(ASSET_LABELS).fillna(acc_df["asset"])

                    try:
                        pivot = acc_df.pivot(index="model", columns="asset_label", values="is_correct")
                        fig_hm = go.Figure(go.Heatmap(
                            z=pivot.values,
                            x=list(pivot.columns),
                            y=list(pivot.index),
                            colorscale=[[0, C_RED],[0.5, C_YELLOW],[1, C_GREEN]],
                            zmin=0, zmax=100,
                            text=[[f"{v:.0f}%" if not pd.isna(v) else "" for v in row] for row in pivot.values],
                            texttemplate="%{text}",
                            hovertemplate="Model: %{y}<br>Asset: %{x}<br>Accuracy: %{z:.1f}%<extra></extra>",
                        ))
                        fig_hm.update_layout(**PLOTLY_BASE, height=350, margin=dict(l=120,r=20,t=40,b=60))
                        st.plotly_chart(fig_hm, use_container_width=True)
                    except Exception as e2:
                        st.dataframe(acc_df, use_container_width=True, hide_index=True)

                elif not meta_df.empty and "ensemble_direction" in meta_df.columns:
                    st.markdown("### Model Directional Accuracy vs Ensemble (Latest only)")
                    st.caption("Install signal_history_full.csv for full historical accuracy.")

                    acc_rows = []
                    for asset in ASSETS:
                        asset_df = meta_df[meta_df["asset"] == asset]
                        for mc in MODEL_COLS[:-1]:
                            if mc not in asset_df.columns:
                                continue
                            ens = asset_df["ensemble_direction"].str.upper()
                            pred_dir = asset_df[mc].apply(lambda v: "LONG" if (not pd.isna(v) and float(v) > 0) else ("SHORT" if (not pd.isna(v) and float(v) < 0) else "NEUTRAL"))
                            acc = (pred_dir == ens).mean()
                            acc_rows.append({
                                "Model": MODEL_NAMES.get(mc, mc),
                                "Asset": ASSET_LABELS.get(asset, asset),
                                "Accuracy": round(acc * 100, 1),
                            })
                    if acc_rows:
                        acc_df2 = pd.DataFrame(acc_rows)
                        try:
                            pivot2 = acc_df2.pivot(index="Model", columns="Asset", values="Accuracy")
                            fig_hm2 = go.Figure(go.Heatmap(
                                z=pivot2.values,
                                x=list(pivot2.columns),
                                y=list(pivot2.index),
                                colorscale=[[0, C_RED],[0.5, C_YELLOW],[1, C_GREEN]],
                                zmin=0, zmax=100,
                                text=[[f"{v:.0f}%" for v in row] for row in pivot2.values],
                                texttemplate="%{text}",
                                hovertemplate="Model: %{y}<br>Asset: %{x}<br>Accuracy: %{z:.1f}%<extra></extra>",
                            ))
                            fig_hm2.update_layout(**PLOTLY_BASE, height=300, margin=dict(l=100,r=20,t=40,b=60))
                            st.plotly_chart(fig_hm2, use_container_width=True)
                        except Exception as e3:
                            st.dataframe(acc_df2, use_container_width=True, hide_index=True)
            except Exception as e4:
                st.error(f"Model accuracy heatmap error: {e4}")
        except Exception as e:
            st.error(f"Models tab error: {e}")

    # ── Tab 8: Ensemble ───────────────────────────────────────────────────
    with tabs[7]:
        try:
            if meta_df.empty:
                st.info("No data.")
            else:
                st.markdown("### Model Agreement Grid")
                st.markdown("Latest vote_count/n_models per asset × horizon. Darker green = higher consensus.")

                try:
                    with st.spinner("Loading ensemble data..."):
                        latest = (
                            meta_df.sort_values("date")
                            .groupby(["asset","horizon"])
                            .last()
                            .reset_index()
                        )
                except Exception as e_grp:
                    st.error(f"Could not group data: {e_grp}")
                    latest = pd.DataFrame()

                if not latest.empty and "vote_count" in latest.columns and "n_models" in latest.columns:
                    try:
                        latest["vote_count"] = pd.to_numeric(latest["vote_count"], errors="coerce")
                        latest["n_models"]   = pd.to_numeric(latest["n_models"], errors="coerce")
                        latest["agreement"]  = latest["vote_count"] / latest["n_models"].replace(0, np.nan)
                        pivot = latest.pivot(index="asset", columns="horizon", values="agreement")
                        pivot.index = [ASSET_LABELS.get(a, a) for a in pivot.index]
                        hz_order = ["short","medium","long"]
                        pivot = pivot.reindex(columns=[h for h in hz_order if h in pivot.columns])
                        pivot.columns = [HORIZON_LABEL.get(c, c) for c in pivot.columns]

                        # Text overlay — use numeric values safely
                        votes_pivot = latest.pivot(index="asset", columns="horizon", values="vote_count")
                        nmod_pivot  = latest.pivot(index="asset", columns="horizon", values="n_models")
                        votes_pivot.index = [ASSET_LABELS.get(a, a) for a in votes_pivot.index]
                        nmod_pivot.index  = [ASSET_LABELS.get(a, a) for a in nmod_pivot.index]
                        votes_pivot = votes_pivot.reindex(columns=[h for h in hz_order if h in votes_pivot.columns])
                        nmod_pivot  = nmod_pivot.reindex(columns=[h for h in hz_order if h in nmod_pivot.columns])
                        votes_pivot.columns = [HORIZON_LABEL.get(c, c) for c in votes_pivot.columns]
                        nmod_pivot.columns  = [HORIZON_LABEL.get(c, c) for c in nmod_pivot.columns]

                        def _safe_frac(r, c):
                            try:
                                v = votes_pivot.iloc[r, c]
                                n = nmod_pivot.iloc[r, c]
                                if pd.isna(v) or pd.isna(n):
                                    return "—"
                                return f"{int(v)}/{int(n)}"
                            except Exception:
                                return "—"

                        text = [[_safe_frac(r, c) for c in range(pivot.shape[1])]
                                for r in range(pivot.shape[0])]

                        # Ensure z data is numeric
                        z_vals = pivot.values.astype(float)

                        fig = go.Figure(go.Heatmap(
                            z=z_vals,
                            x=list(pivot.columns),
                            y=list(pivot.index),
                            colorscale=[[0, C_RED],[0.4, C_YELLOW],[0.7, C_GREEN],[1.0,"#15803d"]],
                            zmin=0, zmax=1,
                            text=text,
                            texttemplate="%{text}",
                            hovertemplate="Asset: %{y}<br>Horizon: %{x}<br>Agreement: %{z:.0%}<extra></extra>",
                        ))
                        fig.update_layout(**PLOTLY_BASE, height=350, margin=dict(l=80,r=20,t=40,b=60))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e_hm:
                        st.error(f"Agreement heatmap error: {e_hm}")
                        # Fallback table
                        if "vote_count" in latest.columns:
                            st.dataframe(latest[["asset","horizon","vote_count","n_models"]].fillna("—"),
                                         use_container_width=True, hide_index=True)

                    # Direction overlay table
                    try:
                        if "ensemble_direction" in latest.columns:
                            hz_order = ["short","medium","long"]
                            dir_pivot = latest.pivot(index="asset", columns="horizon", values="ensemble_direction")
                            dir_pivot.index   = [ASSET_LABELS.get(a, a) for a in dir_pivot.index]
                            dir_pivot         = dir_pivot.reindex(columns=[h for h in hz_order if h in dir_pivot.columns])
                            dir_pivot.columns = [HORIZON_LABEL.get(c, c) for c in dir_pivot.columns]
                            st.markdown("**Current ensemble directions:**")
                            st.dataframe(dir_pivot, use_container_width=True)
                    except Exception as e_dir:
                        st.error(f"Direction table error: {e_dir}")
                else:
                    st.info("vote_count or n_models columns not found in data.")
        except Exception as e:
            st.error(f"Ensemble tab error: {e}")

    # ── Tab 9: Overview ───────────────────────────────────────────────────
    with tabs[8]:
        try:
            st.markdown("""
## Intermarket Ratio ML Pipeline

An end-to-end machine learning system that generates **directional signals** for 4 assets
(SPY, QQQ, Gold, BTC) across **3 horizons** (3-day, 10-day, 40-day) using 5 ML models
combined into a meta-ensemble.

### Phase Summary
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data collection (OHLCV + intermarket ratios) | ✅ Complete |
| 2 | Feature engineering (200+ features per asset) | ✅ Complete |
| 3 | Model training — LightGBM + XGBoost + Ridge + RF | ✅ Complete |
| 4 | TFT (Temporal Fusion Transformer) multi-horizon | ✅ Complete |
| 5 | Meta-ensemble (weighted voting) | ✅ Complete |
| 6 | Signal logging + history tracking | ✅ Complete |
| 7 | Telegram alerts (morning 9:30 AM ET) | ✅ Complete |
| 8A | Dashboard + hub integration | ✅ Complete |

### How to Interpret Signals

| Signal | Meaning | Action |
|--------|---------|--------|
| 🟢 LONG + HIGH conf | ≥4 models agree bullish | Consider long position |
| 🟢 LONG + MED conf | 3 models agree bullish | Watch for confirmation |
| 🔴 SHORT + HIGH conf | ≥4 models agree bearish | Consider hedge/short |
| ⚪ NEUTRAL | Models disagree | No directional edge |

**Key rule**: Only act on HIGH confidence signals where vote_count ≥ 4/5.
Short-horizon (3-day) signals are noisier; 40-day signals carry more weight.

### Data cadence
- Short (3-day): hourly bars, updated daily morning
- Medium (10-day): daily bars, updated daily morning
- Long (40-day): weekly bars, updated weekly
""")
        except Exception as e:
            st.error(f"Overview tab error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CAN SLIM SECTION
# ══════════════════════════════════════════════════════════════════════════════

elif section == "📈 CAN SLIM":
    st.title("📈 CAN SLIM + SEPA/VCP Screener")

    try:
        cs_df      = load_canslim_rankings()
        full_cs_df = load_canslim_full()
        pat_df     = load_canslim_patterns()
    except Exception as e:
        st.error(f"Failed to load CAN SLIM data: {e}")
        cs_df      = pd.DataFrame()
        full_cs_df = pd.DataFrame()
        pat_df     = pd.DataFrame()

    tabs = st.tabs([
        "🏆 Top Setups",
        "📊 Analytics",
        "🔍 Stock Drilldown",
        "📈 Backtest",
        "📖 About",
    ])

    # ── Tab 1: Top Setups ─────────────────────────────────────────────────
    with tabs[0]:
        try:
            if cs_df.empty:
                st.warning("No CAN SLIM data found at data/canslim/composite_rankings.csv")
            else:
                elite = cs_df[cs_df["composite_score"].fillna(0) >= 80].copy() if "composite_score" in cs_df.columns else cs_df.copy()

                # Filters
                fc1, fc2 = st.columns(2)
                with fc1:
                    tier_opts = sorted(cs_df["tier"].dropna().unique()) if "tier" in cs_df.columns else []
                    sel_tiers = st.multiselect("Tier", options=tier_opts, default=[t for t in ["A+","A"] if t in tier_opts])
                with fc2:
                    sector_opts = sorted(cs_df["sector"].dropna().unique()) if "sector" in cs_df.columns else []
                    sel_sectors = st.multiselect("Sector", options=sector_opts, default=sector_opts)

                filtered = elite.copy()
                if sel_tiers and "tier" in filtered.columns:
                    filtered = filtered[filtered["tier"].isin(sel_tiers)]
                if sel_sectors and "sector" in filtered.columns:
                    filtered = filtered[filtered["sector"].isin(sel_sectors)]
                if "composite_score" in filtered.columns:
                    filtered = filtered.sort_values("composite_score", ascending=False)

                today_str = datetime.now().strftime("%Y-%m-%d")
                st.success(f"✅ Screened today ({today_str}): {len(filtered)} setups")

                display_cols = ["ticker","tier","composite_score","best_pattern","rs_rating",
                                "trend_template_pass","squeeze_fired","price","sector"]
                show_cols = [c for c in display_cols if c in filtered.columns]
                disp = filtered[show_cols].copy().reset_index(drop=True)
                # Add derived columns
                disp["🔥 Active"] = disp["squeeze_fired"].apply(lambda v: "🔥 Active" if bool(v) else "") if "squeeze_fired" in disp.columns else ""
                disp["📅 Screened"] = today_str
                disp = disp.rename(columns={
                    "ticker": "Ticker", "tier": "Tier", "composite_score": "Score",
                    "best_pattern": "Pattern", "rs_rating": "RS",
                    "trend_template_pass": "Trend✓", "squeeze_fired": "Squeeze",
                    "price": "Price", "sector": "Sector",
                })
                if "Score" in disp.columns:
                    disp["Score"] = disp["Score"].round(1)
                if "RS" in disp.columns:
                    disp["RS"] = pd.to_numeric(disp["RS"], errors="coerce").round(0)
                if "Price" in disp.columns:
                    disp["Price"] = pd.to_numeric(disp["Price"], errors="coerce").round(2)
                if "Trend✓" in disp.columns:
                    disp["Trend✓"] = disp["Trend✓"].apply(lambda v: "✅" if bool(v) else "❌")
                if "Squeeze" in disp.columns:
                    disp["Squeeze"] = disp["Squeeze"].apply(lambda v: "🔥" if bool(v) else "")

                def _tier_color(val):
                    col = TIER_COLOR_MAP.get(str(val), "")
                    return f"color: {col}; font-weight: bold" if col else ""

                styled = disp.style
                if "Tier" in disp.columns:
                    styled = styled.applymap(_tier_color, subset=["Tier"])
                if "Score" in disp.columns:
                    try:
                        styled = styled.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=60, vmax=100)
                    except Exception:
                        pass  # fallback: no gradient

                st.dataframe(styled, use_container_width=True, hide_index=True)

                # TradingView links
                if "Ticker" in disp.columns:
                    tickers = disp["Ticker"].tolist()[:50]
                    links = " · ".join([
                        f'<a href="https://www.tradingview.com/chart/?symbol={t}" target="_blank" '
                        f'style="color:{C_BLUE};">{t}</a>'
                        for t in tickers
                    ])
                    st.markdown(f"<div style='font-size:0.8rem;'>📈 {links}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Top Setups error: {e}")

    # ── Tab 2: Analytics ──────────────────────────────────────────────────
    with tabs[1]:
        try:
            analytics_src = full_cs_df if not full_cs_df.empty else cs_df
            if analytics_src.empty:
                st.info("No data.")
            else:
                # Tier selector
                tier_opts_analytics = ["A+","A","B","C","D"]
                sel_tiers_analytics = st.multiselect(
                    "Show tiers", tier_opts_analytics,
                    default=tier_opts_analytics,
                    key="analytics_tiers"
                )
                if sel_tiers_analytics and "tier" in analytics_src.columns:
                    filtered_full = analytics_src[analytics_src["tier"].isin(sel_tiers_analytics)].copy()
                else:
                    filtered_full = analytics_src.copy()

                st.caption(f"Showing {len(filtered_full)} stocks from full universe ({len(analytics_src)} total)")

                c1, c2 = st.columns(2)

                # Tier distribution
                with c1:
                    st.markdown("#### Tier Distribution")
                    if "tier" in filtered_full.columns:
                        tier_counts = filtered_full["tier"].value_counts().reindex(["A+","A","B","C","D"]).fillna(0).reset_index()
                        tier_counts.columns = ["Tier","Count"]
                        tier_counts["color"] = tier_counts["Tier"].map(TIER_COLOR_MAP).fillna(C_GREY)
                        fig = go.Figure(go.Bar(
                            x=tier_counts["Tier"], y=tier_counts["Count"],
                            marker_color=tier_counts["color"].tolist(),
                            text=tier_counts["Count"].astype(int),
                            textposition="outside",
                        ))
                        fig.update_layout(**PLOTLY_BASE, height=300, showlegend=False,
                                          xaxis_title="Tier", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)

                # Score distribution
                with c2:
                    st.markdown("#### Score Distribution")
                    if "composite_score" in filtered_full.columns:
                        scores = filtered_full["composite_score"].dropna()
                        # Color by tier
                        fig = go.Figure()
                        for tier_val in ["A+","A","B","C","D"]:
                            if "tier" in filtered_full.columns:
                                tier_scores = filtered_full[filtered_full["tier"] == tier_val]["composite_score"].dropna()
                            else:
                                tier_scores = scores
                            if not tier_scores.empty:
                                fig.add_trace(go.Histogram(
                                    x=tier_scores, nbinsx=15, name=tier_val,
                                    marker_color=TIER_COLOR_MAP.get(tier_val, C_GREY), opacity=0.75,
                                ))
                        fig.update_layout(**PLOTLY_BASE, height=300, barmode="overlay",
                                          xaxis_title="Composite Score", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)

                # Sector breakdown
                st.markdown("#### Sector Breakdown")
                if "sector" in filtered_full.columns and "tier" in filtered_full.columns:
                    sec_tier = filtered_full.groupby(["sector","tier"]).size().reset_index(name="Count")
                    sec_order = sec_tier.groupby("sector")["Count"].sum().sort_values(ascending=True).index.tolist()
                    fig = go.Figure()
                    for tier_val in ["A+","A","B","C","D"]:
                        td = sec_tier[sec_tier["tier"] == tier_val]
                        if not td.empty:
                            fig.add_trace(go.Bar(
                                y=td["sector"], x=td["Count"],
                                name=tier_val,
                                orientation="h",
                                marker_color=TIER_COLOR_MAP.get(tier_val, C_GREY),
                                text=td["Count"],
                                textposition="auto",
                            ))
                    fig.update_layout(**PLOTLY_BASE, height=max(300, len(sec_order)*28),
                                      xaxis_title="# Stocks", yaxis_title="",
                                      barmode="stack",
                                      yaxis=dict(categoryorder="array", categoryarray=sec_order))
                    st.plotly_chart(fig, use_container_width=True)

                # Score components by tier
                st.markdown("#### Avg Score Components by Tier")
                score_cols = [c for c in ["technical_score","fundamental_score","pattern_score","market_score"] if c in filtered_full.columns]
                if score_cols and "tier" in filtered_full.columns:
                    avg_by_tier = filtered_full.groupby("tier")[score_cols].mean().reindex(["A+","A","B","C","D"]).dropna(how="all")
                    fig = go.Figure()
                    colors = [C_BLUE, C_GREEN, C_YELLOW, "#a855f7"]
                    labels = {"technical_score":"Technical","fundamental_score":"Fundamental",
                              "pattern_score":"Pattern","market_score":"Market"}
                    for i, col in enumerate(score_cols):
                        fig.add_trace(go.Bar(
                            name=labels.get(col, col),
                            x=avg_by_tier.index.tolist(),
                            y=avg_by_tier[col].tolist(),
                            marker_color=colors[i % len(colors)],
                        ))
                    fig.update_layout(**PLOTLY_BASE, height=350, barmode="group",
                                      xaxis_title="Tier", yaxis_title="Avg Score")
                    st.plotly_chart(fig, use_container_width=True)

                # TradingView links per tier
                if "ticker" in filtered_full.columns and "tier" in filtered_full.columns:
                    st.markdown("#### TradingView Links by Tier")
                    tier_link_parts = []
                    for tier_val in ["A+","A","B","C"]:
                        tier_tickers = filtered_full[filtered_full["tier"] == tier_val]["ticker"].dropna().tolist()[:20]
                        if tier_tickers:
                            links_str = " ".join([
                                f'<a href="https://www.tradingview.com/chart/?symbol={t}" target="_blank" '
                                f'style="color:{TIER_COLOR_MAP.get(tier_val, C_BLUE)};">{t}</a>'
                                for t in tier_tickers
                            ])
                            tier_link_parts.append(f"<b style='color:{TIER_COLOR_MAP.get(tier_val, C_GREY)};'>{tier_val}:</b> {links_str}")
                    if tier_link_parts:
                        st.markdown(
                            "<div style='font-size:0.8rem;line-height:2.2;'>" +
                            " &nbsp;|&nbsp; ".join(tier_link_parts) +
                            "</div>",
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.error(f"Analytics tab error: {e}")

    # ── Tab 3: Stock Drilldown ────────────────────────────────────────────
    with tabs[2]:
        try:
            if cs_df.empty:
                st.info("No data.")
            else:
                elite_df = cs_df[cs_df["composite_score"].fillna(0) >= 80] if "composite_score" in cs_df.columns else cs_df
                tickers  = sorted(elite_df["ticker"].dropna().unique().tolist()) if "ticker" in elite_df.columns else []

                if not tickers:
                    st.info("No A/A+ stocks (composite score ≥ 80) found.")
                else:
                    sel_ticker = st.selectbox("Select ticker (A/A+ only)", options=tickers)
                    row = elite_df[elite_df["ticker"] == sel_ticker]
                    if row.empty:
                        st.warning("No data.")
                    else:
                        r = row.iloc[0]
                        tier  = str(r.get("tier", "?"))
                        score = float(r.get("composite_score", 0))

                        st.markdown(
                            f"## {sel_ticker} &nbsp; {_tier_badge(tier)} &nbsp; "
                            f"<span style='color:{C_GREY};'>Score: {score:.1f}</span>"
                            f" &nbsp; <a href='https://www.tradingview.com/chart/?symbol={sel_ticker}' "
                            f"target='_blank' style='font-size:0.75rem;color:{C_BLUE};'>📈 TradingView</a>",
                            unsafe_allow_html=True,
                        )

                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Composite",  f"{score:.1f}")
                        m2.metric("RS Rating",  f"{r.get('rs_rating', '—')}")
                        m3.metric("Pattern",    str(r.get("best_pattern", "—")))
                        m4.metric("Trend ✓",    "✅" if bool(r.get("trend_template_pass", False)) else "❌")
                        m5.metric("Squeeze",    "🔥" if bool(r.get("squeeze_fired", False)) else "No")

                        st.markdown("---")
                        col_tech, col_fund, col_pat = st.columns(3)

                        with col_tech:
                            st.markdown("**⚡ Technical**")
                            st.write(f"Score: **{r.get('technical_score', '—')}**")
                            st.write(f"RS Rating: **{r.get('rs_rating', '—')}**")
                            st.write(f"Trend Template: {'✅' if bool(r.get('trend_template_pass', False)) else '❌'}")
                            st.write(f"Squeeze: {'🔥 Fired' if bool(r.get('squeeze_fired', False)) else 'No'}")
                            st.write(f"Pattern: **{r.get('best_pattern', '—')}**")

                        with col_fund:
                            st.markdown("**💼 Fundamental**")
                            st.write(f"Score: **{r.get('fundamental_score', '—')}**")
                            st.write(f"Market Score: **{r.get('market_score', '—')}**")
                            st.caption("EPS/Revenue data not available in current export")

                        with col_pat:
                            st.markdown("**🔍 Pattern**")
                            pat_row = pat_df[pat_df["ticker"] == sel_ticker] if not pat_df.empty and "ticker" in pat_df.columns else pd.DataFrame()
                            if not pat_row.empty:
                                pr = pat_row.iloc[0]
                                st.write(f"VCP: {'✅' if bool(pr.get('vcp_found', False)) else '❌'}")
                                st.write(f"VCP Contractions: **{pr.get('vcp_contractions', '—')}**")
                                st.write(f"VCP Depth: **{pr.get('vcp_depth_pct', '—')}%**")
                                st.write(f"Cup: {'✅' if bool(pr.get('cup_found', False)) else '❌'}")
                                st.write(f"Double Bottom: {'✅' if bool(pr.get('double_bottom_found', False)) else '❌'}")
                                st.write(f"Flat Base: {'✅' if bool(pr.get('flat_base_found', False)) else '❌'}")
                            else:
                                st.caption("No pattern data for this ticker")

                        # Score components chart
                        st.markdown("---")
                        sub_scores = {
                            "Technical": r.get("technical_score", 0),
                            "Fundamental": r.get("fundamental_score", 0),
                            "Pattern": r.get("pattern_score", 0),
                            "Market": r.get("market_score", 0),
                        }
                        valid_sub = {k: float(v) for k,v in sub_scores.items() if v is not None and not pd.isna(v)}
                        if valid_sub:
                            fig = go.Figure(go.Bar(
                                x=list(valid_sub.keys()),
                                y=list(valid_sub.values()),
                                marker_color=[C_BLUE, C_GREEN, C_YELLOW, "#a855f7"],
                                text=[f"{v:.1f}" for v in valid_sub.values()],
                                textposition="outside",
                            ))
                            fig.update_layout(**PLOTLY_BASE, height=300, showlegend=False,
                                              yaxis=dict(range=[0,105]), yaxis_title="Score")
                            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Drilldown tab error: {e}")

    # ── Tab 4: Backtest ───────────────────────────────────────────────────
    with tabs[3]:
        try:
            bt = load_canslim_backtest()
            if bt is None:
                st.info("No backtest data at data/canslim/backtest_results.json")
            else:
                period = bt.get("period", "N/A")
                mode   = bt.get("mode", "walkforward").replace("walkforward", "Walk-Forward")
                st.markdown(
                    f"<div style='background:{C_CARD};border-left:4px solid {C_BLUE};"
                    f"padding:14px 18px;border-radius:6px;margin-bottom:16px;'>"
                    f"<div style='color:#e2e8f0;font-weight:700;font-size:1rem;margin-bottom:8px;'>"
                    f"📈 CAN SLIM Long-Only Strategy — {mode} Backtest</div>"
                    f"<div style='color:{C_GREY};font-size:0.82rem;line-height:1.8;'>"
                    f"<b style='color:#cbd5e1;'>Universe:</b> S&P 500 constituents passing Trend Template &nbsp;|&nbsp; "
                    f"<b style='color:#cbd5e1;'>Entry:</b> A / A+ tier (composite score ≥ 80), pattern trigger &nbsp;|&nbsp; "
                    f"<b style='color:#cbd5e1;'>Exit:</b> 30-day hold or 8% stop-loss<br>"
                    f"<b style='color:#cbd5e1;'>Benchmark:</b> SPY buy-and-hold &nbsp;|&nbsp; "
                    f"<b style='color:#cbd5e1;'>Period:</b> {period}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                spy_ret  = float(bt.get("spy_return", 0) or 0)
                strat_ret = float(bt.get("total_return", 0) or 0)
                alpha    = strat_ret - spy_ret

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Total Return",  f"{strat_ret:+.2f}%", delta=f"SPY {spy_ret:+.2f}%")
                m2.metric("Ann. Return",   f"{bt.get('annualized_return', 0):+.2f}%")
                m3.metric("Sharpe",        f"{bt.get('sharpe_ratio', 0):.2f}")
                m4.metric("Max DD",        f"{bt.get('max_drawdown', 0):.2f}%")
                m5.metric("Win Rate",      f"{bt.get('win_rate', 0):.1f}%")
                m6.metric("# Trades",      str(bt.get("num_trades", 0)))

                trades = bt.get("trades")
                # Compute median return from trades list
                median_ret = 0.0
                if trades:
                    trades_df_tmp = pd.DataFrame(trades)
                    if "return_pct" in trades_df_tmp.columns:
                        median_ret = float(trades_df_tmp["return_pct"].median())

                stat_df = pd.DataFrame({
                    "Metric": ["Avg Return/Trade", "Median Return/Trade", "Best Trade", "Worst Trade", "Avg Win", "Avg Loss", "Alpha vs SPY"],
                    "Value": [
                        f"{bt.get('avg_return', 0):+.2f}%",
                        f"{median_ret:+.2f}%",
                        f"{bt.get('best_trade', 0):+.2f}%",
                        f"{bt.get('worst_trade', 0):+.2f}%",
                        f"{bt.get('avg_win', 0):+.2f}%",
                        f"{bt.get('avg_loss', 0):+.2f}%",
                        f"{alpha:+.2f}%",
                    ],
                })
                st.dataframe(stat_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                if trades:
                    st.markdown("---")
                    trades_df = pd.DataFrame(trades)
                    for dc in ["entry_date","exit_date"]:
                        if dc in trades_df.columns:
                            trades_df[dc] = pd.to_datetime(trades_df[dc], errors="coerce")
                    if "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
                        trades_df["holding_days"] = (trades_df["exit_date"] - trades_df["entry_date"]).dt.days.fillna(0).astype(int)

                    st.subheader(f"Trades ({len(trades_df)})")

                    if "return_pct" in trades_df.columns:
                        fig = go.Figure(go.Histogram(
                            x=trades_df["return_pct"], nbinsx=30,
                            marker_color=[C_GREEN if v > 0 else C_RED for v in trades_df["return_pct"]],
                        ))
                        fig.update_layout(**PLOTLY_BASE, height=250,
                                          xaxis_title="Return %", yaxis_title="Count",
                                          title="Return Distribution")
                        st.plotly_chart(fig, use_container_width=True)

                    show_cols = [c for c in ["ticker","entry_date","exit_date","holding_days",
                                             "entry_price","exit_price","return_pct","exit_reason"]
                                 if c in trades_df.columns]
                    disp_trades = trades_df[show_cols].copy()
                    if "return_pct" in disp_trades.columns:
                        disp_trades = disp_trades.sort_values("return_pct", ascending=False)

                    def _ret_color(val):
                        try:
                            return f"color: {C_GREEN}" if float(val) > 0 else f"color: {C_RED}"
                        except Exception:
                            return ""

                    styled_trades = disp_trades.style
                    if "return_pct" in disp_trades.columns:
                        styled_trades = styled_trades.applymap(_ret_color, subset=["return_pct"])
                    st.dataframe(styled_trades, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Backtest tab error: {e}")

    # ── Tab 5: About ──────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
## CAN SLIM + SEPA/VCP Methodology

**CAN SLIM** (William O'Neil / IBD) ranks growth stocks by combining fundamental quality with technical setup strength.

**SEPA/VCP** (Mark Minervini) identifies Volatility Contraction Patterns — price consolidations showing decreasing depth and volume before potential breakouts.

### Trend Template (8 criteria, all must pass)
1. Price > 200-day SMA
2. 200-day SMA trending up (past 20 bars)
3. 150-day SMA > 200-day SMA
4. Price > 150-day SMA
5. Price > 50-day SMA
6. 50-day SMA > 150-day SMA
7. Price within 25% of 52-week high
8. Price > 30% above 52-week low

### Pattern Types
| Pattern | Quality | What to look for |
|---------|---------|-----------------|
| VCP | ★★★★★ | Shrinking price swings; volume dry-up |
| Cup with Handle | ★★★★★ | U-shape 12–33% deep; handle = shallow pullback |
| Double Bottom | ★★★★ | W-shape; pivot = middle peak |
| Flat Base | ★★★ | < 15% depth; quiet institutional accumulation |

### Composite Score Weights
- Technical (30%): RS Rating, trend template, 52W proximity, volume
- Fundamental (30%): EPS growth, revenue growth, ROE, institutional ownership
- Pattern (25%): Pattern type, base depth, volume dry-up
- Market Context (15%): Market regime, sector rank, breadth

### Tier System
| Tier | Score | Action |
|------|-------|--------|
| A+ | 90–100 | Best setup — watch for entry trigger |
| A | 80–89 | Strong — buy on volume surge |
| B | 65–79 | Monitor — not yet actionable |
| C | 50–64 | Early — wait |
| D | < 50 | Pass |
""")


# ══════════════════════════════════════════════════════════════════════════════
# CRYPTO SECTION
# ══════════════════════════════════════════════════════════════════════════════

elif section == "🪙 Crypto":
    st.title("🪙 Crypto Analyzer")

    try:
        cr_df = load_crypto()
    except Exception as e:
        st.error(f"Failed to load crypto data: {e}")
        cr_df = pd.DataFrame()

    def _clean_sym(s):
        return str(s).replace("/USDT","").replace("/BTC","").strip()

    # Filter to latest run_date
    cr_latest = pd.DataFrame()
    if not cr_df.empty and "run_date" in cr_df.columns:
        latest_rd = cr_df["run_date"].max()
        cr_latest = cr_df[cr_df["run_date"] == latest_rd].copy()
        if "Symbol" in cr_latest.columns:
            cr_latest["Symbol_clean"] = cr_latest["Symbol"].apply(_clean_sym)
        if "Rank" not in cr_latest.columns and "Composite" in cr_latest.columns:
            cr_latest = cr_latest.sort_values("Composite", ascending=False).reset_index(drop=True)
            cr_latest["Rank"] = range(1, len(cr_latest)+1)

    tabs = st.tabs([
        "🏆 Rankings",
        "📡 RRG Chart",
        "🌡️ Heatmap",
        "📈 History",
        "🔍 Asset Detail",
        "📖 About",
    ])

    # ── Tab 1: Rankings ───────────────────────────────────────────────────
    with tabs[0]:
        try:
            if cr_latest.empty:
                st.warning("No crypto data at data/crypto/history.parquet")
            else:
                latest_rd_str = str(latest_rd)[:10] if not cr_df.empty else "—"
                st.caption(f"Run date: **{latest_rd_str}** · {len(cr_latest)} assets")

                # Header metrics
                if len(cr_latest):
                    top = cr_latest.sort_values("Composite", ascending=False).iloc[0] if "Composite" in cr_latest.columns else cr_latest.iloc[0]
                    top_sym = _clean_sym(top.get("Symbol", "?"))
                    leaders = cr_latest[cr_latest["RRG Regime"].str.lower() == "leading"]["Symbol"].apply(_clean_sym).tolist() if "RRG Regime" in cr_latest.columns else []
                    h1, h2, h3 = st.columns(3)
                    h1.metric("Top Composite", f"{top_sym} — {top.get('Composite', 0):.1f}")
                    h2.metric("Leading",  f"{len(leaders)} assets")
                    h3.metric("Total",    f"{len(cr_latest)} assets")

                st.markdown("---")

                st.info("💡 **Multi-horizon view**: Short-term (2W), Mid-term (12W), Long-term (52W) signals require additional pipeline runs — coming in next update")

                disp_cols = [c for c in ["Rank","Symbol","Composite","Tier","RS Score","Tech Score",
                                          "Fund Score","RRG Regime","RS %ile","RS Momentum",
                                          "RS Breakout","Golden ✕","Death ✕"]
                             if c in cr_latest.columns]
                disp = cr_latest[disp_cols].copy()
                if "Symbol" in disp.columns:
                    disp["Symbol"] = disp["Symbol"].apply(_clean_sym)
                if "RRG Regime" in disp.columns:
                    disp["RRG Regime"] = disp["RRG Regime"].apply(lambda r: RRG_BADGE.get(str(r).lower(), r))

                def _tier_rrg_color(val):
                    col = RRG_COLOR.get(str(val).lower().split()[-1].strip("🟢🔵🟡🔴"), "")
                    return f"color: {col}; font-weight: bold" if col else ""

                styled = disp.style
                for fmt_col in ["Composite","RS Score","Tech Score","Fund Score","RS %ile"]:
                    if fmt_col in disp.columns:
                        styled = styled.format({fmt_col: "{:.1f}"}, na_rep="—")
                st.dataframe(styled, use_container_width=True, hide_index=True,
                             height=min(700, 40+36*len(disp)))
        except Exception as e:
            st.error(f"Rankings tab error: {e}")

    # ── Tab 2: RRG Chart ──────────────────────────────────────────────────
    with tabs[1]:
        try:
            if cr_latest.empty:
                st.info("No data.")
            elif "RS Momentum" not in cr_latest.columns or "RS Score" not in cr_latest.columns:
                st.info("RS Momentum or RS Score columns not found.")
            else:
                plot_df = cr_latest.copy()
                plot_df["x"] = pd.to_numeric(plot_df["RS Score"], errors="coerce")
                plot_df["y"] = pd.to_numeric(plot_df["RS Momentum"], errors="coerce") - 50
                plot_df["sym_clean"] = plot_df["Symbol"].apply(_clean_sym)
                plot_df["regime"] = plot_df["RRG Regime"].str.lower() if "RRG Regime" in plot_df.columns else "lagging"

                fig = go.Figure()

                # Quadrant shading
                for x0,x1,y0,y1,col in [
                    (50,100,0,55,"rgba(34,197,94,0.07)"),
                    (0,50,0,55,"rgba(59,130,246,0.07)"),
                    (50,100,-55,0,"rgba(234,179,8,0.07)"),
                    (0,50,-55,0,"rgba(239,68,68,0.07)"),
                ]:
                    fig.add_shape(type="rect", x0=x0,x1=x1,y0=y0,y1=y1, fillcolor=col, line_width=0)

                for lbl,x,y in [("🟢 LEADING",75,45),("🔵 IMPROVING",25,45),("🟡 WEAKENING",75,-45),("🔴 LAGGING",25,-45)]:
                    fig.add_annotation(x=x,y=y,text=lbl,showarrow=False, font=dict(size=11,color="rgba(200,200,200,0.4)"))

                fig.add_shape(type="line",x0=50,x1=50,y0=-55,y1=55,line=dict(color="rgba(255,255,255,0.2)",width=1,dash="dash"))
                fig.add_shape(type="line",x0=0,x1=100,y0=0,y1=0,line=dict(color="rgba(255,255,255,0.2)",width=1,dash="dash"))

                for regime_key in ["leading","improving","weakening","lagging"]:
                    subset = plot_df[plot_df["regime"] == regime_key].dropna(subset=["x","y"])
                    if subset.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=subset["x"], y=subset["y"],
                        mode="markers+text",
                        name=RRG_BADGE.get(regime_key, regime_key),
                        text=subset["sym_clean"],
                        textposition="top center",
                        textfont=dict(size=10),
                        marker=dict(
                            size=14, color=RRG_COLOR[regime_key], opacity=0.85,
                            line=dict(width=1,color="rgba(0,0,0,0.3)"),
                        ),
                        hovertemplate="<b>%{text}</b><br>RS Score: %{x:.1f}<br>RS Momentum offset: %{y:.1f}<extra></extra>",
                    ))

                fig.update_layout(
                    **PLOTLY_BASE, height=560,
                    xaxis=dict(title="RS Score", range=[0,100]),
                    yaxis=dict(title="RS Momentum (offset from 50)", range=[-55,55]),
                )
                st.plotly_chart(fig, use_container_width=True)

                r1, r2 = st.columns(2)
                with r1:
                    st.success("🟢 **LEADING** — Outperforming BTC, gaining momentum. Strongest buy signal.")
                    st.info("🔵 **IMPROVING** — Starting to outperform. Early entry opportunity.")
                with r2:
                    st.warning("🟡 **WEAKENING** — Was leading, losing momentum. Reduce exposure.")
                    st.error("🔴 **LAGGING** — Underperforming BTC. Avoid.")
        except Exception as e:
            st.error(f"RRG tab error: {e}")

    # ── Tab 3: Heatmap ────────────────────────────────────────────────────
    with tabs[2]:
        try:
            if cr_latest.empty:
                st.info("No data.")
            else:
                score_cols = [c for c in ["RS Score","Tech Score","Fund Score","Composite"] if c in cr_latest.columns]
                if not score_cols:
                    st.info("No score columns found (expected: RS Score, Tech Score, Fund Score, Composite).")
                else:
                    hm_df = cr_latest.copy()
                    if "Symbol" in hm_df.columns:
                        hm_df = hm_df.sort_values("Composite", ascending=True) if "Composite" in hm_df.columns else hm_df
                        hm_df["sym_clean"] = hm_df["Symbol"].apply(_clean_sym)
                    syms   = hm_df["sym_clean"].tolist() if "sym_clean" in hm_df.columns else hm_df.index.tolist()

                    # Force all score columns to numeric
                    for sc in score_cols:
                        hm_df[sc] = pd.to_numeric(hm_df[sc], errors="coerce")

                    z_data = hm_df[score_cols].values.astype(float)

                    fig = go.Figure(go.Heatmap(
                        z=z_data,
                        x=score_cols,
                        y=syms,
                        colorscale="RdYlGn",
                        zmin=0, zmax=100,
                        text=[[f"{v:.0f}" if not np.isnan(v) else "" for v in row] for row in z_data],
                        texttemplate="%{text}",
                        hovertemplate="Symbol: %{y}<br>Metric: %{x}<br>Score: %{z:.1f}<extra></extra>",
                    ))
                    fig.update_layout(**PLOTLY_BASE, height=max(350,22*len(syms)),
                                      margin=dict(l=80,r=20,t=40,b=60))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Heatmap tab error: {e}")

    # ── Tab 4: History ────────────────────────────────────────────────────
    with tabs[3]:
        try:
            if cr_df.empty:
                st.info("No data.")
            else:
                run_dates = sorted(cr_df["run_date"].dropna().unique()) if "run_date" in cr_df.columns else []
                if len(run_dates) < 2:
                    n_dates = len(run_dates)
                    d = run_dates[0] if run_dates else "?"
                    if n_dates <= 3:
                        st.info(f"{n_dates} daily snapshot(s) available — growing daily as analyzer runs. Showing full table.")
                    else:
                        st.info(f"Only {n_dates} run date(s) ({str(d)[:10]}). Run analyzer again to build trend history.")
                    st.dataframe(cr_latest, use_container_width=True, height=500)
                else:
                    hist = cr_df.copy()
                    if "Symbol" in hist.columns:
                        hist["sym_clean"] = hist["Symbol"].apply(_clean_sym)

                    # Show note when history is short (just started tracking)
                    if len(run_dates) <= 5:
                        st.info(f"📅 {len(run_dates)} daily snapshots ({str(run_dates[0])[:10]} → {str(run_dates[-1])[:10]}) — growing daily as analyzer runs. Score changes reflect composite score evolution over time.")

                    # Date range selector
                    min_d = pd.to_datetime(min(run_dates))
                    max_d = pd.to_datetime(max(run_dates))
                    start_d = st.date_input(
                        "From", value=min_d.date(),
                        min_value=min_d.date(), max_value=max_d.date(),
                        key="crypto_hist_start"
                    )
                    filtered_hist = hist[pd.to_datetime(hist["run_date"]) >= pd.Timestamp(start_d)]

                    # Top 10 by latest composite
                    top10 = cr_latest.sort_values("Composite", ascending=False)["sym_clean"].head(10).tolist() if "sym_clean" in cr_latest.columns else []
                    sel = st.multiselect("Filter assets", sorted(hist["sym_clean"].unique()) if "sym_clean" in hist.columns else [], default=top10)

                    plot_hist = filtered_hist[filtered_hist["sym_clean"].isin(sel)] if sel and "sym_clean" in filtered_hist.columns else filtered_hist
                    if "Composite" in plot_hist.columns:
                        fig = px.line(
                            plot_hist.sort_values("run_date"),
                            x="run_date", y="Composite", color="sym_clean",
                            markers=True,
                            labels={"run_date":"Date","Composite":"Composite Score","sym_clean":"Symbol"},
                        )
                        fig.update_layout(**PLOTLY_BASE, height=500, yaxis=dict(range=[0,100]))
                        st.plotly_chart(fig, use_container_width=True)

                    # Trend direction arrows: compare first vs last run for each symbol
                    if "sym_clean" in filtered_hist.columns and "Composite" in filtered_hist.columns:
                        st.markdown("#### Trend Direction (first → last in selected range)")
                        trend_rows = []
                        for sym_val, grp in filtered_hist.groupby("sym_clean"):
                            grp_sorted = grp.sort_values("run_date")
                            if len(grp_sorted) >= 2:
                                first_score = float(grp_sorted.iloc[0]["Composite"]) if pd.notna(grp_sorted.iloc[0]["Composite"]) else 0
                                last_score  = float(grp_sorted.iloc[-1]["Composite"]) if pd.notna(grp_sorted.iloc[-1]["Composite"]) else 0
                                delta = last_score - first_score
                                if delta > 2:
                                    arrow = "↑"
                                    arrow_color = C_GREEN
                                elif delta < -2:
                                    arrow = "↓"
                                    arrow_color = C_RED
                                else:
                                    arrow = "→"
                                    arrow_color = C_GREY
                                trend_rows.append({
                                    "Symbol": sym_val,
                                    "First": round(first_score, 1),
                                    "Last": round(last_score, 1),
                                    "Change": f"{delta:+.1f}",
                                    "Trend": arrow,
                                })
                        if trend_rows:
                            trend_df = pd.DataFrame(trend_rows).sort_values("Last", ascending=False)
                            st.dataframe(trend_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"History tab error: {e}")

    # ── Tab 5: Asset Detail ───────────────────────────────────────────────
    with tabs[4]:
        try:
            if cr_latest.empty:
                st.info("No data.")
            else:
                syms = [_clean_sym(s) for s in cr_latest["Symbol"].tolist()] if "Symbol" in cr_latest.columns else []
                sel_sym = st.selectbox("Select asset", syms)

                row = cr_latest[cr_latest["Symbol"].apply(_clean_sym) == sel_sym]
                if row.empty:
                    st.warning("Not found.")
                else:
                    r = row.iloc[0]
                    regime = str(r.get("RRG Regime","lagging")).lower()
                    comp   = float(r.get("Composite",0) or 0)

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("RS Score",  f"{r.get('RS Score',0):.1f}")
                        st.markdown(RRG_BADGE.get(regime, regime))
                        st.metric("RS %ile",   f"{r.get('RS %ile',0):.1f}%")
                    with m2:
                        st.metric("Tech Score", f"{r.get('Tech Score',0):.1f}")
                        if r.get("Golden ✕") == "✓":
                            st.success("Golden Cross ✓")
                        if r.get("Death ✕") == "✓":
                            st.error("Death Cross ✗")
                    with m3:
                        st.metric("Fund Score", f"{r.get('Fund Score',0):.1f}")
                        st.metric("Composite",  f"{comp:.1f}")

                    st.markdown("---")
                    score_labels = ["RS Score","Tech Score","Fund Score","Composite"]
                    score_vals   = [float(r.get(c,0) or 0) for c in score_labels]
                    fig = go.Figure(go.Bar(
                        x=score_labels, y=score_vals,
                        marker_color=[C_BLUE,"#8b5cf6",C_GREEN,C_YELLOW],
                        text=[f"{v:.1f}" for v in score_vals], textposition="outside",
                    ))
                    fig.update_layout(**PLOTLY_BASE, height=300, showlegend=False,
                                      yaxis=dict(range=[0,115]))
                    st.plotly_chart(fig, use_container_width=True)

                    # Historical trend for asset
                    if not cr_df.empty and "Symbol" in cr_df.columns:
                        asset_hist = cr_df[cr_df["Symbol"].apply(_clean_sym) == sel_sym].sort_values("run_date")
                        if len(asset_hist) >= 2 and "Composite" in asset_hist.columns:
                            st.markdown("#### Historical Score Trend")
                            fig2 = px.line(asset_hist, x="run_date", y="Composite", markers=True)
                            fig2.update_layout(**PLOTLY_BASE, height=300, yaxis=dict(range=[0,100]))
                            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Asset detail tab error: {e}")

    # ── Tab 6: About ──────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("""
## Crypto Analyzer Methodology

A systematic ranking of top cryptocurrencies by market cap using three evidence-based pillars.

### Three Pillars
| Pillar | Weight | What We Measure |
|--------|--------|----------------|
| RS vs BTC | 30% | Altcoin/BTC ratio, RS trend, momentum, percentile |
| Technical Health | 25% | Price vs 50/200 SMA, golden/death cross, volume |
| Fundamentals | 45% | TVL growth, protocol fees, tokenomics, utility |

### RRG Chart Guide
| Quadrant | Meaning | Action |
|----------|---------|--------|
| 🟢 LEADING | High RS + positive momentum | Buy/Hold |
| 🔵 IMPROVING | Low RS + rising momentum | Watch/Build |
| 🟡 WEAKENING | High RS + slowing momentum | Reduce/Stop |
| 🔴 LAGGING | Low RS + falling momentum | Avoid |

**Rotation path**: Coins typically rotate clockwise. Best entry: LAGGING → IMPROVING.

### Score Guide
| Score | Grade | Meaning |
|-------|-------|---------|
| 80–100 | A | Strongest conviction |
| 65–79 | B | Good across most pillars |
| 50–64 | C | Mixed signals |
| < 50 | D/F | Avoid |

Data sources: Binance → Kraken → OKX (CCXT), CoinGecko, DeFiLlama.
""")


# ══════════════════════════════════════════════════════════════════════════════
# S&P BREADTH SECTION
# ══════════════════════════════════════════════════════════════════════════════

elif section == "📊 S&P Breadth":
    st.title("📊 S&P 500 Market Breadth")

    try:
        idx_df = load_breadth_index()
        sec_df = load_breadth_sector()
        ath_df = load_breadth_ath()
    except Exception as e:
        st.error(f"Failed to load breadth data: {e}")
        idx_df = pd.DataFrame()
        sec_df = pd.DataFrame()
        ath_df = pd.DataFrame()

    def _get_regime(pct200: float) -> tuple[str,str]:
        if pct200 > 70:  return "🟢 BULL — Broad participation", C_GREEN
        if pct200 > 50:  return "🟡 MIXED — Selective market", C_YELLOW
        if pct200 > 30:  return "🟠 CAUTION — Weak breadth", "#f97316"
        return "🔴 BEAR — Broad deterioration", C_RED

    tabs = st.tabs([
        "📊 Dashboard",
        "🌡️ Heatmaps",
        "📈 Time Series",
        "🏭 Sectors",
        "📖 About",
    ])

    # ── Tab 1: Dashboard ──────────────────────────────────────────────────
    with tabs[0]:
        try:
            if idx_df.empty:
                st.warning("No breadth data at data/breadth/breadth_index_weekly_with_wow.csv")
            else:
                latest_row = idx_df.iloc[-1]
                pct200 = float(latest_row.get("above_200sma", 0))
                pct50  = float(latest_row.get("above_50sma", 0))
                h52    = float(latest_row.get("new_52w_high", 0))
                l52    = float(latest_row.get("new_52w_low", 0))
                wow200 = float(latest_row.get("above_200sma_wow", 0) or 0)
                wow50  = float(latest_row.get("above_50sma_wow", 0) or 0)
                wowh   = float(latest_row.get("new_52w_high_wow", 0) or 0)
                wowl   = float(latest_row.get("new_52w_low_wow", 0) or 0)
                regime_label, regime_col = _get_regime(pct200)

                st.markdown(
                    f"<div style='background:{regime_col}22;border:2px solid {regime_col}66;"
                    f"border-radius:10px;padding:12px 20px;display:inline-block;margin-bottom:16px;'>"
                    f"<span style='color:{regime_col};font-size:1.3em;font-weight:800;'>{regime_label}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ATH proximity (index-wide)
                ath_idx_pct = None
                ath_date    = None
                if not ath_df.empty:
                    idx_row_ath = ath_df[ath_df["sector"] == "S&P 500 (Index)"]
                    if not idx_row_ath.empty:
                        ath_idx_pct = float(idx_row_ath.iloc[0]["near_ath_pct"])
                        ath_date    = idx_row_ath.iloc[0].get("date", "")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("% >200 SMA",   f"{pct200:.1f}%",       delta=f"{wow200:+.1f}pp WoW")
                m2.metric("% >50 SMA",    f"{pct50:.1f}%",        delta=f"{wow50:+.1f}pp WoW")
                m3.metric("52W Highs",    f"{h52:.0f}",           delta=f"{wowh:+.0f} WoW")
                m4.metric("52W Lows",     f"{l52:.0f}",           delta=f"{wowl:+.0f} WoW", delta_color="inverse")
                if ath_idx_pct is not None:
                    m5.metric("Near ATH (±3%)", f"{ath_idx_pct:.1f}%",
                              delta=f"as of {ath_date}" if ath_date else None)
                else:
                    m5.metric("Near ATH (±3%)", "—")

                # Sector table
                if not sec_df.empty:
                    st.markdown("---")
                    st.subheader("Sector Breadth")
                    latest_sec = sec_df.groupby("sector").last().reset_index()

                    # Compute MoM (4-week) if enough data
                    sec_4w = None
                    if len(sec_df["date"].unique()) >= 4:
                        cutoff_date = idx_df["date"].max() - pd.Timedelta(weeks=4)
                        old_sec = sec_df[sec_df["date"] <= cutoff_date].groupby("sector").last().reset_index()
                        sec_4w = old_sec.set_index("sector")["above_200sma"].rename("pct200_4w")

                    # Build ATH lookup: sector → near_ath_pct
                    ath_lookup = {}
                    if not ath_df.empty and "sector" in ath_df.columns:
                        for _, ar in ath_df.iterrows():
                            ath_lookup[ar["sector"]] = ar.get("near_ath_pct", None)

                    sector_rows = []
                    for _, sr in latest_sec.iterrows():
                        sname = sr["sector"]
                        etf   = SECTOR_ETF.get(sname, "")
                        p200  = float(sr.get("above_200sma", 0) or 0)
                        p50   = float(sr.get("above_50sma",  0) or 0)
                        h_pct = float(sr.get("new_52w_high", 0) or 0)

                        # WoW
                        sec_hist = sec_df[sec_df["sector"] == sname].sort_values("date")
                        wow_200  = float(sr.get("above_200sma_wow", 0) or 0) if "above_200sma_wow" in sr.index else (
                            p200 - float(sec_hist.iloc[-2]["above_200sma"]) if len(sec_hist) >= 2 else 0
                        )
                        # MoM
                        mom_200 = None
                        if sec_4w is not None and sname in sec_4w.index:
                            mom_200 = p200 - float(sec_4w[sname])

                        # Trend arrow
                        if wow_200 > 3:   trend = "↑↑"
                        elif wow_200 > 0: trend = "↑"
                        elif wow_200 == 0: trend = "→"
                        elif wow_200 > -3: trend = "↓"
                        else:             trend = "↓↓"

                        # ATH proximity
                        near_ath = ath_lookup.get(sname)
                        near_ath_str = f"{near_ath:.1f}%" if near_ath is not None else "—"

                        sector_rows.append({
                            "Sector": f"{sname} ({etf})" if etf else sname,
                            "%>200SMA":   f"{p200:.1f}%",
                            "%>50SMA":    f"{p50:.1f}%",
                            "WoW pp":     f"{wow_200:+.1f}",
                            "MoM pp":     f"{mom_200:+.1f}" if mom_200 is not None else "—",
                            "52W Highs%": f"{h_pct:.1f}%",
                            "Near ATH":   near_ath_str,
                            "Trend":      trend,
                        })

                    sec_tbl = pd.DataFrame(sector_rows).sort_values(
                        "%>200SMA", ascending=False,
                        key=lambda x: x.str.rstrip("%").astype(float),
                        ignore_index=True
                    )
                    st.dataframe(sec_tbl, use_container_width=True, hide_index=True)

                    # Leading / Lagging tiles
                    st.markdown("---")
                    st.subheader("Leading vs Lagging Sectors")
                    raw_sec = latest_sec.copy()
                    raw_sec["pct200"] = pd.to_numeric(raw_sec["above_200sma"], errors="coerce").fillna(0)
                    raw_sec = raw_sec.sort_values("pct200", ascending=False)
                    top3    = raw_sec.head(3)
                    bot3    = raw_sec.tail(3)

                    lead_cols = st.columns(3)
                    for i, (_, sr) in enumerate(top3.iterrows()):
                        with lead_cols[i]:
                            st.markdown(
                                f"<div style='background:{C_GREEN}22;border:1px solid {C_GREEN}55;"
                                f"border-radius:8px;padding:10px;text-align:center;'>"
                                f"<div style='color:{C_GREEN};font-weight:700;font-size:0.85rem;'>{sr['sector']}</div>"
                                f"<div style='color:{C_GREEN};font-size:1.4em;font-weight:800;'>{sr['pct200']:.1f}%</div>"
                                f"<div style='color:{C_GREY};font-size:0.75rem;'>above 200 SMA</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    lag_cols = st.columns(3)
                    for i, (_, sr) in enumerate(bot3.iterrows()):
                        with lag_cols[i]:
                            st.markdown(
                                f"<div style='background:{C_RED}22;border:1px solid {C_RED}55;"
                                f"border-radius:8px;padding:10px;text-align:center;'>"
                                f"<div style='color:{C_RED};font-weight:700;font-size:0.85rem;'>{sr['sector']}</div>"
                                f"<div style='color:{C_RED};font-size:1.4em;font-weight:800;'>{sr['pct200']:.1f}%</div>"
                                f"<div style='color:{C_GREY};font-size:0.75rem;'>above 200 SMA</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
        except Exception as e:
            st.error(f"Dashboard tab error: {e}")

    # ── Tab 2: Heatmaps ───────────────────────────────────────────────────
    with tabs[1]:
        try:
            if sec_df.empty:
                st.info("No sector data.")
            else:
                try:
                    latest_sec = sec_df.groupby("sector").last().reset_index()
                    metrics = [c for c in ["above_50sma","above_200sma","new_52w_high","new_52w_low"] if c in latest_sec.columns]
                    if not metrics:
                        st.info("No metric columns found in breadth_sector_weekly.csv.")
                    else:
                        labels = {"above_50sma":">50SMA","above_200sma":">200SMA","new_52w_high":"52W High%","new_52w_low":"52W Low%"}
                        # ETF labels for y-axis
                        sec_labels = [f"{s} ({SECTOR_ETF.get(s, '')})" if SECTOR_ETF.get(s) else s
                                      for s in latest_sec["sector"].tolist()]
                        x_labels = [labels.get(m, m) for m in metrics]

                        # Force numeric
                        for m in metrics:
                            latest_sec[m] = pd.to_numeric(latest_sec[m], errors="coerce")
                        z_data = latest_sec[metrics].values.astype(float)

                        fig = go.Figure(go.Heatmap(
                            z=z_data, x=x_labels, y=sec_labels,
                            colorscale="RdYlGn",
                            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z_data],
                            texttemplate="%{text}",
                            hovertemplate="Sector: %{y}<br>Metric: %{x}<br>Value: %{z:.1f}%<extra></extra>",
                        ))
                        fig.update_layout(**PLOTLY_BASE, height=max(450, 30*len(sec_labels)),
                                          margin=dict(l=200,r=20,t=40,b=60))
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e2:
                    st.error(f"Breadth heatmap error: {e2}")
        except Exception as e:
            st.error(f"Heatmap tab error: {e}")

    # ── Tab 3: Time Series ────────────────────────────────────────────────
    with tabs[2]:
        try:
            if idx_df.empty:
                st.info("No breadth index data.")
            else:
                # Start date selector
                ts_min_date = idx_df["date"].min().date()
                ts_max_date = idx_df["date"].max().date()
                ts_default  = max(ts_min_date, pd.Timestamp("2022-01-01").date())
                start_date  = st.date_input(
                    "Start date", value=ts_default,
                    min_value=ts_min_date, max_value=ts_max_date,
                    key="ts_start"
                )

                # Toggle checkboxes
                cb1, cb2, cb3 = st.columns(3)
                with cb1:
                    show_50  = st.checkbox("% >50 SMA",  value=True, key="ts_50")
                with cb2:
                    show_200 = st.checkbox("% >200 SMA", value=True, key="ts_200")
                with cb3:
                    show_ath = st.checkbox("Near ATH (±3%)", value=True, key="ts_ath")

                # Filter by start_date
                idx_filtered = idx_df[idx_df["date"] >= pd.Timestamp(start_date)].copy()

                metrics_ts = []
                if show_50 and "above_50sma" in idx_filtered.columns:
                    metrics_ts.append("above_50sma")
                if show_200 and "above_200sma" in idx_filtered.columns:
                    metrics_ts.append("above_200sma")

                # Compute period returns for legend labels
                def _period_ret(df, col):
                    vals = df[col].dropna()
                    if len(vals) >= 2:
                        return float(vals.iloc[-1]) - float(vals.iloc[0])
                    return 0.0

                if metrics_ts:
                    fig = go.Figure()
                    colors_ts = {
                        "above_50sma":  C_YELLOW,
                        "above_200sma": C_GREEN,
                    }
                    labels_ts = {"above_50sma":"% >50 SMA","above_200sma":"% >200 SMA"}
                    for m in metrics_ts:
                        ret_pp = _period_ret(idx_filtered, m)
                        label_with_ret = f"{labels_ts.get(m,m)} ({ret_pp:+.1f}pp since {start_date})"
                        fig.add_trace(go.Scatter(
                            x=idx_filtered["date"], y=idx_filtered[m],
                            mode="lines", name=label_with_ret,
                            line=dict(color=colors_ts.get(m,C_BLUE)),
                            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
                        ))

                    # Near ATH overlay (from ath_df if available)
                    if show_ath and not ath_df.empty:
                        ath_idx_row = ath_df[ath_df["sector"] == "S&P 500 (Index)"]
                        if not ath_idx_row.empty:
                            ath_idx_pct_val = float(ath_idx_row.iloc[0]["near_ath_pct"])
                            ath_date_val    = str(ath_idx_row.iloc[0].get("date", ""))
                            st.caption(f"Near ATH: {ath_idx_pct_val:.1f}% as of {ath_date_val} — time series tracking begins today")

                    fig.add_hline(y=70, line_dash="dot", line_color=C_GREEN, opacity=0.5, annotation_text="Bull threshold")
                    fig.add_hline(y=50, line_dash="dot", line_color=C_YELLOW, opacity=0.5, annotation_text="Neutral")
                    fig.add_hline(y=30, line_dash="dot", line_color=C_RED, opacity=0.5, annotation_text="Bear threshold")
                    rs_breadth = dict(buttons=[
                        dict(count=1,  label="1M",  step="month", stepmode="backward"),
                        dict(count=3,  label="3M",  step="month", stepmode="backward"),
                        dict(count=6,  label="6M",  step="month", stepmode="backward"),
                        dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                        dict(step="all", label="All"),
                    ])
                    fig.update_layout(
                        **PLOTLY_BASE, height=450,
                        xaxis=dict(
                            rangeselector=rs_breadth, rangeslider=dict(visible=False),
                            range=[str(start_date), idx_df["date"].max().strftime("%Y-%m-%d")],
                        ),
                        yaxis=dict(title="% of S&P 500", range=[0,105]),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SMA columns found or all toggles disabled.")

                # Sector Breadth — % above 200 SMA sub-section
                if not sec_df.empty and "above_200sma" in sec_df.columns:
                    st.markdown("---")
                    st.subheader("Sector Breadth — % above 200 SMA")
                    try:
                        # Filter sector data by start_date
                        sec_filtered = sec_df[sec_df["date"] >= pd.Timestamp(start_date)].copy()
                        sec_pivot = sec_filtered.pivot_table(index="date", columns="sector", values="above_200sma", aggfunc="last")
                        sec_pivot = sec_pivot.sort_index()
                        fig2 = go.Figure()
                        for col in sec_pivot.columns:
                            col_vals = sec_pivot[col].dropna()
                            if len(col_vals) >= 2:
                                sec_ret_pp = float(col_vals.iloc[-1]) - float(col_vals.iloc[0])
                                sec_label = f"{col} ({sec_ret_pp:+.1f}pp)"
                            else:
                                sec_label = col
                            fig2.add_trace(go.Scatter(
                                x=sec_pivot.index, y=sec_pivot[col],
                                mode="lines", name=sec_label,
                                hovertemplate=f"{col}<br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
                            ))
                        fig2.update_layout(
                            **PLOTLY_BASE, height=450,
                            xaxis=dict(
                                rangeselector=rs_breadth, rangeslider=dict(visible=False),
                                range=[str(start_date), sec_df["date"].max().strftime("%Y-%m-%d")],
                            ),
                            yaxis=dict(title="% above 200 SMA", range=[0,105]),
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e3:
                        st.error(f"Sector time series error: {e3}")
        except Exception as e:
            st.error(f"Time series tab error: {e}")

    # ── Tab 4: Sectors ────────────────────────────────────────────────────
    with tabs[3]:
        try:
            if sec_df.empty:
                st.info("No sector data.")
            else:
                # Per-sector metrics table
                st.subheader("Current Sector Metrics")
                try:
                    latest_sec2 = sec_df.groupby("sector").last().reset_index()
                    sec_metric_rows = []
                    for _, sr in latest_sec2.iterrows():
                        sname = sr["sector"]
                        etf   = SECTOR_ETF.get(sname, "")
                        p200  = float(pd.to_numeric(sr.get("above_200sma", 0), errors="coerce") or 0)
                        p50   = float(pd.to_numeric(sr.get("above_50sma",  0), errors="coerce") or 0)

                        sec_hist2 = sec_df[sec_df["sector"] == sname].sort_values("date")
                        if len(sec_hist2) >= 2:
                            prev = sec_hist2.iloc[-2]
                            wow200 = p200 - float(pd.to_numeric(prev.get("above_200sma", p200), errors="coerce") or p200)
                            wow50  = p50  - float(pd.to_numeric(prev.get("above_50sma",  p50),  errors="coerce") or p50)
                        else:
                            wow200, wow50 = 0.0, 0.0

                        # ATH proximity from pre-computed data
                        ath_pct_val = None
                        if not ath_df.empty and "sector" in ath_df.columns:
                            ath_row = ath_df[ath_df["sector"] == sname]
                            if not ath_row.empty:
                                ath_pct_val = ath_row.iloc[0].get("near_ath_pct")
                        ath_str = f"{float(ath_pct_val):.1f}%" if ath_pct_val is not None else "—"

                        sec_metric_rows.append({
                            "Sector": f"{sname} ({etf})" if etf else sname,
                            "%>50SMA":    f"{p50:.1f}%",
                            "%>200SMA":   f"{p200:.1f}%",
                            "WoW 200":    f"{wow200:+.1f}pp",
                            "Near ATH":   ath_str,
                        })
                    st.dataframe(pd.DataFrame(sec_metric_rows), use_container_width=True, hide_index=True)
                except Exception as e2:
                    st.error(f"Sector metrics error: {e2}")

                st.markdown("---")
                all_sectors = sorted(sec_df["sector"].unique().tolist())
                sel_secs = st.multiselect("Sectors for chart", options=all_sectors, default=all_sectors[:8])
                plot_sec = sec_df[sec_df["sector"].isin(sel_secs)] if sel_secs else sec_df

                if "above_200sma" in plot_sec.columns:
                    fig = px.line(
                        plot_sec.sort_values("date"),
                        x="date", y="above_200sma", color="sector",
                        labels={"date":"Date","above_200sma":"% >200 SMA","sector":"Sector"},
                    )
                    fig.update_layout(
                        **PLOTLY_BASE, height=600,
                        xaxis=dict(rangeselector=_rangeselector(), rangeslider=dict(visible=False)),
                        yaxis=dict(title="% above 200 SMA", range=[0,105]),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Sectors tab error: {e}")

    # ── Tab 5: About ──────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
## S&P 500 Market Breadth Dashboard

### Market Regime Definitions
| Regime | % S&P 500 >200 SMA | Meaning |
|--------|---------------------|---------|
| 🟢 BULL | > 70% | Broad participation — most stocks in uptrend |
| 🟡 MIXED | 50–70% | Selective market — large-caps leading |
| 🟠 CAUTION | 30–50% | Weak breadth — risk rising |
| 🔴 BEAR | < 30% | Broad deterioration — avoid new longs |

### Breadth Indicators
- **% >200 SMA**: Long-term trend health. Key regime indicator.
- **% >50 SMA**: Short-term momentum participation.
- **52-Week Highs**: New leadership emerging.
- **52-Week Lows**: Distribution / selling pressure.
- **WoW**: Week-over-week change in percentage points.

### How to Use
1. **Bull regime** (>70%): Aggressive — pursue breakouts, add exposure
2. **Mixed regime** (50–70%): Selective — only highest-quality setups
3. **Caution** (30–50%): Defensive — reduce new positions, tighten stops
4. **Bear** (<30%): Avoid new longs. Focus on short setups or cash.

Data: Weekly computation from S&P 500 universe, updated each Monday.
""")


# ══════════════════════════════════════════════════════════════════════════════
# RISK FACTORS SECTION
# ══════════════════════════════════════════════════════════════════════════════

elif section == "⚠️ Risk Factors":
    st.title("⚠️ Market Risk Factors")

    try:
        risk_scores = load_risk_scores()
        risk_ts     = load_risk_timeseries()
    except Exception as e:
        st.error(f"Failed to load risk data: {e}")
        risk_scores = []
        risk_ts = pd.DataFrame()

    SIGNAL_COLORS_RISK = {"green": C_GREEN, "yellow": C_YELLOW, "red": C_RED, "na": C_GREY}
    SIGNAL_LABELS_RISK = {"green": "LOW RISK", "yellow": "ELEVATED", "red": "HIGH RISK", "na": "N/A"}
    SIGNAL_EMOJI_RISK  = {"green": "🟢", "yellow": "🟡", "red": "🔴", "na": "⚪"}

    tabs = st.tabs([
        "📊 Composite Trend",
        "🎯 Multi-Horizon Signals",
        "📋 Current Signals",
    ])

    # ── Tab 1: Composite Trend ────────────────────────────────────────────
    with tabs[0]:
        try:
            if risk_ts.empty:
                st.warning("No composite timeseries found. Tried: composite_timeseries.parquet and composite_timeseries.csv")
                st.info("Run: python run_hub_cache.py or python compute_composite.py to generate this file.")
            else:
                # KPI cards
                if "composite_20" in risk_ts.columns and "composite_50" in risk_ts.columns:
                    last_row = risk_ts.dropna(subset=["composite_20","composite_50"]).iloc[-1]
                    v20 = float(last_row["composite_20"])
                    v50 = float(last_row["composite_50"])

                    def _score_label_risk(v):
                        # Scale is 0–10: ≥7 = low risk, 4–7 = elevated, <4 = high risk
                        if v >= 7:   return f"🟢 Low Risk ({v:.1f}/10)"
                        if v >= 4:   return f"🟡 Elevated ({v:.1f}/10)"
                        return f"🔴 High Risk ({v:.1f}/10)"

                    m1, m2 = st.columns(2)
                    m1.metric("20-Day Composite", _score_label_risk(v20))
                    m2.metric("50-Day Composite", _score_label_risk(v50))

                    st.markdown("---")

                fig = go.Figure()
                if "composite_20" in risk_ts.columns:
                    fig.add_trace(go.Scatter(
                        x=risk_ts["date"], y=risk_ts["composite_20"],
                        mode="lines", name="20-Day MA",
                        line=dict(color=C_BLUE, width=2.5),
                        hovertemplate="%{x|%Y-%m-%d}<br>20d: %{y:.3f}<extra></extra>",
                    ))
                if "composite_50" in risk_ts.columns:
                    fig.add_trace(go.Scatter(
                        x=risk_ts["date"], y=risk_ts["composite_50"],
                        mode="lines", name="50-Day MA",
                        line=dict(color=C_RED, width=2.5),
                        hovertemplate="%{x|%Y-%m-%d}<br>50d: %{y:.3f}<extra></extra>",
                    ))

                fig.add_hline(y=5,   line_dash="dash", line_color=C_GREY, opacity=0.6,
                              annotation_text="Neutral (5.0)", annotation_font_color=C_GREY)
                fig.add_hline(y=7.0, line_dash="dot",  line_color=C_GREEN, opacity=0.5,
                              annotation_text="Low Risk (7+)", annotation_font_color=C_GREEN,
                              annotation_position="top right")
                fig.add_hline(y=4.0, line_dash="dot", line_color=C_RED, opacity=0.5,
                              annotation_text="High Risk (<4)", annotation_font_color=C_RED,
                              annotation_position="bottom right")

                # Range selector with longer history
                rs_risk = dict(buttons=[
                    dict(count=1, label="1Y", step="year",  stepmode="backward"),
                    dict(count=3, label="3Y", step="year",  stepmode="backward"),
                    dict(count=5, label="5Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ])
                fig.update_layout(
                    **PLOTLY_BASE, height=500,
                    xaxis=dict(rangeselector=rs_risk, rangeslider=dict(visible=False)),
                    yaxis=dict(title="Composite Risk Score (0–10, higher = lower risk)", range=[0, 10.5]),
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Composite trend tab error: {e}")

    # ── Tab 2: Multi-Horizon Signals ──────────────────────────────────────
    with tabs[1]:
        try:
            if not risk_scores:
                st.warning("No risk score data at data/risk_factors/latest_scores.json")
            else:
                st.markdown("**15 factors × 3 horizons** — current/1W · 50-day/1M · 3M derived")

                rows = []
                for s in risk_scores:
                    sig_curr = s.get("signal", "na")
                    sig_50   = s.get("signal_50", sig_curr)
                    # Derive 3M: if same as 50d signal → same, else → mixed (yellow)
                    sig_3m   = sig_curr if sig_curr == sig_50 else "yellow"

                    rows.append({
                        "Factor": s.get("factor_name", s.get("name", "—")),
                        "Value":  s.get("display_value", "—"),
                        "Current/1W": sig_curr,
                        "50-Day/1M":  sig_50,
                        "3M (derived)": sig_3m,
                    })

                CELL_BG = {"green":"#052e16","yellow":"#1c1400","red":"#1c0505","na":"#1a1f2e"}
                CELL_FG = {"green":C_GREEN,"yellow":C_YELLOW,"red":C_RED,"na":C_GREY}

                # Build pure HTML table (avoids go.Table rendering issues in Streamlit)
                html_rows = ""
                for r in rows:
                    def _td_signal(sig_key):
                        bg  = CELL_BG.get(sig_key, "#1a1f2e")
                        fg  = CELL_FG.get(sig_key, C_GREY)
                        lbl = f"{SIGNAL_EMOJI_RISK.get(sig_key,'⚪')} {SIGNAL_LABELS_RISK.get(sig_key,'N/A')}"
                        return (f"<td style='background:{bg};color:{fg};font-weight:700;"
                                f"text-align:center;padding:6px 10px;font-size:0.82rem;'>{lbl}</td>")

                    html_rows += (
                        f"<tr>"
                        f"<td style='color:#e2e8f0;padding:6px 10px;font-size:0.83rem;'>{r['Factor']}</td>"
                        f"<td style='color:#94a3b8;text-align:center;padding:6px 10px;font-size:0.82rem;'>{r['Value']}</td>"
                        f"{_td_signal(r['Current/1W'])}"
                        f"{_td_signal(r['50-Day/1M'])}"
                        f"{_td_signal(r['3M (derived)'])}"
                        f"</tr>"
                    )

                html_table = f"""
<div style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;background:{C_CARD};border-radius:8px;overflow:hidden;">
  <thead>
    <tr style="background:#1e3a5f;">
      <th style="color:white;text-align:left;padding:8px 10px;font-size:0.85rem;">Factor</th>
      <th style="color:white;text-align:center;padding:8px 10px;font-size:0.85rem;">Value</th>
      <th style="color:white;text-align:center;padding:8px 10px;font-size:0.85rem;">Current / 1W</th>
      <th style="color:white;text-align:center;padding:8px 10px;font-size:0.85rem;">50-Day / 1M</th>
      <th style="color:white;text-align:center;padding:8px 10px;font-size:0.85rem;">3M (derived)</th>
    </tr>
  </thead>
  <tbody>
    {html_rows}
  </tbody>
</table>
</div>
"""
                st.markdown(html_table, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Multi-horizon tab error: {e}")

    # ── Tab 3: Current Signals ────────────────────────────────────────────
    with tabs[2]:
        try:
            if not risk_scores:
                st.warning("No risk data.")
            else:
                # Composite KPI
                green_count = sum(1 for s in risk_scores if s.get("signal") == "green")
                yellow_count= sum(1 for s in risk_scores if s.get("signal") == "yellow")
                red_count   = sum(1 for s in risk_scores if s.get("signal") == "red")
                total       = len(risk_scores)

                overall = "green" if green_count >= total*0.6 else ("red" if red_count >= total*0.6 else "yellow")
                ov_col  = SIGNAL_COLORS_RISK[overall]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"🟢 Low Risk",  green_count)
                m2.metric(f"🟡 Elevated",  yellow_count)
                m3.metric(f"🔴 High Risk", red_count)
                m4.metric(f"Total Factors", total)

                st.markdown(
                    f"<div style='background:{ov_col}22;border:2px solid {ov_col}66;"
                    f"border-radius:10px;padding:12px 20px;margin:12px 0;'>"
                    f"<span style='color:{ov_col};font-size:1.2em;font-weight:800;'>"
                    f"{SIGNAL_EMOJI_RISK[overall]} Overall: {SIGNAL_LABELS_RISK[overall]}"
                    f" ({green_count}G / {yellow_count}Y / {red_count}R)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

                # Factor cards — 3 column grid, 2 horizons per card
                factor_cols = st.columns(3)
                for i, s in enumerate(risk_scores):
                    signal     = s.get("signal", "na")
                    signal_50  = s.get("signal_50", signal)
                    sig_col    = SIGNAL_COLORS_RISK.get(signal, C_GREY)
                    sig_col_50 = SIGNAL_COLORS_RISK.get(signal_50, C_GREY)
                    name    = s.get("factor_name", s.get("name", f"Factor {i+1}"))
                    desc    = s.get("description", "")
                    val     = s.get("display_value", "—")
                    text    = s.get("text", "")

                    with factor_cols[i % 3]:
                        st.markdown(
                            f"<div style='background:{C_CARD};border:1px solid {sig_col}44;"
                            f"border-left:4px solid {sig_col};border-radius:8px;"
                            f"padding:12px;margin-bottom:8px;'>"
                            f"<div style='font-weight:700;color:#e2e8f0;font-size:0.88rem;margin-bottom:4px;'>{name}</div>"
                            f"<div style='color:{C_GREY};font-size:0.72rem;margin-bottom:6px;'>{desc}</div>"
                            f"<div style='display:flex;gap:8px;'>"
                            f"<div style='flex:1;background:{sig_col}11;border:1px solid {sig_col}33;border-radius:4px;padding:4px 6px;'>"
                            f"<div style='color:{C_GREY};font-size:0.65rem;'>Current</div>"
                            f"<div style='color:{sig_col};font-weight:700;font-size:0.8rem;'>"
                            f"{SIGNAL_EMOJI_RISK[signal]} {SIGNAL_LABELS_RISK[signal]}</div>"
                            f"</div>"
                            f"<div style='flex:1;background:{sig_col_50}11;border:1px solid {sig_col_50}33;border-radius:4px;padding:4px 6px;'>"
                            f"<div style='color:{C_GREY};font-size:0.65rem;'>50-Day</div>"
                            f"<div style='color:{sig_col_50};font-weight:700;font-size:0.8rem;'>"
                            f"{SIGNAL_EMOJI_RISK[signal_50]} {SIGNAL_LABELS_RISK[signal_50]}</div>"
                            f"</div>"
                            f"</div>"
                            f"<div style='color:#e2e8f0;font-size:0.88rem;font-weight:600;margin-top:4px;'>{val}</div>"
                            f"<div style='color:{C_GREY};font-size:0.72rem;margin-top:4px;'>{text}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.error(f"Current signals tab error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SMART MONEY SECTION
# ══════════════════════════════════════════════════════════════════════════════

elif section == "💰 Smart Money":
    st.title("💰 Smart Money — Insider Transactions & 13F Holdings")
    st.markdown("---")

    try:
        sm_df    = load_smart_money_db()
        sm_13f   = load_smart_money_13f()
        db_path  = DATA_ROOT / "smart_money" / "smartmoney.db"
        db_mtime = _file_mtime(db_path)
    except Exception as e:
        sm_df    = pd.DataFrame()
        sm_13f   = pd.DataFrame()
        db_mtime = "—"
        st.error(f"Error loading Smart Money DB: {e}")

    sm_tabs = st.tabs(["🔍 Insider Buys", "🏦 13F Holdings"])

    # ── Smart Money Tab 1: Insider Buys ───────────────────────────────────
    with sm_tabs[0]:
        try:
            st.caption(f"smartmoney.db — last updated: {db_mtime}")
            st.info("ℹ️ Note: Private companies (e.g. Figma) are not required to file Form 4 with the SEC. Only public company insider transactions appear here.")

            if sm_df.empty:
                db_exists = (DATA_ROOT / "smart_money" / "smartmoney.db").exists()
                if db_exists:
                    st.warning(
                        f"smartmoney.db exists (last updated: {db_mtime}) but contains no insider purchase "
                        f"transactions yet. Run: `cd projects/smart-money && python3 run_daily.py --form4-only --days 90 --limit 500`"
                    )
                else:
                    st.markdown(
                        f"<div style='background:{C_CARD};border:1px solid {C_YELLOW}55;"
                        f"border-left:4px solid {C_YELLOW};border-radius:8px;padding:24px;'>"
                        f"<div style='color:{C_YELLOW};font-size:1.1em;font-weight:700;margin-bottom:8px;'>⏳ Insider pipeline not yet run</div>"
                        f"<div style='color:{C_GREY};'>Run: cd projects/smart-money && python3 run_daily.py --form4-only --days 90 --limit 500</div>"
                        f"<div style='color:{C_GREY};margin-top:6px;font-size:0.82rem;'>Last attempted: {db_mtime}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Buys", len(sm_df))
                if "ticker" in sm_df.columns:
                    m2.metric("Unique Tickers", sm_df["ticker"].nunique())
                if "total_value" in sm_df.columns:
                    total_val = sm_df["total_value"].apply(lambda v: float(v) if pd.notna(v) else 0).sum()
                    m3.metric("Total $ Value", f"${total_val/1e6:.1f}M" if total_val >= 1e6 else f"${total_val:,.0f}")
                if "conviction_score" in sm_df.columns:
                    m4.metric("Avg Conviction", f"{pd.to_numeric(sm_df['conviction_score'], errors='coerce').mean():.1f}")

                st.markdown("---")
                st.subheader("Top Insider Buys by Conviction")

                # Format display
                disp_sm = sm_df.copy()
                if "total_value" in disp_sm.columns:
                    disp_sm["total_value"] = disp_sm["total_value"].apply(
                        lambda v: f"${float(v):,.0f}" if pd.notna(v) else "—"
                    )
                if "price_per_share" in disp_sm.columns:
                    disp_sm["price_per_share"] = disp_sm["price_per_share"].apply(
                        lambda v: f"${float(v):.2f}" if pd.notna(v) else "—"
                    )
                if "shares" in disp_sm.columns:
                    disp_sm["shares"] = disp_sm["shares"].apply(
                        lambda v: f"{int(float(v)):,}" if pd.notna(v) else "—"
                    )
                if "conviction_score" in disp_sm.columns:
                    disp_sm["conviction_score"] = pd.to_numeric(disp_sm["conviction_score"], errors="coerce").round(1)

                display_cols_sm = [c for c in ["ticker","insider_name","role_title","shares","price_per_share",
                                                "total_value","transaction_date","conviction_score"]
                                   if c in disp_sm.columns]
                disp_sm = disp_sm[display_cols_sm].head(50)
                disp_sm = disp_sm.rename(columns={
                    "ticker": "Ticker",
                    "insider_name": "Insider",
                    "role_title": "Role",
                    "shares": "Shares",
                    "price_per_share": "Price",
                    "total_value": "Total Value",
                    "transaction_date": "Date",
                    "conviction_score": "Conviction",
                })

                def _conv_color(val):
                    try:
                        v = float(val)
                        if v >= 80: return f"color: {C_GREEN}; font-weight: bold"
                        if v >= 50: return f"color: {C_YELLOW}"
                        return f"color: {C_GREY}"
                    except Exception:
                        return ""

                styled_sm = disp_sm.style
                if "Conviction" in disp_sm.columns:
                    styled_sm = styled_sm.applymap(_conv_color, subset=["Conviction"])
                st.dataframe(styled_sm, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Insider Buys tab error: {e}")

    # ── Smart Money Tab 2: 13F Holdings ───────────────────────────────────
    with sm_tabs[1]:
        try:
            if sm_13f.empty:
                st.info(
                    "No 13F fund holdings data yet. "
                    "Run: `cd projects/smart-money && python3 run_daily.py --13f-only` "
                    "to fetch the latest institutional 13F filings."
                )
                st.markdown("""
**What are 13F filings?**
Institutional investment managers with >$100M AUM must file quarterly 13F reports with the SEC,
disclosing all equity holdings. This provides a ~45-day lagged view of what major funds hold.

**Coming when data is available:**
- Fund name, position size, portfolio weight
- New positions, additions, reductions, exits
- Aggregated smart-money accumulation score per stock
""")
            else:
                # Summary
                n_funds = sm_13f["fund_name"].nunique() if "fund_name" in sm_13f.columns else 0
                n_positions = len(sm_13f)
                total_aum = sm_13f["market_value"].apply(lambda v: float(v) if pd.notna(v) else 0).sum() if "market_value" in sm_13f.columns else 0

                m1, m2, m3 = st.columns(3)
                m1.metric("Funds Tracked", n_funds)
                m2.metric("Positions", n_positions)
                m3.metric("Total AUM Tracked", f"${total_aum/1e9:.1f}B" if total_aum >= 1e9 else f"${total_aum/1e6:.0f}M")

                st.markdown("---")
                st.subheader("Top Holdings by Market Value")

                disp_13f = sm_13f.copy()
                if "market_value" in disp_13f.columns:
                    disp_13f["market_value"] = disp_13f["market_value"].apply(
                        lambda v: f"${float(v)/1e6:.1f}M" if pd.notna(v) and float(v) >= 1e6 else (f"${float(v):,.0f}" if pd.notna(v) else "—")
                    )
                if "pct_portfolio" in disp_13f.columns:
                    disp_13f["pct_portfolio"] = disp_13f["pct_portfolio"].apply(
                        lambda v: f"{float(v):.2f}%" if pd.notna(v) else "—"
                    )
                if "shares" in disp_13f.columns:
                    disp_13f["shares"] = disp_13f["shares"].apply(
                        lambda v: f"{int(float(v)):,}" if pd.notna(v) else "—"
                    )
                display_cols_13f = [c for c in ["fund_name","ticker","company_name","shares","market_value",
                                                  "pct_portfolio","report_date","change_type"]
                                    if c in disp_13f.columns]
                disp_13f = disp_13f[display_cols_13f].rename(columns={
                    "fund_name": "Fund",
                    "ticker": "Ticker",
                    "company_name": "Company",
                    "shares": "Shares",
                    "market_value": "Market Value",
                    "pct_portfolio": "% Portfolio",
                    "report_date": "Report Date",
                    "change_type": "Change",
                })
                st.dataframe(disp_13f, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"13F Holdings tab error: {e}")
