"""
Financial Data Pipeline Dashboard
===================================
Author: Nithin Kumar Reddy Panthula
Description:
    Streamlit dashboard that runs the financial ETL pipeline live
    and visualizes all outputs: price charts, RSI, moving averages,
    volatility, correlation heatmap, performance summary, and top movers.

Usage:
    pip install -r requirements.txt
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import logging
import json
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Data Pipeline Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b92a5 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e8eaf0 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* Section headers */
    .section-header {
        color: #c9d1e3;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-left: 3px solid #4f8ef7;
        padding-left: 10px;
        margin: 24px 0 16px 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #13161f;
        border-right: 1px solid #2d3348;
    }

    /* Pipeline status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-success {
        background-color: #0d3b26;
        color: #34d399;
        border: 1px solid #065f46;
    }
    .status-running {
        background-color: #1c2a4a;
        color: #60a5fa;
        border: 1px solid #1e40af;
    }
    .status-error {
        background-color: #3b0d0d;
        color: #f87171;
        border: 1px solid #991b1b;
    }

    /* Ticker pills */
    .ticker-pill {
        display: inline-block;
        background: #1e2740;
        border: 1px solid #3b4a6b;
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #93c5fd;
        margin: 2px;
    }

    /* Dividers */
    hr { border-color: #2d3348 !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2e;
        border-radius: 8px 8px 0 0;
        color: #8b92a5;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #252b3b;
        color: #e8eaf0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PIPELINE FUNCTIONS (inline for dashboard use)
# ─────────────────────────────────────────────

TICKER_COLORS = {
    "AAPL": "#60a5fa",
    "MSFT": "#34d399",
    "GOOGL": "#fbbf24",
    "AMZN": "#f472b6",
    "META": "#a78bfa",
}

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return (100 - (100 / (1 + rs))).round(2)


@st.cache_data(ttl=300, show_spinner=False)
def run_pipeline(tickers: list, period: str) -> dict:
    """Full ETL pipeline — cached for 5 minutes."""
    raw_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            df = df[~df.index.duplicated(keep="first")].sort_index()
            df["Ticker"] = ticker
            raw_data[ticker] = df
        except Exception as e:
            st.error(f"Failed to fetch {ticker}: {e}")

    enriched = {}
    close_prices = pd.DataFrame()
    validation_results = {}

    for ticker, df in raw_data.items():
        df = df.copy()

        # Validate
        issues = []
        nulls = df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum()
        if nulls.sum() > 0:
            issues.append(f"{nulls.sum()} null values")
        zero_vol = (df["Volume"] == 0).sum()
        if zero_vol > 0:
            issues.append(f"{zero_vol} zero-volume days")
        bad_prices = ((df["Close"] <= 0) | (df["Open"] <= 0)).sum()
        if bad_prices > 0:
            issues.append(f"{bad_prices} invalid prices")
        validation_results[ticker] = issues if issues else ["All checks passed ✓"]

        # Transform
        df["daily_return_%"]      = (df["Close"].pct_change() * 100).round(4)
        df["cumulative_return_%"] = ((1 + df["Close"].pct_change()).cumprod() - 1) * 100
        df["cumulative_return_%"] = df["cumulative_return_%"].round(4)
        df["MA_7"]                = df["Close"].rolling(7).mean().round(2)
        df["MA_30"]               = df["Close"].rolling(30).mean().round(2)
        df["volatility_30d_%"]    = (
            df["daily_return_%"].rolling(30).std() * np.sqrt(252)
        ).round(2)
        df["RSI_14"]              = compute_rsi(df["Close"])
        df["trading_range"]       = (df["High"] - df["Low"]).round(2)

        enriched[ticker] = df
        close_prices = pd.concat(
            [close_prices, df["Close"].rename(ticker).to_frame()], axis=1
        )
        close_prices = close_prices[~close_prices.index.duplicated(keep="first")]

    # Performance summary
    summary_rows = []
    for ticker, df in enriched.items():
        summary_rows.append({
            "Ticker"           : ticker,
            "Start Price"      : round(df["Close"].iloc[0], 2),
            "End Price"        : round(df["Close"].iloc[-1], 2),
            "Cumulative Return": round(df["cumulative_return_%"].iloc[-1], 2),
            "Avg Daily Return" : round(df["daily_return_%"].mean(), 4),
            "Volatility 30d"   : round(df["volatility_30d_%"].iloc[-1], 2),
            "RSI 14"           : round(df["RSI_14"].iloc[-1], 1),
            "Avg Volume"       : int(df["Volume"].mean()),
            "Trading Days"     : len(df),
        })
    performance_summary = pd.DataFrame(summary_rows).sort_values(
        "Cumulative Return", ascending=False
    ).reset_index(drop=True)

    # Correlation matrix
    correlation_matrix = close_prices.pct_change().corr().round(4)

    # Top movers
    all_daily = pd.concat([
        df[["Ticker", "daily_return_%", "Close", "Volume"]].assign(date=df.index)
        for df in enriched.values()
    ]).reset_index(drop=True)
    top_movers = (
        all_daily.iloc[all_daily["daily_return_%"].abs().sort_values(ascending=False).index]
        .head(20).reset_index(drop=True)
    )

    return {
        "enriched"           : enriched,
        "combined_prices"    : close_prices,
        "performance_summary": performance_summary,
        "correlation_matrix" : correlation_matrix,
        "top_movers"         : top_movers,
        "validation_results" : validation_results,
        "run_timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="#13161f",
    plot_bgcolor="#13161f",
    font=dict(color="#8b92a5", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#1e2535", linecolor="#2d3348", showgrid=True),
    yaxis=dict(gridcolor="#1e2535", linecolor="#2d3348", showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2d3348"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def price_chart(enriched: dict, tickers: list) -> go.Figure:
    fig = go.Figure()
    for ticker in tickers:
        df = enriched[ticker]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            name=ticker, mode="lines",
            line=dict(color=TICKER_COLORS.get(ticker, "#fff"), width=2),
            hovertemplate=f"<b>{ticker}</b><br>Date: %{{x|%b %d}}<br>Close: $%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(**CHART_LAYOUT, title="Close Price", height=380,
                      hovermode="x unified")
    return fig


def cumulative_return_chart(enriched: dict, tickers: list) -> go.Figure:
    fig = go.Figure()
    for ticker in tickers:
        df = enriched[ticker]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["cumulative_return_%"],
            name=ticker, mode="lines",
            line=dict(color=TICKER_COLORS.get(ticker, "#fff"), width=2),
            fill="tozeroy", fillcolor=TICKER_COLORS.get(ticker, "#fff").replace(")", ",0.05)").replace("rgb", "rgba") if "rgb" in TICKER_COLORS.get(ticker, "") else "rgba(99,110,250,0.04)",
            hovertemplate=f"<b>{ticker}</b><br>Return: %{{y:.2f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="#2d3348", line_dash="dot")
    fig.update_layout(**CHART_LAYOUT, title="Cumulative Return (%)", height=380,
                      hovermode="x unified")
    return fig


def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=[f"{ticker} — OHLC + Moving Averages", "RSI (14)", "Volume"],
    )
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#34d399", decreasing_line_color="#f87171",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_7"], name="MA 7",
        line=dict(color="#fbbf24", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_30"], name="MA 30",
        line=dict(color="#a78bfa", width=1.5)), row=1, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI 14",
        line=dict(color="#60a5fa", width=1.5), showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line_color="#f87171", line_dash="dash", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_color="#34d399", line_dash="dash", line_width=1, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="#f87171", opacity=0.05, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="#34d399", opacity=0.05, row=2, col=1)
    # Volume
    colors = ["#34d399" if r >= 0 else "#f87171" for r in df["daily_return_%"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, showlegend=False, opacity=0.7), row=3, col=1)

    layout = dict(**CHART_LAYOUT)
    layout["height"] = 640
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    return fig


def volatility_chart(enriched: dict, tickers: list) -> go.Figure:
    fig = go.Figure()
    for ticker in tickers:
        df = enriched[ticker]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["volatility_30d_%"],
            name=ticker, mode="lines",
            line=dict(color=TICKER_COLORS.get(ticker, "#fff"), width=2),
        ))
    fig.update_layout(**CHART_LAYOUT, title="30-Day Annualized Volatility (%)", height=320)
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[[0, "#1a1f2e"], [0.5, "#3b4a6b"], [1, "#4f8ef7"]],
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#8b92a5"),
            bgcolor="#13161f",
            bordercolor="#2d3348",
        ),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Return Correlation Matrix",
        height=380,
    )
    return fig


def risk_return_scatter(performance_summary: pd.DataFrame) -> go.Figure:
    df = performance_summary
    fig = go.Figure()
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        fig.add_trace(go.Scatter(
            x=[row["Volatility 30d"]], y=[row["Cumulative Return"]],
            mode="markers+text",
            marker=dict(
                size=18,
                color=TICKER_COLORS.get(ticker, "#fff"),
                line=dict(width=2, color="white"),
                symbol="circle",
            ),
            text=[ticker],
            textposition="top center",
            textfont=dict(color="#e8eaf0", size=11),
            name=ticker,
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                f"Volatility: {row['Volatility 30d']:.1f}%<br>"
                f"Return: {row['Cumulative Return']:+.2f}%<extra></extra>"
            ),
        ))
    # Quadrant dividers
    mid_vol = df["Volatility 30d"].mean()
    fig.add_vline(x=mid_vol, line_color="#2d3348", line_dash="dot")
    fig.add_hline(y=0, line_color="#2d3348", line_dash="dot")
    fig.update_layout(
        **CHART_LAYOUT,
        title="Risk vs. Return",
        height=380,
        showlegend=False,
        xaxis_title="Volatility 30d (%)",
        yaxis_title="Cumulative Return (%)",
    )
    return fig


def top_movers_chart(top_movers: pd.DataFrame) -> go.Figure:
    df = top_movers.head(10).copy()
    df["date_str"] = df["date"].astype(str).str[:10]
    df["label"] = df["Ticker"] + " " + df["date_str"]
    df = df.sort_values("daily_return_%")
    colors = ["#34d399" if v >= 0 else "#f87171" for v in df["daily_return_%"]]
    fig = go.Figure(go.Bar(
        x=df["daily_return_%"], y=df["label"],
        orientation="h",
        marker_color=colors,
        text=df["daily_return_%"].apply(lambda v: f"{v:+.2f}%"),
        textposition="outside",
        textfont=dict(color="#e8eaf0"),
        hovertemplate="<b>%{y}</b><br>Daily Move: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, title="Top 10 Single-Day Movers", height=400,
                      xaxis_title="Daily Return (%)")
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Config")
    st.markdown("---")

    ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    selected_tickers = st.multiselect(
        "Tickers",
        options=ALL_TICKERS,
        default=ALL_TICKERS,
        help="Select which stocks to analyze",
    )

    period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
                  "1 Year": "1y", "2 Years": "2y"}
    selected_period_label = st.selectbox("Period", list(period_map.keys()), index=2)
    selected_period = period_map[selected_period_label]

    run_btn = st.button("▶  Run Pipeline", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 🔍 Deep Dive")
    selected_ticker = st.selectbox("Select Ticker", options=selected_tickers or ALL_TICKERS)

    st.markdown("---")
    st.markdown(
        "<div style='color:#5a6278;font-size:0.75rem;line-height:1.6'>"
        "📌 <b>Nithin Kumar Reddy Panthula</b><br>"
        "MS Cybersecurity · Auburn University<br>"
        "Atlanta, GA<br><br>"
        "<a href='https://linkedin.com/in/nithin-panthula' style='color:#60a5fa'>LinkedIn</a> · "
        "<a href='https://github.com/nithin2357-hue' style='color:#60a5fa'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(
        "<h1 style='color:#e8eaf0;font-size:1.8rem;font-weight:800;margin-bottom:2px'>"
        "📈 Financial Data Pipeline Dashboard"
        "</h1>"
        "<p style='color:#5a6278;font-size:0.88rem;margin-top:0'>"
        "Live ETL · Feature Engineering · Analytics · Nithin Kumar Reddy Panthula"
        "</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────

if not selected_tickers:
    st.warning("Please select at least one ticker in the sidebar.")
    st.stop()

with col_status:
    status_placeholder = st.empty()

data = None
with st.spinner("Running ETL pipeline... fetching live data from Yahoo Finance"):
    try:
        status_placeholder.markdown(
            "<div style='padding-top:20px'><span class='status-badge status-running'>⏳ Running</span></div>",
            unsafe_allow_html=True
        )
        data = run_pipeline(tuple(selected_tickers), selected_period)
        status_placeholder.markdown(
            f"<div style='padding-top:20px'><span class='status-badge status-success'>✓ Complete</span></div>"
            f"<div style='color:#5a6278;font-size:0.72rem;margin-top:4px'>{data['run_timestamp']}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        status_placeholder.markdown(
            "<span class='status-badge status-error'>✗ Failed</span>", unsafe_allow_html=True
        )
        st.error(f"Pipeline error: {e}")
        st.stop()

enriched         = data["enriched"]
combined_prices  = data["combined_prices"]
perf             = data["performance_summary"]
corr             = data["correlation_matrix"]
top_movers       = data["top_movers"]
validation       = data["validation_results"]

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────

st.markdown("---")
kpi_cols = st.columns(5)

for i, ticker in enumerate(selected_tickers):
    row = perf[perf["Ticker"] == ticker].iloc[0]
    ret  = row["Cumulative Return"]
    rsi  = row["RSI 14"]
    with kpi_cols[i % 5]:
        delta_sign = "+" if ret >= 0 else ""
        rsi_label = "🔴 Overbought" if rsi > 70 else ("🟢 Oversold" if rsi < 30 else "⚪ Neutral")
        st.metric(
            label=f"{ticker}  ·  {rsi_label}",
            value=f"${row['End Price']:,.2f}",
            delta=f"{delta_sign}{ret:.2f}% return",
        )

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────

st.markdown("---")
tab_overview, tab_deepdive, tab_risk, tab_pipeline = st.tabs([
    "📊  Overview",
    "🔍  Deep Dive",
    "⚖️  Risk & Correlation",
    "🔧  Pipeline"
])

# ── OVERVIEW TAB ──────────────────────────────
with tab_overview:
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(price_chart(enriched, selected_tickers), use_container_width=True)
    with col_r:
        st.plotly_chart(cumulative_return_chart(enriched, selected_tickers), use_container_width=True)

    st.plotly_chart(volatility_chart(enriched, selected_tickers), use_container_width=True)

    st.markdown("<div class='section-header'>Performance Summary</div>", unsafe_allow_html=True)

    def style_table(df):
        def color_return(val):
            color = "#34d399" if val > 0 else "#f87171"
            return f"color: {color}; font-weight: 600"
        def color_rsi(val):
            if val > 70:   return "color: #f87171; font-weight: 600"
            elif val < 30: return "color: #34d399; font-weight: 600"
            return "color: #e8eaf0"
        return (
            df.style
            .format({
                "Start Price"      : "${:.2f}",
                "End Price"        : "${:.2f}",
                "Cumulative Return": "{:+.2f}%",
                "Avg Daily Return" : "{:+.4f}%",
                "Volatility 30d"   : "{:.2f}%",
                "Avg Volume"       : "{:,.0f}",
            })
            .applymap(color_return, subset=["Cumulative Return", "Avg Daily Return"])
            .applymap(color_rsi,    subset=["RSI 14"])
            .set_properties(**{"background-color": "#1a1f2e", "color": "#c9d1e3"})
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#252b3b"),
                    ("color", "#8b92a5"),
                    ("font-size", "0.78rem"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "0.04em"),
                ]},
                {"selector": "tr:hover td", "props": [("background-color", "#252b3b")]},
            ])
        )

    st.dataframe(style_table(perf), use_container_width=True, hide_index=True)


# ── DEEP DIVE TAB ─────────────────────────────
with tab_deepdive:
    if selected_ticker not in enriched:
        st.warning(f"{selected_ticker} data not available.")
    else:
        df_ticker = enriched[selected_ticker]
        st.plotly_chart(candlestick_chart(df_ticker, selected_ticker), use_container_width=True)

        st.markdown("<div class='section-header'>Latest Metrics</div>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("RSI (14)", f"{df_ticker['RSI_14'].iloc[-1]:.1f}")
        m2.metric("MA 7", f"${df_ticker['MA_7'].iloc[-1]:.2f}")
        m3.metric("MA 30", f"${df_ticker['MA_30'].iloc[-1]:.2f}")
        m4.metric("Volatility 30d", f"{df_ticker['volatility_30d_%'].iloc[-1]:.2f}%")
        m5.metric("Avg Volume", f"{df_ticker['Volume'].mean()/1e6:.2f}M")

        # MA crossover signal
        ma7_last  = df_ticker["MA_7"].iloc[-1]
        ma30_last = df_ticker["MA_30"].iloc[-1]
        if ma7_last > ma30_last:
            st.success(f"📈 **Bullish Signal** — MA7 (${ma7_last:.2f}) is above MA30 (${ma30_last:.2f}). Potential upward momentum.")
        else:
            st.warning(f"📉 **Bearish Signal** — MA7 (${ma7_last:.2f}) is below MA30 (${ma30_last:.2f}). Potential downward pressure.")

        st.markdown("<div class='section-header'>Recent Daily Data</div>", unsafe_allow_html=True)
        display_cols = ["Open", "High", "Low", "Close", "Volume",
                        "daily_return_%", "MA_7", "MA_30", "RSI_14", "volatility_30d_%"]
        st.dataframe(
            df_ticker[display_cols].tail(30).iloc[::-1].style
            .format({
                "Open": "${:.2f}", "High": "${:.2f}", "Low": "${:.2f}", "Close": "${:.2f}",
                "Volume": "{:,.0f}", "daily_return_%": "{:+.2f}%",
                "MA_7": "${:.2f}", "MA_30": "${:.2f}",
                "RSI_14": "{:.1f}", "volatility_30d_%": "{:.2f}%",
            })
            .applymap(
                lambda v: "color: #34d399" if isinstance(v, str) and v.startswith("+") else
                          "color: #f87171" if isinstance(v, str) and v.startswith("-") else "",
                subset=["daily_return_%"]
            )
            .set_properties(**{"background-color": "#1a1f2e", "color": "#c9d1e3"}),
            use_container_width=True,
            height=400,
        )


# ── RISK & CORRELATION TAB ────────────────────
with tab_risk:
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
    with col_r:
        st.plotly_chart(risk_return_scatter(perf), use_container_width=True)

    st.plotly_chart(top_movers_chart(top_movers), use_container_width=True)

    st.markdown("<div class='section-header'>Risk Classification</div>", unsafe_allow_html=True)
    risk_df = perf[["Ticker", "Cumulative Return", "Volatility 30d", "RSI 14"]].copy()
    risk_df["Sharpe Proxy"] = (
        risk_df["Cumulative Return"] / risk_df["Volatility 30d"].replace(0, np.nan)
    ).round(2)
    mean_vol = risk_df["Volatility 30d"].mean()
    risk_df["Risk Category"] = risk_df.apply(lambda r: (
        "🟢 Low Risk / High Return"  if r["Cumulative Return"] > 0 and r["Volatility 30d"] < mean_vol else
        "🟡 High Risk / High Return" if r["Cumulative Return"] > 0 else
        "🔵 Low Risk / Low Return"   if r["Volatility 30d"] < mean_vol else
        "🔴 High Risk / Low Return"
    ), axis=1)
    st.dataframe(
        risk_df.style
        .format({"Cumulative Return": "{:+.2f}%", "Volatility 30d": "{:.2f}%", "Sharpe Proxy": "{:.2f}"})
        .set_properties(**{"background-color": "#1a1f2e", "color": "#c9d1e3"}),
        use_container_width=True,
        hide_index=True,
    )


# ── PIPELINE TAB ──────────────────────────────
with tab_pipeline:
    st.markdown("<div class='section-header'>Pipeline Architecture</div>", unsafe_allow_html=True)
    st.code("""
Yahoo Finance API
        │
        ▼
  [ STAGE 1: EXTRACT ]
    • Fetch OHLCV data via yfinance for each ticker
    • Normalize timezone, deduplicate, sort by date
        │
        ▼
  [ STAGE 2: VALIDATE ]
    • Null / missing value detection
    • Zero / negative price anomaly check
    • Zero-volume day detection
    • Date gap analysis (gaps > 5 calendar days)
        │
        ▼
  [ STAGE 3: TRANSFORM ]
    • Daily & cumulative returns
    • 7-day and 30-day moving averages
    • 30-day annualized rolling volatility
    • 14-day Relative Strength Index (RSI)
    • Daily trading range (High - Low)
        │
        ▼
  [ STAGE 4: LOAD ]
    • Per-ticker enriched CSVs (5 files)
    • Combined prices, performance summary, correlation matrix
    • Top movers dataset
    • JSON pipeline summary report
    • Timestamped audit log
""", language="text")

    st.markdown("<div class='section-header'>Data Validation Results</div>", unsafe_allow_html=True)
    val_cols = st.columns(len(selected_tickers))
    for i, ticker in enumerate(selected_tickers):
        with val_cols[i]:
            issues = validation.get(ticker, ["No data"])
            all_ok = issues == ["All checks passed ✓"]
            color  = "#34d399" if all_ok else "#fbbf24"
            icon   = "✅" if all_ok else "⚠️"
            st.markdown(
                f"<div style='background:#1a1f2e;border:1px solid #2d3348;border-radius:10px;padding:12px;'>"
                f"<div style='color:{color};font-weight:700;font-size:0.9rem'>{icon} {ticker}</div>"
                + "".join(
                    f"<div style='color:#8b92a5;font-size:0.78rem;margin-top:4px'>• {issue}</div>"
                    for issue in issues
                )
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-header'>Feature Engineering Reference</div>", unsafe_allow_html=True)
    features = pd.DataFrame([
        {"Feature": "daily_return_%",       "Formula": "pct_change(Close) × 100",                   "Use Case": "Day-over-day momentum"},
        {"Feature": "cumulative_return_%",  "Formula": "(∏(1 + daily_return) − 1) × 100",           "Use Case": "Total period performance"},
        {"Feature": "MA_7",                 "Formula": "Rolling mean(Close, 7)",                     "Use Case": "Short-term trend"},
        {"Feature": "MA_30",                "Formula": "Rolling mean(Close, 30)",                    "Use Case": "Medium-term trend"},
        {"Feature": "volatility_30d_%",     "Formula": "Rolling std(daily_return, 30) × √252",       "Use Case": "Annualized risk"},
        {"Feature": "RSI_14",               "Formula": "100 − 100/(1 + avg_gain/avg_loss)",          "Use Case": "Overbought/oversold signal"},
        {"Feature": "trading_range",        "Formula": "High − Low",                                 "Use Case": "Intraday price spread"},
    ])
    st.dataframe(
        features.style.set_properties(**{"background-color": "#1a1f2e", "color": "#c9d1e3"})
        .set_table_styles([{"selector": "th", "props": [
            ("background-color", "#252b3b"), ("color", "#8b92a5"),
            ("font-size", "0.78rem"), ("text-transform", "uppercase"),
        ]}]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='section-header'>Technologies Used</div>", unsafe_allow_html=True)
    tech_cols = st.columns(4)
    technologies = [
        ("🐍 Python",     "Pipeline orchestration & feature engineering"),
        ("📦 yfinance",   "Live Yahoo Finance API client"),
        ("🐼 Pandas",     "Data ingestion, cleaning, transformation"),
        ("🔢 NumPy",      "Numerical ops & volatility calculations"),
        ("📊 Plotly",     "Interactive financial charts"),
        ("🌊 Streamlit",  "Real-time dashboard & UI"),
        ("🗃️ SQL",        "BI analytics on processed datasets"),
        ("📝 Logging",    "Pipeline audit trail & observability"),
    ]
    for i, (name, desc) in enumerate(technologies):
        with tech_cols[i % 4]:
            st.markdown(
                f"<div style='background:#1a1f2e;border:1px solid #2d3348;border-radius:10px;"
                f"padding:12px 14px;margin-bottom:8px'>"
                f"<div style='color:#e8eaf0;font-weight:700;font-size:0.88rem'>{name}</div>"
                f"<div style='color:#5a6278;font-size:0.75rem;margin-top:3px'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#3d4459;font-size:0.78rem;padding:8px 0 16px'>"
    "Financial Data Pipeline Dashboard · Nithin Kumar Reddy Panthula · "
    "MS Cybersecurity, Auburn University at Montgomery · Atlanta, GA"
    "</div>",
    unsafe_allow_html=True,
)