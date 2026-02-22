"""
Stock Agent — Streamlit Dashboard
Visualize stock prices and financial news
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")

# ─── Load Data ────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_prices():
    files = sorted(DATA_RAW_DIR.glob("prices_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True).drop_duplicates()

@st.cache_data(ttl=60)
def load_news():
    files = sorted(DATA_RAW_DIR.glob("news_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["ticker", "title"])

@st.cache_data(ttl=60)
def load_finetune():
    files = sorted(DATA_PROCESSED_DIR.glob("finetune_data_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["prompt"])

# ─── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("📈 Stock Agent")
st.sidebar.markdown("---")

prices_df = load_prices()
news_df = load_news()
finetune_df = load_finetune()

if not prices_df.empty:
    tickers = prices_df["ticker"].unique().tolist()
    selected_tickers = st.sidebar.multiselect("Select Tickers", tickers, default=tickers)
else:
    selected_tickers = []

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last update:** {datetime.now().strftime('%H:%M:%S')}")
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# ─── Main ─────────────────────────────────────────────────────────
st.title("📊 Stock Agent Dashboard")

if prices_df.empty:
    st.warning("⚠️ No data found. Run `python main.py --scrape` first.")
    st.stop()

# Filter
filtered_prices = prices_df[prices_df["ticker"].isin(selected_tickers)]

# ─── KPIs ─────────────────────────────────────────────────────────
st.subheader("📈 Latest Prices")
latest = filtered_prices.groupby("ticker").last().reset_index()

cols = st.columns(len(latest))
for i, (_, row) in enumerate(latest.iterrows()):
    with cols[i]:
        st.metric(
            label=row["ticker"],
            value=f"${row['price']:,.2f}",
            delta=f"H: ${row['high']:,.2f} | L: ${row['low']:,.2f}",
        )

st.markdown("---")

# ─── Price Chart ──────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💹 Price Comparison")
    fig = go.Figure()
    for ticker in selected_tickers:
        ticker_df = filtered_prices[filtered_prices["ticker"] == ticker]
        fig.add_trace(go.Scatter(
            x=ticker_df["timestamp"],
            y=ticker_df["price"],
            mode="lines+markers",
            name=ticker,
        ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=400,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Volume")
    fig2 = px.bar(
        latest,
        x="ticker",
        y="volume",
        color="ticker",
        template="plotly_dark",
        height=400,
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ─── Market Cap & PE ──────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("🏦 Market Cap")
    latest_cap = latest.dropna(subset=["market_cap"])
    if not latest_cap.empty:
        fig3 = px.pie(
            latest_cap,
            values="market_cap",
            names="ticker",
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("📉 52W High vs Low")
    latest_52 = latest.dropna(subset=["52w_high", "52w_low"])
    if not latest_52.empty:
        fig4 = go.Figure()
        for _, row in latest_52.iterrows():
            fig4.add_trace(go.Bar(
                name=row["ticker"],
                x=[row["ticker"]],
                y=[row["52w_high"] - row["52w_low"]],
                base=row["52w_low"],
            ))
        fig4.update_layout(
            yaxis_title="Price Range (USD)",
            template="plotly_dark",
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ─── News Feed ────────────────────────────────────────────────────
st.subheader("📰 Latest Financial News")

if not news_df.empty:
    filtered_news = news_df[news_df["ticker"].isin(selected_tickers)]
    filtered_news = filtered_news.sort_values("timestamp", ascending=False)

    ticker_filter = st.selectbox("Filter by ticker", ["All"] + selected_tickers)
    if ticker_filter != "All":
        filtered_news = filtered_news[filtered_news["ticker"] == ticker_filter]

    for _, row in filtered_news.head(20).iterrows():
        with st.expander(f"[{row['ticker']}] {row['title']}"):
            st.markdown(f"**Summary:** {row.get('summary', 'N/A')}")
            st.markdown(f"**Published:** {row.get('published', 'N/A')}")
            if row.get("link"):
                st.markdown(f"[Read more]({row['link']})")
else:
    st.info("No news data available.")

st.markdown("---")

# ─── Fine-tuning Data ─────────────────────────────────────────────
st.subheader("🧠 Fine-tuning Dataset")
if not finetune_df.empty:
    st.metric("Total training samples", len(finetune_df))
    st.dataframe(finetune_df[["ticker", "prompt", "completion"]].head(10), use_container_width=True)
else:
    st.info("No fine-tuning data available.")
    