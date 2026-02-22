"""
Stock Agent — Web Scraper
Scrapes Yahoo Finance prices + financial news
"""

import yfinance as yf
import feedparser
import pandas as pd
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

# ─── Config ───────────────────────────────────────────────────────
TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN"]

NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.investing.com/rss/news.rss",
]

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─── Scraper Prix Yahoo Finance ────────────────────────────────────
def scrape_prices(tickers: list[str]) -> pd.DataFrame:
    """Scrape stock prices from Yahoo Finance"""
    console.print("[bold cyan]📈 Scraping stock prices...[/bold cyan]")
    records = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d", interval="1m")

            if hist.empty:
                console.print(f"[yellow]⚠️  No data for {ticker}[/yellow]")
                continue

            latest = hist.iloc[-1]
            info = stock.info

            records.append({
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "price": round(latest["Close"], 2),
                "open": round(latest["Open"], 2),
                "high": round(latest["High"], 2),
                "low": round(latest["Low"], 2),
                "volume": int(latest["Volume"]),
                "market_cap": info.get("marketCap", None),
                "pe_ratio": info.get("trailingPE", None),
                "52w_high": info.get("fiftyTwoWeekHigh", None),
                "52w_low": info.get("fiftyTwoWeekLow", None),
            })
            console.print(f"[green]✅ {ticker}: ${round(latest['Close'], 2)}[/green]")

        except Exception as e:
            console.print(f"[red]❌ Error scraping {ticker}: {e}[/red]")

    df = pd.DataFrame(records)
    return df


# ─── Scraper News ──────────────────────────────────────────────────
def scrape_news(tickers: list[str]) -> pd.DataFrame:
    """Scrape financial news from RSS feeds"""
    console.print("\n[bold cyan]📰 Scraping financial news...[/bold cyan]")
    records = []

    for ticker in tickers:
        feed_url = NEWS_FEEDS[0].format(ticker=ticker)
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:5]:  # 5 news par ticker
                records.append({
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": "Yahoo Finance RSS",
                })

            console.print(f"[green]✅ {ticker}: {len(feed.entries[:5])} articles[/green]")

        except Exception as e:
            console.print(f"[red]❌ Error scraping news for {ticker}: {e}[/red]")

    df = pd.DataFrame(records)
    return df


# ─── Affichage ─────────────────────────────────────────────────────
def display_prices(df: pd.DataFrame):
    table = Table(title="📊 Stock Prices", show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="cyan")
    table.add_column("Price", style="green")
    table.add_column("Open")
    table.add_column("High")
    table.add_column("Low")
    table.add_column("Volume")

    for _, row in df.iterrows():
        table.add_row(
            row["ticker"],
            f"${row['price']}",
            f"${row['open']}",
            f"${row['high']}",
            f"${row['low']}",
            f"{row['volume']:,}",
        )
    console.print(table)


# ─── Main ──────────────────────────────────────────────────────────
def run():
    console.print("[bold magenta]🤖 Stock Agent Starting...[/bold magenta]\n")

    # Scrape prices
    prices_df = scrape_prices(TICKERS)
    display_prices(prices_df)

    # Scrape news
    news_df = scrape_news(TICKERS)

    return prices_df, news_df


if __name__ == "__main__":
    prices_df, news_df = run()
    console.print(f"\n[bold green]✅ Done! {len(prices_df)} stocks, {len(news_df)} articles[/bold green]")