"""
Stock Agent — Main Orchestrator
Scrape → Save → Version → Fine-tune
"""

import argparse
from rich.console import Console
from rich.panel import Panel

from src.scraper.agent import run as scrape
from src.storage.storage import (
    save_prices,
    save_news,
    process_news_for_finetuning,
    dvc_add_and_commit,
    load_finetune_data,
)
from src.finetune.finetune import train, summarize

console = Console()


# ─── Pipeline complet ─────────────────────────────────────────────
def pipeline_scrape():
    """Scrape + Save + DVC version"""
    console.print(Panel("🤖 Stock Agent — Scrape Pipeline", style="bold magenta"))

    # 1. Scrape
    prices_df, news_df = scrape()

    # 2. Save
    prices_path = save_prices(prices_df)
    news_path = save_news(news_df)

    # 3. Process for fine-tuning
    finetune_path = process_news_for_finetuning(news_df)

    # 4. DVC versioning
    paths = [p for p in [prices_path, news_path, finetune_path] if p]
    dvc_add_and_commit(paths, f"data: scrape {len(prices_df)} stocks, {len(news_df)} articles")

    console.print(Panel(
        f"✅ Scraping done!\n"
        f"📈 {len(prices_df)} stocks scraped\n"
        f"📰 {len(news_df)} articles scraped\n"
        f"🧠 {len(finetune_path and open(finetune_path).readlines()) - 1 if finetune_path else 0} fine-tuning samples",
        style="bold green"
    ))


def pipeline_finetune():
    """Fine-tune Qwen2.5-0.5B on scraped data"""
    console.print(Panel("🧠 Stock Agent — Fine-tune Pipeline", style="bold magenta"))

    df = load_finetune_data()
    if df.empty:
        console.print("[red]❌ No data found. Run scrape first: python main.py --scrape[/red]")
        return

    console.print(f"[cyan]📊 {len(df)} training samples available[/cyan]")
    train()


def pipeline_inference(headline: str):
    """Summarize a headline using fine-tuned model"""
    console.print(Panel("🔍 Stock Agent — Inference", style="bold magenta"))
    console.print(f"[cyan]Headline: {headline}[/cyan]\n")
    summary = summarize(headline)
    console.print(f"[green]Summary: {summary}[/green]")


# ─── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Agent")
    parser.add_argument("--scrape", action="store_true", help="Scrape + save + version data")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune Qwen2.5-0.5B")
    parser.add_argument("--summarize", type=str, help="Summarize a headline", metavar="HEADLINE")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if args.all:
        pipeline_scrape()
        pipeline_finetune()
    elif args.scrape:
        pipeline_scrape()
    elif args.finetune:
        pipeline_finetune()
    elif args.summarize:
        pipeline_inference(args.summarize)
    else:
        parser.print_help()