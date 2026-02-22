"""
Stock Agent — Storage Manager
Saves data to CSV + DVC versioning
"""

import pandas as pd
import subprocess
from datetime import datetime
from pathlib import Path
from rich.console import Console

console = Console()

# ─── Config ───────────────────────────────────────────────────────
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─── Save CSV ─────────────────────────────────────────────────────
def save_prices(df: pd.DataFrame) -> Path:
    """Save prices to CSV with timestamp"""
    if df.empty:
        console.print("[yellow]⚠️  No prices to save[/yellow]")
        return None

    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = DATA_RAW_DIR / f"prices_{date_str}.csv"

    if filepath.exists():
        # Append to existing file
        existing = pd.read_csv(filepath)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["ticker", "timestamp"])

    df.to_csv(filepath, index=False)
    console.print(f"[green]💾 Prices saved → {filepath}[/green]")
    return filepath


def save_news(df: pd.DataFrame) -> Path:
    """Save news to CSV with timestamp"""
    if df.empty:
        console.print("[yellow]⚠️  No news to save[/yellow]")
        return None

    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = DATA_RAW_DIR / f"news_{date_str}.csv"

    if filepath.exists():
        existing = pd.read_csv(filepath)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["ticker", "title"])

    df.to_csv(filepath, index=False)
    console.print(f"[green]💾 News saved → {filepath}[/green]")
    return filepath


# ─── Process Data ─────────────────────────────────────────────────
def process_news_for_finetuning(news_df: pd.DataFrame) -> Path:
    """
    Process news into prompt/completion format for fine-tuning
    Format: {"prompt": "Summarize: <title>", "completion": "<summary>"}
    """
    if news_df.empty:
        return None

    records = []
    for _, row in news_df.iterrows():
        if row.get("title") and row.get("summary"):
            records.append({
                "ticker": row["ticker"],
                "prompt": f"Summarize this financial news headline: {row['title']}",
                "completion": row["summary"],
                "source": row.get("source", ""),
                "timestamp": row.get("timestamp", ""),
            })

    processed_df = pd.DataFrame(records)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = DATA_PROCESSED_DIR / f"finetune_data_{date_str}.csv"
    processed_df.to_csv(filepath, index=False)
    console.print(f"[green]⚙️  Processed data → {filepath} ({len(processed_df)} samples)[/green]")
    return filepath


# ─── DVC Versioning ───────────────────────────────────────────────
def dvc_add_and_commit(filepaths: list[Path], message: str = None):
    """Add files to DVC and commit to git"""
    if not filepaths:
        return

    console.print("\n[bold cyan]📦 DVC versioning...[/bold cyan]")

    for filepath in filepaths:
        if filepath and filepath.exists():
            # dvc add
            result = subprocess.run(
                ["dvc", "add", str(filepath)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                console.print(f"[green]✅ DVC tracked: {filepath.name}[/green]")
            else:
                console.print(f"[red]❌ DVC error: {result.stderr}[/red]")

    # git add + commit
    commit_msg = message or f"data: update {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    subprocess.run(["git", "add", "."], capture_output=True)
    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        console.print(f"[green]✅ Git commit: {commit_msg}[/green]")
    else:
        console.print(f"[yellow]ℹ️  Git: {result.stdout.strip()}[/yellow]")


# ─── Load Data ────────────────────────────────────────────────────
def load_latest_prices() -> pd.DataFrame:
    """Load the most recent prices CSV"""
    files = sorted(DATA_RAW_DIR.glob("prices_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


def load_latest_news() -> pd.DataFrame:
    """Load the most recent news CSV"""
    files = sorted(DATA_RAW_DIR.glob("news_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


def load_finetune_data() -> pd.DataFrame:
    """Load all processed fine-tuning data"""
    files = sorted(DATA_PROCESSED_DIR.glob("finetune_data_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["prompt"])