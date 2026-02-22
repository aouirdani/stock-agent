"""
Stock Agent — Fine-tuning
Fine-tune Qwen2.5-0.5B on financial news summarization
Using LoRA (PEFT) for efficient training on Mac M1
"""

import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from rich.console import Console

console = Console()

# ─── Config ───────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODELS_DIR = Path("models")
DATA_PROCESSED_DIR = Path("data/processed")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# LoRA config — léger pour M1
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # rank — plus petit = plus rapide
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

# Training config optimisé M1
TRAINING_ARGS = TrainingArguments(
    output_dir=str(MODELS_DIR / "qwen-finance"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=False,             # pas de fp16 sur M1
    bf16=False,
    optim="adamw_torch",
    report_to="none",       # désactive wandb par défaut
    push_to_hub=False,
)


# ─── Load & Prepare Data ──────────────────────────────────────────
def load_data() -> Dataset:
    """Load processed fine-tuning data"""
    files = sorted(DATA_PROCESSED_DIR.glob("finetune_data_*.csv"))

    if not files:
        console.print("[red]❌ No fine-tuning data found. Run the scraper first![/red]")
        return None

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["prompt"])
    df = df.dropna(subset=["prompt", "completion"])

    console.print(f"[green]✅ Loaded {len(df)} training samples[/green]")
    return Dataset.from_pandas(df[["prompt", "completion"]])


def format_prompt(sample: dict, tokenizer) -> dict:
    """Format prompt for Qwen2.5 chat format"""
    messages = [
        {"role": "system", "content": "You are a financial analyst. Summarize financial news clearly and concisely."},
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["completion"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# ─── Load Model ───────────────────────────────────────────────────
def load_model():
    """Load Qwen2.5-0.5B with LoRA for M1"""
    console.print(f"[bold cyan]🤖 Loading {MODEL_NAME}...[/bold cyan]")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # float32 pour M1
        trust_remote_code=True,
        device_map="cpu",           # charger sur CPU d'abord
    )

    # Appliquer LoRA
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Déplacer sur MPS si disponible
    if torch.backends.mps.is_available():
        model = model.to("mps")
        console.print("[green]✅ Model on MPS (Apple GPU)[/green]")
    else:
        console.print("[yellow]⚠️  Running on CPU[/yellow]")

    return model, tokenizer


# ─── Train ────────────────────────────────────────────────────────
def train():
    console.print("[bold magenta]🚀 Starting Fine-tuning...[/bold magenta]\n")

    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/cyan]")

    # Load data
    dataset = load_data()
    if dataset is None:
        return

    # Load model
    model, tokenizer = load_model()

    # Format dataset
    dataset = dataset.map(
        lambda x: format_prompt(x, tokenizer),
        remove_columns=dataset.column_names
    )

    console.print(f"[cyan]📊 Training samples: {len(dataset)}[/cyan]")

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=TRAINING_ARGS,
    )

    console.print("\n[bold cyan]🏋️  Training started...[/bold cyan]")
    trainer.train()

    # Save model
    output_path = MODELS_DIR / "qwen-finance-lora"
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    console.print(f"\n[bold green]✅ Model saved → {output_path}[/bold green]")


# ─── Inference ────────────────────────────────────────────────────
def summarize(headline: str, model_path: str = None) -> str:
    """Use fine-tuned model to summarize a financial headline"""
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    path = model_path or str(MODELS_DIR / "qwen-finance-lora")
    model = PeftModel.from_pretrained(base_model, path)

    if torch.backends.mps.is_available():
        model = model.to("mps")

    messages = [
        {"role": "system", "content": "You are a financial analyst. Summarize financial news clearly and concisely."},
        {"role": "user", "content": f"Summarize this financial news headline: {headline}"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    if torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    train()