"""
Push fine-tuned Qwen2.5-0.5B model to HuggingFace Hub
"""

from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from rich.console import Console
import torch

console = Console()

# ─── Config ───────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = Path("models/qwen-finance-lora")
HF_REPO = "Sigmafox/qwen2.5-0.5B-finance-summarizer"


def push_to_hub():
    console.print("[bold magenta]🚀 Pushing model to HuggingFace Hub...[/bold magenta]\n")

    # 1. Crée le repo sur HF
    api = HfApi()
    try:
        api.create_repo(
            repo_id=HF_REPO,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        console.print(f"[green]✅ Repo created: {HF_REPO}[/green]")
    except Exception as e:
        console.print(f"[yellow]ℹ️  Repo: {e}[/yellow]")

    # 2. Charge le modèle base + LoRA
    console.print("[cyan]Loading base model...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # 3. Merge LoRA dans le modèle base
    console.print("[cyan]Merging LoRA weights...[/cyan]")
    model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    model = model.merge_and_unload()
    console.print("[green]✅ LoRA merged[/green]")

    # 4. Push tokenizer
    console.print("[cyan]Pushing tokenizer...[/cyan]")
    tokenizer.push_to_hub(HF_REPO)

    # 5. Push model
    console.print("[cyan]Pushing model (this may take a few minutes)...[/cyan]")
    model.push_to_hub(
        HF_REPO,
        commit_message="Add fine-tuned Qwen2.5-0.5B on financial news summarization",
    )

    # 6. Crée le README
    readme = f"""---
language: en
license: apache-2.0
base_model: {BASE_MODEL}
tags:
- finance
- summarization
- qwen2.5
- lora
- fine-tuned
---

# Qwen2.5-0.5B Finance Summarizer

Fine-tuned version of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) on financial news summarization.

## Training
- **Base model:** {BASE_MODEL}
- **Method:** LoRA (r=8, alpha=16)
- **Task:** Financial news summarization
- **Data:** Yahoo Finance RSS news headlines

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("{HF_REPO}")
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO}")

messages = [
    {{"role": "system", "content": "You are a financial analyst. Summarize financial news clearly."}},
    {{"role": "user", "content": "Summarize this financial news headline: Apple reports record quarterly earnings"}},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```
"""

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Add README",
    )

    console.print(f"\n[bold green]✅ Model pushed successfully![/bold green]")
    console.print(f"[cyan]🔗 https://huggingface.co/{HF_REPO}[/cyan]")


if __name__ == "__main__":
    push_to_hub()