"""
finetune/train_lora.py
QLoRA fine-tuning script using PEFT + bitsandbytes + TRL's SFTTrainer.

Designed for RTX 4070 (8GB VRAM).
Uses 4-bit quantization (QLoRA) to fit a 3B or 8B model in 8GB VRAM.

Run:
    python finetune/train_lora.py
    python finetune/train_lora.py --config configs/config.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml

logger = logging.getLogger(__name__)


def check_dependencies() -> None:
    missing = []
    for pkg in ["torch", "transformers", "peft", "bitsandbytes", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages: {missing}\n"
            "Run: pip install torch transformers peft bitsandbytes trl datasets accelerate"
        )


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train(cfg: dict, dry_run: bool = False) -> None:
    check_dependencies()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    ft_cfg = cfg["finetune"]
    train_cfg = ft_cfg["training"]
    lora_cfg = ft_cfg["lora"]

    base_model_id = ft_cfg["base_model"]
    dataset_path = ft_cfg["dataset_path"]
    output_dir = ft_cfg["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LORA FINE-TUNING")
    logger.info(f"  Base model:  {base_model_id}")
    logger.info(f"  Dataset:     {dataset_path}")
    logger.info(f"  Output:      {output_dir}")
    logger.info(f"  Load 4-bit:  {train_cfg['load_in_4bit']}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN — no training, just checking setup")

    # ── GPU check ─────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Fine-tuning requires a GPU (RTX 4070 or similar)."
        )
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    # ── Load dataset ──────────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Fine-tuning dataset not found: {dataset_path}\n"
            "Run: python finetune/dataset_builder.py first"
        )
    data = load_jsonl(dataset_path)
    logger.info(f"Loaded {len(data)} training examples")

    # Train/validation split (85/15)
    split_idx = int(len(data) * 0.85)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    if dry_run:
        logger.info("Dry run complete — dataset loaded successfully")
        return

    # ── Quantization config (QLoRA) ───────────────────────────
    bnb_config = None
    if train_cfg["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",             # NF4 = better than int4 for LLMs
            bnb_4bit_compute_dtype=torch.bfloat16 if train_cfg["bf16"] else torch.float16,
            bnb_4bit_use_double_quant=True,         # double quantization saves ~0.4 bits/param
        )

    # ── Load tokenizer ────────────────────────────────────────
    logger.info(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load base model ───────────────────────────────────────
    logger.info(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if train_cfg["bf16"] else torch.float16,
    )

    # Required before PEFT with 4-bit
    if train_cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Training arguments ────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=10,
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=True,           # saves VRAM at cost of speed
        report_to="none",
        run_name=f"lora-{base_model_id.split('/')[-1]}",
        dataloader_num_workers=0,              # set 0 on Windows; increase on Linux
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=train_cfg["max_seq_length"],
        packing=False,
    )

    # ── Trainer ───────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Training complete. Model saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Check setup without training")
    args = parser.parse_args()

    config_path = _ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
