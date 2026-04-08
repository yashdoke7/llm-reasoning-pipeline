"""
train_lora.py  —  Fine-tune a model with LoRA on a targeted JSONL dataset.

Key changes:
  - Reads dataset path from CLI or config
  - Validates category distribution before training
  - Logs category breakdown at start
  - Supports weighted sampling to enforce target_ratio
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── dataset helpers ────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping bad line in {path}: {e}")
    return rows


def build_targeted_dataset(
    source_files: list[str],
    target_category: str,
    target_ratio: float,
    n_total: int,
    output_path: str | Path,
) -> list[dict]:
    """
    Build a training dataset with target_ratio of samples from target_category.
    Uses existing JSONL files — no regeneration needed.
    """
    by_cat: dict[str, list] = defaultdict(list)

    for fpath in source_files:
        p = Path(fpath)
        if not p.exists():
            logger.warning(f"  [skip] {fpath} not found")
            continue
        rows = load_jsonl(p)
        for row in rows:
            cat = row.get("category", "unknown")
            by_cat[cat].append(row)
        logger.info(f"  Loaded {len(rows)} rows from {fpath}")

    logger.info("Available categories:")
    for cat, items in by_cat.items():
        logger.info(f"  {cat}: {len(items)} samples")

    n_target = int(n_total * target_ratio)
    n_other  = n_total - n_target

    target_pool = by_cat.get(target_category, [])
    other_pool  = [s for cat, items in by_cat.items()
                   if cat != target_category for s in items]

    if len(target_pool) == 0:
        logger.error(f"NO samples found for target category '{target_category}'!")
        logger.error("Check that your source JSONL files contain this category key.")
        sys.exit(1)

    if len(target_pool) < n_target:
        logger.warning(f"Only {len(target_pool)} target samples, need {n_target}. "
                       f"Using all available + adjusting ratio.")
        n_target = len(target_pool)
        n_other = max(int(n_target * (1 - target_ratio) / target_ratio), 0)

    result = random.sample(target_pool, n_target)
    if other_pool:
        result += random.sample(other_pool, min(n_other, len(other_pool)))
    random.shuffle(result)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in result:
            f.write(json.dumps(row) + "\n")

    final_by_cat: dict[str, int] = defaultdict(int)
    for row in result:
        final_by_cat[row.get("category", "unknown")] += 1

    logger.info(f"\nTargeted dataset created: {output_path} ({len(result)} samples)")
    for cat, n in sorted(final_by_cat.items()):
        pct = 100 * n / len(result)
        marker = " ← TARGET" if cat == target_category else ""
        logger.info(f"  {cat}: {n} ({pct:.1f}%){marker}")

    return result


def validate_dataset(dataset: list[dict], target_category: str, target_ratio: float):
    """Warns if distribution is far from intended ratio."""
    by_cat: dict[str, int] = defaultdict(int)
    for row in dataset:
        by_cat[row.get("category", "unknown")] += 1

    n = len(dataset)
    actual_ratio = by_cat.get(target_category, 0) / n if n > 0 else 0

    if abs(actual_ratio - target_ratio) > 0.10:
        logger.warning(
            f"Dataset ratio for '{target_category}' is {actual_ratio:.2f}, "
            f"expected ~{target_ratio:.2f}. Consider rebuilding dataset."
        )
    else:
        logger.info(f"Dataset ratio check passed: {target_category} = {actual_ratio:.2f}")


def summarize_dataset(dataset: list[dict], target_category: str) -> dict[str, float | int]:
    by_cat: dict[str, int] = defaultdict(int)
    for row in dataset:
        by_cat[row.get("category", "unknown")] += 1

    total = len(dataset)
    target_count = by_cat.get(target_category, 0)
    target_ratio = (target_count / total) if total else 0.0

    logger.info("Effective dataset distribution:")
    for cat, count in sorted(by_cat.items()):
        pct = (100.0 * count / total) if total else 0.0
        marker = " ← TARGET" if cat == target_category else ""
        logger.info(f"  {cat}: {count} ({pct:.1f}%){marker}")

    return {
        "total": total,
        "target_count": target_count,
        "target_ratio": target_ratio,
    }


# ── Ollama-based training (via unsloth or direct fine-tune) ────────────────────

def format_training_prompt(row: dict) -> str:
    """
    Convert a JSONL row into a training prompt.
    Supports multiple field name conventions.
    """
    problem, solution = _extract_problem_solution(row)
    category = row.get("category", "general")

    return f"""### Category: {category}

### Problem:
{problem}

### Step-by-Step Solution:
{solution}"""


def _extract_problem_solution(row: dict) -> tuple[str, str]:
    """
    Robustly extract (problem, solution) from different dataset schemas:
      - problem/solution
      - prompt/response
      - input/output (alpaca-like)
      - messages[{role,user/assistant}]
      - text with <|user|> ... <|assistant|> blocks
    """
    problem = str(row.get("problem", row.get("prompt", row.get("input", ""))) or "").strip()
    solution = str(row.get("solution", row.get("response", row.get("output", ""))) or "").strip()

    # Chat-style messages schema.
    if (not problem or not solution) and isinstance(row.get("messages"), list):
        user_msgs = []
        assistant_msgs = []
        for m in row["messages"]:
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                user_msgs.append(content)
            elif role == "assistant":
                assistant_msgs.append(content)
        if not problem and user_msgs:
            problem = user_msgs[-1]
        if not solution and assistant_msgs:
            solution = assistant_msgs[-1]

    # Packed text schema used by this repo.
    if (not problem or not solution) and row.get("text"):
        text = str(row.get("text", ""))
        if not problem:
            m_user = re.search(r"<\|user\|>\s*(.+?)\s*<\|assistant\|>", text, flags=re.DOTALL)
            if m_user:
                problem = m_user.group(1).strip()
        if not solution:
            m_asst = re.search(r"<\|assistant\|>\s*(.+)$", text, flags=re.DOTALL)
            if m_asst:
                solution = m_asst.group(1).strip()

    return problem, solution


def save_as_alpaca_jsonl(dataset: list[dict], output_path: str | Path):
    """Save dataset in Alpaca fine-tune format for unsloth/axolotl."""
    alpaca_rows = []
    dropped = 0
    for row in dataset:
        problem, solution = _extract_problem_solution(row)
        if not problem or not solution:
            dropped += 1
            continue
        alpaca_rows.append({
            "instruction": problem,
            "input": "",
            "output": solution,
            "category": row.get("category", "unknown"),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in alpaca_rows:
            f.write(json.dumps(row) + "\n")

    logger.info(
        f"Alpaca-format dataset saved: {output_path} ({len(alpaca_rows)} rows)"
        + (f" | dropped={dropped} empty rows" if dropped else "")
    )
    return alpaca_rows


def run_unsloth_training(cfg: dict, dataset_path: str, output_dir: str):
    """
    Run fine-tuning via unsloth (fast LoRA training).
    Falls back to axolotl config if unsloth not available.
    """
    ft_cfg = cfg.get("finetune", {})
    base_model = ft_cfg.get("base_model", "qwen2.5:3b")
    lora_rank  = ft_cfg.get("lora_rank", 16)
    lora_alpha = ft_cfg.get("lora_alpha", 32)
    lr         = ft_cfg.get("learning_rate", 2e-4)
    epochs     = ft_cfg.get("epochs", 3)
    batch_size = ft_cfg.get("batch_size", 4)

    try:
        from unsloth import FastLanguageModel
        import torch
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset

        logger.info(f"Training with unsloth | base={base_model} | rank={lora_rank} | epochs={epochs}")

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing=True,
        )

        # Load dataset
        rows = load_jsonl(dataset_path)
        texts = [format_training_prompt(r) for r in rows]
        hf_dataset = Dataset.from_dict({"text": texts})

        # Train
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=ft_cfg.get("warmup_steps", 50),
                num_train_epochs=epochs,
                learning_rate=lr,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                output_dir=output_dir,
                save_steps=100,
                save_total_limit=2,
            ),
        )
        trainer.train()

        # Save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to: {output_dir}")

    except ImportError:
        logger.warning("unsloth not installed. Falling back to transformers+peft training.")
        _run_transformers_fallback(cfg, dataset_path, output_dir)


def _run_transformers_fallback(cfg: dict, dataset_path: str, output_dir: str):
    """
    Fallback trainer for environments where unsloth is unavailable.
    Produces a LoRA adapter in output_dir compatible with merge_adapter.py.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        logger.error(
            "Missing fallback training dependencies. Install with:\n"
            "  pip install torch transformers peft bitsandbytes trl datasets accelerate"
        )
        raise

    ft_cfg = cfg.get("finetune", {})
    base_model = ft_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
    lora_rank = int(ft_cfg.get("lora_rank", 16))
    lora_alpha = int(ft_cfg.get("lora_alpha", 32))
    lr = float(ft_cfg.get("learning_rate", 1e-4))
    epochs = float(ft_cfg.get("epochs", 2))
    batch_size = int(ft_cfg.get("batch_size", 1))

    rows = load_jsonl(dataset_path)
    if not rows:
        raise RuntimeError(f"No rows found in dataset: {dataset_path}")

    texts = []
    for row in rows:
        instr = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        if not instr or not out:
            continue
        if inp:
            text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
            text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
        texts.append({"text": text})

    if not texts:
        raise RuntimeError("No valid instruction/output rows after parsing Alpaca dataset.")

    logger.info(
        f"Fallback training start | base={base_model} | rows={len(texts)} | "
        f"epochs={epochs} | lr={lr}"
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = Dataset.from_list(texts)
    split_idx = max(1, int(len(ds) * 0.9))
    train_ds = ds.select(range(split_idx))
    eval_ds = ds.select(range(split_idx, len(ds))) if split_idx < len(ds) else None

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,
        learning_rate=lr,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_total_limit=2,
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
        packing=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )

    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Fallback training complete. Adapter saved at: {output_dir}")


def _generate_axolotl_config(cfg: dict, dataset_path: str, output_dir: str):
    """Fallback: write an axolotl config.yml for manual training."""
    ft_cfg = cfg.get("finetune", {})
    axolotl_cfg = {
        "base_model": ft_cfg.get("base_model", "qwen2.5:3b"),
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "load_in_4bit": True,
        "adapter": "lora",
        "lora_r": ft_cfg.get("lora_rank", 16),
        "lora_alpha": ft_cfg.get("lora_alpha", 32),
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "datasets": [{"path": dataset_path, "type": "alpaca"}],
        "output_dir": output_dir,
        "num_epochs": ft_cfg.get("epochs", 3),
        "micro_batch_size": ft_cfg.get("batch_size", 4),
        "learning_rate": ft_cfg.get("learning_rate", 2e-4),
        "sequence_len": 2048,
    }
    axolotl_path = Path(output_dir).parent / "axolotl_config.yml"
    axolotl_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml as _yaml
    with open(axolotl_path, "w", encoding="utf-8") as f:
        _yaml.dump(axolotl_cfg, f, default_flow_style=False)
    logger.info(f"Axolotl config written: {axolotl_path}")
    logger.info(f"Run manually: axolotl train {axolotl_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune with targeted LoRA dataset")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model",  default=None, help="Base model name override")
    p.add_argument("--data",   default=None, help="Training JSONL path override")
    p.add_argument("--source-files", nargs="+", default=None,
                   help="Override source JSONL files used to build targeted dataset")
    p.add_argument("--output", default=None, help="Output checkpoint dir override")
    p.add_argument("--use-prebuilt-dataset", action="store_true",
                   help="Use --data as final training dataset directly (skip targeted dataset rebuilding)")
    p.add_argument("--build-dataset-only", action="store_true",
                   help="Only build the targeted dataset, don't train")
    p.add_argument("--target-category", default=None,
                   help="Override target category from config")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        # Allow running from any cwd by resolving relative to project root.
        cfg_path = PROJECT_ROOT / args.config

    if not cfg_path.exists():
        logger.error(
            "Config not found: %s\n"
            "Try: python finetune/train_lora.py --config configs/config.yaml",
            args.config,
        )
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ft_cfg   = cfg.get("finetune", {})
    data_cfg = cfg.get("dataset", {})

    # Resolve parameters
    base_model      = args.model  or ft_cfg.get("base_model", "qwen2.5:3b")
    target_category = args.target_category or ft_cfg.get("target_category", "multistep_arithmetic")
    target_ratio    = ft_cfg.get("target_ratio", 0.70)
    n_total         = ft_cfg.get("total_samples", 400)
    source_files    = args.source_files or data_cfg.get("train_source_files", data_cfg.get("source_files", []))
    dataset_output  = args.data or data_cfg.get("output", "finetune/data/targeted_dataset.jsonl")
    checkpoint_dir  = args.output or f"finetune/checkpoints/{ft_cfg.get('output_model', 'finetuned-v3')}"

    logger.info("=" * 60)
    logger.info("LORA FINE-TUNE STARTING")
    logger.info(f"  Base model:       {base_model}")
    logger.info(f"  Target category:  {target_category} ({int(target_ratio*100)}%)")
    logger.info(f"  Total samples:    {n_total}")
    logger.info(f"  Dataset output:   {dataset_output}")
    logger.info(f"  Checkpoint dir:   {checkpoint_dir}")
    logger.info("=" * 60)

    if args.use_prebuilt_dataset:
        if not args.data:
            logger.error("--use-prebuilt-dataset requires --data <jsonl_path>.")
            sys.exit(1)
        if not Path(args.data).exists():
            logger.error(f"Prebuilt dataset not found: {args.data}")
            sys.exit(1)

        logger.info(f"Using prebuilt dataset: {args.data}")
        dataset = load_jsonl(args.data)
        if not dataset:
            logger.error(f"Prebuilt dataset is empty: {args.data}")
            sys.exit(1)

        effective = summarize_dataset(dataset, target_category)
        logger.info(
            f"Using prebuilt dataset size: {effective['total']} rows | "
            f"target={target_category} ({effective['target_ratio']:.2f})"
        )

        # Still log distribution against current target category for visibility.
        validate_dataset(dataset, target_category, target_ratio)

        alpaca_path = Path(args.data).with_suffix(".alpaca.jsonl")
        save_as_alpaca_jsonl(dataset, alpaca_path)
    else:
        # Step 1: Build targeted dataset from source files
        dataset = build_targeted_dataset(
            source_files=source_files,
            target_category=target_category,
            target_ratio=target_ratio,
            n_total=n_total,
            output_path=dataset_output,
        )

        # Step 2: Validate distribution
        validate_dataset(dataset, target_category, target_ratio)

        # Step 3: Save in Alpaca format for training frameworks
        alpaca_path = Path(dataset_output).with_suffix(".alpaca.jsonl")
        save_as_alpaca_jsonl(dataset, alpaca_path)

    if args.build_dataset_only:
        logger.info("--build-dataset-only set. Stopping before training.")
        return

    # Step 4: Train
    run_unsloth_training(cfg, str(alpaca_path), checkpoint_dir)

    logger.info("\nDone! Next steps:")
    logger.info(f"  1. Run: python finetune/merge_adapter.py --adapter {checkpoint_dir}")
    logger.info(f"  2. Run: python finetune/quantize.py --model <merged_path>")
    logger.info(f"  3. Register: ollama create {ft_cfg.get('output_model')} -f Modelfile.merged")
    logger.info(f"  4. Eval: python experiments/run_comparison_eval.py --config {args.config}")


if __name__ == "__main__":
    main()
