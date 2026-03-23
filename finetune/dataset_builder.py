"""
finetune/dataset_builder.py
Builds the fine-tuning dataset from Phase 1 failure analysis.

Strategy:
  1. Read failure_report.json to identify the weakest category
  2. Load the corresponding dataset
  3. Generate high-quality reasoning traces using a stronger model
  4. Filter to only traces that produce correct final answers
  5. Save as JSONL in instruction-tuning format

Run:
    python finetune/dataset_builder.py
    python finetune/dataset_builder.py --category multistep_arithmetic --samples 3000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml
from tqdm import tqdm

from models.groq_client import get_client as _get_groq_client
from models import get_client
from eval.task_decomposer import TaskDecomposer, parse_reasoning_trace
from data_loaders.gsm8k_loader import load_gsm8k
from data_loaders.crass_loader import load_crass
from data_loaders.toolbench_loader import load_toolbench
from data_loaders.factual_synthetic import load_factual_synthetic
from experiments.run_baseline_eval import (
    task_to_decomposer_fields,
    ground_truth_for_category,
    load_category_tasks,
)

logger = logging.getLogger(__name__)


# ── Trace generation prompt ───────────────────────────────────
# Stronger model generates "gold standard" traces

_TRACE_GEN_SYSTEM = """You are an expert reasoning tutor. Generate a perfect, step-by-step reasoning trace for the given problem.

Requirements:
- Number each step: Step 1:, Step 2:, etc.
- Each step must contain exactly one logical operation or inference
- Each step must be explicit, not skipping any computation
- End with: Final Answer: <answer>
- The final answer MUST be correct
- Do not make any factual errors"""


def _format_training_example(
    system: str,
    user_prompt: str,
    assistant_response: str,
) -> dict:
    """
    Format as instruction-tuning JSONL entry.
    Compatible with TRL's SFTTrainer using the 'text' field format.
    """
    # Chat template format (works with most HuggingFace models)
    text = (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{user_prompt}\n"
        f"<|assistant|>\n{assistant_response}"
    )
    return {
        "text": text,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
    }


def _is_correct_trace(
    raw_response: str,
    ground_truth: str,
    category: str,
    quality_client=None,
    quality_model: str | None = None,
) -> bool:
    """
    Quick correctness check before adding to training set.

    For arithmetic: strict numeric comparison (objective).
    For non-arithmetic: content word overlap fallback.

    Note: this is NOT the same as evaluation-time checking — here we're
    filtering training data, so recall matters more than precision.
    A slightly lenient filter is acceptable because we're generating
    traces with a strong model that is usually correct.
    """
    if not ground_truth:
        return True  # no ground truth — include anyway
    _, final_answer = parse_reasoning_trace(raw_response)
    if not final_answer:
        return False

    import re
    pred = final_answer.strip().lower()
    truth = ground_truth.strip().lower()

    if pred == truth:
        return True

    # Arithmetic: strict numeric check
    if category == "multistep_arithmetic":
        pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
        truth_nums = re.findall(r"-?\d+(?:\.\d+)?", truth)
        if pred_nums and truth_nums:
            try:
                return abs(float(pred_nums[-1]) - float(truth_nums[-1])) < 1e-6
            except ValueError:
                pass
        return False

    # Factual: content word overlap (lower bar for training data recall)
    if category == "factual_consistency":
        truth_words = set(truth.split())
        pred_words = set(pred.split())
        if truth_words and len(truth_words & pred_words) / len(truth_words) > 0.45:
            return True

    # Tool-use: check tool names appear
    if category == "tool_use_planning":
        tool_names = re.findall(r"(?:call\s+)?(\w+)\s+with", truth, re.I)
        if tool_names:
            matched = sum(1 for t in tool_names if t.lower() in pred)
            return matched / len(tool_names) >= 0.5

    # Counterfactual: content word overlap
    if category == "causal_counterfactual":
        stop = {'the', 'a', 'an', 'is', 'was', 'were', 'would', 'have', 'has',
                'had', 'been', 'be', 'to', 'of', 'and', 'in', 'that', 'it',
                'not', 'this', 'if', 'then', 'so', 'but', 'or', 'for', 'with'}
        truth_content = set(re.findall(r'\w+', truth)) - stop
        pred_content = set(re.findall(r'\w+', pred)) - stop
        if truth_content and len(truth_content & pred_content) / len(truth_content) > 0.4:
            return True

    # Optional final semantic check via independent judge for non-arithmetic tasks.
    if quality_client and quality_model and category != "multistep_arithmetic":
        try:
            judge_prompt = (
                f"Category: {category}\n\n"
                f"Ground truth answer: {ground_truth}\n\n"
                f"Candidate answer: {final_answer}\n\n"
                "Return JSON: {\"correct\": true|false, \"reason\": \"...\"}."
            )
            raw = quality_client.complete_json(
                model=quality_model,
                user_prompt=judge_prompt,
                system_prompt=(
                    "You are a strict evaluator. Mark correct=true only if the candidate "
                    "captures the key factual content of ground truth."
                ),
                temperature=0.0,
                max_tokens=120,
            )
            return bool(raw.get("correct", False))
        except Exception:
            # Fall through to conservative rejection if semantic judge fails.
            return False

    return False


def build_dataset(
    category: str,
    tasks: list[dict],
    cfg: dict,
    target_size: int = 3000,
    output_path: str = "datasets/data/finetune_dataset.jsonl",
    provider: str = "groq",
    trace_model_override: str = None,
    quality_client=None,
    quality_model: str | None = None,
) -> int:
    """
    Generate fine-tuning dataset for a given category.

    Args:
        category:    Task category to generate traces for
        tasks:       List of task dicts (from dataset loaders)
        cfg:         Full config dict
        target_size: Desired number of training examples
        output_path: Output JSONL file path

    Returns:
        Number of examples written
    """
    client = get_client(cfg, provider=provider)
    decomposer = TaskDecomposer()
    trace_model = trace_model_override or cfg["models"]["trace_gen_model"]
    eval_cfg = cfg["eval"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    written = 0
    attempted = 0
    skipped_incorrect = 0

    # We may need to loop through tasks multiple times if target > len(tasks)
    task_pool = tasks * (max(1, target_size // len(tasks) + 2))

    logger.info(f"Building fine-tuning dataset: category={category} target={target_size}")
    logger.info(f"  Trace generator model: {trace_model}")
    logger.info(f"  Output: {output_path}")

    with open(output_path, "w", encoding="utf-8") as fout:
        with tqdm(total=target_size, desc=f"Generating {category} traces") as pbar:
            for task_dict in task_pool:
                if written >= target_size:
                    break

                attempted += 1
                # Detect real category if mixed
                real_cat = task_dict.get("category", category)
                task_id = task_dict.get("id", f"{real_cat}_{attempted:05d}")
                fields = task_to_decomposer_fields(task_dict, real_cat)
                gt = ground_truth_for_category(task_dict, real_cat)

                # Build prompt
                user_prompt = decomposer.build_prompt(real_cat, **fields)
                system = _TRACE_GEN_SYSTEM

                try:
                    resp = client.complete(
                        model=trace_model,
                        user_prompt=user_prompt,
                        system_prompt=system,
                        temperature=0.3,    # slight diversity for augmentation
                        max_tokens=eval_cfg["max_tokens"],
                    )
                    raw = resp.content

                except Exception as e:
                    logger.warning(f"[{task_id}] Generation error: {e}")
                    continue

                # Verify correctness before adding
                if gt and not _is_correct_trace(
                    raw,
                    gt,
                    real_cat,
                    quality_client=quality_client,
                    quality_model=quality_model,
                ):
                    skipped_incorrect += 1
                    if skipped_incorrect % 50 == 0:
                        logger.debug(f"Skipped {skipped_incorrect} incorrect traces so far")
                    continue

                example = _format_training_example(
                    system=system,
                    user_prompt=user_prompt,
                    assistant_response=raw.strip(),
                )
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)

    logger.info(
        f"Dataset built: {written} examples written | "
        f"{skipped_incorrect} incorrect traces skipped | "
        f"{attempted} total attempted"
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset from Phase 1 failures")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--category", default=None, help="Override target category")
    parser.add_argument("--samples", type=int, default=None, help="Target training examples")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--provider", choices=["groq", "ollama", "openai"], default="ollama",
                        help="LLM provider for trace generation. openai=GPT-4.1/GPT-5.3 (best quality)")
    parser.add_argument("--trace-model", default=None,
                        help="Model for generating traces (e.g. qwen2.5:7b)")
    parser.add_argument(
        "--quality-provider",
        choices=["groq", "ollama", "openai"],
        default=None,
        help="Optional independent provider for correctness filtering of generated traces",
    )
    parser.add_argument(
        "--quality-model",
        default=None,
        help="Optional model for independent correctness filtering (e.g. qwen2.5:14b or gpt-4o-mini)",
    )
    args = parser.parse_args()

    config_path = _ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load tasks (use training split for fine-tuning, not test)
    cfg_for_load = dict(cfg)
    cfg_for_load["datasets"]["gsm8k"]["split"] = "train"
    cfg_for_load["eval"]["samples_per_category"] = 5000  # large pool

    # Determine target category from failure report
    category = args.category
    
    # NEW: If category is "mixed", combine multiple failure types
    if category == "mixed":
        logger.info("Building mixed dataset (Arithmetic + Factual + Tool-Use + Counterfactual)")
        tasks = []
        # Weighted mix: 40% arithmetic (hardest), 20% each other
        tasks.extend(load_category_tasks("multistep_arithmetic", cfg_for_load)[:1200])
        tasks.extend(load_category_tasks("factual_consistency", cfg_for_load)[:600])
        tasks.extend(load_category_tasks("tool_use_planning", cfg_for_load)[:600])
        tasks.extend(load_category_tasks("causal_counterfactual", cfg_for_load)[:600])
        logger.info(f"Total mixed tasks pool: {len(tasks)}")
    elif not category:
        failure_path = cfg["paths"]["failure_report"]
        if os.path.exists(failure_path):
            with open(failure_path) as f:
                failure_report = json.load(f)
            category = failure_report.get("finetune_target_category", "multistep_arithmetic")
            logger.info(f"Using failure report target category: {category}")
        else:
            category = "multistep_arithmetic"
            logger.warning(f"No failure report found — defaulting to {category}")

    if category != "mixed":
        # Load tasks for single category
        tasks = load_category_tasks(category, cfg_for_load)

    # Use "mixed" as the category label for file generation if mixed
    if not category:
        category = "mixed"

    target_size = args.samples or 3000
    output = args.output or cfg["finetune"]["dataset_path"]

    quality_client = None
    quality_model = None
    if args.quality_provider:
        quality_client = get_client(cfg, provider=args.quality_provider)
        quality_model = args.quality_model or cfg["models"]["evaluator_model"]
        logger.info(
            f"Using independent quality judge: {args.quality_provider}/{quality_model}"
        )

    n = build_dataset(
        category=category,
        tasks=tasks,
        cfg=cfg,
        target_size=target_size,
        output_path=output,
        provider=args.provider,
        trace_model_override=args.trace_model,
        quality_client=quality_client,
        quality_model=quality_model,
    )

    logger.info(f"\nFine-tuning dataset complete: {n} examples at {output}")


if __name__ == "__main__":
    main()
