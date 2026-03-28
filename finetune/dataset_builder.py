"""
finetune/dataset_builder.py
Build fine-tuning datasets from:
  - Real failure cases in outputs/raw_results*.json (preferred)
  - Category pools (legacy/general baseline)
  - Hybrid of failures + balanced top-up

Run examples:
  python finetune/dataset_builder.py --strategy failure --samples 1200
  python finetune/dataset_builder.py --strategy hybrid --samples 1600 --categories multistep_arithmetic factual_consistency tool_use_planning causal_counterfactual
  python finetune/dataset_builder.py --strategy category --category mixed --samples 1200
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml
from tqdm import tqdm

from models import get_client
from eval.task_decomposer import TaskDecomposer, parse_reasoning_trace
from experiments.run_baseline_eval import (
    task_to_decomposer_fields,
    ground_truth_for_category,
    load_category_tasks,
)

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = [
    "multistep_arithmetic",
    "factual_consistency",
    "tool_use_planning",
    "causal_counterfactual",
]


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
    category: str,
    source: str,
) -> dict:
    text = (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{user_prompt}\n"
        f"<|assistant|>\n{assistant_response}"
    )
    return {
        "text": text,
        "category": category,
        "source": source,
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
    if not ground_truth:
        return True
    _, final_answer = parse_reasoning_trace(raw_response)
    if not final_answer:
        return False

    import re
    pred = final_answer.strip().lower()
    truth = ground_truth.strip().lower()

    if pred == truth:
        return True

    if category in ("multistep_arithmetic", "physics_reasoning"):
        pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
        truth_nums = re.findall(r"-?\d+(?:\.\d+)?", truth)
        if pred_nums and truth_nums:
            try:
                return abs(float(pred_nums[-1]) - float(truth_nums[-1])) < 1e-6
            except ValueError:
                pass
        return False

    if category == "factual_consistency":
        truth_words = set(truth.split())
        pred_words = set(pred.split())
        if truth_words and len(truth_words & pred_words) / len(truth_words) > 0.45:
            return True

    if category == "tool_use_planning":
        tool_names = re.findall(r"(?:call\s+)?(\w+)\s+with", truth, re.I)
        if tool_names:
            matched = sum(1 for t in tool_names if t.lower() in pred)
            return matched / len(tool_names) >= 0.5

    if category == "causal_counterfactual":
        stop = {'the', 'a', 'an', 'is', 'was', 'were', 'would', 'have', 'has',
                'had', 'been', 'be', 'to', 'of', 'and', 'in', 'that', 'it',
                'not', 'this', 'if', 'then', 'so', 'but', 'or', 'for', 'with'}
        truth_content = set(re.findall(r'\w+', truth)) - stop
        pred_content = set(re.findall(r'\w+', pred)) - stop
        if truth_content and len(truth_content & pred_content) / len(truth_content) > 0.4:
            return True

    if quality_client and quality_model and category not in ("multistep_arithmetic", "physics_reasoning"):
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
            return False

    return False


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _failure_score(result: dict) -> tuple[int, list[str]]:
    ev = result.get("evaluation", {}) or {}
    hall_summary = (((result.get("hallucination", {}) or {}).get("summary", {})) or {})
    drift = result.get("drift", {}) or {}

    reasons: list[str] = []
    score = 0

    num_invalid = int(ev.get("num_invalid") or 0)
    if num_invalid > 0:
        reasons.append(f"invalid_steps={num_invalid}")
        score += num_invalid * 3

    if ev.get("final_answer_correct") is False:
        reasons.append("wrong_final_answer")
        score += 4

    if ev.get("error_propagated"):
        reasons.append("error_propagated")
        score += 3

    flagged_steps = int(hall_summary.get("flagged_steps") or 0)
    if flagged_steps > 0:
        reasons.append(f"hallucination_flags={flagged_steps}")
        score += flagged_steps * 2

    if drift.get("has_drift"):
        reasons.append("drift_detected")
        score += 2

    return score, reasons


def _extract_failure_tasks(
    raw_results_path: str,
    include_categories: Optional[set[str]] = None,
    max_tasks: Optional[int] = None,
) -> list[dict]:
    if not os.path.exists(raw_results_path):
        raise FileNotFoundError(
            f"Raw results file not found: {raw_results_path}. "
            "Run experiments/run_baseline_eval.py first."
        )

    raw = _load_json(raw_results_path)
    extracted: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for model_name, by_category in raw.items():
        if not isinstance(by_category, dict):
            continue
        for category, task_results in by_category.items():
            if include_categories and category not in include_categories:
                continue
            for item in task_results or []:
                score, reasons = _failure_score(item)
                if score <= 0:
                    continue
                task = item.get("task", {}) or {}
                task_id = str(task.get("task_id") or f"{category}_{len(extracted):05d}")
                key = (category, task_id)
                if key in seen:
                    continue
                seen.add(key)
                extracted.append(
                    {
                        "id": task_id,
                        "category": category,
                        "prompt": task.get("prompt", ""),
                        "ground_truth": task.get("ground_truth", ""),
                        "source": "failure_mining",
                        "failure_score": score,
                        "failure_reasons": reasons,
                        "model": model_name,
                    }
                )

    extracted.sort(
        key=lambda x: (x.get("failure_score", 0), len(x.get("ground_truth", ""))),
        reverse=True,
    )
    if max_tasks:
        extracted = extracted[:max_tasks]

    logger.info(
        f"Extracted {len(extracted)} failure tasks from {raw_results_path}"
    )
    return extracted


def _resolve_target_category(category_arg: Optional[str], cfg: dict) -> str:
    if category_arg:
        return category_arg
    failure_path = cfg["paths"]["failure_report"]
    if os.path.exists(failure_path):
        fr = _load_json(failure_path)
        cat = fr.get("finetune_target_category")
        if cat:
            logger.info(f"Using failure report target category: {cat}")
            return cat
    logger.warning("No category provided and no failure report target found. Using multistep_arithmetic.")
    return "multistep_arithmetic"


def _load_category_pool(
    cfg_for_load: dict,
    categories: list[str],
    per_category_limit: int,
) -> list[dict]:
    pool: list[dict] = []
    for cat in categories:
        try:
            cat_tasks = load_category_tasks(cat, cfg_for_load)
        except Exception as e:
            logger.warning(f"Skipping category {cat}: {e}")
            continue
        random.shuffle(cat_tasks)
        if per_category_limit > 0:
            if not cat_tasks:
                selected = []
            elif len(cat_tasks) >= per_category_limit:
                selected = cat_tasks[:per_category_limit]
            else:
                # Oversample small category pools to maintain category balance.
                selected = []
                while len(selected) < per_category_limit:
                    batch = copy.deepcopy(cat_tasks)
                    random.shuffle(batch)
                    selected.extend(batch)
                selected = selected[:per_category_limit]
        else:
            selected = cat_tasks
        for t in selected:
            t_copy = dict(t)
            t_copy["source"] = "category_pool"
            pool.append(t_copy)
    return pool


def _build_task_pool(
    strategy: str,
    cfg: dict,
    cfg_for_load: dict,
    target_size: int,
    category: Optional[str],
    categories: Optional[list[str]],
    raw_results_path: str,
    max_failure_tasks: Optional[int],
) -> tuple[str, list[dict]]:
    selected_categories = categories or DEFAULT_CATEGORIES
    selected_set = set(selected_categories)
    if category and category not in ("mixed", "all"):
        selected_categories = [category]
        selected_set = {category}

    if strategy == "failure":
        tasks = _extract_failure_tasks(
            raw_results_path=raw_results_path,
            include_categories=selected_set,
            max_tasks=max_failure_tasks,
        )
        return ("mixed" if len(selected_set) > 1 else next(iter(selected_set))), tasks

    if strategy == "category":
        target_cat = _resolve_target_category(category, cfg)
        if target_cat == "mixed":
            per = max(1, target_size // len(selected_categories))
            tasks = _load_category_pool(cfg_for_load, selected_categories, per)
            return "mixed", tasks
        tasks = _load_category_pool(cfg_for_load, [target_cat], per_category_limit=target_size)
        return target_cat, tasks

    if strategy == "mixed":
        per = max(1, target_size // len(selected_categories))
        tasks = _load_category_pool(cfg_for_load, selected_categories, per)
        return "mixed", tasks

    if strategy == "hybrid":
        failure_tasks = _extract_failure_tasks(
            raw_results_path=raw_results_path,
            include_categories=selected_set,
            max_tasks=max_failure_tasks,
        )
        reserve = min(len(failure_tasks), max(1, int(target_size * 0.6)))
        seed = failure_tasks[:reserve]
        needed = max(0, target_size - len(seed))
        if needed > 0:
            per = max(1, needed // len(selected_categories))
            topup = _load_category_pool(cfg_for_load, selected_categories, per)
            random.shuffle(topup)
            seed.extend(topup[:needed])
        return "mixed", seed

    raise ValueError(f"Unknown strategy: {strategy}")


def _build_prompt_and_ground_truth(
    task_dict: dict,
    fallback_category: str,
    decomposer: TaskDecomposer,
) -> tuple[str, str, str]:
    real_cat = task_dict.get("category", fallback_category)
    if task_dict.get("prompt"):
        user_prompt = str(task_dict["prompt"])
    else:
        fields = task_to_decomposer_fields(task_dict, real_cat)
        user_prompt = decomposer.build_prompt(real_cat, **fields)

    gt = task_dict.get("ground_truth")
    if gt is None:
        gt = ground_truth_for_category(task_dict, real_cat)
    return real_cat, user_prompt, str(gt or "")


def build_dataset(
    category: str,
    tasks: list[dict],
    cfg: dict,
    target_size: int = 1200,
    output_path: str = "data_loaders/data/finetune_dataset.jsonl",
    provider: str = "ollama",
    trace_model_override: str = None,
    quality_client=None,
    quality_model: str | None = None,
    random_seed: int = 42,
    gen_max_tokens: int | None = None,
    retries: int = 1,
    keep_alive: str | int | None = "20m",
    apply_correctness_filter: bool = True,
) -> int:
    if not tasks:
        logger.error("No tasks available to build dataset.")
        return 0

    random.seed(random_seed)
    client = get_client(cfg, provider=provider)
    decomposer = TaskDecomposer()
    trace_model = trace_model_override or cfg["models"]["trace_gen_model"]
    eval_cfg = cfg["eval"]
    trace_max_tokens = int(
        gen_max_tokens
        or cfg.get("finetune", {}).get("dataset_gen_max_tokens")
        or min(int(eval_cfg["max_tokens"]), 512)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    written = 0
    attempted = 0
    skipped_incorrect = 0
    task_pool = copy.deepcopy(tasks)
    random.shuffle(task_pool)

    repeat = max(1, target_size // max(1, len(task_pool)) + 2)
    task_pool = task_pool * repeat

    logger.info(f"Building fine-tuning dataset: category={category} target={target_size}")
    logger.info(f"  Trace generator model: {trace_model}")
    logger.info(f"  Trace generation max_tokens: {trace_max_tokens}")
    logger.info(f"  Retry on generation errors: {retries}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Task pool size (before repeat): {len(tasks)}")

    with open(output_path, "w", encoding="utf-8") as fout:
        with tqdm(total=target_size, desc=f"Generating {category} traces") as pbar:
            for task_dict in task_pool:
                if written >= target_size:
                    break

                attempted += 1
                real_cat, user_prompt, gt = _build_prompt_and_ground_truth(task_dict, category, decomposer)
                task_id = task_dict.get("id", f"{real_cat}_{attempted:05d}")
                source = task_dict.get("source", "dataset_pool")

                raw = None
                current_max_tokens = trace_max_tokens
                last_error = None
                for attempt in range(retries + 1):
                    try:
                        resp = client.complete(
                            model=trace_model,
                            user_prompt=user_prompt,
                            system_prompt=_TRACE_GEN_SYSTEM,
                            temperature=0.25,
                            max_tokens=current_max_tokens,
                            keep_alive=keep_alive,
                        )
                        raw = resp.content
                        break
                    except Exception as e:
                        last_error = e
                        if "timed out" in str(e).lower():
                            current_max_tokens = max(256, current_max_tokens // 2)
                        if attempt < retries:
                            logger.warning(
                                f"[{task_id}] Generation error (attempt {attempt + 1}/{retries + 1}): {e}"
                            )
                if raw is None:
                    logger.warning(f"[{task_id}] Generation error: {last_error}")
                    continue

                if apply_correctness_filter and gt and not _is_correct_trace(
                    raw,
                    gt,
                    real_cat,
                    quality_client=quality_client,
                    quality_model=quality_model,
                ):
                    skipped_incorrect += 1
                    continue

                example = _format_training_example(
                    system=_TRACE_GEN_SYSTEM,
                    user_prompt=user_prompt,
                    assistant_response=raw.strip(),
                    category=real_cat,
                    source=source,
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
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--strategy",
        choices=["failure", "category", "mixed", "hybrid"],
        default="failure",
        help="failure=from raw failed tasks, category=single-category pool, mixed=balanced pool, hybrid=failures + balanced top-up",
    )
    parser.add_argument("--category", default=None, help="Single target category or 'mixed'")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category list for mixed/hybrid strategy",
    )
    parser.add_argument("--samples", type=int, default=1200, help="Target training examples")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--provider", choices=["groq", "ollama", "openai"], default="ollama")
    parser.add_argument("--trace-model", default=None)
    parser.add_argument("--raw-results", default=None, help="Path to raw_results.json for failure mining")
    parser.add_argument("--max-failure-tasks", type=int, default=None, help="Cap mined failures before expansion")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--quality-provider",
        choices=["groq", "ollama", "openai"],
        default=None,
        help="Optional independent provider for correctness filtering",
    )
    parser.add_argument(
        "--quality-model",
        default=None,
        help="Optional model for independent correctness filtering",
    )
    parser.add_argument(
        "--gen-max-tokens",
        type=int,
        default=None,
        help="Trace generation max tokens (default: finetune.dataset_gen_max_tokens or min(eval.max_tokens, 512))",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retries per generation call after transient errors/timeouts",
    )
    parser.add_argument(
        "--keep-alive",
        default="20m",
        help="Model keep_alive passed to Ollama client (e.g. 20m). Use 'none' to disable.",
    )
    parser.add_argument(
        "--disable-correctness-filter",
        action="store_true",
        help="Accept generated traces without final-answer correctness filtering (useful for balanced general-data generation).",
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

    cfg_for_load = copy.deepcopy(cfg)
    if "datasets" in cfg_for_load and "gsm8k" in cfg_for_load["datasets"]:
        cfg_for_load["datasets"]["gsm8k"]["split"] = "train"
    cfg_for_load["eval"]["samples_per_category"] = max(args.samples, 2000)

    raw_results_path = args.raw_results or os.path.join(cfg["paths"]["outputs"], "raw_results.json")

    category_label, tasks = _build_task_pool(
        strategy=args.strategy,
        cfg=cfg,
        cfg_for_load=cfg_for_load,
        target_size=args.samples,
        category=args.category,
        categories=args.categories,
        raw_results_path=raw_results_path,
        max_failure_tasks=args.max_failure_tasks,
    )

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
        category=category_label,
        tasks=tasks,
        cfg=cfg,
        target_size=args.samples,
        output_path=output,
        provider=args.provider,
        trace_model_override=args.trace_model,
        quality_client=quality_client,
        quality_model=quality_model,
        random_seed=args.random_seed,
        gen_max_tokens=args.gen_max_tokens,
        retries=max(0, args.retries),
        keep_alive=(None if str(args.keep_alive).lower() == "none" else args.keep_alive),
        apply_correctness_filter=not args.disable_correctness_filter,
    )

    logger.info(f"\nFine-tuning dataset complete: {n} examples at {output}")


if __name__ == "__main__":
    main()
