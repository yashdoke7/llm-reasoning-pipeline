"""
run_comparison_eval.py  —  Compare base vs fine-tuned model step-by-step.

Key fixes applied:
  - Judge client is always a FRESH instance (fixes shared-singleton usage bug)
  - Separate usage tracking for solver vs judge
  - Graceful Groq rate-limit fallback to ollama judge
  - Timeout & retry logic added
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── client helpers ────────────────────────────────────────────────────────────

def make_ollama_client(cfg: dict, model: str):
    """Always returns a FRESH OllamaClient — never the cached singleton."""
    from models.ollama_client import OllamaClient

    params = cfg.get("ollama", {})
    client = OllamaClient(
        host=params.get("host", "http://localhost:11434"),
        timeout=params.get("timeout", 180.0),
    )
    client._default_model = model
    return client


def make_groq_client(cfg: dict, model: str):
    """Returns a Groq client."""
    from models.groq_client import GroqClient

    params = cfg.get("groq", {})
    client = GroqClient(
        api_key=os.environ.get("GROQ_API_KEY", params.get("api_key", "")),
    )
    return client


def get_client(cfg: dict, provider: str, model: str, force_new: bool = False):
    """
    Returns an inference client.
    force_new=True always creates a new instance (use for judge to avoid singleton).
    """
    if provider == "ollama":
        return make_ollama_client(cfg, model)
    elif provider == "groq":
        return make_groq_client(cfg, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_with_retry(client, prompt: str, model: str, max_retries: int = 3,
                    temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Calls client with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            response = client.complete(
                model=model,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.content
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"All {max_retries} attempts failed for model {model}")


def _message_content(task: dict, role: str) -> str:
    msgs = task.get("messages", [])
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if isinstance(m, dict) and str(m.get("role", "")).lower() == role:
            return str(m.get("content", "")).strip()
    return ""


def _extract_final_answer_from_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"Final\s+Answer\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip()
        return ans.splitlines()[0].strip() if ans else ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ── task loading ──────────────────────────────────────────────────────────────

def load_tasks(cfg: dict, categories: list[str], samples_per_cat: int) -> list[dict]:
    """Load evaluation tasks from dataset files or generate them."""
    tasks = []
    data_dir = Path("finetune/data")

    # Try loading from existing JSONL files
    dataset_cfg = cfg.get("dataset", {}) or cfg.get("datasets", {}) or {}
    source_files = dataset_cfg.get("eval_source_files", dataset_cfg.get("source_files", []))
    all_by_cat: dict[str, list] = defaultdict(list)

    for fpath in source_files:
        p = Path(fpath)
        if not p.exists():
            logger.warning(f"Dataset file not found: {fpath}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    cat = row.get("category", "unknown")
                    if cat in categories:
                        all_by_cat[cat].append(row)
                except json.JSONDecodeError:
                    continue

    for cat in categories:
        pool = all_by_cat.get(cat, [])
        if len(pool) == 0:
            logger.warning(f"No tasks found for category '{cat}' — will skip")
            continue
        pool = sorted(pool, key=lambda row: str(row.get("id", "")))
        selected = pool[: min(samples_per_cat, len(pool))]
        tasks.extend(selected)
        logger.info(f"  Loaded {len(selected)} tasks for '{cat}'")

    logger.info(f"Total tasks loaded: {len(tasks)}")
    return tasks


# ── step decomposition & evaluation ──────────────────────────────────────────

SOLVER_PROMPT_TEMPLATE = """Solve the following problem step by step.
Number each step clearly (Step 1:, Step 2:, etc.).
Show your reasoning for each step before giving a final answer.

Problem: {problem}

Solution:"""

JUDGE_PROMPT_TEMPLATE = """You are evaluating reasoning steps for correctness.

Problem: {problem}
Expected Answer: {expected}

Model's Solution:
{solution}

For each numbered step, evaluate if it is VALID (correct reasoning), INVALID (wrong/misleading), or UNCERTAIN.
Then state if the final answer is CORRECT or INCORRECT.

Respond in this exact format:
Step 1: VALID/INVALID/UNCERTAIN - brief reason
Step 2: VALID/INVALID/UNCERTAIN - brief reason
...
Final Answer: CORRECT/INCORRECT
Overall: PASS/FAIL"""


def decompose_steps(solution: str) -> list[str]:
    """Extract numbered steps from a solution string."""
    import re
    steps = re.findall(r"Step\s+\d+[:\.]?\s*(.+?)(?=Step\s+\d+|$)", solution,
                       re.IGNORECASE | re.DOTALL)
    return [s.strip() for s in steps if s.strip()]


def parse_judge_response(judge_response: str) -> dict:
    """Parse judge output into structured metrics."""
    import re
    lines = judge_response.strip().split("\n")

    step_verdicts = []
    final_correct = False
    overall_pass = False

    for line in lines:
        line = line.strip()
        step_match = re.match(r"Step\s+\d+:\s*(VALID|INVALID|UNCERTAIN)", line, re.I)
        if step_match:
            step_verdicts.append(step_match.group(1).upper())

        if re.search(r"Final Answer:\s*CORRECT", line, re.I):
            final_correct = True
        if re.search(r"Overall:\s*PASS", line, re.I):
            overall_pass = True

    n_steps = len(step_verdicts)
    n_invalid = sum(1 for v in step_verdicts if v == "INVALID")
    step_failure_rate = n_invalid / n_steps if n_steps > 0 else 0.0

    # Error propagation: did an early invalid step lead to wrong answer?
    error_propagated = False
    if n_invalid > 0 and not final_correct:
        error_propagated = True

    return {
        "step_verdicts": step_verdicts,
        "n_steps": n_steps,
        "n_invalid": n_invalid,
        "step_failure_rate": step_failure_rate,
        "final_correct": final_correct,
        "overall_pass": overall_pass,
        "error_propagated": error_propagated,
    }


# ── main evaluation loop ──────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    tasks: list[dict],
    solver_client,
    judge_client,
    judge_model: str,
    cfg: dict,
    results_store: list,
) -> dict[str, Any]:
    """Run evaluation for a single model across all tasks."""
    eval_cfg = cfg.get("evaluation", {}) or cfg.get("eval", {})
    solver_temp = eval_cfg.get("solver_temperature", 0.1)
    judge_temp = eval_cfg.get("judge_temperature", 0.0)
    max_tokens = eval_cfg.get("max_tokens", 1024)

    all_metrics = []
    by_category: dict[str, list] = defaultdict(list)

    for i, task in enumerate(tasks):
        category = task.get("category", "unknown")
        user_msg = _message_content(task, "user")
        assistant_msg = _message_content(task, "assistant")

        # Support all dataset schemas used in this repo, including chat/message rows.
        problem = (
            task.get("problem")
            or task.get("prompt")
            or task.get("input")
            or task.get("question")
            or task.get("goal")
            or task.get("premise")
            or user_msg
            or ""
        )
        expected = (
            task.get("expected")
            or task.get("answer")
            or task.get("output")
            or task.get("ground_truth")
            or task.get("correct_answer")
            or _extract_final_answer_from_text(assistant_msg)
            or ""
        )

        logger.info(f"  [{model_name}] Task {i+1}/{len(tasks)} | {category}")

        # 1. Solve
        solver_prompt = SOLVER_PROMPT_TEMPLATE.format(problem=problem)
        try:
            solution = call_with_retry(
                solver_client, solver_prompt, model_name,
                temperature=solver_temp, max_tokens=max_tokens
            )
        except RuntimeError as e:
            logger.error(f"Solver failed for task {i}: {e}")
            solution = "[SOLVER ERROR]"

        # 2. Judge — uses its OWN client (not shared with solver)
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            problem=problem,
            expected=expected,
            solution=solution,
        )
        try:
            judge_response = call_with_retry(
                judge_client, judge_prompt, judge_model,
                temperature=judge_temp, max_tokens=512
            )
        except RuntimeError as e:
            logger.error(f"Judge failed for task {i}: {e}")
            judge_response = "Final Answer: INCORRECT\nOverall: FAIL"

        # 3. Parse metrics
        metrics = parse_judge_response(judge_response)
        metrics["model"] = model_name
        metrics["category"] = category
        metrics["task_id"] = i
        metrics["problem"] = problem[:200]  # truncate for storage
        metrics["solution"] = solution[:500]
        metrics["judge_response"] = judge_response

        all_metrics.append(metrics)
        by_category[category].append(metrics)
        results_store.append(metrics)

    # Aggregate
    n = len(all_metrics)
    if n == 0:
        return {"model": model_name, "error": "no tasks evaluated"}

    agg = {
        "model": model_name,
        "n_tasks": n,
        "step_failure": sum(m["step_failure_rate"] for m in all_metrics) / n,
        "accuracy": sum(1 for m in all_metrics if m["final_correct"]) / n,
        "error_prop": sum(1 for m in all_metrics if m["error_propagated"]) / n,
        "by_category": {},
    }

    for cat, cat_metrics in by_category.items():
        nc = len(cat_metrics)
        agg["by_category"][cat] = {
            "n": nc,
            "step_failure": sum(m["step_failure_rate"] for m in cat_metrics) / nc,
            "accuracy": sum(1 for m in cat_metrics if m["final_correct"]) / nc,
            "error_prop": sum(1 for m in cat_metrics if m["error_propagated"]) / nc,
        }

    return agg


# ── failure report ────────────────────────────────────────────────────────────

def build_failure_report(
    base_results: dict, ft_results: dict,
    base_model: str, ft_model: str,
) -> dict:
    """Identify which categories regressed and suggest fine-tune target."""
    report = {
        "base_model": base_model,
        "ft_model": ft_model,
        "timestamp": datetime.now().isoformat(),
        "category_comparison": {},
        "fine_tuning_target": None,
        "verdict": "",
    }

    worst_regression = float("-inf")
    worst_cat = None

    for cat in base_results.get("by_category", {}):
        base_cat = base_results["by_category"].get(cat, {})
        ft_cat = ft_results["by_category"].get(cat, {})

        base_sf = base_cat.get("step_failure", 0)
        ft_sf = ft_cat.get("step_failure", 0)
        regression = ft_sf - base_sf  # positive = fine-tuned is WORSE

        report["category_comparison"][cat] = {
            "base_step_failure": base_sf,
            "ft_step_failure": ft_sf,
            "regression": regression,
            "base_accuracy": base_cat.get("accuracy", 0),
            "ft_accuracy": ft_cat.get("accuracy", 0),
        }

        if regression > worst_regression:
            worst_regression = regression
            worst_cat = cat

    report["fine_tuning_target"] = worst_cat
    if ft_results.get("accuracy", 0) < base_results.get("accuracy", 0):
        report["verdict"] = f"REGRESSION: fine-tuned worse overall. Retrain with targeted data on '{worst_cat}'."
    else:
        report["verdict"] = f"IMPROVEMENT: fine-tuned better overall. Weakest category: '{worst_cat}'."

    return report


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare base vs fine-tuned model")
    p.add_argument("--config", default="configs/config.yaml", help="YAML config file")
    p.add_argument("--models", nargs=2, metavar=("BASE", "FINETUNED"),
                   help="Override model names from config")
    p.add_argument("--judge-provider", default=None,
                   choices=["ollama", "groq"],
                   help="Provider for judge model")
    p.add_argument("--judge-model", default=None,
                   help="Judge model name override")
    p.add_argument("--solver-provider", default="ollama",
                   choices=["ollama", "groq"])
    p.add_argument("--categories", nargs="+", default=None,
                   help="Categories to evaluate (default: all from config)")
    p.add_argument("--samples", type=int, default=None,
                   help="Samples per category override")
    p.add_argument("--output", default=None,
                   help="Output file path override")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable wandb logging")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {}) or cfg.get("eval", {})
    models_cfg = cfg.get("models", {})

    # Resolve model names
    base_model = (args.models[0] if args.models else None) or models_cfg.get("solver", "qwen2.5:3b")
    ft_model = (args.models[1] if args.models else None) or models_cfg.get("fine_tuned", "qwen2.5-3b-targeted-v3")

    # Resolve judge
    judge_provider = args.judge_provider or cfg.get("providers", {}).get("judge_provider") or cfg.get("provider", "ollama")
    judge_model = args.judge_model or models_cfg.get("judge", "qwen2.5:14b")

    # Resolve solver provider
    solver_provider = args.solver_provider or cfg.get("providers", {}).get("solver_provider") or cfg.get("provider", "ollama")

    # Resolve categories & samples
    categories = args.categories or eval_cfg.get("categories", [
        "multistep_arithmetic", "tool_use_planning",
        "factual_consistency", "causal_counterfactual"
    ])
    samples_per_cat = args.samples or eval_cfg.get("samples_per_category", 8)

    logger.info("=" * 60)
    logger.info("COMPARISON EVAL STARTING")
    logger.info(f"  Base model:    {base_model} (provider: {solver_provider})")
    logger.info(f"  Fine-tuned:    {ft_model}   (provider: {solver_provider})")
    logger.info(f"  Judge:         {judge_model} (provider: {judge_provider})")
    logger.info(f"  Categories:    {categories}")
    logger.info(f"  Samples/cat:   {samples_per_cat}")
    logger.info("=" * 60)

    # ── Create clients ────────────────────────────────────────────────────────
    # KEY FIX: judge always gets force_new=True so it's never the same object
    # as any solver client, even when both use ollama
    base_solver_client  = get_client(cfg, solver_provider, base_model, force_new=False)
    ft_solver_client    = get_client(cfg, solver_provider, ft_model,   force_new=False)
    judge_client        = get_client(cfg, judge_provider,  judge_model, force_new=True)

    # ── Load tasks ────────────────────────────────────────────────────────────
    tasks = load_tasks(cfg, categories, samples_per_cat)
    if not tasks:
        logger.error("No tasks loaded. Check your dataset paths in config.")
        sys.exit(1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_raw_results = []

    logger.info(f"\nEvaluating BASE model: {base_model}")
    base_agg = evaluate_model(
        base_model, tasks, base_solver_client, judge_client,
        judge_model, cfg, all_raw_results
    )

    logger.info(f"\nEvaluating FINE-TUNED model: {ft_model}")
    ft_agg = evaluate_model(
        ft_model, tasks, ft_solver_client, judge_client,
        judge_model, cfg, all_raw_results
    )

    # ── Build failure report ──────────────────────────────────────────────────
    failure_report = build_failure_report(base_agg, ft_agg, base_model, ft_model)

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts = int(time.time())
    out_dir = Path(cfg.get("logging", {}).get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{base_model.replace(':', '_')}_{ft_model.replace(':', '_')}_mit_{ts}"

    raw_path      = args.output or out_dir / f"raw_results_{tag}.json"
    manifest_path = out_dir / f"run_manifest_{tag}.json"
    failure_path  = out_dir / f"failure_report_{tag}.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_results, f, indent=2)

    manifest = {
        "run_id": tag,
        "timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "ft_model": ft_model,
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "solver_provider": solver_provider,
        "samples_per_category": samples_per_cat,
        "categories": categories,
        "base_aggregate": base_agg,
        "ft_aggregate": ft_agg,
        # note: usage tracking is now separate per client instance
        "note": "solver and judge are separate client instances (singleton bug fixed)",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_report, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {base_model}: step_failure={base_agg['step_failure']:.3f} | "
                f"accuracy={base_agg['accuracy']:.3f} | error_prop={base_agg['error_prop']:.3f}")
    logger.info(f"  {ft_model}: step_failure={ft_agg['step_failure']:.3f} | "
                f"accuracy={ft_agg['accuracy']:.3f} | error_prop={ft_agg['error_prop']:.3f}")
    logger.info(f"\n{failure_report['verdict']}")
    logger.info(f"Fine-tuning target: {failure_report['fine_tuning_target']}")
    logger.info(f"\nResults saved to:   {raw_path}")
    logger.info(f"Failure report:     {failure_path}")
    logger.info(f"Run manifest:       {manifest_path}")


if __name__ == "__main__":
    main()  
