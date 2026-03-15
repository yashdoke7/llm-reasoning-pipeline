"""
experiments/run_comparison_eval.py
Phase 2: Compare baseline vs fine-tuned vs quantized models
using the same evaluation harness from Phase 1.

Loads GGUF models via llama-cpp-python for local inference.
Falls back to Groq API if local inference is unavailable.

Run:
    python experiments/run_comparison_eval.py
    python experiments/run_comparison_eval.py --category multistep_arithmetic --samples 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml

from models.groq_client import get_client
from eval.task_decomposer import TaskDecomposer
from eval.step_evaluator import StepEvaluator
from eval.metrics import MetricsAggregator
from experiments.run_baseline_eval import (
    load_category_tasks,
    task_to_decomposer_fields,
    ground_truth_for_category,
)

logger = logging.getLogger(__name__)


# ── GGUF local inference ──────────────────────────────────────

class GGUFInferenceClient:
    """
    Minimal inference client for GGUF models via llama-cpp-python.
    Drop-in replacement for GroqClient for local model benchmarking.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 35) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "Run: pip install llama-cpp-python"
                " (or: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python for GPU)"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        logger.info(f"Loading GGUF model: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,   # 35 layers on GPU, rest on CPU
            verbose=False,
        )
        self._model_path = model_path

    def complete(
        self,
        model: str,           # ignored (local model)
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """Mimics GroqClient.complete() interface."""
        from dataclasses import dataclass

        @dataclass
        class _FakeResponse:
            content: str

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        output = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = output["choices"][0]["message"]["content"]
        return _FakeResponse(content=content)


# ── Comparison runner ─────────────────────────────────────────

def run_comparison(
    category: str,
    tasks: list[dict],
    models_to_compare: list[dict],   # [{"name": str, "type": "groq"|"gguf", "id_or_path": str}]
    cfg: dict,
    output_path: str,
    use_wandb: bool = True,
) -> dict:
    """
    Run the same evaluation for each model and produce a comparison report.

    models_to_compare format:
        {"name": "Llama3-8B-baseline", "type": "groq", "id_or_path": "llama-3-8b-8192"}
        {"name": "Llama3-3B-LoRA-Q4", "type": "gguf", "id_or_path": "outputs/quantized/model_Q4_K_M.gguf"}
    """
    groq_client = get_client(cfg)
    decomposer = TaskDecomposer()
    step_evaluator = StepEvaluator(
        client=groq_client,
        evaluator_model=cfg["models"]["evaluator_model"],
    )
    aggregator = MetricsAggregator(
        wandb_project=cfg["api"]["wandb_project"],
        use_wandb=use_wandb,
    )
    aggregator.init_run(
        run_name=f"comparison_{category}_{int(time.time())}",
        config={"category": category, "models": [m["name"] for m in models_to_compare]},
    )

    all_results = {}
    eval_cfg = cfg["eval"]

    for model_spec in models_to_compare:
        model_name = model_spec["name"]
        logger.info(f"\nBenchmarking: {model_name}")

        # Get inference client
        if model_spec["type"] == "gguf":
            try:
                inference = GGUFInferenceClient(model_spec["id_or_path"])
                model_id = "local_gguf"
            except Exception as e:
                logger.error(f"Failed to load GGUF {model_spec['id_or_path']}: {e}")
                continue
        else:
            inference = groq_client
            model_id = model_spec["id_or_path"]

        model_results = []
        for i, task_dict in enumerate(tasks):
            task_id = task_dict.get("id", f"{category}_{i:04d}")
            fields = task_to_decomposer_fields(task_dict, category)
            gt = ground_truth_for_category(task_dict, category)

            decomposed = decomposer.build(
                task_id=task_id,
                category=category,
                ground_truth=gt,
                **fields,
            )
            try:
                resp = inference.complete(
                    model=model_id,
                    user_prompt=decomposed.prompt,
                    system_prompt=decomposer.get_system_prompt(category),
                    temperature=eval_cfg["temperature"],
                    max_tokens=eval_cfg["max_tokens"],
                )
                decomposed = decomposer.parse_response(decomposed, resp.content)
            except Exception as e:
                logger.error(f"[{task_id}] Inference error: {e}")
                continue

            task_eval = step_evaluator.evaluate(decomposed, model_name=model_name)
            aggregator.add_task_eval(task_eval)

            model_results.append(step_evaluator.to_dict(task_eval))
            if (i + 1) % 10 == 0:
                logger.info(f"  {model_name}: {i+1}/{len(tasks)}")

        all_results[model_name] = model_results

    # Finalize and save
    metrics = aggregator.finalize(output_path=output_path)
    aggregator.finish_run()

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Model':<30} {'Step Fail':>10} {'Accuracy':>10} {'Error Prop':>12}")
    logger.info("-" * 70)
    for m in metrics["model_summary"]:
        logger.info(
            f"{m['model']:<30} "
            f"{m['overall_step_failure_rate']:>10.3f} "
            f"{m['overall_final_accuracy']:>10.3f} "
            f"{m['overall_error_propagation_rate']:>12.3f}"
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 comparison evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config_path = _ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Determine category from failure report
    category = args.category
    if not category:
        failure_path = cfg["paths"]["failure_report"]
        if os.path.exists(failure_path):
            with open(failure_path) as f:
                fr = json.load(f)
            category = fr.get("finetune_target_category", "multistep_arithmetic")
        else:
            category = "multistep_arithmetic"

    # Load tasks (use test split for eval)
    cfg["eval"]["samples_per_category"] = args.samples
    tasks = load_category_tasks(category, cfg)

    # Define models to compare
    quant_dir = cfg["finetune"]["quantization"]["output_dir"]
    models_to_compare = [
        # Baseline from Groq API
        {
            "name": "Llama3.1-8B-baseline",
            "type": "groq",
            "id_or_path": "llama-3.1-8b-instant",
        },
    ]

    # Add quantized models if they exist
    for quant_fmt in ["Q4_K_M", "Q8_0"]:
        gguf_path = os.path.join(quant_dir, f"model_{quant_fmt.lower()}.gguf")
        if os.path.exists(gguf_path):
            models_to_compare.append({
                "name": f"LoRA-finetuned-{quant_fmt}",
                "type": "gguf",
                "id_or_path": gguf_path,
            })
        else:
            logger.warning(f"GGUF not found, skipping: {gguf_path}")

    output_path = os.path.join(cfg["paths"]["outputs"], "comparison_results.json")
    run_comparison(
        category=category,
        tasks=tasks,
        models_to_compare=models_to_compare,
        cfg=cfg,
        output_path=output_path,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
