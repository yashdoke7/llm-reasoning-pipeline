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

from models import get_client
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


class HFInferenceClient:
    """
    Minimal inference client for merged HuggingFace models.
    Useful fallback when llama-cpp-python is unavailable on Windows.
    """

    def __init__(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers/torch not installed. Run: pip install transformers torch"
            )

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"HF model directory not found: {model_path}")

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def complete(
        self,
        model: str,  # ignored
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs,
    ):
        from dataclasses import dataclass

        @dataclass
        class _FakeResponse:
            content: str

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Prefer native chat templates when available.
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._torch.cuda.is_available():
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        out = self._model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _FakeResponse(content=content)


# ── Comparison runner ─────────────────────────────────────────

def run_comparison(
    category: str,
    tasks: list[dict],
    models_to_compare: list[dict],   # [{"name": str, "type": "ollama"|"groq"|"openai"|"gguf", "id_or_path": str}]
    cfg: dict,
    output_path: str,
    judge_provider: str = "ollama",
    judge_model: Optional[str] = None,
    use_wandb: bool = True,
) -> dict:
    """
    Run the same evaluation for each model and produce a comparison report.

    models_to_compare format:
        {"name": "Llama3-8B-baseline", "type": "groq", "id_or_path": "llama-3-8b-8192"}
        {"name": "Llama3-3B-LoRA-Q4", "type": "gguf", "id_or_path": "outputs/quantized/model_Q4_K_M.gguf"}
    """
    judge_client = get_client(cfg, provider=judge_provider)
    if not judge_model:
        if judge_provider == "ollama":
            judge_model = cfg["models"].get("local_judge_model", "qwen2.5:14b")
        else:
            judge_model = cfg["models"]["evaluator_model"]
    
    decomposer = TaskDecomposer()
    step_evaluator = StepEvaluator(
        client=judge_client,
        evaluator_model=judge_model,
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
    inference_clients: dict[str, object] = {}
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
        elif model_spec["type"] == "hf":
            try:
                inference = HFInferenceClient(model_spec["id_or_path"])
                model_id = "local_hf"
            except Exception as e:
                logger.error(f"Failed to load HF model {model_spec['id_or_path']}: {e}")
                continue
        else:
            provider = model_spec["type"]
            if provider not in inference_clients:
                inference_clients[provider] = get_client(cfg, provider=provider)
            inference = inference_clients[provider]
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
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip baseline model and evaluate only finetuned model")
    parser.add_argument("--base-model", default="qwen2.5:3b", 
                        help="Base model for comparison (default: qwen2.5:3b for Ollama, or llama-3.1-8b-instant for Groq)")
    parser.add_argument("--base-type", choices=["ollama", "groq"], default="ollama",
                        help="Provider for base model")
    parser.add_argument("--finetuned-model", default=None,
                        help="Explicit path to finetuned GGUF model (e.g., outputs/quantized/model_q4_k_m.gguf). If not provided, auto-detects Q4_K_M if available.")
    parser.add_argument("--judge-provider", choices=["ollama", "groq", "openai"], default="ollama",
                        help="Provider for step evaluator judge (default: ollama)")
    parser.add_argument("--judge-model", default=None,
                        help="Model for step evaluator judge (default: local_judge_model from config)")
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
    models_to_compare = []
    if not args.skip_base:
        models_to_compare.append(
            {
                "name": f"{args.base_model}-base",
                "type": args.base_type,
                "id_or_path": args.base_model,
            }
        )

    # Add finetuned model (explicit or auto-detect)
    finetuned_path = args.finetuned_model
    if not finetuned_path:
        # Auto-detect Q4_K_M (best quality/size tradeoff)
        finetuned_path = os.path.join(quant_dir, "model_q4_k_m.gguf")
    
    if os.path.exists(finetuned_path):
        if os.path.isdir(finetuned_path):
            models_to_compare.append({
                "name": "Qwen-3B-LoRA-HF",
                "type": "hf",
                "id_or_path": finetuned_path,
            })
        else:
            quant_level = "Q4_K_M" if "q4_k_m" in finetuned_path.lower() else "Q8_0" if "q8_0" in finetuned_path.lower() else "FP16"
            models_to_compare.append({
                "name": f"Qwen-3B-LoRA-{quant_level}",
                "type": "gguf",
                "id_or_path": finetuned_path,
            })
        logger.info(f"Using finetuned model: {finetuned_path}")
    else:
        logger.error(f"Finetuned model not found at: {finetuned_path}")
        logger.info("Skipping finetuned model. Run finetune/train_lora.py → merge_adapter.py → quantize.py first.")

    output_path = os.path.join(cfg["paths"]["outputs"], "comparison_results.json")
    run_comparison(
        category=category,
        tasks=tasks,
        models_to_compare=models_to_compare,
        cfg=cfg,
        output_path=output_path,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
