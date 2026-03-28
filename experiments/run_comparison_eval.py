"""
experiments/run_comparison_eval.py
Phase 2 comparison evaluation (base vs fine-tuned / quantized).

Improvements over previous version:
  - Uses independent judge for final-answer checking (no self-grading bias)
  - Supports multiple categories in one run
  - Supports custom JSONL dataset (for physics-specific evaluation)
  - Writes timestamped comparison outputs + latest aliases
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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
from eval.hallucination_scorer import HallucinationScorer
from eval.drift_detector import DriftDetector
from experiments.run_baseline_eval import (
    load_category_tasks,
    task_to_decomposer_fields,
    ground_truth_for_category,
)

logger = logging.getLogger(__name__)


@dataclass
class SimpleResponse:
    content: str


class GGUFInferenceClient:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 35) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "Run: pip install llama-cpp-python"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        logger.info(f"Loading GGUF model: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def complete(
        self,
        model: str,  # ignored
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs,
    ) -> SimpleResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        output = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return SimpleResponse(content=output["choices"][0]["message"]["content"])


class HFInferenceClient:
    def __init__(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers/torch not installed. Run: pip install transformers torch")

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
    ) -> SimpleResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

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
        return SimpleResponse(content=content)


def _load_custom_jsonl(path: str, category: str, max_samples: int) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Custom dataset not found: {path}")
    tasks: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and len(tasks) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = (row.get("question") or row.get("problem") or row.get("prompt") or "").strip()
            if not question:
                continue
            tasks.append(
                {
                    "id": row.get("id", f"{category}_{i:04d}"),
                    "category": category,
                    "question": question,
                    "ground_truth": (
                        row.get("ground_truth")
                        or row.get("final_answer")
                        or row.get("answer")
                        or ""
                    ),
                }
            )
    logger.info(f"Loaded {len(tasks)} custom tasks from {path} (category={category})")
    return tasks


def _safe_task_fields(task_dict: dict, category: str) -> dict:
    try:
        return task_to_decomposer_fields(task_dict, category)
    except Exception:
        # Fallback for custom categories that still expose question text
        if "question" in task_dict:
            return {"question": task_dict["question"]}
        if "prompt" in task_dict:
            return {"question": task_dict["prompt"]}
        raise


def _default_output_path(outputs_dir: str, categories: list[str], run_suffix: str) -> str:
    if len(categories) == 1:
        return os.path.join(outputs_dir, f"comparison_{categories[0]}_results_{run_suffix}.json")
    return os.path.join(outputs_dir, f"comparison_results_{run_suffix}.json")


def run_comparison(
    categories: list[str],
    tasks_by_category: dict[str, list[dict]],
    models_to_compare: list[dict],
    cfg: dict,
    output_path: str,
    judge_provider: str,
    judge_model: str,
    use_wandb: bool,
    with_diagnostics: bool,
) -> tuple[dict, dict]:
    judge_client = get_client(cfg, provider=judge_provider)

    decomposer = TaskDecomposer()
    step_evaluator = StepEvaluator(
        client=judge_client,
        evaluator_model=judge_model,
        judge_client=judge_client,
        judge_model=judge_model,
    )
    hall_scorer = HallucinationScorer(client=judge_client, model=judge_model) if with_diagnostics else None
    drift_detector = DriftDetector(client=judge_client, model=judge_model) if with_diagnostics else None

    aggregator = MetricsAggregator(
        wandb_project=cfg["api"]["wandb_project"],
        use_wandb=use_wandb,
    )
    aggregator.init_run(
        run_name=f"comparison_{'_'.join(categories)}_{int(time.time())}",
        config={"categories": categories, "models": [m["name"] for m in models_to_compare]},
    )

    run_start = time.time()
    all_results: dict[str, dict[str, list[dict]]] = {}
    inference_clients: dict[str, object] = {}
    eval_cfg = cfg["eval"]
    attempted = 0
    completed = 0

    for model_spec in models_to_compare:
        model_name = model_spec["name"]
        logger.info(f"\nBenchmarking: {model_name}")
        all_results[model_name] = {}

        if model_spec["type"] == "gguf":
            try:
                inference = GGUFInferenceClient(model_spec["id_or_path"])
                model_id = "local_gguf"
            except Exception as e:
                logger.error(f"Failed to load GGUF model {model_spec['id_or_path']}: {e}")
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

        for category in categories:
            tasks = tasks_by_category.get(category, [])
            all_results[model_name][category] = []
            logger.info(f"  Category: {category} ({len(tasks)} tasks)")

            for i, task_dict in enumerate(tasks):
                attempted += 1
                task_id = task_dict.get("id", f"{category}_{i:04d}")
                fields = _safe_task_fields(task_dict, category)
                gt = task_dict.get("ground_truth") or ground_truth_for_category(task_dict, category)

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
                    task_eval = step_evaluator.evaluate(decomposed, model_name=model_name)
                    if with_diagnostics and hall_scorer and drift_detector:
                        hall_scores = hall_scorer.score_steps(task_id, decomposed.steps, decomposed.prompt[:400])
                        drift_result = drift_detector.detect(task_id, decomposed.prompt, decomposed.steps, category)
                        aggregator.add_task_eval(task_eval, hall_scores, drift_result)
                    else:
                        aggregator.add_task_eval(task_eval)
                    all_results[model_name][category].append(step_evaluator.to_dict(task_eval))
                    completed += 1
                except Exception as e:
                    logger.error(f"[{task_id}] Inference/eval error: {e}")
                    continue

                if (i + 1) % 10 == 0:
                    logger.info(f"    {model_name} [{category}] {i + 1}/{len(tasks)}")

    metrics = aggregator.finalize(output_path=output_path)
    aggregator.finish_run()

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Model':<30} {'Step Fail':>10} {'Accuracy':>10} {'Error Prop':>12}")
    logger.info("-" * 70)
    for m in metrics.get("model_summary", []):
        logger.info(
            f"{m['model']:<30} "
            f"{m['overall_step_failure_rate']:>10.3f} "
            f"{m['overall_final_accuracy']:>10.3f} "
            f"{m['overall_error_propagation_rate']:>12.3f}"
        )

    manifest = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "duration_seconds": round(time.time() - run_start, 2),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "categories": categories,
        "models": models_to_compare,
        "tasks_attempted": attempted,
        "tasks_completed": completed,
        "tasks_missing": max(0, attempted - completed),
        "with_diagnostics": with_diagnostics,
        "output_eval_results": output_path,
    }
    return metrics, {"raw_results": all_results, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 comparison evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--category", default=None, help="Single category (legacy flag)")
    parser.add_argument("--categories", nargs="+", default=None, help="Evaluate multiple categories")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--base-model", default="qwen2.5:3b")
    parser.add_argument("--base-type", choices=["ollama", "groq", "openai"], default="ollama")
    parser.add_argument("--finetuned-model", default=None)
    parser.add_argument("--judge-provider", choices=["ollama", "groq", "openai"], default="ollama")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--output", default=None, help="Explicit eval output JSON path")
    parser.add_argument("--with-diagnostics", action="store_true", help="Compute hallucination/drift metrics too")
    parser.add_argument("--custom-dataset", default=None, help="Custom JSONL dataset path (e.g. physics_eval.jsonl)")
    parser.add_argument("--custom-category", default="physics_reasoning", help="Category label for custom dataset")
    args = parser.parse_args()

    config_path = _ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    outputs_dir = cfg["paths"]["outputs"]
    os.makedirs(outputs_dir, exist_ok=True)

    if args.categories:
        categories = args.categories
    elif args.category:
        categories = [args.category]
    elif args.custom_dataset:
        categories = [args.custom_category]
    else:
        failure_path = cfg["paths"]["failure_report"]
        if os.path.exists(failure_path):
            with open(failure_path, encoding="utf-8") as f:
                fr = json.load(f)
            categories = [fr.get("finetune_target_category", "multistep_arithmetic")]
        else:
            categories = ["multistep_arithmetic"]

    cfg["eval"]["samples_per_category"] = args.samples
    tasks_by_category: dict[str, list[dict]] = {}
    if args.custom_dataset:
        cat = args.custom_category
        tasks_by_category[cat] = _load_custom_jsonl(args.custom_dataset, cat, args.samples)
        categories = [cat]
    else:
        for category in categories:
            tasks_by_category[category] = load_category_tasks(category, cfg)

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

    finetuned_path = args.finetuned_model
    if not finetuned_path:
        q4 = os.path.join(quant_dir, "model_q4_k_m.gguf")
        fp16 = os.path.join(quant_dir, "model_fp16.gguf")
        hf_dir = os.path.join(cfg["paths"]["outputs"], "merged_model")
        if os.path.exists(q4):
            finetuned_path = q4
        elif os.path.exists(fp16):
            finetuned_path = fp16
        elif os.path.isdir(hf_dir):
            finetuned_path = hf_dir

    if finetuned_path and os.path.exists(finetuned_path):
        if os.path.isdir(finetuned_path):
            models_to_compare.append(
                {
                    "name": "Qwen-3B-LoRA-HF",
                    "type": "hf",
                    "id_or_path": finetuned_path,
                }
            )
        else:
            lower = finetuned_path.lower()
            quant_level = "Q4_K_M" if "q4_k_m" in lower else "Q8_0" if "q8_0" in lower else "FP16"
            models_to_compare.append(
                {
                    "name": f"Qwen-3B-LoRA-{quant_level}",
                    "type": "gguf",
                    "id_or_path": finetuned_path,
                }
            )
        logger.info(f"Using finetuned model: {finetuned_path}")
    else:
        logger.warning("No finetuned model found. Running base-only comparison.")

    if not models_to_compare:
        raise RuntimeError("No models to compare. Check base/finetuned model arguments.")

    if args.judge_model:
        judge_model = args.judge_model
    elif args.judge_provider == "ollama":
        judge_model = cfg["models"].get("local_judge_model", "qwen2.5:14b")
    else:
        judge_model = cfg["models"]["evaluator_model"]

    run_suffix = f"{'_'.join(categories)}_{int(time.time())}"
    output_path = args.output or _default_output_path(outputs_dir, categories, run_suffix)
    metrics, extras = run_comparison(
        categories=categories,
        tasks_by_category=tasks_by_category,
        models_to_compare=models_to_compare,
        cfg=cfg,
        output_path=output_path,
        judge_provider=args.judge_provider,
        judge_model=judge_model,
        use_wandb=not args.no_wandb,
        with_diagnostics=args.with_diagnostics,
    )

    raw_path = os.path.join(outputs_dir, f"comparison_raw_{run_suffix}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(extras["raw_results"], f, indent=2)

    manifest = extras["manifest"]
    manifest["output_raw_results"] = raw_path
    manifest_path = os.path.join(outputs_dir, f"comparison_manifest_{run_suffix}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Update latest aliases only for standard runs (no explicit custom output path).
    if args.output is None:
        latest_eval = os.path.join(outputs_dir, "comparison_results.json")
        with open(output_path, encoding="utf-8") as fsrc, open(latest_eval, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())

        latest_manifest = os.path.join(outputs_dir, "comparison_manifest.json")
        with open(manifest_path, encoding="utf-8") as fsrc, open(latest_manifest, "w", encoding="utf-8") as fdst:
            fdst.write(fsrc.read())

        if len(categories) == 1:
            cat_alias = os.path.join(outputs_dir, f"comparison_{categories[0]}_results.json")
            with open(output_path, encoding="utf-8") as fsrc, open(cat_alias, "w", encoding="utf-8") as fdst:
                fdst.write(fsrc.read())

    logger.info(f"Saved comparison metrics:  {output_path}")
    logger.info(f"Saved comparison raw:      {raw_path}")
    logger.info(f"Saved comparison manifest: {manifest_path}")


if __name__ == "__main__":
    main()
