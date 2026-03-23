"""
experiments/run_baseline_eval.py
Phase 1: Full baseline evaluation across all models and task categories.

Run:
    python experiments/run_baseline_eval.py
    python experiments/run_baseline_eval.py --categories multistep_arithmetic factual_consistency
    python experiments/run_baseline_eval.py --models llama-3-8b-8192 --samples 20 --no-wandb
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import yaml

from models.groq_client import GroqClient, DailyLimitExhaustedError
from models import get_client
from eval.task_decomposer import TaskDecomposer
from eval.step_evaluator import StepEvaluator
from eval.hallucination_scorer import HallucinationScorer
from eval.drift_detector import DriftDetector
from eval.metrics import MetricsAggregator
from mitigation.retriever import WikipediaRetriever
from mitigation.regrounder import Regrounder, InterventionMode
from data_loaders.gsm8k_loader import load_gsm8k, sample_to_dict as gsm8k_to_dict
from data_loaders.crass_loader import load_crass, sample_to_dict as crass_to_dict
from data_loaders.toolbench_loader import load_toolbench, sample_to_dict as tool_to_dict, format_tools_for_prompt
from data_loaders.factual_synthetic import load_factual_synthetic, sample_to_dict as factual_to_dict


# ── Logging setup ─────────────────────────────────────────────

def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(levelname)s | %(message)s"),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(cfg["paths"]["logs"], "baseline_eval.log"),
                mode="a",
            ),
        ],
    )


logger = logging.getLogger(__name__)


def _dataset_source_summary(category: str, cfg: dict) -> dict:
    ds_cfg = cfg.get("datasets", {})
    source = {"category": category, "kind": "unknown", "path": None, "exists": None}

    if category == "multistep_arithmetic":
        source["kind"] = "huggingface_gsm8k"
        source["path"] = ds_cfg.get("gsm8k", {}).get("cache_dir")
        source["exists"] = os.path.exists(source["path"]) if source["path"] else None
        return source

    if category == "causal_counterfactual":
        p = ds_cfg.get("crass", {}).get("path")
        source["path"] = p
        if p and os.path.exists(p):
            source["kind"] = "counterfactual_file"
            source["exists"] = True
        else:
            source["kind"] = "counterfactual_fallback"
            source["exists"] = False
        return source

    if category == "tool_use_planning":
        p = ds_cfg.get("toolbench", {}).get("path")
        source["path"] = p
        if p and os.path.exists(p):
            source["kind"] = "tooluse_generated_file"
            source["exists"] = True
        else:
            source["kind"] = "tooluse_builtin_fallback"
            source["exists"] = False
        return source

    if category == "factual_consistency":
        p = ds_cfg.get("factual_synthetic", {}).get("path")
        source["path"] = p
        if p and os.path.exists(p):
            source["kind"] = "factual_generated_file"
            source["exists"] = True
        else:
            source["kind"] = "factual_builtin_fallback"
            source["exists"] = False
        return source

    return source


# ── Dataset loader ────────────────────────────────────────────

def load_category_tasks(category: str, cfg: dict) -> list[dict]:
    """Load tasks for a given category and return as list of dicts."""
    ds_cfg = cfg["datasets"]
    n = cfg["eval"]["samples_per_category"]

    if category == "multistep_arithmetic":
        samples = load_gsm8k(
            split=ds_cfg["gsm8k"]["split"],
            max_samples=n,
            cache_dir=ds_cfg["gsm8k"]["cache_dir"],
        )
        return [gsm8k_to_dict(s) for s in samples]

    elif category == "causal_counterfactual":
        samples = load_crass(
            path=ds_cfg["crass"]["path"],
            max_samples=n,
        )
        return [crass_to_dict(s) for s in samples]

    elif category == "tool_use_planning":
        samples = load_toolbench(
            path=ds_cfg["toolbench"]["path"],
            max_samples=n,
        )
        return [tool_to_dict(s) for s in samples]

    elif category == "factual_consistency":
        samples = load_factual_synthetic(
            cache_path=ds_cfg["factual_synthetic"]["path"],
            max_samples=n,
        )
        return [factual_to_dict(s) for s in samples]

    else:
        raise ValueError(f"Unknown category: {category}")


def task_to_decomposer_fields(task_dict: dict, category: str) -> dict:
    """Extract the fields needed by TaskDecomposer.build for a given category."""
    if category == "multistep_arithmetic":
        return {"question": task_dict["question"]}
    elif category == "factual_consistency":
        return {"context": task_dict["context"], "questions": task_dict["questions"]}
    elif category == "tool_use_planning":
        return {
            "goal": task_dict["goal"],
            "available_tools": task_dict["available_tools"],
            "constraints": task_dict.get("constraints", []),
        }
    elif category == "causal_counterfactual":
        return {
            "premise": task_dict["premise"],
            "question": task_dict["question"],
        }
    return {}


def ground_truth_for_category(task_dict: dict, category: str) -> str:
    if category == "multistep_arithmetic":
        return task_dict.get("ground_truth", "")
    elif category == "factual_consistency":
        answers = task_dict.get("answers", [])
        return "; ".join(answers) if answers else ""
    elif category == "causal_counterfactual":
        return task_dict.get("correct_answer", "")
    elif category == "tool_use_planning":
        steps = task_dict.get("correct_plan", [])
        return " | ".join(steps) if steps else "Plan complete."
    return ""


# ── Main eval loop ────────────────────────────────────────────

def run_model_category(
    model_id: str,
    display_name: str,
    category: str,
    tasks: list[dict],
    client: GroqClient,
    decomposer: TaskDecomposer,
    step_evaluator: StepEvaluator,
    hall_scorer: HallucinationScorer,
    drift_detector: DriftDetector,
    regrounder: Regrounder,
    aggregator: MetricsAggregator,
    eval_cfg: dict,
    run_mitigation: bool = True,
    mitigation_eval_mode: str = "none",
) -> list[dict]:
    """
    Run evaluation for one (model, category) pair.
    Returns list of raw result dicts for saving.
    """
    raw_results = []
    total = len(tasks)
    logger.info(f"  [{display_name}] [{category}] Starting {total} tasks")

    for i, task_dict in enumerate(tasks):
        task_id = task_dict.get("id", f"{category}_{i:04d}")

        # ── 1. Build prompt ──────────────────────────────────
        fields = task_to_decomposer_fields(task_dict, category)
        gt = ground_truth_for_category(task_dict, category)
        decomposed = decomposer.build(
            task_id=task_id,
            category=category,
            ground_truth=gt,
            **fields,
        )

        # ── 2. Generate reasoning trace ───────────────────────
        try:
            # For Ollama, force unload the solver model after generation to free VRAM for the judge
            # This prevents 500 errors when running large models alongside the judge
            call_kwargs = {}
            if client.__class__.__name__ == "OllamaClient":
                call_kwargs["keep_alive"] = 0

            resp = client.complete(
                model=model_id,
                user_prompt=decomposed.prompt,
                system_prompt=decomposer.get_system_prompt(category),
                temperature=eval_cfg["temperature"],
                max_tokens=eval_cfg["max_tokens"],
                **call_kwargs,
            )
            
            # Brief pause to allow Ollama to unload the model from VRAM
            if client.__class__.__name__ == "OllamaClient":
                time.sleep(2.0)
                
            decomposed = decomposer.parse_response(decomposed, resp.content)
        except Exception as e:
            logger.error(f"[{task_id}] Generation error: {e}")
            continue

        # ── 3. Step-level evaluation ──────────────────────────
        task_eval = step_evaluator.evaluate(decomposed, model_name=display_name)

        # ── 4. Hallucination scoring ──────────────────────────
        hall_scores = hall_scorer.score_steps(
            task_id=task_id,
            steps=decomposed.steps,
            context=decomposed.prompt[:400],
        )
        hall_summary = hall_scorer.aggregate(hall_scores)

        # ── 5. Drift detection ────────────────────────────────
        drift_result = drift_detector.detect(
            task_id=task_id,
            problem=decomposed.prompt,
            steps=decomposed.steps,
            category=category,
        )

        # ── 6. Mitigation (for flagged tasks OR wrong answers) ──
        mitigation_results = {}
        rg_result = None
        rp_result = None
        if run_mitigation and (
            task_eval.num_invalid > 0
            or hall_summary["flagged_steps"] > 0
            or task_eval.final_answer_correct is False
        ):
            suspicious = [
                c for s in hall_scores for c in s.suspicious_claims
            ]
            # RAG reground
            rg_result = regrounder.reground(
                task=decomposed,
                model=model_id,
                suspicious_claims=suspicious,
                mode=InterventionMode.FULL,
            )
            # Simple reprompt (ablation baseline)
            rp_result = regrounder.reground(
                task=decomposed,
                model=model_id,
                mode=InterventionMode.REPROMPT,
            )
            mitigation_results = {
                "rag_reground": regrounder.to_dict(rg_result),
                "reprompt": regrounder.to_dict(rp_result),
            }

        # Optional: score aggregate metrics on mitigated outputs.
        # This does not change raw per-step verdicts, only which final answer is
        # considered for final-accuracy / propagation aggregation.
        effective_answer_mode = "original"
        if mitigation_eval_mode != "none" and decomposed.ground_truth:
            mode_to_answer = {}
            if rg_result and rg_result.success:
                mode_to_answer["rag"] = rg_result.regrounded_answer
            if rp_result and rp_result.success:
                mode_to_answer["reprompt"] = rp_result.regrounded_answer

            if mitigation_eval_mode in ("rag", "reprompt") and mitigation_eval_mode in mode_to_answer:
                candidate = mode_to_answer[mitigation_eval_mode]
                candidate_correct = step_evaluator.check_answer_correctness(
                    candidate,
                    decomposed.ground_truth,
                    decomposed.category,
                )
                if candidate_correct is not None:
                    task_eval.final_answer_correct = candidate_correct
                    task_eval.error_propagated = (
                        task_eval.first_failure_step is not None
                        and candidate_correct is False
                    )
                    effective_answer_mode = mitigation_eval_mode

            elif mitigation_eval_mode == "best":
                best_mode = "original"
                best_correct = task_eval.final_answer_correct

                for cand_mode, cand_answer in mode_to_answer.items():
                    cand_correct = step_evaluator.check_answer_correctness(
                        cand_answer,
                        decomposed.ground_truth,
                        decomposed.category,
                    )
                    if cand_correct is True and best_correct is not True:
                        best_correct = True
                        best_mode = cand_mode

                if best_correct is not None:
                    task_eval.final_answer_correct = best_correct
                    task_eval.error_propagated = (
                        task_eval.first_failure_step is not None
                        and best_correct is False
                    )
                    effective_answer_mode = best_mode

        # ── 7. Register with aggregator ───────────────────────
        aggregator.add_task_eval(task_eval, hall_scores, drift_result)

        # ── 8. Store raw result ───────────────────────────────
        raw_result = {
            "task": decomposer.to_dict(decomposed),
            "evaluation": step_evaluator.to_dict(task_eval),
            "hallucination": {
                "scores": hall_scorer.to_dict_list(hall_scores),
                "summary": hall_summary,
            },
            "drift": drift_detector.to_dict(drift_result),
            "mitigation": mitigation_results,
            "effective_answer_mode": effective_answer_mode,
        }
        raw_results.append(raw_result)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{display_name}] [{category}] {i+1}/{total} | "
                f"failures={task_eval.num_invalid} propagated={task_eval.error_propagated}"
            )

    return raw_results


# ── Entry point ───────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 baseline evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", nargs="+", help="Override model IDs to evaluate")
    parser.add_argument("--categories", nargs="+", help="Override categories to evaluate")
    parser.add_argument("--samples", type=int, help="Override samples per category")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-mitigation", action="store_true", help="Skip mitigation runs")
    parser.add_argument("--output", default=None, help="Override output JSON path")
    parser.add_argument("--provider", choices=["groq", "ollama", "openai"], default=None,
                        help="LLM provider for SOLVER: groq | ollama | openai")
    parser.add_argument("--evaluator", default=None,
                        help="Override evaluator model ID (solver's model used if not set)")
    parser.add_argument("--judge-provider", choices=["groq", "ollama", "openai"], default=None,
                        help="Separate provider for the JUDGE (step evaluator). "
                             "E.g. --judge-provider openai --judge-model gpt-4o-mini "
                             "to use GPT-4o-mini as an independent judge while Ollama solves.")
    parser.add_argument("--judge-model", default=None,
                        help="Model ID for the judge (used with --judge-provider). "
                             "Examples: gpt-4o-mini, gpt-4o, qwen2.5:14b")
    parser.add_argument(
        "--mitigation-metrics",
        choices=["none", "rag", "reprompt", "best"],
        default="none",
        help="How to compute aggregate final accuracy when mitigation is enabled. "
             "none=original only, rag=RAG answer, reprompt=reprompt answer, best=best among original/rag/reprompt",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────
    config_path = _ROOT / args.config
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg)
    os.makedirs(cfg["paths"]["outputs"], exist_ok=True)
    os.makedirs(cfg["paths"]["charts"], exist_ok=True)

    if args.samples:
        cfg["eval"]["samples_per_category"] = args.samples

    categories = args.categories or cfg["eval"]["categories"]
    models = args.models or [m["id"] for m in cfg["models"]["eval_models"]]
    model_display = {m["id"]: m["display_name"] for m in cfg["models"]["eval_models"]}

    use_wandb = not args.no_wandb
    run_mitigation = not args.no_mitigation
    provider = args.provider or cfg.get("provider", "groq")

    # Judge provider — separate from solver provider if specified
    judge_provider = args.judge_provider or provider
    judge_model_override = args.judge_model

    run_start_ts = time.time()

    logger.info("=" * 60)
    logger.info("LLM REASONING EVALUATION — PHASE 1 BASELINE")
    logger.info(f"  Models:     {models}")
    logger.info(f"  Categories: {categories}")
    logger.info(f"  Samples:    {cfg['eval']['samples_per_category']}")
    logger.info(f"  Mitigation: {run_mitigation}")
    logger.info(f"  Mitigation metrics mode: {args.mitigation_metrics}")
    logger.info(f"  Solver:     {provider}")
    logger.info(f"  Judge:      {judge_provider}" + (f" ({judge_model_override})" if judge_model_override else ""))
    logger.info("=" * 60)

    # ── Init components ───────────────────────────────────────
    client = get_client(cfg, provider=provider)
    decomposer = TaskDecomposer()

    # Determine evaluator model and client
    # --judge-provider groq                           → Groq 70B as independent judge
    # --judge-provider openai --judge-model gpt-4o    → GPT-4o as independent judge
    # (default) same as solver                        → original behaviour
    uses_separate_judge = judge_provider != provider
    if judge_model_override:
        judge_client = get_client(cfg, provider=judge_provider)
        evaluator_model = judge_model_override
        logger.info(f"  Independent judge: {judge_provider}/{evaluator_model}")
    elif uses_separate_judge:
        judge_client = get_client(cfg, provider=judge_provider)
        evaluator_model = cfg["models"]["evaluator_model"]
        logger.info(f"  Independent judge: {judge_provider}/{evaluator_model}")
    else:
        judge_client = client
        evaluator_model = args.evaluator or cfg["models"]["evaluator_model"]
        if provider == "ollama" and not args.evaluator:
            evaluator_model = cfg.get("models", {}).get("local_judge_model", models[0])
            logger.info(f"  Ollama mode: using '{evaluator_model}' as evaluator model")

    step_evaluator = StepEvaluator(
        client=judge_client,
        evaluator_model=evaluator_model,
        # Pass separate judge for answer verification (eliminates self-grading for non-arithmetic)
        judge_client=judge_client if uses_separate_judge or judge_model_override else None,
        judge_model=evaluator_model if uses_separate_judge or judge_model_override else None,
    )
    hall_scorer = HallucinationScorer(
        client=judge_client,
        model=evaluator_model,
    )
    drift_detector = DriftDetector(
        client=judge_client,
        model=evaluator_model,
    )
    retriever = WikipediaRetriever(
        top_k=cfg["mitigation"]["retriever"]["top_k"],
        max_chars_per_doc=cfg["mitigation"]["retriever"]["max_chars_per_doc"],
    )
    regrounder = Regrounder(client=client, retriever=retriever)
    aggregator = MetricsAggregator(
        wandb_project=cfg["api"]["wandb_project"],
        wandb_entity=cfg["api"].get("wandb_entity"),
        use_wandb=use_wandb,
    )

    run_name = f"baseline_{'_'.join(m.split('-')[0] for m in models)}_{int(time.time())}"
    aggregator.init_run(
        run_name=run_name,
        config={
            "models": models,
            "categories": categories,
            "samples_per_category": cfg["eval"]["samples_per_category"],
        },
    )

    # ── Pre-load all datasets ─────────────────────────────────
    logger.info("Loading datasets...")
    category_tasks: dict[str, list[dict]] = {}
    dataset_sources: dict[str, dict] = {}
    for cat in categories:
        tasks = load_category_tasks(cat, cfg)
        category_tasks[cat] = tasks
        dataset_sources[cat] = _dataset_source_summary(cat, cfg)
        logger.info(f"  {cat}: {len(tasks)} tasks loaded")
        ds = dataset_sources[cat]
        logger.info(
            f"    source={ds['kind']}"
            + (f" path={ds['path']}" if ds.get("path") else "")
        )

    # ── Main eval loop ────────────────────────────────────────
    all_raw_results = {}
    daily_limit_hit = False

    for model_id in models:
        if daily_limit_hit:
            break
        display = model_display.get(model_id, model_id)
        logger.info(f"\nEvaluating model: {display} ({model_id})")
        all_raw_results[display] = {}

        for category in categories:
            if daily_limit_hit:
                break
            tasks = category_tasks[category]
            logger.info(f"  Category: {category} ({len(tasks)} tasks)")

            try:
                results = run_model_category(
                    model_id=model_id,
                    display_name=display,
                    category=category,
                    tasks=tasks,
                    client=client,
                    decomposer=decomposer,
                    step_evaluator=step_evaluator,
                    hall_scorer=hall_scorer,
                    drift_detector=drift_detector,
                    regrounder=regrounder,
                    aggregator=aggregator,
                    eval_cfg=cfg["eval"],
                    run_mitigation=run_mitigation,
                    mitigation_eval_mode=args.mitigation_metrics,
                )
            except DailyLimitExhaustedError:
                logger.error("=" * 60)
                logger.error("DAILY TOKEN LIMIT (TPD) EXHAUSTED — saving partial results")
                logger.error("Re-run tomorrow or upgrade at console.groq.com/settings/billing")
                logger.error("=" * 60)
                daily_limit_hit = True
                results = []
            all_raw_results[display][category] = results
            logger.info(f"  Done: {len(results)} results")

    # ── Finalize metrics ──────────────────────────────────────
    # Build unique filenames: model_name(s) + mitigation flag + timestamp
    model_tag = "_".join(m.replace(":", "-").replace("/", "-") for m in models)
    mit_tag = "mit" if run_mitigation else "nomir"
    run_ts = int(time.time())
    run_suffix = f"{model_tag}_{mit_tag}_{run_ts}"

    output_path = args.output or os.path.join(
        cfg["paths"]["outputs"], f"eval_results_{run_suffix}.json"
    )
    metrics = aggregator.finalize(output_path=output_path)
    aggregator.finish_run()

    # Save raw results
    raw_path = os.path.join(cfg["paths"]["outputs"], f"raw_results_{run_suffix}.json")
    with open(raw_path, "w") as f:
        json.dump(all_raw_results, f, indent=2)

    # Save failure report separately (used by finetune/dataset_builder.py)
    failure_path = os.path.join(
        cfg["paths"]["outputs"], f"failure_report_{run_suffix}.json"
    )
    with open(failure_path, "w") as f:
        json.dump(metrics["failure_report"], f, indent=2)

    # Save run manifest for reproducibility and debugging.
    completed_tasks = 0
    attempted_tasks = 0
    for model_id in models:
        display = model_display.get(model_id, model_id)
        by_cat = all_raw_results.get(display, {})
        for cat in categories:
            attempted_tasks += len(category_tasks.get(cat, []))
            completed_tasks += len(by_cat.get(cat, []))

    solver_usage = None
    judge_usage = None
    try:
        if hasattr(client, "get_usage_summary"):
            solver_usage = client.get_usage_summary()
    except Exception:
        solver_usage = None
    try:
        if hasattr(judge_client, "get_usage_summary"):
            judge_usage = judge_client.get_usage_summary()
    except Exception:
        judge_usage = None

    run_manifest = {
        "run_suffix": run_suffix,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "duration_seconds": round(time.time() - run_start_ts, 2),
        "solver_provider": provider,
        "judge_provider": judge_provider,
        "judge_model": evaluator_model,
        "models": models,
        "categories": categories,
        "samples_per_category": cfg["eval"]["samples_per_category"],
        "mitigation_enabled": run_mitigation,
        "mitigation_metrics_mode": args.mitigation_metrics,
        "dataset_sources": dataset_sources,
        "tasks_attempted": attempted_tasks,
        "tasks_completed": completed_tasks,
        "tasks_missing": max(0, attempted_tasks - completed_tasks),
        "partial_run": completed_tasks < attempted_tasks,
        "daily_limit_hit": daily_limit_hit,
        "output_files": {
            "eval_results": output_path,
            "raw_results": raw_path,
            "failure_report": failure_path,
        },
        "solver_usage": solver_usage,
        "judge_usage": judge_usage,
    }

    manifest_path = os.path.join(
        cfg["paths"]["outputs"], f"run_manifest_{run_suffix}.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    # Also save a "latest" symlink-style copy for downstream tools
    for src, dst in [
        (output_path, os.path.join(cfg["paths"]["outputs"], "eval_results.json")),
        (raw_path, os.path.join(cfg["paths"]["outputs"], "raw_results.json")),
        (failure_path, cfg["paths"]["failure_report"]),
        (manifest_path, os.path.join(cfg["paths"]["outputs"], "run_manifest.json")),
    ]:
        with open(src) as f_in:
            data = f_in.read()
        with open(dst, "w") as f_out:
            f_out.write(data)

    # ── Print summary ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE — SUMMARY")
    logger.info("=" * 60)
    for m in metrics["model_summary"]:
        logger.info(
            f"  {m['model']}: "
            f"step_failure={m['overall_step_failure_rate']:.3f} | "
            f"accuracy={m['overall_final_accuracy']:.3f} | "
            f"error_prop={m['overall_error_propagation_rate']:.3f}"
        )

    logger.info(f"\nFine-tuning target: {metrics['failure_report'].get('finetune_target_category')}")
    logger.info(f"Results saved to:   {output_path}")
    logger.info(f"Failure report:     {failure_path}")
    logger.info(f"Run manifest:       {manifest_path}")


if __name__ == "__main__":
    main()
