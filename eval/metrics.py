"""
eval/metrics.py
Aggregates step-level evaluations into task-level and run-level metrics.
Logs everything to Weights & Biases.

Metrics computed:
  - Step failure rate (per category, per model, per step index)
  - Error propagation rate
  - Hallucination rate
  - Drift rate
  - Final answer accuracy
  - Cross-model comparison matrices
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Per-run results container ─────────────────────────────────

@dataclass
class RunMetrics:
    model_name: str
    category: str
    total_tasks: int = 0
    tasks_with_failures: int = 0
    total_steps: int = 0
    invalid_steps: int = 0
    uncertain_steps: int = 0
    error_propagation_count: int = 0
    final_answer_correct: int = 0
    final_answer_evaluated: int = 0
    hallucinated_tasks: int = 0          # tasks with ≥1 hallucination flag
    drifted_tasks: int = 0

    # Step-index breakdown: {step_idx: {"invalid": int, "total": int}}
    step_index_stats: dict = field(default_factory=lambda: defaultdict(lambda: {"invalid": 0, "total": 0}))

    @property
    def step_failure_rate(self) -> float:
        return self.invalid_steps / self.total_steps if self.total_steps else 0.0

    @property
    def task_failure_rate(self) -> float:
        return self.tasks_with_failures / self.total_tasks if self.total_tasks else 0.0

    @property
    def error_propagation_rate(self) -> float:
        return self.error_propagation_count / self.tasks_with_failures if self.tasks_with_failures else 0.0

    @property
    def final_accuracy(self) -> float:
        return self.final_answer_correct / self.final_answer_evaluated if self.final_answer_evaluated else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated_tasks / self.total_tasks if self.total_tasks else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "category": self.category,
            "total_tasks": self.total_tasks,
            "step_failure_rate": round(self.step_failure_rate, 4),
            "task_failure_rate": round(self.task_failure_rate, 4),
            "error_propagation_rate": round(self.error_propagation_rate, 4),
            "final_accuracy": round(self.final_accuracy, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "drifted_tasks": self.drifted_tasks,
            "total_steps": self.total_steps,
            "invalid_steps": self.invalid_steps,
        }


# ── Aggregator ────────────────────────────────────────────────

class MetricsAggregator:
    """
    Aggregates evaluation results and logs to W&B.

    Usage:
        agg = MetricsAggregator(wandb_project="llm-reasoning-eval")
        agg.init_run(run_name="baseline_llama3")
        for task_eval in results:
            agg.add_task_eval(task_eval, hallucination_scores, drift_result)
        summary = agg.finalize()
        agg.finish_run()
    """

    def __init__(
        self,
        wandb_project: str = "llm-reasoning-eval",
        wandb_entity: Optional[str] = None,
        use_wandb: bool = True,
    ) -> None:
        self._project = wandb_project
        self._entity = wandb_entity
        self._use_wandb = use_wandb
        self._wandb_run = None
        # {(model, category) -> RunMetrics}
        self._metrics: dict[tuple, RunMetrics] = {}
        # Raw records for failure analysis
        self._raw_records: list[dict] = []

    def init_run(
        self,
        run_name: str,
        config: Optional[dict] = None,
    ) -> None:
        """Initialize a W&B run."""
        if not self._use_wandb:
            logger.info(f"W&B disabled. Run: {run_name}")
            return
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self._project,
                entity=self._entity,
                name=run_name,
                config=config or {},
                reinit=True,
            )
            logger.info(f"W&B run initialized: {run_name}")
        except ImportError:
            logger.warning("wandb not installed — metrics will not be logged to W&B")
            self._use_wandb = False
        except Exception as e:
            logger.warning(f"W&B init failed: {e} — continuing without W&B")
            self._use_wandb = False

    def add_task_eval(
        self,
        task_eval,              # TaskEvaluation
        hallucination_scores: list = None,
        drift_result=None,      # DriftResult
    ) -> None:
        """Register a single task evaluation result."""
        key = (task_eval.model_name, task_eval.category)
        if key not in self._metrics:
            self._metrics[key] = RunMetrics(
                model_name=task_eval.model_name,
                category=task_eval.category,
            )
        m = self._metrics[key]

        m.total_tasks += 1
        m.total_steps += task_eval.num_steps
        m.invalid_steps += task_eval.num_invalid
        m.uncertain_steps += task_eval.num_uncertain

        if task_eval.num_invalid > 0:
            m.tasks_with_failures += 1

        if task_eval.error_propagated:
            m.error_propagation_count += 1

        if task_eval.final_answer_correct is not None:
            m.final_answer_evaluated += 1
            if task_eval.final_answer_correct:
                m.final_answer_correct += 1

        # Step-index breakdown
        for se in task_eval.step_evaluations:
            m.step_index_stats[se.step_index]["total"] += 1
            if se.verdict and se.verdict.value == "INVALID":
                m.step_index_stats[se.step_index]["invalid"] += 1

        # Hallucination
        if hallucination_scores:
            flagged = any(s.is_flagged for s in hallucination_scores)
            if flagged:
                m.hallucinated_tasks += 1

        # Drift
        if drift_result and drift_result.has_drift:
            m.drifted_tasks += 1

        # Store raw record
        record = {
            "task_id": task_eval.task_id,
            "model": task_eval.model_name,
            "category": task_eval.category,
            "num_steps": task_eval.num_steps,
            "num_invalid": task_eval.num_invalid,
            "error_propagated": task_eval.error_propagated,
            "final_answer_correct": task_eval.final_answer_correct,
            "first_failure_step": task_eval.first_failure_step,
        }
        self._raw_records.append(record)

        # Log to W&B per task
        if self._use_wandb and self._wandb_run:
            try:
                import wandb
                self._wandb_run.log({
                    f"{task_eval.model_name}/{task_eval.category}/num_invalid": task_eval.num_invalid,
                    f"{task_eval.model_name}/{task_eval.category}/error_propagated": int(task_eval.error_propagated),
                    f"{task_eval.model_name}/{task_eval.category}/final_correct": int(task_eval.final_answer_correct or 0),
                })
            except Exception as e:
                logger.debug(f"W&B per-task log error: {e}")

    def finalize(self, output_path: Optional[str] = None) -> dict:
        """
        Compute final aggregate metrics, generate W&B summary tables,
        and optionally save to a JSON file.

        Returns:
            dict with per-(model, category) metrics and overall summary
        """
        all_metrics = [m.to_dict() for m in self._metrics.values()]

        # Step-index failure breakdown (which step fails most)
        step_failure_breakdown = self._compute_step_breakdown()

        # Model comparison matrix
        model_summary = self._compute_model_summary()

        # Failure mode report (for fine-tuning targeting)
        failure_report = self._compute_failure_report()

        result = {
            "per_run_metrics": all_metrics,
            "step_failure_breakdown": step_failure_breakdown,
            "model_summary": model_summary,
            "failure_report": failure_report,
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Metrics saved to {output_path}")

        # Log summary tables to W&B
        if self._use_wandb and self._wandb_run:
            self._log_wandb_tables(all_metrics, step_failure_breakdown)

        return result

    def _compute_step_breakdown(self) -> dict:
        """Per step-index: what fraction of evaluations were INVALID."""
        combined: dict[int, dict] = defaultdict(lambda: {"invalid": 0, "total": 0})
        for m in self._metrics.values():
            for idx, stats in m.step_index_stats.items():
                combined[idx]["invalid"] += stats["invalid"]
                combined[idx]["total"] += stats["total"]
        breakdown = {}
        for idx in sorted(combined.keys()):
            total = combined[idx]["total"]
            invalid = combined[idx]["invalid"]
            breakdown[str(idx)] = {
                "failure_rate": round(invalid / total, 4) if total else 0.0,
                "total": total,
                "invalid": invalid,
            }
        return breakdown

    def _compute_model_summary(self) -> list[dict]:
        """Aggregate metrics per model across all categories."""
        by_model: dict[str, list] = defaultdict(list)
        for m in self._metrics.values():
            by_model[m.model_name].append(m)

        summary = []
        for model, metrics_list in by_model.items():
            total_tasks = sum(m.total_tasks for m in metrics_list)
            total_steps = sum(m.total_steps for m in metrics_list)
            invalid_steps = sum(m.invalid_steps for m in metrics_list)
            ep_count = sum(m.error_propagation_count for m in metrics_list)
            tasks_with_failures = sum(m.tasks_with_failures for m in metrics_list)
            correct = sum(m.final_answer_correct for m in metrics_list)
            evaluated = sum(m.final_answer_evaluated for m in metrics_list)

            summary.append({
                "model": model,
                "total_tasks": total_tasks,
                "overall_step_failure_rate": round(invalid_steps / total_steps, 4) if total_steps else 0.0,
                "overall_error_propagation_rate": round(ep_count / tasks_with_failures, 4) if tasks_with_failures else 0.0,
                "overall_final_accuracy": round(correct / evaluated, 4) if evaluated else 0.0,
                "per_category": {
                    m.category: {
                        "step_failure_rate": round(m.step_failure_rate, 4),
                        "final_accuracy": round(m.final_accuracy, 4),
                    }
                    for m in metrics_list
                },
            })
        return summary

    def _compute_failure_report(self) -> dict:
        """
        Identify which (model, category) combinations have the highest failure rates.
        Used to select fine-tuning targets.
        """
        ranked = sorted(
            self._metrics.values(),
            key=lambda m: m.step_failure_rate,
            reverse=True,
        )
        worst = [
            {
                "model": m.model_name,
                "category": m.category,
                "step_failure_rate": round(m.step_failure_rate, 4),
                "error_propagation_rate": round(m.error_propagation_rate, 4),
                "recommendation": f"High-priority fine-tuning target for {m.category}",
            }
            for m in ranked[:5]
        ]
        return {
            "worst_performing": worst,
            "finetune_target_category": ranked[0].category if ranked else None,
            "finetune_target_model": ranked[0].model_name if ranked else None,
        }

    def _log_wandb_tables(self, all_metrics: list, step_breakdown: dict) -> None:
        try:
            import wandb

            # Summary table
            table = wandb.Table(
                columns=["model", "category", "step_failure_rate",
                         "task_failure_rate", "error_propagation_rate",
                         "final_accuracy", "hallucination_rate"],
                data=[
                    [m["model"], m["category"], m["step_failure_rate"],
                     m["task_failure_rate"], m["error_propagation_rate"],
                     m["final_accuracy"], m["hallucination_rate"]]
                    for m in all_metrics
                ],
            )
            self._wandb_run.log({"eval_summary_table": table})

            # Step breakdown table
            step_table = wandb.Table(
                columns=["step_index", "failure_rate", "total_evaluations"],
                data=[
                    [int(idx), v["failure_rate"], v["total"]]
                    for idx, v in step_breakdown.items()
                ],
            )
            self._wandb_run.log({"step_failure_breakdown": step_table})

            logger.info("W&B tables logged")
        except Exception as e:
            logger.warning(f"W&B table log failed: {e}")

    def finish_run(self) -> None:
        if self._use_wandb and self._wandb_run:
            try:
                self._wandb_run.finish()
                logger.info("W&B run finished")
            except Exception:
                pass

    def get_raw_records(self) -> list[dict]:
        return self._raw_records
