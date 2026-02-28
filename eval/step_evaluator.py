"""
eval/step_evaluator.py
Step-level meta-evaluation: grades each reasoning step as VALID, INVALID, or UNCERTAIN.

For each step in a reasoning trace, a secondary LLM call evaluates:
  - Logical validity given prior steps
  - Factual accuracy (basic check)
  - Consistency with the original problem

This is the core differentiator of the project — most benchmarks only grade
the final answer. We grade every intermediate step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models.groq_client import GroqClient
from eval.task_decomposer import DecomposedTask, ReasoningStep

logger = logging.getLogger(__name__)


# ── Enums and models ──────────────────────────────────────────

class StepVerdict(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class StepEvaluation:
    step_index: int
    step_text: str
    verdict: StepVerdict
    confidence: float         # 0.0 – 1.0
    explanation: str
    error_type: Optional[str] = None   # "logic_error" | "hallucination" | "contradiction" | None


@dataclass
class TaskEvaluation:
    task_id: str
    category: str
    model_name: str
    step_evaluations: list[StepEvaluation] = field(default_factory=list)
    final_answer_correct: Optional[bool] = None
    final_answer: str = ""
    ground_truth: str = ""
    num_steps: int = 0
    num_invalid: int = 0
    num_uncertain: int = 0
    first_failure_step: Optional[int] = None   # step index of first INVALID
    error_propagated: bool = False              # did a step failure lead to wrong final answer?
    eval_error: Optional[str] = None


# ── Prompts ───────────────────────────────────────────────────

_STEP_EVAL_SYSTEM = """You are a rigorous reasoning evaluator. Your job is to assess whether a single reasoning step is valid.

You will be given:
1. The original problem
2. All previous reasoning steps (context)
3. The current step to evaluate

Evaluate the current step on three criteria:
- LOGICAL VALIDITY: Is the inference logically sound given the previous steps?
- FACTUAL ACCURACY: Does the step make any factual claims? If so, are they accurate?
- CONSISTENCY: Does the step contradict anything in the previous steps or the problem?

Respond ONLY with a valid JSON object in this exact format:
{
  "verdict": "VALID" | "INVALID" | "UNCERTAIN",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<one sentence explaining your verdict>",
  "error_type": null | "logic_error" | "hallucination" | "contradiction"
}

Rules:
- VALID: The step is logically sound, factually reasonable, and consistent.
- INVALID: The step contains a clear logical error, factual hallucination, or contradiction.
- UNCERTAIN: You cannot determine validity without external knowledge, or the step is ambiguous.
- error_type must be null when verdict is VALID or UNCERTAIN.
- Be strict but fair. Do not mark a step INVALID just because it could be expressed better."""


def _build_step_eval_prompt(
    problem: str,
    prior_steps: list[ReasoningStep],
    current_step: ReasoningStep,
    category: str,
) -> str:
    prior_text = ""
    if prior_steps:
        prior_text = "\n".join(
            f"Step {s.index}: {s.text}" for s in prior_steps
        )
        prior_text = f"\n\nPrevious steps:\n{prior_text}"

    return (
        f"Category: {category}\n\n"
        f"Original problem:\n{problem}"
        f"{prior_text}\n\n"
        f"Current step to evaluate (Step {current_step.index}):\n"
        f"{current_step.text}"
    )


# ── Final answer checker ──────────────────────────────────────

def _check_answer_correctness(
    predicted: str,
    ground_truth: str,
    category: str,
) -> Optional[bool]:
    """
    Heuristic final answer comparison.
    Returns True/False/None (None = cannot determine).
    """
    if not predicted or not ground_truth:
        return None

    pred = predicted.strip().lower()
    truth = ground_truth.strip().lower()

    # Exact match
    if pred == truth:
        return True

    # Numeric match (strip commas, spaces, units)
    import re
    def extract_numbers(s: str) -> list[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", s)

    pred_nums = extract_numbers(pred)
    truth_nums = extract_numbers(truth)
    if pred_nums and truth_nums:
        try:
            if abs(float(pred_nums[-1]) - float(truth_nums[-1])) < 1e-6:
                return True
        except ValueError:
            pass

    # Substring match for long factual answers
    if category == "factual_consistency":
        # truth words present in predicted
        truth_words = set(truth.split())
        pred_words = set(pred.split())
        overlap = truth_words & pred_words
        if truth_words and len(overlap) / len(truth_words) > 0.6:
            return True

    return False


# ── Main evaluator ────────────────────────────────────────────

class StepEvaluator:
    """
    Evaluates each reasoning step in a completed DecomposedTask.

    Usage:
        evaluator = StepEvaluator(groq_client, evaluator_model="llama-3-8b-8192")
        task_eval = evaluator.evaluate(decomposed_task, model_name="mixtral-8x7b-32768")
    """

    def __init__(
        self,
        client: GroqClient,
        evaluator_model: str = "llama-3-8b-8192",
        temperature: float = 0.05,
        max_tokens: int = 512,
        skip_single_step: bool = False,
    ) -> None:
        self._client = client
        self._model = evaluator_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._skip_single_step = skip_single_step

    def evaluate(
        self,
        task: DecomposedTask,
        model_name: str = "unknown",
    ) -> TaskEvaluation:
        """
        Run step-level evaluation on a completed reasoning task.

        Args:
            task:       DecomposedTask with steps already parsed
            model_name: Name of the model that generated the reasoning (for logging)

        Returns:
            TaskEvaluation with per-step verdicts and aggregate stats
        """
        result = TaskEvaluation(
            task_id=task.task_id,
            category=task.category,
            model_name=model_name,
            final_answer=task.final_answer,
            ground_truth=task.ground_truth,
            num_steps=len(task.steps),
        )

        if task.parse_error:
            result.eval_error = f"Task parse error: {task.parse_error}"
            return result

        if not task.steps:
            result.eval_error = "No steps to evaluate"
            return result

        # Evaluate each step
        for i, step in enumerate(task.steps):
            prior = task.steps[:i]
            try:
                step_eval = self._evaluate_step(
                    problem=task.prompt,
                    prior_steps=prior,
                    current_step=step,
                    category=task.category,
                )
                result.step_evaluations.append(step_eval)
                logger.debug(
                    f"[{task.task_id}] Step {step.index}: {step_eval.verdict.value} "
                    f"(conf={step_eval.confidence:.2f})"
                )
            except Exception as e:
                logger.error(f"[{task.task_id}] Step {step.index} eval error: {e}")
                result.step_evaluations.append(
                    StepEvaluation(
                        step_index=step.index,
                        step_text=step.text,
                        verdict=StepVerdict.UNCERTAIN,
                        confidence=0.0,
                        explanation=f"Evaluation error: {str(e)}",
                    )
                )

        # Aggregate stats
        result.num_invalid = sum(
            1 for se in result.step_evaluations if se.verdict == StepVerdict.INVALID
        )
        result.num_uncertain = sum(
            1 for se in result.step_evaluations if se.verdict == StepVerdict.UNCERTAIN
        )

        # First failure step
        for se in result.step_evaluations:
            if se.verdict == StepVerdict.INVALID:
                result.first_failure_step = se.step_index
                break

        # Final answer correctness
        result.final_answer_correct = _check_answer_correctness(
            task.final_answer, task.ground_truth, task.category
        )

        # Error propagation: step failed AND final answer wrong
        if result.first_failure_step is not None and result.final_answer_correct is False:
            result.error_propagated = True

        return result

    def _evaluate_step(
        self,
        problem: str,
        prior_steps: list[ReasoningStep],
        current_step: ReasoningStep,
        category: str,
    ) -> StepEvaluation:
        prompt = _build_step_eval_prompt(problem, prior_steps, current_step, category)

        raw = self._client.complete_json(
            model=self._model,
            user_prompt=prompt,
            system_prompt=_STEP_EVAL_SYSTEM,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        verdict_str = str(raw.get("verdict", "UNCERTAIN")).upper().strip()
        try:
            verdict = StepVerdict(verdict_str)
        except ValueError:
            verdict = StepVerdict.UNCERTAIN

        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        error_type = raw.get("error_type")
        if verdict == StepVerdict.VALID:
            error_type = None

        return StepEvaluation(
            step_index=current_step.index,
            step_text=current_step.text,
            verdict=verdict,
            confidence=confidence,
            explanation=raw.get("explanation", ""),
            error_type=error_type,
        )

    def to_dict(self, result: TaskEvaluation) -> dict:
        return {
            "task_id": result.task_id,
            "category": result.category,
            "model_name": result.model_name,
            "num_steps": result.num_steps,
            "num_invalid": result.num_invalid,
            "num_uncertain": result.num_uncertain,
            "first_failure_step": result.first_failure_step,
            "error_propagated": result.error_propagated,
            "final_answer": result.final_answer,
            "ground_truth": result.ground_truth,
            "final_answer_correct": result.final_answer_correct,
            "eval_error": result.eval_error,
            "step_evaluations": [
                {
                    "step_index": se.step_index,
                    "step_text": se.step_text,
                    "verdict": se.verdict.value,
                    "confidence": se.confidence,
                    "explanation": se.explanation,
                    "error_type": se.error_type,
                }
                for se in result.step_evaluations
            ],
        }
